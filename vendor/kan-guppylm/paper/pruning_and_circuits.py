"""
Interpretability leverage experiments for the paper:
  #1  Structured pruning — sweep τ thresholds, measure zero-shot val_loss degradation
  #2  Functional archetypes — FPCA on edge curves for all 6 layers
  #3  Domain-selective circuits — fish vs. generic prompt activation diff heatmaps

After running, call:
  python paper/pruning_and_circuits.py --retrain   # fine-tune pruned model (adds ~30 min)

Results saved to:
  paper/pruning_results.json
  paper/figures/pruning_sparsity.pdf
  paper/figures/fpca_all_layers.pdf
  paper/figures/circuits_heatmap.pdf

Usage:
  uv run --with . python paper/pruning_and_circuits.py [--retrain] [--no-prune] [--no-fpca] [--no-circuits]
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

FISH_PROMPTS = [
    "are you hungry",
    "what do you eat",
    "do you like your tank",
    "what is the temperature of your water",
    "are you lonely",
    "what do you do all day",
    "what color are you",
    "do you have friends in the tank",
]

GENERIC_PROMPTS = [
    "what is money",
    "what is the internet",
    "what is love",
    "tell me something interesting",
    "what is time",
    "can you help me",
    "what is the universe",
    "what is a computer",
]


# ── model loading ──────────────────────────────────────────────────────────────

def load_kan_model(ckpt_path=None):
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM
    path = ckpt_path or ROOT / "checkpoints/best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = KANpreyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ── edge function reconstruction (vectorised) ──────────────────────────────────

@torch.no_grad()
def reconstruct_layer(module, n_points=200):
    """Return curves (out, in, n_pts) and x_grids (in, n_pts)."""
    in_f = module.in_features
    order = module.spline_order

    x_grids = np.zeros((in_f, n_points))
    for i in range(in_f):
        g = module.grid[i].cpu().numpy()
        lo, hi = g[order], g[-order - 1]
        margin = (hi - lo) * 0.05
        x_grids[i] = np.linspace(lo - margin, hi + margin, n_points)

    dev = module.grid.device
    x_diag = torch.tensor(x_grids, dtype=torch.float32).T.to(dev)  # (n_pts, in_f)
    bases_all = module.b_splines(x_diag)                            # (n_pts, in_f, n_basis)
    spline_all = torch.einsum("pin,oin->poi", bases_all, module.spline_weight)
    base_all = F.silu(x_diag).unsqueeze(1) * module.base_weight.unsqueeze(0)
    curves = (base_all + spline_all).detach().cpu().numpy().transpose(1, 2, 0)
    return curves, x_grids   # (out, in, n_pts), (in, n_pts)


def compute_metrics_2d(curves, x_grids):
    """
    Vectorised metrics for all (out, in) edge pairs.
    Returns nls, activity, roughness — each shape (out, in).
    """
    out_f, in_f, n = curves.shape

    # Activity: ||f|| / sqrt(n)
    activity = np.linalg.norm(curves, axis=2) / np.sqrt(n)

    # NLS: per input-channel vectorised affine fit
    nls = np.zeros((out_f, in_f))
    for i in range(in_f):
        x = x_grids[i]
        A = np.column_stack([x, np.ones_like(x)])     # (n, 2)
        f_i = curves[:, i, :]                          # (out_f, n)
        coeffs, _, _, _ = np.linalg.lstsq(A, f_i.T, rcond=None)  # (2, out_f)
        f_lin = (A @ coeffs).T                         # (out_f, n)
        nls[:, i] = (np.linalg.norm(f_i - f_lin, axis=1)
                     / (np.linalg.norm(f_i, axis=1) + 1e-8))

    roughness = np.mean(np.abs(np.diff(curves, n=2, axis=2)), axis=2)
    return nls, activity, roughness


# ── #1  Structured pruning ─────────────────────────────────────────────────────

def get_kan_layers(model):
    from kanprey.kan_layers import KANLinear
    return {n: m for n, m in model.named_modules() if isinstance(m, KANLinear)}


def build_pruning_masks(kan_layers, tau_act_frac=None, tau_nls=None, mode="and"):
    """
    Build pruning masks with three modes:
      'and'      : prune if activity < tau_act_frac*mean AND nls < tau_nls
      'act_only' : prune if activity < tau_act_frac*mean
      'nls_only' : prune if nls < tau_nls
    Returns masks dict and aggregate pruned fraction.
    """
    masks, total_edges, total_pruned = {}, 0, 0
    for name, module in kan_layers.items():
        curves, x_grids = reconstruct_layer(module)
        nls, activity, _ = compute_metrics_2d(curves, x_grids)
        layer_mean_act = activity.mean()
        if mode == "act_only":
            mask = activity < tau_act_frac * layer_mean_act
        elif mode == "nls_only":
            mask = nls < tau_nls
        else:  # "and"
            mask = (activity < tau_act_frac * layer_mean_act) & (nls < tau_nls)
        masks[name] = mask
        total_edges += mask.size
        total_pruned += mask.sum()
    return masks, total_pruned / total_edges


def apply_pruning_masks(model, masks):
    """Return a deep-copied model with prunable edges zeroed out."""
    from kanprey.kan_layers import KANLinear
    pruned = copy.deepcopy(model)
    for name, module in pruned.named_modules():
        if isinstance(module, KANLinear) and name in masks:
            mask = torch.tensor(masks[name], dtype=torch.bool)   # (out, in)
            with torch.no_grad():
                module.spline_weight[mask] = 0.0
                module.base_weight[mask] = 0.0
    return pruned


@torch.no_grad()
def evaluate_val_loss(model, val_loader, device, max_batches=80):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def pruning_sweep(model, val_loader, device):
    print("\n── #1  Pruning sweep ─────────────────────────────────────────────")
    kan_layers = get_kan_layers(model)
    baseline = evaluate_val_loss(model, val_loader, device)
    print(f"  Baseline val_loss = {baseline:.4f}")

    results = []

    # Activity-only sweep
    print("\n  [Activity-only]")
    for tau_act in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        masks, pruned_frac = build_pruning_masks(kan_layers, tau_act_frac=tau_act, mode="act_only")
        pruned_model = apply_pruning_masks(model, masks).to(device)
        vl = evaluate_val_loss(pruned_model, val_loader, device)
        delta = vl - baseline
        results.append({"mode": "act_only", "tau_act": tau_act, "tau_nls": None,
                         "pruned_frac": float(pruned_frac), "val_loss": float(vl), "delta": float(delta)})
        print(f"    τ_act={tau_act:.1f}  pruned={pruned_frac:.1%}  val_loss={vl:.4f}  Δ={delta:+.4f}")

    # NLS-only sweep
    print("\n  [NLS-only]")
    for tau_nls in [0.05, 0.1, 0.2, 0.3, 0.5]:
        masks, pruned_frac = build_pruning_masks(kan_layers, tau_nls=tau_nls, mode="nls_only")
        pruned_model = apply_pruning_masks(model, masks).to(device)
        vl = evaluate_val_loss(pruned_model, val_loader, device)
        delta = vl - baseline
        results.append({"mode": "nls_only", "tau_act": None, "tau_nls": tau_nls,
                         "pruned_frac": float(pruned_frac), "val_loss": float(vl), "delta": float(delta)})
        print(f"    τ_nls={tau_nls:.2f}  pruned={pruned_frac:.1%}  val_loss={vl:.4f}  Δ={delta:+.4f}")

    return baseline, results


def pick_best_threshold(results, max_delta=0.01):
    """Highest pruning fraction (any mode) with val_loss increase ≤ max_delta."""
    valid = [r for r in results if r["delta"] <= max_delta]
    if not valid:
        valid = [min(results, key=lambda r: r["delta"])]
    return max(valid, key=lambda r: r["pruned_frac"])


def plot_pruning(baseline, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, mode, color, label in [
        (axes[0], "act_only", "#2980b9", "Activity-only pruning"),
        (axes[1], "nls_only", "#e67e22", "NLS-only pruning"),
    ]:
        pts = [r for r in results if r["mode"] == mode]
        pts.sort(key=lambda r: r["pruned_frac"])
        xs = [r["pruned_frac"] * 100 for r in pts]
        ys = [r["val_loss"] for r in pts]
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.5, markersize=5, label=label)
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Baseline")
        ax.axhline(baseline + 0.01, color="gray", linestyle=":", linewidth=1, label="Baseline +0.01")
        ax.set_xlabel("Pruned edges (%)")
        ax.set_ylabel("Val loss (zero-shot)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES / "pruning_sparsity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ── #1b  Fine-tune pruned model ───────────────────────────────────────────────

def finetune_pruned(best, model, val_loader, device):
    """Apply best threshold, fine-tune 3000 steps, return pruned+finetuned model."""
    from kanprey.config import TrainConfig
    from kanprey.dataset import get_dataloader, load_tokenizer

    mode = best.get("mode", "and")
    print(f"\n── #1b  Fine-tuning pruned model "
          f"(mode={mode}, pruned={best['pruned_frac']:.1%}) ──")

    kan_layers = get_kan_layers(model)
    masks, _ = build_pruning_masks(kan_layers,
                                   tau_act_frac=best.get("tau_act"),
                                   tau_nls=best.get("tau_nls"),
                                   mode=mode)
    pruned = apply_pruning_masks(model, masks).to(device)
    pruned.train()

    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))
    train_loader = get_dataloader("train", tokenizer, max_seq_len=128, batch_size=32)

    ft_steps = 3000
    optimizer = torch.optim.AdamW(pruned.parameters(), lr=1e-4, weight_decay=0.1)
    best_vl, best_state = float("inf"), None
    t0 = time.time()

    train_iter = iter(train_loader)
    for step in range(1, ft_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = pruned(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pruned.parameters(), 1.0)
        optimizer.step()

        if step % 300 == 0:
            vl = evaluate_val_loss(pruned, val_loader, device)
            elapsed = time.time() - t0
            print(f"  [step {step:4d}]  train_loss={loss.item():.4f}  "
                  f"val_loss={vl:.4f}  elapsed={elapsed:.0f}s")
            if vl < best_vl:
                best_vl = vl
                best_state = copy.deepcopy(pruned.state_dict())

    pruned.load_state_dict(best_state)
    ckpt_path = ROOT / "checkpoints/pruned_finetuned/best.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save({"model": best_state, "val_loss": best_vl,
                "pruned_frac": best["pruned_frac"],
                "tau_act": best["tau_act"], "tau_nls": best["tau_nls"]},
               ckpt_path)
    print(f"  Fine-tuned best val_loss = {best_vl:.4f}  (saved to {ckpt_path})")
    return pruned, best_vl


# ── #2  Functional archetypes (FPCA) ──────────────────────────────────────────

def run_fpca(curves, n_components=4):
    """SVD-based functional PCA. Returns components (n_comp, n_pts) and var_ratio."""
    X = curves.reshape(-1, curves.shape[2])            # (out*in, n_pts)
    X -= X.mean(axis=0, keepdims=True)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    var_ratio = (s[:n_components] ** 2) / (s ** 2).sum()
    return Vt[:n_components], var_ratio                # (n_comp, n_pts), (n_comp,)


def analyse_fpca(kan_layers):
    print("\n── #2  Functional archetypes (FPCA) ──────────────────────────────")
    n_layers = len(kan_layers)
    n_comp = 4
    fig, axes = plt.subplots(n_layers, n_comp,
                             figsize=(9, 1.8 * n_layers), sharex=True)
    x_norm = np.linspace(0, 1, 200)
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
    layer_var_ratios = {}

    for row, (name, module) in enumerate(kan_layers.items()):
        curves, _ = reconstruct_layer(module)
        components, var_ratio = run_fpca(curves, n_components=n_comp)
        layer_var_ratios[name] = var_ratio.tolist()
        cumvar = var_ratio.cumsum()
        print(f"  {name}: PC1-4 explain {cumvar[-1]:.1%} of variance "
              f"(PC1={var_ratio[0]:.1%})")

        for k in range(n_comp):
            ax = axes[row, k] if n_layers > 1 else axes[k]
            ax.plot(x_norm, components[k], color=colors[k], linewidth=1.2)
            ax.axhline(0, color="gray", linewidth=0.4)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(f"fPC{k+1} ({var_ratio[k]:.1%})", fontsize=8)
            if k == 0:
                short = name.split(".")[-3] if "blocks" in name else name
                ax.set_ylabel(f"L{row}", fontsize=8, rotation=0, labelpad=18)

    fig.supxlabel("Normalised x", fontsize=9)
    fig.supylabel("Component value", fontsize=9)
    fig.suptitle("Functional PCA — KAN edge function archetypes (all layers)", fontsize=10)
    fig.tight_layout()
    out = FIGURES / "fpca_all_layers.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
    return layer_var_ratios


# ── #3  Domain-selective circuits ─────────────────────────────────────────────

def analyse_circuits(model, tokenizer, kan_layers, layer_idx=2):
    """
    For the chosen layer, compute per-edge output diff:
      diff[j,i] = f_{i,j}(fish_mean_input[i]) - f_{i,j}(generic_mean_input[i])

    Uses the actual reconstructed 1D curves evaluated at mean input values.
    """
    print(f"\n── #3  Domain-selective circuits (layer {layer_idx}) ─────────────")

    target_name = f"blocks.{layer_idx}.ffn.kan"
    target_module = dict(model.named_modules()).get(target_name)
    if target_module is None:
        print(f"  Layer {target_name} not found — skipping")
        return None

    # Collect mean pre-KAN activations per category via forward hook
    captured = {"fish": [], "generic": []}

    def make_hook(cat):
        def _hook(mod, inp, out):
            captured[cat].append(inp[0].detach().cpu().float())  # (1, seq, d_in)
        return _hook

    fmt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    dev = next(model.parameters()).device
    model.eval()
    for cat, prompts in [("fish", FISH_PROMPTS), ("generic", GENERIC_PROMPTS)]:
        handle = target_module.register_forward_hook(make_hook(cat))
        for p in prompts:
            ids = tokenizer.encode(fmt.format(p)).ids
            with torch.no_grad():
                model(torch.tensor([ids], dtype=torch.long).to(dev))
        handle.remove()

    # Mean input per category → (d_in,)
    fish_mean = torch.cat(captured["fish"], dim=1).mean(dim=(0, 1)).numpy()   # (384,)
    gen_mean  = torch.cat(captured["generic"], dim=1).mean(dim=(0, 1)).numpy()

    print(f"  Fish mean input norm: {np.linalg.norm(fish_mean):.3f}")
    print(f"  Generic mean input norm: {np.linalg.norm(gen_mean):.3f}")

    # Reconstruct curves and evaluate at mean input values
    curves, x_grids = reconstruct_layer(target_module, n_points=400)
    out_f, in_f, n_pts = curves.shape

    def eval_at(mean_input):
        """Evaluate f_{i,j}(mean_input[i]) for all (j,i) via linear interpolation."""
        vals = np.zeros((out_f, in_f))
        for i in range(in_f):
            xi = np.clip(mean_input[i], x_grids[i, 0], x_grids[i, -1])
            vals[:, i] = np.array([
                np.interp(xi, x_grids[i], curves[j, i]) for j in range(out_f)
            ])
        return vals

    fish_out  = eval_at(fish_mean)
    gen_out   = eval_at(gen_mean)
    diff = fish_out - gen_out   # (out, in) = (384, 384)

    # Magnitude of domain selectivity per edge
    top_k = 20
    flat_idx = np.argsort(np.abs(diff).ravel())[::-1][:top_k]
    top_edges = [(np.unravel_index(idx, diff.shape), diff.ravel()[idx])
                 for idx in flat_idx]
    print(f"  Top {top_k} most domain-selective edges:")
    for (j, i), d in top_edges[:5]:
        print(f"    out={j:3d}, in={i:3d}  diff={d:+.4f}")

    # Plot heatmap (subsampled to 64×64 for readability)
    stride = max(1, out_f // 64)
    diff_small = diff[::stride, ::stride][:64, :64]
    vmax = np.percentile(np.abs(diff_small), 97)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    for ax, data, title in zip(axes,
            [fish_out[::stride, ::stride][:64,:64],
             gen_out[::stride, ::stride][:64,:64],
             diff_small],
            ["Fish prompts", "Generic prompts", "Diff (Fish − Generic)"]):
        vm = np.percentile(np.abs(data), 97)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Input channel (subsampled)", fontsize=8)
        ax.set_ylabel("Output channel (subsampled)", fontsize=8)

    fig.suptitle(f"Domain-selective circuits — Layer {layer_idx} KAN FFN", fontsize=10)
    fig.tight_layout()
    out_path = FIGURES / "circuits_heatmap.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")

    return {"layer": layer_idx,
            "fish_mean_norm": float(np.linalg.norm(fish_mean)),
            "gen_mean_norm": float(np.linalg.norm(gen_mean)),
            "diff_max": float(np.abs(diff).max()),
            "diff_mean": float(np.abs(diff).mean()),
            "top_edges": [{"out": int(j), "in": int(i), "diff": float(d)}
                          for (j, i), d in top_edges]}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain",     action="store_true", help="Fine-tune pruned model")
    parser.add_argument("--no-prune",    action="store_true")
    parser.add_argument("--no-fpca",     action="store_true")
    parser.add_argument("--no-circuits", action="store_true")
    args = parser.parse_args()

    from kanprey.dataset import get_dataloader, load_tokenizer
    from kanprey.train import detect_device

    device = detect_device()
    print(f"Device: {device}")

    print("Loading KANpreyLM...")
    model, cfg = load_kan_model()
    model.to(device)

    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))
    val_loader = get_dataloader("test", tokenizer, max_seq_len=128, batch_size=32)

    kan_layers = get_kan_layers(model)
    print(f"Found {len(kan_layers)} KANLinear layers")

    results = {}

    # ── #1  Pruning sweep ─────────────────────────────────────────────────────
    if not args.no_prune:
        baseline, sweep = pruning_sweep(model, val_loader, device)
        best = pick_best_threshold(sweep, max_delta=0.005)
        print(f"\n  Best threshold: τ_act={best['tau_act']}, τ_nls={best['tau_nls']} "
              f"→ {best['pruned_frac']:.1%} pruned, Δval_loss={best['delta']:+.4f}")
        plot_pruning(baseline, sweep)

        results["pruning"] = {
            "baseline_val_loss": float(baseline),
            "sweep": sweep,
            "best_threshold": best,
        }

        if args.retrain:
            _, ft_val_loss = finetune_pruned(best, model, val_loader, device)
            results["pruning"]["finetuned_val_loss"] = ft_val_loss

    # ── #2  FPCA ─────────────────────────────────────────────────────────────
    if not args.no_fpca:
        layer_var_ratios = analyse_fpca(kan_layers)
        results["fpca"] = layer_var_ratios

    # ── #3  Circuits ──────────────────────────────────────────────────────────
    if not args.no_circuits:
        circuits = analyse_circuits(model, tokenizer, kan_layers, layer_idx=2)
        if circuits:
            results["circuits"] = circuits

    # ── Save ──────────────────────────────────────────────────────────────────
    out_json = Path(__file__).parent / "pruning_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == "__main__":
    main()
