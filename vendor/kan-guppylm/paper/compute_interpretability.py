"""
Standalone interpretability analysis for the paper.
Computes NLS, activity, roughness for all KAN edge functions, runs FPCA,
performs Mann-Whitney U test against MLP baseline, and generates figures.

Saves results to paper/interp_results.json and figures to paper/figures/.

Usage:
    uv run --with . python paper/compute_interpretability.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

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


# ── model loading ─────────────────────────────────────────────────────────────

def load_kan_model(ckpt_path):
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = KANpreyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_original_guppy():
    orig_dir = str(ROOT / "../guppylm-original")
    sys.path.insert(0, orig_dir)
    import importlib.util, os, json as _json
    spec = importlib.util.spec_from_file_location("config_orig", os.path.join(orig_dir, "config.py"))
    cfg_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg_mod)
    spec2 = importlib.util.spec_from_file_location("model_orig", os.path.join(orig_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(mod)
    with open(os.path.join(orig_dir, "config.json")) as f:
        cd = _json.load(f)
    cfg = cfg_mod.GuppyConfig(
        vocab_size=cd.get("vocab_size", 4096),
        max_seq_len=cd.get("max_position_embeddings", 128),
        d_model=cd.get("hidden_size", 384),
        n_layers=cd.get("num_hidden_layers", 6),
        n_heads=cd.get("num_attention_heads", 6),
        ffn_hidden=cd.get("intermediate_size", 768),
        dropout=0.0,
    )
    state = torch.load(os.path.join(orig_dir, "pytorch_model.bin"),
                       map_location="cpu", weights_only=False)
    model = mod.KanpreyLM(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg


# ── KAN edge function reconstruction ─────────────────────────────────────────

@torch.no_grad()
def reconstruct_layer(module, n_points=200):
    """Return curves (out, in, n_pts) and x_grids (in, n_pts)."""
    from kanprey.kan_layers import KANLinear
    in_f = module.in_features
    out_f = module.out_features
    order = module.spline_order

    x_grids = np.zeros((in_f, n_points))
    for i in range(in_f):
        g = module.grid[i].cpu().numpy()
        lo, hi = g[order], g[-order - 1]
        margin = (hi - lo) * 0.05
        x_grids[i] = np.linspace(lo - margin, hi + margin, n_points)

    # Build x_diag (n_pts, in_f): column i holds the grid points for channel i.
    # Call b_splines ONCE — it evaluates each channel's basis independently.
    x_diag = torch.tensor(x_grids, dtype=torch.float32).T  # (n_pts, in_f)
    bases_all = module.b_splines(x_diag)  # (n_pts, in_f, n_basis)

    # spline component: einsum over basis dimension → (n_pts, out_f, in_f)
    spline_all = torch.einsum("pin,oin->poi", bases_all, module.spline_weight)
    # base component: SiLU(x_i) * base_weight[j,i] → (n_pts, out_f, in_f)
    base_all = F.silu(x_diag).unsqueeze(1) * module.base_weight.unsqueeze(0)

    # curves shape: (out_f, in_f, n_pts)
    curves = (base_all + spline_all).detach().cpu().numpy().transpose(1, 2, 0)
    return curves, x_grids


# ── per-function metrics ──────────────────────────────────────────────────────

def compute_metrics(curves, x_grids):
    """
    curves: (out, in, n_pts)
    x_grids: (in, n_pts)
    Returns arrays of shape (out*in,): nls, activity, roughness
    """
    out_f, in_f, n = curves.shape
    nls_all, act_all, rough_all = [], [], []

    for i in range(in_f):
        x = x_grids[i]  # (n,)
        for j in range(out_f):
            f = curves[j, i]  # (n,)
            norm_f = np.linalg.norm(f) + 1e-8

            # Activity
            act = norm_f / np.sqrt(n)
            act_all.append(act)

            # NLS: fit affine, measure residual
            A = np.column_stack([x, np.ones_like(x)])
            coeffs, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
            f_lin = A @ coeffs
            nls = np.linalg.norm(f - f_lin) / norm_f
            nls_all.append(nls)

            # Roughness: mean |second finite difference|
            rough = np.mean(np.abs(np.diff(f, n=2)))
            rough_all.append(rough)

    return np.array(nls_all), np.array(act_all), np.array(rough_all)


# ── MLP effective edge functions ──────────────────────────────────────────────

def mlp_edge_nls(orig_model, n_points=200, x_range=(-3.0, 3.0)):
    """
    f_MLP_{i,j}(x) = sum_k W2[j,k] * ReLU(W1[k,i] * x)
    Vectorized: compute all (i,j) curves at once via batched matmul.
    Returns NLS for all sampled (i,j) pairs.
    """
    block = list(orig_model.blocks)[0]
    ffn = block.ffn
    children = [c for c in ffn.children() if isinstance(c, torch.nn.Linear)]
    W1 = children[0].weight.detach().cpu().numpy()  # (hidden, d_model_in)
    W2 = children[1].weight.detach().cpu().numpy()  # (d_model_out, hidden)

    d_in  = W1.shape[1]   # 384
    hidden = W1.shape[0]  # 768
    d_out  = W2.shape[0]  # 384
    x_vals = np.linspace(x_range[0], x_range[1], n_points)  # (n_pts,)

    # Vectorized over all x at once:
    # acts[k, i, p] = max(W1[k,i] * x_vals[p], 0)  →  (hidden, d_in, n_pts)
    # f[j, i, p]   = W2[j,:] @ acts[:, i, :]        →  (d_out, d_in, n_pts)
    acts = np.maximum(W1[:, :, np.newaxis] * x_vals[np.newaxis, np.newaxis, :], 0)
    # acts: (hidden, d_in, n_pts); W2: (d_out, hidden)
    f_all = np.einsum("jk,kip->jip", W2, acts)  # (d_out, d_in, n_pts)

    # Compute NLS for all (j, i) pairs
    A = np.column_stack([x_vals, np.ones_like(x_vals)])  # (n_pts, 2)
    # Fit affine: f_lin[j,i,:] = x_vals * a[j,i] + b[j,i]
    # Vectorized lstsq: solve A @ [a,b]^T = f_all[j,i,:] for all (j,i)
    f_flat = f_all.reshape(-1, n_points).T  # (n_pts, d_out*d_in)
    coeffs, _, _, _ = np.linalg.lstsq(A, f_flat, rcond=None)  # (2, d_out*d_in)
    f_lin = (A @ coeffs).T.reshape(d_out, d_in, n_points)

    nls = np.linalg.norm(f_all - f_lin, axis=2) / (np.linalg.norm(f_all, axis=2) + 1e-8)
    return nls.flatten()


# ── FPCA ─────────────────────────────────────────────────────────────────────

def run_fpca(curves, n_components=4):
    """Functional PCA on (out*in, n_pts) curve matrix. Returns scores, components, var_ratio."""
    out_f, in_f, n_pts = curves.shape
    X = curves.reshape(-1, n_pts)

    # Normalize to common x-domain [0,1] by treating each row as sampled on [0,1]
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Covariance in function space: (n_pts, n_pts) — use SVD on X directly
    # Truncated SVD via numpy
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]  # (n_components, n_pts)
    scores = U[:, :n_components] * s[:n_components]
    var_ratio = (s[:n_components] ** 2) / (s ** 2).sum()
    return scores, components, var_ratio


# ── token-conditioned activation ──────────────────────────────────────────────

def token_activation_heatmap(model, tokenizer, layer_idx=0):
    from kanprey.kan_layers import KANLinear

    fish_prompts = [
        "are you hungry", "what do you eat", "do you like your tank",
        "what is the temperature", "are you lonely", "what do you do all day",
        "what color are you", "do you have friends",
    ]
    generic_prompts = [
        "what is money", "what is the internet", "what is love",
        "goodbye", "what is time", "can you talk",
        "what is the universe", "what is a computer",
    ]

    # Find KANLinear module for this layer
    target_name = f"blocks.{layer_idx}.ffn.kan"
    target_module = dict(model.named_modules()).get(target_name)
    if target_module is None:
        return None, None

    activations = {"fish": [], "generic": []}

    def hook_fn(category):
        def _hook(module, inp, out):
            activations[category].append(inp[0].detach().cpu())
        return _hook

    for category, prompts in [("fish", fish_prompts), ("generic", generic_prompts)]:
        handle = target_module.register_forward_hook(hook_fn(category))
        for prompt in prompts:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(text).ids
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                model(x)
        handle.remove()

    # Average activation (mean over tokens and prompts) → (d_model,)
    fish_mean = torch.cat(activations["fish"], dim=1).mean(dim=(0, 1)).numpy()
    gen_mean = torch.cat(activations["generic"], dim=1).mean(dim=(0, 1)).numpy()

    # Build outer-product activity matrix: diff[j,i] ~ how much channel i activates
    # differently for fish vs generic. Use the pre-activation mean as a proxy.
    diff = np.outer(fish_mean - gen_mean, fish_mean - gen_mean)
    return fish_mean, gen_mean, diff


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    kan_model = load_kan_model(ROOT / "checkpoints/best.pt")

    print("Loading original GuppyLM...")
    try:
        orig_model, orig_cfg = load_original_guppy()
        has_orig = True
    except Exception as e:
        print(f"  Could not load original GuppyLM: {e}")
        has_orig = False

    from kanprey.dataset import load_tokenizer
    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))

    from kanprey.kan_layers import KANLinear
    kan_layers = {name: mod for name, mod in kan_model.named_modules()
                  if isinstance(mod, KANLinear)}
    print(f"Found {len(kan_layers)} KANLinear layers: {list(kan_layers.keys())}")

    # ── per-layer metrics ─────────────────────────────────────────────────────
    all_nls, all_act, all_rough = [], [], []
    layer_results = {}

    for name, module in kan_layers.items():
        print(f"  Processing {name} ({module.in_features}×{module.out_features})...")
        curves, x_grids = reconstruct_layer(module, n_points=200)
        nls, act, rough = compute_metrics(curves, x_grids)
        all_nls.append(nls)
        all_act.append(act)
        all_rough.append(rough)

        # FPCA for this layer
        scores, components, var_ratio = run_fpca(curves)

        layer_results[name] = {
            "nls_median": float(np.median(nls)),
            "nls_p10": float(np.percentile(nls, 10)),
            "nls_p90": float(np.percentile(nls, 90)),
            "active_frac": float((act > 0.01).mean()),
            "dead_frac": float((act <= 0.01).mean()),
            "nonlinear_frac": float((nls > 0.1).mean()),
            "roughness_median": float(np.median(rough)),
            "fpca_var_ratio": var_ratio[:4].tolist(),
        }
        print(f"    NLS median={layer_results[name]['nls_median']:.3f}, "
              f"nonlinear={layer_results[name]['nonlinear_frac']:.1%}, "
              f"dead={layer_results[name]['dead_frac']:.1%}")

    all_nls_flat = np.concatenate(all_nls)
    all_act_flat = np.concatenate(all_act)

    global_results = {
        "n_functions": int(all_nls_flat.size),
        "nls_median": float(np.median(all_nls_flat)),
        "nonlinear_frac": float((all_nls_flat > 0.1).mean()),
        "dead_frac": float((all_act_flat <= 0.01).mean()),
        "active_frac": float((all_act_flat > 0.01).mean()),
    }
    print(f"\nGlobal: {global_results['n_functions']:,} functions, "
          f"nonlinear={global_results['nonlinear_frac']:.1%}, "
          f"dead={global_results['dead_frac']:.1%}")

    # ── MLP baseline comparison ───────────────────────────────────────────────
    mw_result = None
    if has_orig:
        print("\nComputing MLP effective edge NLS...")
        try:
            mlp_nls = mlp_edge_nls(orig_model, n_points=200)
            stat, pval = mannwhitneyu(all_nls_flat, mlp_nls, alternative="greater")
            mw_result = {"statistic": float(stat), "pvalue": float(pval),
                         "mlp_nls_median": float(np.median(mlp_nls)),
                         "kan_nls_median": float(np.median(all_nls_flat))}
            print(f"  Mann-Whitney U: stat={stat:.0f}, p={pval:.4e}")
            print(f"  KAN NLS median={mw_result['kan_nls_median']:.3f}, "
                  f"MLP NLS median={mw_result['mlp_nls_median']:.3f}")
        except Exception as e:
            print(f"  MLP comparison failed: {e}")

    # ── Figure: NLS histogram ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(all_nls_flat, bins=80, density=True, alpha=0.75,
            color="#2980b9", label=f"KAN ({global_results['n_functions']:,} edges)")
    if mw_result is not None:
        ax.hist(mlp_nls, bins=80, density=True, alpha=0.65,
                color="#e67e22", label=f"MLP effective edges")
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=1, label="NLS = 0.1 threshold")
    ax.set_xlabel("Nonlinearity Score (NLS)")
    ax.set_ylabel("Density")
    ax.set_title("KAN edge function nonlinearity distribution")
    ax.set_yscale("log")
    if mw_result:
        pval = mw_result["pvalue"]
        pval_str = f"p = {pval:.2e}" if pval > 1e-300 else "p < 10⁻³⁰⁰"
        ax.text(0.98, 0.97, f"Mann-Whitney U\n{pval_str}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend()
    fig.tight_layout()
    nls_fig = FIGURES / "nls_hist.pdf"
    fig.savefig(nls_fig)
    plt.close(fig)
    print(f"\nSaved {nls_fig}")

    # ── Figure: FPCA components for first layer ───────────────────────────────
    first_layer_name = list(kan_layers.keys())[0]
    first_module = kan_layers[first_layer_name]
    curves0, x_grids0 = reconstruct_layer(first_module, n_points=200)
    _, components, var_ratio = run_fpca(curves0, n_components=4)

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.5), sharey=False)
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
    x_norm = np.linspace(0, 1, 200)
    for k in range(4):
        axes[k].plot(x_norm, components[k], color=colors[k], linewidth=1.5)
        axes[k].axhline(0, color="gray", linewidth=0.5)
        axes[k].set_title(f"fPC{k+1} ({var_ratio[k]:.1%})", fontsize=9)
        axes[k].set_xlabel("Normalised $x$", fontsize=8)
        axes[k].tick_params(labelsize=7)
    axes[0].set_ylabel("Component value")
    fig.suptitle(f"Functional PCA — {first_layer_name}", fontsize=10)
    fig.tight_layout()
    fpca_fig = FIGURES / "fpca_components.pdf"
    fig.savefig(fpca_fig)
    plt.close(fig)
    print(f"Saved {fpca_fig}")

    # ── Figure: token-conditioned activation heatmap ──────────────────────────
    print("\nComputing token-conditioned activations...")
    try:
        fish_mean, gen_mean, diff = token_activation_heatmap(kan_model, tokenizer, layer_idx=0)
        # Subsample to a 64×64 grid for visualisation
        s = 6  # stride
        diff_small = diff[::s, ::s][:64, :64]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        vmax = np.percentile(np.abs(diff_small), 98)
        im = ax.imshow(diff_small, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Fish − Generic activation diff")
        ax.set_xlabel("Input channel (subsampled)")
        ax.set_ylabel("Input channel (subsampled)")
        ax.set_title("Token-conditioned activation — Layer 0 FFN")
        fig.tight_layout()
        act_fig = FIGURES / "activation_heatmap.pdf"
        fig.savefig(act_fig)
        plt.close(fig)
        print(f"Saved {act_fig}")
    except Exception as e:
        print(f"  Activation heatmap failed: {e}")

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "global": global_results,
        "layers": layer_results,
        "mann_whitney": mw_result,
    }
    out_json = Path(__file__).parent / "interp_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")
    print("\n=== PAPER NUMBERS ===")
    print(f"  Total KAN edge functions: {global_results['n_functions']:,}")
    print(f"  Meaningfully nonlinear (NLS > 0.1): {global_results['nonlinear_frac']:.1%}")
    print(f"  Dead edges (activity ≤ 0.01): {global_results['dead_frac']:.1%}")
    if mw_result:
        print(f"  Mann-Whitney p-value (KAN > MLP): {mw_result['pvalue']:.2e}")


if __name__ == "__main__":
    main()
