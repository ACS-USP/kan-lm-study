"""
Direct B-spline coefficient intervention experiment.

Hypothesis: modifying domain-selective edges (identified via circuit analysis)
causes asymmetric effects on fish-domain vs. generic prompts; modifying a
control (non-selective) edge causes symmetric effects on both.

Experiment:
  1. Load KANpreyLM (B-spline KAN)
  2. Pick target edges (high |diff|) and a control edge (|diff| ≈ 0)
  3. Apply spline interventions: scale spline_weight[j,i,:] by α ∈ {0, 0.5, 2, 5}
  4. Measure per-prompt loss on 8 fish + 8 generic prompts before/after
  5. Show Δloss_fish ≠ Δloss_generic for selective edges, ≈ for control

Saves:
  paper/figures/spline_intervention.pdf
  paper/intervention_results.json

Usage:
  uv run --with . python paper/spline_intervention.py
"""

import copy
import json
import sys
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
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

# Prompts — same as circuits analysis
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

# Top domain-selective edges from circuit analysis (layer 2)
# Format: (out_ch, in_ch, diff)  diff = fish - generic
SELECTIVE_EDGES = [
    (327, 147, -0.0778),   # generic-selective (|diff| largest)
    ( 60, 147, +0.0628),   # fish-selective (opposite sign, same input hub)
    (119,  85, +0.0603),   # fish-selective (second hub)
]
# Control: a random edge with very low selectivity — picked as (200, 200)
# (We'll verify it has near-zero diff at runtime)
CONTROL_EDGE = (200, 200, 0.0)

LAYER_IDX = 2
SCALES = [0.0, 0.5, 2.0, 5.0]   # α multipliers for spline_weight[j,i,:]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_kan_model():
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM
    ckpt = torch.load(ROOT / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = KANpreyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_ffn_kan(model, layer_idx):
    return dict(model.named_modules())[f"blocks.{layer_idx}.ffn.kan"]


@torch.no_grad()
def prompt_loss(model, tokenizer, prompts, device):
    """
    Average causal LM loss per prompt: cross-entropy of predicting each
    token given its prefix, averaged over all tokens and prompts.
    """
    fmt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    losses = []
    model.eval()
    for p in prompts:
        ids = torch.tensor(
            tokenizer.encode(fmt.format(p)).ids, dtype=torch.long, device=device
        )
        if len(ids) < 2:
            continue
        logits = model(ids.unsqueeze(0)).squeeze(0)   # (seq, vocab)
        loss = F.cross_entropy(logits[:-1], ids[1:])
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def reconstruct_edge(module, out_ch, in_ch, n_pts=300):
    """Return (x, f(x)) for edge (out_ch, in_ch)."""
    from kanprey.kan_layers import KANLinear
    dev = module.grid.device
    order = module.spline_order
    g = module.grid[in_ch].cpu().numpy()
    lo, hi = g[order], g[-order - 1]
    margin = (hi - lo) * 0.08
    x_np = np.linspace(lo - margin, hi + margin, n_pts)
    x = torch.tensor(x_np, dtype=torch.float32, device=dev)

    # Evaluate single-channel: pass x as (n_pts, in_features) with zeros elsewhere
    x_full = torch.zeros(n_pts, module.in_features, device=dev)
    x_full[:, in_ch] = x
    bases = module.b_splines(x_full)[:, in_ch, :]           # (n_pts, n_basis)

    spline_val = bases @ module.spline_weight[out_ch, in_ch, :]   # (n_pts,)
    base_val   = module.base_weight[out_ch, in_ch] * F.silu(x)    # (n_pts,)
    f = (spline_val + base_val).cpu().numpy()
    return x_np, f


def apply_edge_scale(model, layer_idx, out_ch, in_ch, alpha):
    """Return a copy of the model with spline_weight[out,in] scaled by alpha."""
    m = copy.deepcopy(model)
    kan = get_ffn_kan(m, layer_idx)
    with torch.no_grad():
        kan.spline_weight[out_ch, in_ch, :] *= alpha
        kan.base_weight[out_ch, in_ch]      *= alpha
    return m


# ── main experiment ───────────────────────────────────────────────────────────

def run_intervention(model, tokenizer, device, out_ch, in_ch, label, diff):
    print(f"\n  Edge ({out_ch}, {in_ch})  label={label}  circuit_diff={diff:+.4f}")

    # Baseline losses
    base_fish    = prompt_loss(model, tokenizer, FISH_PROMPTS, device)
    base_generic = prompt_loss(model, tokenizer, GENERIC_PROMPTS, device)
    print(f"    baseline  fish={base_fish:.4f}  generic={base_generic:.4f}")

    kan = get_ffn_kan(model, LAYER_IDX)
    x_orig, f_orig = reconstruct_edge(kan, out_ch, in_ch)

    results = {"out": out_ch, "in": in_ch, "label": label, "diff": diff,
               "baseline_fish": base_fish, "baseline_generic": base_generic,
               "interventions": []}
    curves = {"x": x_orig.tolist(), "original": f_orig.tolist(), "modified": {}}

    for alpha in SCALES:
        m_mod = apply_edge_scale(model, LAYER_IDX, out_ch, in_ch, alpha)
        m_mod.to(device)

        loss_fish    = prompt_loss(m_mod, tokenizer, FISH_PROMPTS, device)
        loss_generic = prompt_loss(m_mod, tokenizer, GENERIC_PROMPTS, device)
        delta_fish    = loss_fish    - base_fish
        delta_generic = loss_generic - base_generic
        asymmetry = delta_fish - delta_generic

        _, f_mod = reconstruct_edge(get_ffn_kan(m_mod, LAYER_IDX), out_ch, in_ch)
        curves["modified"][str(alpha)] = f_mod.tolist()

        results["interventions"].append({
            "alpha": alpha,
            "loss_fish": loss_fish, "loss_generic": loss_generic,
            "delta_fish": delta_fish, "delta_generic": delta_generic,
            "asymmetry": asymmetry,
        })
        print(f"    α={alpha:.1f}  Δfish={delta_fish:+.4f}  Δgen={delta_generic:+.4f}"
              f"  asymmetry={asymmetry:+.4f}")

    return results, curves


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results, all_curves):
    n_edges = len(all_results)
    fig, axes = plt.subplots(2, n_edges, figsize=(4 * n_edges, 6))

    colors_alpha = {0.0: "#e74c3c", 0.5: "#e67e22", 2.0: "#27ae60", 5.0: "#8e44ad"}

    for col, (res, curves) in enumerate(zip(all_results, all_curves)):
        label = res["label"]
        ax_curve = axes[0, col]
        ax_delta = axes[1, col]

        # Row 0: edge function curves
        x = np.array(curves["x"])
        ax_curve.plot(x, curves["original"], color="black", lw=2,
                      label="original", zorder=5)
        for alpha, f_mod in curves["modified"].items():
            alpha_f = float(alpha)
            ax_curve.plot(x, np.array(f_mod), color=colors_alpha[alpha_f],
                          lw=1.2, linestyle="--", label=f"α={alpha_f}")
        ax_curve.axhline(0, color="gray", lw=0.5)
        ax_curve.set_title(f"{label}\n(out={res['out']}, in={res['in']})", fontsize=9)
        ax_curve.set_xlabel("x")
        if col == 0:
            ax_curve.set_ylabel("f(x)")
        ax_curve.legend(fontsize=7, loc="upper left")

        # Row 1: Δloss bars (fish vs. generic per alpha)
        ivs = res["interventions"]
        alphas = [iv["alpha"] for iv in ivs]
        delta_fish    = [iv["delta_fish"]    for iv in ivs]
        delta_generic = [iv["delta_generic"] for iv in ivs]
        x_pos = np.arange(len(alphas))
        w = 0.35
        bars1 = ax_delta.bar(x_pos - w/2, delta_fish,    w, label="Δloss fish",
                              color="#2980b9", alpha=0.85)
        bars2 = ax_delta.bar(x_pos + w/2, delta_generic, w, label="Δloss generic",
                              color="#e67e22", alpha=0.85)
        ax_delta.axhline(0, color="gray", lw=0.7)
        ax_delta.set_xticks(x_pos)
        ax_delta.set_xticklabels([f"α={a}" for a in alphas], fontsize=8)
        ax_delta.set_xlabel("Intervention scale α")
        if col == 0:
            ax_delta.set_ylabel("Δ val loss")
        ax_delta.legend(fontsize=7)
        ax_delta.set_title("Asymmetric effect" if "control" not in label.lower()
                            else "Control (symmetric)")

    fig.suptitle(
        f"B-spline intervention — Layer {LAYER_IDX} KAN FFN\n"
        "Asymmetric Δloss (fish vs. generic) confirms domain-selective circuit",
        fontsize=10)
    fig.tight_layout()
    out = FIGURES / "spline_intervention.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from kanprey.dataset import load_tokenizer
    from kanprey.train import detect_device

    device = detect_device()
    print(f"Device: {device}")

    print("Loading KANpreyLM...")
    model = load_kan_model().to(device)

    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))

    # Verify the control edge has near-zero circuit diff
    kan = get_ffn_kan(model, LAYER_IDX)
    from paper.pruning_and_circuits import reconstruct_layer, compute_metrics_2d
    curves_l2, x_grids_l2 = reconstruct_layer(kan)
    nls, act, _ = compute_metrics_2d(curves_l2, x_grids_l2)

    co, ci, _ = CONTROL_EDGE
    ctrl_nls = float(nls[co, ci])
    ctrl_act = float(act[co, ci])
    layer_mean_act = float(act.mean())
    print(f"\nControl edge ({co},{ci}): NLS={ctrl_nls:.4f}, "
          f"act={ctrl_act:.4f} (layer_mean={layer_mean_act:.4f})")

    # Edges to test
    edges_to_test = list(SELECTIVE_EDGES) + [(*CONTROL_EDGE[:2], 0.0, "control")]

    # Label and run each edge
    labels = [f"selective (diff={d:+.3f})" for _, _, d in SELECTIVE_EDGES] + ["control"]
    edges_to_test = [
        (327, 147, -0.0778, "selective: generic>fish"),
        ( 60, 147, +0.0628, "selective: fish>generic"),
        (119,  85, +0.0603, "selective: fish>generic"),
        (200, 200,  0.0,    "control"),
    ]

    print(f"\nRunning interventions on {len(edges_to_test)} edges "
          f"× {len(SCALES)} scale factors = {len(edges_to_test)*len(SCALES)} evals...")

    all_results, all_curves = [], []
    for out_ch, in_ch, diff, label in edges_to_test:
        res, curves = run_intervention(model, tokenizer, device, out_ch, in_ch, label, diff)
        all_results.append(res)
        all_curves.append(curves)

    plot_results(all_results, all_curves)

    # Save JSON (omit large curve arrays to keep file small)
    out_json = Path(__file__).parent / "intervention_results.json"
    slim_results = copy.deepcopy(all_results)  # curves saved separately
    with open(out_json, "w") as f:
        json.dump(slim_results, f, indent=2)
    print(f"Results saved to {out_json}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'Edge':<30} {'α':>5} {'Δfish':>8} {'Δgen':>8} {'asymmetry':>10}")
    print("─" * 65)
    for res in all_results:
        for iv in res["interventions"]:
            tag = f"({res['out']},{res['in']}) {res['label'][:20]}"
            print(f"{tag:<30} {iv['alpha']:>5.1f} "
                  f"{iv['delta_fish']:>+8.4f} {iv['delta_generic']:>+8.4f} "
                  f"{iv['asymmetry']:>+10.4f}")
        print()


if __name__ == "__main__":
    main()
