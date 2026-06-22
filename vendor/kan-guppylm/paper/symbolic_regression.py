"""
Symbolic regression of KAN edge functions.

Fits a library of closed-form candidates to the top-50 most active edges per
layer and reports coverage (fraction with R² > 0.99).

Usage:
    uv run python paper/symbolic_regression.py [--checkpoint checkpoints/kat2/best.pt]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kanprey.model import KATpreyLM, KANpreyLM
from kanprey.config import ModelConfig


# ── Candidate symbolic forms ──────────────────────────────────────────────────

def fit_linear(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]

def candidate_forms(x):
    """Return dict of candidate arrays for fitting, given x (n_pts,)."""
    return {
        "linear":   np.vstack([x, np.ones_like(x)]).T,
        "quadratic": np.vstack([x**2, x, np.ones_like(x)]).T,
        "cubic":    np.vstack([x**3, x**2, x, np.ones_like(x)]).T,
        "sigmoid":  np.vstack([1 / (1 + np.exp(-x)), np.ones_like(x)]).T,
        "tanh":     np.vstack([np.tanh(x), np.ones_like(x)]).T,
        "swish":    np.vstack([x / (1 + np.exp(-x)), np.ones_like(x)]).T,
    }

def best_fit(x: np.ndarray, y: np.ndarray):
    """Return (best_form_name, R², fitted_y) for the edge function."""
    ss_tot = np.var(y) * len(y)
    if ss_tot < 1e-12:
        return "linear", 1.0, np.full_like(y, y.mean())

    best_name, best_r2, best_pred = "linear", -np.inf, y
    forms = candidate_forms(x)
    for name, A in forms.items():
        try:
            coeffs, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coeffs
            ss_res = np.sum((y - pred) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2, best_name, best_pred = r2, name, pred
        except Exception:
            continue
    return best_name, best_r2, best_pred


# ── Edge function reconstruction ─────────────────────────────────────────────

@torch.no_grad()
def reconstruct_layer(kan_layer, n_pts: int = 200):
    """
    Reconstruct all edge functions for a KANLinear layer.

    Returns:
        x_vals: (in_features, n_pts) — per-channel x grid
        curves: (out_features, in_features, n_pts) — f_{j,i}(x)
        activity: (out_features, in_features) — ||f||_2
    """
    device = kan_layer.base_weight.device
    in_f = kan_layer.in_features
    out_f = kan_layer.out_features

    # Per-channel grid range from learned knots
    grid = kan_layer.grid.cpu().numpy()  # (in, n_knots)
    x_min = grid[:, kan_layer.spline_order]          # first internal knot
    x_max = grid[:, -(kan_layer.spline_order + 1)]   # last internal knot

    x_vals = np.stack([
        np.linspace(x_min[i], x_max[i], n_pts) for i in range(in_f)
    ])  # (in, n_pts)

    curves = np.zeros((out_f, in_f, n_pts))
    for i in range(in_f):
        xi = torch.tensor(x_vals[i], dtype=torch.float32, device=device)
        # Evaluate base path: SiLU(x) * W_base[:, i]
        base = torch.nn.functional.silu(xi).unsqueeze(1) * kan_layer.base_weight[:, i]
        # Evaluate spline path
        xi_batch = xi.unsqueeze(0)  # (1, n_pts)
        # b_splines expects (batch, in_features); feed each channel one at a time
        # We use a trick: create a (n_pts, in_f) tensor with xi in column i, zeros elsewhere
        xi_full = torch.zeros(n_pts, in_f, device=device)
        xi_full[:, i] = xi
        splines = kan_layer.b_splines(xi_full)       # (n_pts, in_f, n_basis)
        sp_i = splines[:, i, :]                       # (n_pts, n_basis)
        # spline contribution to all outputs: (out_f, n_pts)
        spline_out = (kan_layer.spline_weight[:, i, :] @ sp_i.T)  # (out_f, n_pts)
        curves[:, i, :] = (base.T + spline_out).cpu().numpy()

    activity = np.sqrt(np.mean(curves ** 2, axis=-1))  # (out_f, in_f)
    return x_vals, curves, activity


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_symbolic_regression(model, top_k: int = 50, n_pts: int = 200, r2_thresh: float = 0.99):
    results = {}
    all_forms = []
    all_r2 = []

    for block_idx, block in enumerate(model.blocks):
        kan = block.ffn.kan
        print(f"  Layer {block_idx}: reconstructing {kan.out_features}×{kan.in_features} edges…")
        x_vals, curves, activity = reconstruct_layer(kan, n_pts)

        # Select top-k edges by activity
        flat_idx = np.argsort(activity.ravel())[::-1][:top_k]
        j_idxs, i_idxs = np.unravel_index(flat_idx, activity.shape)

        layer_results = []
        for rank, (j, i) in enumerate(zip(j_idxs, i_idxs)):
            x = x_vals[i]
            y = curves[j, i, :]
            name, r2, pred = best_fit(x, y)
            layer_results.append({
                "j": int(j), "i": int(i),
                "activity": float(activity[j, i]),
                "best_form": name, "r2": float(r2),
            })
            all_forms.append(name)
            all_r2.append(r2)

        results[f"layer_{block_idx}"] = layer_results

    # Summary
    r2_arr = np.array(all_r2)
    coverage = float(np.mean(r2_arr >= r2_thresh))
    form_counts = {}
    for f in all_forms:
        form_counts[f] = form_counts.get(f, 0) + 1
    results["summary"] = {
        "coverage_r2_099": coverage,
        "mean_r2": float(r2_arr.mean()),
        "median_r2": float(np.median(r2_arr)),
        "form_counts": form_counts,
    }
    print(f"\n  Coverage (R²≥{r2_thresh}): {coverage:.1%}")
    print(f"  Mean R²: {r2_arr.mean():.4f}  Median: {np.median(r2_arr):.4f}")
    print(f"  Form distribution: {form_counts}")
    return results


def make_figure(model, results, out_path: Path, n_pts: int = 200, n_show: int = 12):
    """Plot representative edges with their best symbolic fit."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()

    shown = 0
    for block_idx, block in enumerate(model.blocks):
        if shown >= n_show:
            break
        kan = block.ffn.kan
        x_vals, curves, activity = reconstruct_layer(kan, n_pts)
        layer_res = results[f"layer_{block_idx}"][:4]  # top-4 per layer
        for rec in layer_res:
            if shown >= n_show:
                break
            j, i = rec["j"], rec["i"]
            x = x_vals[i]
            y = curves[j, i, :]
            name, r2, pred = best_fit(x, y)

            ax = axes[shown]
            ax.plot(x, y, "b-", lw=1.5, label="KAN edge")
            ax.plot(x, pred, "r--", lw=1.2, label=f"{name} ($R^2$={r2:.3f})")
            ax.set_title(f"L{block_idx} ({j},{i})", fontsize=8)
            ax.set_xlabel("$x$", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
            shown += 1

    for ax in axes[shown:]:
        ax.set_visible(False)

    fig.suptitle("KAN Edge Functions with Best Symbolic Fits", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/kat2/best.pt")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k most active edges to analyze per layer")
    parser.add_argument("--r2-thresh", type=float, default=0.99)
    parser.add_argument("--out-json", default="paper/symbolic_results.json")
    parser.add_argument("--out-fig", default="paper/figures/symbolic_regression.pdf")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", ModelConfig())
    model_type = ckpt.get("model_type", "kat")

    if model_type == "kat":
        model = KATpreyLM(model_cfg)
    else:
        model = KANpreyLM(model_cfg)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {model_type} model from {args.checkpoint}")

    print("Running symbolic regression…")
    results = run_symbolic_regression(model, top_k=args.top_k, r2_thresh=args.r2_thresh)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {args.out_json}")

    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    make_figure(model, results, Path(args.out_fig))


if __name__ == "__main__":
    main()
