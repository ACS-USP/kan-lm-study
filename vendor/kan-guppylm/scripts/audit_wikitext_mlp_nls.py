#!/usr/bin/env python3
"""
Cross-domain MLP edge-function NLS audit on Wikitext-103 GPT-2-small.

Loads the scale MLP checkpoint and computes effective-edge NLS for the
first FFN layer, using the same formula as paper/compute_interpretability.py.

Usage:
    cd ~/Documents/GitHub/kan-guppylm
    source .venv/bin/activate
    python scripts/audit_wikitext_mlp_nls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CKPT_PATH = Path("checkpoints/scale/mlp_gpt2/best.pt")
OUT_JSON = Path("results/wikitext_mlp_nls.json")


def mlp_edge_nls_from_weights(W1, b1, W2, b2, n_points=200, x_range=(-3.0, 3.0)):
    """
    Compute NLS for all (j,i) effective edges in an MLP FFN.

    W1: (hidden, d_in)  — fc1 weight
    b1: (hidden,)       — fc1 bias
    W2: (d_out, hidden) — fc2 weight
    b2: (d_out,)        — fc2 bias

    Uses ReLU for consistency with compute_interpretability.py.
    """
    W1 = W1.detach().cpu().numpy()
    b1 = b1.detach().cpu().numpy()
    W2 = W2.detach().cpu().numpy()
    b2 = b2.detach().cpu().numpy()

    d_in = W1.shape[1]
    hidden = W1.shape[0]
    d_out = W2.shape[0]
    x_vals = np.linspace(x_range[0], x_range[1], n_points)

    # acts[k, i, p] = ReLU(W1[k,i] * x[p] + b1[k])
    acts = np.maximum(W1[:, :, np.newaxis] * x_vals[np.newaxis, np.newaxis, :] + b1[:, np.newaxis, np.newaxis], 0)

    # f[j, i, p] = W2[j,:] @ acts[:, i, :] + b2[j]
    f_all = np.einsum("jk,kip->jip", W2, acts) + b2[:, np.newaxis, np.newaxis]

    # Fit affine: f_lin = a*x + b
    A = np.column_stack([x_vals, np.ones_like(x_vals)])
    f_flat = f_all.reshape(-1, n_points).T

    coeffs, _, _, _ = np.linalg.lstsq(A, f_flat, rcond=None)
    f_lin = (A @ coeffs).T.reshape(d_out, d_in, n_points)

    nls = np.linalg.norm(f_all - f_lin, axis=2) / (np.linalg.norm(f_all, axis=2) + 1e-8)
    return nls.flatten()


def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    # Extract first FFN layer weights
    W1 = state["blocks.0.ffn.fc1.weight"]
    b1 = state["blocks.0.ffn.fc1.bias"]
    W2 = state["blocks.0.ffn.fc2.weight"]
    b2 = state["blocks.0.ffn.fc2.bias"]

    print(f"  fc1: {W1.shape}  |  fc2: {W2.shape}")

    nls = mlp_edge_nls_from_weights(W1, b1, W2, b2, n_points=200)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    np.save("results/wikitext_mlp_nls_raw.npy", nls)
    print(f"Raw NLS saved to results/wikitext_mlp_nls_raw.npy")
    median_nls = float(np.median(nls))
    mean_nls = float(np.mean(nls))
    p10 = float(np.percentile(nls, 10))
    p90 = float(np.percentile(nls, 90))
    nonlinear_frac = float((nls > 0.1).mean())

    print(f"\nResults for blocks.0.ffn:")
    print(f"  NLS median: {median_nls:.4f}")
    print(f"  NLS mean:   {mean_nls:.4f}")
    print(f"  NLS p10:    {p10:.4f}")
    print(f"  NLS p90:    {p90:.4f}")
    print(f"  Nonlinear (>0.1): {nonlinear_frac:.1%}")
    print(f"  Total edges: {len(nls):,}")

    # Compare with GuppyLM MLP (from paper)
    guppylm_median = 0.268
    diff = median_nls - guppylm_median
    print(f"\nComparison with GuppyLM MLP median NLS ({guppylm_median}):")
    print(f"  Wikitext-103 median: {median_nls:.4f}")
    print(f"  Difference: {diff:+.4f} ({diff/guppylm_median*100:+.1f}%)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "source": "wikitext-103-gpt2-small",
        "checkpoint": str(CKPT_PATH),
        "layer": "blocks.0.ffn",
        "fc1_shape": list(W1.shape),
        "fc2_shape": list(W2.shape),
        "n_edges": len(nls),
        "nls": {
            "median": median_nls,
            "mean": mean_nls,
            "std": float(np.std(nls)),
            "min": float(np.min(nls)),
            "max": float(np.max(nls)),
            "p10": p10,
            "p90": p90,
        },
        "nonlinear_frac": nonlinear_frac,
        "guppylm_comparison": {
            "guppylm_median_nls": guppylm_median,
            "difference": diff,
            "pct_diff": diff / guppylm_median * 100.0,
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()
