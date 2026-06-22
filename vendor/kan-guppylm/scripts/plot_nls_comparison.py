#!/usr/bin/env python3
"""
Generate side-by-side NLS histograms: GuppyLM MLP vs Wikitext-103 GPT-2-small MLP.

Usage:
    cd ~/Documents/GitHub/kan-guppylm
    source .venv/bin/activate
    python scripts/plot_nls_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES = Path(__file__).parent.parent / "docs" / "figures"
FIGURES.mkdir(exist_ok=True)


def main():
    # Load Wikitext results
    # Load Wikitext results
    wiki_nls = np.load("results/wikitext_mlp_nls_raw.npy")
    wiki_median = float(np.median(wiki_nls))


    # Load GuppyLM MLP NLS from the paper's compute_interpretability.py output
    # We need to re-run the GuppyLM MLP audit or use cached values.
    # For speed, let's just re-run the mlp_edge_nls on the GuppyLM MLP checkpoint.
    import torch
    from kanprey.model import MLPTransformer
    from kanprey.config import ModelConfig

    ckpt = torch.load("checkpoints/mlp_s42/best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model = MLPTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Reconstruct using the same logic as compute_interpretability.py
    block = list(model.blocks)[0]
    ffn = block.ffn
    W1 = ffn.fc1.weight.detach().cpu().numpy()
    b1 = ffn.fc1.bias.detach().cpu().numpy()
    W2 = ffn.fc2.weight.detach().cpu().numpy()
    b2 = ffn.fc2.bias.detach().cpu().numpy()

    d_in = W1.shape[1]
    hidden = W1.shape[0]
    d_out = W2.shape[0]
    x_vals = np.linspace(-3.0, 3.0, 200)

    acts = np.maximum(W1[:, :, np.newaxis] * x_vals[np.newaxis, np.newaxis, :] + b1[:, np.newaxis, np.newaxis], 0)
    f_all = np.einsum("jk,kip->jip", W2, acts) + b2[:, np.newaxis, np.newaxis]

    A = np.column_stack([x_vals, np.ones_like(x_vals)])
    f_flat = f_all.reshape(-1, 200).T
    coeffs, _, _, _ = np.linalg.lstsq(A, f_flat, rcond=None)
    f_lin = (A @ coeffs).T.reshape(d_out, d_in, 200)
    guppy_nls = np.linalg.norm(f_all - f_lin, axis=2) / (np.linalg.norm(f_all, axis=2) + 1e-8)
    guppy_nls = guppy_nls.flatten()
    guppy_median = float(np.median(guppy_nls))

    print(f"GuppyLM MLP median NLS: {guppy_median:.4f}")
    print(f"Wikitext-103 MLP median NLS: {wiki_median:.4f}")

    # Plot side-by-side histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(0, 1.0, 51)

    ax = axes[0]
    ax.hist(guppy_nls, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(guppy_median, color="darkred", linestyle="--", linewidth=2, label=f"median={guppy_median:.3f}")
    ax.set_xlabel("NLS (nonlinearity score)")
    ax.set_ylabel("Count")
    ax.set_title("GuppyLM MLP (layer 0)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.0)

    ax = axes[1]
    ax.hist(np.asarray(wiki_nls).flatten(), bins=bins, color="forestgreen", edgecolor="white", alpha=0.8)
    ax.axvline(wiki_median, color="darkred", linestyle="--", linewidth=2, label=f"median={wiki_median:.3f}")
    ax.set_xlabel("NLS (nonlinearity score)")
    ax.set_title("Wikitext-103 GPT-2-small MLP (layer 0)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.0)

    fig.suptitle("Effective-edge NLS distributions across domains", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES / "nls_cross_domain.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {out_path}")

    # Also save a comparison table
    table = {
        "guppylm_mlp": {
            "median_nls": guppy_median,
            "mean_nls": float(np.mean(guppy_nls)),
            "nonlinear_frac": float((guppy_nls > 0.1).mean()),
            "n_edges": int(len(guppy_nls)),
        },
        "wikitext_mlp": {
            "median_nls": wiki_median,
            "mean_nls": float(np.mean(wiki_nls)),
            "nonlinear_frac": float((wiki_nls > 0.1).mean()),
            "n_edges": int(len(wiki_nls)),
        },
    }
    with open("results/nls_cross_domain_table.json", "w") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
