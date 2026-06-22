"""
Generate all paper figures from experimental data.

Run after:
  1. Ablation training runs complete (checkpoints/mlpedge_h{1,2,5,10,20}/)
  2. marimo notebook run (kanpy/interpret.py) — save NLS/activity CSVs
  3. Training log CSVs exist in each checkpoint dir

Usage:
  uv run --with . python paper/make_figures.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
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


# ── Figure 1: Architecture diagram ───────────────────────────────────────────
# Drawn manually — skip here; source should be a TikZ file or drawn in Inkscape.


# ── Figure 2: Training loss curves ───────────────────────────────────────────

def plot_loss_curves():
    configs = [
        # (label, color, ls, ckpt_dir)
        ("KANprey (B-spline FFN)", "#e67e22", ":", "checkpoints"),
        ("KATprey v2 (B-spline + KAN attn)", "#2980b9", "-", "checkpoints/kat2"),
        ("MLPEdge (h=5)", "#27ae60", "-", "checkpoints/mlpedge"),
    ]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color, ls, ckpt_dir in configs:
        log = ROOT / ckpt_dir / "train_log.csv"
        if not log.exists():
            print(f"  [skip] {label}: {log} not found")
            continue
        steps, val_losses = [], []
        with open(log) as f:
            for row in csv.DictReader(f):
                vl = row.get("val_loss", row.get("train_loss", ""))
                try:
                    val_losses.append(float(vl))
                    steps.append(int(row["step"]))
                except ValueError:
                    pass
        ax.plot(steps, val_losses, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss during training")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "loss_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Spline vs MLPEdge edge visualization ───────────────────────────

def plot_edge_compare(layer_idx=0, n_pairs=6):
    sys.path.insert(0, str(ROOT))
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM, MLPEdgepreyLM
    from kanprey.kan_layers import KANLinear, MLPEdgeLinear
    import torch.nn.functional as F

    device = torch.device("cpu")

    def load(ckpt_path, cls):
        ckpt = torch.load(ROOT / ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt.get("model_cfg", ModelConfig())
        model = cls(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    try:
        kan_model = load("checkpoints/best.pt", KANpreyLM)
        mlp_model = load("checkpoints/mlpedge/best.pt", MLPEdgepreyLM)
    except FileNotFoundError as e:
        print(f"  [skip] edge compare: {e}")
        return

    # Get layer modules
    kan_ffn = dict(kan_model.named_modules())[f"blocks.{layer_idx}.ffn.kan"]
    mlp_ffn = dict(mlp_model.named_modules())[f"blocks.{layer_idx}.ffn.edge"]

    assert isinstance(kan_ffn, KANLinear)
    assert isinstance(mlp_ffn, MLPEdgeLinear)

    # Sample (i,j) pairs
    rng = np.random.default_rng(42)
    in_dim = kan_ffn.in_features
    out_dim = kan_ffn.out_features
    pairs = [(int(rng.integers(0, in_dim)), int(rng.integers(0, out_dim)))
             for _ in range(n_pairs)]

    fig, axes = plt.subplots(2, n_pairs, figsize=(2.2 * n_pairs, 4), sharey=False)

    for col, (i, j) in enumerate(pairs):
        # KAN spline curve
        g = kan_ffn.grid[i].cpu().numpy()
        x_range = np.linspace(g[kan_ffn.spline_order], g[-kan_ffn.spline_order - 1], 200)
        x_t = torch.tensor(x_range, dtype=torch.float32).unsqueeze(1)  # (200, 1)
        with torch.no_grad():
            bases = kan_ffn.b_splines(x_t)  # (200, 1, n_basis)
            spline_out = torch.einsum("bik,oik->bo", bases,
                                     kan_ffn.spline_weight[:, i:i+1, :])[:, 0]  # (200,)
            base_out = F.silu(x_t[:, 0]) * kan_ffn.base_weight[j, i]
            kan_curve = (base_out + spline_out[:]).cpu().numpy()

        # MLPEdge curve: H = σ(x * W1[i,:] + b1[i,:]), f_{i,j} = H @ W2[j, i, :]
        x_t2 = torch.tensor(x_range, dtype=torch.float32)
        with torch.no_grad():
            H = mlp_ffn.activation(x_t2.unsqueeze(1) * mlp_ffn.W1[i, :] + mlp_ffn.b1[i, :])  # (200, h)
            mlp_curve = (H @ mlp_ffn.W2[j, i, :]).cpu().numpy()  # (200,)

        for row, (curve, title) in enumerate([(kan_curve, "KAN"), (mlp_curve, "MLPEdge")]):
            ax = axes[row][col]
            ax.plot(x_range, curve, linewidth=1.2, color="#2980b9" if row == 0 else "#27ae60")
            ax.set_title(f"$f_{{{i},{j}}}$", fontsize=9)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            if col == 0:
                ax.set_ylabel(title)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Layer {layer_idx}: learned edge functions $f_{{i,j}}(x)$",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    out = FIGURES / "edge_compare.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Hidden-size ablation ───────────────────────────────────────────

def plot_ablation():
    hidden_sizes = [1, 2, 5, 10, 20]
    val_losses = []
    latencies_ms = []

    import time

    sys.path.insert(0, str(ROOT))
    from kanprey.config import ModelConfig
    from kanprey.model import MLPEdgepreyLM
    from kanprey.dataset import load_tokenizer

    device = torch.device("cpu")
    tokenizer = load_tokenizer(str(ROOT / "tokenizer.json"))

    for h in hidden_sizes:
        ckpt_path = ROOT / f"checkpoints/mlpedge_h{h}/best.pt"
        if not ckpt_path.exists():
            print(f"  [skip] h={h}: checkpoint not found")
            val_losses.append(None)
            latencies_ms.append(None)
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        val_losses.append(float(ckpt.get("val_loss", float("nan"))))

        # Measure latency
        cfg = ckpt.get("model_cfg", ModelConfig())
        model = MLPEdgepreyLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        prompt = "<|im_start|>user\nare you hungry<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        times = []
        with torch.no_grad():
            for _ in range(5):
                t0 = time.perf_counter()
                model.generate(x, max_new_tokens=64, temperature=0.7, top_k=50)
                times.append(time.perf_counter() - t0)
        latencies_ms.append(np.mean(times[1:]) * 1000)

    valid = [(h, v, l) for h, v, l in zip(hidden_sizes, val_losses, latencies_ms)
             if v is not None]
    if not valid:
        print("  [skip] ablation: no checkpoints found")
        return

    hs, vs, ls = zip(*valid)
    x = np.arange(len(hs))
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - 0.2, vs, 0.35, label="Val loss", color="#2980b9", alpha=0.8)
    bars2 = ax2.bar(x + 0.2, ls, 0.35, label="Latency (ms)", color="#e67e22", alpha=0.8)

    ax1.set_xlabel("Per-edge hidden size $h$")
    ax1.set_ylabel("Validation loss", color="#2980b9")
    ax2.set_ylabel("Inference latency (ms)", color="#e67e22")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(h) for h in hs])
    ax1.set_title("MLPEdge hidden-size ablation")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "ablation.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures...")
    plot_loss_curves()
    plot_edge_compare()
    plot_ablation()
    print("Done. Remaining figures (NLS histogram, activation heatmap) "
          "are generated by kanpy/interpret.py — run it and save outputs to paper/figures/.")
