#!/usr/bin/env python
"""plot_prune_compare.py — KAN vs MLP pruning baseline (Review #7, M4).

Plots Delta validation loss (loss - unpruned baseline) vs. fraction of FFN
capacity pruned, so the KAN edge curve and the MLP neuron curves are comparable
despite different unpruned baselines. Reads:
  results/prune_curve/prune_curve.csv          (KAN grid-2 activity vs random)
  results/prune_mlp_curve/prune_mlp_curve.csv  (MLP magnitude/activation/random)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1] / "experiments" / "results"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        out.append(float(v) if v not in ("", None) else None)
    return out


kan = load(EXP / "prune_curve" / "prune_curve.csv")
mlp = load(EXP / "prune_mlp_curve" / "prune_mlp_curve.csv")

kf = [float(r["fraction"]) for r in kan]
ka = col(kan, "val_loss_activity")
kr = col(kan, "val_loss_random")
kbase = ka[0]

mf = [float(r["fraction"]) for r in mlp]
mm = col(mlp, "val_loss_magnitude")
mac = col(mlp, "val_loss_activation")
mr = col(mlp, "val_loss_random")
mbase = mm[0]


def delta(ys, base):
    return [(y - base) if y is not None else None for y in ys]


plt.figure(figsize=(6.2, 4.2))
plt.plot(kf, delta(ka, kbase), "o-", color="#009E73", lw=2, label="KAN edge, activity-ranked")
plt.plot(kf, delta(kr, kbase), "o:", color="#009E73", alpha=0.55, label="KAN edge, random")
plt.plot(mf, delta(mac, mbase), "s-", color="#2F6BFF", lw=2, label="MLP neuron, activation saliency")
plt.plot(mf, delta(mm, mbase), "^--", color="#D55E00", label="MLP neuron, weight magnitude")
plt.plot(mf, delta(mr, mbase), "s:", color="#2F6BFF", alpha=0.55, label="MLP neuron, random")
plt.axhline(0.005, color="gray", lw=0.8, ls=":")
plt.text(0.012, 0.0065, "+0.005 nats", fontsize=8, color="gray")
plt.ylim(-0.005, 0.12)
plt.xlabel("fraction of FFN capacity pruned")
plt.ylabel(r"$\Delta$ validation loss vs. unpruned (nats)")
plt.title("Pruning robustness: KAN edge audit vs. MLP neuron baselines")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = Path(__file__).resolve().parent / "prune_compare.pdf"
plt.savefig(out)
plt.savefig(out.with_suffix(".png"), dpi=150)
print("wrote", out)
