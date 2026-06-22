#!/usr/bin/env python
"""Plot M1 grid-sweep audit results (trained vs random-init) vs basis capacity."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

n_basis = [5, 8, 13, 23]           # grid 2, 5, 10, 20
# trained to convergence (grid 10/20 retrained at lr=3e-4, 16k steps)
val_loss = [0.288, 0.312, 0.371, 3.025]

fpca_tr   = [99.93, 91.0, 75.3, 62.3]
fpca_rand = [99.93, 91.5, 64.5, 37.2]
cf_tr     = [97.0, 18.7, 13.3, 1.0]
cf_rand   = [65.0, 4.0, 0.0, 0.0]

out = Path(__file__).resolve().parent / "results" / "gridsweep_audit" / "gridsweep_capacity.png"
out.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

ax[0].plot(n_basis, fpca_tr, "o-", label="trained")
ax[0].plot(n_basis, fpca_rand, "s--", label="random init")
ax[0].set_title("Top-4 fPCA variance (min over layers)")
ax[0].set_xlabel("n_basis (= grid_size + spline_order)")
ax[0].set_ylabel("% variance in top-4 PCs")
ax[0].legend(); ax[0].set_ylim(30, 102)

ax[1].plot(n_basis, cf_tr, "o-", label="trained")
ax[1].plot(n_basis, cf_rand, "s--", label="random init")
ax[1].set_title("Closed-form coverage (R² > 0.99, top-50/layer)")
ax[1].set_xlabel("n_basis")
ax[1].set_ylabel("% edges fit by 6-fn library")
ax[1].legend(); ax[1].set_ylim(-3, 102)

ax[2].plot(n_basis, val_loss, "o-", color="crimson")
ax[2].set_title("Trained validation loss (8000 steps)")
ax[2].set_xlabel("n_basis")
ax[2].set_ylabel("best val loss (nats)")
ax[2].axhline(0.288, color="gray", ls=":", lw=0.8, label="grid-2 baseline")
ax[2].legend()

for a in ax:
    xt = [5, 8, 13, 23]
    a.set_xticks(xt)
    a.set_xticklabels([f"{nb}\n(g{g})" for nb, g in zip(xt, [2, 5, 10, 20])])

fig.suptitle("M1: KAN edge interpretability vs. basis capacity (GuppyLM, seed 42)")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
