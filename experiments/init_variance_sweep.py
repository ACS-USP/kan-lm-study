#!/usr/bin/env python
"""
Reconcile Finding 2: the paper's random-init grid-2 control reports top-4 fPCA
= 82.3% with "high-frequency oscillations", while a fresh-init grid-2 KAN
(spline_weight ~ N(0,0.1), the model's actual init) gives ~99.9%.

This sweeps the spline-weight init std to find what (if anything) reproduces the
paper's 82.3% / high-frequency control, characterizing whether the paper's
control used a non-default (rougher) initialization.

Run with PYTHONPATH=<kan-guppylm>:
  PYTHONPATH=/.../kan-guppylm uv run python init_variance_sweep.py
"""
import sys
from pathlib import Path

REPO = "/Users/felippealves/Documents/GitHub/kan-guppylm"
EXP = str(Path(__file__).resolve().parent)
for p in (REPO, EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from kanprey.config import ModelConfig             # noqa: E402
from kanprey.model import KANpreyLM                # noqa: E402
from audit_grid_sweep import audit_model, ffn_kanlinear_modules  # noqa: E402

GRID = 2
STDS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]   # 0.1 = the model's actual init

print(f"grid_size={GRID}  (n_basis = {GRID}+3 = {GRID+3})")
print(f"{'spline_std':>10} {'fPCA_top4_min%':>14} {'closedform_R2>.99%':>18} {'median_NLS':>11}")
rows = []
for std in STDS:
    torch.manual_seed(0)
    cfg = ModelConfig(kan_grid_size=GRID)
    m = KANpreyLM(cfg)
    m.eval()
    with torch.no_grad():
        for _, mod in ffn_kanlinear_modules(m):
            mod.spline_weight.normal_(0.0, std)
    res = audit_model(m, f"std{std}", n_points=200, top_k=50, r2_thresh=0.99, tau=0.10)
    cov = res["closed_form_coverage_r2_ge_0.99_pct"]
    print(f"{std:>10} {res['fpca_top4_min_pct']:>14.2f} {cov:>18.2f} {res['median_nls']:>11.3f}")
    rows.append((std, res["fpca_top4_min_pct"], cov, res["median_nls"]))

print("\nReference: paper random-init control = 82.3% fPCA / 34.7% closed-form.")
print("The model's actual init is std=0.1.")
