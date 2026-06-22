#!/usr/bin/env python
"""
audit_grid_sweep.py — Experiment M1 (Review #5).

Runs the paper's KAN edge-function audit across one or more checkpoints (e.g.
trained at different grid sizes) and writes per-checkpoint JSON + a combined CSV.

It reuses:
  * the exact edge reconstruction from kanprey/interpret.py
    (curves f_{j,i}(x) = base_weight * SiLU(x) + sum_k spline_weight[...,k] B(x)),
  * the six-function closed-form library from paper/symbolic_regression.py
    {linear, quadratic, cubic, sigmoid, tanh, x*SiLU(x)}.

Metrics per checkpoint (pooled over the 6 FFN KANLinear layers):
  median NLS, %nonlinear (NLS>tau), %inactive (activity<=0.01),
  per-layer top-4 fPCA variance (min/mean), pooled top-k closed-form coverage.

USAGE (copy into kan-guppylm/scripts/ and run from the kan-guppylm repo root):

  python scripts/audit_grid_sweep.py \
      --checkpoints grid2=checkpoints/gridsweep/kan_grid2_s42/best.pt \
                    grid5=checkpoints/gridsweep/kan_grid5_s42/best.pt \
      --random-control --n-points 200 --top-k 50 --r2-thresh 0.99 \
      --out results/gridsweep_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# --- make `import kanprey` work whether run from repo root or scripts/ -------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kanprey.config import ModelConfig                      # noqa: E402
from kanprey.kan_layers import KANLinear                    # noqa: E402
from kanprey.model import KANpreyLM, KATpreyLM              # noqa: E402


# --------------------------------------------------------------------------- #
# Model loading (mirrors kanprey/interpret.py::_load_kan)
# --------------------------------------------------------------------------- #
def load_kan_model(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    cls = KATpreyLM if ckpt.get("model_type", "kan") == "kat" else KANpreyLM
    model = cls(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("val_loss", None), cfg


def fresh_model(cfg, device: str = "cpu"):
    """Randomly initialized model with the same config (random-init control)."""
    torch.manual_seed(0)
    model = KANpreyLM(cfg)
    model.eval()
    return model.to(device)


def ffn_kanlinear_modules(model):
    """All FFN KANLinear modules (excludes attention KANLinear by name)."""
    mods = []
    for name, m in model.named_modules():
        if isinstance(m, KANLinear) and "att" not in name.lower():
            mods.append((name, m))
    if not mods:  # fall back: take every KANLinear and warn
        mods = [(n, m) for n, m in model.named_modules() if isinstance(m, KANLinear)]
        print("  [warn] no 'ffn'-named KANLinear found; auditing ALL KANLinear modules")
    return mods


# --------------------------------------------------------------------------- #
# Edge reconstruction (verbatim logic from kanprey/interpret.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_layer_functions(module: KANLinear, n_points: int = 200):
    """Returns curves (out,in,n_points), x_grids (in,n_points), x_norm (in,n_points)."""
    module = module.to("cpu")
    in_f = module.in_features
    order = module.spline_order

    x_grids = torch.zeros(in_f, n_points)
    for i in range(in_f):
        g = module.grid[i]
        x_min = g[order].item()
        x_max = g[-(order + 1)].item()
        margin = 0.05 * max(x_max - x_min, 1e-6)
        x_grids[i] = torch.linspace(x_min - margin, x_max + margin, n_points)

    bases = module.b_splines(x_grids.T)                       # (n_points, in, n_basis)
    spline_curves = torch.einsum("pin,oin->oip", bases, module.spline_weight)
    silu_x = F.silu(x_grids)                                  # (in, n_points)
    base_curves = module.base_weight.unsqueeze(-1) * silu_x.unsqueeze(0)
    curves = (spline_curves + base_curves).numpy()

    x_min_v = x_grids.min(dim=1, keepdim=True).values
    x_max_v = x_grids.max(dim=1, keepdim=True).values
    x_norm = ((x_grids - x_min_v) / (x_max_v - x_min_v + 1e-8)).numpy()
    return curves, x_grids.numpy(), x_norm


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def nls_activity(curves_flat: np.ndarray, x_norm_row: np.ndarray):
    """NLS (deviation from best affine fit) and activity (rms) per curve."""
    P = curves_flat.shape[1]
    activity = np.linalg.norm(curves_flat, axis=1) / np.sqrt(P)
    A = np.column_stack([x_norm_row, np.ones(P)])
    coeffs, _, _, _ = np.linalg.lstsq(A, curves_flat.T, rcond=None)
    f_lin = (A @ coeffs).T
    nls = np.linalg.norm(curves_flat - f_lin, axis=1) / (np.linalg.norm(curves_flat, axis=1) + 1e-8)
    return nls, activity


def fpca_top4_variance(curves_flat: np.ndarray) -> float:
    """Fraction of curve variance captured by the top-4 functional PCs (centered SVD)."""
    centered = curves_flat - curves_flat.mean(axis=0, keepdims=True)
    # economy SVD; singular values^2 are proportional to explained variance
    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    total = float(np.sum(s ** 2))
    if total < 1e-12:
        return 1.0
    return float(np.sum(s[:4] ** 2) / total)


def _candidate_designs(x: np.ndarray) -> dict[str, np.ndarray]:
    silu = x / (1.0 + np.exp(-x))
    one = np.ones_like(x)
    return {
        "linear":    np.column_stack([x, one]),
        "quadratic": np.column_stack([x ** 2, x, one]),
        "cubic":     np.column_stack([x ** 3, x ** 2, x, one]),
        "sigmoid":   np.column_stack([1.0 / (1.0 + np.exp(-x)), one]),
        "tanh":      np.column_stack([np.tanh(x), one]),
        "xsilu":     np.column_stack([x * silu, one]),
    }


def best_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
    ss_tot = float(np.var(y) * len(y))
    if ss_tot < 1e-12:
        return "constant", 1.0
    best_name, best_r2 = "linear", -np.inf
    for name, A in _candidate_designs(x).items():
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot
        if r2 > best_r2:
            best_name, best_r2 = name, r2
    return best_name, best_r2


# --------------------------------------------------------------------------- #
# Per-checkpoint audit
# --------------------------------------------------------------------------- #
def audit_model(model, label: str, n_points: int, top_k: int, r2_thresh: float, tau: float):
    layers = ffn_kanlinear_modules(model)
    all_nls, all_act = [], []
    per_layer_fpca = []
    pooled_top_r2 = []
    best_forms: dict[str, int] = {}
    total_funcs = 0

    for name, module in layers:
        curves, x_grids, x_norm = reconstruct_layer_functions(module, n_points=n_points)
        out_f, in_f, P = curves.shape
        total_funcs += out_f * in_f
        cf = curves.reshape(out_f * in_f, P)
        xn = np.tile(x_norm, (out_f, 1, 1)).reshape(out_f * in_f, P)

        nls, act = nls_activity(cf, xn[0])
        all_nls.append(nls)
        all_act.append(act)
        per_layer_fpca.append(fpca_top4_variance(cf))

        # closed-form fit on the top-k most active edges (real x domain)
        order = np.argsort(act)[::-1][:top_k]
        xg = np.tile(x_grids, (out_f, 1, 1)).reshape(out_f * in_f, P)
        for idx in order:
            name_fit, r2 = best_fit_r2(xg[idx], cf[idx])
            pooled_top_r2.append(r2)
            if r2 >= r2_thresh:
                best_forms[name_fit] = best_forms.get(name_fit, 0) + 1

    nls = np.concatenate(all_nls)
    act = np.concatenate(all_act)
    r2_arr = np.array(pooled_top_r2)

    return {
        "label": label,
        "n_ffn_layers": len(layers),
        "n_edge_functions": int(total_funcs),
        "n_basis": int(layers[0][1].n_basis),
        "median_nls": float(np.median(nls)),
        f"pct_nonlinear_gt_{tau}": float(np.mean(nls > tau) * 100.0),
        "pct_inactive_le_0.01": float(np.mean(act <= 0.01) * 100.0),
        "fpca_top4_min_pct": float(np.min(per_layer_fpca) * 100.0),
        "fpca_top4_mean_pct": float(np.mean(per_layer_fpca) * 100.0),
        "fpca_top4_per_layer_pct": [round(v * 100.0, 4) for v in per_layer_fpca],
        f"closed_form_coverage_r2_ge_{r2_thresh}_pct": float(np.mean(r2_arr >= r2_thresh) * 100.0),
        "closed_form_mean_r2": float(r2_arr.mean()),
        "closed_form_best_form_counts": best_forms,
    }


def main():
    ap = argparse.ArgumentParser(description="Grid-size sweep edge audit (Review #5, M1).")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label=path/to/best.pt entries (label optional; defaults to path stem)")
    ap.add_argument("--random-control", action="store_true",
                    help="also audit a fresh untrained model with each checkpoint's config")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--r2-thresh", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.10, help="nonlinearity threshold")
    ap.add_argument("--out", type=str, default="results/gridsweep_audit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in args.checkpoints:
        label, _, path = entry.partition("=")
        if not path:
            path, label = label, Path(label).parent.name
        print(f"\n=== auditing {label}  ({path}) ===")
        model, val_loss, cfg = load_kan_model(path)
        res = audit_model(model, label, args.n_points, args.top_k, args.r2_thresh, args.tau)
        res["best_val_loss"] = val_loss
        res["checkpoint"] = path
        rows.append(res)
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("fpca_top4_per_layer_pct", "closed_form_best_form_counts")},
                         indent=2))
        (out_dir / f"audit_{label}.json").write_text(json.dumps(res, indent=2))

        if args.random_control:
            print(f"--- random-init control for {label} ---")
            rc = audit_model(fresh_model(cfg), f"{label}_randinit",
                             args.n_points, args.top_k, args.r2_thresh, args.tau)
            rc["best_val_loss"] = None
            rc["checkpoint"] = "RANDOM_INIT"
            rows.append(rc)
            (out_dir / f"audit_{label}_randinit.json").write_text(json.dumps(rc, indent=2))

    # combined CSV
    import csv
    cov_key = f"closed_form_coverage_r2_ge_{args.r2_thresh}_pct"
    nl_key = f"pct_nonlinear_gt_{args.tau}"
    cols = ["label", "n_basis", "best_val_loss", "median_nls", nl_key,
            "pct_inactive_le_0.01", "fpca_top4_min_pct", "fpca_top4_mean_pct", cov_key,
            "closed_form_mean_r2", "n_edge_functions"]
    csv_path = out_dir / "gridsweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
