#!/usr/bin/env python
"""BabyLM Phase 3.4 post-training sanity check.

Verifies, for all 61 runs:
  - present in manifest with exit_code == 0
  - best.pt exists on disk and is non-trivially sized
  - manifest best_val_loss is in the sane BabyLM range (1.0 < CE < 8.0), not NaN
  - manifest best_val_loss agrees with min(val_loss) in train_log.csv
Then computes per-architecture mean/std/min/max and n=10 Welch CIs vs MLP
for the critical rows.
"""
import json
import csv
import math
from pathlib import Path

CKPT = Path("/Users/felippealves/Documents/GitHub/kan-guppylm/checkpoints/babylm")
manifest = json.loads((CKPT / "manifest.json").read_text())
runs = manifest["runs"]

# Expected tier structure from the implementation plan
EXPECTED = {
    "mlp":              ("mlp_s",                list(range(42, 52))),  # 10 (critical)
    "swiglu":           ("swiglu_s",             list(range(42, 52))),  # 10 (critical)
    "chebyshev_d3_g8":  ("chebyshev_d3_g8_s",    list(range(42, 52))),  # 10 (critical)
    "grkan_canonical":  ("grkan_canonical_s",    list(range(42, 52))),  # 10 (critical)
    "grkan_square":     ("grkan_square_s",       list(range(42, 47))),  # 5  (supporting)
    "kan_grid2":        ("kan_grid2_s",          list(range(42, 47))),  # 5  (supporting)
    "mlpedge_h8":       ("mlpedge_h8_s",         list(range(42, 47))),  # 5  (supporting)
    "kat_grid2":        ("kat_grid2_s",          [42, 43, 44]),         # 3  (low)
    "mlpedge_h5":       ("mlpedge_h5_s",         [42, 43, 44]),         # 3  (low)
}

problems = []
arch_losses = {}

print("=" * 78)
print("PER-RUN VERIFICATION")
print("=" * 78)

n_ok = 0
n_total = 0
for arch, (prefix, seeds) in EXPECTED.items():
    losses = []
    for seed in seeds:
        n_total += 1
        ckpt_name = f"{prefix}{seed}"
        # find the run_key in manifest (model/ckpt_name)
        keys = [k for k in runs if k.endswith("/" + ckpt_name)]
        if not keys:
            problems.append(f"MISSING in manifest: {ckpt_name}")
            continue
        entry = runs[keys[0]]

        # exit code
        if entry.get("exit_code") != 0:
            problems.append(f"NONZERO EXIT: {ckpt_name} exit={entry.get('exit_code')}")
            continue

        # best.pt on disk
        bp = CKPT / ckpt_name / "best.pt"
        if not bp.exists():
            problems.append(f"MISSING best.pt: {ckpt_name}")
            continue
        if bp.stat().st_size < 1_000_000:
            problems.append(f"TINY best.pt ({bp.stat().st_size}B): {ckpt_name}")
            continue

        # val loss sanity
        vl = entry.get("best_val_loss")
        if vl is None or (isinstance(vl, float) and math.isnan(vl)):
            problems.append(f"NaN/None best_val_loss: {ckpt_name}")
            continue
        if not (1.0 < vl < 8.0):
            problems.append(f"OUT-OF-RANGE val loss {vl}: {ckpt_name}")
            continue

        # cross-check against train_log.csv
        log = CKPT / ckpt_name / "train_log.csv"
        if log.exists():
            with open(log) as f:
                rows = list(csv.DictReader(f))
            log_losses = [float(r["val_loss"]) for r in rows
                          if r.get("val_loss") not in (None, "")]
            if log_losses:
                log_min = min(log_losses)
                if abs(log_min - vl) > 0.01:
                    problems.append(
                        f"MANIFEST/LOG MISMATCH {ckpt_name}: manifest={vl} log_min={log_min:.4f}")
        losses.append(vl)
        n_ok += 1

    arch_losses[arch] = losses
    if losses:
        n = len(losses)
        mean = sum(losses) / n
        std = (sum((x - mean) ** 2 for x in losses) / (n - 1)) ** 0.5 if n > 1 else 0.0
        print(f"  {arch:<18} n={n:<2}  ok={len(losses)}/{len(seeds)}  "
              f"mean={mean:.4f}  std={std:.4f}  min={min(losses):.4f}  max={max(losses):.4f}")

print()
print(f"Runs verified OK: {n_ok}/{n_total}")
print(f"Manifest claims:  total_runs={manifest.get('total_runs')}  "
      f"total_failed={manifest.get('total_failed')}  "
      f"total_elapsed_h={manifest.get('total_elapsed_h')}")

print()
print("=" * 78)
print("PROBLEMS")
print("=" * 78)
if problems:
    for p in problems:
        print("  ✗ " + p)
else:
    print("  none — all 61 runs valid (exit 0, best.pt present, loss in range, log-consistent)")

# ---- n=10 Welch CIs vs MLP on critical rows ----
def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, v, n

def welch(a, b):
    """Return (mean_diff a-b, ci_lo, ci_hi, t, df) at 95%."""
    ma, va, na = stats(a)
    mb, vb, nb = stats(b)
    se = math.sqrt(va / na + vb / nb)
    diff = ma - mb
    t = diff / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return diff, t, df, se

print()
print("=" * 78)
print("HEADLINE: critical-row means vs MLP (BabyLM validation cross-entropy)")
print("=" * 78)
mlp = arch_losses["mlp"]
mlp_mean = sum(mlp) / len(mlp)
print(f"  MLP-4x-GELU baseline: {mlp_mean:.4f}  (n={len(mlp)})\n")
try:
    from scipy import stats as ss
    have_scipy = True
except Exception:
    have_scipy = False

for arch in ["swiglu", "chebyshev_d3_g8", "grkan_canonical"]:
    xs = arch_losses[arch]
    m = sum(xs) / len(xs)
    diff, t, df, se = welch(xs, mlp)
    if have_scipy:
        tcrit = ss.t.ppf(0.975, df)
        p = 2 * ss.t.sf(abs(t), df)
    else:
        tcrit = 2.10  # ~t(18) approx
        p = float("nan")
    lo, hi = diff - tcrit * se, diff + tcrit * se
    sign = "BEATS MLP" if hi < 0 else ("WORSE than MLP" if lo > 0 else "n.s. vs MLP")
    pstr = f"p={p:.2e}" if have_scipy else "p=?(no scipy)"
    print(f"  {arch:<18} mean={m:.4f}  Δ(arch−MLP)={diff:+.4f}  "
          f"95%CI=[{lo:+.4f},{hi:+.4f}]  t({df:.1f})={t:.2f}  {pstr}  → {sign}")
print()
print("scipy available:", have_scipy)
