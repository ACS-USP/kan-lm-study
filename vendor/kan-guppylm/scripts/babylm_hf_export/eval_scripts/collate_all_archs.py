"""Collate fast BLiMP/EWoK across ALL architectures (critical + supporting + low)
into one table with per-arch mean +/- 95% CI."""
import re, math
from pathlib import Path
from scipy import stats as ss

RESULTS = Path("/Users/felippealves/Documents/GitHub/babylm-eval/results")

# arch key -> (result-dir prefix, n_seeds, val_loss mean, tier, label)
ARCHS = [
    ("mlp",          "mlp",             10, 3.8199, "critical", "MLP-4x-GELU"),
    ("swiglu",       "swiglu",          10, 3.7700, "critical", "SwiGLU"),
    ("chebyshev",    "chebyshev",       10, 3.7809, "critical", "Chebyshev d3 g8"),
    ("grkan",        "grkan",           10, 3.7997, "critical", "GR-KAN canonical"),
    ("grkan_square", "grkan_square",     5, 3.7995, "support",  "GR-KAN square"),
    ("mlpedge_h5",   "mlpedge_h5",       3, 4.0127, "low",      "MLPEdge h5"),
    ("mlpedge_h8",   "mlpedge_h8",       5, 4.0180, "support",  "MLPEdge h8"),
    ("kat_grid2",    "kat_grid2",        3, 4.0745, "low",      "KAT grid2"),
    ("kan_grid2",    "kan_grid2",        5, 4.0787, "support",  "KAN grid2"),
]

def read_avg(name, task, ds):
    p = RESULTS / name / "main" / "zero_shot" / "causal" / task / ds / "best_temperature_report.txt"
    if not p.exists(): return None
    m = re.search(r"### AVERAGE ACCURACY\s*\n([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else None

def collect(prefix, n, task, ds):
    # critical archs use plain prefix (mlp_s42); others use the full prefix too
    seeds = range(42, 42+n) if n != 3 else [42,43,44]
    # critical n=10 -> 42..51; n=5 -> 42..46; n=3 -> 42..44
    if n == 10: seeds = range(42,52)
    elif n == 5: seeds = range(42,47)
    else: seeds = range(42,45)
    return [v for s in seeds if (v := read_avg(f"{prefix}_s{s}", task, ds)) is not None]

def stat(xs):
    n=len(xs); m=sum(xs)/n
    sd=(sum((x-m)**2 for x in xs)/(n-1))**0.5 if n>1 else 0.0
    ci=ss.t.ppf(0.975,n-1)*sd/math.sqrt(n) if n>1 else 0.0
    return m,sd,ci,n

print(f"{'Architecture':<18}{'tier':<9}{'val CE':>8}{'BLiMP':>8}{'±CI':>7}{'EWoK':>8}{'±CI':>7}{'n':>4}")
print("-"*68)
for key,prefix,n,val,tier,label in ARCHS:
    b=collect(prefix,n,"blimp","blimp_fast")
    e=collect(prefix,n,"ewok","ewok_fast")
    if not b: print(f"{label:<18}{tier:<9}{val:>8.4f}  (no data)"); continue
    bm,_,bci,bn=stat(b); em,_,eci,en=stat(e)
    print(f"{label:<18}{tier:<9}{val:>8.4f}{bm:>8.2f}{bci:>7.2f}{em:>8.2f}{eci:>7.2f}{bn:>4}")
print("\nBLiMP chance=50%, EWoK chance=50%.")
