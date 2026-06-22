"""Collate per-seed BabyLM zero-shot results into per-architecture
mean +/- 95% CI, with Welch t-tests vs MLP. n=10 seeds per critical arch."""
import re
import math
from pathlib import Path
from scipy import stats as ss

RESULTS = Path("/Users/felippealves/Documents/GitHub/babylm-eval/results")
ARCHS = ["mlp", "swiglu", "chebyshev", "grkan"]
LABEL = {"mlp": "MLP-4x-GELU", "swiglu": "SwiGLU",
         "chebyshev": "Chebyshev d3 g8", "grkan": "GR-KAN canonical"}
VAL = {"mlp": 3.8199, "swiglu": 3.7700, "chebyshev": 3.7809, "grkan": 3.7997}
SEEDS = list(range(42, 52))


def read_avg(arch, seed, task, ds):
    p = RESULTS / f"{arch}_s{seed}" / "main" / "zero_shot" / "causal" / task / ds / "best_temperature_report.txt"
    if not p.exists():
        return None
    m = re.search(r"### AVERAGE ACCURACY\s*\n([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else None


def collect(task, ds):
    return {a: [v for s in SEEDS if (v := read_avg(a, s, task, ds)) is not None] for a in ARCHS}


def stats(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    ci = ss.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return m, sd, ci, n


def welch(a, b):
    ma, va, na = sum(a)/len(a), (sum((x-sum(a)/len(a))**2 for x in a)/(len(a)-1)), len(a)
    mb, vb, nb = sum(b)/len(b), (sum((x-sum(b)/len(b))**2 for x in b)/(len(b)-1)), len(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return ma-mb, float('nan'), float('nan')
    t = (ma-mb)/se
    df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    p = 2*ss.t.sf(abs(t), df)
    return ma-mb, t, p


for task, ds, title in [("blimp", "blimp_fast", "BLiMP (n=10 seeds)"),
                         ("ewok", "ewok_fast", "EWoK (n=10 seeds)")]:
    data = collect(task, ds)
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Architecture':<18}{'val CE':>8}{'mean':>8}{'sd':>7}{'95% CI':>16}{'n':>4}")
    for a in sorted(ARCHS, key=lambda a: VAL[a]):
        if not data[a]:
            print(f"{LABEL[a]:<18}  (no data)"); continue
        m, sd, ci, n = stats(data[a])
        print(f"{LABEL[a]:<18}{VAL[a]:>8.4f}{m:>8.2f}{sd:>7.2f}   [{m-ci:5.2f},{m+ci:5.2f}]{n:>4}")
    # Welch vs MLP
    if data["mlp"]:
        print(f"\n  vs MLP-4x-GELU baseline (Welch t-test):")
        for a in ["swiglu", "chebyshev", "grkan"]:
            if data[a] and len(data[a]) > 1 and len(data["mlp"]) > 1:
                d, t, p = welch(data[a], data["mlp"])
                verdict = "n.s." if (p != p or p >= 0.05) else ("BETTER" if d > 0 else "WORSE")
                print(f"    {LABEL[a]:<18} Δ={d:+.2f} pts  t={t:+.2f}  p={p:.3f}  -> {verdict}")
    print()
