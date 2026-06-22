"""Collate BabyLM zero-shot fast results into a comparison table, alongside
the validation-loss ranking, for the 4 critical architectures."""
import re
from pathlib import Path

RESULTS = Path("/Users/felippealves/Documents/GitHub/babylm-eval/results")

MODELS = ["mlp_best", "swiglu_best", "chebyshev_best", "grkan_best"]
LABEL = {"mlp_best": "MLP-4x-GELU", "swiglu_best": "SwiGLU",
         "chebyshev_best": "Chebyshev d3 g8", "grkan_best": "GR-KAN canonical"}
VAL_LOSS = {"mlp_best": 3.8199, "swiglu_best": 3.7700,
            "chebyshev_best": 3.7809, "grkan_best": 3.7997}  # n=10 means

# (task, dataset_dir) -> column label
COLS = [("blimp", "blimp_fast", "BLiMP"),
        ("blimp", "supplement_fast", "BLiMP-suppl"),
        ("ewok", "ewok_fast", "EWoK")]


def read_avg(model, task, dataset):
    p = RESULTS / model / "main" / "zero_shot" / "causal" / task / dataset / "best_temperature_report.txt"
    if not p.exists():
        return None
    txt = p.read_text()
    m = re.search(r"### AVERAGE ACCURACY\s*\n([0-9.]+)", txt)
    return float(m.group(1)) if m else None


rows = {}
for model in MODELS:
    rows[model] = {label: read_avg(model, task, ds) for task, ds, label in COLS}

# Table
hdr = f"{'Architecture':<18}{'val CE':>9}{'BLiMP':>10}{'BLiMP-suppl':>14}{'EWoK':>9}"
print(hdr)
print("-" * len(hdr))
for model in sorted(MODELS, key=lambda m: VAL_LOSS[m]):
    r = rows[model]
    def fmt(v): return f"{v:.2f}" if v is not None else "  —"
    print(f"{LABEL[model]:<18}{VAL_LOSS[model]:>9.4f}"
          f"{fmt(r['BLiMP']):>10}{fmt(r['BLiMP-suppl']):>14}{fmt(r['EWoK']):>9}")

print()
# Ranking comparison: does BLiMP order match val-loss order?
val_rank = sorted(MODELS, key=lambda m: VAL_LOSS[m])  # best (lowest) first
blimp_vals = {m: rows[m]["BLiMP"] for m in MODELS if rows[m]["BLiMP"] is not None}
if len(blimp_vals) == len(MODELS):
    blimp_rank = sorted(MODELS, key=lambda m: -blimp_vals[m])  # best (highest) first
    print("val-loss ranking (best→worst):", [LABEL[m] for m in val_rank])
    print("BLiMP    ranking (best→worst):", [LABEL[m] for m in blimp_rank])
    print("Rankings match:", val_rank == blimp_rank)
