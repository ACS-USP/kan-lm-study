#!/bin/bash
# Full (non-fast) BLiMP + supplement, n=10 seeds, 4 critical architectures.
set -u
export PYTHONPATH=/Users/felippealves/Documents/GitHub/kan-guppylm
PY=/Users/felippealves/Documents/GitHub/babylm-eval/.venv/bin/python
EXP=/Users/felippealves/Documents/GitHub/kan-guppylm/scripts/babylm_hf_export
TOK=/Users/felippealves/Documents/GitHub/kan-guppylm/tokenizer_babylm.json
CK=/Users/felippealves/Documents/GitHub/kan-guppylm/checkpoints/babylm
HF=/Users/felippealves/Documents/GitHub/kan-guppylm/hf_models/babylm/seeds
DATA=/Users/felippealves/Documents/GitHub/babylm-eval/evaluation_data/full_eval
cd /Users/felippealves/Documents/GitHub/babylm-eval
mkdir -p "$HF"
avg_from () { grep -A1 "AVERAGE ACCURACY" "$1" 2>/dev/null | tail -1; }
prefix_for () { case "$1" in
  mlp) echo "mlp";; swiglu) echo "swiglu";;
  chebyshev) echo "chebyshev_d3_g8";; grkan) echo "grkan_canonical";; esac; }

t0=$(date +%s)
for arch in mlp swiglu chebyshev grkan; do
  pfx=$(prefix_for "$arch")
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    src="$CK/${pfx}_s${seed}/best.pt"; name="${arch}_s${seed}"; dst="$HF/$name"
    [ -f "$src" ] || { echo "MISSING $src"; continue; }
    $PY "$EXP/convert_to_hf.py" "$src" "$dst" --tokenizer "$TOK" >/dev/null 2>&1
    $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$dst" \
      --backend causal --task blimp --data_path "$DATA/blimp_filtered" >/dev/null 2>&1
    b=$(avg_from "results/$name/main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt")
    $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$dst" \
      --backend causal --task blimp --data_path "$DATA/supplement_filtered" >/dev/null 2>&1
    s=$(avg_from "results/$name/main/zero_shot/causal/blimp/supplement_filtered/best_temperature_report.txt")
    echo "$name  BLiMP_full=$b  SUPPL_full=$s  (t=$(( $(date +%s) - t0 ))s)"
    rm -rf "$dst"
  done
done
echo "FULLSET SWEEP COMPLETE ($(( $(date +%s) - t0 ))s)"
