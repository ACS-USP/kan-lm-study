#!/bin/bash
# Full n=10 seed sweep: convert + BLiMP + EWoK for all 10 seeds of each of the
# 4 critical architectures. Produces per-seed reports for CI computation.
# (No associative arrays — macOS bash 3.2 compatible.)
set -u
export PYTHONPATH=/Users/felippealves/Documents/GitHub/kan-guppylm
PY=/Users/felippealves/Documents/GitHub/babylm-eval/.venv/bin/python
EXP=/Users/felippealves/Documents/GitHub/kan-guppylm/scripts/babylm_hf_export
TOK=/Users/felippealves/Documents/GitHub/kan-guppylm/tokenizer_babylm.json
CK=/Users/felippealves/Documents/GitHub/kan-guppylm/checkpoints/babylm
HF=/Users/felippealves/Documents/GitHub/kan-guppylm/hf_models/babylm/seeds
DATA=/Users/felippealves/Documents/GitHub/babylm-eval/evaluation_data/fast_eval
cd /Users/felippealves/Documents/GitHub/babylm-eval
mkdir -p "$HF"

prefix_for () {  # arch -> checkpoint dir prefix
  case "$1" in
    mlp)       echo "mlp" ;;
    swiglu)    echo "swiglu" ;;
    chebyshev) echo "chebyshev_d3_g8" ;;
    grkan)     echo "grkan_canonical" ;;
  esac
}

avg_from () {  # report path -> AVERAGE ACCURACY value
  grep -A1 "AVERAGE ACCURACY" "$1" 2>/dev/null | tail -1
}

t0=$(date +%s)
for arch in mlp swiglu chebyshev grkan; do
  pfx=$(prefix_for "$arch")
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    src="$CK/${pfx}_s${seed}/best.pt"
    name="${arch}_s${seed}"
    dst="$HF/$name"
    if [ ! -f "$src" ]; then echo "MISSING $src"; continue; fi
    $PY "$EXP/convert_to_hf.py" "$src" "$dst" --tokenizer "$TOK" >/dev/null 2>&1
    $PY -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$dst" --backend causal --task blimp \
      --data_path "$DATA/blimp_fast" >/dev/null 2>&1
    blimp=$(avg_from "results/$name/main/zero_shot/causal/blimp/blimp_fast/best_temperature_report.txt")
    $PY -m evaluation_pipeline.sentence_zero_shot.run \
      --model_path_or_name "$dst" --backend causal --task ewok \
      --data_path "$DATA/ewok_fast" >/dev/null 2>&1
    ewok=$(avg_from "results/$name/main/zero_shot/causal/ewok/ewok_fast/best_temperature_report.txt")
    echo "$name  BLiMP=$blimp  EWoK=$ewok  (t=$(( $(date +%s) - t0 ))s)"
    rm -rf "$dst"
  done
done
echo "SEED SWEEP COMPLETE ($(( $(date +%s) - t0 ))s)"
