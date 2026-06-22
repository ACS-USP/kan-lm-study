#!/bin/bash
# Extend zero-shot BLiMP+EWoK to the supporting + low-priority architectures.
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

avg_from () { grep -A1 "AVERAGE ACCURACY" "$1" 2>/dev/null | tail -1; }

# "checkpoint_prefix:seedlist"
JOBS="
grkan_square:42 43 44 45 46
kan_grid2:42 43 44 45 46
mlpedge_h8:42 43 44 45 46
kat_grid2:42 43 44
mlpedge_h5:42 43 44
"

t0=$(date +%s)
echo "$JOBS" | while IFS=: read pfx seeds; do
  [ -z "$pfx" ] && continue
  for seed in $seeds; do
    src="$CK/${pfx}_s${seed}/best.pt"
    name="${pfx}_s${seed}"
    dst="$HF/$name"
    if [ ! -f "$src" ]; then echo "MISSING $src"; continue; fi
    $PY "$EXP/convert_to_hf.py" "$src" "$dst" --tokenizer "$TOK" >/dev/null 2>&1
    $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$dst" \
      --backend causal --task blimp --data_path "$DATA/blimp_fast" >/dev/null 2>&1
    blimp=$(avg_from "results/$name/main/zero_shot/causal/blimp/blimp_fast/best_temperature_report.txt")
    $PY -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name "$dst" \
      --backend causal --task ewok --data_path "$DATA/ewok_fast" >/dev/null 2>&1
    ewok=$(avg_from "results/$name/main/zero_shot/causal/ewok/ewok_fast/best_temperature_report.txt")
    echo "$name  BLiMP=$blimp  EWoK=$ewok  (t=$(( $(date +%s) - t0 ))s)"
    rm -rf "$dst"
  done
done
echo "EXTEND SWEEP COMPLETE ($(( $(date +%s) - t0 ))s)"
