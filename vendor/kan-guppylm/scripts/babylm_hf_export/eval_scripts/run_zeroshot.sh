#!/bin/bash
# Run BLiMP / BLiMP-supplement / EWoK fast zero-shot for the 4 critical architectures.
set -u
export PYTHONPATH=/Users/felippealves/Documents/GitHub/kan-guppylm
PY=/Users/felippealves/Documents/GitHub/babylm-eval/.venv/bin/python
HF=/Users/felippealves/Documents/GitHub/kan-guppylm/hf_models/babylm
DATA=/Users/felippealves/Documents/GitHub/babylm-eval/evaluation_data/fast_eval
cd /Users/felippealves/Documents/GitHub/babylm-eval

run () {  # model_dir task data_subdir
  local model=$1 task=$2 sub=$3
  echo ""
  echo "============================================================"
  echo "RUN model=$(basename $model) task=$task data=$sub"
  echo "============================================================"
  $PY -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$HF/$model" --backend causal \
    --task "$task" --data_path "$DATA/$sub" --save_predictions 2>&1 \
    | tr '\r' '\n' | grep -vE 'it/s\]?$' | tail -8
}

for m in mlp_best swiglu_best chebyshev_best grkan_best; do
  # MLP blimp already done, but re-run for uniformity is cheap; skip to save time:
  if [ "$m" != "mlp_best" ]; then run "$m" blimp blimp_fast; fi
  run "$m" blimp supplement_fast
  run "$m" ewok ewok_fast
done
echo ""
echo "ALL ZERO-SHOT RUNS COMPLETE"
