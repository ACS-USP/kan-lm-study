#!/usr/bin/env python
"""
Confound-removal retraining for M1: train higher-grid KANs with a lower LR and
longer horizon so they reach grid-2-comparable validation loss, then the edge
audit reflects capacity, not undertraining.

Disables periodic step_*.pt saves (save_interval huge) to avoid filling disk;
only best.pt is kept.

Run from anywhere with PYTHONPATH=<kan-guppylm repo>:
  PYTHONPATH=/.../kan-guppylm uv run python retrain_lowlr.py \
      --grid-size 20 --lr 1e-4 --steps 12000 \
      --checkpoint-dir checkpoints/gridsweep/kan_grid20_s42_lr1e4
"""
import argparse
import sys
from pathlib import Path

REPO = "/Users/felippealves/Documents/GitHub/kan-guppylm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kanprey.config import ModelConfig, TrainConfig
from kanprey.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--grid-size", type=int, required=True)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--min-lr", type=float, default=1e-5)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--checkpoint-dir", required=True)
a = ap.parse_args()

Path(a.checkpoint_dir).mkdir(parents=True, exist_ok=True)

mc = ModelConfig(kan_grid_size=a.grid_size)
tc = TrainConfig(
    learning_rate=a.lr,
    min_lr=a.min_lr,
    max_steps=a.steps,
    batch_size=32,
    seed=a.seed,
    checkpoint_dir=a.checkpoint_dir,
    save_interval=10**9,   # keep only best.pt — avoid filling disk with step_*.pt
)
print(f"[retrain] grid_size={a.grid_size} lr={a.lr} min_lr={a.min_lr} steps={a.steps} "
      f"seed={a.seed} -> {a.checkpoint_dir}")
train(mc, tc, model_type="kan")
