"""
Reconstruct train_log.csv from periodic step checkpoints (which store val_loss=train_loss).
Run once to backfill CSVs for existing checkpoint dirs that predate the CSV logging feature.

Note: step checkpoints store train_loss (labeled val_loss), not the true val_loss.
The best.pt val_loss is the authoritative number. This script gives an approximate
loss curve from training snapshots.

Also parses training.log (text) for KANprey which has step-level val_loss.

Usage:
  uv run --with . python paper/extract_logs_from_checkpoints.py
"""

import csv
import re
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


def extract_from_step_ckpts(ckpt_dir: Path):
    """Read val_loss from step_*.pt files (stored as train snapshot loss)."""
    rows = []
    for p in sorted(ckpt_dir.glob("step_*.pt")):
        step = int(p.stem.split("_")[1])
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        # step checkpoints store: {step, model, val_loss} where val_loss = train_loss at that step
        v = ckpt.get("val_loss", float("nan"))
        rows.append((step, v))
    if not rows:
        return
    out = ckpt_dir / "train_log.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
        for step, v in rows:
            w.writerow([step, f"{v:.4f}", "—", "—", "—"])
    print(f"Wrote {len(rows)} rows → {out}")


def extract_from_training_log(log_path: Path, out_dir: Path):
    """Parse the text training.log for KANprey (has true val_loss per eval step)."""
    pattern = re.compile(
        r"\[step\s+(\d+)\]\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+lr=([\d.e+-]+)\s+elapsed=([\d.]+)s"
    )
    rows = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step, tl, vl, lr, el = m.groups()
                rows.append((int(step), tl, vl, lr, el))
    if not rows:
        print(f"  No matches in {log_path}")
        return
    out = out_dir / "train_log.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    # KANprey — has text log with true val_loss
    log = ROOT / "training.log"
    ckpt = ROOT / "checkpoints"
    if log.exists():
        extract_from_training_log(log, ckpt)

    # KAT v2 and MLPEdge — reconstruct from step checkpoints
    for d in ["kat2", "mlpedge"]:
        extract_from_step_ckpts(ROOT / "checkpoints" / d)
