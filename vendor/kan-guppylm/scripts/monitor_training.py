"""
Live monitor for train_scale.py training jobs.

Reads two sources:
  - train.log   — tqdm output (live step / speed / train loss)
  - train_log.csv — eval checkpoints (val loss, ppl, tok/s)

Usage:
    python scripts/monitor_training.py --dir checkpoints/unit0
    python scripts/monitor_training.py --dir checkpoints/unit0 --interval 30
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

TQDM_RE = re.compile(
    r"(\d+)/(\d+)\s+\[(\S+)<(\S+),\s+([\d.]+)it/s,\s+loss=([\d.]+),\s+lr=([\S]+)\]"
)


def parse_tqdm(log_path: Path):
    """Extract the most recent tqdm progress line from train.log."""
    if not log_path.exists():
        return None
    try:
        with open(log_path, "rb") as f:
            # Read last 64 KB — enough to catch the latest tqdm line
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            chunk = f.read().decode("utf-8", errors="replace")
        matches = TQDM_RE.findall(chunk)
        if not matches:
            return None
        step, total, elapsed, remaining, speed, loss, lr = matches[-1]
        return {
            "step": int(step),
            "total": int(total),
            "elapsed": elapsed,
            "remaining": remaining,
            "speed": float(speed),
            "train_loss": float(loss),
            "lr": lr,
        }
    except Exception:
        return None


def parse_csv(csv_path: Path):
    """Read all eval rows from train_log.csv."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    try:
        with open(csv_path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def eta_str(remaining: str) -> str:
    """Format tqdm remaining time string."""
    return remaining.replace(":", "h ", 1).replace(":", "m ", 1) + "s"


def render(out_dir: Path):
    log_path = out_dir / "train.log"
    csv_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"

    live = parse_tqdm(log_path)
    rows = parse_csv(csv_path)

    os.system("clear")
    width = 62
    print("=" * width)
    print(f"  KanpreyLM Training Monitor — {out_dir}")
    print("=" * width)

    if live:
        pct = 100 * live["step"] / live["total"]
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n  Step  {live['step']:,} / {live['total']:,}  ({pct:.1f}%)")
        print(f"  [{bar}]")
        print(f"\n  Train loss : {live['train_loss']:.4f}")
        print(f"  LR         : {live['lr']}")
        print(f"  Speed      : {live['speed']:.2f} it/s")
        print(f"  Elapsed    : {live['elapsed']}")
        print(f"  ETA        : {eta_str(live['remaining'])}")
    else:
        print("\n  Waiting for training to start…")

    print()
    print("-" * width)

    if rows:
        print(f"  {'Step':>7}  {'Val Loss':>9}  {'Val PPL':>8}  {'tok/s':>7}  {'LR':>10}")
        print(f"  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}")
        for r in rows[-10:]:  # last 10 eval checkpoints
            print(
                f"  {int(r['step']):>7,}  {float(r['val_loss']):>9.4f}"
                f"  {float(r['val_ppl']):>8.1f}"
                f"  {int(float(r['tok_per_sec'])):>7,}"
                f"  {r['lr']:>10}"
            )
        best_row = min(rows, key=lambda r: float(r["val_loss"]))
        print()
        print(f"  Best val loss : {float(best_row['val_loss']):.4f}  "
              f"(ppl {float(best_row['val_ppl']):.1f})  at step {best_row['step']}")
        if best_path.exists():
            mtime = time.strftime("%H:%M:%S", time.localtime(best_path.stat().st_mtime))
            print(f"  best.pt saved : {mtime}")
    else:
        next_eval = 1000
        if live:
            remaining_steps = next_eval - live["step"]
            if remaining_steps > 0 and live["speed"] > 0:
                secs = remaining_steps / live["speed"]
                mins = int(secs // 60)
                print(f"  First eval at step {next_eval:,}  (~{mins} min away)")
            else:
                print(f"  First eval at step {next_eval:,}")
        else:
            print("  No eval checkpoints yet.")

    print()
    print("=" * width)
    print(f"  Refreshed at {time.strftime('%H:%M:%S')}   (Ctrl-C to quit)")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/unit0",
                        help="Output directory passed to train_scale.py")
    parser.add_argument("--interval", type=int, default=15,
                        help="Refresh interval in seconds (default 15)")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    print(f"Monitoring {out_dir}  (refreshing every {args.interval}s)")

    try:
        while True:
            render(out_dir)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
