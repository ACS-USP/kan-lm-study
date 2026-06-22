#!/usr/bin/env python
"""
Sequential launcher for the BabyLM seed matrix on local MPS.

Runs one training job at a time, logs progress to a JSON manifest,
continues on failure, and supports resume-after-interrupt.

Usage:
    uv run python scripts/babylm_launch.py              # all 61 runs
    uv run python scripts/babylm_launch.py --dry-run     # print commands only
    uv run python scripts/babylm_launch.py --tier 1      # critical rows only
    uv run python scripts/babylm_launch.py --skip-done   # resume after interrupt
"""
import os
import subprocess
import sys
import time
import json
import csv
from pathlib import Path
from datetime import datetime

# ── Environment ──────────────────────────────────────────────────

def _load_dotenv() -> dict[str, str]:
    """Load .env from the project root and return a dict of HF-relevant vars."""
    env_vars: dict[str, str] = {}
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.exists():
        return env_vars
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("\"'")
        env_vars[key] = val
    return env_vars

_DOTENV = _load_dotenv()

def _subprocess_env() -> dict[str, str]:
    """Environment for each training subprocess.

    - HF_TOKEN from .env (avoids rate-limit warnings)
    - HF_DATASETS_OFFLINE=1 (pre-tokenized cache is built upfront; no network needed)
    - inherit the parent process environment for everything else
    """
    env = os.environ.copy()
    if "HF_TOKEN" in _DOTENV:
        env["HF_TOKEN"] = _DOTENV["HF_TOKEN"]
    env["HF_DATASETS_OFFLINE"] = "1"
    return env


def _build_token_cache_once():
    """Build the pre-tokenized .npy cache if it doesn't exist.

    Called in the launcher process (with network access) before any training
    subprocess runs.  After this, every subprocess runs fully offline.
    """
    from pathlib import Path as _Path
    cache_dir = _Path.home() / ".cache" / "kanprey"
    if (cache_dir / "babylm_8192_train.npy").exists() and \
       (cache_dir / "babylm_8192_val.npy").exists():
        print("── Token cache already exists — skipping build. ──\n")
        return

    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from kanprey.dataset_babylm import (
        load_babylm_tokenizer, BabyLMDataset, DEFAULT_TOKENIZER_PATH,
    )
    tok = load_babylm_tokenizer(DEFAULT_TOKENIZER_PATH)
    print("── Pre-building token cache (one-time) ──")
    for split in ("train", "val"):
        BabyLMDataset(split=split, tokenizer=tok)
    print("── Cache ready.  All training runs will use HF_DATASETS_OFFLINE=1. ──\n")


# ── Seed matrix ──────────────────────────────────────────────────
# (model, checkpoint_prefix, seeds, extra_cli_args)
TIER1 = [
    # Critical: 10 seeds each — carry the replacement claim
    ("mlp",    "mlp",                list(range(42, 52)), ""),
    ("swiglu", "swiglu",             list(range(42, 52)), ""),
    ("basis",  "chebyshev_d3_g8",    list(range(42, 52)),
     "--basis-family chebyshev --basis-degree 3 --basis-groups 8 --basis-input-norm tanh"),
    ("grkan",  "grkan_canonical",    list(range(42, 52)), ""),
]
TIER2 = [
    # Supporting: 5 seeds each
    ("grkan",   "grkan_square",      list(range(42, 47)), "--grkan-denominator square"),
    ("kan",     "kan_grid2",         list(range(42, 47)), "--grid-size 2"),
    ("mlpedge", "mlpedge_h8",        list(range(42, 47)), "--mlp-edge-hidden 8"),
]
TIER3 = [
    # Low priority: 3 seeds each
    ("kat",     "kat_grid2",         [42, 43, 44], "--grid-size 2"),
    ("mlpedge", "mlpedge_h5",        [42, 43, 44], "--mlp-edge-hidden 5"),
]

BASE_CMD = (
    "uv run python -m kanprey.train "
    "--dataset babylm "
    "--steps 8000 --batch-size 32 "
    "--model {model} --seed {seed} "
    "--checkpoint-dir checkpoints/babylm/{ckpt_name}"
)

MANIFEST_PATH = Path("checkpoints/babylm/manifest.json")


# ── Manifest helpers ─────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"runs": {}, "started_at": None, "tiers_completed": []}


def save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def extract_best_val_loss(ckpt_name: str) -> float | None:
    """Read best val_loss from the training CSV log."""
    log_path = Path(f"checkpoints/babylm/{ckpt_name}/train_log.csv")
    if not log_path.exists():
        return None
    try:
        with open(log_path) as f:
            reader = csv.DictReader(f)
            losses = [float(row["val_loss"]) for row in reader if row.get("val_loss")]
            return round(min(losses), 6) if losses else None
    except Exception:
        return None


# ── Run one job ──────────────────────────────────────────────────

def run_one(
    model: str,
    ckpt_prefix: str,
    seed: int,
    extra: str,
    manifest: dict,
    skip_done: bool,
) -> bool:
    ckpt_name = f"{ckpt_prefix}_s{seed}"
    run_key = f"{model}/{ckpt_name}"

    # Skip completed runs when resuming
    if skip_done and run_key in manifest["runs"]:
        prev = manifest["runs"][run_key]
        if prev.get("exit_code") == 0:
            ckpt_dir = Path(f"checkpoints/babylm/{ckpt_name}")
            if ckpt_dir.exists() and (ckpt_dir / "best.pt").exists():
                best = prev.get("best_val_loss", "?")
                print(f"  SKIP {run_key} — already complete (best_loss={best})")
                return True

    cmd = BASE_CMD.format(model=model, seed=seed, ckpt_name=ckpt_name)
    if extra:
        cmd += " " + extra

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {run_key}")
    print(f"  {cmd}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, shell=True, env=_subprocess_env())
    elapsed = time.time() - start

    entry = {
        "command": cmd,
        "exit_code": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "finished_at": datetime.now().isoformat(),
    }
    entry["best_val_loss"] = extract_best_val_loss(ckpt_name)

    manifest["runs"][run_key] = entry
    save_manifest(manifest)

    status = "OK" if result.returncode == 0 else f"FAIL (exit={result.returncode})"
    print(f"  → {status}  elapsed={elapsed/60:.1f}min  best_val_loss={entry['best_val_loss']}")
    return result.returncode == 0


# ── Main ─────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    skip_done = "--skip-done" in sys.argv
    tiers_to_run = {"1", "2", "3"}
    for arg in sys.argv[1:]:
        if arg.startswith("--tier="):
            tiers_to_run = {arg.split("=")[1]}

    manifest = load_manifest()
    if manifest["started_at"] is None:
        manifest["started_at"] = datetime.now().isoformat()

    all_tiers: list[tuple[str, list]] = []
    if "1" in tiers_to_run:
        all_tiers.append(("Tier 1 — Critical (10 seeds)", TIER1))
    if "2" in tiers_to_run:
        all_tiers.append(("Tier 2 — Supporting (5 seeds)", TIER2))
    if "3" in tiers_to_run:
        all_tiers.append(("Tier 3 — Low priority (3 seeds)", TIER3))

    total = sum(len(seeds) for _, _, seeds, _ in sum([t for _, t in all_tiers], []))
    print(f"BabyLM seed matrix: {total} runs across {len(all_tiers)} tiers")
    print(f"  dry_run={dry_run}  skip_done={skip_done}")
    if dry_run:
        print("  (printing commands only — not executing)\n")
    else:
        _build_token_cache_once()

    done, failed = 0, 0
    t0 = time.time()

    for tier_name, tier in all_tiers:
        print(f"\n{'─'*60}")
        print(f"  {tier_name}")
        print(f"{'─'*60}")
        for model, ckpt_prefix, seeds, extra in tier:
            for seed in seeds:
                if dry_run:
                    ckpt_name = f"{ckpt_prefix}_s{seed}"
                    cmd = BASE_CMD.format(model=model, seed=seed, ckpt_name=ckpt_name)
                    if extra:
                        cmd += " " + extra
                    print(f"  [{model}/{ckpt_name}] {cmd}")
                    done += 1
                else:
                    ok = run_one(model, ckpt_prefix, seed, extra, manifest, skip_done)
                    done += 1
                    if not ok:
                        failed += 1
                    # Estimate remaining time
                    elapsed = (time.time() - t0) / 3600
                    frac = done / total
                    if frac > 0:
                        eta = elapsed / frac - elapsed
                        print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)  "
                              f"elapsed={elapsed:.1f}h  ETA={eta:.1f}h  failed={failed}")

    elapsed = (time.time() - t0) / 3600

    # Mark completed tiers
    manifest["tiers_completed"] = list(tiers_to_run)
    manifest["total_elapsed_h"] = round(elapsed, 1)
    manifest["total_runs"] = done
    manifest["total_failed"] = failed
    save_manifest(manifest)

    print(f"\n{'='*60}")
    print(f"Done. {done} runs, {failed} failed, {elapsed:.1f} hours.")
    if failed > 0:
        print(f"WARNING: {failed} runs failed. Check manifest: {MANIFEST_PATH}")
        print(f"Re-run with --skip-done after fixing failures.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
