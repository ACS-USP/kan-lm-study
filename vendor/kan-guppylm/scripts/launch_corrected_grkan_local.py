#!/usr/bin/env python3
"""Launch corrected GuppyLM GR-KAN local seed queue.

Runs seeds sequentially on the local device to avoid MPS memory contention.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
STEPS = 8000
BATCH_SIZE = 32
RUN_PREFIX = "grkan_corrected_s"
LOG_DIR = ROOT / "logs" / "grkan_corrected_local"
RESULTS_DIR = ROOT / "results"
PID_PATH = RESULTS_DIR / "grkan_corrected_local_queue.pid"
STATE_PATH = RESULTS_DIR / "grkan_corrected_local_launch.json"
QUEUE_LOG = LOG_DIR / "queue.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_dir(seed: int) -> Path:
    return ROOT / "checkpoints" / f"{RUN_PREFIX}{seed}"


def train_log(seed: int) -> Path:
    return LOG_DIR / f"seed_{seed}.log"


def command(seed: int) -> list[str]:
    return [
        "uv", "run", "python", "-m", "kanprey.train",
        "--model", "grkan",
        "--steps", str(STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--checkpoint-dir", str(checkpoint_dir(seed).relative_to(ROOT)),
        "--seed", str(seed),
    ]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def write_state(state: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def assert_targets_clean() -> None:
    dirty = []
    for seed in SEEDS:
        cdir = checkpoint_dir(seed)
        if cdir.exists() and any(cdir.iterdir()):
            dirty.append(str(cdir.relative_to(ROOT)))
    if dirty:
        raise SystemExit("Refusing to overwrite non-empty checkpoint directories: " + ", ".join(dirty))


def run_queue() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    state = {
        "started_at": utc_now(),
        "status": "running",
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "runs": {},
        "commands": {str(seed): command(seed) for seed in SEEDS},
        "formula_verification": "results/grkan_formula_verification.json",
    }
    write_state(state)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for seed in SEEDS:
        cdir = checkpoint_dir(seed)
        cdir.mkdir(parents=True, exist_ok=True)
        log_path = train_log(seed)
        state = load_state()
        state["runs"][str(seed)] = {
            "status": "running",
            "started_at": utc_now(),
            "checkpoint_dir": str(cdir.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
            "command": command(seed),
        }
        write_state(state)
        with log_path.open("w") as log:
            log.write(f"[{utc_now()}] starting seed {seed}\n")
            log.write("command: " + " ".join(command(seed)) + "\n\n")
            log.flush()
            rc = subprocess.run(command(seed), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            log.write(f"\n[{utc_now()}] finished seed {seed} rc={rc}\n")
        state = load_state()
        state["runs"][str(seed)]["finished_at"] = utc_now()
        state["runs"][str(seed)]["returncode"] = rc
        state["runs"][str(seed)]["status"] = "complete" if rc == 0 else "failed"
        write_state(state)
        if rc != 0:
            state["status"] = "failed"
            state["failed_seed"] = seed
            state["finished_at"] = utc_now()
            write_state(state)
            raise SystemExit(rc)

    state = load_state()
    state["status"] = "complete"
    state["finished_at"] = utc_now()
    write_state(state)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def start() -> None:
    assert_targets_clean()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and is_running(pid):
            raise SystemExit(f"Queue already running with pid {pid}")
    with QUEUE_LOG.open("w") as queue_log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "run-queue"],
            cwd=ROOT,
            stdout=queue_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_PATH.write_text(str(proc.pid))
    state = {
        "started_at": utc_now(),
        "status": "launched",
        "pid": proc.pid,
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "queue_log": str(QUEUE_LOG.relative_to(ROOT)),
        "commands": {str(seed): command(seed) for seed in SEEDS},
        "formula_verification": "results/grkan_formula_verification.json",
    }
    write_state(state)
    print(json.dumps(state, indent=2))


def status() -> None:
    state = load_state()
    pid = None
    running = False
    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text().strip())
            running = is_running(pid)
        except ValueError:
            pid = None
    state["pid"] = pid
    state["pid_running"] = running
    for seed in SEEDS:
        cdir = checkpoint_dir(seed)
        state.setdefault("runs", {}).setdefault(str(seed), {})
        state["runs"][str(seed)].update({
            "checkpoint_dir_exists": cdir.exists(),
            "best_pt_exists": (cdir / "best.pt").exists(),
            "train_log_csv_exists": (cdir / "train_log.csv").exists(),
            "log_exists": train_log(seed).exists(),
        })
    print(json.dumps(state, indent=2))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "run-queue", "status"}:
        raise SystemExit("usage: launch_corrected_grkan_local.py {start|run-queue|status}")
    if sys.argv[1] == "start":
        start()
    elif sys.argv[1] == "run-queue":
        run_queue()
    else:
        status()


if __name__ == "__main__":
    main()
