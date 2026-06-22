# Vendored source: what was pruned

The `vendor/` copies are **curated source subsets** of the three repositories,
not full mirrors. The authoritative full source is the pinned commit of each repo
recorded in `../manifest.json`. We removed material that is **not needed to
reproduce the paper** and that concerns private infrastructure or internal notes,
so the published artifact is smaller and contains nothing operational.

## Removed (all repos, where present)

- **RunPod infrastructure scripts:** `runpod_launch.py`, `runpod_gpu_test.py`,
  `runpod_orchestrate.py`, `runpod_supervisor.py`, `train_watchdog.py`, and their
  tests (`test_runpod_*.py`). These launch/monitor GPU pods; they are operational
  tooling, not part of producing any figure or table.
- **Internal notes / planning:** `dev/` (dev logs, leaderboards, scratch
  notebooks), `nanokan/PLAN.md`, and `nanokan/docs/` (internal considerations).

## Kept (everything needed to reproduce)

- The model/kernel code: `nanochat/` package (incl. the corrected Safe-Padé
  rational GR-KAN kernel), `kanprey/` (GuppyLM/BabyLM models, audit, pruning).
- Training/eval/data scripts actually used for results (`base_train.py`,
  `tok_train.py`, `*_eval.py`, the audit/pruning/profiler/export scripts, etc.),
  task definitions, non-infra tests, configs, tokenizers, and `LICENSE`.

## Benign residue (intentionally left)

- A `runpod` entry remains in `pyproject.toml`/`uv.lock` (a public PyPI package);
  it is now unused but harmless.
- `nanokan/scripts/base_train.py` writes a crash log to a default
  `/runpod-volume/...` path — a non-sensitive default output location.

Verified clean with `gitleaks` (filesystem + history): no secrets, keys, tokens,
emails, or internal hostnames/IPs.
