# KAN-LM Study — Code and Reproduction Artifact

Self-contained code + provenance artifact for **"Auditing and Benchmarking KAN
Feed-Forward Layers in Small Language Models."**

The paper separates two questions: (1) can KAN feed-forward edge functions be
audited in a small LM (yes — and the audit is actionable and corpus-transferable),
and (2) do KAN-family FFNs replace strong MLP baselines (no consistent advantage
on standardized BabyLM benchmarks). This repository contains everything needed to
inspect and rerun the experiments behind the figures and tables.

## What's here

| Path | Contents |
|---|---|
| `vendor/` | Frozen **source-only** snapshots of the three code repos (no `.git`, no checkpoints, no data). See `manifest.json` for commits/roles. |
| `experiments/` | Scripts that produced the paper's figures/tables, plus their small result outputs in `experiments/results/`. |
| `figures/` | Figure PDFs/PNGs as they appear in the paper. |
| `paper/` | LaTeX source and compiled PDF. |
| `environment/` | Per-repo `pyproject.toml` + `uv.lock`. |
| `checkpoints/` | **Not** the checkpoints — an inventory, a checksum script, and a pointer to the external archive (Zenodo/HF). |
| `manifest.json` | Machine-readable provenance: repos, commits, the Safe-Padé kernel correction, and a figure/table → script → data map. |

## Checkpoints live outside git

Model checkpoints (128 `best.pt` files, ~21.7 GB) are **not** in this repo — git is
the wrong tool for large binaries. They are archived separately (Zenodo DOI or
HuggingFace Hub; see `checkpoints/README.md`) and referenced by `manifest.json`.
The reproduction contract is: **clone this repo → download the cited checkpoints
from the archive → rerun.**

## Quick start

```bash
# 1. Environment (per the main repo; uv recommended)
cd vendor/kan-guppylm
uv sync            # or: pip install -e . using environment/kan-guppylm.*

# 2. Offline smoke test — regenerate a figure from committed result CSVs,
#    no checkpoints needed (proves the plotting/repro path works):
cd ../../ && make smoke
```

## Reproducing figures/tables

`make help` lists targets. Two classes:

- **Offline (CSV → figure):** `make fig-prune` regenerates `figures/prune_compare.pdf`
  from `experiments/results/`. No checkpoints needed.
- **From checkpoints:** download the relevant `best.pt` from the archive (see
  `checkpoints/README.md`), then run the script named in `manifest.json` for that
  figure/table, e.g.:
  ```bash
  # MLP pruning baseline (Table 10 / Fig. 3 MLP curves)
  python experiments/prune_mlp.py --checkpoint <path>/mlp_s42/best.pt \
      --tokenizer vendor/kan-guppylm/tokenizer.json
  # BabyLM grid-2 edge audit (Table 9)
  python experiments/audit_grid_sweep.py \
      --checkpoints s42=<path>/babylm/kan_grid2_s42/best.pt ... --random-control
  ```

See `manifest.json` `figures`/`tables` for the exact script + inputs behind each.

## Provenance and corrections

- The corrected rational activation uses the **Safe Padé** denominator
  `Q(x)=1+|b0 x + b1 x^2 + b2 x^3 + b3 x^4|`; pre-fix GR-KAN checkpoints are
  excluded from all reported evidence. See `manifest.json` `kernel_corrections`.
- `vendor/nanokan` is an internal repo vendored here as source. **Confirm it can be
  released publicly before publishing this artifact.**

## License

Code in this artifact and each vendored repo is MIT (see per-repo `LICENSE`).
Datasets retain their own licenses (GuppyLM MIT; BabyLM challenge corpus; Wikitext-103
CC BY-SA 3.0/GFDL; ClimbMix → NVIDIA Nemotron-ClimbMix CC BY-NC 4.0).
