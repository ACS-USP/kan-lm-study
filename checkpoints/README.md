# Checkpoints (archived externally)

Model checkpoints are **not stored in git** — they are large binaries (git is the
wrong tool). They live in an external archive and are referenced by `../manifest.json`.

## Where

**Host: TBD.** Choose one at upload time and record the URL/DOI in `../manifest.json`
(`checkpoints.host`) and in the paper's data-availability statement:

- **Zenodo** — single citable DOI for the whole bundle (recommended for the paper's
  permanent archive).
- **HuggingFace Hub** — model-card UX + `hf_hub_download`; can mint a Zenodo DOI alongside.

## What

The bundle is **every run's final checkpoint** (`best.pt`): ~124 files, ~18 GB.
It includes the **61-run BabyLM Strict-Small matrix** (the four critical
architectures × 10 seeds plus supporting/low-priority rows × fewer seeds) and the
GuppyLM, Wikitext-103 scale, and ClimbMix endpoints.

- `INVENTORY.tsv` — every `best.pt` with its size and repo-relative path.
- `SHA256SUMS` — integrity checksums (generated at upload time; see below).

## Generating checksums

Checksums are computed against your local checkpoint tree at upload time (they are
not committed pre-upload because the bundle composition/host is finalized then):

```bash
CKPT_ROOT=/path/to/kan-guppylm/checkpoints bash gen_checksums.sh
# writes ./SHA256SUMS  (then upload it alongside the checkpoints)
```

## Using a downloaded checkpoint

After downloading, point the experiment scripts at the local path, e.g.:

```bash
python ../experiments/prune_mlp.py \
    --checkpoint /local/checkpoints/mlp_s42/best.pt \
    --tokenizer ../vendor/kan-guppylm/tokenizer.json
```

The figure/table → checkpoint mapping is in `../manifest.json`.
