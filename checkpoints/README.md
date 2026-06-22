# Checkpoints (archived on the HuggingFace Hub)

Model checkpoints are **not stored in git** — they are large binaries (128 files,
~21.7 GB). They are archived on the **HuggingFace Hub** as a model repo with a model
card, and a **DOI** is minted from that repo (DataCite). The code repo references
them via `../manifest.json`.

```
checkpoints/
├── README.md            (this file)
├── INVENTORY.tsv        every best.pt with size + path
├── SHA256SUMS           integrity checksums (committed; verify after download)
├── gen_checksums.sh     regenerate SHA256SUMS from a local tree
└── hf/
    ├── README.md          the HF model card (uploaded as the repo README)
    └── upload_to_hf.py     uploads checkpoints + card to the Hub
```

## How to publish (the part needing your HF account)

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                                   # paste a write token

CKPT_ROOT=/Users/felippealves/Documents/GitHub/kan-guppylm/checkpoints \
HF_REPO=ACS-USP/kan-lm-study-checkpoints \
python checkpoints/hf/upload_to_hf.py           # creates a PRIVATE repo, resumable
```

Then, when you're ready to publish:

1. On the HF repo page → **Settings → Change visibility → Public**.
2. **Settings → "Generate DOI"** (DataCite). Copy the DOI.
3. Record the DOI + repo URL in `../manifest.json` (`checkpoints.host`) and in the
   paper's data-availability statement.

## Integrity

Checksums were generated from the local checkpoint tree at packaging time:

```bash
# regenerate (e.g. if the bundle changes):
CKPT_ROOT=/path/to/kan-guppylm/checkpoints bash gen_checksums.sh
# verify a downloaded copy:
cd /local/download && shasum -a 256 -c /path/to/SHA256SUMS
```

## Using a downloaded checkpoint

```bash
python ../experiments/prune_mlp.py \
    --checkpoint /local/checkpoints/mlp_s42/best.pt \
    --tokenizer ../vendor/kan-guppylm/tokenizer.json
```

The figure/table → checkpoint mapping is in `../manifest.json`. Note these are
`kanprey` checkpoints, not `transformers` models — load them with the vendored
`vendor/kan-guppylm/kanprey` code (see `hf/README.md`).
