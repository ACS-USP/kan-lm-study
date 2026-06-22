#!/usr/bin/env python
"""Upload the KAN-LM study checkpoints + model card to the HuggingFace Hub.

Hosts all `best.pt` files (~129, ~20 GB) plus the model card, inventory, and
checksums in one HF model repo. The repo is created PRIVATE; flip it to public
and mint a DOI from the repo settings when ready (see checkpoints/README.md).

Prereqs
-------
  pip install -U "huggingface_hub[cli]"
  hf auth login                      # paste a write token

Usage
-----
  CKPT_ROOT=/path/to/kan-guppylm/checkpoints \
  HF_REPO=ACS-USP/kan-lm-study-checkpoints \
  python checkpoints/hf/upload_to_hf.py

Notes
-----
  * Uploads only **/best.pt (skips intermediate step_*.pt), mirroring the local tree.
  * upload_large_folder is resumable: re-run if the connection drops.
  * Verify after upload:  shasum -a 256 -c checkpoints/SHA256SUMS  (against a local copy)
"""
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    sys.exit('huggingface_hub not installed. Run: pip install -U "huggingface_hub[cli]"')

HERE = Path(__file__).resolve().parent              # checkpoints/hf
CKPT_DIR = HERE.parent                              # checkpoints/

CKPT_ROOT = os.environ.get("CKPT_ROOT")
REPO = os.environ.get("HF_REPO", "ACS-USP/kan-lm-study-checkpoints")
PRIVATE = os.environ.get("HF_PRIVATE", "1") != "0"

if not CKPT_ROOT or not Path(CKPT_ROOT).is_dir():
    sys.exit("Set CKPT_ROOT to your local checkpoints directory (the one with best.pt files).")

api = HfApi()
print(f"Creating/using model repo {REPO} (private={PRIVATE}) ...")
create_repo(REPO, repo_type="model", private=PRIVATE, exist_ok=True)

# 1) model card + metadata files (small, do first so the repo is browsable immediately)
api.upload_file(path_or_fileobj=str(HERE / "README.md"),
                path_in_repo="README.md", repo_id=REPO, repo_type="model")
for meta in ("INVENTORY.tsv", "SHA256SUMS"):
    p = CKPT_DIR / meta
    if p.exists():
        api.upload_file(path_or_fileobj=str(p), path_in_repo=meta,
                        repo_id=REPO, repo_type="model")
        print(f"  uploaded {meta}")

# 2) checkpoints (best.pt only), resumable large-folder upload, mirroring the tree
print(f"Uploading best.pt files from {CKPT_ROOT} ... (resumable)")
api.upload_large_folder(repo_id=REPO, repo_type="model",
                        folder_path=CKPT_ROOT, allow_patterns=["**/best.pt"])

print(f"\nDone. Review at https://huggingface.co/{REPO}")
print("Next: make the repo public, then Settings -> 'Generate DOI' to mint a DataCite DOI.")
