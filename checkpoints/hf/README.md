---
license: mit
language:
- en
tags:
- kolmogorov-arnold-networks
- kan
- small-language-models
- interpretability
- pruning
- babylm
- negative-results
datasets:
- arman-bd/guppylm-60k-generic
pipeline_tag: text-generation
---

# KAN-LM Study — Model Checkpoints

Trained checkpoints for the paper **"Auditing and Benchmarking KAN Feed-Forward
Layers in Small Language Models."** These are the model weights behind every
figure and table; the code, experiment scripts, and provenance manifest live in
the companion repository.

- **Code + reproduction:** https://github.com/ACS-USP/kan-lm-study
- **Paper:** see `paper/` in the code repo
- **DOI:** _TBD — generated from this repository's settings (DataCite)._

> These are **not** `transformers`-loadable models. They are `kanprey` checkpoints
> (custom KAN/MLP transformer). Load them with the vendored `kan-guppylm` code in
> the companion repo (`vendor/kan-guppylm/kanprey`), not `AutoModel`.

## What's here

`best.pt` for every training run (128 files, ~21.7 GB), organized by regime:

| Regime | Path prefix | Contents |
|---|---|---|
| GuppyLM screen | `mlp_s*`, `swiglu_s*`, `kan_grid2_s*`, `grkan_corrected_s*`, `basis_confirm/*`, `kat_s*`, `mlpedge_*` | 3-seed architecture screen (d=384, 6 layers, vocab 2,393) |
| BabyLM Strict-Small | `babylm/<arch>_s42..s51` | the 61-run matrix (4 critical × 10 seeds + support/low rows), vocab 8,192 |
| Grid-size sweep | `gridsweep/*` | KAN grid 2/5/10/20 for the interpretability-vs-capacity sweep |
| Wikitext-103 scale | `scale/mlp`, `scale/mlpedge_h8` | GPT-2-small parameter-matched stress test |

- `INVENTORY.tsv` — every `best.pt` with size and path.
- `SHA256SUMS` — integrity checksums; verify with `shasum -a 256 -c SHA256SUMS`.

(The 286M ClimbMix GR-KAN stress-test checkpoints are large and tracked separately;
see the code repo's `manifest.json`.)

## Provenance and correctness

- The corrected rational activation uses the **Safe Padé** denominator
  `Q(x)=1+|b0 x + b1 x^2 + b2 x^3 + b3 x^4|`. **Pre-fix GR-KAN checkpoints are
  excluded** from all reported evidence and are not in this collection.
- Repo commits, the kernel correction, and a figure/table → script → checkpoint map
  are in `manifest.json` in the code repo.

## Loading a checkpoint

```python
import torch
from kanprey.config import ModelConfig          # from vendor/kan-guppylm
from kanprey.model import KANpreyLM, MLPTransformer

ckpt = torch.load("babylm/grkan_canonical_s42/best.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["model_cfg"]
model = (MLPTransformer if ckpt["model_type"] == "mlp" else KANpreyLM)(cfg)
model.load_state_dict(ckpt["model"]); model.eval()
```

## Licenses

Weights: MIT. Training data retains its own licenses — GuppyLM (MIT), BabyLM
challenge corpus, Wikitext-103 (CC BY-SA 3.0/GFDL), ClimbMix → NVIDIA
Nemotron-ClimbMix (CC BY-NC 4.0, research use).

## Citation

```bibtex
@misc{alves2026kanlm,
  title  = {Auditing and Benchmarking KAN Feed-Forward Layers in Small Language Models},
  author = {Alves, Felippe},
  year   = {2026},
  note   = {Code: https://github.com/ACS-USP/kan-lm-study}
}
```
