# KanpreyLM

A research testbed for **Kolmogorov-Arnold Network (KAN)** architectures in language models, using [GuppyLM](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic) — a small fish-personality chatbot — as the experimental subject.

Three model variants are implemented and compared against the original MLP baseline:

| Model | FFN | Attention | Val loss | Eval (16 prompts) | Inference |
|-------|-----|-----------|----------|-------------------|-----------|
| GuppyLM (baseline) | Linear→ReLU→Linear | dot-product | — | 8–9/16 | ~180 ms |
| KANpreyLM | KANLinear (B-spline) | dot-product | 0.2894 | 8/16 | ~880 ms |
| KATpreyLM | KANLinear (B-spline) | KAN kernel (KAT) | 0.2867 | 14–15/16 | ~1200 ms |
| **MLPEdgepreyLM** | **MLP-per-edge** | **dot-product** | **0.2854** | **14/16** | **~680 ms** |

All KAN variants trained for 8 000 steps with prompt-response masking (loss computed on assistant tokens only).

---

## What's in here

### Three model variants

**KANpreyLM** — replaces the 2-layer ReLU FFN with a single `KANLinear` layer (EfficientKAN, B-spline basis). Each edge `f_{i,j}(x)` is a learnable 1D B-spline function, giving ~147 K visualisable curves per model.

**KATpreyLM** — fully KAN transformer. Adds KAN feature maps to the Q and K projections in every attention head (KAT attention), turning the dot-product similarity kernel into a learned non-Euclidean kernel `K(q,k) = φ(q)·φ(k)`.

**MLPEdgepreyLM** — novel variant proposed here. Replaces each B-spline edge function with a tiny MLP (R→R):

```
h_i    = σ(x_i · W1[i] + b1[i])     # (hidden,) per input channel
f_{i,j} = W2[j,i] · h_i              # scalar per edge
```

Same KAN topology (additive decomposition over edges), no grid update mechanism, standard matmul ops throughout. Matches B-spline quality at 4× faster training and 1.8× faster inference.

### Interpretability notebook

`kanprey/interpret.py` is a [marimo](https://marimo.io) notebook implementing a rigorous interpretability study:

- Per-function statistics (NLS, activity, roughness, monotonicity, symmetry) across all ~147 K spline functions
- Functional PCA and K-means clustering of learned curve shapes
- Pruning / sparsity analysis with interactive thresholds
- Token-conditioned activation study (fish-domain vs generic prompts)
- MLP baseline comparison with Mann-Whitney U test
- KAT attention kernel visualisation (PSD check, learned vs linear contours)
- Interactive explorer: select any layer and (input, output) channel to see the exact learned 1D function

```bash
uv run --with . marimo edit kanprey/interpret.py
```

---

## Project structure

```
kanprey/
  config.py        ModelConfig, TrainConfig
  kan_layers.py    KANLinear (B-spline), MLPEdgeLinear (MLP-per-edge)
  model.py         KANpreyLM, KATpreyLM, MLPEdgepreyLM
  dataset.py       KanpreyDataset with prompt-response masking
  dataset_wikitext.py  Wikitext-103 dataloader (GPT-2 BPE, tiktoken)
  train.py         Training loop (cosine LR, grid update, checkpointing)
  inference.py     Chat interface
  interpret.py     Marimo interpretability notebook

scripts/
  train_scale.py   GPT-2 scale training (Wikitext-103, MLP vs MLPEdge)
  runpod_launch.py RunPod GPU pod launcher for scaling experiments

checkpoints/
  best.pt              KANpreyLM (8 000 steps, val_loss=0.2894)
  kat2/best.pt         KATpreyLM (8 000 steps, val_loss=0.2867)
  mlpedge/best.pt      MLPEdgepreyLM (8 000 steps, val_loss=0.2854)
  kan_8k_masked/best.pt  KANpreyLM fair baseline (8 000 steps, masked)

compare_with_original.py   Head-to-head 3-way comparison script
tokenizer.json             BPE tokenizer (vocab=2 393)
```

---

## Quickstart

**Requirements:** Python 3.13+, [uv](https://github.com/astral-sh/uv)

```bash
# Install
git clone https://github.com/HCAI-USP/kanprey-lm
cd kanprey-lm
uv sync

# Chat with the best model (MLPEdge)
uv run python -m kanprey.inference \
  --checkpoint checkpoints/mlpedge/best.pt \
  --prompt "are you hungry"

# 3-way comparison (requires ../guppylm-original)
uv run python compare_with_original.py \
  --kan-ckpt  checkpoints/kat2/best.pt \
  --kat-ckpt  checkpoints/mlpedge/best.pt \
  --orig-dir  ../guppylm-original

# Train a new variant
uv run python -m kanprey.train --model kan      --steps 8000
uv run python -m kanprey.train --model kat      --steps 8000 --checkpoint-dir checkpoints/kat2
uv run python -m kanprey.train --model mlpedge  --steps 8000 --checkpoint-dir checkpoints/mlpedge

# GPT-2 scale experiment (requires GPU)
uv run python scripts/train_scale.py --model mlpedge --steps 20000 \
  --batch-size 32 --grad-accum 4 --output-dir checkpoints/scale_mlpedge

# Interpretability notebook
uv run marimo edit kanprey/interpret.py
```

---

## Key findings

### Prompt-response masking matters

The first KATpreyLM run (4 000 steps, no masking) achieved val_loss 0.4202. Training loss was computed over all tokens including the user prompt, diluting the signal.

Adding `y[:prompt_len - 1] = -100` to mask prompt tokens dropped val_loss to **0.2867** and produced clean, in-character responses.

### MLPEdge: a novel KAN variant

Replacing B-spline edge functions with tiny learned MLPs (R→R) preserves the KAN topology while eliminating the grid update mechanism. The einsum structure is identical:

```python
# B-spline KAN
splines = module.b_splines(x)                          # custom Cox-de Boor recursion
out = einsum("bik,oik->bo", splines, spline_weight)

# MLP-edge KAN  (this work)
H = σ(einsum("bi,ih->bih", x, W1) + b1)               # standard matmul + activation
out = einsum("bih,oih->bo", H, W2)                     # (B, out)
```

Results at matched parameter count (hidden=5 ≈ n_basis=5):

| | KAT spline | MLPEdge |
|---|---|---|
| Val loss | 0.2867 | **0.2854** |
| Training time | 43 min | **10 min** |
| Inference latency | ~1 200 ms | **~680 ms** |
| Grid update step | required | not needed |

---

## Model architecture details

### KANLinear (B-spline)

```
forward(x):
    base_out  = SiLU(x) @ base_weight.T          # (B, out)  residual linear path
    splines   = b_splines(x)                      # (B, in, n_basis)  Cox-de Boor
    spline_out = einsum("bik,oik->bo", splines, spline_weight)
    return base_out + spline_out
```

Parameters: `out×in×n_basis` (spline) + `out×in` (base). Grid knots adapted once after warm-up via `update_grid_all()`.

### MLPEdgeLinear

```
forward(x):
    H   = σ(einsum("bi,ih->bih", x, W1) + b1)   # (B, in, hidden)
    out = einsum("bih,oih->bo", H, W2) + b_out    # (B, out)
    return out
```

Parameters: `in×hidden` (W1) + `in×hidden` (b1) + `out×in×hidden` (W2) + `out` (b_out). No grid.

### KATAttention

Q and K projections are passed through per-head `KANLinear(head_dim → head_dim)` feature maps before the dot-product, replacing the implicit linear kernel with a learned kernel `K(q,k) = φ(q)·φ(k)`.

---

## Dataset

[arman-bd/guppylm-60k-generic](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic) — 60 K instruction-response pairs in ChatML format, first-person fish-personality chatbot. BPE tokeniser trained on the same data (vocab=2 393).

---

## Citation

If you use this code or the MLPEdge finding, please cite:

```bibtex
@misc{kanpreylm-2025,
  title   = {KanpreyLM: KAN Variants in a Small Language Model Testbed},
  author  = {Alves, Felippe},
  year    = {2025},
  url     = {https://github.com/HCAI-USP/kanprey-lm}
}
```

---

## References

- Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- Yang & Wang, "KAT: Kolmogorov-Arnold Transformer", ICLR 2025.
- Lau, "EfficientKAN", 2024. [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)
- Agarwal et al., "Neural Additive Models", NeurIPS 2021. [arXiv:2004.13912](https://arxiv.org/abs/2004.13912)
- Tang et al., "PowerMLP: An Efficient Version of KAN", AAAI 2025. [arXiv:2412.13571](https://arxiv.org/abs/2412.13571)
