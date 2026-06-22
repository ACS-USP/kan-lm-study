#!/usr/bin/env python
"""
prune_mlp.py — MLP pruning baseline (Review #7, M4).

Reviewer #7 (M4) asks whether the KAN *activity*-guided pruning curve is better
than ordinary MLP pruning at matched sparsity, since the published curve only
compares activity-ranked vs. random pruning *within* the KAN. This script builds
the missing baseline: structured hidden-neuron pruning of the GuppyLM MLP FFN,
ranked by (a) weight-magnitude saliency and (b) data-driven activation magnitude,
with a random control, evaluated on the SAME GuppyLM test split and evaluator as
the KAN curve (docs/experiments/results/prune_curve/prune_curve.csv).

Prunable unit: an MLP FFN hidden neuron h (the 4x-expansion dimension). Pruning a
fraction p of the 6*d_ffn hidden neurons removes a fraction p of FFN parameters
and FLOPs, the same matched-sparsity axis as pruning a fraction p of KAN edges.

Saliency rankings (global, across all 6 layers' hidden neurons):
  * magnitude : s_h = ||W_in[h,:], b_in[h]||_2 * ||W_out[:,h]||_2
  * activation: s_h = mean_t |GELU(W_in x_t + b_in)[h]| over the test split

Zeroing a neuron: W_in[h,:]=0, b_in[h]=0, W_out[:,h]=0 (GELU(0)=0 -> no contribution).

USAGE (copy into kan-guppylm/scripts/ and run from the kan-guppylm repo root):

  uv run python scripts/prune_mlp.py \
      --checkpoint checkpoints/mlp_s42/best.pt --tokenizer tokenizer.json \
      --fractions 0 0.1 0.2 0.3 0.4 0.5 \
      --out results/prune_mlp_curve
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kanprey.config import ModelConfig                       # noqa: E402
from kanprey.dataset import get_dataloader, load_tokenizer   # noqa: E402
from kanprey.model import MLPTransformer                     # noqa: E402
from kanprey.train import evaluate                           # noqa: E402


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mlp_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    assert ckpt.get("model_type", "mlp") == "mlp", f"not an MLP checkpoint: {path}"
    model = MLPTransformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def ffn_modules(model):
    """List of (name, MLPFFN) modules, one per transformer block."""
    return [(f"blocks.{i}.ffn", b.ffn) for i, b in enumerate(model.blocks)]


@torch.no_grad()
def magnitude_saliency(model) -> list[np.ndarray]:
    """Per-neuron weight-magnitude saliency for each FFN: ||W_in row|| * ||W_out col||."""
    sals = []
    for _, ffn in ffn_modules(model):
        w_in = ffn.fc1.weight.detach()            # (hidden, d_model)
        b_in = ffn.fc1.bias.detach()              # (hidden,)
        w_out = ffn.fc2.weight.detach()           # (d_model, hidden)
        in_norm = torch.sqrt((w_in ** 2).sum(dim=1) + b_in ** 2)   # (hidden,)
        out_norm = torch.linalg.norm(w_out, dim=0)                 # (hidden,)
        sals.append((in_norm * out_norm).cpu().numpy())
    return sals


@torch.no_grad()
def activation_saliency(model, loader, device, max_batches: int) -> list[np.ndarray]:
    """Per-neuron mean |GELU(fc1 x)| over the split (data-driven activation magnitude)."""
    mods = ffn_modules(model)
    acc = [torch.zeros(ffn.fc1.weight.shape[0], device=device) for _, ffn in mods]
    counts = 0
    handles = []
    store: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_m, _inp, out):
            store[idx] = F.gelu(out)              # (B, T, hidden) pre-activation -> activation
        return hook

    for idx, (_, ffn) in enumerate(mods):
        handles.append(ffn.fc1.register_forward_hook(make_hook(idx)))

    model.eval().to(device)
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        model(x)
        for idx in range(len(mods)):
            a = store[idx].abs().reshape(-1, store[idx].shape[-1])   # (B*T, hidden)
            acc[idx] += a.sum(dim=0)
            if idx == 0:
                counts += a.shape[0]
    for h in handles:
        h.remove()
    return [(acc[idx] / max(counts, 1)).cpu().numpy() for idx in range(len(mods))]


def apply_neuron_prune(model, pruned_bool: np.ndarray, sizes, bounds):
    """Zero pruned hidden neurons in-place (W_in row, b_in, W_out col)."""
    mods = ffn_modules(model)
    with torch.no_grad():
        for mi, (_, ffn) in enumerate(mods):
            pm = pruned_bool[bounds[mi]:bounds[mi + 1]]            # (hidden,)
            keep_in = torch.tensor((~pm).astype(np.float32))       # (hidden,)
            ffn.fc1.weight.mul_(keep_in.unsqueeze(1))
            ffn.fc1.bias.mul_(keep_in)
            ffn.fc2.weight.mul_(keep_in.unsqueeze(0))


def run_ranking(checkpoint, order, fractions, sizes, bounds, N, val_loader, device, max_eval):
    """Prune the lowest-saliency neurons at each fraction; return val losses."""
    losses = []
    for p in fractions:
        n_prune = int(round(p * N))
        pruned = np.zeros(N, bool)
        pruned[order[:n_prune]] = True
        model, _ = load_mlp_model(checkpoint)
        apply_neuron_prune(model, pruned, sizes, bounds)
        model.to(device)
        losses.append(evaluate(model, val_loader, device, max_batches=max_eval))
    return losses


def main():
    ap = argparse.ArgumentParser(description="MLP hidden-neuron pruning baseline (Review #7, M4).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--dataset-name", default="arman-bd/guppylm-60k-generic")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-eval-batches", type=int, default=10_000, help="eval batches (default full test split)")
    ap.add_argument("--act-batches", type=int, default=10_000, help="batches for activation saliency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/prune_mlp_curve")
    args = ap.parse_args()

    device = detect_device()
    print(f"Device: {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base, cfg = load_mlp_model(args.checkpoint)
    mods = ffn_modules(base)
    sizes = [ffn.fc1.weight.shape[0] for _, ffn in mods]
    bounds = np.cumsum([0] + sizes)
    N = int(bounds[-1])
    print(f"FFN layers: {len(mods)}, hidden/layer: {sizes[0]}, total prunable neurons: {N:,}")

    tokenizer = load_tokenizer(args.tokenizer)
    val_loader = get_dataloader("test", tokenizer, batch_size=args.batch_size,
                                max_seq_len=cfg.max_seq_len, dataset_name=args.dataset_name,
                                shuffle=False)

    # --- saliency rankings (ascending: lowest-saliency pruned first) ----------
    mag = np.concatenate(magnitude_saliency(base))
    mag_order = np.argsort(mag)
    act_loader = get_dataloader("test", tokenizer, batch_size=args.batch_size,
                                max_seq_len=cfg.max_seq_len, dataset_name=args.dataset_name,
                                shuffle=False)
    act = np.concatenate(activation_saliency(base, act_loader, device, args.act_batches))
    act_order = np.argsort(act)
    rng = np.random.default_rng(args.seed)
    rand_order = rng.permutation(N)

    vl_mag = run_ranking(args.checkpoint, mag_order, args.fractions, sizes, bounds, N,
                         val_loader, device, args.max_eval_batches)
    vl_act = run_ranking(args.checkpoint, act_order, args.fractions, sizes, bounds, N,
                         val_loader, device, args.max_eval_batches)
    vl_rand = run_ranking(args.checkpoint, rand_order, args.fractions, sizes, bounds, N,
                          val_loader, device, args.max_eval_batches)

    rows = []
    for i, p in enumerate(args.fractions):
        row = {"fraction": p, "n_pruned": int(round(p * N)),
               "val_loss_magnitude": round(vl_mag[i], 5),
               "val_loss_activation": round(vl_act[i], 5),
               "val_loss_random": round(vl_rand[i], 5)}
        rows.append(row)
        print(f"p={p:.2f}  mag={vl_mag[i]:.4f}  act={vl_act[i]:.4f}  rand={vl_rand[i]:.4f}")

    csv_path = out_dir / "prune_mlp_curve.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
