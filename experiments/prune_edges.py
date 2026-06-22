#!/usr/bin/env python
"""
prune_edges.py — Experiment M2 (Review #5).

Activity-guided edge pruning of a trained GuppyLM KAN, with a random-pruning
control. Demonstrates that the audit's per-edge ACTIVITY scores are *actionable*:
zeroing low-activity edges degrades validation loss far slower than zeroing
random edges. Optional masked finetune shows recovery.

Edge importance == reconstructed-curve activity (||f||/sqrt(P)), i.e. the SAME
metric the paper uses for its "inactive <= 0.01" count, so the prune is defined
in the audit's own units.

USAGE (copy into kan-guppylm/scripts/ and run from the kan-guppylm repo root):

  python scripts/prune_edges.py \
      --checkpoint checkpoints/kan_grid2_s42/best.pt --tokenizer tokenizer.json \
      --fractions 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
      --random-control --out results/prune_curve

  # with short masked finetune for recovery:
  python scripts/prune_edges.py --checkpoint ... --tokenizer tokenizer.json \
      --fractions 0.5 0.7 0.9 --finetune-steps 800 --out results/prune_finetune
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
from kanprey.kan_layers import KANLinear                     # noqa: E402
from kanprey.model import KANpreyLM, KATpreyLM               # noqa: E402
from kanprey.train import evaluate                           # noqa: E402


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_kan_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    cls = KATpreyLM if ckpt.get("model_type", "kan") == "kat" else KANpreyLM
    model = cls(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def ffn_kanlinear_modules(model):
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, KANLinear) and "att" not in n.lower()]
    if not mods:
        mods = [(n, m) for n, m in model.named_modules() if isinstance(m, KANLinear)]
    return mods


@torch.no_grad()
def edge_activity(module: KANLinear, n_points: int = 200) -> np.ndarray:
    """Per-edge activity ||f_{j,i}||/sqrt(P), shape (out, in). Matches the paper's metric."""
    module = module.to("cpu")
    in_f, order = module.in_features, module.spline_order
    x_grids = torch.zeros(in_f, n_points)
    for i in range(in_f):
        g = module.grid[i]
        x_min, x_max = g[order].item(), g[-(order + 1)].item()
        margin = 0.05 * max(x_max - x_min, 1e-6)
        x_grids[i] = torch.linspace(x_min - margin, x_max + margin, n_points)
    bases = module.b_splines(x_grids.T)
    spline_curves = torch.einsum("pin,oin->oip", bases, module.spline_weight)
    base_curves = module.base_weight.unsqueeze(-1) * F.silu(x_grids).unsqueeze(0)
    curves = (spline_curves + base_curves).numpy()           # (out, in, P)
    out_f = curves.shape[0]
    return (np.linalg.norm(curves.reshape(out_f * in_f, n_points), axis=1)
            / np.sqrt(n_points)).reshape(out_f, in_f)


def apply_prune(model, pruned_bool: np.ndarray, shapes, bounds):
    """Zero pruned edges in-place; return per-module keep masks (cpu float tensors)."""
    keep_masks = []
    mods = ffn_kanlinear_modules(model)
    with torch.no_grad():
        for mi, (_, m) in enumerate(mods):
            out_f, in_f = shapes[mi]
            pm = pruned_bool[bounds[mi]:bounds[mi + 1]].reshape(out_f, in_f)
            keep = torch.tensor((~pm).astype(np.float32))
            m.base_weight.mul_(keep)
            m.spline_weight.mul_(keep.unsqueeze(-1))
            keep_masks.append(keep)
    return keep_masks


def masked_finetune(model, keep_masks, train_loader, steps, device, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mods = ffn_kanlinear_modules(model)
    dev_masks = [k.to(device) for k in keep_masks]
    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                                # re-zero pruned edges every step
            for (_, m), keep in zip(mods, dev_masks):
                m.base_weight.mul_(keep)
                m.spline_weight.mul_(keep.unsqueeze(-1))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Activity-guided edge pruning (Review #5, M2).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--dataset-name", default="arman-bd/guppylm-60k-generic")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--finetune-steps", type=int, default=0)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batches", type=int, default=10_000, help="eval batches (default = full test split)")
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/prune_curve")
    args = ap.parse_args()

    device = detect_device()
    print(f"Device: {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- compute per-edge activity once on the base (cpu) model --------------
    base_model, cfg = load_kan_model(args.checkpoint)
    mods = ffn_kanlinear_modules(base_model)
    print(f"FFN KANLinear layers: {len(mods)}")
    activities, shapes = [], []
    for name, m in mods:
        a = edge_activity(m, args.n_points)
        activities.append(a)
        shapes.append(a.shape)
        print(f"  {name:30s} {a.shape[0]}x{a.shape[1]}  median_act={np.median(a):.4f}")
    act_concat = np.concatenate([a.ravel() for a in activities])
    sizes = [s[0] * s[1] for s in shapes]
    bounds = np.cumsum([0] + sizes)
    N = int(bounds[-1])
    print(f"Total FFN edges: {N:,}")
    sort_order = np.argsort(act_concat)                       # ascending: lowest activity first
    rng = np.random.default_rng(args.seed)
    rand_order = rng.permutation(N)

    # --- dataloaders ---------------------------------------------------------
    tokenizer = load_tokenizer(args.tokenizer)
    val_loader = get_dataloader("test", tokenizer, batch_size=args.batch_size,
                                max_seq_len=cfg.max_seq_len, dataset_name=args.dataset_name,
                                shuffle=False)
    train_loader = None
    if args.finetune_steps > 0:
        train_loader = get_dataloader("train", tokenizer, batch_size=args.batch_size,
                                      max_seq_len=cfg.max_seq_len, dataset_name=args.dataset_name)

    rows = []
    for p in args.fractions:
        n_prune = int(round(p * N))

        # activity-ranked prune
        pruned = np.zeros(N, bool)
        pruned[sort_order[:n_prune]] = True
        m_act, _ = load_kan_model(args.checkpoint)
        keep = apply_prune(m_act, pruned, shapes, bounds)
        m_act.to(device)
        vl_act = evaluate(m_act, val_loader, device, max_batches=args.max_batches)

        vl_ft = None
        if args.finetune_steps > 0 and n_prune > 0:
            m_act = masked_finetune(m_act, keep, train_loader, args.finetune_steps, device, args.finetune_lr)
            vl_ft = evaluate(m_act, val_loader, device, max_batches=args.max_batches)

        # random-pruning control
        vl_rand = None
        if args.random_control and n_prune > 0:
            pruned_r = np.zeros(N, bool)
            pruned_r[rand_order[:n_prune]] = True
            m_rand, _ = load_kan_model(args.checkpoint)
            apply_prune(m_rand, pruned_r, shapes, bounds)
            m_rand.to(device)
            vl_rand = evaluate(m_rand, val_loader, device, max_batches=args.max_batches)

        row = {"fraction": p, "n_pruned": n_prune,
               "val_loss_activity": round(vl_act, 5),
               "val_loss_activity_finetuned": (round(vl_ft, 5) if vl_ft is not None else ""),
               "val_loss_random": (round(vl_rand, 5) if vl_rand is not None else "")}
        rows.append(row)
        print(f"p={p:.2f}  n_pruned={n_prune:>7,}  "
              f"act={vl_act:.4f}"
              + (f"  rand={vl_rand:.4f}" if vl_rand is not None else "")
              + (f"  ft={vl_ft:.4f}" if vl_ft is not None else ""))

    csv_path = out_dir / "prune_curve.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # --- plot (optional) -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr = [r["fraction"] for r in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(fr, [r["val_loss_activity"] for r in rows], "o-", label="activity-ranked prune")
        if args.random_control:
            yr = [r["val_loss_random"] for r in rows]
            plt.plot(fr, [v if v != "" else np.nan for v in yr], "s--", label="random prune")
        if args.finetune_steps > 0:
            yf = [r["val_loss_activity_finetuned"] for r in rows]
            plt.plot(fr, [v if v != "" else np.nan for v in yf], "^:", label=f"+{args.finetune_steps}-step finetune")
        plt.axhline(rows[0]["val_loss_activity"], color="gray", lw=0.8, ls=":", label="unpruned baseline")
        plt.xlabel("fraction of FFN edges pruned")
        plt.ylabel("validation loss (nats)")
        plt.title("Activity-guided edge pruning vs. random")
        plt.legend()
        plt.tight_layout()
        png = out_dir / "prune_curve.png"
        plt.savefig(png, dpi=150)
        print(f"Wrote {png}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
