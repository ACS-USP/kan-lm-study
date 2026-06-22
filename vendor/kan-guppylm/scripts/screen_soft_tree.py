"""Bounded GuppyLM screen: soft-tree basis vs param-matched baselines.

Trains each ~11.6M variant for a fixed short step budget with identical seed,
data order, optimizer and LR schedule, then reports validation cross-entropy
(assistant-token-only). This is a fast local signal, NOT the full 10k-step,
multi-seed result used for publication claims.

Usage:
    uv run python -m scripts.screen_soft_tree --steps 600 --eval-every 150
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from kanprey.config import ModelConfig, TrainConfig
from kanprey.dataset import get_dataloader, load_tokenizer
from kanprey.model import BasisKANpreyLM, MLPTransformer
from kanprey.train import autocast_ctx, detect_device, evaluate, get_lr


def build(model_type: str, cfg: ModelConfig) -> torch.nn.Module:
    if model_type == "mlp":
        return MLPTransformer(cfg)
    return BasisKANpreyLM(cfg)


def run_variant(label: str, model_type: str, family: str, steps: int,
                eval_every: int, device: torch.device, tok, base: TrainConfig,
                tree_depth: int | None = None, tree_steepness: float | None = None):
    # Identical seed -> identical init RNG and identical shuffled data order.
    torch.manual_seed(base.seed)
    tree_kw = {}
    if tree_depth is not None:
        tree_kw["basis_tree_depth"] = tree_depth
    if tree_steepness is not None:
        tree_kw["basis_tree_steepness"] = tree_steepness
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), basis_family=family, **tree_kw)
    train_loader = get_dataloader(
        "train", tok, batch_size=base.batch_size, max_seq_len=cfg.max_seq_len,
        dataset_name=base.dataset_name,
    )
    val_loader = get_dataloader(
        "test", tok, batch_size=base.batch_size, max_seq_len=cfg.max_seq_len,
        dataset_name=base.dataset_name, shuffle=False,
    )
    model = build(model_type, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=base.learning_rate,
        weight_decay=base.weight_decay, betas=base.betas,
    )
    sched = TrainConfig(
        max_steps=steps, warmup_steps=min(base.warmup_steps, max(1, steps // 5)),
        learning_rate=base.learning_rate, min_lr=base.min_lr,
    )
    best = float("inf")
    init_val = None
    t0 = time.time()
    step = 0
    model.train()
    while step < steps:
        for x, y in train_loader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)
            lr = get_lr(step, sched)
            for g in opt.param_groups:
                g["lr"] = lr
            with autocast_ctx(device):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.grad_clip)
            opt.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                vl = evaluate(model, val_loader, device, max_batches=30)
                if init_val is None:
                    init_val = vl
                best = min(best, vl)
                print(f"  [{label:9s}] step {step:4d}  val {vl:.4f}  best {best:.4f}  ({time.time()-t0:.0f}s)")
    return {"label": label, "params": n_params, "best": best, "secs": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated variant labels to run (default: all)")
    ap.add_argument("--tree-depth", type=int, default=None,
                    help="Override soft-tree depth for soft_tree variants")
    ap.add_argument("--tree-steepness", type=float, default=None,
                    help="Override soft-tree init steepness for soft_tree variants")
    args = ap.parse_args()

    device = detect_device()
    print(f"Device: {device}  |  steps={args.steps}  seed={args.seed}")
    tok = load_tokenizer("tokenizer.json")
    base = TrainConfig(seed=args.seed, batch_size=args.batch_size)

    variants = [
        ("mlp", "mlp", "chebyshev"),       # GELU-MLP control
        ("chebyshev", "basis", "chebyshev"),  # best-screened polynomial basis
        ("soft_tree", "basis", "soft_tree"),  # proposed
    ]
    only = set(args.only.split(",")) if args.only else None
    results = []
    for label, mtype, family in variants:
        if only is not None and label not in only:
            continue
        print(f"\n=== {label} ({mtype}/{family}) ===")
        td = args.tree_depth if family == "soft_tree" else None
        ts = args.tree_steepness if family == "soft_tree" else None
        results.append(run_variant(label, mtype, family, args.steps,
                                    args.eval_every, device, tok, base,
                                    tree_depth=td, tree_steepness=ts))

    print("\n" + "=" * 56)
    print(f"{'variant':12s} {'params':>12s} {'best_val':>10s} {'secs':>7s}")
    print("-" * 56)
    for r in sorted(results, key=lambda r: r["best"]):
        print(f"{r['label']:12s} {r['params']:>12,d} {r['best']:>10.4f} {r['secs']:>7.0f}")


if __name__ == "__main__":
    main()
