"""
GPT-2 scale training script for the MLPEdge vs MLP FFN comparison.

Trains on Wikitext-103 using GPT-2 small architecture.
Designed to run both locally (for testing) and on RunPod (for full training).

Usage:
    # Local quick test (~10 steps)
    uv run python scripts/train_scale.py --model mlp --steps 100 --output-dir /tmp/scale_test

    # Full experiment on GPU
    uv run python scripts/train_scale.py --model mlpedge --steps 50000 \
        --batch-size 32 --grad-accum 4 --output-dir checkpoints/scale_mlpedge

    # Standard MLP baseline
    uv run python scripts/train_scale.py --model mlp --steps 50000 \
        --batch-size 32 --grad-accum 4 --output-dir checkpoints/scale_mlp
"""

import argparse
import csv
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kanprey.config import ModelConfig, TrainConfig
from kanprey.model import (
    MLPTransformer, MLPEdgepreyLM, GRKANpreyLM,
    MoEGRKANpreyLM, ModuleChainLM, LoopGRKANpreyLM, BasisKANpreyLM,
)
from kanprey.dataset_wikitext import get_wikitext_loaders
from kanprey.optim import configure_optimizers


# ── GPT-2 small config ────────────────────────────────────────────────────────

GPT2_SMALL = dict(
    d_model=768,
    n_heads=12,
    n_layers=12,
    max_seq_len=1024,
    dropout=0.1,
    mlp_edge_hidden=5,
)

# ── Local 9M config (for quick experiments and MultKAN ablation) ──────────────
# d_model=384, n_heads=6, n_layers=6 → ~9M params depending on FFN variant.
# Uses seq_len=128 to match the existing small-model training setup.
LOCAL_9M = dict(
    d_model=384,
    n_heads=6,
    n_layers=6,
    max_seq_len=128,
    dropout=0.1,
    mlp_edge_hidden=5,
    # GR-KAN defaults (grkan_m=5, grkan_n=4, grkan_groups=8, grkan_expand=4)
    # are already set in ModelConfig; no overrides needed here
)

# ── Modular unit config ───────────────────────────────────────────────────────
# Each unit has unit_n_layers=3 blocks (half LOCAL_9M depth) → ~5M per unit.
# Two units chained = 6 blocks total, matching LOCAL_9M depth at same d_model.
# MoE config: 8 GR-KAN experts, top_k=2, so 25% of expert params activate/token.
MODULE_UNIT = dict(
    d_model=384,
    n_heads=6,
    n_layers=6,       # total depth (unit_n_layers splits this into units)
    max_seq_len=128,
    dropout=0.1,
    unit_n_layers=3,  # layers per composable unit
    # MoE settings
    n_moe_experts=8,
    moe_top_k=2,
    load_balance_coeff=0.01,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    # MPS float16 autocast adds ~10% overhead on Apple Silicon due to fp16/fp32
    # conversion at each layer boundary without CUDA-style tensor cores to recoup it.
    return torch.autocast("cpu", enabled=False)


def get_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def perplexity(val_loss: float) -> float:
    return math.exp(val_loss)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = detect_device()
    print(f"Device: {device}")

    # Dataset — seq_len depends on model
    _local_models = ("grkan", "mlp_local", "moe_grkan", "unit_grkan", "unit0_grkan")
    seq_len = LOCAL_9M["max_seq_len"] if args.model in _local_models else GPT2_SMALL["max_seq_len"]
    print("Loading Wikitext-103…")
    train_loader, val_loader, vocab_size = get_wikitext_loaders(
        batch_size=args.batch_size,
        max_seq_len=seq_len,
        num_workers=args.num_workers,
    )

    # Model
    if args.model == "grkan":
        cfg = ModelConfig(vocab_size=vocab_size, **LOCAL_9M)
        model = GRKANpreyLM(cfg).to(device)
        label = "GRKAN-9M"
    elif args.model == "mlp_local":
        cfg = ModelConfig(vocab_size=vocab_size, **LOCAL_9M)
        model = MLPTransformer(cfg).to(device)
        label = "MLP-9M"
    elif args.model == "moe_grkan":
        # Sparse MoE: 8 GR-KAN experts, top_k=2, local scale
        cfg = ModelConfig(vocab_size=vocab_size, **MODULE_UNIT)
        model = MoEGRKANpreyLM(cfg).to(device)
        label = f"MoE-GRKAN-{cfg.n_moe_experts}exp-top{cfg.moe_top_k}"
    elif args.model == "unit_grkan":
        # Full modular chain: 2 units × 3 layers each = 6 layers total
        cfg = ModelConfig(vocab_size=vocab_size, **MODULE_UNIT)
        n_units = cfg.n_layers // cfg.unit_n_layers
        model = ModuleChainLM(cfg, n_units=n_units).to(device)
        label = f"ModuleChain-GRKAN-{n_units}x{cfg.unit_n_layers}L"
        if args.freeze_unit is not None:
            # Freeze earlier units when training the next stage
            for u in range(args.freeze_unit):
                model.freeze_unit(u)
                print(f"  Unit {u} frozen.")
        if args.load_unit is not None:
            unit_idx, ckpt_path = args.load_unit
            model.load_unit_from_checkpoint(int(unit_idx), ckpt_path, device=str(device))
            print(f"  Loaded unit {unit_idx} from {ckpt_path}")
    elif args.model == "unit0_grkan":
        # Train a single 3-layer unit as a standalone GR-KAN model (Stage 0)
        cfg = ModelConfig(vocab_size=vocab_size,
                          **{**MODULE_UNIT, "n_layers": MODULE_UNIT["unit_n_layers"]})
        model = GRKANpreyLM(cfg).to(device)
        label = "GRKAN-Unit0"
    elif args.model == "loop_grkan":
        # Ouro-style LoopLM: one 3-layer unit applied T_max times (weight-tied)
        # with learned exit gate + entropy-regularized training objective.
        # Optional EqR interventions: --loop-init-std (RI) and --loop-noise-std (NI).
        cfg = ModelConfig(
            vocab_size=vocab_size,
            **{**MODULE_UNIT, "n_layers": MODULE_UNIT["unit_n_layers"]},
            loop_t_max=args.loop_t_max,
            loop_beta=args.loop_beta,
            loop_init_std=args.loop_init_std,
            loop_noise_std=args.loop_noise_std,
        )
        model = LoopGRKANpreyLM(cfg).to(device)
        label = f"LoopGRKAN-T{cfg.loop_t_max}"
        if args.freeze_body:
            model.body.requires_grad_(False)
            print("  Body frozen (Stage II gate fine-tuning).")
    elif args.model == "basis":
        # Grouped function-basis KAN FFN (e.g. soft_tree) at GPT-2-small scale —
        # the exact d_model=768/12L/Wikitext-103 setup where MLPEdge lost 26% ppl,
        # for an apples-to-apples scaling test. basis_expand=4 (default) keeps the
        # FFN at 768->3072->768, parameter-matched to MLP-GPT2; the basis coeffs
        # add only basis_groups * (2**depth) params per activation (negligible).
        cfg = ModelConfig(
            vocab_size=vocab_size,
            **GPT2_SMALL,
            basis_family=args.basis_family,
            basis_groups=args.basis_groups,
            basis_input_norm=args.basis_input_norm,
            basis_degree=args.basis_degree,
            basis_tree_depth=args.basis_tree_depth,
            basis_tree_steepness=args.basis_tree_steepness,
        )
        model = BasisKANpreyLM(cfg).to(device)
        label = f"Basis-{args.basis_family}-GPT2"
    else:
        mlp_edge_hidden = 8 if args.model == "mlpedge_matched" else GPT2_SMALL["mlp_edge_hidden"]
        cfg = ModelConfig(
            vocab_size=vocab_size,
            **{**GPT2_SMALL, "mlp_edge_hidden": mlp_edge_hidden},
        )
        if args.model in ("mlpedge", "mlpedge_matched"):
            model = MLPEdgepreyLM(cfg).to(device)
            label = "MLPEdge-GPT2-matched" if args.model == "mlpedge_matched" else "MLPEdge-GPT2"
        else:
            model = MLPTransformer(cfg).to(device)
            label = "MLP-GPT2"

    if args.load_checkpoint:
        ckpt = torch.load(args.load_checkpoint, map_location=str(device), weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  load_checkpoint: {len(missing)} missing keys (expected for new heads)")
        print(f"  Loaded full model weights from {args.load_checkpoint}")

    if args.grad_checkpoint:
        model.set_gradient_checkpointing(True)
        print("Gradient checkpointing: ON (recomputes activations during backward)")

    if args.compile:
        # aot_eager traces + fuses ops without requiring Triton/CUDA.
        # Works on MPS and CPU; gives ~10-30% speedup by eliminating Python
        # dispatch overhead on repeated forward+backward calls.
        backend = "inductor" if device.type == "cuda" else "aot_eager"
        model = torch.compile(model, backend=backend)
        print(f"torch.compile: ON  (backend={backend})")

    n_params = count_params(model)
    print(f"\n{label}  |  params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  d_model={cfg.d_model}  n_heads={cfg.n_heads}  n_layers={cfg.n_layers}")
    print(f"  vocab={vocab_size}  seq_len={cfg.max_seq_len}")
    print(f"  effective batch = {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print()

    if args.optimizer == "muon":
        print("Optimizer: Muon (matrices) + AdamW (embeddings/scalars)")
        optimizer = configure_optimizers(
            model,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=0.1,
            device_type=device.type,
        )
    else:
        print("Optimizer: AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=device.type == "cuda",
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_dir = Path(args.volume_dir) if args.volume_dir else None
    if vol_dir:
        vol_dir.mkdir(parents=True, exist_ok=True)
        print(f"Persistent volume backup: {vol_dir}")

    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_s", "tok_per_sec"])

    best_val = float("inf")
    step = 0
    t0 = time.time()
    tok_count = 0

    train_iter = iter(train_loader)
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        # Gradient accumulation
        accum_loss = 0.0
        for acc_step in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with autocast_ctx(device):
                # LoopGRKANpreyLM computes its own loss (expected LM + entropy)
                # internally; all other models return logits.
                if isinstance(model, LoopGRKANpreyLM):
                    loss = model(x, targets=y)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    # Add load-balance auxiliary loss for MoE models
                    if hasattr(model, "load_balance_loss"):
                        loss = loss + model.load_balance_loss()
                loss = loss / args.grad_accum

            loss.backward()
            accum_loss += loss.item()
            tok_count += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}")

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            tok_per_sec = tok_count / elapsed
            ppl = perplexity(val_loss)
            print(f"\n[step {step:6d}]  train={accum_loss:.4f}  val={val_loss:.4f}  "
                  f"ppl={ppl:.1f}  lr={lr:.2e}  {tok_per_sec:,.0f} tok/s  {elapsed/3600:.2f}h")
            log_writer.writerow([step, f"{accum_loss:.4f}", f"{val_loss:.4f}",
                                  f"{ppl:.2f}", f"{lr:.2e}", f"{elapsed:.1f}", f"{tok_per_sec:.0f}"])
            log_file.flush()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_cfg": cfg,
                    "model_type": args.model,
                    "n_params": n_params,
                }, out_dir / "best.pt")

            # Back up best.pt and log to persistent volume after every eval.
            # If the community pod is interrupted, the volume retains the last
            # best checkpoint. The copy is fast (<1 s for ~500 MB).
            if vol_dir and (out_dir / "best.pt").exists():
                shutil.copy2(out_dir / "best.pt", vol_dir / "best.pt")
                shutil.copy2(log_path, vol_dir / "train_log.csv")

        if step % args.save_interval == 0:
            ckpt_path = out_dir / f"step_{step:07d}.pt"
            torch.save({"step": step, "model": model.state_dict()}, ckpt_path)
            if vol_dir:
                shutil.copy2(ckpt_path, vol_dir / ckpt_path.name)

    pbar.close()
    log_file.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/3600:.2f}h  |  best val_loss={best_val:.4f}  ppl={perplexity(best_val):.1f}")

    if vol_dir:
        used_mb = sum(f.stat().st_size for f in vol_dir.rglob("*") if f.is_file()) / 1e6
        print(f"Volume {vol_dir}: {used_mb:.0f} MB written")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 scale MLPEdge vs MLP comparison")
    parser.add_argument("--model",
                        choices=["mlp", "mlpedge", "mlpedge_matched", "basis", "grkan",
                                 "mlp_local", "moe_grkan", "unit_grkan", "unit0_grkan", "loop_grkan"],
                        default="mlp",
                        help="mlp=standard GPT-2 FFN (124M), mlpedge=MLPEdge h=5 (103M), "
                             "mlpedge_matched=MLPEdge h=8 parameter-matched to MLP (124M), "
                             "grkan=GR-KAN rational FFN (KAT ICLR 2025) at local scale, "
                             "mlp_local=standard MLP at same local scale (baseline for grkan), "
                             "moe_grkan=sparse MoE of GR-KAN experts (8 experts, top-2), "
                             "unit0_grkan=single 3-layer GR-KAN unit (Stage 0 of modular chain), "
                             "unit_grkan=full modular chain (2 units × 3 layers)")
    parser.add_argument("--load-checkpoint", type=str, default=None, metavar="PATH",
                        help="Load full model state_dict from PATH before training. "
                             "Use for joint fine-tuning after staged training.")
    parser.add_argument("--freeze-unit", type=int, default=None, metavar="N",
                        help="(unit_grkan only) Freeze units 0..N-1 before training. "
                             "Use with --load-unit to implement progressive unit training.")
    parser.add_argument("--load-unit", nargs=2, metavar=("IDX", "CKPT"), default=None,
                        help="(unit_grkan only) Load checkpoint CKPT into unit IDX. "
                             "Example: --load-unit 0 checkpoints/unit0/best.pt")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume-dir", type=str, default=None,
                        help="Path to persistent RunPod network volume for checkpoint backup. "
                             "best.pt and train_log.csv are copied here after every eval interval, "
                             "so progress survives community-cloud interruptions.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to reduce activation VRAM by ~10x "
                             "at the cost of ~33%% extra compute (recomputes each block during backward)")
    # ── LoopGRKAN options ──────────────────────────────────────────────────────
    parser.add_argument("--loop-t-max", type=int, default=4,
                        help="(loop_grkan) Max recurrent steps T_max (default 4, Ouro uses 4 for stability)")
    parser.add_argument("--loop-beta", type=float, default=0.1,
                        help="(loop_grkan) Entropy regularization coefficient β (default 0.1; "
                             "reduce to 0.05 for Stage II)")
    parser.add_argument("--loop-init-std", type=float, default=0.0,
                        help="(loop_grkan) EqR RI: std of random perturbation added to z₀ "
                             "(0=disabled; 0.1 recommended)")
    parser.add_argument("--loop-noise-std", type=float, default=0.0,
                        help="(loop_grkan) EqR NI: std of per-step additive noise "
                             "(0=disabled; 0.01 recommended)")
    parser.add_argument("--freeze-body", action="store_true",
                        help="(loop_grkan Stage II) Freeze body blocks, train only exit gate")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile (aot_eager on MPS/CPU, inductor on CUDA). "
                             "Eliminates Python dispatch overhead; ~10-30%% faster after warm-up.")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                        help="adamw=standard AdamW (default, works everywhere); "
                             "muon=Muon for 2D weight matrices + AdamW for embeddings/scalars. "
                             "Muon orthogonalizes momentum via Newton-Schulz-5, converging ~2× "
                             "faster per token on CUDA. Falls back to AdamW on MPS/CPU automatically.")
    # ── Grouped function-basis KAN FFN options (--model basis) ──────────────────
    parser.add_argument("--basis-family", type=str, default="soft_tree",
                        help="(basis) univariate basis: soft_tree, chebyshev, legendre, gaussian, "
                             "inverse_quadratic, wendland, triangular_hat, quadratic_hat, relu_power")
    parser.add_argument("--basis-groups", type=int, default=8,
                        help="(basis) group-shared coefficient groups; d_model and hidden must divide by it")
    parser.add_argument("--basis-input-norm", type=str, default="tanh",
                        choices=["none", "tanh", "clamp"],
                        help="(basis) input normalization applied before the basis")
    parser.add_argument("--basis-degree", type=int, default=5,
                        help="(basis) polynomial degree for chebyshev/legendre families")
    parser.add_argument("--basis-tree-depth", type=int, default=3,
                        help="(basis, soft_tree) oblivious-tree depth; n_basis = 2**depth")
    parser.add_argument("--basis-tree-steepness", type=float, default=1.0,
                        help="(basis, soft_tree) initial gate steepness β (GuppyLM optimum ≈1; β>=4 saturates)")
    args = parser.parse_args()
    train(args)
