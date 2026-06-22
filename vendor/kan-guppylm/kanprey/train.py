"""Training loop for KanpreyLM."""

import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from kanprey.config import ModelConfig, TrainConfig
from kanprey.dataset import get_dataloader, load_tokenizer, train_tokenizer
from kanprey.model import KANpreyLM, KATpreyLM, MLPEdgepreyLM, MLPTransformer, GRKANpreyLM, BasisKANpreyLM, SwiGLUTransformer


def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(model: KANpreyLM, loader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(model_cfg: ModelConfig | None = None, train_cfg: TrainConfig | None = None, model_type: str = "kan", dataset: str = "guppylm"):
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    torch.manual_seed(train_cfg.seed)
    device = detect_device()
    print(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────
    if dataset == "babylm":
        from kanprey.dataset_babylm import (
            train_babylm_tokenizer, load_babylm_tokenizer,
            get_babylm_dataloader, PAD_ID as _PAD_ID,
        )
        tok_path = train_cfg.babylm_tokenizer_path
        if not Path(tok_path).exists():
            print("Training BabyLM tokenizer…")
            tokenizer = train_babylm_tokenizer(
                vocab_size=model_cfg.vocab_size,
                save_path=tok_path,
                dataset_path=train_cfg.babylm_dataset_path,
            )
        else:
            tokenizer = load_babylm_tokenizer(tok_path)
            print(f"Loaded BabyLM tokenizer from {tok_path}")
        actual_vocab = tokenizer.get_vocab_size()
        model_cfg.vocab_size = actual_vocab
        print("Loading BabyLM dataset…")
        train_loader = get_babylm_dataloader(
            "train", tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            batch_size=train_cfg.batch_size,
            dataset_path=train_cfg.babylm_dataset_path,
        )
        val_loader = get_babylm_dataloader(
            "val", tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            batch_size=train_cfg.batch_size,
            dataset_path=train_cfg.babylm_dataset_path,
        )
    else:
        tok_path = train_cfg.tokenizer_path
        if not Path(tok_path).exists():
            print("Training tokenizer…")
            tokenizer = train_tokenizer(
                vocab_size=model_cfg.vocab_size,
                save_path=tok_path,
                dataset_name=train_cfg.dataset_name,
            )
        else:
            tokenizer = load_tokenizer(tok_path)
            print(f"Loaded tokenizer from {tok_path}")
        actual_vocab = tokenizer.get_vocab_size()
        if actual_vocab != model_cfg.vocab_size:
            print(f"Note: tokenizer vocab={actual_vocab} (requested {model_cfg.vocab_size}); using actual.")
            model_cfg.vocab_size = actual_vocab
        print("Loading dataset…")
        train_loader = get_dataloader(
            "train", tokenizer,
            batch_size=train_cfg.batch_size,
            max_seq_len=model_cfg.max_seq_len,
            dataset_name=train_cfg.dataset_name,
        )
        val_loader = get_dataloader(
            "test", tokenizer,
            batch_size=train_cfg.batch_size,
            max_seq_len=model_cfg.max_seq_len,
            dataset_name=train_cfg.dataset_name,
            shuffle=False,
        )
        pad_id = PAD_ID

    # Model
    if model_type == "kat":
        model = KATpreyLM(model_cfg).to(device)
        label = "KATpreyLM"
    elif model_type == "mlpedge":
        model = MLPEdgepreyLM(model_cfg).to(device)
        label = "MLPEdgepreyLM"
    elif model_type == "mlp":
        model = MLPTransformer(model_cfg).to(device)
        label = "MLP-Transformer"
    elif model_type == "swiglu":
        model = SwiGLUTransformer(model_cfg).to(device)
        label = "SwiGLU-Transformer"
    elif model_type == "grkan":
        model = GRKANpreyLM(model_cfg).to(device)
        label = "GRKANpreyLM"
    elif model_type == "basis":
        model = BasisKANpreyLM(model_cfg).to(device)
        label = f"BasisKANpreyLM[{model_cfg.basis_family}]"
    else:
        model = KANpreyLM(model_cfg).to(device)
        label = "KanpreyLM"
    summary = model.param_summary()
    print(f"\n{label}  |  total params: {summary['total']:,}")
    print(f"  embedding : {summary['embedding']:,}")
    print(f"  attention : {summary['attention']:,}")
    print(f"  kan_ffn   : {summary['kan_ffn']:,}")
    if model_type == "kat":
        print(f"  ffn grid_size={model_cfg.kan_grid_size}  kat_grid_size={model_cfg.kat_grid_size}  spline_order={model_cfg.kan_spline_order}")
    elif model_type == "mlpedge":
        print(f"  mlp_edge_hidden={model_cfg.mlp_edge_hidden}")
    elif model_type == "mlp":
        print(f"  standard 2-layer GELU FFN (4× expansion)")
    elif model_type == "grkan":
        print(f"  grkan m={model_cfg.grkan_m} n={model_cfg.grkan_n} groups={model_cfg.grkan_groups} denominator={model_cfg.grkan_denominator}")
    elif model_type == "basis":
        print(
            f"  basis_family={model_cfg.basis_family} degree={model_cfg.basis_degree} "
            f"groups={model_cfg.basis_groups} centers={model_cfg.basis_centers} "
            f"width_scale={model_cfg.basis_width_scale} input_norm={model_cfg.basis_input_norm} "
            f"expand={model_cfg.basis_expand}"
        )
    else:
        print(f"  grid_size={model_cfg.kan_grid_size}  spline_order={model_cfg.kan_spline_order}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=tuple(train_cfg.betas),
        weight_decay=train_cfg.weight_decay,
    )

    Path(train_cfg.checkpoint_dir).mkdir(exist_ok=True)
    log_path = Path(train_cfg.checkpoint_dir) / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val_loss = float("inf")
    step = 0
    grid_updated = False
    t0 = time.time()

    train_iter = iter(train_loader)
    model.train()

    pbar = tqdm(total=train_cfg.max_steps, desc="Training")

    while step < train_cfg.max_steps:
        # Refresh iterator when exhausted
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Learning rate schedule
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        # Adapt KAN grids once after warm-up
        if not grid_updated and step >= train_cfg.grid_update_step:
            print(f"\n[step {step}] Updating KAN grids…")
            model.update_grid_all(x)
            grid_updated = True
            print("  Done.")

        # Evaluate
        if step % train_cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            print(f"\n[step {step:5d}]  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  "
                  f"lr={lr:.2e}  elapsed={elapsed:.0f}s")
            log_writer.writerow([step, f"{loss.item():.4f}", f"{val_loss:.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
            log_file.flush()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "model_type": model_type,
                }
                torch.save(ckpt, f"{train_cfg.checkpoint_dir}/best.pt")

        # Periodic checkpoint
        if step % train_cfg.save_interval == 0:
            torch.save(
                {"step": step, "model": model.state_dict(), "val_loss": loss.item()},
                f"{train_cfg.checkpoint_dir}/step_{step:06d}.pt",
            )

    pbar.close()
    log_file.close()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s  |  best val_loss={best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["kan", "kat", "mlpedge", "mlp", "grkan", "basis", "swiglu"], default="kan",
                        help="Model variant: kan=KAN-FFN, kat=KAT+KAN-FFN, mlpedge=MLP-per-edge FFN, mlp=standard MLP transformer, grkan=GR-KAN rational FFN, basis=grouped alternative basis FFN, swiglu=SwiGLU gated MLP")
    parser.add_argument("--dataset", choices=["guppylm", "babylm"], default="guppylm",
                        help="Dataset: guppylm (ChatML, assistant-only loss) or babylm (standard LM)")
    parser.add_argument("--grid-size", type=int, default=5,
                        help="KAN-FFN grid size (5=expressive, 2=param-matched)")
    parser.add_argument("--kat-grid-size", type=int, default=3,
                        help="KAT attention Q/K grid size (only used with --model kat)")
    parser.add_argument("--mlp-edge-hidden", type=int, default=5,
                        help="Hidden units per edge MLP (only used with --model mlpedge)")
    parser.add_argument("--grkan-m", type=int, default=5,
                        help="GR-KAN numerator polynomial degree")
    parser.add_argument("--grkan-n", type=int, default=4,
                        help="GR-KAN denominator polynomial degree")
    parser.add_argument("--grkan-groups", type=int, default=8,
                        help="GR-KAN rational group count")
    parser.add_argument("--grkan-denominator", choices=["abs", "softplus", "square"], default="abs",
                        help="GR-KAN denominator ablation")
    parser.add_argument("--basis-family", choices=[
        "chebyshev", "legendre", "gaussian", "inverse_quadratic",
        "wendland", "triangular_hat", "quadratic_hat", "relu_power", "soft_tree",
    ], default="chebyshev")
    parser.add_argument("--basis-degree", type=int, default=5)
    parser.add_argument("--basis-groups", type=int, default=8)
    parser.add_argument("--basis-centers", type=int, default=8)
    parser.add_argument("--basis-width-scale", type=float, default=1.5)
    parser.add_argument("--basis-input-norm", choices=["none", "tanh", "clamp"], default="tanh")
    parser.add_argument("--basis-relu-power", type=int, default=2)
    parser.add_argument("--basis-expand", type=int, default=4)
    parser.add_argument("--basis-tree-depth", type=int, default=3,
                        help="Soft-tree depth (only --basis-family soft_tree); n_leaves=2**depth")
    parser.add_argument("--basis-tree-steepness", type=float, default=4.0,
                        help="Initial soft-tree gate steepness (only --basis-family soft_tree)")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m_cfg = ModelConfig(
        kan_grid_size=args.grid_size,
        kat_grid_size=args.kat_grid_size,
        mlp_edge_hidden=args.mlp_edge_hidden,
        grkan_m=args.grkan_m,
        grkan_n=args.grkan_n,
        grkan_groups=args.grkan_groups,
        grkan_denominator=args.grkan_denominator,
        basis_family=args.basis_family,
        basis_degree=args.basis_degree,
        basis_groups=args.basis_groups,
        basis_centers=args.basis_centers,
        basis_width_scale=args.basis_width_scale,
        basis_input_norm=args.basis_input_norm,
        basis_relu_power=args.basis_relu_power,
        basis_expand=args.basis_expand,
        basis_tree_depth=args.basis_tree_depth,
        basis_tree_steepness=args.basis_tree_steepness,
    )
    t_cfg = TrainConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    train(m_cfg, t_cfg, model_type=args.model, dataset=args.dataset)
