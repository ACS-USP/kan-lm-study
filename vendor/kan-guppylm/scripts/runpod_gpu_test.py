"""
GPU environment smoke test for RunPod pods.

Run this before committing to a long training job to verify:
  - CUDA is available (not a CPU-only PyTorch install)
  - The repo and model code import correctly
  - A small forward + backward pass works on this GPU
  - Gradient checkpointing works
  - Prints VRAM usage so you can sanity-check memory headroom

Usage (on the pod):
    /opt/conda/bin/python scripts/runpod_gpu_test.py

Exit codes:
    0  — all checks passed
    1  — something is broken (error printed to stderr)
"""

import sys
import time
from pathlib import Path

# Allow running from the repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))


def fail(msg: str):
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check_cuda():
    import torch
    print(f"PyTorch version : {torch.__version__}")
    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() returned False.\n"
            "  Most likely cause: PyTorch was installed from default PyPI (CPU-only build).\n"
            "  Fix: use /opt/conda/bin/python instead of uv/pip-installed Python."
        )
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"CUDA device     : {name}")
    print(f"VRAM total      : {vram_gb:.1f} GB")
    return device


def check_matmul(device):
    import torch
    print("Matrix multiply … ", end="", flush=True)
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = a @ b
    assert c.shape == (1024, 1024)
    torch.cuda.synchronize()
    print("OK")


def check_model(device):
    import torch
    import torch.nn.functional as F
    from kanprey.config import ModelConfig
    from kanprey.model import MLPTransformer

    print("Model import    … ", end="", flush=True)
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        dropout=0.0,
    )
    model = MLPTransformer(cfg).to(device)
    model.set_gradient_checkpointing(True)
    model.train()
    print("OK")

    print("Forward pass    … ", end="", flush=True)
    B, T = 4, 128
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    t0 = time.time()
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    assert logits.shape == (B, T, cfg.vocab_size), f"unexpected logits shape {logits.shape}"
    print(f"OK  ({elapsed*1000:.0f} ms, loss={loss.item():.3f})")

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak VRAM used  : {peak_mb:.0f} MB")


def main():
    print("=" * 50)
    print("RunPod GPU smoke test")
    print("=" * 50)

    device = check_cuda()
    check_matmul(device)
    check_model(device)

    print("=" * 50)
    print("ALL CHECKS PASSED — safe to launch training.")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
