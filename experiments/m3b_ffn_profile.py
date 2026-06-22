#!/usr/bin/env python
"""
M3b: decompose the Chebyshev BasisKANFFN forward time and measure torch.compile
on the WHOLE FFN block (not just the isolated basis), to fix the over-strong
"fusion would substantially close the gap" claim.

Breakdown: MLP FFN, BasisKANFFN (eager), its two matmuls only, its two basis
activations only, and BasisKANFFN under torch.compile end-to-end.

Run with PYTHONPATH=<kan-guppylm>:
  PYTHONPATH=/.../kan-guppylm uv run python m3b_ffn_profile.py
"""
import sys, time
REPO = "/Users/felippealves/Documents/GitHub/kan-guppylm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from kanprey.config import ModelConfig
from kanprey.basis_layers import BasisKANFFN


def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()


def bench(fn, dev, iters=50, warmup=10):
    for _ in range(warmup):
        fn(); sync(dev)
    ts = []
    for _ in range(iters):
        sync(dev); t0 = time.perf_counter()
        fn(); sync(dev)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]  # median ms


class MLPFFN(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.l1 = nn.Linear(d, dff); self.l2 = nn.Linear(dff, d)
    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


def main():
    dev = device()
    print(f"device={dev}  torch={torch.__version__}")
    cfg = ModelConfig(basis_family="chebyshev", basis_degree=3, basis_groups=8,
                      basis_input_norm="tanh")
    d, dff = cfg.d_model, cfg.d_model * cfg.basis_expand
    B = 32 * 128  # batch*seq flattened
    x = torch.randn(B, d, device=dev)

    mlp = MLPFFN(d, dff).to(dev).eval()
    ffn = BasisKANFFN(cfg).to(dev).eval()

    with torch.no_grad():
        t_mlp = bench(lambda: mlp(x), dev)
        t_ffn = bench(lambda: ffn(x), dev)
        # matmuls only (replace basis with identity)
        t_mm = bench(lambda: ffn.linear2(ffn.linear1(x)), dev)
        # basis activations only (both)
        h = ffn.linear1(ffn.basis1(x))
        t_basis = bench(lambda: (ffn.basis1(x), ffn.basis2(h)), dev)

        # torch.compile on the WHOLE FFN block
        t_comp = None
        try:
            cffn = torch.compile(ffn, mode="reduce-overhead")
            t_comp = bench(lambda: cffn(x), dev, iters=30, warmup=20)
        except Exception as e:
            print(f"[torch.compile failed] {e}")

    print(f"\n{'component':<34}{'median ms':>12}")
    print(f"{'MLP FFN (GELU, 4x)':<34}{t_mlp:>12.3f}")
    print(f"{'BasisKANFFN Chebyshev (eager)':<34}{t_ffn:>12.3f}")
    print(f"{'  - two matmuls only':<34}{t_mm:>12.3f}")
    print(f"{'  - two basis activations only':<34}{t_basis:>12.3f}")
    if t_comp:
        print(f"{'BasisKANFFN + torch.compile':<34}{t_comp:>12.3f}")
    print()
    print(f"eager FFN slowdown vs MLP:        {t_ffn / t_mlp:.2f}x")
    print(f"matmuls account for:              {t_mm / t_ffn * 100:.0f}% of eager FFN")
    print(f"basis+overhead accounts for:      {(t_ffn - t_mm) / t_ffn * 100:.0f}% of eager FFN")
    if t_comp:
        print(f"compiled FFN slowdown vs MLP:     {t_comp / t_mlp:.2f}x")
        print(f"compile end-to-end speedup:       {t_ffn / t_comp:.2f}x")


if __name__ == "__main__":
    main()
