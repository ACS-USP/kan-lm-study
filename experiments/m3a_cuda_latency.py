#!/usr/bin/env python
"""
M3a: CUDA latency for the GuppyLM FFN variants (MLP / Chebyshev / rational
GR-KAN), eager and with whole-block torch.compile, so the "slower" verdict is
not gated on the immature MPS backend.

Self-contained except for kanprey (BasisKANFFN, GRKANFFN). Designed to run a
few seconds of GPU time. Run on the pod with the repo on PYTHONPATH:
  PYTHONPATH=<repo> python m3a_cuda_latency.py
"""
import sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from kanprey.config import ModelConfig
from kanprey.basis_layers import BasisKANFFN
from kanprey.model import GRKANFFN


def dev():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def sync(d):
    if d.type == "cuda": torch.cuda.synchronize()
    elif d.type == "mps": torch.mps.synchronize()


def bench(fn, d, iters=50, warmup=15):
    for _ in range(warmup):
        fn(); sync(d)
    ts = []
    for _ in range(iters):
        sync(d); t0 = time.perf_counter(); fn(); sync(d)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2]


class MLPFFN(nn.Module):
    def __init__(self, d, dff):
        super().__init__(); self.l1 = nn.Linear(d, dff); self.l2 = nn.Linear(dff, d)
    def forward(self, x): return self.l2(F.gelu(self.l1(x)))


def main():
    d = dev()
    gpu = torch.cuda.get_device_name(0) if d.type == "cuda" else str(d)
    print(f"device={d} ({gpu}) torch={torch.__version__}")
    cfg = ModelConfig(basis_family="chebyshev", basis_degree=3, basis_groups=8,
                      basis_input_norm="tanh")
    D, DFF = cfg.d_model, cfg.d_model * cfg.basis_expand
    B = 32 * 128
    x = torch.randn(B, D, device=d)
    tokens = B

    builds = {
        "MLP-GELU-4x": MLPFFN(D, DFF).to(d).eval(),
        "Chebyshev-d3-g8": BasisKANFFN(cfg).to(d).eval(),
        "Rational-GRKAN-g8": GRKANFFN(cfg).to(d).eval(),
    }
    out = {"device": gpu, "torch": torch.__version__, "batch_tokens": tokens, "results": {}}
    with torch.no_grad():
        base = None
        for name, m in builds.items():
            t_eager = bench(lambda: m(x), d)
            t_comp = None
            try:
                cm = torch.compile(m, mode="reduce-overhead")
                t_comp = bench(lambda: cm(x), d, iters=30, warmup=20)
            except Exception as e:
                print(f"[compile failed {name}] {e}")
            if name == "MLP-GELU-4x":
                base = t_eager
            out["results"][name] = {
                "eager_ms": round(t_eager, 3),
                "compiled_ms": round(t_comp, 3) if t_comp else None,
                "eager_tok_per_s": round(tokens / (t_eager/1000)),
                "compiled_tok_per_s": round(tokens / (t_comp/1000)) if t_comp else None,
            }
    # slowdowns vs MLP
    mlp_e = out["results"]["MLP-GELU-4x"]["eager_ms"]
    mlp_c = out["results"]["MLP-GELU-4x"]["compiled_ms"]
    for name, r in out["results"].items():
        r["eager_slowdown_vs_mlp"] = round(r["eager_ms"]/mlp_e, 2)
        if r["compiled_ms"] and mlp_c:
            r["compiled_slowdown_vs_mlp"] = round(r["compiled_ms"]/mlp_c, 2)

    print(json.dumps(out, indent=2))
    with open("m3a_cuda_latency_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote m3a_cuda_latency_result.json")


if __name__ == "__main__":
    main()
