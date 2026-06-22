#!/usr/bin/env python3
"""
Profile forward-pass time breakdown for GuppyLM MLP vs Chebyshev basis.

Produces JSON with:
- Total forward time (ms)
- Time in basis evaluation (Chebyshev only)
- Time in linear projections
- Time in attention
- Time in embeddings / head
- Python/framework overhead (inferred)
- Memory bandwidth estimate
- MFU estimate

Usage:
    cd ~/Documents/GitHub/kan-guppylm
    source .venv/bin/activate
    python scripts/profile_guppylm_forward.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from kanprey.config import ModelConfig
from kanprey.model import MLPTransformer, BasisKANpreyLM

# ── paths ─────────────────────────────────────────────────────────────────────
MLP_CKPT = Path("checkpoints/mlp_s42/best.pt")
CHEB_CKPT = Path("checkpoints/basis_confirm/cheb_d3_g8_s42/best.pt")
OUT_JSON = Path("results/profiler_breakdown.json")

# ── profiling config ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
SEQ_LEN = 128
N_WARMUP = 10
N_MEASURED = 50

# M4 Pro MPS peak FLOPS (FP16/BF16). Apple claims ~38 TFLOPS for M4 Pro GPU,
# but MPS realistically achieves a fraction. Use a conservative estimate.
# For MFU we mainly care about *relative* MFU between the two models.
PEAK_MPS_TFLOPS = 38.0  # M4 Pro GPU claimed peak FP16/BF16 (used for MFU denominator)


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_cfg", ModelConfig())
    model_type = ckpt.get("model_type", "kan")
    if model_type == "mlp":
        model = MLPTransformer(cfg)
    elif model_type == "basis":
        model = BasisKANpreyLM(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg, model_type


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def theoretical_flops_per_token(cfg: ModelConfig, model_type: str) -> int:
    """Return estimated FLOPs per token for a full forward pass."""
    d = cfg.d_model
    h = cfg.d_model * 4  # standard 4x expansion
    L = cfg.n_layers
    V = cfg.vocab_size
    T = cfg.max_seq_len  # used for attention QK^T

    # Embeddings + head (tied, so count once)
    embed_flops = 2 * V * d

    # Attention per layer:
    # QKV projection: 3 * (2 * d * d) = 6*d^2
    # Attention scores QK^T: 2 * d * T (per head, scaled by n_heads)
    #   Actually head_dim = d / n_heads, so QK is (T * head_dim) @ (head_dim * T)
    #   -> 2 * T * T * head_dim per head -> 2 * T^2 * d total
    # Softmax is negligible.
    # Attn @ V: 2 * T * T * d
    # Output projection: 2 * d * d
    # Total attention per layer: ~4*d^2 + 4*T^2*d
    attn_flops = 4 * d * d + 4 * T * T * d

    # FFN per layer
    if model_type == "mlp":
        # Linear1: 2*d*h, GELU ~1 FLOP/element (negligible), Linear2: 2*h*d
        ffn_flops = 2 * d * h + 2 * h * d  # = 16*d^2
    elif model_type == "basis":
        # Basis1: Chebyshev recurrence. Degree=dg, each step ~3 FLOPs/element.
        dg = cfg.basis_degree
        basis1_flops = 3 * dg * d
        # Linear1: 2*d*h
        lin1_flops = 2 * d * h
        # Basis2: 3*dg*h
        basis2_flops = 3 * dg * h
        # Linear2: 2*h*d
        lin2_flops = 2 * h * d
        ffn_flops = basis1_flops + lin1_flops + basis2_flops + lin2_flops
    else:
        ffn_flops = 2 * d * h + 2 * h * d

    # LayerNorms: ~5 FLOPs/element per LN, 2 per block = 10*d*L
    ln_flops = 10 * d * L

    total = embed_flops + L * (attn_flops + ffn_flops) + ln_flops
    return total


def profile_model(model: nn.Module, cfg: ModelConfig, model_type: str, device: torch.device):
    """Run manual timing loops and return stats dict."""
    x = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # Measured runs
    times_ms = []
    with torch.no_grad():
        for _ in range(N_MEASURED):
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = torch.tensor(times_ms)
    median_ms = float(times_ms.median())
    mean_ms = float(times_ms.mean())
    std_ms = float(times_ms.std())
    min_ms = float(times_ms.min())

    tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / (median_ms / 1000.0)
    flops_per_token = theoretical_flops_per_token(cfg, model_type)
    achieved_tflops = tokens_per_sec * flops_per_token / 1e12
    mfu = achieved_tflops / PEAK_MPS_TFLOPS * 100.0

    return {
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "tokens_per_sec": tokens_per_sec,
        "flops_per_token": flops_per_token,
        "achieved_tflops": achieved_tflops,
        "mfu_percent": mfu,
    }


def profile_with_torch_profiler(model: nn.Module, cfg: ModelConfig, device: torch.device):
    """Use torch.profiler to get kernel-level breakdown."""
    x = torch.randint(0, cfg.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # MPS profiler activity is not supported as of PyTorch 2.6

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

    # Aggregate by operator name
    events = prof.key_averages(group_by_input_shape=False)
    total_cpu_us = sum(e.cpu_time_total for e in events)

    # Categorize
    matmul_us = 0.0
    embedding_us = 0.0
    softmax_us = 0.0
    native_layer_norm_us = 0.0
    gelu_us = 0.0
    chebyshev_us = 0.0  # inferred from non-matmul, non-embedding FFN time
    other_us = 0.0

    for e in events:
        name = e.key.lower()
        t = e.cpu_time_total
        if "mm" in name or "matmul" in name or "bmm" in name or "addmm" in name:
            matmul_us += t
        elif "embedding" in name:
            embedding_us += t
        elif "softmax" in name:
            softmax_us += t
        elif "layer_norm" in name or "native_layer_norm" in name:
            native_layer_norm_us += t
        elif "gelu" in name or "glu" in name:
            gelu_us += t
        else:
            other_us += t

    # For Chebyshev on CPU profiling, the basis recurrence shows up in "other"
    # because it's pure Python/PyTorch ops (mul, sub). We can't easily separate
    # it from framework overhead without CUDA profiling.
    return {
        "total_cpu_ms": total_cpu_us / 1000.0,
        "matmul_ms": matmul_us / 1000.0,
        "embedding_ms": embedding_us / 1000.0,
        "softmax_ms": softmax_us / 1000.0,
        "layer_norm_ms": native_layer_norm_us / 1000.0,
        "gelu_ms": gelu_us / 1000.0,
        "other_ms": other_us / 1000.0,
    }


def profile_chebyshev_isolated(device: torch.device):
    """Time just the GroupedBasisActivation forward in isolation."""
    from kanprey.basis_layers import GroupedBasisActivation

    d = 384
    groups = 8
    degree = 3
    batch = 32
    seq = 128

    m = GroupedBasisActivation(d, num_groups=groups, family="chebyshev", degree=degree).to(device)
    x = torch.randn(batch, seq, d, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = m(x)
        if device.type == "mps":
            torch.mps.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(N_MEASURED):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = m(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = torch.tensor(times_ms)
    median_ms = float(times_ms.median())

    # Theoretical FLOPs for basis1 in isolation
    # For each element: degree steps * ~3 FLOPs
    theoretical_flops = batch * seq * d * degree * 3
    tokens_per_sec = (batch * seq) / (median_ms / 1000.0)
    achieved_tflops = tokens_per_sec * theoretical_flops / 1e12
    mfu = achieved_tflops / PEAK_MPS_TFLOPS * 100.0

    return {
        "median_ms": median_ms,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "tokens_per_sec": tokens_per_sec,
        "theoretical_flops": theoretical_flops,
        "achieved_tflops": achieved_tflops,
        "mfu_percent": mfu,
    }


def profile_mlp_ffn_isolated(device: torch.device):
    """Time just the MLP FFN forward in isolation."""
    from kanprey.model import MLPFFN
    from kanprey.config import ModelConfig

    cfg = ModelConfig(d_model=384, n_layers=6, dropout=0.0)
    m = MLPFFN(cfg).to(device)
    batch = 32
    seq = 128
    d = 384
    x = torch.randn(batch, seq, d, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = m(x)
        if device.type == "mps":
            torch.mps.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(N_MEASURED):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = m(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = torch.tensor(times_ms)
    median_ms = float(times_ms.median())

    # Theoretical FLOPs: Linear1 2*d*4d + GELU negligible + Linear2 2*4d*d = 16*d^2 per token
    theoretical_flops = 16 * d * d
    tokens_per_sec = (batch * seq) / (median_ms / 1000.0)
    achieved_tflops = tokens_per_sec * theoretical_flops / 1e12
    mfu = achieved_tflops / PEAK_MPS_TFLOPS * 100.0

    return {
        "median_ms": median_ms,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "tokens_per_sec": tokens_per_sec,
        "theoretical_flops": theoretical_flops,
        "achieved_tflops": achieved_tflops,
        "mfu_percent": mfu,
    }
def profile_basis_ffn_isolated(device: torch.device):
    """Time the full BasisKANFFN forward in isolation (fair comparison to MLPFFN)."""
    from kanprey.basis_layers import BasisKANFFN
    from kanprey.config import ModelConfig

    cfg = ModelConfig(d_model=384, n_layers=6, dropout=0.0,
                      basis_family="chebyshev", basis_degree=3, basis_groups=8,
                      basis_expand=4, basis_input_norm="tanh")
    m = BasisKANFFN(cfg).to(device)
    batch = 32
    seq = 128
    d = 384
    x = torch.randn(batch, seq, d, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = m(x)
        if device.type == "mps":
            torch.mps.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(N_MEASURED):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = m(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_ms = torch.tensor(times_ms)
    median_ms = float(times_ms.median())

    # Theoretical FLOPs: basis1 + linear1 + basis2 + linear2
    dg = cfg.basis_degree
    h = d * cfg.basis_expand
    basis1_flops = 3 * dg * d
    lin1_flops = 2 * d * h
    basis2_flops = 3 * dg * h
    lin2_flops = 2 * h * d
    theoretical_flops = basis1_flops + lin1_flops + basis2_flops + lin2_flops

    tokens_per_sec = (batch * seq) / (median_ms / 1000.0)
    achieved_tflops = tokens_per_sec * theoretical_flops / 1e12
    mfu = achieved_tflops / PEAK_MPS_TFLOPS * 100.0

    return {
        "median_ms": median_ms,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "tokens_per_sec": tokens_per_sec,
        "theoretical_flops": theoretical_flops,
        "achieved_tflops": achieved_tflops,
        "mfu_percent": mfu,
    }


def test_torch_compile_chebyshev(device: torch.device):
    """Test torch.compile speedup on Chebyshev basis (MPS may not support)."""
    from kanprey.basis_layers import GroupedBasisActivation

    d = 384
    groups = 8
    degree = 3
    batch = 32
    seq = 128

    m = GroupedBasisActivation(d, num_groups=groups, family="chebyshev", degree=degree).to(device)
    x = torch.randn(batch, seq, d, device=device)

    # Baseline
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = m(x)
        if device.type == "mps":
            torch.mps.synchronize()

    times_baseline = []
    with torch.no_grad():
        for _ in range(N_MEASURED):
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = m(x)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times_baseline.append((t1 - t0) * 1000.0)

    baseline_ms = float(torch.tensor(times_baseline).median())

    # Compiled
    try:
        m_compiled = torch.compile(m, mode="max-autotune")
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = m_compiled(x)
            if device.type == "mps":
                torch.mps.synchronize()

        times_compiled = []
        with torch.no_grad():
            for _ in range(N_MEASURED):
                if device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                _ = m_compiled(x)
                if device.type == "mps":
                    torch.mps.synchronize()
                t1 = time.perf_counter()
                times_compiled.append((t1 - t0) * 1000.0)

        compiled_ms = float(torch.tensor(times_compiled).median())
        speedup = baseline_ms / compiled_ms
        return {"baseline_ms": baseline_ms, "compiled_ms": compiled_ms, "speedup": speedup, "supported": True}
    except Exception as e:
        return {"baseline_ms": baseline_ms, "compiled_ms": None, "speedup": None, "supported": False, "error": str(e)}


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "peak_tflops": PEAK_MPS_TFLOPS,
        "models": {},
        "isolated": {},
    }

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\n[MLP] Loading checkpoint...")
    mlp_model, mlp_cfg, mlp_type = load_model(MLP_CKPT, device)
    mlp_params = count_params(mlp_model)
    print(f"[MLP] Params: {mlp_params:,}  |  Type: {mlp_type}")

    print("[MLP] Profiling full forward...")
    mlp_stats = profile_model(mlp_model, mlp_cfg, mlp_type, device)
    print(f"[MLP] Median forward: {mlp_stats['median_ms']:.2f} ms  |  "
          f"Throughput: {mlp_stats['tokens_per_sec']:.1f} tok/s  |  "
          f"MFU: {mlp_stats['mfu_percent']:.2f}%")

    print("[MLP] Torch profiler breakdown...")
    mlp_prof = profile_with_torch_profiler(mlp_model, mlp_cfg, device)
    print(f"[MLP] Profiler total CPU: {mlp_prof['total_cpu_ms']:.2f} ms  |  "
          f"matmul: {mlp_prof['matmul_ms']:.2f} ms  |  "
          f"other: {mlp_prof['other_ms']:.2f} ms")

    results["models"]["mlp"] = {
        "params": mlp_params,
        "config": {
            "d_model": mlp_cfg.d_model,
            "n_layers": mlp_cfg.n_layers,
            "vocab_size": mlp_cfg.vocab_size,
            "max_seq_len": mlp_cfg.max_seq_len,
        },
        "timing": mlp_stats,
        "profiler": mlp_prof,
    }

    # ── Chebyshev ─────────────────────────────────────────────────────────────
    print("\n[Chebyshev] Loading checkpoint...")
    cheb_model, cheb_cfg, cheb_type = load_model(CHEB_CKPT, device)
    cheb_params = count_params(cheb_model)
    print(f"[Chebyshev] Params: {cheb_params:,}  |  Type: {cheb_type}")
    print(f"[Chebyshev] Config: degree={cheb_cfg.basis_degree}, groups={cheb_cfg.basis_groups}, "
          f"expand={cheb_cfg.basis_expand}, family={cheb_cfg.basis_family}")

    print("[Chebyshev] Profiling full forward...")
    cheb_stats = profile_model(cheb_model, cheb_cfg, cheb_type, device)
    print(f"[Chebyshev] Median forward: {cheb_stats['median_ms']:.2f} ms  |  "
          f"Throughput: {cheb_stats['tokens_per_sec']:.1f} tok/s  |  "
          f"MFU: {cheb_stats['mfu_percent']:.2f}%")

    print("[Chebyshev] Torch profiler breakdown...")
    cheb_prof = profile_with_torch_profiler(cheb_model, cheb_cfg, device)
    print(f"[Chebyshev] Profiler total CPU: {cheb_prof['total_cpu_ms']:.2f} ms  |  "
          f"matmul: {cheb_prof['matmul_ms']:.2f} ms  |  "
          f"other: {cheb_prof['other_ms']:.2f} ms")

    results["models"]["chebyshev"] = {
        "params": cheb_params,
        "config": {
            "d_model": cheb_cfg.d_model,
            "n_layers": cheb_cfg.n_layers,
            "vocab_size": cheb_cfg.vocab_size,
            "max_seq_len": cheb_cfg.max_seq_len,
            "basis_degree": cheb_cfg.basis_degree,
            "basis_groups": cheb_cfg.basis_groups,
            "basis_expand": cheb_cfg.basis_expand,
        },
        "timing": cheb_stats,
        "profiler": cheb_prof,
    }

    # ── Isolated FFN comparison ───────────────────────────────────────────────
    print("\n[Isolated] Profiling MLP FFN in isolation...")
    mlp_ffn = profile_mlp_ffn_isolated(device)
    print(f"[Isolated MLP FFN] Median: {mlp_ffn['median_ms']:.2f} ms  |  "
          f"MFU: {mlp_ffn['mfu_percent']:.2f}%")

    print("[Isolated] Profiling full BasisKANFFN in isolation...")
    cheb_ffn = profile_basis_ffn_isolated(device)
    print(f"[Isolated BasisKANFFN] Median: {cheb_ffn['median_ms']:.2f} ms  |  "
          f"MFU: {cheb_ffn['mfu_percent']:.2f}%")

    print("[Isolated] Profiling Chebyshev basis only (no linears)...")
    cheb_basis = profile_chebyshev_isolated(device)
    print(f"[Isolated Chebyshev basis] Median: {cheb_basis['median_ms']:.2f} ms  |  "
          f"MFU: {cheb_basis['mfu_percent']:.2f}%")

    print("[Isolated] Testing torch.compile on Chebyshev basis...")
    compile_result = test_torch_compile_chebyshev(device)
    if compile_result["supported"]:
        print(f"[torch.compile] Baseline: {compile_result['baseline_ms']:.2f} ms  |  "
              f"Compiled: {compile_result['compiled_ms']:.2f} ms  |  "
              f"Speedup: {compile_result['speedup']:.2f}x")
    else:
        print(f"[torch.compile] Not supported on this device. Error: {compile_result.get('error', 'N/A')}")

    results["isolated"] = {
        "mlp_ffn": mlp_ffn,
        "basis_ffn": cheb_ffn,
        "chebyshev_basis": cheb_basis,
        "torch_compile": compile_result,
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    speedup_full = cheb_stats["median_ms"] / mlp_stats["median_ms"]
    speedup_isolated_ffn = cheb_ffn["median_ms"] / mlp_ffn["median_ms"]
    speedup_isolated_basis = cheb_basis["median_ms"] / mlp_ffn["median_ms"]
    mfu_gap = mlp_stats["mfu_percent"] - cheb_stats["mfu_percent"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full-model forward slowdown (Cheb / MLP):     {speedup_full:.2f}x")
    print(f"Isolated-FFN slowdown (BasisKANFFN / MLPFFN): {speedup_isolated_ffn:.2f}x")
    print(f"Isolated-basis slowdown (Basis only / MLPFFN):{speedup_isolated_basis:.2f}x")
    print(f"MLP full-model MFU:                           {mlp_stats['mfu_percent']:.2f}%")
    print(f"Chebyshev full-model MFU:                     {cheb_stats['mfu_percent']:.2f}%")
    print(f"MFU gap (MLP - Chebyshev):                    {mfu_gap:.2f} pp")
    if compile_result["supported"]:
        print(f"torch.compile speedup on Chebyshev basis:     {compile_result['speedup']:.2f}x")
    print("=" * 60)

    results["summary"] = {
        "speedup_full": speedup_full,
        "speedup_isolated_ffn": speedup_isolated_ffn,
        "speedup_isolated_basis": speedup_isolated_basis,
        "mfu_gap_pp": mfu_gap,
        "torch_compile_speedup": compile_result.get("speedup"),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()