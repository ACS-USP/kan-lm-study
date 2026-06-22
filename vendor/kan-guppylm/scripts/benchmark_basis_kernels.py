#!/usr/bin/env python3
"""Microbenchmark Gaussian/RBF basis candidates before training.

This is a local harness only.  It does not launch training or RunPod jobs.  The
same formulas are written as small functions so tests can verify correctness and
future Triton kernels can use the output as a reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


@dataclass(frozen=True)
class BenchResult:
    name: str
    shape: tuple[int, int]
    device: str
    dtype: str
    forward_ms: float
    backward_ms: float
    forward_max_error: float
    grad_max_error: float
    output_mean: float
    output_std: float
    peak_memory_bytes: int | None


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def centers_and_width(n_centers: int, width_scale: float, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    if n_centers < 2:
        raise ValueError("n_centers must be >= 2")
    if width_scale <= 0:
        raise ValueError("width_scale must be positive")
    centers = torch.linspace(-1.0, 1.0, n_centers, device=device, dtype=dtype)
    return centers, width_scale / float(n_centers - 1)


def squared_radius(x: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    r = (x.unsqueeze(-1) - centers) / width
    return r * r


def gaussian_exact(x: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp(-squared_radius(x, centers, width))


def gaussian_exp2(x: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    return torch.exp2(-squared_radius(x, centers, width) / LN2)


def gaussian_clamped(x: torch.Tensor, centers: torch.Tensor, width: float, z_max: float = 16.0) -> torch.Tensor:
    return torch.exp(-squared_radius(x, centers, width).clamp_max(z_max))


def gaussian_lut_linear(
    x: torch.Tensor,
    centers: torch.Tensor,
    width: float,
    *,
    z_max: float = 16.0,
    table_size: int = 2048,
) -> torch.Tensor:
    z = squared_radius(x, centers, width).clamp(0.0, z_max)
    table_x = torch.linspace(0.0, z_max, table_size, device=x.device, dtype=x.dtype)
    table_y = torch.exp(-table_x)
    scaled = z * ((table_size - 1) / z_max)
    i0 = scaled.floor().to(torch.long).clamp(0, table_size - 2)
    frac = scaled - i0.to(scaled.dtype)
    y0 = table_y[i0]
    y1 = table_y[i0 + 1]
    return y0 + frac * (y1 - y0)


def inverse_quadratic(x: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    return torch.reciprocal(1.0 + squared_radius(x, centers, width))


def wendland_c2(x: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    r = ((x.unsqueeze(-1) - centers) / width).abs()
    t = F.relu(1.0 - r)
    return t.pow(4) * (4.0 * r + 1.0)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    x_base: torch.Tensor,
    centers: torch.Tensor,
    width: float,
    repeats: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # Warmup
    for _ in range(2):
        x = x_base.detach().clone().requires_grad_(True)
        y = fn(x, centers, width)
        y.square().mean().backward()
    synchronize(x_base.device)

    f_total = 0.0
    b_total = 0.0
    y_last: torch.Tensor | None = None
    grad_last: torch.Tensor | None = None
    for _ in range(repeats):
        x = x_base.detach().clone().requires_grad_(True)
        t0 = time.perf_counter()
        y = fn(x, centers, width)
        synchronize(x_base.device)
        t1 = time.perf_counter()
        y.square().mean().backward()
        synchronize(x_base.device)
        t2 = time.perf_counter()
        f_total += t1 - t0
        b_total += t2 - t1
        y_last = y.detach()
        grad_last = x.grad.detach()
    assert y_last is not None and grad_last is not None
    return (f_total / repeats) * 1000.0, (b_total / repeats) * 1000.0, y_last, grad_last


def benchmark_one(
    name: str,
    fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    x_base: torch.Tensor,
    centers: torch.Tensor,
    width: float,
    repeats: int,
    y_ref: torch.Tensor,
    grad_ref: torch.Tensor,
) -> BenchResult:
    f_ms, b_ms, y, grad = time_forward_backward(fn, x_base, centers, width, repeats)
    mem = None
    if x_base.device.type == "cuda":
        mem = torch.cuda.max_memory_allocated(x_base.device)
    return BenchResult(
        name=name,
        shape=tuple(x_base.shape),
        device=str(x_base.device),
        dtype=str(x_base.dtype),
        forward_ms=f_ms,
        backward_ms=b_ms,
        forward_max_error=float((y - y_ref).abs().max().item()),
        grad_max_error=float((grad - grad_ref).abs().max().item()),
        output_mean=float(y.mean().item()),
        output_std=float(y.std(unbiased=False).item()),
        peak_memory_bytes=mem,
    )


def run_benchmarks(
    *,
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    centers_count: int,
    width_scale: float,
    repeats: int,
    seed: int,
) -> list[BenchResult]:
    torch.manual_seed(seed)
    x_base = torch.randn(shape, device=device, dtype=dtype).clamp(-4.0, 4.0)
    centers, width = centers_and_width(centers_count, width_scale, device=device, dtype=dtype)
    f_ref, b_ref, y_ref, grad_ref = time_forward_backward(gaussian_exact, x_base, centers, width, repeats)
    results = [
        BenchResult(
            name="exact_torch_exp",
            shape=tuple(shape),
            device=str(device),
            dtype=str(dtype),
            forward_ms=f_ref,
            backward_ms=b_ref,
            forward_max_error=0.0,
            grad_max_error=0.0,
            output_mean=float(y_ref.mean().item()),
            output_std=float(y_ref.std(unbiased=False).item()),
            peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        )
    ]
    candidates: list[tuple[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]]] = [
        ("exp2_rewrite", gaussian_exp2),
        ("clamped_exp_z16", gaussian_clamped),
        ("lut_linear_z16_2048", gaussian_lut_linear),
        ("inverse_quadratic", inverse_quadratic),
        ("wendland_c2", wendland_c2),
    ]
    for name, fn in candidates:
        results.append(benchmark_one(name, fn, x_base, centers, width, repeats, y_ref, grad_ref))
    return results


def parse_shape(raw: str) -> tuple[int, int]:
    left, right = raw.lower().replace(" ", "").split("x", 1)
    return int(left), int(right)


def dtype_from_name(raw: str) -> torch.dtype:
    if raw == "float32":
        return torch.float32
    if raw == "float16":
        return torch.float16
    if raw == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="4096x384", help="activation shape as rows x cols")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--centers", type=int, default=8)
    parser.add_argument("--width-scale", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/basis_gaussian_microbench.json"))
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = dtype_from_name(args.dtype)
    results = run_benchmarks(
        shape=parse_shape(args.shape),
        device=device,
        dtype=dtype,
        centers_count=args.centers,
        width_scale=args.width_scale,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = [asdict(r) for r in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
