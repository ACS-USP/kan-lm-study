#!/usr/bin/env python3
"""Verify canonical Safe Padé GR-KAN formula before local reruns.

Checks the local PyTorch/MPS path used by GuppyLM-scale training. CUDA/Triton
kernel parity must be checked separately on RunPod/H100 for d12 runs.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from kanprey.config import ModelConfig
from kanprey.kan_layers import GroupRational
from kanprey.model import GRKANpreyLM


def canonical_reference(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, groups: int) -> torch.Tensor:
    """Canonical Safe Padé: P(x) / (1 + |sum_i b_i x^(i+1)|)."""
    d_g = x.shape[-1] // groups
    x_g = x.reshape(-1, groups, d_g)

    powers = [torch.ones_like(x_g)]
    for _ in range(1, a.numel()):
        powers.append(powers[-1] * x_g)
    num = sum(a[i] * powers[i] for i in range(a.numel()))

    denom_poly = torch.zeros_like(x_g)
    x_power = x_g
    for i in range(b.shape[1]):
        denom_poly = denom_poly + b[:, i].view(1, groups, 1) * x_power
        x_power = x_power * x_g
    denom = 1.0 + denom_poly.abs()
    return (num / denom).reshape_as(x)


def wrong_reference(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, groups: int) -> torch.Tensor:
    """Retracted formula: P(x) / (1 + sum_i |b_i| |x|^(i+1))."""
    d_g = x.shape[-1] // groups
    x_g = x.reshape(-1, groups, d_g)

    powers = [torch.ones_like(x_g)]
    for _ in range(1, a.numel()):
        powers.append(powers[-1] * x_g)
    num = sum(a[i] * powers[i] for i in range(a.numel()))

    denom = torch.ones_like(x_g)
    x_abs_power = x_g.abs()
    for i in range(b.shape[1]):
        denom = denom + b[:, i].abs().view(1, groups, 1) * x_abs_power
        x_abs_power = x_abs_power * x_g.abs()
    return (num / denom).reshape_as(x)


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    if not torch.allclose(got, expected, atol=atol, rtol=rtol):
        max_abs = (got - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs={max_abs:.6g}")


def formula_adversarial() -> dict[str, float]:
    torch.manual_seed(1)
    layer = GroupRational(d_in=4, num_groups=2, m=2, n=2, init="identity").double()
    with torch.no_grad():
        layer.a.copy_(torch.tensor([0.25, -0.5, 1.25], dtype=torch.float64))
        layer.b.copy_(torch.tensor([[1.0, -1.0], [-0.75, 0.5]], dtype=torch.float64))
    x = torch.tensor(
        [[[-2.0, -0.5, 0.5, 2.0], [1.5, -1.25, 0.75, -0.25]]],
        dtype=torch.float64,
    )
    got = layer(x)
    canonical = canonical_reference(x, layer.a, layer.b, groups=2)
    wrong = wrong_reference(x, layer.a, layer.b, groups=2)
    assert_close("GroupRational vs canonical", got, canonical, atol=1e-12, rtol=1e-12)
    diff_wrong = (got - wrong).abs().max().item()
    if diff_wrong < 1e-3:
        raise AssertionError(f"adversarial case did not separate wrong formula: diff={diff_wrong:.6g}")
    return {"max_abs_vs_wrong_formula": diff_wrong}


def gradcheck_canonical() -> dict[str, float | bool]:
    torch.manual_seed(2)
    layer = GroupRational(d_in=4, num_groups=2, m=2, n=2, init="random").double()
    with torch.no_grad():
        layer.b.copy_(torch.tensor([[0.7, -0.4], [-0.3, 0.9]], dtype=torch.float64))
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True) * 0.4

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return layer(inp)

    ok = torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)
    return {"gradcheck": bool(ok)}


def tiny_training_smoke() -> dict[str, float | str | int | bool]:
    torch.manual_seed(3)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        grkan_groups=4,
        grkan_m=3,
        grkan_n=2,
        grkan_expand=2,
    )
    model = GRKANpreyLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses: list[float] = []
    model.train()
    for _ in range(10):
        idx = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len), device=device)
        targets = idx.roll(shifts=-1, dims=1)
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise AssertionError("non-finite smoke-test loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise AssertionError(f"non-finite gradient in {name}")
        optimizer.step()
        for name, param in model.named_parameters():
            if "rat" in name and not torch.isfinite(param).all():
                raise AssertionError(f"non-finite rational parameter in {name}")
        losses.append(float(loss.detach().cpu()))

    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints/grkan_gate0_smoke.pt")
    torch.save({"model": model.state_dict(), "model_cfg": cfg, "losses": losses}, ckpt_path)
    reloaded = GRKANpreyLM(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(state["model"])
    return {
        "device": str(device),
        "steps": 10,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoint": str(ckpt_path),
        "checkpoint_load_ok": True,
    }


def main() -> None:
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "rational_cuda_backend_available": bool(GroupRational.cuda_available()),
        "formula_adversarial": formula_adversarial(),
        "gradcheck": gradcheck_canonical(),
        "smoke": tiny_training_smoke(),
    }
    out_path = Path("results/grkan_formula_verification.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
