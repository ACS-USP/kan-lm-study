"""Grouped function-basis activations for KAN-style FFNs.

These layers keep the GR-KAN scaling shape: a cheap channel-wise nonlinear
basis with coefficients shared within groups, followed by ordinary dense linear
maps.  They are intentionally written in plain PyTorch first so local GuppyLM
screening can validate math and stability before any Triton/H100 kernel work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

BasisFamily = Literal[
    "chebyshev",
    "legendre",
    "gaussian",
    "inverse_quadratic",
    "wendland",
    "triangular_hat",
    "quadratic_hat",
    "relu_power",
    "soft_tree",
]
InputNorm = Literal["none", "tanh", "clamp"]
InitMode = Literal["identity", "swish", "gelu", "random"]


@dataclass(frozen=True)
class BasisDiagnostics:
    family: str
    input_min: float
    input_max: float
    normalized_min: float
    normalized_max: float
    basis_mean: list[float]
    basis_std: list[float]
    mean_entropy: float | None
    max_center_occupancy: float | None


def _uniform_centers(n: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    if n < 2:
        raise ValueError("basis_centers must be >= 2")
    return torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)


def _rbf_width(n_centers: int, width_scale: float) -> float:
    if n_centers < 2:
        raise ValueError("basis_centers must be >= 2")
    if width_scale <= 0:
        raise ValueError("basis_width_scale must be positive")
    return width_scale / float(n_centers - 1)


class GroupedBasisActivation(nn.Module):
    """Group-shared univariate basis activation.

    Input and output shapes are identical.  For every scalar channel value, the
    layer evaluates a small basis and contracts it with coefficients shared by
    the channel's group:

    ``y[..., group, channel] = sum_k coeff[group, k] * phi_k(x)``.

    This is the same cheap placement as GR-KAN's group-rational activation, not
    a full per-edge KAN layer.  It is therefore suitable for local screening and
    has a direct fused-kernel path: generate basis values in registers and reduce
    with group coefficients.
    """

    def __init__(
        self,
        d_in: int,
        num_groups: int = 8,
        family: BasisFamily = "chebyshev",
        degree: int = 5,
        centers: int = 8,
        width_scale: float = 1.5,
        input_norm: InputNorm = "tanh",
        relu_power: int = 2,
        depth: int = 3,
        steepness: float = 4.0,
        init: InitMode = "identity",
    ):
        super().__init__()
        if d_in % num_groups != 0:
            raise ValueError(f"d_in={d_in} must be divisible by num_groups={num_groups}")
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if relu_power < 1:
            raise ValueError("relu_power must be >= 1")
        self.d_in = d_in
        self.g = num_groups
        self.d_g = d_in // num_groups
        self.family = family
        self.degree = degree
        self.centers = centers
        self.width_scale = width_scale
        self.input_norm = input_norm
        self.relu_power = relu_power
        self.depth = depth
        self.steepness = steepness
        self.n_basis = self._basis_count(family, degree, centers, depth)

        if self._uses_centers:
            self.register_buffer("center_grid", _uniform_centers(centers))
        else:
            self.register_buffer("center_grid", torch.empty(0))
        if self.family == "soft_tree":
            if depth < 1:
                raise ValueError("depth (soft-tree depth) must be >= 1")
            # Learnable split thresholds, shared across groups like `center_grid`
            # but trainable (the "learnable knot" property). Spread inside the
            # normalized input range; gates saturate harmlessly if they drift out.
            init_thresholds = torch.linspace(-1.0, 1.0, depth + 2)[1:-1].clone()
            self.tree_thresholds = nn.Parameter(init_thresholds)
            # Per-split gate steepness, parameterized through softplus so it stays
            # positive. Init moderate to keep gates soft (avoids early saturation).
            raw_steepness = math.log(math.expm1(steepness))
            self.tree_log_steepness = nn.Parameter(torch.full((depth,), float(raw_steepness)))

        self.coeff = nn.Parameter(torch.zeros(num_groups, self.n_basis))
        self._init_coeffs(init)

    @property
    def _uses_centers(self) -> bool:
        return self.family in {
            "gaussian",
            "inverse_quadratic",
            "wendland",
            "triangular_hat",
            "quadratic_hat",
            "relu_power",
        }

    @staticmethod
    def _basis_count(family: BasisFamily, degree: int, centers: int, depth: int = 1) -> int:
        if family in {"chebyshev", "legendre"}:
            return degree + 1
        if family == "relu_power":
            return centers * 2
        if family in {"gaussian", "inverse_quadratic", "wendland", "triangular_hat", "quadratic_hat"}:
            return centers
        if family == "soft_tree":
            return 2 ** depth
        raise ValueError(f"unknown basis family {family!r}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "none":
            return x
        if self.input_norm == "tanh":
            return torch.tanh(x)
        if self.input_norm == "clamp":
            return x.clamp(-1.0, 1.0)
        raise ValueError(f"unknown input_norm {self.input_norm!r}")

    def basis_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return basis values with shape ``x.shape + (n_basis,)``."""
        z = self._normalize(x)
        if self.family == "chebyshev":
            return self._chebyshev(z)
        if self.family == "legendre":
            return self._legendre(z)
        if self.family == "gaussian":
            return self._gaussian(z)
        if self.family == "inverse_quadratic":
            return self._inverse_quadratic(z)
        if self.family == "wendland":
            return self._wendland(z)
        if self.family == "triangular_hat":
            return self._triangular_hat(z)
        if self.family == "quadratic_hat":
            h = self._triangular_hat(z)
            return h * h
        if self.family == "relu_power":
            return self._relu_power(z)
        if self.family == "soft_tree":
            return self._soft_tree(z)
        raise ValueError(f"unknown basis family {self.family!r}")

    def _chebyshev(self, z: torch.Tensor) -> torch.Tensor:
        vals = [torch.ones_like(z), z]
        for _ in range(1, self.degree):
            vals.append(2.0 * z * vals[-1] - vals[-2])
        return torch.stack(vals[: self.degree + 1], dim=-1)

    def _legendre(self, z: torch.Tensor) -> torch.Tensor:
        vals = [torch.ones_like(z), z]
        for n in range(1, self.degree):
            vals.append(((2 * n + 1) * z * vals[-1] - n * vals[-2]) / (n + 1))
        return torch.stack(vals[: self.degree + 1], dim=-1)

    def _centered_radius(self, z: torch.Tensor) -> torch.Tensor:
        centers = self.center_grid.to(device=z.device, dtype=z.dtype)
        h = _rbf_width(self.centers, self.width_scale)
        return (z.unsqueeze(-1) - centers) / h

    def _gaussian(self, z: torch.Tensor) -> torch.Tensor:
        r = self._centered_radius(z)
        return torch.exp(-(r * r))

    def _inverse_quadratic(self, z: torch.Tensor) -> torch.Tensor:
        r = self._centered_radius(z)
        return torch.reciprocal(1.0 + r * r)

    def _triangular_hat(self, z: torch.Tensor) -> torch.Tensor:
        r = self._centered_radius(z).abs()
        return F.relu(1.0 - r)

    def _wendland(self, z: torch.Tensor) -> torch.Tensor:
        r = self._centered_radius(z).abs()
        t = F.relu(1.0 - r)
        return t.pow(4) * (4.0 * r + 1.0)

    def _relu_power(self, z: torch.Tensor) -> torch.Tensor:
        centers = self.center_grid.to(device=z.device, dtype=z.dtype)
        right = F.relu(z.unsqueeze(-1) - centers).pow(self.relu_power)
        left = F.relu(centers - z.unsqueeze(-1)).pow(self.relu_power)
        return torch.cat([right, left], dim=-1)

    def _soft_tree(self, z: torch.Tensor) -> torch.Tensor:
        """Oblivious soft-tree leaf memberships (partition of unity).

        Each of ``depth`` learnable thresholds defines a logistic gate; the
        membership of leaf ``ell in {0,1}^depth`` is the product of the gate
        (or its complement) for each split.  Returns shape ``z.shape + (2**depth,)``
        whose final axis sums to 1, so ``sum_ell mu_ell * v_ell`` is a convex
        combination of the leaf values (bounded, saturating).
        """
        thresholds = self.tree_thresholds.to(device=z.device, dtype=z.dtype)
        beta = F.softplus(self.tree_log_steepness.to(device=z.device, dtype=z.dtype))
        gates = torch.sigmoid((z.unsqueeze(-1) - thresholds) * beta)  # (..., depth)
        mu = torch.ones((*z.shape, 1), device=z.device, dtype=z.dtype)
        for k in range(self.depth):
            gk = gates[..., k : k + 1]
            mu = torch.cat([mu * (1.0 - gk), mu * gk], dim=-1)
        return mu

    @torch.no_grad()
    def _init_coeffs(self, init: InitMode) -> None:
        if init == "random":
            nn.init.normal_(self.coeff, std=0.02)
            return
        sample = torch.linspace(-3.0, 3.0, 2048)
        basis = self.basis_values(sample).to(torch.float64)
        if init == "identity":
            target = sample.to(torch.float64)
        elif init == "swish":
            target = (sample * torch.sigmoid(sample)).to(torch.float64)
        elif init == "gelu":
            target = F.gelu(sample).to(torch.float64)
        else:
            raise ValueError(f"unknown init {init!r}")
        # Ridge-stabilized least squares.  The small diagonal prevents unstable
        # coefficients for localized bases whose columns can be nearly collinear.
        eye = torch.eye(basis.shape[1], dtype=basis.dtype)
        coeff = torch.linalg.solve(basis.T @ basis + 1e-6 * eye, basis.T @ target)
        self.coeff.copy_(coeff.to(dtype=self.coeff.dtype).unsqueeze(0).expand_as(self.coeff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_g = x.reshape(-1, self.g, self.d_g)
        basis = self.basis_values(x_g)
        out = torch.einsum("ngdk,gk->ngd", basis, self.coeff)
        return out.reshape(shape)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> BasisDiagnostics:
        x_flat = x.detach().reshape(-1, self.d_in)
        z = self._normalize(x_flat)
        basis = self.basis_values(x_flat).reshape(-1, self.n_basis)
        mean = basis.mean(dim=0)
        std = basis.std(dim=0, unbiased=False)
        entropy: float | None = None
        max_occ: float | None = None
        if self._uses_centers or self.family == "soft_tree":
            weights = basis.clamp_min(0)
            mass = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = weights / mass
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = float(ent.mean().item())
            max_occ = float(probs.mean(dim=0).max().item())
        return BasisDiagnostics(
            family=self.family,
            input_min=float(x_flat.min().item()),
            input_max=float(x_flat.max().item()),
            normalized_min=float(z.min().item()),
            normalized_max=float(z.max().item()),
            basis_mean=[float(v) for v in mean.cpu()],
            basis_std=[float(v) for v in std.cpu()],
            mean_entropy=entropy,
            max_center_occupancy=max_occ,
        )


class BasisKANFFN(nn.Module):
    """Two-layer FFN using grouped basis activations around dense projections."""

    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.d_model * cfg.basis_expand
        self.basis1 = GroupedBasisActivation(
            cfg.d_model,
            num_groups=cfg.basis_groups,
            family=cfg.basis_family,
            degree=cfg.basis_degree,
            centers=cfg.basis_centers,
            width_scale=cfg.basis_width_scale,
            input_norm=cfg.basis_input_norm,
            relu_power=cfg.basis_relu_power,
            depth=cfg.basis_tree_depth,
            steepness=cfg.basis_tree_steepness,
            init="identity",
        )
        self.linear1 = nn.Linear(cfg.d_model, hidden)
        self.basis2 = GroupedBasisActivation(
            hidden,
            num_groups=cfg.basis_groups,
            family=cfg.basis_family,
            degree=cfg.basis_degree,
            centers=cfg.basis_centers,
            width_scale=cfg.basis_width_scale,
            input_norm=cfg.basis_input_norm,
            relu_power=cfg.basis_relu_power,
            depth=cfg.basis_tree_depth,
            steepness=cfg.basis_tree_steepness,
            init="swish",
        )
        self.linear2 = nn.Linear(hidden, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="linear")
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity="linear")
        self.linear2.weight.data.mul_(1.0 / math.sqrt(2.0))
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.basis1(x))
        return self.drop(self.linear2(self.basis2(h)))

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> dict[str, BasisDiagnostics]:
        h = self.linear1(self.basis1(x))
        return {
            "basis1": self.basis1.diagnostics(x),
            "basis2": self.basis2.diagnostics(h),
        }
