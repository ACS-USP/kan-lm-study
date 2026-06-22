"""
Edge-function layers for KAN-style networks.

KANLinear     — B-spline basis (EfficientKAN, Liu et al. 2024 / Lau 2024).
MLPEdgeLinear — Each edge f_{i,j}(x) is a tiny MLP (R→R) instead of a spline.
                Same topology as KAN; no grid update required.
GroupRational — Group-rational activation for GR-KAN (Yang & Wang, ICLR 2025).
                Safe Padé rational function with shared numerator, per-group denominator.
                Uses the official rational_kat_cu CUDA extension when available
                (install: pip install rational-kat-cu), falling back to a pure-PyTorch
                Horner implementation. The CUDA backend avoids the ~123× training
                slowdown from backward-pass memory stalls identified in FlashKAT
                (Raffel & Chen, arXiv 2505.13813).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Optional CUDA backend for GroupRational ───────────────────────────────────
# rational_kat_cu provides a fused CUDA kernel for the group-rational forward and
# backward pass, fixing the memory-bound gradient accumulation bottleneck in the
# pure-PyTorch Horner loop (FlashKAT, arXiv 2505.13813).
#
# Install: pip install rational-kat-cu
# Source:  https://github.com/Adamdad/rational_kat_cu
try:
    from rational_kat_cu import rat_cuda  # type: ignore
    _RAT_CUDA_AVAILABLE = True
except ImportError:
    _RAT_CUDA_AVAILABLE = False


class KANLinear(nn.Module):
    """
    Single KAN layer: in_features -> out_features via B-spline basis + residual.

    Forward: y = base_weight(silu(x)) + spline_weight · B(x)

    where B(x) are (grid_size + spline_order) B-spline basis values evaluated
    for each input feature, and spline_weight has shape (out, in, n_basis).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: nn.Module | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order  # number of B-spline basis functions

        self.base_activation = base_activation or nn.SiLU()

        # Residual linear path: out x in
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        # Spline weights: out x in x n_basis
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.n_basis)
        )

        # Grid: (spline_order + grid_size + 1) knots, shared across all (in, out)
        # Shape: (in_features, grid_size + 2*spline_order + 1)
        n_knots = grid_size + 2 * spline_order + 1
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        # Extend grid with spline_order extra knots on each end
        step = (grid_range[1] - grid_range[0]) / grid_size
        left = grid[0] - step * torch.arange(spline_order, 0, -1)
        right = grid[-1] + step * torch.arange(1, spline_order + 1)
        grid = torch.cat([left, grid, right])  # (n_knots,)
        # Broadcast to (in_features, n_knots)
        self.register_buffer("grid", grid.unsqueeze(0).expand(in_features, -1).clone())

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions via Cox-de Boor recursion.

        Args:
            x: (batch, in_features)
        Returns:
            bases: (batch, in_features, n_basis)  n_basis = grid_size + spline_order
        """
        x = x.unsqueeze(-1)  # (B, in, 1) — broadcast over knot dimension

        # Order-0: indicator 1 on [t_i, t_{i+1})
        # grid: (in, n_knots),  grid[:,:-1]: (in, n_knots-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        # bases: (B, in, n_knots-1)

        # Cox-de Boor recursion.  After k iterations, bases has n_knots-1-k columns.
        for k in range(1, self.spline_order + 1):
            n = bases.shape[-1]  # n_knots - k before this step

            # Knot slices for i=0..n-2  (n-1 output bases)
            t_i   = self.grid[:, :n - 1]        # (in, n-1)
            t_ik  = self.grid[:, k:n - 1 + k]   # (in, n-1)   t_{i+k}
            t_i1  = self.grid[:, 1:n]            # (in, n-1)   t_{i+1}
            t_ik1 = self.grid[:, k + 1:n + k]   # (in, n-1)   t_{i+k+1}

            left  = (x - t_i)   / (t_ik  - t_i ).clamp(min=1e-8) * bases[..., :-1]
            right = (t_ik1 - x) / (t_ik1 - t_i1).clamp(min=1e-8) * bases[..., 1:]
            bases = left + right  # (B, in, n-1)

        # Final: (B, in, n_knots-1-spline_order) = (B, in, grid_size+spline_order) = (B, in, n_basis)
        assert bases.shape[-1] == self.n_basis, f"{bases.shape[-1]} != {self.n_basis}"
        return bases.contiguous()

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Adapt grid knots to cover the actual distribution of activations in x.
        Call once after warm-up (e.g. at step 200) on a representative batch.

        Args:
            x: (batch, in_features) — sample of activations
        """
        batch = x.shape[0]
        splines = self.b_splines(x)  # (B, in, n_basis)

        # Re-fit spline coefficients to current activations
        # spline_output = einsum("bin,oin->bo", splines, spline_weight)
        # We keep existing weights and only update the grid positions.

        # Build new adaptive grid from quantiles of x
        x_sorted, _ = x.sort(dim=0)  # (B, in)
        step = batch // self.grid_size
        grid_adaptive = x_sorted[
            [max(0, min(i * step, batch - 1)) for i in range(self.grid_size + 1)], :
        ].T  # (in, grid_size+1)

        # Add margins and extend with extra knots
        grid_range = grid_adaptive[:, -1] - grid_adaptive[:, 0]
        grid_adaptive = grid_adaptive + torch.stack(
            [
                -margin * grid_range,
                *([torch.zeros_like(grid_range)] * (self.grid_size - 1)),
                margin * grid_range,
            ],
            dim=1,
        )[:, : self.grid_size + 1]

        step_size = (grid_adaptive[:, -1] - grid_adaptive[:, 0]) / self.grid_size
        left = grid_adaptive[:, 0:1] - step_size.unsqueeze(1) * torch.arange(
            self.spline_order, 0, -1, device=x.device
        )
        right = grid_adaptive[:, -1:] + step_size.unsqueeze(1) * torch.arange(
            1, self.spline_order + 1, device=x.device
        )
        new_grid = torch.cat([left, grid_adaptive, right], dim=1)  # (in, n_knots)
        self.grid.copy_(new_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features) or (batch, seq, in_features)
        Returns:
            y: same leading dims, (out_features,) last dim
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (B, in)

        # Residual path
        base_out = F.linear(self.base_activation(x_flat), self.base_weight)  # (B, out)

        # Spline path: evaluate basis then contract
        splines = self.b_splines(x_flat)  # (B, in, n_basis)
        # spline_weight: (out, in, n_basis) -> einsum bik,oik->bo
        spline_out = torch.einsum("bik,oik->bo", splines, self.spline_weight)  # (B, out)

        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


class MLPEdgeLinear(nn.Module):
    """
    KAN-topology layer where each edge f_{i,j}(x_i) is a tiny MLP (R→R).

    Architecture per edge:
        h_i    = activation(x_i * W1[i] + b1[i])   shape (hidden,)
        f_{i,j} = W2[j,i] · h_i                    scalar

    Full forward (vectorised):
        H   = activation(einsum("bi,ih->bih", x, W1) + b1)   (B, in, hidden)
        out = einsum("bih,oih->bo", H, W2) + b_out            (B, out)

    This is structurally identical to KANLinear's spline path — the only
    difference is that the basis H is produced by a learned linear+activation
    step rather than a fixed B-spline evaluation.  No grid; no grid update.

    Parameters vs KANLinear(in, out, grid=G, order=k), n_basis = G+k:
        KANLinear    : out*in*n_basis + out*in              (spline + base)
        MLPEdgeLinear: out*in*hidden  + in*hidden*2 + out   (W2 + W1/b1 + b_out)
    With hidden == n_basis the parameter counts are nearly equal.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 5,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden
        self.n_basis = hidden          # expose same attribute as KANLinear for inventory code
        self.spline_order = 0          # sentinel — no grid/spline order concept
        self.activation = activation or nn.SiLU()

        # First layer: one (hidden,) unit per input channel
        self.W1 = nn.Parameter(torch.empty(in_features, hidden))
        self.b1 = nn.Parameter(torch.zeros(in_features, hidden))

        # Second layer: one (hidden,) → scalar per (out, in) edge
        self.W2 = nn.Parameter(torch.empty(out_features, in_features, hidden))

        # Output bias (shared across in_features, as in a standard linear layer)
        self.b_out = nn.Parameter(torch.zeros(out_features))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.normal_(self.W2, mean=0.0, std=0.1 / math.sqrt(self.in_features))

    def update_grid(self, x: torch.Tensor, **_kwargs):
        """No-op: MLPEdge has no grid to update."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)          # (B, in)

        # First layer: per-input-channel linear + activation → (B, in, hidden)
        H = self.activation(
            torch.einsum("bi,ih->bih", x_flat, self.W1) + self.b1
        )

        # Second layer: contract over (in, hidden) per output → (B, out)
        out = torch.einsum("bih,oih->bo", H, self.W2) + self.b_out

        return out.reshape(*orig_shape[:-1], self.out_features)


class GroupRational(nn.Module):
    """
    Group-Rational activation for GR-KAN (Yang & Wang, ICLR 2025).

    Applies g learnable rational functions to groups of input channels.  The
    baseline denominator is the corrected Safe Padé form:

        F(x) = (a₀ + a₁x + … + aₘxᵐ) / (1 + |b₁x + … + bₙxⁿ|)

    Local ablations can swap the denominator to ``softplus`` or ``square`` while
    preserving the same grouped layout and coefficient shapes.
    """

    def __init__(
        self,
        d_in: int,
        num_groups: int = 8,
        m: int = 5,
        n: int = 4,
        init: str = "identity",
        denominator: str = "abs",
    ):
        super().__init__()
        if d_in % num_groups != 0:
            raise ValueError(f"d_in={d_in} must be divisible by num_groups={num_groups}")
        if m < 1 or n < 1:
            raise ValueError("m and n must both be >= 1")
        if denominator not in {"abs", "softplus", "square"}:
            raise ValueError(f"unknown rational denominator {denominator!r}")
        self.d_in = d_in
        self.g = num_groups
        self.d_g = d_in // num_groups
        self.m = m
        self.n = n
        self.denominator = denominator

        # Shared numerator: a₀, a₁, …, aₘ
        self.a = nn.Parameter(torch.zeros(m + 1))
        # Per-group denominator: g × n, where b[i, k] = b_{k+1} for group i
        self.b = nn.Parameter(torch.zeros(num_groups, n))

        self._init_coeffs(init)

    def _zero_denominator_value(self) -> float:
        if self.denominator == "softplus":
            return 1.0 + math.log(2.0)
        return 1.0

    @torch.no_grad()
    def _init_coeffs(self, init: str):
        denom0 = self._zero_denominator_value()
        if init == "identity":
            # F(x) = x at b=0 for every denominator mode.
            self.a.zero_()
            self.a[1] = denom0
            self.b.zero_()
        elif init == "swish":
            # Fit numerator to denom0 * Swish when b=0.
            x_fit = torch.linspace(-4.0, 4.0, 2000)
            y_fit = denom0 * x_fit * torch.sigmoid(x_fit)
            X = torch.stack([x_fit ** i for i in range(self.m + 1)], dim=1)
            a_fit = torch.linalg.lstsq(X, y_fit.unsqueeze(1)).solution.squeeze()
            self.a.copy_(a_fit[: self.m + 1])
            self.b.zero_()
        else:
            nn.init.normal_(self.a, std=0.1)
            self.b.zero_()

    def _denominator(self, q: torch.Tensor) -> torch.Tensor:
        if self.denominator == "abs":
            return 1.0 + q.abs()
        if self.denominator == "softplus":
            return 1.0 + F.softplus(q)
        if self.denominator == "square":
            return 1.0 + q * q
        raise RuntimeError(f"unknown rational denominator {self.denominator!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.d_in)  # (N, d_in)

        if self.denominator == "abs" and _RAT_CUDA_AVAILABLE and x.is_cuda:
            # Fused CUDA kernel implements the corrected Safe Padé denominator.
            out = rat_cuda(x_flat, self.a, self.b)
            return out.reshape(shape)

        # ── Pure-PyTorch fallback (Horner's method) ───────────────────────────
        x_g = x_flat.reshape(-1, self.g, self.d_g)   # (N, g, d_g)

        num = self.a[self.m]
        for i in range(self.m - 1, -1, -1):
            num = self.a[i] + x_g * num

        d = self.b[:, -1].view(1, self.g, 1)
        for i in range(self.n - 2, -1, -1):
            d = self.b[:, i].view(1, self.g, 1) + x_g * d
        q = x_g * d

        return (num / self._denominator(q)).reshape(shape)

    @staticmethod
    def cuda_available() -> bool:
        return _RAT_CUDA_AVAILABLE
