"""
Muon optimizer for KanpreyLM.

Muon (Momentum + Orthogonalization) applies Newton-Schulz-5 orthogonalization to
the momentum buffer before each weight update. For 2D+ weight matrices this produces
near-orthogonal updates that converge significantly faster per token than AdamW —
empirically 2–3× fewer steps to reach the same validation loss.

Usage — split params into two groups then pass to configure_optimizers():
    matrix_params, scalar_params = split_param_groups(model)
    optimizer = configure_optimizers(matrix_params, scalar_params, lr=3e-4)

Reference:
    Kosson et al. 2024, "Muon: Momentum + Orthogonalization"
    Jordan et al. 2024 (used in nanochat / karpathy)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Newton-Schulz-5 orthogonalization ─────────────────────────────────────────

def _newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Compute the orthogonal factor of G via 5 Newton-Schulz iterations.

    Approximates G(G^T G)^{-1/2}, i.e., the polar factor U in G = US.
    Coefficients (a, b, c) fit a degree-5 polynomial to x^{-1/2} on [0.1, 1.9].
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315

    # Work in bfloat16 for speed; restore dtype at end
    orig_dtype = G.dtype
    X = G.to(torch.bfloat16 if G.device.type == "cuda" else torch.float32)
    X = X / (X.norm() + 1e-7)

    # Newton-Schulz works on tall/square matrices; transpose if wide
    transposed = X.size(-2) < X.size(-1)
    if transposed:
        X = X.mT

    for _ in range(steps):
        A = X @ X.mT
        X = a * X + b * (A @ X) + c * (A @ A @ X)

    if transposed:
        X = X.mT

    return X.to(orig_dtype)


# ── Muon optimizer ────────────────────────────────────────────────────────────

class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Newton-Schulz orthogonalization for 2D+ weight matrices.

    For each matrix parameter W with gradient G:
        buf  ←  momentum · buf + G          (momentum buffer)
        Ŵ   ←  NS5(buf)                    (orthogonalize)
        Ŵ   ←  Ŵ · max(1, m/n)^0.5        (scale by aspect ratio)
        W   ←  W − lr · Ŵ

    The aspect-ratio scaling keeps the effective learning rate consistent
    regardless of whether the weight is tall or wide.

    Only use this optimizer for parameters with ndim >= 2 that are true weight
    matrices (not embeddings). Pass embeddings and scalars to AdamW instead.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if g.ndim >= 2:
                    update = _newtonschulz5(buf, steps=ns_steps)
                    # Aspect-ratio normalisation (matches nanochat)
                    scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
                    update = update * scale
                else:
                    update = buf

                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(update, alpha=-lr)

        return loss


# ── Parameter group helpers ───────────────────────────────────────────────────

def split_param_groups(model: nn.Module) -> tuple[list, list]:
    """
    Split model parameters into two lists:
      - matrix_params: 2D+ weight matrices suitable for Muon
      - scalar_params: embeddings, biases, LayerNorm weights (use AdamW)

    Embeddings are excluded from Muon even though they are 2D, because their
    gradients are sparse (only a subset of rows receive gradient per step),
    which breaks the full-matrix orthogonalization assumption.
    """
    matrix_params = []
    scalar_params = []

    embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in embedding_ids or p.ndim < 2:
            scalar_params.append(p)
        else:
            matrix_params.append(p)

    return matrix_params, scalar_params


def configure_optimizers(
    model: nn.Module,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    adamw_betas: tuple = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.Optimizer:
    """
    Build a combined Muon + AdamW optimizer.

    Returns a single AdamW when running on CPU/MPS (Newton-Schulz is fast only on CUDA).
    On CUDA, uses Muon for all 2D weight matrices and AdamW for the rest.

    For LR scheduling, update both param groups by iterating optimizer.param_groups.
    """
    if device_type != "cuda":
        # Newton-Schulz on MPS is not meaningfully faster than AdamW
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=adamw_betas,
            weight_decay=weight_decay,
            fused=False,
        )

    matrix_params, scalar_params = split_param_groups(model)

    n_matrix = sum(p.numel() for p in matrix_params)
    n_scalar = sum(p.numel() for p in scalar_params)
    print(f"  Muon  → {len(matrix_params):3d} tensors  {n_matrix/1e6:.2f}M params")
    print(f"  AdamW → {len(scalar_params):3d} tensors  {n_scalar/1e6:.2f}M params")

    muon = Muon(matrix_params, lr=lr * 10, momentum=muon_momentum)
    adamw = torch.optim.AdamW(
        scalar_params,
        lr=lr,
        betas=adamw_betas,
        weight_decay=weight_decay,
        fused=True,
    )

    # Return as a ComboOptimizer so the training loop can call .step() / .zero_grad()
    # on a single object and update LR uniformly across both.
    return _ComboOptimizer(muon, adamw)


class _ComboOptimizer:
    """Thin wrapper combining Muon (matrices) + AdamW (scalars) into one object."""

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self._muon = muon
        self._adamw = adamw
        # Expose both param_groups for LR scheduling
        self.param_groups = muon.param_groups + adamw.param_groups

    def step(self):
        self._muon.step()
        self._adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, sd: dict):
        self._muon.load_state_dict(sd["muon"])
        self._adamw.load_state_dict(sd["adamw"])
