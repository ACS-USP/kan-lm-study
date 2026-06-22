"""
Sparse Mixture-of-Experts layers using GR-KAN (rational function) experts.

Each expert is a two-layer GR-KAN FFN identical in structure to GRKANFFN
(Yang & Wang, ICLR 2025). A linear router selects Top-K experts per token;
only those experts compute, keeping active FLOPs equal to the dense baseline.

References
----------
- KAT paper (ICLR 2025): rational expert FFNs
- Switch Transformer (Fedus et al.): Top-1 routing + load balance loss
- Mixtral (Mistral AI): Top-2 routing
- ∞-MoE (arXiv 2601.17680): MoE at GPT-2 scale
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kanprey.kan_layers import GroupRational


class ExpertRouter(nn.Module):
    """
    Linear router: produces soft expert probabilities and hard Top-K selection.

    Returns (indices, weights) where indices is (B*T, top_k) and weights is
    the normalized softmax probability for each selected expert.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        assert 1 <= top_k <= n_experts
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (N, d_model) flattened tokens
        Returns:
            indices: (N, top_k) selected expert indices
            weights: (N, top_k) normalized routing weights (sum to 1 per token)
            router_probs: (N, n_experts) full softmax probabilities (for load-balance loss)
        """
        logits = self.gate(x)                              # (N, n_experts)
        router_probs = F.softmax(logits, dim=-1)           # (N, n_experts)
        weights, indices = router_probs.topk(self.top_k, dim=-1)  # (N, top_k) each
        # Re-normalize selected weights so they sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return indices, weights, router_probs


def load_balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary loss encouraging uniform expert utilization.

    L = n_experts × Σᵢ (fraction_routed_i × mean_prob_i)

    When routing is perfectly uniform this equals 1.0. Penalises collapse
    to a single expert. Multiply by a small coefficient (e.g. 1e-2) and
    add to the language-modeling loss.
    """
    n = router_probs.size(-1)
    # fraction of tokens routed to each expert (hard assignment via argmax)
    top1 = router_probs.argmax(dim=-1)                     # (N,)
    counts = torch.bincount(top1, minlength=n).float()     # (n,)
    fraction = counts / counts.sum()                       # (n,)
    mean_prob = router_probs.mean(dim=0)                   # (n,)
    return n * (fraction * mean_prob).sum()


class GRKANExpertFFN(nn.Module):
    """
    Single GR-KAN expert FFN (identical structure to GRKANFFN in model.py).

    rat1(x) → Linear1(·) → rat2(h) → Linear2(·)

    rat1 is initialized to identity; rat2 approximates Swish.
    """

    def __init__(self, d_model: int, expand: int = 4,
                 m: int = 5, n: int = 4, groups: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        hidden = d_model * expand

        self.rat1 = GroupRational(d_model, groups, m, n, init="identity")
        self.linear1 = nn.Linear(d_model, hidden)
        self.rat2 = GroupRational(hidden, groups, m, n, init="swish")
        self.linear2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="linear")
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity="linear")
        self.linear2.weight.data.mul_(1.0 / math.sqrt(2.8178))
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.rat1(x))
        return self.drop(self.linear2(self.rat2(h)))


class MoEGRKANFFN(nn.Module):
    """
    Sparse MoE FFN: n_experts independent GR-KAN FFNs with Top-K routing.

    At each forward pass:
      1. Route each token to top_k experts.
      2. Run only selected experts on the tokens assigned to them.
      3. Weighted-sum expert outputs back into the residual stream.

    Active parameters per token = top_k / n_experts × total_expert_params.
    With top_k=2, n_experts=8: 25% of expert params activate per token.
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 expand: int = 4, m: int = 5, n: int = 4, groups: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = ExpertRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([
            GRKANExpertFFN(d_model, expand, m, n, groups, dropout)
            for _ in range(n_experts)
        ])

        # Accumulated during forward for the caller to add to loss.
        self._last_load_balance_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out: (B, T, d_model)

        Side-effect: sets self._last_load_balance_loss (scalar tensor).
        """
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)                      # (N, d_model)

        indices, weights, router_probs = self.router(x_flat)
        # indices, weights: (N, top_k)

        self._last_load_balance_loss = load_balance_loss(router_probs)

        out = torch.zeros_like(x_flat)                    # (N, d_model)

        # Dispatch: for each expert, gather the tokens routed to it.
        for k in range(self.top_k):
            expert_idx = indices[:, k]                    # (N,) which expert for slot k
            w_k = weights[:, k]                           # (N,) weight for slot k

            for e in range(self.n_experts):
                mask = expert_idx == e                    # (N,) bool
                if not mask.any():
                    continue
                x_e = x_flat[mask]                        # (n_e, d_model)
                y_e = self.experts[e](x_e)                # (n_e, d_model)
                out[mask] += w_k[mask].unsqueeze(-1) * y_e

        return out.reshape(B, T, C)
