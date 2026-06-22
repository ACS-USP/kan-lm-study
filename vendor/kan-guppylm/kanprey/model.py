"""
Two model variants:

KANpreyLM  — standard attention + KAN feed-forward (B-spline replaces ReLU FFN).
KATpreyLM  — KAT attention + KAN feed-forward (fully KAN transformer).

KAT attention replaces the implicit linear kernel K(q,k)=q·k with a learned
kernel K(q,k)=φ(q)·φ(k) where φ is a position-wise KANLinear map applied
independently to each head's query and key vectors before the dot product.
The softmax, value aggregation, and output projection stay standard.
This is an open research direction: the KAT paper (Yang & Wang, ICLR 2025)
keeps attention unchanged and only replaces FFNs with rational-basis KAN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kanprey.config import ModelConfig
from kanprey.kan_layers import KANLinear, MLPEdgeLinear, GroupRational
from kanprey.basis_layers import BasisKANFFN
from kanprey.moe_layers import MoEGRKANFFN


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        # F.scaled_dot_product_attention uses Flash Attention on Ampere+ GPUs.
        # It never materialises the full [B, H, T, T] attention matrix, cutting
        # activation memory from ~19 GB to ~1 GB at B=32, T=1024, H=12.
        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class KANFFN(nn.Module):
    """Replaces the 2-layer ReLU FFN with a single KANLinear layer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kan = KANLinear(
            in_features=cfg.d_model,
            out_features=cfg.d_model,
            grid_size=cfg.kan_grid_size,
            spline_order=cfg.kan_spline_order,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.kan(x))


class KATAttention(nn.Module):
    """
    KAT (Kolmogorov-Arnold Transformer) attention.

    Replaces the implicit linear kernel K(q,k) = q·k / √d with a learned kernel
    K(q,k) = φ(q)·φ(k) / √d, where φ is a KANLinear feature map applied
    position-wise to each head's query and key vectors independently.

    This turns attention from a fixed dot-product similarity into a learned
    positive-semidefinite kernel function approximated by B-spline basis functions.
    Value aggregation and output projection remain standard linear layers.

    Why Q and K only (not V):
      - Q and K determine *which* tokens attend to *which* — the scoring function.
      - V carries the *content* being aggregated — no benefit to nonlinear scoring here.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Shared KAN feature maps for Q and K (separate weights, same architecture).
        # Operate on head_dim (64) not d_model (384) — cheap.
        # grid=3 on head_dim=64: params = 64×64×(3+3) + 64×64 = 28,672 per KAN
        self.kan_q = KANLinear(
            self.head_dim, self.head_dim,
            grid_size=cfg.kat_grid_size,
            spline_order=cfg.kat_spline_order,
        )
        self.kan_k = KANLinear(
            self.head_dim, self.head_dim,
            grid_size=cfg.kat_grid_size,
            spline_order=cfg.kat_spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
        # q, k, v: (B, n_heads, T, head_dim)

        # Apply KAN feature maps position-wise across all heads simultaneously.
        # Flatten (B, n_heads, T) into a single batch dimension for KANLinear.
        BHT = B * self.n_heads * T
        q = self.kan_q(q.reshape(BHT, self.head_dim)).view(B, self.n_heads, T, self.head_dim)
        k = self.kan_k(k.reshape(BHT, self.head_dim)).view(B, self.n_heads, T, self.head_dim)

        # Flash Attention in KAN feature space — same memory saving as Attention.
        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = KANFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class KATBlock(nn.Module):
    """Transformer block with KAT attention + KAN FFN — fully KAN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = KATAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = KANFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class _KanpreyBase(nn.Module):
    """Shared generate / param_summary logic for both model variants."""

    cfg: ModelConfig
    _gradient_checkpointing: bool = False

    def set_gradient_checkpointing(self, enabled: bool = True):
        self._gradient_checkpointing = enabled

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, KANLinear):
                module._init_weights()

    def param_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        attn_params = sum(
            p.numel() for block in self.blocks for p in block.attn.parameters()
        )
        ffn_params = sum(
            p.numel() for block in self.blocks for p in block.ffn.parameters()
        )
        emb_params = (
            sum(p.numel() for p in self.tok_emb.parameters())
            + sum(p.numel() for p in self.pos_emb.parameters())
        )
        return {
            "total": total,
            "embedding": emb_params,
            "attention": attn_params,
            "kan_ffn": ffn_params,
            "other": total - attn_params - ffn_params - emb_params,
        }


class KANpreyLM(_KanpreyBase):
    """Standard attention + KAN FFN (B-spline). Our baseline KAN model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (same as GuppyLM)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """Call after warm-up to adapt all KAN grids to actual activation distributions.

        Args:
            x_sample: (B, T) token-ID tensor — a representative training batch.
        """
        activations: dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(_module, inp, _out):
                activations[name] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            return hook

        handles = [
            block.ffn.kan.register_forward_hook(make_hook(f"block_{i}"))
            for i, block in enumerate(self.blocks)
        ]

        with torch.no_grad():
            self(x_sample)

        for h in handles:
            h.remove()

        for i, block in enumerate(self.blocks):
            key = f"block_{i}"
            if key in activations:
                block.ffn.kan.update_grid(activations[key])

class KATpreyLM(_KanpreyBase):
    """
    Fully KAN transformer: KAT attention + KAN FFN.

    Every learnable transformation in each block is KAN-based:
      - Q and K feature maps: KANLinear(head_dim → head_dim)  [KAT attention]
      - Feed-forward:         KANLinear(d_model → d_model)     [KAN FFN]
    Standard linear layers are retained only for V/output projections and
    the token/positional embeddings (these don't benefit from spline basis).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([KATBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """Adapt all KAN grids (FFN and attention Q/K maps) to real activation distributions."""
        activations: dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(_module, inp, _out):
                activations[name] = inp[0].detach().reshape(-1, inp[0].shape[-1])
            return hook

        handles = []
        for i, block in enumerate(self.blocks):
            handles.append(block.ffn.kan.register_forward_hook(make_hook(f"ffn_{i}")))
            handles.append(block.attn.kan_q.register_forward_hook(make_hook(f"attn_q_{i}")))
            handles.append(block.attn.kan_k.register_forward_hook(make_hook(f"attn_k_{i}")))

        with torch.no_grad():
            self(x_sample)

        for h in handles:
            h.remove()

        for i, block in enumerate(self.blocks):
            if f"ffn_{i}" in activations:
                block.ffn.kan.update_grid(activations[f"ffn_{i}"])
            if f"attn_q_{i}" in activations:
                block.attn.kan_q.update_grid(activations[f"attn_q_{i}"])
            if f"attn_k_{i}" in activations:
                block.attn.kan_k.update_grid(activations[f"attn_k_{i}"])


class MLPEdgeFFN(nn.Module):
    """Replaces the KAN-FFN with an MLP-edge layer (same topology, learned basis)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.edge = MLPEdgeLinear(
            in_features=cfg.d_model,
            out_features=cfg.d_model,
            hidden=cfg.mlp_edge_hidden,
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.edge(x))


class MLPEdgeBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MLPEdgeFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MLPEdgepreyLM(_KanpreyBase):
    """
    Standard attention + MLP-edge FFN.

    Each FFN edge f_{i,j}(x_i) is a tiny learned MLP (R→R) instead of a B-spline.
    Same KAN topology (additive decomposition per edge), no grid mechanism.
    Comparable parameter count to KANpreyLM at grid=2 when hidden=5.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MLPEdgeBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: MLPEdge has no grids to update."""


class GRKANFFN(nn.Module):
    """
    Group-Rational KAN FFN (KAT, Yang & Wang ICLR 2025).

    Replaces the standard MLP FFN with two GR-KAN layers:

        rat₁(x) → Linear₁(·) → rat₂(h) → Linear₂(·)

    where each GR-KAN layer = GroupRational activation followed by a linear map.
    rat₁ is initialized to identity; rat₂ is initialized to approximate Swish.
    This matches the weight-transfer-compatible structure from the paper (Fig. 4).

    Parameter overhead over an equivalent MLP: (m+1) + 2·n·g — negligible.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.grkan_expand
        g = cfg.grkan_groups
        m, n = cfg.grkan_m, cfg.grkan_n

        self.rat1 = GroupRational(cfg.d_model, g, m, n, init="identity", denominator=cfg.grkan_denominator)
        self.linear1 = nn.Linear(cfg.d_model, hidden)
        self.rat2 = GroupRational(hidden, g, m, n, init="swish", denominator=cfg.grkan_denominator)
        self.linear2 = nn.Linear(hidden, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Variance-preserving init for linear layers (He init, adjusted for rational α).
        # rat₁ ≈ identity (α=1) → standard kaiming; rat₂ ≈ swish (α≈2.82) → scale by 1/√α.
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="linear")
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity="linear")
        self.linear2.weight.data.mul_(1.0 / math.sqrt(2.8178))
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(self.rat1(x))
        return self.drop(self.linear2(self.rat2(h)))


class GRKANBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = GRKANFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GRKANpreyLM(_KanpreyBase):
    """
    Standard attention + GR-KAN FFN (Kolmogorov-Arnold Transformer, ICLR 2025).

    Replaces the MLP FFN with Group-Rational KAN layers: rational activations
    (Safe Padé, m=5/n=4) with parameter sharing across groups of edges, making
    the parameter count virtually identical to a standard MLP transformer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GRKANBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: rational functions have no grid to update."""


# ── Looped GR-KAN (Ouro + EqR) ────────────────────────────────────────────────

class LoopGRKANpreyLM(_KanpreyBase):
    """
    Looped GR-KAN language model combining Ouro-style recurrent depth with
    EqR attractor-shaping interventions.

    Architecture (Ouro, arXiv 2510.25741):
        emb(x) → [RI noise] → x₀
        for t in 1..T_max:
            x_t = body(x_{t-1})  +  [NI noise]   ← same ModuleUnit every step
            λ_t = sigmoid(exit_gate(mean(x_t)))   ← per-step exit probability

    Training loss (Stage I):
        L = Σ_t p_φ(t|x) · L_LM^(t)  −  β · H(p_φ(·|x))
        where p_φ is the learned exit distribution (survival × exit prob)
        and H is entropy (prevents collapse to always using T_max).

    EqR enhancements (arXiv 2605.21488, toggleable via config):
        RI: z₀ ~ N(0, σ²I) added at embedding output  [loop_init_std > 0]
        NI: x_{t+1} = x_t + (1-λ)·(body(x_t) - x_t) + β·ε  [loop_noise_std > 0]

    Inference: runs until CDF(exit) > loop_exit_threshold or T_max steps.
    Breadth scaling: run B restarts (via RI), pick lowest mean residual.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.body = ModuleUnit(cfg, n_layers=cfg.unit_n_layers)
        self.exit_gate = nn.Linear(cfg.d_model, 1, bias=True)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        # expose self.blocks for _KanpreyBase.param_summary compatibility
        self.blocks = self.body.blocks
        self._init_weights()
        nn.init.zeros_(self.exit_gate.weight)
        nn.init.zeros_(self.exit_gate.bias)

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        if self.cfg.loop_init_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.cfg.loop_init_std
        return x

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """One recurrent step with optional NI damping+noise (EqR)."""
        x_new = self.body(x)
        if self.cfg.loop_noise_std > 0 and self.training:
            lam = self.cfg.loop_damping
            noise = torch.randn_like(x) * self.cfg.loop_noise_std
            x_new = x + (1.0 - lam) * (x_new - x) + noise
        return x_new

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Training (targets provided): returns scalar loss = expected LM loss − β·entropy.
        Inference (targets=None):    returns logits (B, T, vocab_size).
        """
        x = self._embed(idx)
        T_max = self.cfg.loop_t_max
        if self.training and targets is not None:
            return self._forward_train(x, targets, T_max)
        return self._forward_infer(x, T_max)

    def _forward_train(self, x: torch.Tensor, targets: torch.Tensor,
                       T_max: int) -> torch.Tensor:
        per_step_loss, exit_probs = [], []
        survival = torch.ones(x.shape[0], device=x.device)

        for t in range(1, T_max + 1):
            x = self._step(x)
            lam_t = torch.sigmoid(self.exit_gate(x.mean(dim=1))).squeeze(-1)  # (B,)
            logits = self.head(self.ln_f(x))
            loss_t = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            per_step_loss.append(loss_t)
            p_t = lam_t * survival
            survival = survival * (1.0 - lam_t)
            if t == T_max:
                p_t = p_t + survival
            exit_probs.append(p_t.mean())

        p_stack = torch.stack(exit_probs)
        l_stack = torch.stack(per_step_loss)
        expected_loss = (p_stack * l_stack).sum()
        entropy = -(p_stack * (p_stack + 1e-8).log()).sum()
        return expected_loss - self.cfg.loop_beta * entropy

    def _forward_infer(self, x: torch.Tensor, T_max: int) -> torch.Tensor:
        threshold = self.cfg.loop_exit_threshold
        cdf = torch.zeros(x.shape[0], device=x.device)
        survival = torch.ones(x.shape[0], device=x.device)
        last_logits = None
        for t in range(1, T_max + 1):
            x = self.body(x)
            lam_t = torch.sigmoid(self.exit_gate(x.mean(dim=1))).squeeze(-1)
            cdf = cdf + survival * lam_t
            survival = survival * (1.0 - lam_t)
            last_logits = self.head(self.ln_f(x))
            if cdf.mean().item() >= threshold:
                break
        return last_logits

    def forward_breadth(self, idx: torch.Tensor, B: int = 4) -> torch.Tensor:
        """
        EqR breadth scaling: B independent RI restarts; select by lowest residual.
        Single-example only (batch size 1).
        """
        assert idx.shape[0] == 1
        T_max = self.cfg.loop_t_max
        best_logits, best_residual = None, float("inf")
        init_std = max(self.cfg.loop_init_std, 0.1)

        with torch.no_grad():
            for _ in range(B):
                B2, T = idx.shape
                pos = torch.arange(T, device=idx.device).unsqueeze(0)
                x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
                x = x + torch.randn_like(x) * init_std
                x_prev = x.clone()
                residuals = []
                for t in range(1, T_max + 1):
                    x = self.body(x)
                    if t >= T_max - 1:
                        residuals.append((x - x_prev).norm(dim=-1).mean().item())
                    x_prev = x.clone()
                logits = self.head(self.ln_f(x))
                mean_res = sum(residuals) / max(len(residuals), 1)
                if mean_res < best_residual:
                    best_residual, best_logits = mean_res, logits
        return best_logits

    def loop_residuals(self, idx: torch.Tensor) -> list[float]:
        """Return ||x_t - x_{t-1}||₂ per loop step (attractor convergence diagnostic)."""
        with torch.no_grad():
            x = self._embed(idx)
            x_prev = x.clone()
            residuals = []
            for _ in range(self.cfg.loop_t_max):
                x = self.body(x)
                residuals.append((x - x_prev).norm(dim=-1).mean().item())
                x_prev = x.clone()
        return residuals

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: rational functions have no grid to update."""


class MLPFFN(nn.Module):
    """Standard 2-layer FFN: Linear → GELU → Linear (GPT-2 style, 4× expansion)."""

    def __init__(self, cfg: ModelConfig, expand: int = 4):
        super().__init__()
        hidden = cfg.d_model * expand
        self.fc1 = nn.Linear(cfg.d_model, hidden)
        self.fc2 = nn.Linear(hidden, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MLPBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MLPFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MLPTransformer(_KanpreyBase):
    """
    Standard GPT-style transformer (MLP FFN, dot-product attention).

    Used as the fair baseline for the GPT-2 scale comparison.
    Architecture: d_model, n_heads, n_layers, FFN=4×d_model expansion + GELU.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MLPBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: standard MLP has no grids."""


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (Shazeer 2020, arXiv:2002.05202).

    down(SiLU(x W_gate) * (x W_up)). Hidden = 8/3 d_model so the three
    bias-free projections have the same weight count as a 4x GELU FFN
    (3 * d * (8d/3) = 8 d^2 = d*4d + 4d*d), giving a parameter-matched
    modern-FFN baseline.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = (cfg.d_model * 8) // 3   # 1024 for d_model=384
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class SwiGLUBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SwiGLUTransformer(_KanpreyBase):
    """GPT-style transformer with a parameter-matched SwiGLU FFN baseline."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SwiGLUBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: SwiGLU has no grids."""


class BasisKANBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = BasisKANFFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class BasisKANpreyLM(_KanpreyBase):
    """Standard attention + grouped function-basis KAN FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([BasisKANBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: grouped basis activations use fixed centers/recurrences."""

    @torch.no_grad()
    def basis_diagnostics(self, x_sample: torch.Tensor) -> dict[str, dict]:
        activations: dict[str, torch.Tensor] = {}

        def make_hook(name):
            def hook(_module, inp, _out):
                activations[name] = inp[0].detach()
            return hook

        handles = [
            block.ffn.register_forward_hook(make_hook(f"block_{i}"))
            for i, block in enumerate(self.blocks)
        ]
        self(x_sample)
        for h in handles:
            h.remove()
        out: dict[str, dict] = {}
        for i, block in enumerate(self.blocks):
            key = f"block_{i}"
            if key in activations:
                out[key] = {
                    name: diag.__dict__
                    for name, diag in block.ffn.diagnostics(activations[key]).items()
                }
        return out

# ── MoE GR-KAN ────────────────────────────────────────────────────────────────

class MoEGRKANBlock(nn.Module):
    """Transformer block with standard attention + sparse MoE GR-KAN FFN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MoEGRKANFFN(
            d_model=cfg.d_model,
            n_experts=cfg.n_moe_experts,
            top_k=cfg.moe_top_k,
            expand=cfg.grkan_expand,
            m=cfg.grkan_m,
            n=cfg.grkan_n,
            groups=cfg.grkan_groups,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MoEGRKANpreyLM(_KanpreyBase):
    """
    Standard attention + sparse Mixture-of-Experts GR-KAN FFN.

    Each transformer block contains n_moe_experts independent GR-KAN FFN experts.
    A linear router selects top_k experts per token; only those activate.
    Active FLOPs ≈ top_k/n_experts of a dense GR-KAN model.

    An auxiliary load-balance loss is accumulated during forward and exposed
    via self.load_balance_loss() for the training loop to add to the LM loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MoEGRKANBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._load_balance_coeff = cfg.load_balance_coeff
        self._init_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def load_balance_loss(self) -> torch.Tensor:
        """Sum of per-block load-balance losses, weighted by load_balance_coeff."""
        total = sum(
            b.ffn._last_load_balance_loss
            for b in self.blocks
            if b.ffn._last_load_balance_loss is not None
        )
        return self._load_balance_coeff * total

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: rational functions have no grid to update."""


# ── Sequential Unit Composition ───────────────────────────────────────────────

class ModuleUnit(nn.Module):
    """
    A self-contained group of GR-KAN transformer blocks forming one composable unit.

    During stand-alone training, a unit acts as a full LM with its own head.
    During composition (ModuleChainLM), heads are stripped and residual streams
    are chained: Unit-N receives the output of Unit-(N-1) as input.

    All units must share the same d_model (the residual stream is the interface).
    The tokenizer and embedding are shared or frozen after Unit-0 trains.
    """

    def __init__(self, cfg: ModelConfig, n_layers: int | None = None):
        super().__init__()
        depth = n_layers if n_layers is not None else cfg.unit_n_layers
        self.blocks = nn.ModuleList([GRKANBlock(cfg) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class ModuleChainLM(_KanpreyBase):
    """
    Compose multiple independently-trained ModuleUnit instances into one LM.

    Architecture:
        shared embedding → Unit-0 → Unit-1 → … → Unit-N → shared head

    Training protocol (progressive):
      1. Train a standalone GRKANpreyLM with unit_n_layers blocks as Unit-0.
      2. Load Unit-0 weights into ModuleChainLM (units[0]), freeze them.
      3. Train units[1] end-to-end with units[0] frozen.
      4. Repeat, adding one unit per stage.
      5. Optional: fine-tune all units jointly at the end.

    The residual stream dimension d_model is the interface — all units must
    share the same d_model. The embedding and head are weight-tied and shared.
    """

    def __init__(self, cfg: ModelConfig, n_units: int = 2,
                 layers_per_unit: int | None = None):
        super().__init__()
        self.cfg = cfg
        depth = layers_per_unit or cfg.unit_n_layers

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.units = nn.ModuleList([ModuleUnit(cfg, depth) for _ in range(n_units)])
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

        # blocks is expected by _KanpreyBase.param_summary — expose for compatibility
        self.blocks = nn.ModuleList(
            [block for unit in self.units for block in unit.blocks]
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for unit in self.units:
            x = unit(x)
        return self.head(x)

    def freeze_unit(self, unit_idx: int):
        """Freeze all parameters in a trained unit (call before training the next)."""
        for p in self.units[unit_idx].parameters():
            p.requires_grad_(False)

    def unfreeze_unit(self, unit_idx: int):
        for p in self.units[unit_idx].parameters():
            p.requires_grad_(True)

    def load_unit_from_checkpoint(self, unit_idx: int, ckpt_path: str,
                                  device: str = "cpu"):
        """Load blocks from a standalone GRKANpreyLM checkpoint into units[unit_idx]."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        # Extract only the block weights (key prefix: "blocks.")
        unit_state = {
            k.removeprefix("blocks."): v
            for k, v in state.items()
            if k.startswith("blocks.")
        }
        # Map flat block indices to unit-local indices
        self.units[unit_idx].blocks.load_state_dict(
            {k: v for k, v in unit_state.items()}, strict=False
        )

    def update_grid_all(self, x_sample: torch.Tensor):
        """No-op: rational functions have no grid to update."""
