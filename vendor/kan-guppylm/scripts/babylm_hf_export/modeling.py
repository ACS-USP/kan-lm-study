"""HuggingFace causal-LM wrapper around a native KanpreyLM checkpoint.

The wrapper *holds* the real kanprey nn.Module (`self.lm`) and calls it directly —
the architecture is never re-implemented, so there is no risk of the eval model
diverging from the trained one. `self.lm(input_ids)` already returns logits of
shape (B, T, vocab); we package them into a `CausalLMOutput` so the BabyLM
evaluation pipeline (which reads `output["logits"]`) works unchanged.

Requires `kanprey` to be importable (set PYTHONPATH to the kan-guppylm repo when
running the pipeline with trust_remote_code).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutput

from .model_configuration import KanpreyConfig

from kanprey.config import ModelConfig
from kanprey.model import (
    MLPTransformer,
    SwiGLUTransformer,
    GRKANpreyLM,
    BasisKANpreyLM,
    KANpreyLM,
    KATpreyLM,
    MLPEdgepreyLM,
)

_BUILDERS = {
    "mlp": MLPTransformer,
    "swiglu": SwiGLUTransformer,
    "grkan": GRKANpreyLM,
    "basis": BasisKANpreyLM,
    "kan": KANpreyLM,
    "kat": KATpreyLM,
    "mlpedge": MLPEdgepreyLM,
}


class KanpreyModel(PreTrainedModel):
    """Base trunk: returns last_hidden_state (post-ln_f, pre-LM-head). Used by
    the GLUE fine-tuning pipeline (AutoModel + its own ClassifierHead). Shares
    the same submodule layout as all kanprey LM classes (tok_emb/pos_emb/drop/
    blocks/ln_f)."""
    config_class = KanpreyConfig
    base_model_prefix = "lm"
    supports_gradient_checkpointing = False

    def __init__(self, config: KanpreyConfig, **kwargs):
        super().__init__(config, **kwargs)
        kcfg = ModelConfig(**config.kanprey_cfg)
        self.lm = _BUILDERS[config.kanprey_model_type](kcfg)
        self.max_seq_len = kcfg.max_seq_len

    def get_input_embeddings(self):
        return self.lm.tok_emb

    def set_input_embeddings(self, value):
        self.lm.tok_emb = value

    def _trunk(self, idx: torch.Tensor) -> torch.Tensor:
        m = self.lm
        T = idx.size(1)
        if T <= self.max_seq_len:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = m.drop(m.tok_emb(idx) + m.pos_emb(pos))
            for blk in m.blocks:
                x = blk(x)
            return m.ln_f(x)
        parts = []
        for s in range(0, T, self.max_seq_len):
            chunk = idx[:, s : s + self.max_seq_len]
            t = chunk.size(1)
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = m.drop(m.tok_emb(chunk) + m.pos_emb(pos))
            for blk in m.blocks:
                x = blk(x)
            parts.append(m.ln_f(x))
        return torch.cat(parts, dim=1)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        h = self._trunk(input_ids)
        if return_dict is False:
            return (h,)
        return BaseModelOutput(last_hidden_state=h)


class KanpreyForCausalLM(PreTrainedModel):
    config_class = KanpreyConfig
    base_model_prefix = "lm"
    supports_gradient_checkpointing = False
    _supports_sdpa = False
    _supports_flash_attn_2 = False

    def __init__(self, config: KanpreyConfig, **kwargs):
        super().__init__(config, **kwargs)
        kcfg = ModelConfig(**config.kanprey_cfg)
        if config.kanprey_model_type not in _BUILDERS:
            raise ValueError(
                f"Unsupported kanprey_model_type={config.kanprey_model_type!r}; "
                f"expected one of {sorted(_BUILDERS)}"
            )
        self.lm = _BUILDERS[config.kanprey_model_type](kcfg)
        self.max_seq_len = kcfg.max_seq_len

    # ── HF embedding plumbing (weight tying matches kanprey: head.weight is tok_emb.weight) ──
    def get_input_embeddings(self):
        return self.lm.tok_emb

    def set_input_embeddings(self, value):
        self.lm.tok_emb = value

    def get_output_embeddings(self):
        return self.lm.head

    def set_output_embeddings(self, value):
        self.lm.head = value

    def can_generate(self) -> bool:
        return True

    def _run(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the fixed-context kanprey model, returning T-length logits.

        kanprey uses learned positional embeddings capped at max_seq_len. For
        sequences longer than the context window we score in non-overlapping
        windows and concatenate, which keeps the output length aligned with the
        input (BLiMP/EWoK fast items are short, so this path is rarely hit)."""
        T = input_ids.size(1)
        if T <= self.max_seq_len:
            return self.lm(input_ids)
        parts = [
            self.lm(input_ids[:, start : start + self.max_seq_len])
            for start in range(0, T, self.max_seq_len)
        ]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,  # accepted, intentionally unused
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        # attention_mask is ignored on purpose: the eval pipeline right-pads and the
        # model applies its own causal mask, so padded positions never affect the
        # logits at real positions.
        logits = self._run(input_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return CausalLMOutput(loss=loss, logits=logits)
