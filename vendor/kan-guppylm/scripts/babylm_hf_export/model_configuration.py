"""HuggingFace config for KanpreyLM checkpoints (BabyLM eval pipeline wrapper).

Carries the *entire* kanprey ModelConfig as a dict (`kanprey_cfg`) plus the
model-family selector (`kanprey_model_type`). The modeling file reconstructs the
exact kanprey architecture via `ModelConfig(**kanprey_cfg)`, so no field can be
silently dropped. Standard HF fields (vocab_size, hidden_size, ...) are mirrored
for tooling that expects them.
"""
from __future__ import annotations

from transformers import PretrainedConfig


class KanpreyConfig(PretrainedConfig):
    model_type = "kanprey"

    def __init__(
        self,
        kanprey_model_type: str = "mlp",
        kanprey_cfg: dict | None = None,
        **kwargs,
    ):
        self.kanprey_model_type = kanprey_model_type
        self.kanprey_cfg = kanprey_cfg or {}
        # Mirror common fields for HF tooling (read-only convenience).
        self.vocab_size = self.kanprey_cfg.get("vocab_size", kwargs.pop("vocab_size", 8192))
        self.hidden_size = self.kanprey_cfg.get("d_model", 384)
        self.num_attention_heads = self.kanprey_cfg.get("n_heads", 6)
        self.num_hidden_layers = self.kanprey_cfg.get("n_layers", 6)
        self.max_position_embeddings = self.kanprey_cfg.get("max_seq_len", 128)
        super().__init__(**kwargs)
