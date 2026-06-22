"""Convert a native KanpreyLM checkpoint into a trust_remote_code HF repo
consumable by the BabyLM evaluation-pipeline-2025.

Produces, in <out_dir>:
  config.json              (auto_map -> modeling.KanpreyForCausalLM, full kanprey_cfg)
  modeling.py              (copied wrapper)
  model_configuration.py   (copied config class)
  pytorch_model.bin        (state dict, keys prefixed `lm.` to match the wrapper)
  tokenizer.json           (the BabyLM BPE tokenizer)
  tokenizer_config.json / special_tokens_map.json

Usage:
  python convert_to_hf.py <checkpoint.pt> <out_dir> [--tokenizer tokenizer_babylm.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import torch

TEMPLATE_DIR = Path(__file__).parent


def convert(ckpt_path: str, out_dir: str, tokenizer_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    model_cfg = ckpt["model_cfg"]
    model_type = ckpt.get("model_type", "mlp")
    cfg_dict = dataclasses.asdict(model_cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Custom code files
    for fname in ("modeling.py", "model_configuration.py",
                  "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(TEMPLATE_DIR / fname, out / fname)

    # config.json
    config = {
        "model_type": "kanprey",
        "architectures": ["KanpreyForCausalLM"],
        "auto_map": {
            "AutoConfig": "model_configuration.KanpreyConfig",
            "AutoModel": "modeling.KanpreyModel",
            "AutoModelForCausalLM": "modeling.KanpreyForCausalLM",
        },
        "kanprey_model_type": model_type,
        "kanprey_cfg": cfg_dict,
        "vocab_size": cfg_dict["vocab_size"],
        "hidden_size": cfg_dict["d_model"],
        "num_attention_heads": cfg_dict["n_heads"],
        "num_hidden_layers": cfg_dict["n_layers"],
        "max_position_embeddings": cfg_dict["max_seq_len"],
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # Weights: prefix every key with `lm.` to match KanpreyForCausalLM.lm
    prefixed = {f"lm.{k}": v for k, v in state.items()}
    torch.save(prefixed, out / "pytorch_model.bin")

    # Tokenizer
    shutil.copy(tokenizer_path, out / "tokenizer.json")

    return {
        "out_dir": str(out),
        "model_type": model_type,
        "vocab_size": cfg_dict["vocab_size"],
        "n_state_keys": len(state),
        "best_val_loss": ckpt.get("val_loss"),
        "step": ckpt.get("step"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer", default="tokenizer_babylm.json")
    args = ap.parse_args()
    info = convert(args.checkpoint, args.out_dir, args.tokenizer)
    print(json.dumps(info, indent=2))
