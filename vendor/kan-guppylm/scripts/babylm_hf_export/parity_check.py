"""Parity smoke test: HF-wrapped model must produce IDENTICAL logits to the
native KanpreyLM checkpoint. Also exercises tokenizer loading + offset mapping
exactly as the BabyLM pipeline does.

Run in the eval venv with PYTHONPATH=<kan-guppylm>:
  PYTHONPATH=/path/to/kan-guppylm .venv/bin/python parity_check.py <ckpt> <out_dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from convert_to_hf import convert  # local module (run from this dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer", default="tokenizer_babylm.json")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    print(f"[1] Converting {args.checkpoint} -> {args.out_dir}")
    info = convert(args.checkpoint, args.out_dir, args.tokenizer)
    print(f"    {info}")

    device = torch.device("cpu")

    # Native model
    print("[2] Loading native KanpreyLM via kanprey.inference.load_model")
    from kanprey.inference import load_model as load_native
    native, native_cfg = load_native(args.checkpoint, device)
    native.eval()

    # HF model
    print("[3] Loading HF model via AutoModelForCausalLM(trust_remote_code=True)")
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    hf = AutoModelForCausalLM.from_pretrained(args.out_dir, trust_remote_code=True)
    hf.to(device).eval()

    n_native = sum(p.numel() for p in native.parameters())
    n_hf = sum(p.numel() for p in hf.parameters())
    print(f"    params: native={n_native:,}  hf={n_hf:,}  match={n_native == n_hf}")

    # Logit parity on identical random input
    print("[4] Logit parity check")
    torch.manual_seed(0)
    vocab = native_cfg.vocab_size
    T = min(64, native_cfg.max_seq_len)
    x = torch.randint(0, vocab, (2, T), device=device)
    with torch.no_grad():
        lo_native = native(x)
        out_hf = hf(input_ids=x)
        lo_hf = out_hf["logits"]
    assert lo_native.shape == lo_hf.shape, f"shape mismatch {lo_native.shape} vs {lo_hf.shape}"
    max_abs = (lo_native - lo_hf).abs().max().item()
    print(f"    shapes {tuple(lo_native.shape)}  max|Δlogits|={max_abs:.3e}  (tol={args.tol})")

    # Argmax agreement (next-token predictions identical)
    agree = (lo_native.argmax(-1) == lo_hf.argmax(-1)).float().mean().item()
    print(f"    argmax agreement={agree*100:.2f}%")

    # Tokenizer loading + offset mapping (as the pipeline does)
    print("[5] Tokenizer load + offset mapping (pipeline path)")
    tok = AutoTokenizer.from_pretrained(args.out_dir, trust_remote_code=True)
    enc = tok("The cat sat on the mat.", return_offsets_mapping=True)
    print(f"    is_fast={tok.is_fast}  pad_id={tok.pad_token_id}  "
          f"n_ids={len(enc['input_ids'])}  has_offsets={'offset_mapping' in enc}")
    assert tok.is_fast, "tokenizer must be fast (offset mapping required by pipeline)"
    assert "offset_mapping" in enc, "offset mapping missing"

    ok = (max_abs < args.tol) and (lo_native.shape == lo_hf.shape) and (n_native == n_hf)
    print()
    print("PARITY:", "PASS ✓" if ok else "FAIL ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
