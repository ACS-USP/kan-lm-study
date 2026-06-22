"""
Head-to-head comparison: original GuppyLM vs KanpreyLM vs KATpreyLM.

Usage:
    # 2-way (original vs KAN-FFN):
    uv run --with . python compare_with_original.py \
        --kan-ckpt checkpoints/best.pt --kan-tok tokenizer.json \
        --orig-dir ../guppylm-original

    # 3-way (add KAT):
    uv run --with . python compare_with_original.py \
        --kan-ckpt checkpoints/best.pt --kan-tok tokenizer.json \
        --kat-ckpt checkpoints/kat/best.pt \
        --orig-dir ../guppylm-original
"""

import argparse
import sys
import time
import textwrap
import torch

# ── eval suite ────────────────────────────────────────────────────────────────

PROMPTS = [
    "hello",
    "how are you",
    "are you hungry",
    "what do you eat",
    "what is your name",
    "do you like your tank",
    "what is the temperature",
    "are you lonely",
    "what do you do all day",
    "what is the internet",
    "do you have friends",
    "are you scared",
    "what is money",
    "goodbye",
    "what color are you",
    "can you talk",
]

EVAL_KEYWORDS = {
    "hello":                  ["hi", "hello", "water", "swim", "tank", "hey"],
    "how are you":            ["ok", "good", "fine", "swim", "water", "feel"],
    "are you hungry":         ["yes", "food", "hungry", "eat", "flakes"],
    "what do you eat":        ["flakes", "food", "eat", "pellet", "hungry"],
    "what is your name":      ["guppy", "fish", "name", "i am", "i'm"],
    "do you like your tank":  ["tank", "like", "home", "water", "yes", "no"],
    "what is the temperature":["water", "cold", "warm", "temperature", "degrees"],
    "are you lonely":         ["lonely", "alone", "tank", "friend", "fish"],
    "what do you do all day": ["swim", "water", "tank", "float", "look"],
    "what is the internet":   ["don't know", "wet", "water", "what is", "know"],
    "do you have friends":    ["fish", "friend", "alone", "tank", "no", "yes"],
    "are you scared":         ["scared", "no", "fine", "safe", "tank", "water"],
    "what is money":          ["don't know", "wet", "what is", "know", "water"],
    "goodbye":                ["bye", "goodbye", "water", "swim", "ok"],
    "what color are you":     ["color", "blue", "orange", "silver", "shiny", "scale"],
    "can you talk":           ["yes", "no", "talk", "water", "bubble", "speak"],
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_original_guppy(orig_dir: str, device: torch.device):
    sys.path.insert(0, orig_dir)
    from tokenizers import Tokenizer as _Tok
    import importlib.util, os, json

    spec = importlib.util.spec_from_file_location("config", os.path.join(orig_dir, "config.py"))
    config_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(config_mod)
    GuppyConfig = config_mod.GuppyConfig

    spec2 = importlib.util.spec_from_file_location("model_orig", os.path.join(orig_dir, "model.py"))
    model_mod = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(model_mod)
    GuppyLM = model_mod.GuppyLM

    with open(os.path.join(orig_dir, "config.json")) as f:
        cfg_data = json.load(f)

    cfg = GuppyConfig(
        vocab_size=cfg_data.get("vocab_size", 4096),
        max_seq_len=cfg_data.get("max_position_embeddings", 128),
        d_model=cfg_data.get("hidden_size", 384),
        n_layers=cfg_data.get("num_hidden_layers", 6),
        n_heads=cfg_data.get("num_attention_heads", 6),
        ffn_hidden=cfg_data.get("intermediate_size", 768),
        dropout=0.0,
    )

    state = torch.load(os.path.join(orig_dir, "pytorch_model.bin"),
                       map_location=device, weights_only=False)
    model = KanpreyLM(cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = _Tok.from_file(os.path.join(orig_dir, "tokenizer.json"))
    n = sum(p.numel() for p in model.parameters())
    return model, tokenizer, cfg, n


def load_kan_variant(ckpt_path: str, tok_path: str, device: torch.device):
    """Load either KANpreyLM or KATpreyLM from checkpoint (type is stored in ckpt)."""
    sys.path.insert(0, ".")
    from kanprey.config import ModelConfig
    from kanprey.model import KANpreyLM, KATpreyLM, MLPEdgepreyLM
    from kanprey.dataset import load_tokenizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", ModelConfig())
    model_type = ckpt.get("model_type", "kan")

    if model_type == "kat":
        model = KATpreyLM(model_cfg).to(device)
    elif model_type == "mlpedge":
        model = MLPEdgepreyLM(model_cfg).to(device)
    else:
        model = KANpreyLM(model_cfg).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(tok_path)
    n = sum(p.numel() for p in model.parameters())
    return model, tokenizer, model_cfg, model_type, n


# ── inference helpers ─────────────────────────────────────────────────────────

def respond_original(prompt, model, tokenizer, device, cfg,
                     temperature=0.7, top_k=50, max_tokens=64):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(text).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    out, _ = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    elapsed = time.perf_counter() - t0
    new_ids = out[0, len(ids):].tolist()
    resp = tokenizer.decode(new_ids)
    if "<|im_end|>" in resp:
        resp = resp[:resp.index("<|im_end|>")]
    if "<|im_start|>" in resp:
        resp = resp[:resp.index("<|im_start|>")]
    return resp.strip(), elapsed


def respond_kan(prompt, model, tokenizer, device,
                temperature=0.7, top_k=50, max_tokens=64):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(text).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    elapsed = time.perf_counter() - t0
    new_ids = out[0, len(ids):].tolist()
    resp = tokenizer.decode(new_ids)
    if "<|im_end|>" in resp:
        resp = resp[:resp.index("<|im_end|>")]
    return resp.strip(), elapsed


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kan-ckpt",    default="checkpoints/best.pt")
    parser.add_argument("--kan-tok",     default="tokenizer.json")
    parser.add_argument("--kat-ckpt",    default=None,
                        help="Path to KAT checkpoint (omit for 2-way comparison)")
    parser.add_argument("--kat-tok",     default="tokenizer.json")
    parser.add_argument("--orig-dir",    default="../guppylm-original")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--max-tokens",  type=int,   default=64)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading original GuppyLM…")
    orig_model, orig_tok, orig_cfg, orig_n = load_original_guppy(args.orig_dir, device)

    print("Loading KAN/KAT preyLM (primary checkpoint)…")
    kan_model, kan_tok, kan_cfg, kan_type, kan_n = load_kan_variant(
        args.kan_ckpt, args.kan_tok, device)

    kat_model = kat_tok = kat_cfg = kat_n = None
    if args.kat_ckpt:
        print("Loading KATpreyLM (secondary checkpoint)…")
        kat_model, kat_tok, kat_cfg, _, kat_n = load_kan_variant(
            args.kat_ckpt, args.kat_tok, device)

    three_way = kat_model is not None

    # ── labels ──────────────────────────────────────────────────────────────
    kan_label = "KATpreyLM" if kan_type == "kat" else "KanpreyLM"
    kan_desc  = (f"grid_ffn={kan_cfg.kan_grid_size} kat={kan_cfg.kat_grid_size}"
                 if kan_type == "kat" else f"grid={kan_cfg.kan_grid_size}")

    # ── table layout ────────────────────────────────────────────────────────
    col = 36 if three_way else 53
    sep = "  "

    header_orig = f"ORIGINAL GuppyLM"
    header_kan  = kan_label
    header_kat  = "KATpreyLM" if three_way else ""

    W = col * (3 if three_way else 2) + len(sep) * (2 if three_way else 1) + 4

    print("\n" + "=" * W)
    if three_way:
        print(f"  {header_orig:<{col}}{sep}{header_kan:<{col}}{sep}{header_kat}")
        print(f"  {f'{orig_n/1e6:.2f}M | FFN: ReLU':<{col}}"
              f"{sep}{f'{kan_n/1e6:.2f}M | {kan_desc}':<{col}}"
              f"{sep}{f'{kat_n/1e6:.2f}M | grid_ffn={kat_cfg.kan_grid_size} kat={kat_cfg.kat_grid_size}'}")
    else:
        print(f"  {header_orig:<{col}}{sep}{header_kan}")
        print(f"  {f'{orig_n/1e6:.2f}M | FFN: 384→768→384 ReLU':<{col}}"
              f"{sep}{f'{kan_n/1e6:.2f}M | {kan_desc}'}")
    print("=" * W)

    orig_pass = kan_pass = kat_pass = 0
    orig_times = []; kan_times = []; kat_times = []

    for prompt in PROMPTS:
        orig_resp, orig_t = respond_original(
            prompt, orig_model, orig_tok, device, orig_cfg,
            args.temperature, args.top_k, args.max_tokens)
        kan_resp, kan_t = respond_kan(
            prompt, kan_model, kan_tok, device,
            args.temperature, args.top_k, args.max_tokens)
        orig_times.append(orig_t); kan_times.append(kan_t)

        kat_resp = kat_t = None
        if three_way:
            kat_resp, kat_t = respond_kan(
                prompt, kat_model, kat_tok, device,
                args.temperature, args.top_k, args.max_tokens)
            kat_times.append(kat_t)

        kws = EVAL_KEYWORDS.get(prompt, [])
        o_hit = any(k in orig_resp.lower() for k in kws)
        k_hit = any(k in kan_resp.lower() for k in kws)
        t_hit = any(k in kat_resp.lower() for k in kws) if kat_resp else False
        orig_pass += o_hit; kan_pass += k_hit; kat_pass += t_hit

        print(f"\n  Prompt: \"{prompt}\"")
        orig_lines = textwrap.wrap(orig_resp or "(empty)", col - 4)
        kan_lines  = textwrap.wrap(kan_resp  or "(empty)", col - 4)
        kat_lines  = textwrap.wrap(kat_resp  or "(empty)", col - 4) if kat_resp else []
        n_lines = max(len(orig_lines), len(kan_lines), len(kat_lines), 1)

        for i in range(n_lines):
            o = orig_lines[i] if i < len(orig_lines) else ""
            k = kan_lines[i]  if i < len(kan_lines)  else ""
            t = kat_lines[i]  if i < len(kat_lines)  else ""
            o_tag = ("✓" if o_hit else "✗") if i == 0 else " "
            k_tag = ("✓" if k_hit else "✗") if i == 0 else " "
            t_tag = ("✓" if t_hit else "✗") if i == 0 else " "
            prefix_o = f"  [{o_tag}] "
            prefix_k = f"[{k_tag}] "
            prefix_t = f"[{t_tag}] " if three_way else ""
            if three_way:
                print(f"{prefix_o}{o:<{col - 2}}{sep}{prefix_k}{k:<{col - 4}}{sep}{prefix_t}{t}")
            else:
                print(f"{prefix_o}{o:<{col - 2}}{sep}{prefix_k}{k}")

    # ── summary ─────────────────────────────────────────────────────────────
    avg_orig = sum(orig_times) / len(orig_times)
    avg_kan  = sum(kan_times)  / len(kan_times)
    avg_kat  = sum(kat_times)  / len(kat_times) if kat_times else None

    print("\n" + "=" * W)
    print("SUMMARY")
    print("=" * W)
    label_w = 18
    rows = [
        ("Parameters",  f"{orig_n/1e6:.2f}M",         f"{kan_n/1e6:.2f}M",         f"{kat_n/1e6:.2f}M" if kat_n else "—"),
        ("Vocab size",  str(orig_cfg.vocab_size),       str(kan_cfg.vocab_size),      str(kat_cfg.vocab_size) if kat_cfg else "—"),
        ("FFN type",    "Linear→ReLU→Linear",           f"KANLinear (g={kan_cfg.kan_grid_size})", f"KANLinear (g={kat_cfg.kan_grid_size})" if kat_cfg else "—"),
        ("Attn type",   "dot-product",                  "dot-product" if kan_type=="kan" else "KAN kernel", "KAN kernel"),
        ("Eval score",  f"{orig_pass}/{len(PROMPTS)}",  f"{kan_pass}/{len(PROMPTS)}",  f"{kat_pass}/{len(PROMPTS)}" if three_way else "—"),
        ("Avg latency", f"{avg_orig*1000:.0f} ms",      f"{avg_kan*1000:.0f} ms",     f"{avg_kat*1000:.0f} ms" if avg_kat else "—"),
    ]
    for label, o_val, k_val, t_val in rows:
        if three_way:
            print(f"  {label:<{label_w}}: {o_val:<{col - label_w}}{sep}{k_val:<{col - 4}}{sep}{t_val}")
        else:
            print(f"  {label:<{label_w}}: {o_val:<{col - label_w}}{sep}{k_val}")
    print("=" * W)


if __name__ == "__main__":
    main()
