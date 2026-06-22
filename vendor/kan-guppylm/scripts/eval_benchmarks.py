"""
Benchmark evaluation for KanpreyLM models.

Evaluates a checkpoint on four standard tasks:

  1. WikiText-103 perplexity  — language modelling baseline
  2. LAMBADA accuracy         — long-range next-word prediction
  3. ARC-Easy (0-shot)        — multiple-choice science questions
  4. HellaSwag (0-shot)       — multiple-choice sentence completion

All tasks use the GPT-2 BPE tokenizer (tiktoken, 50 257 vocab) to stay
compatible with the training setup.

Scoring method for MCQ (ARC, HellaSwag):
  For each candidate completion, compute the mean per-token log-probability
  given the context (length-normalised to avoid bias toward short options).
  The candidate with the highest score is the model's answer.

Baseline reference numbers (from published papers):
  Model           | WT-103 ppl | LAMBADA | ARC-E  | HellaSwag
  ------------------------------------------------------------
  GPT-2 117M      |   37.50    | 45.99%  |   —    |   31.1%
  OPT-125M        |     —      |   —     | 22.87% |   31.47%
  Random (4-way)  |     —      |   —     | 25.00% |   25.00%

Usage:
    # Evaluate best checkpoint
    python scripts/eval_benchmarks.py --checkpoint checkpoints/unit0/best.pt \\
        --model unit0_grkan

    # Quick debug run (100 examples per task)
    python scripts/eval_benchmarks.py --checkpoint checkpoints/unit0/best.pt \\
        --model unit0_grkan --max-samples 100

    # Evaluate MoE model
    python scripts/eval_benchmarks.py --checkpoint checkpoints/moe/best.pt \\
        --model moe_grkan
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kanprey.config import ModelConfig
from kanprey.model import (
    GRKANpreyLM, MLPTransformer, MoEGRKANpreyLM, ModuleChainLM,
    MLPEdgepreyLM, LoopGRKANpreyLM,
)
from kanprey.dataset_wikitext import WikitextDataset

# ── Reference baselines ───────────────────────────────────────────────────────

BASELINES = {
    "GPT-2 117M":  {"wt103_ppl": 37.50, "lambada_acc": 45.99, "arc_easy": None,  "hellaswag": 31.1},
    "OPT-125M":    {"wt103_ppl": None,   "lambada_acc": None,  "arc_easy": 22.87, "hellaswag": 31.47},
    "Random (4-way)": {"wt103_ppl": None, "lambada_acc": None, "arc_easy": 25.0,  "hellaswag": 25.0},
}

# ── Model configs (must match train_scale.py) ─────────────────────────────────

MODULE_UNIT = dict(
    d_model=384, n_heads=6, n_layers=6, max_seq_len=128, dropout=0.0,
    unit_n_layers=3, n_moe_experts=8, moe_top_k=2, load_balance_coeff=0.01,
)
LOCAL_9M = dict(
    d_model=384, n_heads=6, n_layers=6, max_seq_len=128, dropout=0.0,
)


def load_model(args, vocab_size: int, device: torch.device):
    if args.model == "unit0_grkan":
        cfg = ModelConfig(vocab_size=vocab_size,
                          **{**MODULE_UNIT, "n_layers": MODULE_UNIT["unit_n_layers"]})
        model = GRKANpreyLM(cfg)
    elif args.model == "grkan":
        cfg = ModelConfig(vocab_size=vocab_size, **LOCAL_9M)
        model = GRKANpreyLM(cfg)
    elif args.model == "moe_grkan":
        cfg = ModelConfig(vocab_size=vocab_size, **MODULE_UNIT)
        model = MoEGRKANpreyLM(cfg)
    elif args.model == "unit_grkan":
        cfg = ModelConfig(vocab_size=vocab_size, **MODULE_UNIT)
        n_units = cfg.n_layers // cfg.unit_n_layers
        model = ModuleChainLM(cfg, n_units=n_units)
    elif args.model == "mlp_local":
        cfg = ModelConfig(vocab_size=vocab_size, **LOCAL_9M)
        model = MLPTransformer(cfg)
    elif args.model == "loop_grkan":
        cfg = ModelConfig(vocab_size=vocab_size,
                          **{**MODULE_UNIT, "n_layers": MODULE_UNIT["unit_n_layers"]})
        model = LoopGRKANpreyLM(cfg)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


# ── Task 1: WikiText-103 perplexity ──────────────────────────────────────────

def eval_wikitext103(model, cfg, device, max_samples: int | None = None) -> float:
    """Stride-based perplexity on the WikiText-103 validation set."""
    val_ds = WikitextDataset("validation", max_seq_len=cfg.max_seq_len)
    n = min(len(val_ds), max_samples) if max_samples else len(val_ds)

    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for i in tqdm(range(n), desc="WikiText-103", leave=False):
            x, y = val_ds[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_n += y.numel()

    return math.exp(total_loss / total_n)


# ── Task 2: LAMBADA accuracy ──────────────────────────────────────────────────

def eval_lambada(model, cfg, enc, device, max_samples: int | None = None) -> float:
    """
    LAMBADA: predict the last word of a passage given all preceding context.
    Accuracy = fraction of examples where the top-1 predicted token matches
    the first token of the target last word (standard evaluation protocol).
    """
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    n = min(len(ds), max_samples) if max_samples else len(ds)

    correct, total = 0, 0
    with torch.no_grad():
        for i in tqdm(range(n), desc="LAMBADA", leave=False):
            text = ds[i]["text"]
            # Split into context (all but last word) and target (last word)
            parts = text.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            context, target = parts

            ctx_ids = enc.encode(context)
            tgt_ids = enc.encode(" " + target)  # space prefix for BPE consistency
            if not tgt_ids:
                continue

            # Truncate context to fit model's max_seq_len
            max_ctx = cfg.max_seq_len - 1
            ctx_ids = ctx_ids[-max_ctx:]

            input_ids = torch.tensor([ctx_ids], device=device)
            logits = model(input_ids)          # (1, T, vocab)
            next_token_logits = logits[0, -1]  # (vocab,)
            predicted = next_token_logits.argmax().item()

            if predicted == tgt_ids[0]:
                correct += 1
            total += 1

    return 100.0 * correct / total if total > 0 else 0.0


# ── Task 3: ARC-Easy (0-shot) ─────────────────────────────────────────────────

def _score_completion(model, enc, context_ids: list[int],
                      completion: str, device, max_seq_len: int) -> float:
    """
    Mean per-token log-probability of `completion` given `context_ids`.
    Length-normalised to avoid favouring short answers.
    """
    comp_ids = enc.encode(" " + completion)
    if not comp_ids:
        return float("-inf")

    full = context_ids[-max_seq_len + len(comp_ids):] + comp_ids
    input_ids = torch.tensor([full[:-1]], device=device)
    target_ids = torch.tensor([full[1:]], device=device)

    with torch.no_grad():
        logits = model(input_ids)  # (1, T, vocab)

    # Only score the completion tokens (last len(comp_ids) positions)
    n_ctx = len(full) - 1
    n_comp = len(comp_ids)
    comp_start = n_ctx - n_comp

    comp_logits = logits[0, comp_start:]        # (n_comp, vocab)
    comp_targets = target_ids[0, comp_start:]   # (n_comp,)

    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_log_probs = log_probs.gather(1, comp_targets.unsqueeze(1)).squeeze(1)
    return token_log_probs.mean().item()


def eval_arc_easy(model, cfg, enc, device, max_samples: int | None = None) -> float:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    n = min(len(ds), max_samples) if max_samples else len(ds)

    correct, total = 0, 0
    for i in tqdm(range(n), desc="ARC-Easy", leave=False):
        item = ds[i]
        question = item["question"]
        choices = item["choices"]
        answer_key = item["answerKey"]

        labels = choices["label"]
        texts = choices["text"]
        context = f"Question: {question}\nAnswer:"
        ctx_ids = enc.encode(context)

        scores = [
            _score_completion(model, enc, ctx_ids, t, device, cfg.max_seq_len)
            for t in texts
        ]
        pred_label = labels[scores.index(max(scores))]

        if pred_label == answer_key:
            correct += 1
        total += 1

    return 100.0 * correct / total if total > 0 else 0.0


# ── Task 4: HellaSwag (0-shot) ────────────────────────────────────────────────

def eval_hellaswag(model, cfg, enc, device, max_samples: int | None = None) -> float:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    n = min(len(ds), max_samples) if max_samples else len(ds)

    correct, total = 0, 0
    for i in tqdm(range(n), desc="HellaSwag", leave=False):
        item = ds[i]
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        ctx_ids = enc.encode(ctx)
        scores = [
            _score_completion(model, enc, ctx_ids, e, device, cfg.max_seq_len)
            for e in endings
        ]
        pred = scores.index(max(scores))

        if pred == label:
            correct += 1
        total += 1

    return 100.0 * correct / total if total > 0 else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark KanpreyLM checkpoints")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt checkpoint file")
    parser.add_argument("--model",
                        choices=["unit0_grkan", "grkan", "moe_grkan",
                                 "unit_grkan", "mlp_local", "loop_grkan"],
                        required=True)
    parser.add_argument("--tasks", nargs="+",
                        choices=["wt103", "lambada", "arc", "hellaswag", "all"],
                        default=["all"],
                        help="Which benchmarks to run (default: all)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap examples per task (for quick testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    run_all = "all" in args.tasks
    run_wt103    = run_all or "wt103"    in args.tasks
    run_lambada  = run_all or "lambada"  in args.tasks
    run_arc      = run_all or "arc"      in args.tasks
    run_hellaswag = run_all or "hellaswag" in args.tasks

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab  # 50,257

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model(args, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | {n_params:,} params ({n_params/1e6:.1f}M)")
    print()

    results: dict[str, float | None] = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "n_params": n_params,
    }
    t0 = time.time()

    if run_wt103:
        print("── WikiText-103 perplexity ──────────────────────────────────────")
        ppl = eval_wikitext103(model, cfg, device, args.max_samples)
        results["wt103_ppl"] = round(ppl, 2)
        print(f"  WikiText-103 perplexity: {ppl:.2f}")
        print(f"  (GPT-2 117M baseline: 37.50)")
        print()

    if run_lambada:
        print("── LAMBADA accuracy ─────────────────────────────────────────────")
        acc = eval_lambada(model, cfg, enc, device, args.max_samples)
        results["lambada_acc"] = round(acc, 2)
        print(f"  LAMBADA accuracy: {acc:.2f}%")
        print(f"  (GPT-2 117M baseline: 45.99%)")
        print()

    if run_arc:
        print("── ARC-Easy (0-shot) ────────────────────────────────────────────")
        arc = eval_arc_easy(model, cfg, enc, device, args.max_samples)
        results["arc_easy"] = round(arc, 2)
        print(f"  ARC-Easy accuracy: {arc:.2f}%")
        print(f"  (OPT-125M baseline: 22.87% | Random: 25.00%)")
        print()

    if run_hellaswag:
        print("── HellaSwag (0-shot) ───────────────────────────────────────────")
        hs = eval_hellaswag(model, cfg, enc, device, args.max_samples)
        results["hellaswag"] = round(hs, 2)
        print(f"  HellaSwag accuracy: {hs:.2f}%")
        print(f"  (GPT-2 117M baseline: 31.1% | OPT-125M: 31.47%)")
        print()

    elapsed = time.time() - t0
    results["eval_time_s"] = round(elapsed, 1)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"{'Benchmark':<20} {'This model':>12} {'GPT-2 117M':>12} {'OPT-125M':>12}")
    print("-" * 64)
    for key, label, b_gpt2, b_opt in [
        ("wt103_ppl",  "WikiText-103 ppl ↓", "37.50", "—"),
        ("lambada_acc", "LAMBADA acc ↑",      "45.99%", "—"),
        ("arc_easy",   "ARC-Easy ↑",         "—",      "22.87%"),
        ("hellaswag",  "HellaSwag ↑",        "31.1%",  "31.47%"),
    ]:
        val = results.get(key)
        val_str = f"{val:.2f}" + ("%" if key != "wt103_ppl" else "") if val is not None else "—"
        print(f"{label:<20} {val_str:>12} {b_gpt2:>12} {b_opt:>12}")
    print("=" * 64)
    print(f"Eval time: {elapsed/60:.1f} min")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
