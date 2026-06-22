"""
Side-by-side comparison: KanpreyLM vs original GuppyLM.

Compares parameter counts, inference speed, and generation quality on the same prompts.

Usage:
    # Compare two KAN checkpoints (e.g., grid=5 vs grid=2):
    uv run python compare.py \\
        --ckpt-a checkpoints/best_grid5.pt --label-a "KAN grid=5 (13M)" \\
        --ckpt-b checkpoints/best_grid2.pt --label-b "KAN grid=2 (10M)"
"""

import argparse
import time
import sys
import torch

sys.path.insert(0, ".")
from kanprey.dataset import load_tokenizer
from kanprey.inference import chat_completion, load_model
from kanprey.train import detect_device

PROMPTS = [
    "hello",
    "are you hungry",
    "what is money",
    "do you like your tank",
    "goodbye",
    "what color are you",
    "are you lonely",
    "what do you do all day",
]


def benchmark_speed(model, tokenizer, device, n_runs: int = 20) -> float:
    """Return average tokens/second over n_runs generations."""
    prompt = "hello"
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        chat_completion(prompt, model, tokenizer, device, max_new_tokens=32)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    return 32 / avg  # tokens per second


def run_comparison(
    ckpt_a: str,
    label_a: str,
    ckpt_b: str | None,
    label_b: str | None,
    tokenizer_path: str,
):
    device = detect_device()
    tokenizer = load_tokenizer(tokenizer_path)

    models = [(label_a, load_model(ckpt_a, device))]
    if ckpt_b:
        models.append((label_b, load_model(ckpt_b, device)))

    # Parameter summary
    print("\n" + "=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)
    for label, (model, cfg) in models:
        summary = model.param_summary()
        print(f"\n{label}")
        for k, v in summary.items():
            print(f"  {k:<14}: {v:>12,}")

    # Speed benchmark
    print("\n" + "=" * 70)
    print("INFERENCE SPEED (tokens/sec, 20 runs of 32 new tokens)")
    print("=" * 70)
    for label, (model, _) in models:
        tps = benchmark_speed(model, tokenizer, device)
        print(f"  {label:<35}: {tps:>6.1f} tok/s")

    # Generation comparison
    print("\n" + "=" * 70)
    print("GENERATION COMPARISON  (temperature=0.7, top_k=50)")
    print("=" * 70)
    for prompt in PROMPTS:
        print(f"\nPrompt: \"{prompt}\"")
        for label, (model, _) in models:
            resp = chat_completion(prompt, model, tokenizer, device)
            print(f"  [{label}] {resp}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="First checkpoint path")
    parser.add_argument("--label-a", default="Model A")
    parser.add_argument("--ckpt-b", default=None, help="Second checkpoint (optional)")
    parser.add_argument("--label-b", default="Model B")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    args = parser.parse_args()

    run_comparison(args.ckpt_a, args.label_a, args.ckpt_b, args.label_b, args.tokenizer)
