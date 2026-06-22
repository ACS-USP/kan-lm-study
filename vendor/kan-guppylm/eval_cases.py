"""
16 evaluation prompts mirroring GuppyLM's eval_cases.py for apples-to-apples comparison.

Run:  uv run python eval_cases.py --checkpoint checkpoints/best.pt
"""

import argparse
import json
import sys
import time
import torch

sys.path.insert(0, ".")
from kanprey.dataset import load_tokenizer
from kanprey.inference import chat_completion, load_model
from kanprey.train import detect_device

CASES = [
    # (prompt, expected_keywords_any, notes)
    ("hello", ["hi", "hello", "water", "swim", "tank", "hey"], "greeting"),
    ("how are you", ["ok", "good", "fine", "swim", "water", "feel"], "wellbeing"),
    ("are you hungry", ["yes", "food", "hungry", "eat", "flakes"], "hunger"),
    ("what do you eat", ["flakes", "food", "eat", "pellet", "hungry"], "food"),
    ("what is your name", ["guppy", "fish", "name", "i am", "i'm"], "identity"),
    ("do you like your tank", ["tank", "like", "home", "water", "yes", "no"], "tank opinion"),
    ("what is the temperature", ["water", "cold", "warm", "temperature", "degrees"], "temperature"),
    ("are you lonely", ["lonely", "alone", "tank", "friend", "fish"], "loneliness"),
    ("what do you do all day", ["swim", "water", "tank", "float", "look"], "daily life"),
    ("can you talk", ["yes", "no", "talk", "water", "bubble", "speak"], "capability"),
    ("what is the internet", ["don't know", "wet", "water", "what is", "know"], "confusion"),
    ("do you have friends", ["fish", "friend", "alone", "tank", "no", "yes"], "social"),
    ("are you scared", ["scared", "no", "fine", "safe", "tank", "water"], "fear"),
    ("what is money", ["don't know", "wet", "what is", "know", "water"], "confusion"),
    ("goodbye", ["bye", "goodbye", "water", "swim", "ok"], "farewell"),
    ("what color are you", ["color", "blue", "orange", "silver", "shiny", "scale"], "appearance"),
]


def run_eval(
    checkpoint: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_k: int = 50,
    max_tokens: int = 64,
    quiet: bool = False,
):
    device = detect_device()
    tokenizer = load_tokenizer(tokenizer_path)
    model, _ = load_model(checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    model_type = model.__class__.__name__

    if not quiet:
        print(f"\nEval  |  {model_type}  |  {n_params:,} params  |  checkpoint: {checkpoint}\n")
        print(f"{'Prompt':<30} {'Pass':>4}  {'Latency ms':>10}  Response")
        print("-" * 100)

    passed = 0
    latencies_s = []
    response_lengths = []
    case_rows = []
    for prompt, keywords, note in CASES:
        t0 = time.perf_counter()
        response = chat_completion(
            prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed_s = time.perf_counter() - t0
        hit = any(kw.lower() in response.lower() for kw in keywords)
        passed += hit
        latencies_s.append(elapsed_s)
        response_lengths.append(len(tokenizer.encode(response).ids))
        tag = "PASS" if hit else "FAIL"
        resp_short = response[:60].replace("\n", " ")
        case_rows.append(
            {
                "prompt": prompt,
                "note": note,
                "passed": hit,
                "latency_s": elapsed_s,
                "response": response,
            }
        )
        if not quiet:
            print(f"{prompt:<30} {tag:>4}  {elapsed_s*1000:10.1f}  {resp_short}")

    result = {
        "checkpoint": checkpoint,
        "model_type": model_type,
        "params": n_params,
        "score": passed,
        "total_cases": len(CASES),
        "score_pct": 100 * passed / len(CASES),
        "mean_latency_ms": 1000 * sum(latencies_s) / len(latencies_s),
        "median_latency_ms": 1000 * sorted(latencies_s)[len(latencies_s) // 2],
        "mean_response_tokens": sum(response_lengths) / len(response_lengths),
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "device": str(device),
        "cases": case_rows,
    }

    if not quiet:
        print(
            f"\nScore: {passed}/{len(CASES)}  ({result['score_pct']:.0f}%)"
            f"  |  mean latency: {result['mean_latency_ms']:.1f} ms"
        )
    return result


def run_suite(
    runs: list[tuple[str, str]],
    tokenizer_path: str,
    temperature: float = 0.7,
    top_k: int = 50,
    max_tokens: int = 64,
    json_out: bool = False,
):
    results = []
    for label, checkpoint in runs:
        result = run_eval(
            checkpoint,
            tokenizer_path,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            quiet=json_out,
        )
        result["label"] = label
        results.append(result)

    if json_out:
        print(json.dumps(results, indent=2))
        return results

    print("\n" + "=" * 104)
    print(f"{'Label':<18} {'Score':>7}  {'Mean latency':>14}  {'Params':>12}  {'Model'}")
    print("-" * 104)
    for r in results:
        print(
            f"{r['label']:<18} {r['score']:>2}/{r['total_cases']:<4}"
            f"  {r['mean_latency_ms']:>11.1f} ms  {r['params']:>12,}  {r['model_type']}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run", action="append", default=[], help="LABEL=checkpoint; may repeat")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runs = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run expects LABEL=checkpoint, got: {item}")
        label, checkpoint = item.split("=", 1)
        runs.append((label, checkpoint))

    if runs:
        run_suite(
            runs,
            args.tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            json_out=args.json,
        )
    elif args.checkpoint:
        result = run_eval(
            args.checkpoint,
            args.tokenizer,
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            quiet=args.json,
        )
        if args.json:
            print(json.dumps(result, indent=2))
    else:
        raise SystemExit("Pass --checkpoint or one/more --run LABEL=checkpoint")
