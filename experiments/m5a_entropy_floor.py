#!/usr/bin/env python
"""
M5a: quantify the GuppyLM benchmark's dynamic range. Computes unigram and
bigram (add-k smoothed) baseline cross-entropy on the *assistant-token* loss
the model optimizes (positions where y != -100), to contextualize the trained
models' CE ~= 0.277 (perplexity ~1.32).

Counts n-grams from the train split, evaluates on the test split.

Run with PYTHONPATH=<kan-guppylm>:
  PYTHONPATH=/.../kan-guppylm uv run python m5a_entropy_floor.py
"""
import sys, math
from collections import defaultdict, Counter
REPO = "/Users/felippealves/Documents/GitHub/kan-guppylm"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from kanprey.dataset import KanpreyDataset, load_tokenizer, PAD_ID

TOK = "tokenizer.json"
DATASET = "arman-bd/guppylm-60k-generic"
MAXLEN = 128
K = 0.1  # add-k smoothing


def main():
    tok = load_tokenizer(TOK)
    V = tok.get_vocab_size()
    print(f"vocab={V}  pad_id={PAD_ID}")

    train = KanpreyDataset("train", tok, MAXLEN, DATASET)
    test = KanpreyDataset("test", tok, MAXLEN, DATASET)

    # --- count uni/bigrams from train inputs ---
    uni = Counter()
    bi = defaultdict(Counter)
    for i in range(len(train)):
        x, _ = train[i]
        x = x.tolist()
        prev = None
        for t in x:
            if t == PAD_ID:
                break
            uni[t] += 1
            if prev is not None:
                bi[prev][t] += 1
            prev = t
    total_uni = sum(uni.values())
    print(f"train tokens counted: {total_uni:,}")

    # --- evaluate on test assistant-target positions (y != -100) ---
    nll_uni = nll_bi = 0.0
    n = 0
    logV = math.log(V)
    for i in range(len(test)):
        x, y = test[i]
        x, y = x.tolist(), y.tolist()
        for t in range(len(y)):
            tgt = y[t]
            if tgt == -100 or tgt == PAD_ID:
                continue
            # unigram P(tgt)
            p_uni = (uni.get(tgt, 0) + K) / (total_uni + K * V)
            nll_uni += -math.log(p_uni)
            # bigram P(tgt | x[t])
            ctx = x[t]
            row = bi.get(ctx)
            c_ctx = uni.get(ctx, 0)
            c_bg = row.get(tgt, 0) if row else 0
            p_bi = (c_bg + K) / (c_ctx + K * V)
            nll_bi += -math.log(p_bi)
            n += 1

    ce_uni, ce_bi = nll_uni / n, nll_bi / n
    print(f"\nassistant target tokens evaluated: {n:,}")
    print(f"{'baseline':<22}{'CE (nats)':>12}{'perplexity':>12}")
    print(f"{'uniform (log V)':<22}{logV:>12.3f}{V:>12.1f}")
    print(f"{'unigram':<22}{ce_uni:>12.3f}{math.exp(ce_uni):>12.2f}")
    print(f"{'bigram (add-0.1)':<22}{ce_bi:>12.3f}{math.exp(ce_bi):>12.2f}")
    print(f"{'trained MLP/KAN':<22}{0.277:>12.3f}{math.exp(0.277):>12.2f}")
    print(f"\nHeadroom from bigram->trained: {ce_bi - 0.277:.3f} nats "
          f"({math.exp(ce_bi):.2f} -> {math.exp(0.277):.2f} ppl)")


if __name__ == "__main__":
    main()
