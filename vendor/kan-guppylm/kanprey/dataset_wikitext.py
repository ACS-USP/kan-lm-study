"""
Wikitext-103 dataset loader for GPT-scale experiments.

Uses the GPT-2 BPE tokenizer (tiktoken) so we get a standard 50,257-token
vocabulary — consistent with GPT-2 baseline comparisons.

Memory design
-------------
The naive approach (enc.encode(full_text) → Python list) creates ~3GB of
Python int objects for 103M tokens before any tensor is allocated.  Instead
we encode article by article, convert each to a numpy int32 array immediately,
concatenate the small arrays, save to disk, and memory-map on load.

Peak RAM during first run: ~1.3GB (raw texts + growing numpy arrays).
Subsequent runs: ~0MB (mmap, backed by disk).

Usage:
    from kanprey.dataset_wikitext import get_wikitext_loaders
    train_loader, val_loader, vocab_size = get_wikitext_loaders(
        batch_size=32, max_seq_len=1024, num_workers=0
    )
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import tiktoken
    _TIKTOKEN = True
except ImportError:
    _TIKTOKEN = False

from datasets import load_dataset


_CACHE_DIR = Path.home() / ".cache" / "kanprey"


def _get_tokenizer():
    if not _TIKTOKEN:
        raise ImportError("tiktoken is required: uv add tiktoken")
    return tiktoken.get_encoding("gpt2")


def _build_token_cache(split: str, cache_file: Path) -> None:
    """Encode Wikitext-103 article-by-article to avoid a ~3GB Python-list spike."""
    enc = _get_tokenizer()
    print(f"Building token cache for split='{split}' → {cache_file}")

    raw = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)
    texts = [t for t in raw["text"] if t.strip()]
    del raw
    gc.collect()

    # Encode one article at a time; convert to numpy immediately so Python
    # int objects are freed before the next article is processed.
    arrays: list[np.ndarray] = []
    for text in texts:
        ids = enc.encode(text)
        arrays.append(np.array(ids, dtype=np.int32))
        del ids

    del texts
    gc.collect()

    # np.concatenate peak: sum(arrays) + result ≈ 2 × 400MB = 800MB.
    all_tokens = np.concatenate(arrays)
    del arrays
    gc.collect()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, all_tokens)
    print(f"  Saved {len(all_tokens):,} tokens ({all_tokens.nbytes / 1e6:.0f} MB)")
    del all_tokens
    gc.collect()


class WikitextDataset(Dataset):
    """
    Wikitext-103 windows of length max_seq_len+1, sliced on-the-fly from a
    memory-mapped numpy file.  Steady-state RAM ≈ 0 (OS page cache only).
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 1024,
        stride: int | None = None,
        cache_dir: Path | None = None,
    ):
        stride = stride or max_seq_len
        cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        cache_file = cache_dir / f"wikitext103_{split}_gpt2bpe.npy"

        if not cache_file.exists():
            _build_token_cache(split, cache_file)

        # mmap_mode='r' maps the file into virtual address space without
        # reading it all into RAM — near-zero memory cost.
        self.tokens = np.load(cache_file, mmap_mode="r")
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.n_samples = max(0, (len(self.tokens) - max_seq_len - 1) // stride)
        self.vocab_size = tiktoken.get_encoding("gpt2").n_vocab

        print(f"WikitextDataset [{split}]: {len(self.tokens):,} tokens → "
              f"{self.n_samples:,} windows (seq_len={max_seq_len})")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.stride
        # Copy the small slice into a tensor; avoids keeping mmap pages pinned.
        chunk = torch.from_numpy(
            self.tokens[start: start + self.max_seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def get_wikitext_loaders(
    batch_size: int = 32,
    max_seq_len: int = 1024,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """Returns (train_loader, val_loader, vocab_size)."""
    train_ds = WikitextDataset("train", max_seq_len=max_seq_len)
    val_ds = WikitextDataset("validation", max_seq_len=max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),  # safe to pin when no worker processes
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        drop_last=False,
    )
    return train_loader, val_loader, train_ds.vocab_size
