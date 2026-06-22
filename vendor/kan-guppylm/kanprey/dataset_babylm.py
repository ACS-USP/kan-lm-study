"""BabyLM Strict-Small dataset loader and tokenizer.

BabyLM-community/BabyLM-2026-Strict-Small: 10M-word child-directed text corpus,
~1.1M short passages. Only a train split is provided — we split 90/10 for
train/validation.

Standard LM objective: all-token cross-entropy, no ChatML masking, no role tokens.
Uses a BPE tokenizer trained on the corpus with a configurable vocab size.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

# Minimal special tokens.  ByteLevel BPE has no unknown tokens — every byte is
# representable.  <pad> is the only special token needed.
SPECIAL_TOKENS = ["<pad>"]
PAD_ID = 0

BABYLM_DATASET_PATH = "BabyLM-community/BabyLM-2026-Strict-Small"
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_TOKENIZER_PATH = "tokenizer_babylm.json"
TRAIN_FRAC = 0.9  # 90 % train, 10 % validation
_CACHE_DIR = Path.home() / ".cache" / "kanprey"


# ── Tokenizer ────────────────────────────────────────────────────

def train_babylm_tokenizer(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_path: str = DEFAULT_TOKENIZER_PATH,
    dataset_path: str = BABYLM_DATASET_PATH,
) -> Tokenizer:
    """Train a BPE tokenizer on the BabyLM corpus and save it.

    The tokenizer is cached to disk — subsequent calls with the same path
    will load it rather than retrain.
    """
    if Path(save_path).exists():
        return load_babylm_tokenizer(save_path)

    ds = load_dataset(dataset_path, split="train")

    def text_iter():
        for example in ds:
            yield example["text"]

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(text_iter(), trainer=trainer)
    tokenizer.save(save_path)
    print(f"BabyLM tokenizer saved → {save_path}  (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


def load_babylm_tokenizer(path: str = DEFAULT_TOKENIZER_PATH) -> Tokenizer:
    return Tokenizer.from_file(path)


_SPLIT_SEED = 42  # Fixed seed for the 90/10 train/val partition — shared across all training seeds.


def _cache_path(split: str, vocab_size: int) -> Path:
    """Deterministic path for the pre-tokenized .npy cache."""
    return _CACHE_DIR / f"babylm_{vocab_size}_{split}.npy"

def _build_token_cache(
    tokenizer: Tokenizer,
    dataset_path: str,
    vocab_size: int,
) -> None:
    """Tokenize the entire corpus once, split 90/10, save two .npy files.

    Peak RAM: ~140 MB (raw texts + growing ids arrays for one split at a time).
    Subsequent runs mmap the cache at near-zero memory cost.
    """
    print("Building pre-tokenized BabyLM cache (one-time)…")
    ds = load_dataset(dataset_path, split="train")
    texts = [example["text"] for example in ds]

    # Deterministic shuffle + split using fixed SPLIT_SEED.
    rng = torch.Generator().manual_seed(_SPLIT_SEED)
    n = len(texts)
    perm = torch.randperm(n, generator=rng).tolist()
    split_idx = int(n * TRAIN_FRAC)

    for split, indices in [("train", perm[:split_idx]), ("val", perm[split_idx:])]:
        cache_file = _cache_path(split, vocab_size)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        arrays: list[np.ndarray] = []
        for i in indices:
            ids = tokenizer.encode(texts[i]).ids
            arrays.append(np.array(ids, dtype=np.int32))
            del ids

        all_tokens = np.concatenate(arrays)
        del arrays
        gc.collect()

        np.save(cache_file, all_tokens)
        print(f"  {split}: {len(all_tokens):,} tokens → {cache_file}")
        del all_tokens
        gc.collect()

    print("Pre-tokenized cache ready.")



# ── Dataset ──────────────────────────────────────────────────────

class BabyLMDataset(Dataset):
    """Standard LM dataset over BabyLM text — contiguous fixed-length windows.

    Every token is a target.  No ChatML template, no role masking.

    On first use, tokenizes the entire corpus and caches it as .npy files
    in ~/.cache/kanprey/.  Subsequent runs mmap the cache at near-zero
    RAM and startup cost.

    The train/val split uses a fixed seed (_SPLIT_SEED) so all training seeds
    share the same partition — the training seed only affects model init.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer: Tokenizer | None = None,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        max_seq_len: int = 128,
        dataset_path: str = BABYLM_DATASET_PATH,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or load_babylm_tokenizer(tokenizer_path)
        vocab_size = self.tokenizer.get_vocab_size()

        cache_file = _cache_path(split, vocab_size)
        if not cache_file.exists():
            _build_token_cache(self.tokenizer, dataset_path, vocab_size)

        # mmap_mode='r' maps the file into virtual address space without
        # reading it all into RAM — near-zero memory, OS page cache only.
        self.ids = np.load(cache_file, mmap_mode="r")
        self.n_samples = max(0, len(self.ids) // (max_seq_len + 1))

        print(
            f"BabyLMDataset [{split}]: {len(self.ids):,} tokens → "
            f"{self.n_samples:,} windows (seq_len={max_seq_len})"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * (self.max_seq_len + 1)
        # Copy the small slice into a tensor; avoids keeping mmap pages pinned.
        chunk = torch.from_numpy(
            self.ids[start : start + self.max_seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = PAD_ID):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    x_pad = torch.stack([
        torch.cat([x, torch.full((max_len - x.shape[0],), pad_id)]) for x in xs
    ])
    y_pad = torch.stack([
        torch.cat([y, torch.full((max_len - y.shape[0],), -100)]) for y in ys
    ])
    return x_pad, y_pad


def get_babylm_dataloader(
    split: str,
    tokenizer: Tokenizer | None = None,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    max_seq_len: int = 128,
    batch_size: int = 32,
    dataset_path: str = BABYLM_DATASET_PATH,
    num_workers: int = 0,
) -> DataLoader:
    ds = BabyLMDataset(
        split=split,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        dataset_path=dataset_path,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        drop_last=(split == "train"),
        collate_fn=lambda b: collate_fn(b, PAD_ID),
    )
