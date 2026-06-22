"""Dataset loading and tokenization for KAN-GuppyLM."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from datasets import load_dataset


# Mirror the original GuppyLM token layout exactly:
#   <pad>=0  <|im_start|>=1  <|im_end|>=2
# <|im_start|> doubles as BOS; <|im_end|> doubles as EOS.
# ByteLevel BPE has no unknown tokens (every byte is representable), so no <unk> needed.
SPECIAL_TOKENS = ["<pad>", "<|im_start|>", "<|im_end|>"]
PAD_ID  = 0
BOS_ID  = 1   # reuse <|im_start|> as BOS, same as original GuppyLM
EOS_ID  = 2   # reuse <|im_end|>   as EOS

TEMPLATE = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def format_sample(sample: dict) -> str:
    return TEMPLATE.format(input=sample["input"], output=sample["output"])


def train_tokenizer(
    vocab_size: int = 4096,
    save_path: str = "tokenizer.json",
    dataset_name: str = "arman-bd/guppylm-60k-generic",
) -> Tokenizer:
    """Train a BPE tokenizer on the full dataset and save it."""
    ds = load_dataset(dataset_name)
    texts = [format_sample(s) for split in ds.values() for s in split]

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}  (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)


ASSISTANT_PREFIX = "<|im_start|>assistant\n"


class KanpreyDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 128,
        dataset_name: str = "arman-bd/guppylm-60k-generic",
    ):
        raw = load_dataset(dataset_name, split=split)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.samples = []
        self.prompt_lens = []
        for item in raw:
            text = format_sample(item)
            ids = tokenizer.encode(text).ids
            if len(ids) < 2:
                continue
            ids = ids[: max_seq_len + 1]

            # Compute where the assistant response begins so we can mask the prompt.
            # prompt = everything up to and including "<|im_start|>assistant\n"
            prompt_text = f"<|im_start|>user\n{item['input']}<|im_end|>\n{ASSISTANT_PREFIX}"
            prompt_len = len(tokenizer.encode(prompt_text).ids)

            self.samples.append(ids)
            self.prompt_lens.append(min(prompt_len, len(ids)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        # Only train on the assistant response; mask the user prompt with -100
        # so cross_entropy(ignore_index=-100) skips those positions.
        # y[i] predicts ids[i+1], so mask up to prompt_len-1 positions.
        y[: self.prompt_lens[idx] - 1] = -100
        return x, y


def collate_fn(batch, pad_id: int = PAD_ID):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    x_pad = torch.stack(
        [torch.cat([x, torch.full((max_len - x.shape[0],), pad_id)]) for x in xs]
    )
    y_pad = torch.stack(
        [torch.cat([y, torch.full((max_len - y.shape[0],), -100)]) for y in ys]
    )
    return x_pad, y_pad


def get_dataloader(
    split: str,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 128,
    dataset_name: str = "arman-bd/guppylm-60k-generic",
    shuffle: bool = True,
) -> DataLoader:
    ds = KanpreyDataset(split, tokenizer, max_seq_len, dataset_name)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
