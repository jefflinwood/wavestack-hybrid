"""Datasets helpers for loading TinyStories/FineWeb style corpora."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from datasets import load_dataset


class WaveStackTextDataset(Dataset):
    """Tokenizes HF datasets into fixed-length blocks."""

    def __init__(self, dataset_name: str, split: str, tokenizer, seq_len: int):
        super().__init__()
        self.dataset = load_dataset(dataset_name, split=split, streaming=False)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        text = sample["text"] if "text" in sample else next(iter(sample.values()))
        tokens = self.tokenizer.encode(text)
        input_ids = tokens[: self.seq_len]
        if len(input_ids) < self.seq_len:
            pad = [self.tokenizer.pad_id] * (self.seq_len - len(input_ids))
            input_ids = input_ids + pad
        tensor = torch.tensor(input_ids, dtype=torch.long)
        return {"input_ids": tensor, "labels": tensor}
