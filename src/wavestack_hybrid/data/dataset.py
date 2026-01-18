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
        if dataset_name.startswith("json:"):
            spec = dataset_name.split(":", 1)[1]
            data_files = _parse_data_files(spec)
            self.dataset = load_dataset("json", data_files=data_files, split=split, streaming=False)
        elif dataset_name.startswith("parquet:"):
            spec = dataset_name.split(":", 1)[1]
            data_files = _parse_data_files(spec)
            self.dataset = load_dataset("parquet", data_files=data_files, split=split, streaming=False)
        elif ":" in dataset_name:
            base_name, config_name = dataset_name.split(":", 1)
            self.dataset = load_dataset(base_name, config_name, split=split, streaming=False)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=False)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        if "code" in sample:
            text = sample["code"]
        elif "text" in sample:
            text = sample["text"]
        elif "docstring" in sample:
            text = sample["docstring"]
        else:
            text = next(iter(sample.values()))
        tokens = self.tokenizer.encode(text)
        input_ids = tokens[: self.seq_len]
        input_len = len(input_ids)
        if input_len < self.seq_len:
            pad = [self.tokenizer.pad_id] * (self.seq_len - input_len)
            input_ids = input_ids + pad
        tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = tensor.clone()
        if input_len < self.seq_len:
            labels[input_len:] = -100
        return {"input_ids": tensor, "labels": labels}


def _parse_data_files(spec: str):
    if "=" not in spec:
        return spec
    data_files = {}
    for entry in spec.split(","):
        if not entry.strip():
            continue
        key, value = entry.split("=", 1)
        data_files[key.strip()] = value.strip()
    return data_files
