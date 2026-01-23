#!/usr/bin/env python
"""Run lightweight linguistic probes on Wikitext-2 representations."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from wavestack_hybrid.data.tokenizer import TokenizerWrapper


@dataclass
class ProbeTask:
    name: str
    labels: List[int]
    num_classes: int


def _bin_value(value: float, bins: List[float]) -> int:
    for idx, edge in enumerate(bins):
        if value <= edge:
            return idx
    return len(bins)


def _compute_tasks(input_ids: torch.Tensor, mask: torch.Tensor) -> Dict[str, ProbeTask]:
    tokenizer = TokenizerWrapper()
    punctuation = set(string.punctuation)
    word_re = re.compile(r"[A-Za-z]+")
    digit_re = re.compile(r"[0-9]")

    token_lengths: List[int] = []
    word_counts: List[int] = []
    avg_word_lens: List[float] = []
    cap_ratios: List[float] = []
    punct_ratios: List[float] = []
    digit_present: List[int] = []

    for tokens, token_mask in zip(input_ids, mask):
        valid_tokens = tokens[token_mask].tolist()
        token_lengths.append(len(valid_tokens))
        text = tokenizer.decode(valid_tokens)
        words = word_re.findall(text)
        word_counts.append(len(words))
        if words:
            avg_word_lens.append(sum(len(w) for w in words) / len(words))
            cap_count = sum(1 for w in words if w[0].isupper())
            cap_ratios.append(cap_count / len(words))
        else:
            avg_word_lens.append(0.0)
            cap_ratios.append(0.0)
        if text:
            punct_count = sum(1 for ch in text if ch in punctuation)
            punct_ratios.append(punct_count / max(1, len(text)))
            digit_present.append(1 if digit_re.search(text) else 0)
        else:
            punct_ratios.append(0.0)
            digit_present.append(0)

    tasks: Dict[str, ProbeTask] = {
        "token_length_bin": ProbeTask(
            name="token_length_bin",
            labels=[_bin_value(v, [128, 256, 384]) for v in token_lengths],
            num_classes=4,
        ),
        "word_count_bin": ProbeTask(
            name="word_count_bin",
            labels=[_bin_value(v, [32, 64, 128]) for v in word_counts],
            num_classes=4,
        ),
        "avg_word_len_bin": ProbeTask(
            name="avg_word_len_bin",
            labels=[_bin_value(v, [4.0, 5.5, 7.0]) for v in avg_word_lens],
            num_classes=4,
        ),
        "capitalization_ratio_bin": ProbeTask(
            name="capitalization_ratio_bin",
            labels=[_bin_value(v, [0.05, 0.15, 0.3]) for v in cap_ratios],
            num_classes=4,
        ),
        "punctuation_ratio_bin": ProbeTask(
            name="punctuation_ratio_bin",
            labels=[_bin_value(v, [0.02, 0.05, 0.1]) for v in punct_ratios],
            num_classes=4,
        ),
        "digit_present": ProbeTask(
            name="digit_present",
            labels=digit_present,
            num_classes=2,
        ),
    }
    return tasks


def _train_probe(
    features: torch.Tensor,
    labels: List[int],
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Tuple[float, float]:
    num_samples = features.size(0)
    num_classes = len(set(labels))
    if num_classes < 2:
        return math.nan, math.nan

    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)
    split = int(0.8 * num_samples)
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train = features[train_idx]
    y_train = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    x_val = features[val_idx]
    y_val = torch.tensor([labels[i] for i in val_idx], dtype=torch.long)

    model = nn.Linear(features.size(-1), num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        perm = torch.randperm(x_train.size(0))
        for start in range(0, x_train.size(0), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_logits = model(x_val)
        preds = val_logits.argmax(dim=-1)
        accuracy = (preds == y_val).float().mean().item()

    majority = max(set(y_train.tolist()), key=y_train.tolist().count)
    baseline = (y_val == majority).float().mean().item()
    return accuracy, baseline


def _collect_features(payload: Dict[str, object]) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {"mixed": payload["mixed"].float()}
    lanes = payload.get("lanes")
    if isinstance(lanes, torch.Tensor):
        lane_names = payload.get("lane_names", [])
        for idx, name in enumerate(lane_names):
            features[f"lane_{name}"] = lanes[:, idx, :].float()
    return features


def _write_results(results: Iterable[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for row in results:
            fp.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", required=True, help="Hybrid probe .pt payload.")
    parser.add_argument("--transformer", required=True, help="Transformer probe .pt payload.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=200, help="Probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Probe learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Probe batch size.")
    parser.add_argument(
        "--output",
        default="outputs/probes/wikitext2_probe_results.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    hybrid_payload = torch.load(args.hybrid, map_location="cpu")
    transformer_payload = torch.load(args.transformer, map_location="cpu")

    tasks = _compute_tasks(hybrid_payload["input_ids"], hybrid_payload["mask"])
    results: List[Dict[str, object]] = []

    for label, payload in [("hybrid", hybrid_payload), ("transformer", transformer_payload)]:
        features = _collect_features(payload)
        for task in tasks.values():
            labels = task.labels
            if len(set(labels)) < 2:
                continue
            for rep_name, rep in features.items():
                accuracy, baseline = _train_probe(
                    rep,
                    labels,
                    seed=args.seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                )
                results.append(
                    {
                        "model": label,
                        "representation": rep_name,
                        "task": task.name,
                        "accuracy": accuracy,
                        "baseline": baseline,
                        "num_classes": task.num_classes,
                        "num_samples": rep.size(0),
                    }
                )

    _write_results(results, Path(args.output))
    print(f"[probe] Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
