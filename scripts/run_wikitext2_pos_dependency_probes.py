#!/usr/bin/env python
"""Run POS and dependency probes using spaCy labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
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
    label_names: List[str]


def _load_spacy(model_name: str):
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "spaCy is required for POS/dependency probes. Install with "
            "`uv pip install spacy` and download a model (e.g. "
            "`python -m spacy download en_core_web_sm`)."
        ) from exc
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' not found. Download with "
            f"`python -m spacy download {model_name}`."
        ) from exc


def _token_spans(tokenizer: TokenizerWrapper, token_ids: List[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for tok_id in token_ids:
        piece = tokenizer.decode([tok_id])
        pieces.append(piece)
        start = cursor
        cursor += len(piece)
        spans.append((start, cursor))
    return "".join(pieces), spans


def _align_spacy_to_bpe(
    tokenizer: TokenizerWrapper, token_ids: List[int], mask: List[bool], nlp
) -> Tuple[List[torch.Tensor], List[str], List[str], List[str]]:
    active_ids = [tok for tok, keep in zip(token_ids, mask) if keep]
    text, spans = _token_spans(tokenizer, active_ids)
    doc = nlp(text)

    alignments: List[List[int]] = []
    pos_labels: List[str] = []
    dep_labels: List[str] = []
    head_dirs: List[str] = []
    head_bins: List[str] = []
    bins = [1, 2, 4, 8, 16, 32, 64]

    for token in doc:
        start = token.idx
        end = start + len(token)
        indices = [
            idx
            for idx, (t_start, t_end) in enumerate(spans)
            if t_end > start and t_start < end
        ]
        if not indices:
            continue
        alignments.append(indices)
        pos_labels.append(token.pos_)
        dep_labels.append(token.dep_)
        if token.head.i == token.i:
            head_dirs.append("root")
            head_bins.append("0")
        else:
            direction = "left" if token.head.i < token.i else "right"
            head_dirs.append(direction)
            dist = abs(token.head.i - token.i)
            for edge in bins:
                if dist <= edge:
                    head_bins.append(f"<={edge}")
                    break
            else:
                head_bins.append(f">{bins[-1]}")

    return alignments, pos_labels, dep_labels, head_dirs, head_bins


def _collect_features(
    payload: Dict[str, object]
) -> Dict[str, torch.Tensor]:
    mixed = payload.get("mixed")
    if not isinstance(mixed, torch.Tensor) or mixed.dim() != 3:
        raise ValueError("Probe payload must be extracted with --pool none (mixed shape: N x S x D).")
    features = {"mixed": mixed.float()}
    lanes = payload.get("lanes")
    if isinstance(lanes, torch.Tensor) and lanes.dim() == 4:
        lane_names = payload.get("lane_names", [])
        for idx, name in enumerate(lane_names):
            features[f"lane_{name}"] = lanes[:, idx, :, :].float()
    return features


def _aggregate_token_features(
    features: torch.Tensor, alignments: List[List[int]]
) -> torch.Tensor:
    token_vectors = []
    for indices in alignments:
        vec = features[indices].mean(dim=0)
        token_vectors.append(vec)
    if not token_vectors:
        return torch.empty((0, features.size(-1)))
    return torch.stack(token_vectors, dim=0)


def _prepare_probe_dataset(
    payload: Dict[str, object],
    model_name: str,
    nlp,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, ProbeTask]]:
    tokenizer = TokenizerWrapper()
    input_ids = payload["input_ids"].tolist()
    mask = payload["mask"].tolist()
    reps = _collect_features(payload)

    task_labels: Dict[str, List[str]] = defaultdict(list)
    rep_features: Dict[str, List[torch.Tensor]] = {name: [] for name in reps}

    for sample_idx, (tokens, mask_row) in enumerate(zip(input_ids, mask)):
        alignments, pos, dep, head_dir, head_bins = _align_spacy_to_bpe(
            tokenizer, tokens, mask_row, nlp
        )
        if not alignments:
            continue
        task_labels["pos"].extend(pos)
        task_labels["dep"].extend(dep)
        task_labels["head_dir"].extend(head_dir)
        task_labels["head_dist_bin"].extend(head_bins)
        for rep_name, rep_tensor in reps.items():
            rep_sample = rep_tensor[sample_idx]
            token_feats = _aggregate_token_features(rep_sample, alignments)
            rep_features[rep_name].append(token_feats)

    stacked_features: Dict[str, torch.Tensor] = {}
    for rep_name, tensors in rep_features.items():
        if tensors:
            stacked_features[rep_name] = torch.cat(tensors, dim=0)

    tasks: Dict[str, ProbeTask] = {}
    for name, labels in task_labels.items():
        label_set = sorted(set(labels))
        mapping = {label: idx for idx, label in enumerate(label_set)}
        tasks[name] = ProbeTask(
            name=f"{model_name}_{name}",
            labels=[mapping[label] for label in labels],
            label_names=label_set,
        )

    return stacked_features, tasks


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
    if num_classes < 2 or num_samples < 2:
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


def _write_results(rows: Iterable[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", required=True, help="Hybrid probe .pt payload (pool=none).")
    parser.add_argument("--transformer", required=True, help="Transformer probe .pt payload (pool=none).")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=200, help="Probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Probe learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Probe batch size.")
    parser.add_argument(
        "--output",
        default="outputs/probes/wikitext2_pos_dep_results.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    nlp = _load_spacy(args.spacy_model)
    hybrid_payload = torch.load(args.hybrid, map_location="cpu")
    transformer_payload = torch.load(args.transformer, map_location="cpu")

    hybrid_features, hybrid_tasks = _prepare_probe_dataset(hybrid_payload, "hybrid", nlp)
    transformer_features, transformer_tasks = _prepare_probe_dataset(
        transformer_payload, "transformer", nlp
    )

    results: List[Dict[str, object]] = []
    for rep_name, rep in hybrid_features.items():
        for task_name, task in hybrid_tasks.items():
            accuracy, baseline = _train_probe(
                rep,
                task.labels,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            results.append(
                {
                    "model": "hybrid",
                    "representation": rep_name,
                    "task": task_name,
                    "accuracy": accuracy,
                    "baseline": baseline,
                    "num_classes": len(task.label_names),
                    "num_samples": rep.size(0),
                    "labels": task.label_names,
                }
            )

    for rep_name, rep in transformer_features.items():
        for task_name, task in transformer_tasks.items():
            accuracy, baseline = _train_probe(
                rep,
                task.labels,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
            results.append(
                {
                    "model": "transformer",
                    "representation": rep_name,
                    "task": task_name,
                    "accuracy": accuracy,
                    "baseline": baseline,
                    "num_classes": len(task.label_names),
                    "num_samples": rep.size(0),
                    "labels": task.label_names,
                }
            )

    _write_results(results, Path(args.output))
    print(f"[probe] Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
