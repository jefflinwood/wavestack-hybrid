#!/usr/bin/env python
"""Run a needle-in-haystack recall probe for a single model checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.transformer_baseline import TransformerBaseline
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import _resolve_device


def _token_pool(tokenizer: TokenizerWrapper) -> List[int]:
    candidates = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "black",
        "white",
        "gray",
        "pink",
        "silver",
        "gold",
        "cyan",
        "magenta",
        "apple",
        "orange",
        "banana",
        "grape",
        "lemon",
        "peach",
        "cherry",
        "mango",
        "melon",
        "berry",
        "cloud",
        "river",
        "mountain",
        "forest",
        "ocean",
        "desert",
        "storm",
        "breeze",
        "shadow",
        "ember",
        "stone",
        "steel",
        "copper",
        "amber",
        "jade",
        "sable",
        "ivory",
        "crimson",
        "azure",
    ]
    pool: List[int] = []
    for word in candidates:
        tokens = tokenizer.encode(f" {word}")
        if len(tokens) == 1:
            pool.append(tokens[0])
    if len(pool) < 8:
        raise ValueError("Token pool too small; adjust candidate list.")
    return pool


def _prompt_tokens(
    tokenizer: TokenizerWrapper, template: str
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if template == "explicit":
        key_prompt = tokenizer.encode(" The key is")
        value_prompt = tokenizer.encode(" and the value is")
        query_prompt = tokenizer.encode(" When the key is")
        answer_prompt = tokenizer.encode(", the value is")
        return key_prompt, value_prompt, query_prompt, answer_prompt
    key_prompt = tokenizer.encode("Key:")
    value_prompt = tokenizer.encode(" Value:")
    query_prompt = tokenizer.encode(" Query:")
    answer_prompt = tokenizer.encode(" Answer:")
    return key_prompt, value_prompt, query_prompt, answer_prompt


def _build_sequence(
    key_id: int,
    value_id: int,
    filler: List[int],
    prompts: Tuple[List[int], List[int], List[int], List[int]],
) -> Tuple[List[int], int]:
    key_prompt, value_prompt, query_prompt, answer_prompt = prompts
    tokens: List[int] = []
    tokens.extend(key_prompt)
    tokens.append(key_id)
    tokens.extend(value_prompt)
    tokens.append(value_id)
    tokens.extend(filler)
    tokens.extend(query_prompt)
    tokens.append(key_id)
    tokens.extend(answer_prompt)
    return tokens, value_id


def _base_distance(prompts: Tuple[List[int], List[int], List[int], List[int]]) -> int:
    _, _, query_prompt, answer_prompt = prompts
    return len(query_prompt) + len(answer_prompt) + 2


def _pad_batch(sequences: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    batch = []
    lengths = []
    for seq in sequences:
        lengths.append(len(seq))
        padded = seq + [pad_id] * (max_len - len(seq))
        batch.append(padded)
    return torch.tensor(batch, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def _compute_accuracy(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    top_k: int,
) -> float:
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    last_logits = logits.gather(1, idx).squeeze(1)
    topk = torch.topk(last_logits, k=top_k, dim=-1).indices
    match = (topk == targets.unsqueeze(1)).any(dim=1)
    return match.float().mean().item()


def _compute_pool_accuracy(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    pool_ids: List[int],
    top_k: int,
) -> float:
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    last_logits = logits.gather(1, idx).squeeze(1)
    pool = torch.tensor(pool_ids, device=last_logits.device, dtype=torch.long)
    pool_logits = last_logits.index_select(1, pool)
    topk = torch.topk(pool_logits, k=min(top_k, pool_logits.size(1)), dim=-1).indices
    predicted = pool[topk]
    match = (predicted == targets.unsqueeze(1)).any(dim=1)
    return match.float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--device", default="auto", help="Device override.")
    parser.add_argument("--offsets", default="64,128,256,512,1024,2048,3072", help="Comma offsets.")
    parser.add_argument("--samples", type=int, default=128, help="Samples per offset.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--template",
        choices=("simple", "explicit"),
        default="simple",
        help="Prompt template for the key/value recall probe.",
    )
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    device = _resolve_device(args.device)
    tokenizer = TokenizerWrapper()
    pool = _token_pool(tokenizer)
    prompts = _prompt_tokens(tokenizer, args.template)
    base_distance = _base_distance(prompts)
    offsets = [int(v.strip()) for v in args.offsets.split(",") if v.strip()]

    if experiment.model.architecture == "transformer":
        model = TransformerBaseline(experiment.model)
    else:
        model = HybridWaveStack(experiment.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp, torch.no_grad():
        for offset in offsets:
            filler_len = offset - base_distance
            if filler_len < 0:
                continue
            max_len = (
                len(prompts[0])
                + 1
                + len(prompts[1])
                + 1
                + filler_len
                + len(prompts[2])
                + 1
                + len(prompts[3])
            )
            if max_len > experiment.model.max_seq_len:
                continue

            sequences: List[List[int]] = []
            targets: List[int] = []
            for _ in range(args.samples):
                key_id, value_id = rng.sample(pool, 2)
                filler = [rng.choice(pool) for _ in range(filler_len)]
                seq, target = _build_sequence(key_id, value_id, filler, prompts)
                sequences.append(seq)
                targets.append(target)

            batch, lengths = _pad_batch(sequences, tokenizer.pad_id)
            targets_tensor = torch.tensor(targets, dtype=torch.long)

            batch = batch.to(device)
            lengths = lengths.to(device)
            targets_tensor = targets_tensor.to(device)

            accuracies_top1 = []
            accuracies_top5 = []
            pool_top1 = []
            pool_top5 = []
            for start in range(0, batch.size(0), args.batch_size):
                end = start + args.batch_size
                logits = model(batch[start:end])
                acc1 = _compute_accuracy(logits, lengths[start:end], targets_tensor[start:end], top_k=1)
                acc5 = _compute_accuracy(logits, lengths[start:end], targets_tensor[start:end], top_k=5)
                acc_pool1 = _compute_pool_accuracy(
                    logits,
                    lengths[start:end],
                    targets_tensor[start:end],
                    pool_ids=pool,
                    top_k=1,
                )
                acc_pool5 = _compute_pool_accuracy(
                    logits,
                    lengths[start:end],
                    targets_tensor[start:end],
                    pool_ids=pool,
                    top_k=5,
                )
                accuracies_top1.append(acc1)
                accuracies_top5.append(acc5)
                pool_top1.append(acc_pool1)
                pool_top5.append(acc_pool5)

            top1 = sum(accuracies_top1) / len(accuracies_top1)
            top5 = sum(accuracies_top5) / len(accuracies_top5)
            pool1 = sum(pool_top1) / len(pool_top1)
            pool5 = sum(pool_top5) / len(pool_top5)
            payload = {
                "model": experiment.name,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "template": args.template,
                "offset": offset,
                "samples": args.samples,
                "seq_len": max_len,
                "accuracy_top1": top1,
                "accuracy_top5": top5,
                "accuracy_pool_top1": pool1,
                "accuracy_pool_top5": pool5,
                "base_distance": base_distance,
            }
            fp.write(json.dumps(payload) + "\n")
            print(
                "[recall] offset="
                f"{offset} seq_len={max_len} top1={top1:.3f} top5={top5:.3f} "
                f"pool1={pool1:.3f} pool5={pool5:.3f}"
            )


if __name__ == "__main__":
    main()
