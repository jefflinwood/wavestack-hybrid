#!/usr/bin/env python
"""Extract lane and mixed representations for Wikitext-2 linguistic probes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.transformer_baseline import TransformerBaseline
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import _resolve_device


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (hidden * mask_f).sum(dim=1) / denom


def _masked_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, idx).squeeze(1)


def _pool_hidden(hidden: torch.Tensor, mask: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "mean":
        return _masked_mean(hidden, mask)
    if pool == "last":
        return _masked_last(hidden, mask)
    return hidden


def _pool_lanes(
    lane_outputs: torch.Tensor, mask: torch.Tensor, pool: str
) -> torch.Tensor:
    # lane_outputs: (lanes, batch, seq, dim) -> (batch, lanes, seq, dim)
    lane_outputs = lane_outputs.permute(1, 0, 2, 3)
    if pool == "mean":
        mask_f = mask.float().unsqueeze(1).unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp(min=1.0)
        return (lane_outputs * mask_f).sum(dim=2) / denom
    if pool == "last":
        lengths = mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1, 1).expand(
            -1, lane_outputs.size(1), 1, lane_outputs.size(-1)
        )
        return lane_outputs.gather(2, idx).squeeze(2)
    return lane_outputs


def _iter_batches(loader: Iterable[Dict[str, torch.Tensor]]):
    for batch in loader:
        yield batch["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--device", default="auto", help="Device override (auto/cpu/cuda/mps).")
    parser.add_argument("--split", default="validation", help="Dataset split to extract.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for extraction.")
    parser.add_argument("--max-samples", type=int, default=512, help="Max samples to extract.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for extraction.")
    parser.add_argument(
        "--pool",
        choices=("none", "mean", "last"),
        default="mean",
        help="Pooling strategy over sequence length.",
    )
    parser.add_argument("--output", required=True, help="Output .pt file path.")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    device = _resolve_device(args.device)
    if args.seq_len > experiment.model.max_seq_len:
        raise ValueError(
            f"seq_len {args.seq_len} exceeds model max_seq_len {experiment.model.max_seq_len}"
        )

    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(
        experiment.dataset_name,
        args.split,
        tokenizer,
        args.seq_len,
    )
    if args.max_samples:
        subset_size = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if experiment.model.architecture == "transformer":
        model = TransformerBaseline(experiment.model)
        lane_names: list[str] = []
    else:
        model = HybridWaveStack(experiment.model)
        lane_names = list(model.lane_names)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    all_inputs = []
    all_masks = []
    all_mixed = []
    all_lanes = []

    with torch.no_grad():
        for input_ids in _iter_batches(dataloader):
            input_ids = input_ids.to(device)
            mask = input_ids != tokenizer.pad_id
            if isinstance(model, HybridWaveStack):
                reps = model.forward_representations(input_ids)
                mixed = reps["mixed"]
                lane_outputs = reps["lane_outputs"]
                pooled_mixed = _pool_hidden(mixed, mask, args.pool)
                pooled_lanes = _pool_lanes(lane_outputs, mask, args.pool)
                all_lanes.append(pooled_lanes.cpu())
            else:
                hidden = model.forward_hidden(input_ids)
                pooled_mixed = _pool_hidden(hidden, mask, args.pool)
            all_mixed.append(pooled_mixed.cpu())
            all_inputs.append(input_ids.cpu())
            all_masks.append(mask.cpu())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset": experiment.dataset_name,
        "split": args.split,
        "seq_len": args.seq_len,
        "pool": args.pool,
        "input_ids": torch.cat(all_inputs, dim=0),
        "mask": torch.cat(all_masks, dim=0),
        "mixed": torch.cat(all_mixed, dim=0),
        "lane_names": lane_names,
        "lanes": torch.cat(all_lanes, dim=0) if all_lanes else None,
    }
    torch.save(payload, output_path)
    print(f"[probe] Wrote {payload['input_ids'].size(0)} samples to {output_path}")


if __name__ == "__main__":
    main()
