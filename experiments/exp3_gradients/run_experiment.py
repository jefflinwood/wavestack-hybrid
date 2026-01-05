#!/usr/bin/env python
"""Gradient tracking study runner."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, Subset

import torch

from wavestack_hybrid.analysis.gradient_tracker import GradientTracker
from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.seed import set_seed
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs and dataloader shuffling.")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    if args.device:
        experiment.training.device = args.device
    if args.max_steps:
        experiment.training.max_steps = args.max_steps
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )
    if args.samples:
        dataset = Subset(dataset, list(range(min(args.samples, len(dataset)))))

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    loader_kwargs: dict[str, object] = {
        "batch_size": experiment.training.batch_size,
        "shuffle": True,
        "generator": generator,
        "num_workers": experiment.training.num_workers,
        "pin_memory": experiment.training.pin_memory,
    }
    if experiment.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
        if experiment.training.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    print(
        f"[Gradients] Experiment={experiment.name} samples={len(dataloader.dataset)} "
        f"device={experiment.training.device} max_steps={experiment.training.max_steps}"
    )
    params = experiment.model.get_param_breakdown()
    lane_params = experiment.model.get_lane_param_breakdown()
    flops = experiment.model.get_flop_breakdown(seq_len=experiment.model.max_seq_len)
    flops["total"] = sum(flops.values())
    lane_flops = experiment.model.get_lane_flop_breakdown(seq_len=experiment.model.max_seq_len)

    model = HybridWaveStack(experiment.model)
    tracker = GradientTracker(model)

    trainer = Trainer(model, experiment)
    for step, batch in enumerate(dataloader):
        loss = trainer._training_step(batch)  # pylint: disable=protected-access
        if step % experiment.training.log_interval == 0:
            tracker.log_gradients(step)
        if step >= experiment.training.max_steps:
            break

    print(tracker.summary())
    print(f"[Gradients] Params total: {params.get('total', 0)}")
    print(f"[Gradients] Params breakdown: {params}")
    print(f"[Gradients] Params lanes: {lane_params}")
    print(f"[Gradients] FLOPs total (seq): {flops.get('total', 0):.2e}")
    print(f"[Gradients] FLOPs breakdown (seq): {flops}")
    print(f"[Gradients] FLOPs lanes (seq): {lane_flops}")


if __name__ == "__main__":
    main()
