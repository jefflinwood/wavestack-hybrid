#!/usr/bin/env python
"""Gradient tracking study runner."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.analysis.gradient_tracker import GradientTracker
from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    if args.device:
        experiment.training.device = args.device
    if args.max_steps:
        experiment.training.max_steps = args.max_steps

    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )
    if args.samples:
        dataset = Subset(dataset, list(range(min(args.samples, len(dataset)))))

    dataloader = DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=True)

    print(
        f"[Gradients] Experiment={experiment.name} samples={len(dataloader.dataset)} "
        f"device={experiment.training.device} max_steps={experiment.training.max_steps}"
    )

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


if __name__ == "__main__":
    main()
