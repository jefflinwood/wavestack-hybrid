#!/usr/bin/env python
"""Launch script for expressivity experiments."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--device", default=None, help="Override TrainingConfig.device (auto/cpu/cuda/mps).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override TrainingConfig.max_steps for quick smoke runs.")
    parser.add_argument("--samples", type=int, default=None, help="Limit the number of dataset samples loaded.")
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
        subset_size = min(args.samples, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))

    dataloader = DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=True)

    dataset_size = len(dataloader.dataset)
    print(f"[Expressivity] Experiment={experiment.name} samples={dataset_size} device={experiment.training.device}")
    print(f"[Expressivity] Max steps {experiment.training.max_steps} | Batch size {experiment.training.batch_size}")

    model = HybridWaveStack(experiment.model)
    trainer = Trainer(model, experiment)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
