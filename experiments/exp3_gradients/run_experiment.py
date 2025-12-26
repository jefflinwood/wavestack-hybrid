#!/usr/bin/env python
"""Gradient tracking study runner."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from wavestack_hybrid.analysis.gradient_tracker import GradientTracker
from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(experiment.dataset_name, experiment.train_split, tokenizer, experiment.model.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=True)

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
