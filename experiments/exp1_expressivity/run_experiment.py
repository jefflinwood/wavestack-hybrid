#!/usr/bin/env python
"""Launch script for expressivity experiments."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(experiment.dataset_name, experiment.train_split, tokenizer, experiment.model.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=True)

    model = HybridWaveStack(experiment.model)
    trainer = Trainer(model, experiment)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
