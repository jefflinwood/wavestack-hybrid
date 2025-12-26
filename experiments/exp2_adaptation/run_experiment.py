#!/usr/bin/env python
"""Fine-tuning pipeline for adaptation experiments."""

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
    parser.add_argument("--pretrain-config", required=True)
    parser.add_argument("--finetune-config", required=True)
    args = parser.parse_args()

    tokenizer = TokenizerWrapper()

    pretrain = ExperimentConfig.from_yaml(args.pretrain_config)
    finetune = ExperimentConfig.from_yaml(args.finetune_config)

    model = HybridWaveStack(pretrain.model)
    pretrain_dataset = WaveStackTextDataset(pretrain.dataset_name, pretrain.train_split, tokenizer, pretrain.model.max_seq_len)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=pretrain.training.batch_size, shuffle=True)
    Trainer(model, pretrain).train(pretrain_loader)

    finetune_dataset = WaveStackTextDataset(finetune.dataset_name, finetune.train_split, tokenizer, finetune.model.max_seq_len)
    finetune_loader = DataLoader(finetune_dataset, batch_size=finetune.training.batch_size, shuffle=True)
    Trainer(model, finetune).train(finetune_loader)


if __name__ == "__main__":
    main()
