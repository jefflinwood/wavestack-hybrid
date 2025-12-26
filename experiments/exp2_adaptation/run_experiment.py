#!/usr/bin/env python
"""Fine-tuning pipeline for adaptation experiments."""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def _apply_overrides(experiment: ExperimentConfig, device: str | None, max_steps: int | None):
    if device:
        experiment.training.device = device
    if max_steps:
        experiment.training.max_steps = max_steps


def _build_loader(experiment: ExperimentConfig, tokenizer: TokenizerWrapper, samples: int | None):
    dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )
    if samples:
        dataset = Subset(dataset, list(range(min(samples, len(dataset)))))
    dataloader = DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=True)
    print(
        f"[Adaptation] Stage={experiment.name} samples={len(dataloader.dataset)} "
        f"device={experiment.training.device} max_steps={experiment.training.max_steps}"
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-config", required=True)
    parser.add_argument("--finetune-config", required=True)
    parser.add_argument("--device", default=None, help="Override device for both stages.")
    parser.add_argument("--pretrain-max-steps", type=int, default=None)
    parser.add_argument("--finetune-max-steps", type=int, default=None)
    parser.add_argument("--pretrain-samples", type=int, default=None, help="Subset size for pretrain stage.")
    parser.add_argument("--finetune-samples", type=int, default=None, help="Subset size for finetune stage.")
    args = parser.parse_args()

    tokenizer = TokenizerWrapper()

    pretrain = ExperimentConfig.from_yaml(args.pretrain_config)
    finetune = ExperimentConfig.from_yaml(args.finetune_config)

    _apply_overrides(pretrain, args.device, args.pretrain_max_steps)
    _apply_overrides(finetune, args.device, args.finetune_max_steps)

    model = HybridWaveStack(pretrain.model)

    pretrain_loader = _build_loader(pretrain, tokenizer, args.pretrain_samples)
    Trainer(model, pretrain).train(pretrain_loader)

    finetune_loader = _build_loader(finetune, tokenizer, args.finetune_samples)
    Trainer(model, finetune).train(finetune_loader)


if __name__ == "__main__":
    main()
