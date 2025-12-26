#!/usr/bin/env python
"""Run a short TinyStories-based smoke test to validate the full stack."""

from __future__ import annotations

import argparse
from itertools import cycle

import torch
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig, ModelConfig, TrainingConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="TinyStories-backed smoke test.")
    parser.add_argument("--steps", type=int, default=5, help="Optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dimension.")
    parser.add_argument("--examples", type=int, default=64, help="Number of TinyStories samples to use.")
    parser.add_argument("--split", default="train", help="Dataset split to sample from.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto/cpu/cuda/mps.")
    args = parser.parse_args()

    training = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        log_interval=1,
        save_interval=args.steps + 1,
        use_wandb=False,
        device=args.device,
        mixed_precision=False,
    )
    model_config = ModelConfig(
        vocab_size=50_257,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
    )
    experiment = ExperimentConfig(
        name="tinystories-smoke",
        model=model_config,
        training=training,
        dataset_name="roneneldan/TinyStories",
        train_split=args.split,
        output_dir="./outputs",
        checkpoint_dir="./checkpoints",
    )

    print(f"Loading {args.examples} TinyStories samples from split '{args.split}'...")
    tokenizer = TokenizerWrapper()
    dataset = WaveStackTextDataset(experiment.dataset_name, args.split, tokenizer, experiment.model.max_seq_len)
    subset_indices = list(range(min(args.examples, len(dataset))))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    model = HybridWaveStack(model_config)
    trainer = Trainer(model, experiment)
    trainer.train(cycle(dataloader))
    print("TinyStories smoke test completed successfully.")


if __name__ == "__main__":
    main()
