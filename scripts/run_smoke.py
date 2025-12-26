#!/usr/bin/env python
"""Run a lightweight synthetic training loop to ensure wiring works."""

from __future__ import annotations

import argparse

import torch

from wavestack_hybrid.config import ExperimentConfig, ModelConfig, TrainingConfig
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


class SyntheticLoader:
    """Yields random token batches for a fixed number of steps."""

    def __init__(self, *, steps: int, batch_size: int, seq_len: int, vocab_size: int):
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self):
        for _ in range(self.steps):
            tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            yield {"input_ids": tokens, "labels": tokens.clone()}


def main():
    parser = argparse.ArgumentParser(description="Run a quick synthetic smoke test.")
    parser.add_argument("--steps", type=int, default=5, help="Number of optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size.")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for synthetic tokens.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for the model.")
    parser.add_argument("--vocab-size", type=int, default=256, help="Synthetic vocabulary size.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto/cpu/cuda/mps.")
    args = parser.parse_args()

    training = TrainingConfig(
        max_steps=args.steps,
        batch_size=args.batch_size,
        log_interval=1,
        save_interval=args.steps + 1,  # skip checkpoints
        use_wandb=False,
        device=args.device,
        mixed_precision=False,
    )
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.seq_len,
    )
    experiment = ExperimentConfig(
        name="smoke",
        model=model_config,
        training=training,
        dataset_name="synthetic",
        output_dir="./outputs",
        checkpoint_dir="./checkpoints",
    )

    model = HybridWaveStack(model_config)
    loader = SyntheticLoader(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    trainer = Trainer(model, experiment)
    trainer.train(loader)
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
