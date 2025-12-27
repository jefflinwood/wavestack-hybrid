#!/usr/bin/env python
"""Launch script for expressivity experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Mapping

from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def _append_experiment_log(
    experiment: ExperimentConfig,
    config_path: str,
    samples: int | None,
    summary: Mapping[str, float | int | None],
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    train_loss = summary.get("train_loss")
    eval_loss = summary.get("eval_loss")
    lines = [
        "",
        timestamp,
        f"- Study: exp1_expressivity",
        f"- Experiment: {experiment.name}",
        f"- Config: {config_path}",
        f"- Dataset: {experiment.dataset_name}",
        f"- Device: {experiment.training.device}",
        f"- Max steps: {experiment.training.max_steps}",
        f"- Samples: {samples if samples is not None else 'all'}",
        f"- Train loss: {train_loss:.4f}" if train_loss is not None else "- Train loss: n/a",
        f"- Eval loss: {eval_loss:.4f}" if eval_loss is not None else "- Eval loss: n/a",
    ]
    log_path = Path("EXPERIMENT_LOG.md")
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


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
    val_dataloader = None
    try:
        val_dataset = WaveStackTextDataset(
            experiment.dataset_name,
            experiment.val_split,
            tokenizer,
            experiment.model.max_seq_len,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=experiment.training.batch_size, shuffle=False
        )
    except Exception as exc:  # pragma: no cover - best-effort eval hook
        print(f"[Expressivity] Eval loader unavailable: {exc}")

    dataset_size = len(dataloader.dataset)
    print(f"[Expressivity] Experiment={experiment.name} samples={dataset_size} device={experiment.training.device}")
    print(f"[Expressivity] Max steps {experiment.training.max_steps} | Batch size {experiment.training.batch_size}")

    model = HybridWaveStack(experiment.model)
    trainer = Trainer(model, experiment)
    summary = trainer.train(dataloader, eval_dataloader=val_dataloader)
    _append_experiment_log(experiment, args.config, args.samples, summary)


if __name__ == "__main__":
    main()
