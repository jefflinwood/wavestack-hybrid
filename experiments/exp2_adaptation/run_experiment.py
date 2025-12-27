#!/usr/bin/env python
"""Fine-tuning pipeline for adaptation experiments."""

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


def _build_eval_loader(experiment: ExperimentConfig, tokenizer: TokenizerWrapper):
    try:
        dataset = WaveStackTextDataset(
            experiment.dataset_name,
            experiment.val_split,
            tokenizer,
            experiment.model.max_seq_len,
        )
    except Exception as exc:  # pragma: no cover - best-effort eval hook
        print(f"[Adaptation] Eval loader unavailable for {experiment.name}: {exc}")
        return None
    return DataLoader(dataset, batch_size=experiment.training.batch_size, shuffle=False)


def _append_experiment_log(
    experiment: ExperimentConfig,
    config_path: str,
    stage: str,
    samples: int | None,
    summary: Mapping[str, float | int | None],
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    train_loss = summary.get("train_loss")
    eval_loss = summary.get("eval_loss")
    lines = [
        "",
        timestamp,
        "- Study: exp2_adaptation",
        f"- Stage: {stage}",
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
    pretrain_eval_loader = _build_eval_loader(pretrain, tokenizer)
    pretrain_summary = Trainer(model, pretrain).train(pretrain_loader, pretrain_eval_loader)
    _append_experiment_log(
        pretrain, args.pretrain_config, "pretrain", args.pretrain_samples, pretrain_summary
    )

    finetune_loader = _build_loader(finetune, tokenizer, args.finetune_samples)
    finetune_eval_loader = _build_eval_loader(finetune, tokenizer)
    finetune_summary = Trainer(model, finetune).train(finetune_loader, finetune_eval_loader)
    _append_experiment_log(
        finetune, args.finetune_config, "finetune", args.finetune_samples, finetune_summary
    )


if __name__ == "__main__":
    main()
