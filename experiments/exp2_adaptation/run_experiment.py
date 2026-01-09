#!/usr/bin/env python
"""Fine-tuning pipeline for adaptation experiments."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.models.transformer_baseline import TransformerBaseline
from wavestack_hybrid.training.seed import set_seed
from wavestack_hybrid.training.trainer import Trainer


def _apply_overrides(experiment: ExperimentConfig, device: str | None, max_steps: int | None):
    if device:
        experiment.training.device = device
    if max_steps:
        experiment.training.max_steps = max_steps


def _build_train_loader(
    experiment: ExperimentConfig, tokenizer: TokenizerWrapper, samples: int | None, seed: int | None
):
    base_dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )
    dataset = base_dataset
    if samples:
        dataset = Subset(base_dataset, list(range(min(samples, len(base_dataset)))))
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader_kwargs: dict[str, object] = {
        "batch_size": experiment.training.batch_size,
        "shuffle": True,
        "generator": generator,
        "num_workers": experiment.training.num_workers,
        "pin_memory": experiment.training.pin_memory,
    }
    if experiment.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
        if experiment.training.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)
    print(
        f"[Adaptation] Stage={experiment.name} samples={len(dataloader.dataset)} "
        f"device={experiment.training.device} max_steps={experiment.training.max_steps}"
    )
    return dataloader, base_dataset, dataset


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
    loader_kwargs = {
        "batch_size": experiment.training.batch_size,
        "shuffle": False,
        "num_workers": experiment.training.num_workers,
        "pin_memory": experiment.training.pin_memory,
    }
    if experiment.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
        if experiment.training.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _build_holdout_loss(
    trainer: Trainer,
    experiment: ExperimentConfig,
    base_dataset: WaveStackTextDataset,
    train_dataset: Subset | WaveStackTextDataset,
    samples: int | None,
    seed: int | None,
) -> float | None:
    holdout_size = experiment.training.eval_batches * experiment.training.batch_size
    if holdout_size <= 0 or len(base_dataset) == 0:
        return None
    if samples:
        remaining_indices = list(range(len(base_dataset)))[len(train_dataset) :]
        if remaining_indices:
            candidates = remaining_indices
            use_base_indices = True
        else:
            candidates = list(range(len(train_dataset)))
            use_base_indices = False
    else:
        candidates = list(range(len(base_dataset)))
        use_base_indices = True
    rng = random.Random(seed if seed is not None else 42)
    sample_size = min(holdout_size, len(candidates))
    holdout_indices = rng.sample(candidates, sample_size)
    if not holdout_indices:
        return None
    holdout_base = base_dataset if use_base_indices else train_dataset
    holdout_dataset = Subset(holdout_base, holdout_indices)
    loader_kwargs = {
        "batch_size": experiment.training.batch_size,
        "shuffle": False,
        "num_workers": experiment.training.num_workers,
        "pin_memory": experiment.training.pin_memory,
    }
    if experiment.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
        if experiment.training.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
    holdout_loader = DataLoader(holdout_dataset, **loader_kwargs)
    return trainer.evaluate(holdout_loader)


def _append_experiment_log(
    experiment: ExperimentConfig,
    config_path: str,
    stage: str,
    samples: int | None,
    summary: Mapping[str, float | int | None],
    holdout_loss: float | None,
    seed: int | None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    train_loss = summary.get("train_loss")
    eval_loss = summary.get("eval_loss")
    runtime_s = summary.get("runtime_s")
    tokens_per_s = summary.get("tokens_per_s")
    peak_memory_bytes = summary.get("peak_memory_bytes")
    params = experiment.model.get_param_breakdown()
    lane_params = experiment.model.get_lane_param_breakdown()
    flops = experiment.model.get_flop_breakdown(seq_len=experiment.model.max_seq_len)
    flops["total"] = sum(flops.values())
    lane_flops = experiment.model.get_lane_flop_breakdown(seq_len=experiment.model.max_seq_len)
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
        f"- Seed: {seed}" if seed is not None else "- Seed: n/a",
        f"- Train loss: {train_loss:.4f}" if train_loss is not None else "- Train loss: n/a",
        f"- Eval loss: {eval_loss:.4f}" if eval_loss is not None else "- Eval loss: n/a",
        f"- Holdout loss: {holdout_loss:.4f}" if holdout_loss is not None else "- Holdout loss: n/a",
        f"- Runtime (s): {runtime_s:.2f}" if runtime_s is not None else "- Runtime (s): n/a",
        f"- Tokens/s: {tokens_per_s:.2f}" if tokens_per_s is not None else "- Tokens/s: n/a",
        f"- Peak memory (bytes): {peak_memory_bytes}" if peak_memory_bytes is not None else "- Peak memory (bytes): n/a",
        f"- Params total: {params.get('total', 0)}",
        f"- Params breakdown: {params}",
        f"- Params lanes: {lane_params}",
        f"- FLOPs total (seq): {flops.get('total', 0):.2e}",
        f"- FLOPs breakdown (seq): {flops}",
        f"- FLOPs lanes (seq): {lane_flops}",
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
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs and dataloader shuffling.")
    args = parser.parse_args()

    tokenizer = TokenizerWrapper()

    pretrain = ExperimentConfig.from_yaml(args.pretrain_config)
    finetune = ExperimentConfig.from_yaml(args.finetune_config)

    _apply_overrides(pretrain, args.device, args.pretrain_max_steps)
    _apply_overrides(finetune, args.device, args.finetune_max_steps)
    if args.seed is not None:
        set_seed(args.seed)

    if pretrain.model.architecture == "transformer":
        model = TransformerBaseline(pretrain.model)
    else:
        model = HybridWaveStack(pretrain.model)

    pretrain_loader, pretrain_base, pretrain_dataset = _build_train_loader(
        pretrain, tokenizer, args.pretrain_samples, args.seed
    )
    pretrain_eval_loader = _build_eval_loader(pretrain, tokenizer)
    pretrain_trainer = Trainer(model, pretrain)
    pretrain_summary = pretrain_trainer.train(pretrain_loader, pretrain_eval_loader)

    pretrain_holdout = _build_holdout_loss(
        pretrain_trainer, pretrain, pretrain_base, pretrain_dataset, args.pretrain_samples, args.seed
    )
    _append_experiment_log(
        pretrain,
        args.pretrain_config,
        "pretrain",
        args.pretrain_samples,
        pretrain_summary,
        pretrain_holdout,
        args.seed,
    )

    finetune_loader, finetune_base, finetune_dataset = _build_train_loader(
        finetune, tokenizer, args.finetune_samples, args.seed
    )
    finetune_eval_loader = _build_eval_loader(finetune, tokenizer)
    finetune_trainer = Trainer(model, finetune)
    finetune_summary = finetune_trainer.train(finetune_loader, finetune_eval_loader)
    finetune_holdout = _build_holdout_loss(
        finetune_trainer, finetune, finetune_base, finetune_dataset, args.finetune_samples, args.seed
    )
    _append_experiment_log(
        finetune,
        args.finetune_config,
        "finetune",
        args.finetune_samples,
        finetune_summary,
        finetune_holdout,
        args.seed,
    )


if __name__ == "__main__":
    main()
