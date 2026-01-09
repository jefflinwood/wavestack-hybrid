#!/usr/bin/env python
"""Launch script for expressivity experiments."""

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
from wavestack_hybrid.training.seed import set_seed
from wavestack_hybrid.training.trainer import Trainer


def _append_experiment_log(
    experiment: ExperimentConfig,
    config_path: str,
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
    params["total"] = experiment.model.get_param_count()
    lane_params = experiment.model.get_lane_param_breakdown()
    flops = experiment.model.get_flop_breakdown(seq_len=experiment.model.max_seq_len)
    flops["total"] = sum(flops.values())
    lane_flops = experiment.model.get_lane_flop_breakdown(seq_len=experiment.model.max_seq_len)
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
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--device", default=None, help="Override TrainingConfig.device (auto/cpu/cuda/mps).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override TrainingConfig.max_steps for quick smoke runs.")
    parser.add_argument("--samples", type=int, default=None, help="Limit the number of dataset samples loaded.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs and dataloader shuffling.")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    if args.device:
        experiment.training.device = args.device
    if args.max_steps:
        experiment.training.max_steps = args.max_steps
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = TokenizerWrapper()
    base_dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )

    dataset = base_dataset
    if args.samples:
        subset_size = min(args.samples, len(base_dataset))
        dataset = Subset(base_dataset, list(range(subset_size)))

    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
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
    val_dataloader = None
    try:
        val_dataset = WaveStackTextDataset(
            experiment.dataset_name,
            experiment.val_split,
            tokenizer,
            experiment.model.max_seq_len,
        )
        val_loader_kwargs = {
            "batch_size": experiment.training.batch_size,
            "shuffle": False,
            "num_workers": experiment.training.num_workers,
            "pin_memory": experiment.training.pin_memory,
        }
        if experiment.training.num_workers > 0:
            val_loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
            if experiment.training.prefetch_factor is not None:
                val_loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
        val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)
    except Exception as exc:  # pragma: no cover - best-effort eval hook
        print(f"[Expressivity] Eval loader unavailable: {exc}")

    dataset_size = len(dataloader.dataset)
    print(f"[Expressivity] Experiment={experiment.name} samples={dataset_size} device={experiment.training.device}")
    print(f"[Expressivity] Max steps {experiment.training.max_steps} | Batch size {experiment.training.batch_size}")

    model = HybridWaveStack(experiment.model)
    trainer = Trainer(model, experiment)
    summary = trainer.train(dataloader, eval_dataloader=val_dataloader)

    holdout_loss = None
    holdout_size = experiment.training.eval_batches * experiment.training.batch_size
    holdout_indices = None
    use_base_indices = True
    if holdout_size > 0 and len(base_dataset) > 0:
        if args.samples:
            remaining_indices = list(range(len(base_dataset)))[len(dataset) :]
            if remaining_indices:
                candidates = remaining_indices
                use_base_indices = True
            else:
                candidates = list(range(len(dataset)))
                use_base_indices = False
        else:
            candidates = list(range(len(base_dataset)))
            use_base_indices = True
        rng = random.Random(args.seed if args.seed is not None else 42)
        sample_size = min(holdout_size, len(candidates))
        holdout_indices = rng.sample(candidates, sample_size)
    if holdout_indices:
        holdout_base = base_dataset if use_base_indices else dataset
        holdout_dataset = Subset(holdout_base, holdout_indices)
        holdout_loader_kwargs = {
            "batch_size": experiment.training.batch_size,
            "shuffle": False,
            "num_workers": experiment.training.num_workers,
            "pin_memory": experiment.training.pin_memory,
        }
        if experiment.training.num_workers > 0:
            holdout_loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
            if experiment.training.prefetch_factor is not None:
                holdout_loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
        holdout_loader = DataLoader(holdout_dataset, **holdout_loader_kwargs)
        holdout_loss = trainer.evaluate(holdout_loader)

    _append_experiment_log(experiment, args.config, args.samples, summary, holdout_loss, args.seed)


if __name__ == "__main__":
    main()
