#!/usr/bin/env python
"""Run context-length sweeps for Wikitext-2 experiments."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.data.dataset import WaveStackTextDataset
from wavestack_hybrid.data.tokenizer import TokenizerWrapper
from wavestack_hybrid.models.transformer_baseline import TransformerBaseline
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


def _build_dataloader(
    experiment: ExperimentConfig,
    dataset: Iterable[Mapping[str, torch.Tensor]],
    shuffle: bool,
    seed: int | None,
):
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loader_kwargs: dict[str, object] = {
        "batch_size": experiment.training.batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": experiment.training.num_workers,
        "pin_memory": experiment.training.pin_memory,
    }
    if experiment.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.training.persistent_workers
        if experiment.training.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = experiment.training.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _run_experiment(
    base_config_path: str,
    seq_len: int,
    device: str | None,
    max_steps: int | None,
    samples: int | None,
    seed: int | None,
    batch_size: int | None,
):
    experiment = ExperimentConfig.from_yaml(base_config_path)
    if device:
        experiment.training.device = device
    if max_steps:
        experiment.training.max_steps = max_steps
    if batch_size:
        experiment.training.batch_size = batch_size
    experiment.model.max_seq_len = seq_len
    experiment.name = f"{experiment.name}_ctx{seq_len}"
    experiment.training.log_runtime = True

    checkpoint_root = Path(experiment.checkpoint_dir) / "phase6" / experiment.name
    experiment.checkpoint_dir = str(checkpoint_root)

    if seed is not None:
        set_seed(seed)

    tokenizer = TokenizerWrapper()
    base_dataset = WaveStackTextDataset(
        experiment.dataset_name,
        experiment.train_split,
        tokenizer,
        experiment.model.max_seq_len,
    )

    dataset = base_dataset
    if samples:
        subset_size = min(samples, len(base_dataset))
        dataset = Subset(base_dataset, list(range(subset_size)))

    dataloader = _build_dataloader(experiment, dataset, shuffle=True, seed=seed)

    val_dataloader = None
    try:
        val_dataset = WaveStackTextDataset(
            experiment.dataset_name,
            experiment.val_split,
            tokenizer,
            experiment.model.max_seq_len,
        )
        val_dataloader = _build_dataloader(experiment, val_dataset, shuffle=False, seed=seed)
    except Exception as exc:  # pragma: no cover - best-effort eval hook
        print(f"[ContextSweep] Eval loader unavailable: {exc}")

    dataset_size = len(dataloader.dataset)
    print(
        f"[ContextSweep] Experiment={experiment.name} samples={dataset_size} "
        f"device={experiment.training.device} seq_len={seq_len}"
    )
    print(
        f"[ContextSweep] Max steps {experiment.training.max_steps} | "
        f"Batch size {experiment.training.batch_size}"
    )

    if experiment.model.architecture == "transformer":
        model = TransformerBaseline(experiment.model)
    else:
        model = HybridWaveStack(experiment.model)
    trainer = Trainer(model, experiment)
    summary = trainer.train(dataloader, eval_dataloader=val_dataloader)

    holdout_loss = None
    holdout_size = experiment.training.eval_batches * experiment.training.batch_size
    holdout_indices = None
    use_base_indices = True
    if holdout_size > 0 and len(base_dataset) > 0:
        if samples:
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
        rng = random.Random(seed if seed is not None else 42)
        sample_size = min(holdout_size, len(candidates))
        holdout_indices = rng.sample(candidates, sample_size)
    if holdout_indices:
        holdout_base = base_dataset if use_base_indices else dataset
        holdout_dataset = Subset(holdout_base, holdout_indices)
        holdout_loader = _build_dataloader(experiment, holdout_dataset, shuffle=False, seed=seed)
        holdout_loss = trainer.evaluate(holdout_loader)

    _append_experiment_log(experiment, base_config_path, samples, summary, holdout_loss, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid-config",
        default="experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml",
        help="Hybrid base config path.",
    )
    parser.add_argument(
        "--transformer-config",
        default="experiments/exp1_expressivity/config_AV_transformer_12m_wikitext2_longseq_phase1.yaml",
        help="Transformer base config path.",
    )
    parser.add_argument(
        "--seq-lens",
        default="128,256,512,1024,2048,4096",
        help="Comma-separated sequence lengths to sweep.",
    )
    parser.add_argument("--device", default="auto", help="Training device (auto/cpu/cuda/mps).")
    parser.add_argument("--max-steps", type=int, default=1000, help="Override max steps.")
    parser.add_argument("--samples", type=int, default=8000, help="Limit dataset samples.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    args = parser.parse_args()

    seq_lens = [int(v.strip()) for v in args.seq_lens.split(",") if v.strip()]

    for seq_len in seq_lens:
        _run_experiment(
            args.hybrid_config,
            seq_len=seq_len,
            device=args.device,
            max_steps=args.max_steps,
            samples=args.samples,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    for seq_len in seq_lens:
        _run_experiment(
            args.transformer_config,
            seq_len=seq_len,
            device=args.device,
            max_steps=args.max_steps,
            samples=args.samples,
            seed=args.seed,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
