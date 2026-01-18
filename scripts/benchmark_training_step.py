#!/usr/bin/env python
"""Benchmark training step scaling across sequence lengths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import copy

from wavestack_hybrid.config import ExperimentConfig
from wavestack_hybrid.models.transformer_baseline import TransformerBaseline
from wavestack_hybrid.models.wavestack import HybridWaveStack
from wavestack_hybrid.training.trainer import Trainer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()


def _get_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        if current is not None:
            return int(current())
    return None


def _benchmark(
    model,
    trainer: Trainer,
    device: torch.device,
    seq_lens: list[int],
    batch_size: int,
    steps: int,
    warmup: int,
    vocab_size: int,
    max_seq_len: int,
    output_path: Path | None,
    model_name: str,
):
    model_name = model_name
    model.train()
    results = []
    for seq_len in seq_lens:
        if seq_len > max_seq_len:
            print(f"[train-bench] Skipping seq_len={seq_len} (max_seq_len={max_seq_len})")
            continue
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        batch = {"input_ids": input_ids, "labels": input_ids}
        for _ in range(warmup):
            _ = trainer._compute_loss(batch)  # pylint: disable=protected-access
            trainer.optimizer.zero_grad()
        _sync(device)
        start = time.perf_counter()
        for _ in range(steps):
            loss = trainer._compute_loss(batch)  # pylint: disable=protected-access
            if trainer.grad_scaler is not None:
                trainer.grad_scaler.scale(loss).backward()
                trainer.grad_scaler.unscale_(trainer.optimizer)
                trainer.grad_scaler.step(trainer.optimizer)
                trainer.grad_scaler.update()
            else:
                loss.backward()
                trainer.optimizer.step()
            trainer.optimizer.zero_grad()
        _sync(device)
        total = time.perf_counter() - start
        time_per_step = total / max(1, steps)
        tokens_per_s = (batch_size * seq_len) / time_per_step
        memory = _get_memory_bytes(device)
        record = {
            "model": model_name,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "time_s": time_per_step,
            "tokens_per_s": tokens_per_s,
            "memory_bytes": memory,
        }
        results.append(record)
        if output_path:
            with output_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record) + "\n")
    return results


def _append_experiment_log(
    model_name: str,
    results: list[dict[str, float | int | None]],
    batch_size: int,
    steps: int,
    warmup: int,
    device: str,
) -> None:
    seq_lens = [str(r["seq_len"]) for r in results]
    timing_pairs = ", ".join(
        f"{r['seq_len']}:{r['time_s']*1000:.2f}ms/{r['tokens_per_s']:.1f}tps/{r['memory_bytes'] or 0}B"
        for r in results
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        timestamp,
        "- Study: training_benchmark",
        f"- Model: {model_name}",
        f"- Device: {device}",
        f"- Seq lens: {','.join(seq_lens)}",
        f"- Batch size: {batch_size}",
        f"- Steps: {steps}",
        f"- Warmup: {warmup}",
        f"- Timing: {timing_pairs}",
    ]
    log_path = Path("EXPERIMENT_LOG.md")
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training step scaling benchmark.")
    parser.add_argument("--model", default="both", choices=["hybrid", "transformer", "both"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-lens", default="128,256,512")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-results", action="store_true", help="Append summary to EXPERIMENT_LOG.md.")
    parser.add_argument("--tf-max-seq-len", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device(args.device)

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = ExperimentConfig.from_yaml(args.config)

    if args.model in {"hybrid", "both"}:
        experiment.training.device = str(device)
        experiment.training.mixed_precision = False
        model = HybridWaveStack(experiment.model)
        trainer = Trainer(model, experiment)
        hybrid_results = _benchmark(
            model,
            trainer,
            device,
            seq_lens,
            args.batch_size,
            args.steps,
            args.warmup,
            experiment.model.vocab_size,
            experiment.model.max_seq_len,
            output_path,
            "hybrid",
        )
        if args.log_results:
            _append_experiment_log(
                model_name="hybrid",
                results=hybrid_results,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                device=str(device),
            )

    if args.model in {"transformer", "both"}:
        transformer_experiment = copy.deepcopy(experiment)
        transformer_experiment.training.device = str(device)
        transformer_experiment.training.mixed_precision = False
        transformer_config = transformer_experiment.model
        transformer_config.architecture = "transformer"
        if args.tf_max_seq_len is not None:
            transformer_config.max_seq_len = args.tf_max_seq_len
        model = TransformerBaseline(transformer_config)
        trainer = Trainer(model, transformer_experiment)
        transformer_results = _benchmark(
            model,
            trainer,
            device,
            seq_lens,
            args.batch_size,
            args.steps,
            args.warmup,
            transformer_config.vocab_size,
            transformer_config.max_seq_len,
            output_path,
            "transformer",
        )
        if args.log_results:
            _append_experiment_log(
                model_name="transformer",
                results=transformer_results,
                batch_size=args.batch_size,
                steps=args.steps,
                warmup=args.warmup,
                device=str(device),
            )


if __name__ == "__main__":
    main()
