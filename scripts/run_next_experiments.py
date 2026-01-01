#!/usr/bin/env python
"""Run the immediate next experiment batch (multi-seed baselines, 50m ablations, adaptation sanity)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[next] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def _parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def _run_exp1(
    device: str,
    max_steps: int,
    samples_12m: int,
    samples_50m: int,
    seeds: list[int],
) -> None:
    configs = [
        ("experiments/exp1_expressivity/config_B_hybrid_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_D_neural_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_C_hybrid_50m.yaml", "50m"),
        ("experiments/exp1_expressivity/config_A_neural_50m.yaml", "50m"),
    ]
    for seed in seeds:
        for config_path, size_tag in configs:
            cmd = [
                "uv",
                "run",
                "python",
                str(Path("experiments/exp1_expressivity/run_experiment.py")),
                "--config",
                config_path,
                "--device",
                device,
                "--max-steps",
                str(max_steps),
                "--seed",
                str(seed),
            ]
            if size_tag == "12m" and samples_12m:
                cmd += ["--samples", str(samples_12m)]
            if size_tag == "50m" and samples_50m:
                cmd += ["--samples", str(samples_50m)]
            _run_command(cmd)


def _run_50m_ablations(device: str, max_steps: int, samples: int, seeds: list[int]) -> None:
    configs = [
        "experiments/exp1_expressivity/config_N_hybrid_50m_only_wavelet.yaml",
        "experiments/exp1_expressivity/config_O_hybrid_50m_no_wavelet.yaml",
    ]
    for seed in seeds:
        for config_path in configs:
            cmd = [
                "uv",
                "run",
                "python",
                str(Path("experiments/exp1_expressivity/run_experiment.py")),
                "--config",
                config_path,
                "--device",
                device,
                "--max-steps",
                str(max_steps),
                "--seed",
                str(seed),
            ]
            if samples:
                cmd += ["--samples", str(samples)]
            _run_command(cmd)


def _run_adaptation_sanity(device: str, seed: int | None) -> None:
    cmd = [
        "uv",
        "run",
        "python",
        str(Path("experiments/exp2_adaptation/run_experiment.py")),
        "--pretrain-config",
        "experiments/exp2_adaptation/pretrain_config_sanity.yaml",
        "--finetune-config",
        "experiments/exp2_adaptation/finetune_config_sanity.yaml",
        "--device",
        device,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    _run_command(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the next experiment batch.")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps.")
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds for exp1/ablations.")
    parser.add_argument("--exp1-max-steps", type=int, default=3000)
    parser.add_argument("--exp1-samples-12m", type=int, default=8000)
    parser.add_argument("--exp1-samples-50m", type=int, default=16000)
    parser.add_argument("--ablations-max-steps", type=int, default=3000)
    parser.add_argument("--ablations-samples", type=int, default=16000)
    parser.add_argument("--adaptation-seed", type=int, default=1)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)

    _run_exp1(
        device=args.device,
        max_steps=args.exp1_max_steps,
        samples_12m=args.exp1_samples_12m,
        samples_50m=args.exp1_samples_50m,
        seeds=seeds,
    )
    _run_50m_ablations(
        device=args.device,
        max_steps=args.ablations_max_steps,
        samples=args.ablations_samples,
        seeds=seeds,
    )
    _run_adaptation_sanity(device=args.device, seed=args.adaptation_seed)


if __name__ == "__main__":
    main()
