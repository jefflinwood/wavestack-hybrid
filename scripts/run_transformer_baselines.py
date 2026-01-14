#!/usr/bin/env python
"""Run matched-parameter transformer baselines alongside hybrid configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[transformer-baseline] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def _parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer baseline runner.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--samples-12m", type=int, default=8000)
    parser.add_argument("--samples-50m", type=int, default=16000)
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()

    configs = [
        ("experiments/exp1_expressivity/config_B_hybrid_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_AI_transformer_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_C_hybrid_50m.yaml", "50m"),
        ("experiments/exp1_expressivity/config_AJ_transformer_50m.yaml", "50m"),
    ]

    seeds = _parse_seeds(args.seeds)

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
                args.device,
                "--max-steps",
                str(args.max_steps),
                "--seed",
                str(seed),
            ]
            if size_tag == "12m" and args.samples_12m:
                cmd += ["--samples", str(args.samples_12m)]
            if size_tag == "50m" and args.samples_50m:
                cmd += ["--samples", str(args.samples_50m)]
            _run_command(cmd)


if __name__ == "__main__":
    main()
