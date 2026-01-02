#!/usr/bin/env python
"""Run baseline vs context block experiments for exp1."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[context] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def _parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Context block ablation runner.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()

    configs = [
        "experiments/exp1_expressivity/config_B_hybrid_12m.yaml",
        "experiments/exp1_expressivity/config_Q_hybrid_12m_context_mlp.yaml",
        "experiments/exp1_expressivity/config_R_hybrid_12m_context_conv.yaml",
    ]

    seeds = _parse_seeds(args.seeds)

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
                args.device,
                "--max-steps",
                str(args.max_steps),
                "--samples",
                str(args.samples),
                "--seed",
                str(seed),
            ]
            _run_command(cmd)


if __name__ == "__main__":
    main()
