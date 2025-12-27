#!/usr/bin/env python
"""Run the expressivity experiment suite sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[suite] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all exp1 expressivity configs.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Override device for all runs (auto/cpu/cuda/mps).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override TrainingConfig.max_steps for all runs.",
    )
    parser.add_argument(
        "--samples-12m",
        type=int,
        default=8000,
        help="Sample cap for 12m configs; set to 0 to disable.",
    )
    parser.add_argument(
        "--samples-50m",
        type=int,
        default=16000,
        help="Sample cap for 50m configs; set to 0 to disable.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat the full suite.",
    )
    args = parser.parse_args()

    base_cmd = [
        sys.executable,
        str(Path("experiments/exp1_expressivity/run_experiment.py")),
    ]

    configs = [
        ("experiments/exp1_expressivity/config_B_hybrid_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_D_neural_12m.yaml", "12m"),
        ("experiments/exp1_expressivity/config_C_hybrid_50m.yaml", "50m"),
        ("experiments/exp1_expressivity/config_A_neural_50m.yaml", "50m"),
    ]

    for run_idx in range(args.repeat):
        if args.repeat > 1:
            print(f"[suite] Starting run {run_idx + 1}/{args.repeat}")
        for config_path, size_tag in configs:
            cmd = base_cmd + ["--config", config_path, "--device", args.device]
            if args.max_steps:
                cmd += ["--max-steps", str(args.max_steps)]
            if size_tag == "12m" and args.samples_12m:
                cmd += ["--samples", str(args.samples_12m)]
            if size_tag == "50m" and args.samples_50m:
                cmd += ["--samples", str(args.samples_50m)]
            _run_command(cmd)


if __name__ == "__main__":
    main()
