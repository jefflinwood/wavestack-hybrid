#!/usr/bin/env python
"""Run a small CodeSearchNet hybrid tuning sweep."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[codesearchnet-sweep] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def _parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeSearchNet hybrid sweep runner.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seeds", default="1")
    args = parser.parse_args()

    configs = [
        "experiments/exp1_expressivity/config_AM_hybrid_12m_codesearchnet.yaml",
        "experiments/exp1_expressivity/config_AP_hybrid_12m_codesearchnet_lr1e4.yaml",
        "experiments/exp1_expressivity/config_AQ_hybrid_12m_codesearchnet_recomp_standard.yaml",
        "experiments/exp1_expressivity/config_AR_hybrid_12m_codesearchnet_lane_caps.yaml",
        "experiments/exp1_expressivity/config_AS_hybrid_12m_codesearchnet_no_poly.yaml",
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
                "--samples",
                str(args.samples),
                "--seed",
                str(seed),
            ]
            _run_command(cmd)


if __name__ == "__main__":
    main()
