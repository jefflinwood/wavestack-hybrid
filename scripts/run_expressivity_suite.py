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
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of seeds to run (overrides --repeat).",
    )
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="Include lane ablation configs in the run list.",
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
    if args.include_ablations:
        configs.extend(
            [
                ("experiments/exp1_expressivity/config_E_hybrid_12m_only_poly.yaml", "12m"),
                ("experiments/exp1_expressivity/config_F_hybrid_12m_only_trig.yaml", "12m"),
                ("experiments/exp1_expressivity/config_G_hybrid_12m_only_wavelet.yaml", "12m"),
                ("experiments/exp1_expressivity/config_H_hybrid_12m_no_poly.yaml", "12m"),
                ("experiments/exp1_expressivity/config_I_hybrid_12m_no_trig.yaml", "12m"),
                ("experiments/exp1_expressivity/config_J_hybrid_12m_no_wavelet.yaml", "12m"),
                ("experiments/exp1_expressivity/config_N_hybrid_50m_only_wavelet.yaml", "50m"),
                ("experiments/exp1_expressivity/config_O_hybrid_50m_no_wavelet.yaml", "50m"),
            ]
        )

    if args.seeds:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    else:
        seeds = [None] * args.repeat

    for run_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            label = f"{seed}" if seed is not None else f"{run_idx + 1}"
            print(f"[suite] Starting run {run_idx + 1}/{len(seeds)} (seed={label})")
        for config_path, size_tag in configs:
            cmd = base_cmd + ["--config", config_path, "--device", args.device]
            if args.max_steps:
                cmd += ["--max-steps", str(args.max_steps)]
            if size_tag == "12m" and args.samples_12m:
                cmd += ["--samples", str(args.samples_12m)]
            if size_tag == "50m" and args.samples_50m:
                cmd += ["--samples", str(args.samples_50m)]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            _run_command(cmd)


if __name__ == "__main__":
    main()
