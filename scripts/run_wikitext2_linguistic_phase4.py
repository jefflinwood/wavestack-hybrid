#!/usr/bin/env python
"""Run Phase 4 linguistic follow-ups for Wikitext-2."""

from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


def _run_command(args: Sequence[str]) -> None:
    print(f"[phase4] Running: {' '.join(args)}")
    subprocess.run(list(args), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="Training device (auto/cpu/cuda/mps).")
    parser.add_argument("--max-steps", type=int, default=1000, help="Override max steps for quick checkpoints.")
    parser.add_argument("--samples", type=int, default=8000, help="Limit dataset samples for faster runs.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "experiments/exp1_expressivity/config_AW_hybrid_12m_wikitext2_longseq_wavelet_capacity.yaml",
            "experiments/exp1_expressivity/config_AX_hybrid_12m_wikitext2_longseq_wavelet_only.yaml",
            "experiments/exp1_expressivity/config_AY_hybrid_12m_wikitext2_longseq_lane_diversity.yaml",
        ],
        help="Config paths to run.",
    )
    args = parser.parse_args()

    base_cmd = [
        "uv",
        "run",
        "python",
        "experiments/exp1_expressivity/run_experiment.py",
        "--device",
        args.device,
        "--max-steps",
        str(args.max_steps),
        "--samples",
        str(args.samples),
        "--seed",
        str(args.seed),
    ]

    for config in args.configs:
        _run_command([*base_cmd, "--config", config])


if __name__ == "__main__":
    main()
