#!/usr/bin/env python
"""Run 10k-step long-seq Wikitext-2 training for hybrid and transformer."""

from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


def _run_command(args: Sequence[str]) -> None:
    print(f"[longseq-10k] Running: {' '.join(args)}")
    subprocess.run(list(args), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", help="Training device (auto/cpu/cuda/mps).")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps.")
    parser.add_argument("--samples", type=int, default=8000, help="Sample cap.")
    parser.add_argument("--seed", type=int, default=1, help="Seed.")
    parser.add_argument(
        "--hybrid-config",
        default="experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml",
        help="Hybrid config path.",
    )
    parser.add_argument(
        "--transformer-config",
        default="experiments/exp1_expressivity/config_AV_transformer_12m_wikitext2_longseq_phase1.yaml",
        help="Transformer config path.",
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

    _run_command([*base_cmd, "--config", args.hybrid_config])
    _run_command([*base_cmd, "--config", args.transformer_config])


if __name__ == "__main__":
    main()
