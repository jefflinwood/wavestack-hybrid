#!/usr/bin/env python
"""Run the Phase 3 scaling protocol (sequence length sweeps)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[scaling] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling protocol runner.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config",
        default="experiments/exp1_expressivity/config_B_hybrid_12m.yaml",
        help="Hybrid config path to set max_seq_len.",
    )
    parser.add_argument("--seq-lens", default="512,1024,2048,4096")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tf-max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--output",
        default="outputs/scaling_sweep.jsonl",
        help="JSONL output path for per-sequence timings.",
    )
    args = parser.parse_args()

    cmd = [
        "uv",
        "run",
        "python",
        str(Path("scripts/benchmark_inference.py")),
        "--model",
        "both",
        "--config",
        args.config,
        "--device",
        args.device,
        "--seq-lens",
        args.seq_lens,
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--tf-max-seq-len",
        str(args.tf_max_seq_len),
        "--output",
        args.output,
        "--log-results",
    ]
    _run_command(cmd)


if __name__ == "__main__":
    main()
