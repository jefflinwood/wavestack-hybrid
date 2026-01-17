#!/usr/bin/env python
"""Run the Phase 3 scaling protocol using Wikitext-2 configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run_command(args: list[str]) -> None:
    print(f"[scaling-wikitext2] Running: {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling protocol runner for Wikitext-2.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-lens", default="512,1024,2048,4096")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        default="outputs/scaling_sweep_wikitext2.jsonl",
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
        "experiments/exp1_expressivity/config_AL_hybrid_12m_wikitext2_longseq.yaml",
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
        "4096",
        "--output",
        args.output,
        "--log-results",
    ]
    _run_command(cmd)


if __name__ == "__main__":
    main()
