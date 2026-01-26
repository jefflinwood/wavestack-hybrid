#!/usr/bin/env python
"""Run recall sweeps for hybrid and transformer checkpoints."""

from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


def _run_command(args: Sequence[str]) -> None:
    print(f"[recall-sweep] Running: {' '.join(args)}")
    subprocess.run(list(args), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hybrid-config",
        default="experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml",
        help="Hybrid config path.",
    )
    parser.add_argument(
        "--hybrid-checkpoint",
        default="checkpoints/phase5/hybrid/checkpoint_001000.pt",
        help="Hybrid checkpoint path.",
    )
    parser.add_argument(
        "--transformer-config",
        default="experiments/exp1_expressivity/config_AV_transformer_12m_wikitext2_longseq_phase1.yaml",
        help="Transformer config path.",
    )
    parser.add_argument(
        "--transformer-checkpoint",
        default="checkpoints/phase5/transformer/checkpoint_001000.pt",
        help="Transformer checkpoint path.",
    )
    parser.add_argument("--device", default="auto", help="Device override.")
    parser.add_argument(
        "--offsets",
        default="64,128,256,512,1024,2048,3072",
        help="Comma-separated offsets.",
    )
    parser.add_argument("--samples", type=int, default=128, help="Samples per offset.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--template",
        choices=("simple", "explicit"),
        default="simple",
        help="Prompt template for the recall probe.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/recall",
        help="Output directory for JSONL results.",
    )
    args = parser.parse_args()

    hybrid_out = f"{args.output_dir}/wikitext2_hybrid_recall.jsonl"
    transformer_out = f"{args.output_dir}/wikitext2_transformer_recall.jsonl"

    base_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_context_recall_probe.py",
        "--device",
        args.device,
        "--offsets",
        args.offsets,
        "--samples",
        str(args.samples),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--template",
        args.template,
    ]

    _run_command(
        [
            *base_cmd,
            "--config",
            args.hybrid_config,
            "--checkpoint",
            args.hybrid_checkpoint,
            "--output",
            hybrid_out,
        ]
    )
    _run_command(
        [
            *base_cmd,
            "--config",
            args.transformer_config,
            "--checkpoint",
            args.transformer_checkpoint,
            "--output",
            transformer_out,
        ]
    )


if __name__ == "__main__":
    main()
