#!/usr/bin/env python
"""Extract probe representations for Phase 4 Wikitext-2 variants."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from wavestack_hybrid.config import ExperimentConfig


def _run_command(args: Sequence[str]) -> None:
    print(f"[phase4-probe] Running: {' '.join(args)}")
    subprocess.run(list(args), check=True)


def _checkpoint_path(config_path: str, step: int) -> Path:
    config = ExperimentConfig.from_yaml(config_path)
    ckpt_dir = Path(config.checkpoint_dir)
    return ckpt_dir / f"checkpoint_{step:06d}.pt"


def _stem_from_config(config_path: str) -> str:
    config = ExperimentConfig.from_yaml(config_path)
    return config.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml",
            "experiments/exp1_expressivity/config_AW_hybrid_12m_wikitext2_longseq_wavelet_capacity.yaml",
            "experiments/exp1_expressivity/config_AX_hybrid_12m_wikitext2_longseq_wavelet_only.yaml",
            "experiments/exp1_expressivity/config_AY_hybrid_12m_wikitext2_longseq_lane_diversity.yaml",
        ],
        help="Config paths to extract.",
    )
    parser.add_argument("--device", default="auto", help="Device for extraction.")
    parser.add_argument("--checkpoint-step", type=int, default=1000, help="Checkpoint step to load.")
    parser.add_argument("--seq-len-mean", type=int, default=512, help="Seq len for pooled probes.")
    parser.add_argument("--max-samples-mean", type=int, default=512, help="Samples for pooled probes.")
    parser.add_argument("--seq-len-tokens", type=int, default=256, help="Seq len for token probes.")
    parser.add_argument("--max-samples-tokens", type=int, default=128, help="Samples for token probes.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for extraction.")
    parser.add_argument(
        "--output-dir", default="outputs/probes/phase5", help="Base output directory."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.configs:
        stem = _stem_from_config(config_path)
        checkpoint = _checkpoint_path(config_path, args.checkpoint_step)
        mean_out = output_dir / f"{stem}_mean.pt"
        token_out = output_dir / f"{stem}_tokens.pt"

        _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/extract_wikitext2_probe_reprs.py",
                "--config",
                config_path,
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--split",
                "validation",
                "--seq-len",
                str(args.seq_len_mean),
                "--max-samples",
                str(args.max_samples_mean),
                "--batch-size",
                str(args.batch_size),
                "--pool",
                "mean",
                "--output",
                str(mean_out),
            ]
        )

        _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/extract_wikitext2_probe_reprs.py",
                "--config",
                config_path,
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--split",
                "validation",
                "--seq-len",
                str(args.seq_len_tokens),
                "--max-samples",
                str(args.max_samples_tokens),
                "--batch-size",
                str(args.batch_size),
                "--pool",
                "none",
                "--output",
                str(token_out),
            ]
        )


if __name__ == "__main__":
    main()
