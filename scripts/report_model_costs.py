#!/usr/bin/env python
"""Report parameter and FLOP estimates for a config."""

from __future__ import annotations

import argparse

from wavestack_hybrid.config import ExperimentConfig


def _print_section(title: str, rows: dict[str, float]) -> None:
    print(f"\n{title}")
    for key, value in rows.items():
        print(f"- {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report model parameter/FLOP estimates.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="If set, report per-sequence FLOPs instead of per-token.",
    )
    args = parser.parse_args()

    experiment = ExperimentConfig.from_yaml(args.config)
    model = experiment.model

    param_breakdown = model.get_param_breakdown()
    param_breakdown["total"] = model.get_param_count()
    _print_section("Parameter Estimates", param_breakdown)

    lane_params = model.get_lane_param_breakdown()
    _print_section("Per-Lane Parameter Estimates", lane_params)

    flop_breakdown = model.get_flop_breakdown(seq_len=args.seq_len)
    flop_breakdown["total"] = sum(flop_breakdown.values())
    _print_section(
        "FLOP Estimates (per-sequence)" if args.seq_len else "FLOP Estimates (per-token)",
        flop_breakdown,
    )

    lane_flops = model.get_lane_flop_breakdown(seq_len=args.seq_len)
    _print_section(
        "Per-Lane FLOP Estimates (per-sequence)" if args.seq_len else "Per-Lane FLOP Estimates (per-token)",
        lane_flops,
    )


if __name__ == "__main__":
    main()
