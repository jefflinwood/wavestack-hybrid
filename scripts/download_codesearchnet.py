#!/usr/bin/env python
"""Download CodeSearchNet data via the datasets package."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CodeSearchNet data locally.")
    parser.add_argument("--language", default="python")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/codesearchnet")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    dataset = load_dataset("code_search_net", args.language, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.language}_{args.split}.jsonl"
    dataset.to_json(str(output_path))
    print(f"Wrote {len(dataset)} samples to {output_path}")


if __name__ == "__main__":
    main()
