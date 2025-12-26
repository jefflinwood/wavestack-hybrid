#!/usr/bin/env python
"""Utility script to pre-download datasets."""

from __future__ import annotations

import argparse

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download datasets used by WaveStack experiments.")
    parser.add_argument("--dataset", default="roneneldan/TinyStories", help="Dataset identifier.")
    parser.add_argument("--split", default="train", help="Split to download.")
    args = parser.parse_args()

    print(f"Downloading {args.dataset}:{args.split}...")
    load_dataset(args.dataset, split=args.split, streaming=False)
    print("Download complete.")


if __name__ == "__main__":
    main()
