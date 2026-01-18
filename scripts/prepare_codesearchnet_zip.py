#!/usr/bin/env python
"""Prepare CodeSearchNet JSONL files from a local ZIP."""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path


def _write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CodeSearchNet ZIP to JSONL.")
    parser.add_argument("--zip", dest="zip_path", default="data/python.zip")
    parser.add_argument("--output", default="data/codesearchnet")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {"train": [], "valid": []}
    unmatched = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            lower_name = name.lower()
            if not (lower_name.endswith(".jsonl") or lower_name.endswith(".jsonl.gz")):
                continue
            if "/train/" in lower_name or lower_name.endswith("train.jsonl") or "train" in Path(lower_name).parts:
                split = "train"
            elif (
                "/valid/" in lower_name
                or lower_name.endswith("valid.jsonl")
                or "/validation/" in lower_name
                or "valid" in Path(lower_name).parts
                or "validation" in Path(lower_name).parts
            ):
                split = "valid"
            else:
                unmatched.append(name)
                continue
            with zf.open(name) as fp:
                raw = fp.read()
                if lower_name.endswith(".gz"):
                    raw = gzip.decompress(raw)
                for line in raw.decode("utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    records[split].append(json.loads(line))

    if not records["train"] or not records["valid"]:
        sample = ", ".join(unmatched[:5])
        raise RuntimeError(
            "Missing train/valid records in ZIP; verify file structure. "
            f"Unmatched examples: {sample or 'none'}"
        )

    train_path = output_dir / "python_train.jsonl"
    valid_path = output_dir / "python_valid.jsonl"
    _write_jsonl(records["train"], train_path)
    _write_jsonl(records["valid"], valid_path)

    print(f"Wrote {len(records['train'])} train samples to {train_path}")
    print(f"Wrote {len(records['valid'])} valid samples to {valid_path}")


if __name__ == "__main__":
    main()
