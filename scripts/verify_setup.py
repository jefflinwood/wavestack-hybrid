#!/usr/bin/env python
"""Basic environment checks."""

from __future__ import annotations

import importlib
import sys


REQUIRED = ["torch", "numpy", "tiktoken", "datasets"]


def main():
    missing = []
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)

    import torch

    print(f"Torch version: {torch.__version__}")
    print("Environment looks good!")


if __name__ == "__main__":
    main()
