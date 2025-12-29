"""Seed helpers for experiment reproducibility."""

from __future__ import annotations

import random

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in the environment
    np = None


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG seeds."""

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
