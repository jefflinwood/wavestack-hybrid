"""Lane attribution helpers for qualitative analysis."""

from __future__ import annotations

from typing import Dict

import torch


def lane_attributions(lane_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize lane activations to highlight contribution per token."""

    contributions = {}
    total = sum(tensor.abs() for tensor in lane_outputs.values())
    total = total + 1e-6
    for name, tensor in lane_outputs.items():
        contributions[name] = (tensor.abs() / total).mean(dim=-1)
    return contributions
