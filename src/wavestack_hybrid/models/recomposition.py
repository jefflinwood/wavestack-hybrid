"""Recomposition networks that turn lane signals into logits."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from ..config import RecompositionConfig


def _activation(name: str):
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "swish": nn.SiLU(),
    }[name]


class LaneRecompositionNetwork(nn.Module):
    """MLP that consumes lane-specific features and produces fused representations."""

    def __init__(self, hidden_dim: int, config: RecompositionConfig, capacity: float):
        super().__init__()
        width = max(32, int(hidden_dim * capacity))
        depth = {"shallow": 1, "standard": 2, "deep": 3}[config.depth]
        layers = []
        in_dim = hidden_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(_activation(config.activation))
            layers.append(nn.Dropout(config.dropout))
            in_dim = width
        layers.append(nn.Linear(width, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class RecompositionBundle(nn.Module):
    """Builds recomposition networks for each lane and returns their outputs."""

    def __init__(self, hidden_dim: int, config: RecompositionConfig, lanes: Iterable[str] | None = None):
        super().__init__()
        available = {
            "poly": LaneRecompositionNetwork(hidden_dim, config, config.poly_capacity),
            "trig": LaneRecompositionNetwork(hidden_dim, config, config.trig_capacity),
            "wavelet": LaneRecompositionNetwork(hidden_dim, config, config.wavelet_capacity),
        }
        lane_list = list(lanes) if lanes is not None else list(available.keys())
        missing = [lane for lane in lane_list if lane not in available]
        if missing:
            raise ValueError(f"Unknown lanes requested: {missing}")
        self.lanes = nn.ModuleDict({lane: available[lane] for lane in lane_list})

    def forward(self, lane_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: net(lane_features[name]) for name, net in self.lanes.items()}
