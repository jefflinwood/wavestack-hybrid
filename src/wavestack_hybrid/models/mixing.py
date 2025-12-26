"""Lane mixing utilities."""

from __future__ import annotations

import torch
from torch import nn


class LaneMixer(nn.Module):
    """Combines lane outputs via gating, attention, or small MLP."""

    def __init__(self, hidden_dim: int, num_lanes: int, mixing_type: str = "gated"):
        super().__init__()
        self.mixing_type = mixing_type
        self.num_lanes = num_lanes

        if mixing_type == "gated":
            self.gates = nn.Parameter(torch.zeros(num_lanes, hidden_dim))
        elif mixing_type == "attention":
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:  # mlp
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * num_lanes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, lane_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_outputs: tensor of shape (num_lanes, batch, seq, dim)
        """

        if self.mixing_type == "gated":
            gates = torch.softmax(self.gates, dim=0).view(self.num_lanes, 1, 1, -1)
            return (lane_outputs * gates).sum(dim=0)

        if self.mixing_type == "attention":
            combined = lane_outputs.mean(dim=0)
            attn_scores = torch.matmul(self.query(combined), self.key(combined).transpose(-1, -2))
            attn_scores = attn_scores / combined.size(-1) ** 0.5
            attn = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn, self.value(combined))

        # mlp
        batch, seq, dim = lane_outputs.size(1), lane_outputs.size(2), lane_outputs.size(3)
        flattened = lane_outputs.permute(1, 2, 0, 3).reshape(batch, seq, self.num_lanes * dim)
        return self.mlp(flattened)
