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
            self.gating = nn.Sequential(
                nn.Linear(hidden_dim * num_lanes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_lanes),
            )
        elif mixing_type == "attention":
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:  # mlp mixing
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * num_lanes, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def _check_inputs(self, lane_outputs: torch.Tensor):
        if lane_outputs.size(0) != self.num_lanes:
            raise ValueError(
                f"Lane mixer expected {self.num_lanes} lanes, received {lane_outputs.size(0)}"
            )

    def forward(self, lane_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lane_outputs: tensor of shape (num_lanes, batch, seq, dim)
        """

        self._check_inputs(lane_outputs)

        if self.mixing_type == "gated":
            batch, seq, dim = lane_outputs.size(1), lane_outputs.size(2), lane_outputs.size(3)
            flattened = lane_outputs.permute(1, 2, 0, 3).reshape(batch, seq, self.num_lanes * dim)
            gate_logits = self.gating(flattened)
            weights = torch.softmax(gate_logits, dim=-1).permute(2, 0, 1).unsqueeze(-1)
            return (lane_outputs * weights).sum(dim=0)

        if self.mixing_type == "attention":
            # Treat lanes as tokens and let them attend to each other per position.
            lane_tokens = lane_outputs.permute(1, 2, 0, 3)  # (batch, seq, lanes, dim)
            q = self.query(lane_tokens)
            k = self.key(lane_tokens)
            v = self.value(lane_tokens)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (lane_tokens.size(-1) ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            attended = torch.matmul(attn, v).mean(dim=2)
            return attended

        # mlp mixing
        batch, seq, dim = lane_outputs.size(1), lane_outputs.size(2), lane_outputs.size(3)
        flattened = lane_outputs.permute(1, 2, 0, 3).reshape(batch, seq, self.num_lanes * dim)
        return self.mlp(flattened)
