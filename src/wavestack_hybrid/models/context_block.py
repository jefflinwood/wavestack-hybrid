"""Context blocks for optional pre/post lane mixing."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..config import ContextBlockConfig


class ContextBlock(nn.Module):
    """Residual context block applied before or after the lane mixer."""

    def __init__(self, hidden_dim: int, config: ContextBlockConfig):
        super().__init__()
        self.block_type = config.block_type
        self.depth = max(1, config.depth)
        self.kernel_size = max(1, config.kernel_size)
        self.causal = config.causal

        if self.block_type == "mlp":
            width = max(8, int(hidden_dim * config.hidden_multiplier))
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, width),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(width, hidden_dim),
                        nn.Dropout(config.dropout),
                    )
                    for _ in range(self.depth)
                ]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.kernel_size, padding=0)
                    for _ in range(self.depth)
                ]
            )
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.block_type == "mlp":
            out = hidden_states
            for layer in self.layers:
                out = out + layer(out)
            return out

        out = hidden_states.transpose(1, 2)
        for conv in self.convs:
            if self.causal:
                out_padded = F.pad(out, (self.kernel_size - 1, 0))
            else:
                pad = self.kernel_size // 2
                out_padded = F.pad(out, (pad, pad))
            conv_out = conv(out_padded)
            conv_out = self.activation(conv_out)
            conv_out = self.dropout(conv_out)
            out = out + conv_out
        return out.transpose(1, 2)
