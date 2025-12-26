"""Neural baseline decomposition for ablation runs."""

from __future__ import annotations

from torch import nn


class NeuralDecomposition(nn.Module):
    """Simple stack of linear projections used when analytical lanes are disabled."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.net(hidden_states)
