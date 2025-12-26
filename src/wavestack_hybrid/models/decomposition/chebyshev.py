"""Chebyshev polynomial decomposition utilities."""

from __future__ import annotations

import torch
from torch import nn

from ...config import DecompositionConfig


class ChebyshevDecomposition(nn.Module):
    """Generates Chebyshev polynomial coefficients from hidden states."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.order = config.poly_order
        self.normalization = config.poly_normalization
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return projected Chebyshev coefficients."""

        x = hidden_states
        if self.normalization == "unit":
            denom = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x / denom

        polys = [torch.ones_like(x), x]
        for _ in range(2, self.order):
            polys.append(2 * x * polys[-1] - polys[-2])

        stacked = torch.stack(polys[: self.order], dim=-2)  # (B, S, order, H)
        coeffs = stacked.mean(dim=-2)  # Average over order dimension
        return self.project(coeffs)
