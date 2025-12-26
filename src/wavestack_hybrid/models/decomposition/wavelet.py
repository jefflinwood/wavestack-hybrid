"""Fixed wavelet-inspired pooling layers."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ...config import DecompositionConfig


class WaveletDecomposition(nn.Module):
    """Approximates multi-scale wavelet coefficients via pooled averages."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.wavelet_levels = config.wavelet_levels
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return concatenated multi-scale pooled features."""

        b, seq_len, dim = hidden_states.shape
        pooled_features = []
        signal = hidden_states.transpose(1, 2)  # (B, H, S)

        for level in range(self.wavelet_levels):
            kernel_size = 2 ** (level + 1)
            pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            pooled = pool(signal)
            if pooled.size(-1) < seq_len:
                pad = seq_len - pooled.size(-1)
                pooled = F.pad(pooled, (0, pad))
            pooled = pooled[..., :seq_len]
            pooled_features.append(pooled.transpose(1, 2))

        stacked = torch.stack(pooled_features, dim=-2).mean(dim=-2)
        return self.project(stacked)
