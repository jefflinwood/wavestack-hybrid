"""Fixed wavelet-inspired pooling layers."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ...config import DecompositionConfig


class WaveletDecomposition(nn.Module):
    """Multi-scale smoothing + detail extraction inspired by Haar wavelets."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.wavelet_levels = max(1, config.wavelet_levels)
        self.output_projection = nn.Linear(hidden_dim * 2 * self.wavelet_levels, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return concatenated low/high responses for each level."""

        _, seq_len, _ = hidden_states.shape
        signal = hidden_states.transpose(1, 2)  # (B, H, S)
        features = []

        for level in range(1, self.wavelet_levels + 1):
            kernel_size = 2 ** level
            padding = kernel_size // 2
            low = F.avg_pool1d(signal, kernel_size=kernel_size, stride=1, padding=padding)
            if low.size(-1) < seq_len:
                low = F.pad(low, (0, seq_len - low.size(-1)))
            low = low[..., :seq_len]
            detail = signal - low

            features.append(low.transpose(1, 2))
            features.append(detail.transpose(1, 2))

        concatenated = torch.cat(features, dim=-1)
        return self.output_projection(concatenated)
