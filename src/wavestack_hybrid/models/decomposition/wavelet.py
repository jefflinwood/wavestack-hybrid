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
        self.max_levels = max(1, config.wavelet_levels)
        self.active_levels = self.max_levels
        self.causal = config.causal
        self.output_projection = nn.Linear(hidden_dim * 2 * self.max_levels, hidden_dim)

    def set_active_levels(self, levels: int) -> None:
        self.active_levels = max(1, min(int(levels), self.max_levels))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return concatenated low/high responses for each level."""

        _, seq_len, _ = hidden_states.shape
        signal = hidden_states.transpose(1, 2)  # (B, H, S)
        features = []

        for level in range(1, self.active_levels + 1):
            kernel_size = 2 ** level
            if self.causal:
                padded = F.pad(signal, (kernel_size - 1, 0))
                low = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
                low = low[..., :seq_len]
            else:
                padding = kernel_size // 2
                low = F.avg_pool1d(signal, kernel_size=kernel_size, stride=1, padding=padding)
                if low.size(-1) < seq_len:
                    low = F.pad(low, (0, seq_len - low.size(-1)))
                low = low[..., :seq_len]
            detail = signal - low

            features.append(low.transpose(1, 2))
            features.append(detail.transpose(1, 2))

        missing_levels = self.max_levels - self.active_levels
        if missing_levels > 0:
            zeros = torch.zeros_like(hidden_states)
            for _ in range(missing_levels):
                features.append(zeros)
                features.append(zeros)

        concatenated = torch.cat(features, dim=-1)
        return self.output_projection(concatenated)
