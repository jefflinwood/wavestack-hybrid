"""Fourier decomposition utilities."""

from __future__ import annotations

import torch
from torch import nn

from ...config import DecompositionConfig


class FourierDecomposition(nn.Module):
    """Extracts frequency coefficients using an rFFT."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.num_freqs = config.num_freqs
        self.freq_selection = config.freq_selection
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return truncated frequency-domain features."""

        freq = torch.fft.rfft(hidden_states, dim=1)
        freq = freq[:, : self.num_freqs, :]
        magnitude = torch.abs(freq)
        if self.freq_selection == "learnable":
            gates = torch.sigmoid(self.project.weight.mean(dim=0))[: magnitude.shape[-1]]
            magnitude = magnitude * gates.view(1, 1, -1)
        reduced = magnitude.mean(dim=1, keepdim=True)
        tiled = reduced.expand_as(hidden_states)
        return self.project(tiled)
