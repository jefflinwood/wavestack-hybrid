"""Fourier decomposition utilities."""

from __future__ import annotations

import torch
from torch import nn

from ...config import DecompositionConfig


class FourierDecomposition(nn.Module):
    """Extracts and reconstructs limited frequency components."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.num_freqs = max(1, config.num_freqs)
        self.freq_selection = config.freq_selection
        if config.freq_selection == "learnable":
            self.freq_gates = nn.Parameter(torch.ones(self.num_freqs))
        else:
            self.register_buffer("freq_gates", torch.ones(self.num_freqs), persistent=False)
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return time-domain reconstruction from truncated spectrum."""

        seq_len = hidden_states.size(1)
        freq_full_len = seq_len // 2 + 1
        freq = torch.fft.rfft(hidden_states, dim=1, norm="ortho")
        num_freqs = min(self.num_freqs, freq.size(1))
        truncated = freq[:, :num_freqs, :]
        gates = self.freq_gates[:num_freqs].view(1, num_freqs, 1)
        gated = truncated * gates

        freq_recon = torch.zeros(
            freq.size(0),
            freq_full_len,
            freq.size(2),
            dtype=freq.dtype,
            device=freq.device,
        )
        freq_recon[:, :num_freqs, :] = gated
        reconstructed = torch.fft.irfft(freq_recon, n=seq_len, dim=1, norm="ortho")
        return self.project(reconstructed)
