"""Fourier decomposition utilities."""

from __future__ import annotations

import math

import torch
from torch import nn

from ...config import DecompositionConfig


class FourierDecomposition(nn.Module):
    """Extracts and reconstructs limited frequency components."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.num_freqs = max(1, config.num_freqs)
        self.freq_selection = config.freq_selection
        self.causal = config.causal
        if config.freq_selection == "learnable":
            self.freq_gates = nn.Parameter(torch.ones(self.num_freqs))
        else:
            self.register_buffer("freq_gates", torch.ones(self.num_freqs), persistent=False)
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return time-domain reconstruction from truncated spectrum."""

        seq_len = hidden_states.size(1)
        if self.causal:
            return self._causal_reconstruct(hidden_states, seq_len)
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

    def _causal_reconstruct(self, hidden_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Approximate causal band-limited reconstruction using prefix Fourier features."""

        x = hidden_states
        num_freqs = min(self.num_freqs, seq_len // 2 + 1)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.arange(num_freqs, device=x.device, dtype=x.dtype)
        angles = (2.0 * math.pi / max(1, seq_len)) * positions[:, None] * freqs[None, :]
        cos_basis = torch.cos(angles)
        sin_basis = torch.sin(angles)
        denom = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, seq_len, 1)

        reconstructed = torch.zeros_like(x)
        for freq_idx in range(num_freqs):
            cos_vec = cos_basis[:, freq_idx].view(1, seq_len, 1)
            sin_vec = sin_basis[:, freq_idx].view(1, seq_len, 1)
            coeff_cos = torch.cumsum(x * cos_vec, dim=1) / denom
            coeff_sin = torch.cumsum(x * sin_vec, dim=1) / denom
            gate = self.freq_gates[freq_idx]
            reconstructed = reconstructed + gate * (coeff_cos * cos_vec + coeff_sin * sin_vec)
        return self.project(reconstructed)
