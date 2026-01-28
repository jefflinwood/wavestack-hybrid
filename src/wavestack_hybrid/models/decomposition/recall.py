"""Kernelized recall decomposition lane."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ...config import DecompositionConfig


class RecallDecomposition(nn.Module):
    """Causal linear-time recall using kernelized prefix memory."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.feature_dim = max(8, int(config.recall_features))
        self.decay = float(config.recall_decay)
        self.eps = float(config.recall_epsilon)
        self.q_proj = nn.Linear(hidden_dim, self.feature_dim)
        self.k_proj = nn.Linear(hidden_dim, self.feature_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q_phi = self._phi(q)
        k_phi = self._phi(k)

        decay = float(self.decay)
        if decay <= 0.0:
            kv_cum = k_phi.unsqueeze(-1) * v.unsqueeze(2)
            k_cum = k_phi
        elif decay >= 0.999:
            kv = torch.einsum("bsf,bsd->bsfd", k_phi, v)
            kv_cum = torch.cumsum(kv, dim=1)
            k_cum = torch.cumsum(k_phi, dim=1)
        else:
            seq_len = k_phi.size(1)
            steps = torch.arange(seq_len, device=k_phi.device, dtype=k_phi.dtype)
            powers = torch.pow(torch.tensor(decay, device=k_phi.device, dtype=k_phi.dtype), steps)
            inv_powers = powers.reciprocal()
            k_scaled = k_phi * inv_powers.view(1, seq_len, 1)
            k_cum = torch.cumsum(k_scaled, dim=1) * powers.view(1, seq_len, 1)
            kv_scaled = k_scaled.unsqueeze(-1) * v.unsqueeze(2)
            kv_cum = torch.cumsum(kv_scaled, dim=1) * powers.view(1, seq_len, 1, 1)

        numerator = torch.einsum("bsf,bsfd->bsd", q_phi, kv_cum)
        denom = torch.einsum("bsf,bsf->bs", q_phi, k_cum).unsqueeze(-1)
        denom = denom.clamp_min(self.eps)
        return numerator / denom
