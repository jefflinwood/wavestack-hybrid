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

        if self.decay >= 1.0:
            kv = torch.einsum("bsf,bsd->bsfd", k_phi, v)
            kv_cum = torch.cumsum(kv, dim=1)
            k_cum = torch.cumsum(k_phi, dim=1)
        else:
            batch, seq_len, feat_dim = k_phi.size()
            kv_cum = torch.zeros(batch, seq_len, feat_dim, v.size(-1), device=v.device, dtype=v.dtype)
            k_cum = torch.zeros(batch, seq_len, feat_dim, device=v.device, dtype=v.dtype)
            kv_prev = torch.zeros(batch, feat_dim, v.size(-1), device=v.device, dtype=v.dtype)
            k_prev = torch.zeros(batch, feat_dim, device=v.device, dtype=v.dtype)
            decay = self.decay
            for t in range(seq_len):
                kv_prev = decay * kv_prev + k_phi[:, t, :].unsqueeze(-1) * v[:, t, :].unsqueeze(1)
                k_prev = decay * k_prev + k_phi[:, t, :]
                kv_cum[:, t, :, :] = kv_prev
                k_cum[:, t, :] = k_prev

        numerator = torch.einsum("bsf,bsfd->bsd", q_phi, kv_cum)
        denom = torch.einsum("bsf,bsf->bs", q_phi, k_cum).unsqueeze(-1)
        denom = denom.clamp_min(self.eps)
        return numerator / denom
