"""Chebyshev polynomial decomposition utilities."""

from __future__ import annotations

import torch
from torch import nn

from ...config import DecompositionConfig


class ChebyshevDecomposition(nn.Module):
    """Projects hidden states onto Chebyshev polynomials and reconstructs the signal."""

    def __init__(self, hidden_dim: int, config: DecompositionConfig):
        super().__init__()
        self.max_order = max(1, config.poly_order)
        self.active_order = self.max_order
        self.normalization = config.poly_normalization
        self.causal = config.causal
        self.project = nn.Linear(hidden_dim, hidden_dim)

    def set_active_order(self, order: int) -> None:
        self.active_order = max(1, min(int(order), self.max_order))

    def _basis(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, order: int
    ) -> torch.Tensor:
        """Compute Chebyshev basis functions evaluated at normalized positions."""

        positions = torch.linspace(-1.0, 1.0, steps=seq_len, device=device, dtype=dtype)
        basis = [torch.ones_like(positions)]
        if order > 1:
            basis.append(positions)
        for _ in range(2, order):
            basis.append(2 * positions * basis[-1] - basis[-2])
        return torch.stack(basis[:order], dim=-1)  # (seq_len, order)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return reconstruction from truncated Chebyshev series."""

        batch, seq_len, _ = hidden_states.shape
        x = hidden_states
        if self.normalization == "unit":
            denom = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x / denom

        order = self.active_order
        basis = self._basis(seq_len, x.device, x.dtype, order)  # (seq_len, order)
        if not self.causal:
            # Coefficients per polynomial order
            coeffs = torch.einsum("bsh,so->boh", x, basis) / seq_len
            reconstructed = torch.einsum("boh,so->bsh", coeffs, basis)
            return self.project(reconstructed)

        reconstructed = torch.zeros_like(x)
        denom = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
        for order_idx in range(order):
            basis_vec = basis[:, order_idx].view(1, seq_len, 1)
            weighted = x * basis_vec
            coeffs = torch.cumsum(weighted, dim=1) / denom
            reconstructed = reconstructed + coeffs * basis_vec
        return self.project(reconstructed)
