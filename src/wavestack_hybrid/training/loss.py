"""Loss helpers for WaveStack training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..config import TrainingConfig


def compute_multi_objective_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    reconstruction: Optional[torch.Tensor],
    lane_balance: Optional[torch.Tensor],
    config: TrainingConfig,
) -> torch.Tensor:
    """Combine autoregressive, reconstruction, and orthogonality terms."""

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    autoregressive = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    total = autoregressive * config.alpha_autoregressive

    if reconstruction is not None:
        recon_loss = F.mse_loss(reconstruction, labels.float())
        total = total + recon_loss * config.alpha_reconstruction

    if lane_balance is not None:
        orthogonality = lane_balance.pow(2).mean()
        total = total + orthogonality * config.alpha_orthogonality

    return total
