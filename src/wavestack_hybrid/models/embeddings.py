"""Token + positional embeddings used by WaveStack models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class HybridEmbedding(nn.Module):
    """Simple wrapper that exposes token + learned positional embeddings."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.position_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.hidden_dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.LongTensor, positions: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """Lookup embeddings and combine them."""

        if positions is None:
            positions = torch.arange(
                0,
                input_ids.size(1),
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.position_embed(positions)
        return token_embeddings + position_embeddings
