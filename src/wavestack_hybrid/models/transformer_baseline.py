"""Minimal transformer baseline for fair comparisons."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig


class TransformerBlock(nn.Module):
    """Causal transformer block."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.ln1(hidden)
        attn_out, _ = self.attn(hidden, hidden, hidden, attn_mask=attn_mask, need_weights=False)
        hidden = residual + self.dropout(attn_out)
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden


class TransformerBaseline(nn.Module):
    """GPT-style causal transformer baseline."""

    supports_lanes = False

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.architecture != "transformer":
            raise ValueError("TransformerBaseline requires ModelConfig.architecture='transformer'.")

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.positions = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_dim,
                    config.transformer.num_heads,
                    config.transformer.mlp_ratio,
                    config.transformer.dropout,
                    config.transformer.attn_dropout,
                )
                for _ in range(config.transformer.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.embeddings(input_ids) + self.positions(positions)
        attn_mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        hidden = self.ln(hidden)
        return self.lm_head(hidden)
