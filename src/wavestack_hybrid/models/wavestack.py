"""Complete WaveStack Hybrid model assembly."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ..config import ModelConfig
from .decomposition.chebyshev import ChebyshevDecomposition
from .decomposition.fourier import FourierDecomposition
from .decomposition.neural import NeuralDecomposition
from .decomposition.wavelet import WaveletDecomposition
from .embeddings import HybridEmbedding
from .mixing import LaneMixer
from .recomposition import RecompositionBundle


class HybridWaveStack(nn.Module):
    """High-level module glueing decompositions, recomposition, and mixing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = HybridEmbedding(config.vocab_size, config.hidden_dim, config.max_seq_len)
        self.chebyshev = ChebyshevDecomposition(config.hidden_dim, config.decomposition)
        self.fourier = FourierDecomposition(config.hidden_dim, config.decomposition)
        self.wavelet = WaveletDecomposition(config.hidden_dim, config.decomposition)
        self.neural = (
            NeuralDecomposition(config.hidden_dim, config.neural_decomp_layers)
            if not config.use_analytical_decomp
            else None
        )

        self.recomposition = RecompositionBundle(config.hidden_dim, config.recomposition)
        self.lane_names = list(self.recomposition.lanes.keys())
        if config.num_lanes != len(self.lane_names):
            raise ValueError(
                f"config.num_lanes={config.num_lanes} does not match available lanes {self.lane_names}"
            )
        self.mixer = LaneMixer(config.hidden_dim, len(self.lane_names), config.mixing_type)
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Forward pass returning vocabulary logits."""

        hidden = self.embeddings(input_ids)
        lane_features: Dict[str, torch.Tensor]

        if self.config.use_analytical_decomp:
            lane_features = {
                "poly": self.chebyshev(hidden),
                "trig": self.fourier(hidden),
                "wavelet": self.wavelet(hidden),
            }
        else:
            base = self.neural(hidden)
            lane_features = {
                "poly": base,
                "trig": base,
                "wavelet": base,
            }

        recomposed = self.recomposition(lane_features)
        lane_outputs = torch.stack([recomposed[name] for name in self.lane_names], dim=0)
        mixed = self.mixer(lane_outputs)

        if self.config.use_skip_connections:
            mixed = mixed + hidden

        mixed = self.ln(mixed)
        logits = self.lm_head(mixed)
        return logits
