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
from .context_block import ContextBlock
from .mixing import LaneMixer
from .recomposition import RecompositionBundle


class HybridWaveStack(nn.Module):
    """High-level module glueing decompositions, recomposition, and mixing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        enabled_lanes = list(config.enabled_lanes)
        allowed_lanes = {"poly", "trig", "wavelet"}
        if not enabled_lanes:
            raise ValueError("At least one lane must be enabled.")
        invalid = [lane for lane in enabled_lanes if lane not in allowed_lanes]
        if invalid:
            raise ValueError(f"Invalid lane names in enabled_lanes: {invalid}")

        self.embeddings = HybridEmbedding(config.vocab_size, config.hidden_dim, config.max_seq_len)
        self.chebyshev = ChebyshevDecomposition(config.hidden_dim, config.decomposition)
        self.fourier = FourierDecomposition(config.hidden_dim, config.decomposition)
        self.wavelet = WaveletDecomposition(config.hidden_dim, config.decomposition)
        self.neural = (
            NeuralDecomposition(config.hidden_dim, config.neural_decomp_layers)
            if not config.use_analytical_decomp
            else None
        )

        self.lane_names = enabled_lanes
        self.recomposition = RecompositionBundle(config.hidden_dim, config.recomposition, self.lane_names)
        if config.num_lanes != len(self.lane_names):
            raise ValueError(
                f"config.num_lanes={config.num_lanes} does not match enabled lanes {self.lane_names}"
            )
        self.mixer = LaneMixer(config.hidden_dim, len(self.lane_names), config.mixing_type)
        self.pre_context_block = None
        self.post_context_block = None
        if config.context_block.enabled:
            position = config.context_block.position
            if position in {"pre", "both"}:
                self.pre_context_block = ContextBlock(config.hidden_dim, config.context_block)
            if position in {"post", "both"}:
                self.post_context_block = ContextBlock(config.hidden_dim, config.context_block)
        self.ln = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, return_lanes: bool = False):
        """Forward pass returning vocabulary logits (and lane outputs when requested)."""

        hidden = self.embeddings(input_ids)
        lane_features: Dict[str, torch.Tensor]

        if self.config.use_analytical_decomp:
            lane_features = {}
            if "poly" in self.lane_names:
                lane_features["poly"] = self.chebyshev(hidden)
            if "trig" in self.lane_names:
                lane_features["trig"] = self.fourier(hidden)
            if "wavelet" in self.lane_names:
                lane_features["wavelet"] = self.wavelet(hidden)
        else:
            base = self.neural(hidden)
            lane_features = {name: base for name in self.lane_names}

        recomposed = self.recomposition(lane_features)
        if self.pre_context_block is not None:
            recomposed = {
                name: self.pre_context_block(recomposed[name]) for name in self.lane_names
            }
        lane_outputs = torch.stack([recomposed[name] for name in self.lane_names], dim=0)
        mixed = self.mixer(lane_outputs)
        if self.post_context_block is not None:
            mixed = self.post_context_block(mixed)

        if self.config.use_skip_connections:
            mixed = mixed + hidden

        mixed = self.ln(mixed)
        logits = self.lm_head(mixed)
        if return_lanes:
            return logits, lane_outputs
        return logits
