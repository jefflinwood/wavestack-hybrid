import torch
import pytest

from wavestack_hybrid.config import ModelConfig
from wavestack_hybrid.models.wavestack import HybridWaveStack


def test_model_forward_runs():
    config = ModelConfig(vocab_size=128, hidden_dim=32, max_seq_len=16)
    model = HybridWaveStack(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)


def test_model_raises_on_lane_mismatch():
    config = ModelConfig(num_lanes=2)
    with pytest.raises(ValueError):
        HybridWaveStack(config)
