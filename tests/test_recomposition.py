import torch

from wavestack_hybrid.config import RecompositionConfig
from wavestack_hybrid.models.recomposition import RecompositionBundle


def test_recomposition_bundle_returns_all_lanes():
    bundle = RecompositionBundle(hidden_dim=16, config=RecompositionConfig())
    features = {
        "poly": torch.randn(2, 8, 16),
        "trig": torch.randn(2, 8, 16),
        "wavelet": torch.randn(2, 8, 16),
    }
    outputs = bundle(features)
    assert set(outputs.keys()) == {"poly", "trig", "wavelet"}
    assert all(tensor.shape == (2, 8, 16) for tensor in outputs.values())
