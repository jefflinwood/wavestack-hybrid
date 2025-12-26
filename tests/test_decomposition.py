import torch

from wavestack_hybrid.config import DecompositionConfig
from wavestack_hybrid.models.decomposition.chebyshev import ChebyshevDecomposition
from wavestack_hybrid.models.decomposition.fourier import FourierDecomposition
from wavestack_hybrid.models.decomposition.wavelet import WaveletDecomposition


def _dummy_states():
    return torch.randn(2, 16, 32)


def test_chebyshev_shapes():
    module = ChebyshevDecomposition(32, DecompositionConfig())
    output = module(_dummy_states())
    assert output.shape == (2, 16, 32)


def test_fourier_shapes():
    module = FourierDecomposition(32, DecompositionConfig(num_freqs=8))
    output = module(_dummy_states())
    assert output.shape == (2, 16, 32)


def test_wavelet_shapes():
    module = WaveletDecomposition(32, DecompositionConfig(wavelet_levels=2))
    output = module(_dummy_states())
    assert output.shape == (2, 16, 32)
