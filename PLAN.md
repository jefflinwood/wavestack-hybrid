December 12-26-2025

Execution Plan for Coding Agent: WaveStack 2.0 Hybrid Implementation
Project Structure
wavestack_hybrid/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration dataclasses
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Token + positional embeddings
│   │   ├── decomposition/
│   │   │   ├── __init__.py
│   │   │   ├── chebyshev.py   # Analytical polynomial decomposition
│   │   │   ├── fourier.py     # Analytical FFT decomposition
│   │   │   ├── wavelet.py     # Fixed wavelet decomposition
│   │   │   └── neural.py      # Neural baseline decomposition
│   │   ├── recomposition.py   # Learned recomposition networks
│   │   ├── mixing.py          # Lane mixing layer
│   │   └── wavestack.py       # Complete model assembly
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   ├── loss.py            # Multi-objective loss functions
│   │   └── metrics.py         # Lane specialization metrics
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # TinyStories/FineWeb dataset loaders
│   │   └── tokenizer.py       # Tokenizer wrapper
│   └── analysis/
│       ├── __init__.py
│       ├── interpretability.py # Lane attribution analysis
│       ├── visualization.py    # Coefficient plotting
│       └── gradient_tracker.py # Gradient flow monitoring
├── experiments/
│   ├── exp1_expressivity/
│   │   ├── config_A_neural_50m.yaml
│   │   ├── config_B_hybrid_12m.yaml
│   │   ├── config_C_hybrid_50m.yaml
│   │   ├── config_D_neural_12m.yaml
│   │   └── run_experiment.py
│   ├── exp2_adaptation/
│   │   ├── pretrain_config.yaml
│   │   ├── finetune_config.yaml
│   │   └── run_experiment.py
│   └── exp3_gradients/
│       ├── config.yaml
│       └── run_experiment.py
├── tests/
│   ├── test_decomposition.py
│   ├── test_recomposition.py
│   └── test_model.py
├── scripts/
│   ├── download_data.py
│   └── verify_setup.py
├── requirements.txt
├── README.md
└── setup.py
Phase 1: Core Infrastructure Setup (Days 1-2)
Task 1.1: Project Scaffolding
File: setup.py, requirements.txt, README.md

Instructions:

python
# requirements.txt should include:
"""
torch>=2.0.0
numpy>=1.24.0
tiktoken>=0.5.0
pywt>=1.4.1  # For wavelet transforms
datasets>=2.14.0  # HuggingFace datasets
wandb>=0.15.0  # Experiment tracking
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
PyYAML>=6.0
tqdm>=4.65.0
"""

# setup.py - standard Python package setup
# README.md - project overview, architecture diagram, quick start
Acceptance Criteria:

 pip install -e . successfully installs package
 All dependencies resolve without conflicts
 Project imports work: from wavestack_hybrid.models import HybridWaveStack
Task 1.2: Configuration System
File: src/config.py

Instructions:

python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class DecompositionConfig:
    """Configuration for analytical decomposition layers"""
    # Chebyshev (polynomial) settings
    poly_order: int = 32
    poly_normalization: Literal['unit', 'standard'] = 'unit'
    
    # Fourier (trigonometric) settings
    num_freqs: int = 64
    freq_selection: Literal['learnable', 'fixed'] = 'learnable'
    
    # Wavelet settings
    wavelet_type: Literal['haar', 'db4', 'db8', 'sym4'] = 'db4'
    wavelet_levels: int = 3
    scale_selection: Literal['learnable', 'fixed'] = 'learnable'

@dataclass
class RecompositionConfig:
    """Configuration for learned recomposition networks"""
    depth: Literal['shallow', 'standard', 'deep'] = 'standard'
    poly_capacity: float = 1.0
    trig_capacity: float = 1.0
    wavelet_capacity: float = 1.5
    dropout: float = 0.1
    activation: Literal['gelu', 'relu', 'swish'] = 'gelu'

@dataclass
class ModelConfig:
    """Complete model configuration"""
    # Architecture type
    use_analytical_decomp: bool = True  # False = neural baseline
    
    # Model dimensions
    vocab_size: int = 50257  # GPT-2 tokenizer
    hidden_dim: int = 512
    max_seq_len: int = 512
    
    # Decomposition settings
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    
    # Recomposition settings
    recomposition: RecompositionConfig = field(default_factory=RecompositionConfig)
    
    # Mixing settings
    mixing_type: Literal['gated', 'attention', 'mlp'] = 'gated'
    num_lanes: int = 3
    
    # Training enhancements
    use_skip_connections: bool = False
    use_multi_objective_loss: bool = False
    
    # Neural baseline settings (when use_analytical_decomp=False)
    neural_decomp_layers: int = 3
    
    def get_param_count(self) -> int:
        """Estimate parameter count for this configuration"""
        # TODO: Implement parameter counting logic
        pass

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Loss settings (multi-objective)
    alpha_autoregressive: float = 0.7
    alpha_reconstruction: float = 0.2
    alpha_orthogonality: float = 0.1
    
    # Evaluation
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Logging
    log_interval: int = 100
    use_wandb: bool = True
    project_name: str = "wavestack_hybrid"
    
    # Hardware
    device: str = "mps"  # or "cuda" or "cpu"
    mixed_precision: bool = False  # bf16/fp16 training

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load config from YAML file"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
Acceptance Criteria:

 Can create config objects programmatically
 Can load/save configs from/to YAML
 All experiment configs (A, B, C, D) can be represented
 Parameter count estimation works (at least approximately)
Phase 2: Analytical Decomposition Layers (Days 3-5)
Task 2.1: Chebyshev Polynomial Decomposition
File: src/models/decomposition/chebyshev.py

Instructions:

python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebyshevDecomposition(nn.Module):
    """
    Analytical Chebyshev polynomial decomposition.
    Zero learned parameters in decomposition path.
    
    Mathematical foundation:
    - T_0(x) = 1
    - T_1(x) = x  
    - T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    
    Projection: c_k = <f, T_k> / ||T_k||^2
    """
    
    def __init__(self, hidden_dim: int, poly_order: int = 32, 
                 normalization: str = 'unit'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.poly_order = poly_order
        self.normalization = normalization
        
        # No learned parameters!
        # Register polynomial norms for proper projection
        if normalization == 'unit':
            # ||T_k||^2 = pi/2 for k>0, pi for k=0
            norms = torch.ones(poly_order)
            norms[0] = torch.tensor(torch.pi)
            norms[1:] = torch.tensor(torch.pi / 2)
            self.register_buffer('poly_norms', norms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) embeddings
        Returns:
            coeffs: (batch, seq_len, poly_order) Chebyshev coefficients
        """
        batch, seq_len, dim = x.shape
        
        # Step 1: Normalize to [-1, 1] domain
        x_norm = self._normalize_to_domain(x)
        
        # Step 2: Compute Chebyshev coefficients analytically
        coeffs = self._compute_coefficients(x_norm)
        
        return coeffs
    
    def _normalize_to_domain(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map arbitrary embeddings to [-1, 1] for Chebyshev stability.
        Use per-feature min-max normalization.
        """
        # Per-feature normalization (across batch and sequence)
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        
        # Map to [-1, 1]
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        
        # Clamp to ensure stability
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        
        return x_norm
    
    def _compute_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev coefficients using recurrence relation.
        This is the core analytical decomposition.
        """
        batch, seq_len, dim = x.shape
        coeffs = torch.zeros(batch, seq_len, self.poly_order, 
                            device=x.device, dtype=x.dtype)
        
        # Initialize for recurrence
        T_prev2 = None
        T_prev = None
        
        for k in range(self.poly_order):
            # Evaluate T_k(x) using recurrence
            if k == 0:
                T_k = torch.ones_like(x)
            elif k == 1:
                T_k = x
            else:
                T_k = 2 * x * T_prev - T_prev2
            
            # Compute coefficient via discrete orthogonal projection
            # c_k = <f, T_k> / ||T_k||^2
            # For discrete case: <f, T_k> ≈ mean(f * T_k) * N
            inner_product = (x * T_k).mean(dim=-1)  # (batch, seq_len)
            
            if self.normalization == 'unit':
                coeffs[:, :, k] = inner_product / self.poly_norms[k]
            else:
                coeffs[:, :, k] = inner_product
            
            # Update for next iteration
            T_prev2 = T_prev
            T_prev = T_k
        
        return coeffs
    
    def reconstruct(self, coeffs: torch.Tensor, 
                   num_points: int = None) -> torch.Tensor:
        """
        Optional: Reconstruct signal from Chebyshev coefficients.
        Useful for analysis/visualization.
        
        Args:
            coeffs: (batch, seq_len, poly_order)
            num_points: Number of evaluation points (default: hidden_dim)
        Returns:
            reconstruction: (batch, seq_len, num_points)
        """
        if num_points is None:
            num_points = self.hidden_dim
        
        batch, seq_len, order = coeffs.shape
        
        # Create evaluation points in [-1, 1]
        x_eval = torch.linspace(-1, 1, num_points, device=coeffs.device)
        x_eval = x_eval.view(1, 1, -1).expand(batch, seq_len, -1)
        
        # Reconstruct using Chebyshev basis
        reconstruction = torch.zeros_like(x_eval)
        T_prev2 = None
        T_prev = None
        
        for k in range(order):
            if k == 0:
                T_k = torch.ones_like(x_eval)
            elif k == 1:
                T_k = x_eval
            else:
                T_k = 2 * x_eval * T_prev - T_prev2
            
            reconstruction += coeffs[:, :, k:k+1] * T_k
            
            T_prev2 = T_prev
            T_prev = T_k
        
        return reconstruction

# Unit tests to implement
def test_chebyshev_decomposition():
    """Test basic functionality"""
    batch, seq_len, hidden_dim = 2, 10, 512
    poly_order = 32
    
    decomp = ChebyshevDecomposition(hidden_dim, poly_order)
    
    # Test forward pass
    x = torch.randn(batch, seq_len, hidden_dim)
    coeffs = decomp(x)
    
    assert coeffs.shape == (batch, seq_len, poly_order)
    assert not torch.isnan(coeffs).any()
    assert not torch.isinf(coeffs).any()
    
    # Test reconstruction (should approximately recover input)
    x_recon = decomp.reconstruct(coeffs, num_points=hidden_dim)
    # Note: Won't be perfect due to finite polynomial order
    
    print("✓ Chebyshev decomposition test passed")

def test_chebyshev_orthogonality():
    """Verify Chebyshev polynomials are orthogonal"""
    decomp = ChebyshevDecomposition(512, 16)
    
    # Generate test signal
    x = torch.randn(1, 1, 512)
    coeffs = decomp(x)
    
    # Coefficients should be relatively uncorrelated
    # (not a strict test, but sanity check)
    corr = torch.corrcoef(coeffs[0, 0])
    off_diagonal = corr - torch.eye(16)
    
    assert off_diagonal.abs().mean() < 0.5  # Loose bound
    print("✓ Chebyshev orthogonality test passed")
Acceptance Criteria:

 Forward pass produces correct shape output
 No NaN or Inf values in coefficients
 Reconstruction approximately recovers input (for high poly_order)
 Zero learned parameters in decomposition
 Tests pass
Task 2.2: Fourier (FFT) Decomposition
File: src/models/decomposition/fourier.py

Instructions:

python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierDecomposition(nn.Module):
    """
    Analytical Fourier decomposition using FFT.
    
    Only learned parameters: frequency selection mask (optional).
    Decomposition itself is analytical (FFT algorithm).
    """
    
    def __init__(self, hidden_dim: int, num_freqs: int = 64,
                 freq_selection: str = 'learnable'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.freq_selection = freq_selection
        
        # Learnable frequency selection mask
        if freq_selection == 'learnable':
            self.freq_mask = nn.Parameter(torch.ones(num_freqs))
        else:
            self.register_buffer('freq_mask', torch.ones(num_freqs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) embeddings
        Returns:
            features: (batch, seq_len, num_freqs * 2) 
                      [real_part, imag_part concatenated]
        """
        batch, seq_len, dim = x.shape
        
        # Step 1: Compute FFT along hidden dimension (analytical!)
        # Use rfft for real-valued inputs (more efficient)
        fft_result = torch.fft.rfft(x, dim=-1)  
        # Shape: (batch, seq_len, hidden_dim//2 + 1)
        
        # Step 2: Select top-k frequencies
        fft_truncated = fft_result[..., :self.num_freqs]
        
        # Step 3: Apply learnable frequency selection
        fft_selected = fft_truncated * self.freq_mask.view(1, 1, -1)
        
        # Step 4: Convert complex to real features
        # Option A: Real and imaginary parts
        real_part = fft_selected.real
        imag_part = fft_selected.imag
        
        # Option B: Magnitude and phase (alternative)
        # magnitude = torch.abs(fft_selected)
        # phase = torch.angle(fft_selected)
        
        # Concatenate real and imaginary
        features = torch.cat([real_part, imag_part], dim=-1)
        # Shape: (batch, seq_len, num_freqs * 2)
        
        return features
    
    def get_power_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute power spectrum for analysis/visualization.
        
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            power: (batch, seq_len, num_freqs) power at each frequency
        """
        fft_result = torch.fft.rfft(x, dim=-1)
        fft_truncated = fft_result[..., :self.num_freqs]
        
        # Power = |F[k]|^2
        power = torch.abs(fft_truncated) ** 2
        
        return power
    
    def reconstruct(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct signal from Fourier features (for analysis).
        
        Args:
            features: (batch, seq_len, num_freqs * 2)
        Returns:
            reconstruction: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = features.shape
        
        # Split back into real and imaginary
        real_part = features[..., :self.num_freqs]
        imag_part = features[..., self.num_freqs:]
        
        # Reconstruct complex FFT coefficients
        fft_complex = torch.complex(real_part, imag_part)
        
        # Pad to full FFT length if needed
        if fft_complex.shape[-1] < self.hidden_dim // 2 + 1:
            pad_size = self.hidden_dim // 2 + 1 - fft_complex.shape[-1]
            fft_complex = F.pad(fft_complex, (0, pad_size))
        
        # Inverse FFT
        reconstruction = torch.fft.irfft(fft_complex, n=self.hidden_dim, dim=-1)
        
        return reconstruction

# Unit tests
def test_fourier_decomposition():
    """Test basic FFT functionality"""
    batch, seq_len, hidden_dim = 2, 10, 512
    num_freqs = 64
    
    decomp = FourierDecomposition(hidden_dim, num_freqs)
    
    # Test forward pass
    x = torch.randn(batch, seq_len, hidden_dim)
    features = decomp(x)
    
    assert features.shape == (batch, seq_len, num_freqs * 2)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    
    # Test power spectrum
    power = decomp.get_power_spectrum(x)
    assert power.shape == (batch, seq_len, num_freqs)
    assert (power >= 0).all()  # Power is always non-negative
    
    print("✓ Fourier decomposition test passed")

def test_fourier_parseval():
    """Verify Parseval's theorem (energy conservation)"""
    decomp = FourierDecomposition(512, 256)  # Use many frequencies
    
    x = torch.randn(1, 1, 512)
    
    # Energy in time domain
    time_energy = (x ** 2).sum()
    
    # Energy in frequency domain (Parseval's theorem)
    fft_result = torch.fft.rfft(x, dim=-1)
    freq_energy = (torch.abs(fft_result) ** 2).sum()
    
    # Should be approximately equal (up to normalization)
    ratio = time_energy / freq_energy
    assert 0.9 < ratio < 1.1
    
    print("✓ Parseval's theorem test passed")
Acceptance Criteria:

 FFT decomposition produces correct output shape
 Power spectrum is non-negative
 Parseval's theorem holds (energy conservation)
 Reconstruction approximately recovers input
 Only freq_mask is learned (if freq_selection='learnable')
 Tests pass
Task 2.3: Wavelet Decomposition
File: src/models/decomposition/wavelet.py

Instructions:

python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class WaveletDecomposition(nn.Module):
    """
    Fixed wavelet decomposition using PyWavelets library.
    
    Learned parameters: Only scale weighting (optional).
    Wavelet filters themselves are fixed.
    """
    
    def __init__(self, hidden_dim: int, wavelet: str = 'db4', 
                 levels: int = 3, scale_selection: str = 'learnable'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wavelet_name = wavelet
        self.levels = levels
        self.scale_selection = scale_selection
        
        # Get wavelet filter coefficients (FIXED, not learned)
        wavelet_obj = pywt.Wavelet(wavelet)
        
        # Register filter banks as buffers (not parameters)
        self.register_buffer('dec_lo', 
                           torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32))
        self.register_buffer('dec_hi',
                           torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32))
        self.register_buffer('rec_lo',
                           torch.tensor(wavelet_obj.rec_lo, dtype=torch.float32))
        self.register_buffer('rec_hi',
                           torch.tensor(wavelet_obj.rec_hi, dtype=torch.float32))
        
        # Learnable scale weights
        num_scales = levels + 1  # approximation + detail at each level
        if scale_selection == 'learnable':
            self.scale_weights = nn.Parameter(torch.ones(num_scales))
        else:
            self.register_buffer('scale_weights', torch.ones(num_scales))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) embeddings
        Returns:
            features: (batch, seq_len, (levels+1) * approx_hidden_dim)
                      Multi-scale wavelet coefficients
        """
        batch, seq_len, dim = x.shape
        
        # Apply DWT along hidden dimension
        # We'll process each sequence position independently
        all_coeffs = []
        current_signal = x
        
        for level in range(self.levels):
            # Convolve with low-pass (approximation) and high-pass (detail)
            approx, detail = self._dwt_step(current_signal)
            
            # Apply scale weight to detail coefficients
            detail = detail * self.scale_weights[level]
            all_coeffs.append(detail)
            
            # Continue decomposition on approximation
            current_signal = approx
        
        # Add final approximation with its scale weight
        current_signal = current_signal * self.scale_weights[-1]
        all_coeffs.append(current_signal)
        
        # Reverse to go from coarse to fine
        all_coeffs = all_coeffs[::-1]
        
        # Interpolate all scales to same length and concatenate
        all_coeffs_aligned = []
        for coeff in all_coeffs:
            # Interpolate to original sequence length
            coeff_interp = F.interpolate(
                coeff.transpose(1, 2),  # (batch, hidden_dim, seq_len_scaled)
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)
            all_coeffs_aligned.append(coeff_interp)
        
        # Concatenate along feature dimension
        features = torch.cat(all_coeffs_aligned, dim=-1)
        # Shape: (batch, seq_len, (levels+1) * hidden_dim)
        
        return features
    
    def _dwt_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single level of DWT using convolution + downsampling.
        
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            approx: (batch, seq_len//2, hidden_dim) low-frequency
            detail: (batch, seq_len//2, hidden_dim) high-frequency
        """
        batch, seq_len, dim = x.shape
        
        # Transpose for conv1d: (batch, hidden_dim, seq_len)
        x_t = x.transpose(1, 2)
        
        # Prepare filters for grouped convolution
        # Each hidden dimension gets its own filter
        filter_len = len(self.dec_lo)
        
        # Low-pass filter (approximation)
        lo_filter = self.dec_lo.view(1, 1, -1).expand(dim, 1, -1)
        approx = F.conv1d(x_t, lo_filter, groups=dim, padding=filter_len//2)
        
        # High-pass filter (detail)
        hi_filter = self.dec_hi.view(1, 1, -1).expand(dim, 1, -1)
        detail = F.conv1d(x_t, hi_filter, groups=dim, padding=filter_len//2)
        
        # Downsample by 2 (standard DWT)
        approx = approx[..., ::2]
        detail = detail[..., ::2]
        
        # Transpose back: (batch, seq_len//2, hidden_dim)
        approx = approx.transpose(1, 2)
        detail = detail.transpose(1, 2)
        
        return approx, detail
    
    def get_scale_energies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy at each wavelet scale (for analysis).
        
        Returns:
            energies: (batch, seq_len, levels+1) energy per scale
        """
        batch, seq_len, dim = x.shape
        
        energies = torch.zeros(batch, seq_len, self.levels + 1, device=x.device)
        current_signal = x
        
        for level in range(self.levels):
            approx, detail = self._dwt_step(current_signal)
            
            # Energy = sum of squares
            detail_interp = F.interpolate(
                detail.transpose(1, 2), size=seq_len, mode='linear'
            ).transpose(1, 2)
            
            energies[:, :, level] = (detail_interp ** 2).sum(dim=-1)
            current_signal = approx
        
        # Final approximation energy
        approx_interp = F.interpolate(
            current_signal.transpose(1, 2), size=seq_len, mode='linear'
        ).transpose(1, 2)
        energies[:, :, -1] = (approx_interp ** 2).sum(dim=-1)
        
        return energies

# Unit tests
def test_wavelet_decomposition():
    """Test basic wavelet functionality"""
    batch, seq_len, hidden_dim = 2, 128, 512  # seq_len must be power of 2
    levels = 3
    
    decomp = WaveletDecomposition(hidden_dim, wavelet='db4', levels=levels)
    
    # Test forward pass
    x = torch.randn(batch, seq_len, hidden_dim)
    features = decomp(x)
    
    expected_dim = (levels + 1) * hidden_dim
    assert features.shape == (batch, seq_len, expected_dim)
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()
    
    print("✓ Wavelet decomposition test passed")

def test_wavelet_energy_conservation():
    """Verify energy is approximately conserved across scales"""
    decomp = WaveletDecomposition(512, 'haar', levels=2)
    
    x = torch.randn(1, 128, 512)
    
    # Input energy
    input_energy = (x ** 2).sum()
    
    # Multi-scale energies
    scale_energies = decomp.get_scale_energies(x)
    total_scale_energy = scale_energies.sum()
    
    # Should be approximately equal
    ratio = input_energy / total_scale_energy
    assert 0.8 < ratio < 1.2  # Loose bound due to interpolation
    
    print("✓ Wavelet energy conservation test passed")
Acceptance Criteria:

 Wavelet decomposition produces correct output shape
 Works with different wavelet families (haar, db4, db8)
 Energy approximately conserved across scales
 Only scale_weights are learned (if scale_selection='learnable')
 Tests pass
Task 2.4: Neural Baseline Decomposition (for comparison)
File: src/models/decomposition/neural.py

Instructions:

python
import torch
import torch.nn as nn

class NeuralDecomposition(nn.Module):
    """
    Fully learned neural decomposition (baseline for comparison).
    This is what Config A and D use instead of analytical decomposition.
    """
    
    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        layers = []
        in_dim = hidden_dim
        
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            features: (batch, seq_len, output_dim)
        """
        return self.network(x)
Acceptance Criteria:

 Simple feedforward network
 Matches interface of analytical decomposition layers
 Tests pass
Phase 3: Recomposition Networks (Day 6)
Task 3.1: Lane Recomposition
File: src/models/recomposition.py

Instructions:

python
import torch
import torch.nn as nn

class LaneRecomposition(nn.Module):
    """
    Learned recomposition network.
    Maps decomposition coefficients back to hidden dimension.
    
    This is where expressivity lives in the hybrid model.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 depth: str = 'standard', capacity_multiplier: float = 1.0,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        # Depth configurations
        depth_map = {
            'shallow': 1,
            'standard': 2,
            'deep': 4,
        }
        num_layers = depth_map[depth]
        
        # Activation functions
        activation_map = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
        }
        act_fn = activation_map[activation]
        
        # Build network
        intermediate_dim = int(hidden_dim * capacity_multiplier)
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            out_dim = intermediate_dim if i < num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(act_fn)
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.Dropout(dropout))
            
            in_dim = out_dim
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) coefficients from decomposition
        Returns:
            features: (batch, seq_len, hidden_dim)
        """
        return self.network(x)

# Test
def test_recomposition():
    """Test recomposition with different depths"""
    batch, seq_len = 2, 10
    input_dim, hidden_dim = 32, 512
    
    for depth in ['shallow', 'standard', 'deep']:
        recomp = LaneRecomposition(input_dim, hidden_dim, depth=depth)
        
        x = torch.randn(batch, seq_len, input_dim)
        out = recomp(x)
        
        assert out.shape == (batch, seq_len, hidden_dim)
        print(f"✓ Recomposition test passed for depth={depth}")
Acceptance Criteria:

 Supports different depth configurations
 Supports capacity multipliers for parameter allocation
 Tests pass for all depths
Phase 4: Model Assembly (Day 7)
Task 4.1: Embeddings
File: src/models/embeddings.py

Instructions:

python
import torch
import torch.nn as nn

class WaveStackEmbedding(nn.Module):
    """Token and positional embeddings"""
    
    def __init__(self, vocab_size: int, hidden_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, hidden_dim)
        """
        batch, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embed(input_ids)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        
        # Add together
        embeddings = token_emb + pos_emb
        
        return embeddings
Acceptance Criteria:

 Standard embedding implementation
 Tests pass
Task 4.2: Lane Mixing
File: src/models/mixing.py

Instructions:

python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossLaneMixing(nn.Module):
    """
    Mixing layer to integrate information across lanes.
    """
    
    def __init__(self, hidden_dim: int, num_lanes: int = 3, 
                 mixing_type: str = 'gated'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lanes = num_lanes
        self.mixing_type = mixing_type
        
        if mixing_type == 'gated':
            # Gated mixing: learn importance of each lane per position
            self.gate = nn.Linear(num_lanes * hidden_dim, num_lanes)
            self.output_proj = nn.Linear(num_lanes * hidden_dim, hidden_dim)
            
        elif mixing_type == 'attention':
            # Cross-lane attention
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=8, batch_first=True
            )
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            
        elif mixing_type == 'mlp':
            # Simple MLP mixing
            self.mlp = nn.Sequential(
                nn.Linear(num_lanes * hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
    
    def forward(self, lane_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            lane_features: List of (batch, seq_len, hidden_dim) tensors
        Returns:
            mixed: (batch, seq_len, hidden_dim)
        """
        if self.mixing_type == 'gated':
            return self._gated_mixing(lane_features)
        elif self.mixing_type == 'attention':
            return self._attention_mixing(lane_features)
        else:  # mlp
            return self._mlp_mixing(lane_features)
    
    def _gated_mixing(self, lane_features: list[torch.Tensor]) -> torch.Tensor:
        # Stack lanes
        stacked = torch.stack(lane_features, dim=-2)
        # (batch, seq_len, num_lanes, hidden_dim)
        
        # Flatten for gate computation
        flat = stacked.flatten(-2, -1)
        # (batch, seq_len, num_lanes * hidden_dim)
        
        # Compute gates
        gates = F.softmax(self.gate(flat), dim=-1)
        # (batch, seq_len, num_lanes)
        
        # Apply gates
        weighted = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        # (batch, seq_len, hidden_dim)
        
        # Output projection
        output = self.output_proj(flat)
        
        return output
    
    def _attention_mixing(self, lane_features: list[torch.Tensor]) -> torch.Tensor:
        # Use first lane as query, all lanes as key/value
        query = lane_features[0]
        key_value = torch.cat(lane_features, dim=1)
        # (batch, seq_len * num_lanes, hidden_dim)
        
        attended, _ = self.cross_attn(query, key_value, key_value)
        output = self.output_proj(attended)
        
        return output
    
    def _mlp_mixing(self, lane_features: list[torch.Tensor]) -> torch.Tensor:
        # Simple concatenate and project
        concat = torch.cat(lane_features, dim=-1)
        # (batch, seq_len, num_lanes * hidden_dim)
        
        output = self.mlp(concat)
        
        return output

# Test
def test_mixing():
    """Test different mixing strategies"""
    batch, seq_len, hidden_dim = 2, 10, 512
    num_lanes = 3
    
    lane_features = [
        torch.randn(batch, seq_len, hidden_dim) for _ in range(num_lanes)
    ]
    
    for mixing_type in ['gated', 'attention', 'mlp']:
        mixer = CrossLaneMixing(hidden_dim, num_lanes, mixing_type)
        
        output = mixer(lane_features)
        
        assert output.shape == (batch, seq_len, hidden_dim)
        print(f"✓ Mixing test passed for type={mixing_type}")
Acceptance Criteria:

 Supports gated, attention, and MLP mixing
 Tests pass for all types
Task 4.3: Complete Model
File: src/models/wavestack.py

Instructions:

python
import torch
import torch.nn as nn
from typing import Dict, Optional

from .embeddings import WaveStackEmbedding
from .decomposition.chebyshev import ChebyshevDecomposition
from .decomposition.fourier import FourierDecomposition
from .decomposition.wavelet import WaveletDecomposition
from .decomposition.neural import NeuralDecomposition
from .recomposition import LaneRecomposition
from .mixing import CrossLaneMixing

class HybridWaveStack(nn.Module):
    """
    Complete WaveStack 2.0 model with hybrid symbolic-neural architecture.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = WaveStackEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            max_seq_len=config.max_seq_len
        )
        
        # Build lanes based on config
        self._build_lanes(config)
        
        # Lane mixing
        self.lane_mixing = CrossLaneMixing(
            hidden_dim=config.hidden_dim,
            num_lanes=config.num_lanes,
            mixing_type=config.mixing_type
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie embeddings to output (optional but common)
        # self.output_proj.weight = self.embeddings.token_embed.weight
    
    def _build_lanes(self, config):
        """Build decomposition and recomposition for each lane"""
        if config.use_analytical_decomp:
            # === HYBRID APPROACH ===
            
            # Polynomial lane
            self.poly_decomp = ChebyshevDecomposition(
                hidden_dim=config.hidden_dim,
                poly_order=config.decomposition.poly_order,
                normalization=config.decomposition.poly_normalization
            )
            self.poly_recomp = LaneRecomposition(
                input_dim=config.decomposition.poly_order,
                hidden_dim=config.hidden_dim,
                depth=config.recomposition.depth,
                capacity_multiplier=config.recomposition.poly_capacity,
                dropout=config.recomposition.dropout
            )
            
            # Trigonometric lane
            self.trig_decomp = FourierDecomposition(
                hidden_dim=config.hidden_dim,
                num_freqs=config.decomposition.num_freqs,
                freq_selection=config.decomposition.freq_selection
            )
            self.trig_recomp = LaneRecomposition(
                input_dim=config.decomposition.num_freqs * 2,
                hidden_dim=config.hidden_dim,
                depth=config.recomposition.depth,
                capacity_multiplier=config.recomposition.trig_capacity,
                dropout=config.recomposition.dropout
            )
            
            # Wavelet lane
            self.wavelet_decomp = WaveletDecomposition(
                hidden_dim=config.hidden_dim,
                wavelet=config.decomposition.wavelet_type,
                levels=config.decomposition.wavelet_levels,
                scale_selection=config.decomposition.scale_selection
            )
            wavelet_output_dim = (config.decomposition.wavelet_levels + 1) * config.hidden_dim
            self.wavelet_recomp = LaneRecomposition(
                input_dim=wavelet_output_dim,
                hidden_dim=config.hidden_dim,
                depth=config.recomposition.depth,
                capacity_multiplier=config.recomposition.wavelet_capacity,
                dropout=config.recomposition.dropout
            )
            
        else:
            # === NEURAL BASELINE ===
            output_dim = config.hidden_dim  # Same for all lanes
            
            self.poly_decomp = NeuralDecomposition(
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.neural_decomp_layers
            )
            self.poly_recomp = nn.Identity()  # No recomposition needed
            
            self.trig_decomp = NeuralDecomposition(
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.neural_decomp_layers
            )
            self.trig_recomp = nn.Identity()
            
            self.wavelet_decomp = NeuralDecomposition(
                hidden_dim=config.hidden_dim,
                output_dim=output_dim,
                num_layers=config.neural_decomp_layers
            )
            self.wavelet_recomp = nn.Identity()
        
        # Optional skip connections
        if config.use_skip_connections:
            self.skip_alpha = nn.Parameter(torch.tensor(0.9))
            self.skip_path = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, input_ids: torch.Tensor, 
                return_lane_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            return_lane_features: If True, return intermediate lane features
        Returns:
            dict with 'logits' and optionally 'lane_features'
        """
        # Embed
        x = self.embeddings(input_ids)
        
        # === Lane processing ===
        # Decomposition (analytical or neural)
        poly_coeffs = self.poly_decomp(x)
        trig_coeffs = self.trig_decomp(x)
        wavelet_coeffs = self.wavelet_decomp(x)
        
        # Recomposition (learned)
        poly_features = self.poly_recomp(poly_coeffs)
        trig_features = self.trig_recomp(trig_coeffs)
        wavelet_features = self.wavelet_recomp(wavelet_coeffs)
        
        # Optional skip connection
        if hasattr(self, 'skip_alpha'):
            skip_features = self.skip_path(x)
            poly_features = self.skip_alpha * poly_features + (1 - self.skip_alpha) * skip_features
        
        # Mix lanes
        lane_features_list = [poly_features, trig_features, wavelet_features]
        mixed = self.lane_mixing(lane_features_list)
        
        # Output projection
        logits = self.output_proj(mixed)
        
        output = {'logits': logits}
        
        if return_lane_features:
            output['lane_features'] = {
                'polynomial': poly_features,
                'trigonometric': trig_features,
                'wavelet': wavelet_features,
            }
            output['coefficients'] = {
                'polynomial': poly_coeffs,
                'trigonometric': trig_coeffs,
                'wavelet': wavelet_coeffs,
            }
        
        return output
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Count parameters"""
        if non_embedding:
            return sum(p.numel() for n, p in self.named_parameters() 
                      if 'embeddings' not in n and p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Test
def test_hybrid_wavestack():
    """Test complete model"""
    from ..config import ModelConfig, DecompositionConfig, RecompositionConfig
    
    config = ModelConfig(
        vocab_size=1000,
        hidden_dim=256,
        max_seq_len=128,
        use_analytical_decomp=True,
        decomposition=DecompositionConfig(
            poly_order=16,
            num_freqs=32,
            wavelet_levels=2
        ),
        recomposition=RecompositionConfig(
            depth='standard'
        )
    )
    
    model = HybridWaveStack(config)
    
    # Test forward pass
    batch, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch, seq_len))
    
    output = model(input_ids, return_lane_features=True)
    
    assert output['logits'].shape == (batch, seq_len, 1000)
    assert 'lane_features' in output
    assert 'coefficients' in output
    
    print(f"✓ HybridWaveStack test passed")
    print(f"  Total params: {model.get_num_params():,}")
    print(f"  Non-embedding params: {model.get_num_params(non_embedding=True):,}")
Acceptance Criteria:

 Model assembles correctly with analytical decomposition
 Model assembles correctly with neural baseline
 Forward pass produces correct output shapes
 Can return intermediate lane features for analysis
 Parameter counting works
 Tests pass
Phase 5: Training Infrastructure (Days 8-9)
Task 5.1: Loss Functions
File: src/training/loss.py

Instructions:

python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class StandardLanguageModelingLoss(nn.Module):
    """Standard cross-entropy loss for next-token prediction"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        Returns:
            loss: scalar
        """
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss to encourage lane specialization.
    
    Components:
    1. Autoregressive loss (next-token prediction)
    2. Reconstruction loss (lanes preserve information)
    3. Orthogonality loss (lanes learn different features)
    """
    
    def __init__(self, alpha_ar: float = 0.7, alpha_recon: float = 0.2,
                 alpha_ortho: float = 0.1):
        super().__init__()
        self.alpha_ar = alpha_ar
        self.alpha_recon = alpha_recon
        self.alpha_ortho = alpha_ortho
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lane_features: Dict[str, torch.Tensor],
                embeddings: torch.Tensor) -> tuple[torch.Tensor, Dict]:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            lane_features: Dict with 'polynomial', 'trigonometric', 'wavelet'
            embeddings: (batch, seq_len, hidden_dim) original embeddings
        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual losses
        """
        # 1. Autoregressive loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # 2. Reconstruction loss
        # Each lane should preserve input information
        recon_loss = 0
        for lane_name, features in lane_features.items():
            # MSE between lane features and embeddings
            recon_loss += F.mse_loss(features, embeddings)
        recon_loss /= len(lane_features)
        
        # 3. Orthogonality loss
        # Lanes should learn different representations
        lanes = list(lane_features.values())
        ortho_loss = 0
        num_pairs = 0
        
        for i in range(len(lanes)):
            for j in range(i + 1, len(lanes)):
                # Minimize cosine similarity
                cos_sim = F.cosine_similarity(
                    lanes[i].view(-1, lanes[i].size(-1)),
                    lanes[j].view(-1, lanes[j].size(-1)),
                    dim=-1
                ).abs().mean()
                ortho_loss += cos_sim
                num_pairs += 1
        
        if num_pairs > 0:
            ortho_loss /= num_pairs
        
        # Total loss
        total_loss = (
            self.alpha_ar * ce_loss +
            self.alpha_recon * recon_loss +
            self.alpha_ortho * ortho_loss
        )
        
        loss_dict = {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'recon_loss': recon_loss.item(),
            'ortho_loss': ortho_loss.item(),
        }
        
        return total_loss, loss_dict
Acceptance Criteria:

 Standard CE loss works
 Multi-objective loss computes all components
 Tests pass
Task 5.2: Metrics & Tracking
File: src/training/metrics.py

Instructions:

python
import torch
import numpy as np
from typing import Dict

class LaneSpecializationMetrics:
    """Compute metrics for lane specialization analysis"""
    
    @staticmethod
    def lane_utilization_entropy(lane_features: Dict[str, torch.Tensor]) -> float:
        """
        Compute entropy of lane utilization.
        High entropy = balanced usage, Low entropy = some lanes dominate
        
        Args:
            lane_features: Dict of (batch, seq, hidden) tensors
        Returns:
            entropy: scalar in [0, log(num_lanes)]
        """
        # Compute energy per lane
        energies = []
        for features in lane_features.values():
            energy = (features ** 2).sum().item()
            energies.append(energy)
        
        # Normalize to probabilities
        total_energy = sum(energies)
        probs = [e / (total_energy + 1e-8) for e in energies]
        
        # Compute entropy
        entropy = -sum(p * np.log(p + 1e-8) for p in probs)
        
        return entropy
    
    @staticmethod
    def lane_specialization_index(coeffs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Measure how specialized each lane's coefficients are.
        High values = concentrated on few basis functions (specialized)
        Low values = spread across many basis functions (general)
        
        Args:
            coeffs: Dict of coefficient tensors
        Returns:
            specialization scores per lane
        """
        scores = {}
        
        for lane_name, coeff_tensor in coeffs.items():
            # Compute L2 norm per coefficient position
            # Shape: (batch, seq, num_coeffs) -> (num_coeffs,)
            coeff_norms = coeff_tensor.norm(dim=(0, 1))
            
            # Normalize to distribution
            coeff_dist = coeff_norms / (coeff_norms.sum() + 1e-8)
            
            # Compute entropy (lower = more specialized)
            entropy = -(coeff_dist * torch.log(coeff_dist + 1e-8)).sum().item()
            
            # Convert to specialization score (higher = more specialized)
            max_entropy = np.log(len(coeff_dist))
            specialization = 1.0 - (entropy / max_entropy)
            
            scores[lane_name] = specialization
        
        return scores
    
    @staticmethod
    def cross_lane_orthogonality(lane_features: Dict[str, torch.Tensor]) -> float:
        """
        Measure orthogonality between lanes.
        1.0 = perfectly orthogonal, 0.0 = perfectly aligned
        
        Args:
            lane_features: Dict of (batch, seq, hidden) tensors
        Returns:
            orthogonality score in [0, 1]
        """
        lanes = list(lane_features.values())
        
        # Flatten to vectors
        lane_vecs = [l.reshape(-1) for l in lanes]
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(lane_vecs)):
            for j in range(i + 1, len(lane_vecs)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    lane_vecs[i].unsqueeze(0),
                    lane_vecs[j].unsqueeze(0)
                ).abs().item()
                similarities.append(cos_sim)
        
        # Orthogonality = 1 - average similarity
        avg_similarity = np.mean(similarities)
        orthogonality = 1.0 - avg_similarity
        
        return orthogonality

def compute_perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity"""
    return np.exp(loss)
Acceptance Criteria:

 Metrics compute without errors
 Values are in expected ranges
 Tests pass
Task 5.3: Gradient Tracker
File: src/analysis/gradient_tracker.py

Instructions:

python
import torch
from collections import defaultdict
from typing import Dict, List
import numpy as np

class GradientTracker:
    """
    Track gradient statistics during training for Experiment 3.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.history = defaultdict(list)
    
    def log_gradients(self, step: int):
        """
        Call after loss.backward(), before optimizer.step()
        
        Args:
            step: Current training step
        """
        # Embedding gradients
        if hasattr(self.model, 'embeddings'):
            if self.model.embeddings.token_embed.weight.grad is not None:
                grad = self.model.embeddings.token_embed.weight.grad
                self.history['embeddings_norm'].append({
                    'step': step,
                    'value': grad.norm().item()
                })
                self.history['embeddings_mean'].append({
                    'step': step,
                    'value': grad.abs().mean().item()
                })
        
        # Lane-specific gradients
        for lane_name in ['poly', 'trig', 'wavelet']:
            # Decomposition gradients (neural baseline only)
            decomp = getattr(self.model, f'{lane_name}_decomp', None)
            if decomp is not None:
                grad_norms = []
                for param in decomp.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                
                if grad_norms:
                    self.history[f'{lane_name}_decomp_grad'].append({
                        'step': step,
                        'value': np.mean(grad_norms)
                    })
            
            # Recomposition gradients
            recomp = getattr(self.model, f'{lane_name}_recomp', None)
            if recomp is not None:
                grad_norms = []
                for param in recomp.parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                
                if grad_norms:
                    self.history[f'{lane_name}_recomp_grad'].append({
                        'step': step,
                        'value': np.mean(grad_norms)
                    })
        
        # Lane gradient balance
        lane_grads = []
        for lane_name in ['poly', 'trig', 'wavelet']:
            key = f'{lane_name}_recomp_grad'
            if key in self.history and self.history[key]:
                lane_grads.append(self.history[key][-1]['value'])
        
        if lane_grads:
            self.history['lane_grad_variance'].append({
                'step': step,
                'value': np.var(lane_grads)
            })
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        summary = {}
        
        for key, values in self.history.items():
            if values:
                vals = [v['value'] for v in values]
                summary[key] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'final': vals[-1],
                }
        
        return summary
    
    def save(self, path: str):
        """Save history to file"""
        import json
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
Acceptance Criteria:

 Tracks gradients for all components
 Can save/load history
 Summary statistics work
[Due to length, I'll provide the remaining tasks in summary form. Let me know if you want any expanded]

Phase 6: Training Loop (Day 10)
Task 6.1: src/training/trainer.py - Main training loop with:

Optimizer setup (AdamW)
Learning rate scheduling
Gradient accumulation
Mixed precision (optional)
Checkpoint saving/loading
WandB logging
Evaluation loop
Phase 7: Data Pipeline (Day 11)
Task 7.1: src/data/dataset.py - Dataset loaders for:

TinyStories
FineWeb-Edu (for later)
Python code (The Stack)
Task 7.2: src/data/tokenizer.py - Tokenizer wrapper

Phase 8: Experiment Configs (Day 12)
Task 8.1: Create YAML configs for all experiments:

experiments/exp1_expressivity/config_A_neural_50m.yaml
experiments/exp1_expressivity/config_B_hybrid_12m.yaml
experiments/exp1_expressivity/config_C_hybrid_50m.yaml
experiments/exp1_expressivity/config_D_neural_12m.yaml
Phase 9: Experiment Runners (Day 13)
Task 9.1: experiments/exp1_expressivity/run_experiment.py Task 9.2: experiments/exp2_adaptation/run_experiment.py Task 9.3: experiments/exp3_gradients/run_experiment.py

Phase 10: Analysis Tools (Day 14)
Task 10.1: src/analysis/interpretability.py - Lane attribution Task 10.2: src/analysis/visualization.py - Plotting utilities
