"""Dataclasses describing WaveStack Hybrid configuration surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Dict, Literal, get_type_hints


@dataclass
class DecompositionConfig:
    """Analytical decomposition lane configuration."""

    causal: bool = False

    # Chebyshev (polynomial) settings
    poly_order: int = 8
    poly_normalization: Literal["unit", "standard"] = "unit"

    # Fourier (trigonometric) settings
    num_freqs: int = 64
    freq_selection: Literal["learnable", "fixed"] = "learnable"

    # Wavelet settings
    wavelet_type: Literal["haar", "db4", "db8", "sym4"] = "db4"
    wavelet_levels: int = 3
    scale_selection: Literal["learnable", "fixed"] = "learnable"


@dataclass
class RecompositionConfig:
    """Configuration for the learned recomposition stack."""

    depth: Literal["shallow", "standard", "deep"] = "standard"
    poly_capacity: float = 1.0
    trig_capacity: float = 1.0
    wavelet_capacity: float = 1.5
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "swish"] = "gelu"


@dataclass
class ModelConfig:
    """Top-level model layout."""

    use_analytical_decomp: bool = True  # False => neural-only baseline
    enabled_lanes: list[str] = field(default_factory=lambda: ["poly", "trig", "wavelet"])

    # Dimensions
    vocab_size: int = 50_257
    hidden_dim: int = 512
    max_seq_len: int = 512

    # Subsystems
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    recomposition: RecompositionConfig = field(default_factory=RecompositionConfig)

    # Mixing
    mixing_type: Literal["gated", "attention", "mlp"] = "gated"
    num_lanes: int = 3

    # Enhancements
    use_skip_connections: bool = False
    use_multi_objective_loss: bool = False
    neural_decomp_layers: int = 3

    def get_param_count(self) -> int:
        """Rough parameter estimate used for experiment planning."""

        # Token + positional embeddings
        embed_params = self.vocab_size * self.hidden_dim + self.max_seq_len * self.hidden_dim

        # Lane-specific decompositions (cheap analytical filters approximated here)
        lane_overheads = {
            "poly": self.decomposition.poly_order,
            "trig": self.decomposition.num_freqs * 2,
            "wavelet": self.decomposition.wavelet_levels * 16,
        }

        lane_params = 0
        lane_capacity_map = {
            "poly": self.recomposition.poly_capacity,
            "trig": self.recomposition.trig_capacity,
            "wavelet": self.recomposition.wavelet_capacity,
        }

        for lane_name in self.enabled_lanes:
            base = lane_overheads[lane_name]
            # Analytical parameters + recomposition MLP
            hidden = int(self.hidden_dim * lane_capacity_map[lane_name])
            lane_params += base * hidden + hidden * self.hidden_dim

        # Mixing mechanism estimate
        num_lanes = len(self.enabled_lanes)
        if self.mixing_type == "gated":
            mixing_params = num_lanes * self.hidden_dim * 2
        elif self.mixing_type == "attention":
            mixing_params = self.hidden_dim ** 2
        else:  # mlp
            mixing_params = self.hidden_dim * (self.hidden_dim // 2)

        # Optional neural decomposition stack (baseline)
        neural_params = 0
        if not self.use_analytical_decomp:
            neural_params = self.neural_decomp_layers * (self.hidden_dim ** 2)

        return embed_params + lane_params + mixing_params + neural_params


@dataclass
class TrainingConfig:
    """Optimization hyperparameters."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    alpha_autoregressive: float = 0.7
    alpha_reconstruction: float = 0.2
    alpha_orthogonality: float = 0.1

    eval_interval: int = 1000
    eval_batches: int = 8
    save_interval: int = 5000
    log_interval: int = 100

    use_wandb: bool = True
    project_name: str = "wavestack_hybrid"

    device: str = "auto"
    mixed_precision: bool = False


@dataclass
class ExperimentConfig:
    """Full experiment settings pulled from YAML or CLI overrides."""

    name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"

    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML."""

        import yaml

        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        def _build_dataclass(target_cls, values):
            if not isinstance(values, dict):
                return values
            hints = get_type_hints(target_cls)
            kwargs: Dict[str, object] = {}
            for key, field_value in values.items():
                hint = hints.get(key)
                if hint and is_dataclass(hint) and isinstance(field_value, dict):
                    kwargs[key] = _build_dataclass(hint, field_value)
                else:
                    kwargs[key] = field_value
            return target_cls(**kwargs)

        return cls(
            name=data.get("name", "unnamed-experiment"),
            model=_build_dataclass(ModelConfig, data.get("model", {})),
            training=_build_dataclass(TrainingConfig, data.get("training", {})),
            dataset_name=data.get("dataset_name", "roneneldan/TinyStories"),
            train_split=data.get("train_split", "train"),
            val_split=data.get("val_split", "validation"),
            output_dir=data.get("output_dir", "./outputs"),
            checkpoint_dir=data.get("checkpoint_dir", "./checkpoints"),
        )

    def to_yaml(self, path: str | Path):
        """Persist the configuration to disk."""

        import yaml

        with open(path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(asdict(self), fp, sort_keys=False)
