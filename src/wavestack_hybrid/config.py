"""Dataclasses describing WaveStack Hybrid configuration surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Dict, Literal, get_type_hints


@dataclass
class DecompositionConfig:
    """Analytical decomposition lane configuration."""

    causal: bool = False
    schedule: bool = False
    schedule_steps: int = 1000
    poly_order_min: int = 4
    num_freqs_min: int = 16
    wavelet_levels_min: int = 1

    # Chebyshev (polynomial) settings
    poly_order: int = 4
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
class ContextBlockConfig:
    """Optional context block configuration."""

    enabled: bool = False
    position: Literal["pre", "post", "both"] = "both"
    block_type: Literal["mlp", "conv"] = "mlp"
    depth: int = 1
    hidden_multiplier: float = 2.0
    dropout: float = 0.1
    kernel_size: int = 3
    causal: bool = True


@dataclass
class ModelConfig:
    """Top-level model layout."""

    use_analytical_decomp: bool = True  # False => neural-only baseline
    enabled_lanes: list[str] = field(default_factory=lambda: ["poly", "trig", "wavelet"])
    context_block: ContextBlockConfig = field(default_factory=ContextBlockConfig)

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

        lane_params = sum(self._lane_param_breakdown().values())

        # Mixing mechanism estimate
        num_lanes = len(self.enabled_lanes)
        mixing_params = self._mixing_param_count(num_lanes)

        # Optional neural decomposition stack (baseline)
        neural_params = 0
        if not self.use_analytical_decomp:
            neural_params = self.neural_decomp_layers * (self.hidden_dim ** 2)

        context_params = self._context_param_count()

        return embed_params + lane_params + mixing_params + neural_params + context_params

    def get_param_breakdown(self) -> Dict[str, int]:
        """Return parameter estimates grouped by major subsystem."""

        num_lanes = len(self.enabled_lanes)
        breakdown: Dict[str, int] = {
            "embeddings": self.vocab_size * self.hidden_dim + self.max_seq_len * self.hidden_dim,
            "lanes": sum(self._lane_param_breakdown().values()),
            "mixing": self._mixing_param_count(num_lanes),
            "context": self._context_param_count(),
        }
        if not self.use_analytical_decomp:
            breakdown["neural_decomp"] = self._neural_param_count()
        return breakdown

    def get_lane_param_breakdown(self) -> Dict[str, int]:
        """Return per-lane parameter estimates."""

        return self._lane_param_breakdown()

    def get_flop_breakdown(self, seq_len: int | None = None) -> Dict[str, float]:
        """Approximate FLOPs per forward pass (linear layers only).

        Returns per-token FLOPs when seq_len is None, otherwise per-sequence FLOPs.
        Analytical basis/FFT costs are not included in this estimate.
        """

        per_token = self._flops_per_token()
        if seq_len is None:
            return per_token
        return {name: value * seq_len for name, value in per_token.items()}

    def get_lane_flop_breakdown(self, seq_len: int | None = None) -> Dict[str, float]:
        """Return per-lane FLOP estimates for recomposition + projections."""

        per_token = self._lane_flops_per_token()
        if seq_len is None:
            return per_token
        return {name: value * seq_len for name, value in per_token.items()}

    def _lane_param_breakdown(self) -> Dict[str, int]:
        lane_capacity_map = {
            "poly": self.recomposition.poly_capacity,
            "trig": self.recomposition.trig_capacity,
            "wavelet": self.recomposition.wavelet_capacity,
        }
        lane_params: Dict[str, int] = {}
        for lane_name in self.enabled_lanes:
            width = max(32, int(self.hidden_dim * lane_capacity_map[lane_name]))
            recomposition_params = self._recomposition_param_count(width)
            if lane_name == "wavelet":
                decomp_params = (self.hidden_dim * (self.hidden_dim * 2 * self.decomposition.wavelet_levels)) + self.hidden_dim
            else:
                decomp_params = (self.hidden_dim * self.hidden_dim) + self.hidden_dim
            lane_params[lane_name] = recomposition_params + decomp_params
        return lane_params

    def _recomposition_param_count(self, width: int) -> int:
        depth = {"shallow": 1, "standard": 2, "deep": 3}[self.recomposition.depth]
        params = 0
        in_dim = self.hidden_dim
        for _ in range(depth):
            params += in_dim * width + width
            in_dim = width
        params += width * self.hidden_dim + self.hidden_dim
        return params

    def _mixing_param_count(self, num_lanes: int) -> int:
        if self.mixing_type == "gated":
            return (self.hidden_dim * num_lanes) * self.hidden_dim + self.hidden_dim + self.hidden_dim * num_lanes + num_lanes
        if self.mixing_type == "attention":
            return 3 * (self.hidden_dim * self.hidden_dim)
        return (self.hidden_dim * num_lanes) * self.hidden_dim + self.hidden_dim + self.hidden_dim * self.hidden_dim + self.hidden_dim

    def _context_param_count(self) -> int:
        if not self.context_block.enabled:
            return 0
        block_count = 2 if self.context_block.position == "both" else 1
        depth = max(1, self.context_block.depth)
        if self.context_block.block_type == "mlp":
            width = max(8, int(self.hidden_dim * self.context_block.hidden_multiplier))
            per_block = (self.hidden_dim * width + width) + (width * self.hidden_dim + self.hidden_dim)
            return block_count * depth * per_block
        per_conv = (self.hidden_dim * self.hidden_dim * self.context_block.kernel_size) + self.hidden_dim
        return block_count * depth * per_conv

    def _neural_param_count(self) -> int:
        params = 0
        for _ in range(self.neural_decomp_layers):
            params += self.hidden_dim * self.hidden_dim + self.hidden_dim
            params += 2 * self.hidden_dim
        return params

    def _flops_per_token(self) -> Dict[str, float]:
        num_lanes = len(self.enabled_lanes)
        per_token: Dict[str, float] = {
            "lanes": sum(self._lane_flops_per_token().values()),
            "mixing": self._mixing_flops_per_token(num_lanes),
            "context": self._context_flops_per_token(),
            "lm_head": 2.0 * self.hidden_dim * self.vocab_size,
        }
        if not self.use_analytical_decomp:
            per_token["neural_decomp"] = self._neural_flops_per_token()
        return per_token

    def _lane_flops_per_token(self) -> Dict[str, float]:
        lane_capacity_map = {
            "poly": self.recomposition.poly_capacity,
            "trig": self.recomposition.trig_capacity,
            "wavelet": self.recomposition.wavelet_capacity,
        }
        flops: Dict[str, float] = {}
        for lane_name in self.enabled_lanes:
            width = max(32, int(self.hidden_dim * lane_capacity_map[lane_name]))
            recomposition_flops = self._recomposition_flops_per_token(width)
            decomp_flops = 0.0
            if self.use_analytical_decomp:
                if lane_name == "wavelet":
                    decomp_flops = 2.0 * self.hidden_dim * (
                        self.hidden_dim * 2 * self.decomposition.wavelet_levels
                    )
                else:
                    decomp_flops = 2.0 * self.hidden_dim * self.hidden_dim
            flops[lane_name] = recomposition_flops + decomp_flops
        return flops

    def _recomposition_flops_per_token(self, width: int) -> float:
        depth = {"shallow": 1, "standard": 2, "deep": 3}[self.recomposition.depth]
        flops = 0.0
        in_dim = self.hidden_dim
        for _ in range(depth):
            flops += 2.0 * in_dim * width
            in_dim = width
        flops += 2.0 * width * self.hidden_dim
        return flops

    def _mixing_flops_per_token(self, num_lanes: int) -> float:
        if self.mixing_type == "gated":
            return (
                2.0 * (self.hidden_dim * num_lanes) * self.hidden_dim
                + 2.0 * self.hidden_dim * num_lanes
                + self.hidden_dim * num_lanes
            )
        if self.mixing_type == "attention":
            lane_tokens = num_lanes
            qkv = 3.0 * (2.0 * self.hidden_dim * self.hidden_dim) * lane_tokens
            attn_scores = 2.0 * lane_tokens * lane_tokens * self.hidden_dim
            attn_apply = 2.0 * lane_tokens * lane_tokens * self.hidden_dim
            return qkv + attn_scores + attn_apply
        return 2.0 * (self.hidden_dim * num_lanes) * self.hidden_dim + 2.0 * self.hidden_dim * self.hidden_dim

    def _context_flops_per_token(self) -> float:
        if not self.context_block.enabled:
            return 0.0
        block_count = 2 if self.context_block.position == "both" else 1
        depth = max(1, self.context_block.depth)
        if self.context_block.block_type == "mlp":
            width = max(8, int(self.hidden_dim * self.context_block.hidden_multiplier))
            per_block = 2.0 * self.hidden_dim * width + 2.0 * width * self.hidden_dim
            return block_count * depth * per_block
        per_conv = 2.0 * self.hidden_dim * self.hidden_dim * self.context_block.kernel_size
        return block_count * depth * per_conv

    def _neural_flops_per_token(self) -> float:
        flops = 0.0
        for _ in range(self.neural_decomp_layers):
            flops += 2.0 * self.hidden_dim * self.hidden_dim
        return flops


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
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    alpha_autoregressive: float = 0.7
    alpha_reconstruction: float = 0.2
    alpha_orthogonality: float = 0.1

    eval_interval: int = 1000
    eval_batches: int = 8
    save_interval: int = 5000
    log_interval: int = 100

    lane_diversity: bool = False
    lane_diversity_metric: Literal["cosine", "energy"] = "cosine"
    log_lane_stats: bool = False
    log_runtime: bool = False
    log_memory: bool = False

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
