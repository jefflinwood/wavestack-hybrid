from pathlib import Path

from wavestack_hybrid.config import ExperimentConfig, RecompositionConfig


def test_experiment_config_from_yaml_parses_nested(tmp_path):
    yaml_content = """
name: test
model:
  hidden_dim: 128
  recomposition:
    depth: deep
    wavelet_capacity: 2.0
training:
  device: cpu
  max_steps: 10
"""
    path = Path(tmp_path) / "config.yaml"
    path.write_text(yaml_content)

    config = ExperimentConfig.from_yaml(path)
    assert isinstance(config.model.recomposition, RecompositionConfig)
    assert config.model.recomposition.depth == "deep"
    assert config.model.recomposition.wavelet_capacity == 2.0
