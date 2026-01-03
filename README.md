# WaveStack Hybrid

WaveStack Hybrid explores a multi-lane large language model architecture where analytical decompositions (Chebyshev polynomials, Fourier bases, fixed wavelets) feed learned recomposition networks. This repository hosts research code, experiment configs, and analysis tools for iterating on that idea.

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   uv pip install -e .
   ```
2. Verify the environment:
   ```bash
   uv run python scripts/verify_setup.py
   ```
3. Download sample data to unblock quick experiments:
   ```bash
   uv run python scripts/download_data.py --dataset tinystories
   ```
4. Run a synthetic smoke test to ensure the model and trainer wire up (pick a device with `--device` or rely on auto-detection):
   ```bash
    uv run python scripts/run_smoke.py --steps 3
   ```
5. Run a TinyStories-backed smoke test (same `--device` flag applies and defaults to auto-detection):
   ```bash
   uv run python scripts/run_tinystories_smoke.py --steps 3 --examples 64
   ```

Device selection defaults to `auto`, which prefers CUDA, then Apple Silicon MPS, then CPU. Override this behavior by setting `--device` on scripts or `TrainingConfig.device` in experiment configs.

## Repository Layout
- `src/wavestack_hybrid/`: Python package with model components, training utilities, and analysis helpers.
- `experiments/`: Reproducible experiment setups with YAML configs per study.
- `tests/`: Pytest suite covering decomposition lanes, recomposition heads, and the assembled model.
- `scripts/`: Operational helpers for bootstrapping data or checking dependencies.
- `pyproject.toml`: Project metadata, build backend configuration, and dependency list.

## Running Experiments
- Expressivity study:
  ```bash
  uv run python experiments/exp1_expressivity/run_experiment.py \
    --config experiments/exp1_expressivity/config_C_hybrid_50m.yaml \
    --device auto --max-steps 1000 --samples 2048
  ```
  Seeded runs can be reproduced with `--seed`, which also fixes dataloader shuffling:
  ```bash
  uv run python experiments/exp1_expressivity/run_experiment.py \
    --config experiments/exp1_expressivity/config_B_hybrid_12m.yaml \
    --device auto --max-steps 3000 --samples 8000 --seed 1
  ```
  To run the whole expressivity suite (optionally with ablations) and multiple seeds:
  ```bash
  uv run python scripts/run_expressivity_suite.py --device auto --max-steps 3000 --seeds 1,2,3
  uv run python scripts/run_expressivity_suite.py --device auto --max-steps 3000 --include-ablations --seeds 1,2
  ```
  Runtime/memory logging can be enabled in the YAML configs:
  ```yaml
  training:
    log_runtime: true
    log_memory: true
  ```
- Inference scaling benchmark (Hybrid vs Transformer baseline):
  ```bash
  uv run python scripts/benchmark_inference.py \
    --model both \
    --config experiments/exp1_expressivity/config_B_hybrid_12m.yaml \
    --device auto \
    --seq-lens 64,128,256,512 \
    --batch-size 8 --steps 10
  ```
- Adaptation study (pretrain + finetune):
  ```bash
  uv run python experiments/exp2_adaptation/run_experiment.py \
    --pretrain-config experiments/exp2_adaptation/pretrain_config.yaml \
    --finetune-config experiments/exp2_adaptation/finetune_config.yaml \
    --device auto --pretrain-max-steps 1000 --finetune-max-steps 500
  ```
- Gradient study:
  ```bash
  uv run python experiments/exp3_gradients/run_experiment.py \
    --config experiments/exp3_gradients/config.yaml --device auto --samples 1024
  ```
All runners accept `--device` and optional step/sample limits to keep local iterations lightweight.

## Status
This project is under active construction following the execution plan in `PLAN.md`. Refer to `AGENTS.md` for contributor guidelines. Issues and PRs are welcome once core infrastructure is stabilized.
