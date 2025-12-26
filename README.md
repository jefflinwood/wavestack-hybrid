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

## Repository Layout
- `src/wavestack_hybrid/`: Python package with model components, training utilities, and analysis helpers.
- `experiments/`: Reproducible experiment setups with YAML configs per study.
- `tests/`: Pytest suite covering decomposition lanes, recomposition heads, and the assembled model.
- `scripts/`: Operational helpers for bootstrapping data or checking dependencies.
- `pyproject.toml`: Project metadata, build backend configuration, and dependency list.

## Status
This project is under active construction following the execution plan in `PLAN.md`. Refer to `AGENTS.md` for contributor guidelines. Issues and PRs are welcome once core infrastructure is stabilized.
