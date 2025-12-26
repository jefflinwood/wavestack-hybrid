# Repository Guidelines

## Project Structure & Module Organization
The Python package lives under `src/wavestack_hybrid`; embed layers, decomposition lanes, recomposition networks, and the trainer/data utilities mirror the breakdown shown in `PLAN.md`. Experiments are grouped by study in `experiments/exp*/`, each with YAML configs and a `run_experiment.py`. Shared integration tests live in `tests/`, while automation helpers such as dataset downloaders sit in `scripts/`. Root-level `pyproject.toml` and `README.md` control packaging, dependencies, and onboarding context.

## Build, Test, and Development Commands
- `uv pip install -e .` — install the package in editable mode so local module changes are immediately importable (uv manages the venv and dependency resolver).
- `uv run python scripts/download_data.py --dataset tinystories` — fetches datasets referenced by the training configs; use explicit flags per corpus.
- `uv run python experiments/exp1_expressivity/run_experiment.py --config experiments/exp1_expressivity/config_C_hybrid_50m.yaml` — launches a tracked experiment using the selected configuration.
- `uv run pytest tests/` — run the full test suite; add `-k test_recomposition` to scope to a single component.
- `uv run python scripts/run_smoke.py --steps 3 --device auto` — synthetic sanity check; `--device` accepts `auto/cpu/cuda/mps`.
- `uv run python scripts/run_tinystories_smoke.py --steps 3 --examples 64 --device auto` — miniature TinyStories pass using the same device selection.
- Experiment runners support overrides to keep iterations short:
  - Expressivity: `uv run python experiments/exp1_expressivity/run_experiment.py --config ... --device auto --samples 2048 --max-steps 1000`
  - Adaptation: `uv run python experiments/exp2_adaptation/run_experiment.py --pretrain-config ... --finetune-config ... --device auto --pretrain-max-steps 1000 --finetune-max-steps 500`
  - Gradients: `uv run python experiments/exp3_gradients/run_experiment.py --config ... --device auto --samples 1024`

## Coding Style & Naming Conventions
Use Python 3.10+, 4-space indentation, and type annotations for every public function. Favor dataclasses (see `src/config.py`) for structured configs, and keep module-level constants in `UPPER_SNAKE_CASE`. File names are lowercase_with_underscores; class names follow PascalCase (e.g., `HybridWaveStack`). Follow Torch semantics for tensor naming (`embeddings`, `poly_lane`, etc.) and document non-obvious math with short docstrings.

## Testing Guidelines
Author `pytest` tests under `tests/` with filenames mirroring the target module (`test_decomposition.py`). Individual tests should describe behavior (`test_fourier_lane_respects_max_freq`). Where possible, introduce regression fixtures for TinyStories snippets to avoid large downloads. Aim for >85% statement coverage on critical paths (decomposition, recomposition, trainer). Run `pytest --maxfail=1 --disable-warnings -q` before pushing to keep CI fast.

## Commit & Pull Request Guidelines
History currently shows imperative, descriptive messages (`Initial PLAN.md`); continue that style with concise summaries (e.g., `Add wavelet decoder config`). Scope each PR to one logical change set, link the motivating issue, and include: short description, testing commands executed, config or data updates, and screenshots/metrics for experiment-oriented work. Rebase onto main before requesting review to keep the linear history intact.

## Security & Configuration Tips
Store API tokens (e.g., Hugging Face, WandB) in environment variables and consume them through `config.py` rather than hard-coding. Large checkpoints and datasets should remain in external storage; add new cache paths to `.gitignore`. When sharing experiment configs, scrub any absolute filesystem paths or secrets.
