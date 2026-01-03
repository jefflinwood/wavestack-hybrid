# WaveStack Hybrid Design

This document explains the core architecture, data flow, and experiment tooling for WaveStack Hybrid.

## Architecture Overview
- **Embedding**: Token + positional embeddings produce a `(batch, seq, hidden)` stream.
- **Decomposition lanes**:
  - **Chebyshev**: Projects the sequence onto Chebyshev bases and reconstructs a smoothed signal.
  - **Fourier**: Reconstructs a band-limited version of the sequence from truncated frequencies.
  - **Wavelet**: Multi-scale low/high responses derived via pooling.
- **Recomposition**: Each lane has its own small MLP stack that maps lane features back to `hidden_dim`.
- **Mixing**: Lane outputs are fused via a gating, attention, or MLP mixer.
- **Output head**: LayerNorm + linear projection to vocabulary logits.

Key configuration switches:
- `model.use_analytical_decomp`: uses analytical lanes when `true`, otherwise a neural-only baseline.
- `model.enabled_lanes`: selects a subset of lanes for ablations.
- `model.decomposition.causal`: uses causal variants of analytical lanes to prevent future-token leakage.

## Data Flow
1. Dataset sample text is tokenized into fixed-length sequences.
2. `input_ids` and `labels` are identical token blocks.
3. The model produces logits for each position.
4. The loss applies an autoregressive shift (`labels[:, 1:]` vs `logits[:, :-1]`).

## Losses
The primary objective is an autoregressive cross-entropy loss with optional secondary terms:
- Reconstruction loss (MSE) when a reconstruction tensor is supplied.
- Lane orthogonality penalty when a lane-balance tensor is supplied.

Loss weights live in `TrainingConfig` as `alpha_*` fields.

## Training Loop
The `Trainer`:
- Cycles the dataloader until `max_steps` is reached.
- Logs average training loss every `log_interval`.
- Runs lightweight evaluation every `eval_interval` using `eval_batches`.
- Writes metrics to `outputs/<experiment>_metrics.jsonl`.
- Optionally writes checkpoints to `checkpoints/`.
- Optional runtime/memory stats can be logged per interval when `training.log_runtime` or `training.log_memory` are enabled.

## Evaluation
- Default evaluation uses the dataset `validation` split.
- For additional sanity checks, the experiment runners compute a fixed-seed holdout loss from the train split (or remaining samples when `--samples` is used).
- Results are appended to `EXPERIMENT_LOG.md`.
- When enabled, runtime/tokens-per-second and peak memory entries are also appended to `EXPERIMENT_LOG.md`.

## Experiments
- **Expressivity (exp1)**: Hybrid vs neural baselines at matched parameter sizes, plus lane ablations.
- **Adaptation (exp2)**: Pretrain + finetune pipeline for transfer behavior.
- **Gradients (exp3)**: Gradient norm tracking to surface optimization issues.

## Where to Look
- Model assembly: `src/wavestack_hybrid/models/wavestack.py`
- Decompositions: `src/wavestack_hybrid/models/decomposition/`
- Recomposition + mixing: `src/wavestack_hybrid/models/recomposition.py`, `src/wavestack_hybrid/models/mixing.py`
- Trainer + loss: `src/wavestack_hybrid/training/`
- Experiment runners: `experiments/exp*/run_experiment.py`
