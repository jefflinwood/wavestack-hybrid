# WaveStack Hybrid – Remaining Execution Plan

The initial infrastructure (phases 1–10) is in place: package scaffolding, analytical lane implementations, recomposition/mixing logic, experiment configs, and smoke tests. The focus now is on validating the analytical-lane hypothesis and benchmarking WaveStack against attention-driven models.

## Phase 11 – Analytical Lane Experiments (Days 15-17)
- **Context Blocks:** Prototype optional lightweight convolutional or residual MLP blocks before/after the lane mixer to study non-attention context sharing.
- **Lane Diversity Regularizers:** Introduce configurable similarity/energy penalties plus per-lane logging to quantify specialization.
- **Frequency/Scale Scheduling:** Add schedulers for Chebyshev order, Fourier bins, and wavelet levels so analytical capacity can ramp up during training.

## Phase 12 – Lane Variants & Ablations (Days 18-19)
- **Lane Toggle & Extensions:** Add config switches to disable lanes or plug in extra analytical transforms (e.g., spline, FIR filters) to map parameter/quality trade-offs.
- **Parameter & FLOP Accounting:** Extend `ModelConfig.get_param_count()` and related tooling to report per-lane parameter counts and FLOP estimates so comparisons with attention baselines are grounded.

## Phase 13 – Evaluation Harness (Day 20)
- **Benchmark Runner:** Build scripts/notebooks to run matched-parameter experiments versus a reference Transformer (e.g., GPT-style) capturing loss, accuracy, runtime, and memory.
- **Findings Report:** Document where WaveStack requires more parameters, where it wins (latency, stability, interpretability), and outline next research directions.
