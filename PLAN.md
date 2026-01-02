# WaveStack Hybrid – Remaining Execution Plan

The initial infrastructure (phases 1–10) is in place: package scaffolding, analytical lane implementations, recomposition/mixing logic, experiment configs, and smoke tests. The focus now is on validating the analytical-lane hypothesis and benchmarking WaveStack against attention-driven models.

## Phase 11 – Analytical Lane Experiments (Days 15-17)
- **Context Blocks:** Prototype optional lightweight convolutional or residual MLP blocks before/after the lane mixer to study non-attention context sharing. (completed: conv context block improves results, promoted to default)
- **Lane Diversity Regularizers:** Introduce configurable similarity/energy penalties plus per-lane logging to quantify specialization. (pending)
- **Frequency/Scale Scheduling:** Add schedulers for Chebyshev order, Fourier bins, and wavelet levels so analytical capacity can ramp up during training. (pending)
- **Causal Lanes:** Enforce causal analytical lanes to avoid future-token leakage and re-run expressivity baselines. (completed)
- **Poly Tuning:** Reduce Chebyshev order after ablations (current default poly_order=4). (completed)

## Phase 12 – Lane Variants & Ablations (Days 18-19)
- **Lane Toggle & Extensions:** Add config switches to disable lanes or plug in extra analytical transforms (e.g., spline, FIR filters) to map parameter/quality trade-offs. (mostly completed: `enabled_lanes`, 12m/50m ablations)
- **Parameter & FLOP Accounting:** Extend `ModelConfig.get_param_count()` and related tooling to report per-lane parameter counts and FLOP estimates so comparisons with attention baselines are grounded. (pending)

## Phase 13 – Evaluation Harness (Day 20)
- **Benchmark Runner:** Build scripts/notebooks to run matched-parameter experiments versus a reference Transformer (e.g., GPT-style) capturing loss, accuracy, runtime, and memory. (partial: inference scaling script + expressivity suite)
- **Findings Report:** Document where WaveStack requires more parameters, where it wins (latency, stability, interpretability), and outline next research directions. (pending)

## Next Experiments (Immediate)
- **Multi-seed Baselines:** Run 2–3 seeds for hybrid_12m vs neural_12m and hybrid_50m vs neural_50m using causal lanes and poly_order=4.
- **50m Lane Ablations:** Re-run hybrid_50m_only_wavelet and hybrid_50m_no_wavelet with poly_order=4 to validate lane importance at scale.
- **Adaptation Sanity:** Short pretrain+finetune pass with poly_order=4 to confirm transfer behavior under the new defaults.
