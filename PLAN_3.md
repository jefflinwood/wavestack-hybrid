# WaveStack Hybrid – Phase 3 Plan (Scaling Behavior)

This phase focuses on empirically determining whether WaveStack Hybrid scales linearly or quadratically with sequence length and how it compares to transformer baselines under matched conditions.

## Phase 1 – Scaling Protocol
- **Controlled Sweep:** Fix batch size, model params, and device; vary sequence length across a wide range (e.g., 128 → 2048 or 4096).
- **Stable Seeds:** Use a fixed seed and fixed input data size to ensure repeatability.
- **Warmup & Timing:** Use warmup iterations before timing; record average step time over multiple iterations.
- **Memory Tracking:** Capture peak memory during sweeps to validate O(S) vs O(S^2) behavior.

## Phase 2 – Matched Baselines
- **Hybrid vs Transformer:** Use the matched‑param transformer configs and hybrid configs already in the repo.
- **Training vs Inference:** Benchmark both forward‑only inference and a short training step to see scaling in each regime.
- **Baseline Sanity:** Confirm no NaNs in transformer runs and validate consistent loss curves across lengths.

## Phase 3 – Reporting
- **Scaling Exponent:** Fit a power law `time ~ seq_len^p` and report `p`.
- **Tables + Plots:** Produce a summary table (seq_len, time, tokens/s, memory) and a small plot of log‑log slopes.
- **Findings:** Summarize whether scaling appears linear (p≈1) or quadratic (p≈2), with notes on caveats.

## Phase 4 – Follow-ups (If Needed)
- **Causal Fourier Optimization:** If hybrid scaling is worse than expected, optimize the causal Fourier path.
- **Context Block Placement:** If scaling is linear but throughput lags, test pre/post placement for speed.
- **Sequence Packing:** Introduce packed sequences to reduce padding overhead at high lengths.

## Execution Notes
- Keep runs on the same hardware and environment to avoid confounds.
- Log results in `EXPERIMENT_LOG.md` and add a summary entry to `FINDINGS.md`.
