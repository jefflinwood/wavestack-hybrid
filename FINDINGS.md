# WaveStack Hybrid Findings

This file collects experiment outcomes, takeaways, and open questions. Add new entries in reverse-chronological order.

## 2026-01-01 — Multi-Seed Expressivity (Causal, poly_order=4)
- **Setup:** TinyStories, 3 seeds, 3k steps, 8k/16k samples, MPS.
- **Runtime:** ~1 day wall time on M1 Max (64 GB RAM) for the full batch (baselines + 50m ablations + adaptation sanity).
- **Outcome:** Hybrid consistently beats neural baselines at 12m and 50m on eval and holdout losses.
- **Representative ranges:**
  - Hybrid 12m eval ~0.737–0.741; holdout ~0.814–0.846.
  - Neural 12m eval ~1.036–1.038; holdout ~1.087–1.137.
  - Hybrid 50m eval ~0.617–0.619; holdout ~0.848–0.882.
  - Neural 50m eval ~1.003–1.005; holdout ~1.048–1.125.
- **Interpretation:** Advantage is stable across seeds after enforcing causal lanes and lowering poly order.
- **Next check:** Extend to longer training horizons (e.g., 10k steps) to confirm gaps persist.

## 2026-01-02 — Context Blocks (12m, Causal)
- **Setup:** 2 seeds, 3k steps, 8k samples. Compared baseline vs MLP vs causal conv context blocks (pre+post).
- **Outcome:** Conv context block consistently improved eval/holdout vs baseline; MLP context was neutral.
- **Representative results:**
  - Baseline eval ~0.737–0.741; holdout ~0.831–0.846.
  - Context MLP eval ~0.739–0.743; holdout ~0.832–0.847.
  - Context conv eval ~0.728–0.730; holdout ~0.823–0.836.
- **Action:** Promoted conv context block to default in expressivity configs.

## 2026-01-03 — Scheduling (12m, Causal)
- **Setup:** 2 seeds, 3k steps, 8k samples. Linear schedule ramped `poly_order`, `num_freqs`, `wavelet_levels` over 1k steps.
- **Outcome:** Scheduled vs static showed no meaningful difference at 3k steps.
- **Representative results:**
  - Seed 1 baseline eval 0.7284 / holdout 0.8354 vs scheduled eval 0.7287 / holdout 0.8348.
  - Seed 2 baseline eval 0.7296 / holdout 0.8225 vs scheduled eval 0.7296 / holdout 0.8225.
- **Interpretation:** Scheduling may need longer horizons or different ramp settings; parked for larger hardware.

## 2026-01-04 — Long-Horizon Expressivity (10k Steps)
- **Setup:** 10k steps, TinyStories, seed 1, sample caps 8k (12m) / 16k (50m).
- **Outcome:** Hybrid maintains a strong advantage at 10k steps.
- **Representative results:**
  - Hybrid 12m eval 0.8859 / holdout 1.0365.
  - Neural 12m eval 1.1825 / holdout 1.3058.
  - Hybrid 50m eval 0.5128 / holdout 0.7628.
  - Neural 50m eval 1.0861 / holdout 1.1878.

## 2026-01-04 — Lane Diversity Follow-up (12m)
- **Setup:** 2 seeds, 3k steps, 8k samples; compared baseline vs cosine and energy diversity penalties.
- **Outcome:** Both diversity regularizers provide a small, consistent improvement over baseline.
- **Representative averages:**
  - Baseline avg eval 0.7361 / holdout 0.8335.
  - Cosine diversity avg eval 0.7291 / holdout 0.8315.
  - Energy diversity avg eval 0.7291 / holdout 0.8313.
- **Interpretation:** Gains are modest; cosine/energy are effectively tied. Defaulting to cosine is reasonable.

## 2026-01-01 — 50m Lane Ablations (Causal, poly_order=4)
- **Setup:** 3 seeds, 3k steps, 16k samples.
- **Outcome:** Wavelet-only and no-wavelet variants are close to each other but both lag the full hybrid.
- **Representative ranges:**
  - Only wavelet eval ~0.679–0.684; holdout ~0.916–0.953.
  - No wavelet eval ~0.683–0.685; holdout ~0.929–0.975.
- **Interpretation:** Wavelet lane is strong, but full multi-lane mixing still helps.

## 2025-12-30 — Poly Order Tuning (12m, Causal)
- **Setup:** 3k steps, 8k samples.
- **Outcome:** Lower Chebyshev order performs better than the original order 32.
- **Result:** `poly_order=4` slightly beats `poly_order=8` on eval/holdout; default set to 4.
- **Interpretation:** The polynomial lane is over-capacity at higher order and benefits from a smaller basis.

## 2025-12-28 — Lane Ablations (12m, Causal)
- **Setup:** 3k steps, 8k samples.
- **Outcome:** Wavelet > trig > poly as single-lane performers.
- **Interpretation:** The poly lane is weakest on its own and can slightly hurt when over-parameterized.
- **Next check:** Tune poly capacity/order (completed) and re-evaluate.

## 2025-12-27 — Causal Lanes Fix
- **Issue:** Analytical lanes leaked future tokens (non-causal FFT, wavelet pooling, and full-sequence polynomial bases).
- **Fix:** Added causal variants for Chebyshev, Fourier, and wavelet lanes; re-ran baselines.
- **Impact:** Losses rose to realistic levels, but hybrid advantage persisted.

## Open Questions
- Does the hybrid advantage persist at longer horizons (10k–50k steps)?
- Do tuned poly settings generalize to non-TinyStories corpora?
- How do per-lane parameters/FLOPs compare to attention baselines at matched quality?
- Can the causal Fourier lane be optimized to reduce compute overhead?
