# WaveStack Hybrid Findings

This file collects experiment outcomes, takeaways, and open questions. Add new entries in reverse-chronological order.

## 2026-01-01 — Multi-Seed Expressivity (Causal, poly_order=4)
- **Setup:** TinyStories, 3 seeds, 3k steps, 8k/16k samples, MPS.
- **Outcome:** Hybrid consistently beats neural baselines at 12m and 50m on eval and holdout losses.
- **Representative ranges:**
  - Hybrid 12m eval ~0.737–0.741; holdout ~0.814–0.846.
  - Neural 12m eval ~1.036–1.038; holdout ~1.087–1.137.
  - Hybrid 50m eval ~0.617–0.619; holdout ~0.848–0.882.
  - Neural 50m eval ~1.003–1.005; holdout ~1.048–1.125.
- **Interpretation:** Advantage is stable across seeds after enforcing causal lanes and lowering poly order.
- **Next check:** Extend to longer training horizons (e.g., 10k steps) to confirm gaps persist.

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
