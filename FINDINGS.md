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

## 2026-01-05 — Small Corpus (Wikitext-2)
- **Setup:** 2 seeds, 3k steps, 8k samples; hybrid vs neural on `wikitext-2-raw-v1`.
- **Outcome:** Neural baseline slightly outperforms hybrid at 12m on this small corpus.
- **Representative results:**
  - Hybrid eval ~0.704–0.705; holdout ~0.692–0.702.
  - Neural eval ~0.689–0.693; holdout ~0.665–0.680.
- **Interpretation:** Hybrid gains may be dataset‑dependent; revisit with longer horizons or adjusted hyperparameters.

## 2026-01-07 — Wikitext-2 Long Follow-up (10k Steps)
- **Setup:** 10k steps, 2 seeds, 8k samples; hybrid vs neural on `wikitext-2-raw-v1`.
- **Outcome:** Neural baseline remains substantially better; hybrid underperforms at longer horizons.
- **Representative results:**
  - Hybrid eval ~1.138–1.140; holdout ~1.122–1.125.
  - Neural eval ~0.864–0.874; holdout ~0.840–0.849.
- **Interpretation:** The hybrid advantage does not transfer to Wikitext‑2 with current settings; investigate dataset‑specific tuning or decomposition adaptations.

## 2026-01-15 — Wikitext-2 Matched-Param Transformer (12m)
- **Setup:** 2 seeds, 3k steps, 8k samples; matched‑param transformer vs hybrid/neural on Wikitext‑2.
- **Outcome:** Transformer baseline outperforms both hybrid and neural at 12m.
- **Representative results:**
  - Transformer eval ~0.632; holdout ~0.606–0.617.
  - Neural eval ~0.689–0.693; holdout ~0.665–0.680.
  - Hybrid eval ~0.704–0.705; holdout ~0.692–0.702.
- **Interpretation:** On Wikitext‑2, transformer remains best; hybrid currently lags even at matched params.

## 2026-01-08 — Wikitext-2 Tuning Sweep (Hybrid)
- **Setup:** 10k steps, seed 1, 8k samples; swept lower LR, higher dropout, and lane capacity reductions.
- **Outcome:** Lowering LR to 1e‑4 significantly improves eval/holdout; other tweaks do not help.
- **Representative results:**
  - Baseline long eval 1.1371 / holdout 1.1227.
  - LR 1e‑4 eval 0.9401 / holdout 0.9279.
  - Dropout 0.2 eval 1.1462 / holdout 1.1303.
  - Lane caps eval 1.1474 / holdout 1.1220.
- **Interpretation:** Wikitext‑2 likely needs a lower LR; next step is re-running hybrid vs neural with LR 1e‑4 on both for parity.

## 2026-01-13 — Matched-Param Transformer Baselines (TinyStories)
- **Setup:** 2 seeds, 3k steps, TinyStories; matched-param transformer vs hybrid at ~12m.
- **Outcome:** Hybrid significantly outperforms the matched transformer baseline.
- **Representative results:**
  - Hybrid eval ~0.728–0.730; holdout ~0.823–0.835.
  - Transformer eval ~1.048; holdout ~1.117–1.146.
- **50m snapshot:** Transformer eval ~1.020; holdout ~1.063–1.104, while hybrid eval ~0.599–0.602 and holdout ~0.826–0.857.
- **Interpretation:** Even before optimization, the hybrid architecture is ahead at matched parameter counts on TinyStories.

## 2026-01-15 — Scaling Sweep (Inference, MPS)
- **Setup:** Sequence lengths 128/256/512, batch size 8, 10 steps, 2 warmup steps; same hybrid/transformer configs.
- **Outcome:** Both models appear near‑linear in this range; transformer is faster, hybrid is slightly lower memory.
- **Representative results:**
  - Hybrid exponent ~0.95; timing `128:44.32ms/23106.6tps/375MB`, `256:82.93ms/24695.7tps/581MB`, `512:166.30ms/24630.0tps/993MB`.
  - Transformer exponent ~1.07; timing `128:24.02ms/42638.1tps/518MB`, `256:47.78ms/42864.2tps/724MB`, `512:106.47ms/38469.9tps/1137MB`.
- **Interpretation:** No quadratic behavior is visible at 512 tokens; a longer sequence sweep (with higher max_seq_len) is needed to differentiate scaling.

## 2026-01-15 — Scaling Sweep (Long-Seq, MPS)
- **Setup:** Sequence lengths 512/1024/2048/4096, batch size 8, 10 steps, 2 warmup steps; `max_seq_len=4096`.
- **Outcome:** Hybrid remains near‑linear; transformer shows super‑linear scaling consistent with quadratic attention.
- **Representative results:**
  - Hybrid exponent ~1.00; timing `512:165.81ms/24703.0tps/998MB`, `1024:330.36ms/24797.0tps/1822MB`, `2048:659.53ms/24841.9tps/3469MB`, `4096:1318.81ms/24846.6tps/6762MB`.
  - Transformer exponent ~1.48; timing `512:106.97ms/38289.5tps/1183MB`, `1024:260.02ms/31505.4tps/2008MB`, `2048:724.75ms/22606.6tps/3666MB`, `4096:2295.93ms/14272.2tps/6966MB`.
- **Interpretation:** At long sequence lengths, transformer runtime scales super‑linearly while hybrid stays near‑linear, supporting the scaling hypothesis.

## 2026-01-15 — Scaling Sweep (Long-Seq, CPU)
- **Setup:** Sequence lengths 128/256/512/1024/2048/4096, batch size 4, 5 steps, 1 warmup; `max_seq_len=4096`.
- **Outcome:** Hybrid remains near‑linear; transformer shows super‑linear scaling on CPU with a steep throughput drop at long lengths.
- **Representative results:**
  - Hybrid exponent ~0.94; tokens/s ~6.6k–8.7k across lengths.
  - Transformer exponent ~1.28; tokens/s drops from ~7.5k (128) to ~2.8k (4096).
- **Interpretation:** CPU scaling reinforces the linear vs super‑linear behavior; hybrid is more stable at long sequences.

## 2026-01-15 — Scaling Sweep (Long-Seq, Wikitext-2, MPS)
- **Setup:** Sequence lengths 512/1024/2048/4096, batch size 8, 10 steps, 2 warmup; `max_seq_len=4096`.
- **Outcome:** Hybrid remains near‑linear; transformer shows super‑linear scaling on Wikitext‑2.
- **Representative results:**
  - Hybrid exponent ~1.00; timing `512:166.32ms/24626.7tps/998MB`, `1024:330.12ms/24815.5tps/1822MB`, `2048:658.56ms/24878.6tps/3469MB`, `4096:1318.23ms/24857.6tps/6762MB`.
  - Transformer exponent ~1.48; timing `512:106.84ms/38338.7tps/1183MB`, `1024:259.69ms/31545.3tps/2008MB`, `2048:725.26ms/22590.5tps/3666MB`, `4096:2295.27ms/14276.3tps/6966MB`.
- **Interpretation:** Long‑seq scaling behavior is consistent across datasets; hybrid stays near‑linear while transformer trends super‑linear.

## 2026-01-18 — CodeSearchNet Hybrid Run
- **Setup:** 10k steps, seed 1, 8k samples; CodeSearchNet Python JSONL (train/valid).
- **Outcome:** Training loss is reasonable; eval/holdout are higher than TinyStories, as expected for code data.
- **Results:** Train loss 0.3534, eval loss 3.5519, holdout loss 3.3570.
- **Notes:** Dataset now uses `code` column with pad masking for loss.

## 2026-01-18 — CodeSearchNet Baselines (12m)
- **Setup:** 10k steps, seed 1, 8k samples; hybrid vs neural vs matched-param transformer.
- **Outcome:** Transformer baseline performs best, neural second, hybrid worst on eval/holdout.
- **Results:**
  - Hybrid eval 3.5519 / holdout 3.3570.
  - Neural eval 3.1363 / holdout 2.9842.
  - Transformer eval 2.7744 / holdout 2.6473.
- **Note:** Runtime/tokens-per-second logs look anomalous; re-run with `training.log_runtime: true` if runtime comparisons are needed.

## 2026-01-20 — CodeSearchNet Hybrid Sweep
- **Setup:** 10k steps, seed 1, 8k samples; swept LR and minor architecture tweaks.
- **Outcome:** Lowering LR to 1e‑4 is the clear win; other tweaks help less or regress.
- **Results:**
  - Baseline eval 3.5486 / holdout 3.3495.
  - LR 1e‑4 eval 2.5838 / holdout 2.4565.
  - Recomp standard eval 2.8696 / holdout 2.7237.
  - Lane caps eval 3.1510 / holdout 2.9800.
  - No poly eval 2.9919 / holdout 2.8223.
- **Interpretation:** CodeSearchNet benefits strongly from a lower LR; adopt 1e‑4 for code runs and re-compare to baselines.

## 2026-01-20 — Wikitext-2 Long-Seq Phase 1 (1k Steps)
- **Setup:** Long-seq hybrid vs matched transformer, 1k steps, seed 1, 8k samples; `max_seq_len=4096`, batch size 8.
- **Outcome:** Both models are still high-loss at 1k steps; hybrid is slightly better on eval/holdout and serves as a checkpoint for probes.
- **Results:**
  - Hybrid eval 4.5102 / holdout 4.3923.
  - Transformer eval 4.8949 / holdout 4.7103.
- **Interpretation:** Early checkpoints are adequate for linguistic probing; do not over-interpret loss gaps at this short horizon.

## 2026-01-20 — Wikitext-2 Linguistic Probes (Phase 2)
- **Setup:** 512 validation samples, pooled mean representations; linear probes on heuristic linguistic tasks.
- **Outcome:** Hybrid lanes capture strong surface linguistic signals; some lanes slightly outperform the mixed representation on specific tasks.
- **Representative results (accuracy):**
  - Token length bin: hybrid lane poly/trig 0.951 vs transformer mixed 0.961.
  - Word count bin: hybrid lane poly/trig 0.874 vs transformer mixed 0.932.
  - Avg word length bin: hybrid lane trig 0.903 vs transformer mixed 0.854.
  - Capitalization ratio bin: hybrid lane wavelet 0.845 vs transformer mixed 0.825.
  - Punctuation ratio bin: hybrid mixed 0.806 vs transformer mixed 0.796.
  - Digit presence: hybrid mixed/wavelet 0.922 vs transformer mixed 0.883.
- **Summary table (best per model):**
  - `avg_word_len_bin` | hybrid `lane_trig` 0.903 | transformer `mixed` 0.854
  - `capitalization_ratio_bin` | hybrid `lane_wavelet` 0.845 | transformer `mixed` 0.825
  - `digit_present` | hybrid `mixed` 0.922 | transformer `mixed` 0.883
  - `punctuation_ratio_bin` | hybrid `mixed` 0.806 | transformer `mixed` 0.796
  - `token_length_bin` | hybrid `lane_poly` 0.951 | transformer `mixed` 0.961
  - `word_count_bin` | hybrid `lane_poly` 0.874 | transformer `mixed` 0.932
- **Interpretation:** Hybrid lanes appear to separate different surface cues (trig for word length, wavelet for capitalization/digits). These are heuristic probes, not full POS/dependency tasks; deeper probes still needed.

## 2026-01-21 — Wikitext-2 POS/Dependency Probes (Phase 3)
- **Setup:** 128 validation samples, token-level representations with spaCy tags; linear probes on POS, dependency label, head direction, and head distance bins.
- **Outcome:** Wavelet lane carries the strongest syntactic signal; hybrid mixed also outperforms transformer mixed across POS/dep tasks.
- **Representative results (accuracy):**
  - POS: hybrid wavelet 0.804 vs hybrid mixed 0.760 vs transformer mixed 0.625.
  - Dependency label: hybrid wavelet 0.659 vs hybrid mixed 0.602 vs transformer mixed 0.508.
  - Head direction: hybrid wavelet 0.768 vs hybrid mixed 0.741 vs transformer mixed 0.667.
  - Head distance bin: hybrid wavelet 0.538 vs hybrid mixed 0.485 vs transformer mixed 0.421.
- **Summary table (best per model):**
  - `pos` | hybrid `lane_wavelet` 0.804 | transformer `mixed` 0.625
  - `dep` | hybrid `lane_wavelet` 0.659 | transformer `mixed` 0.508
  - `head_dir` | hybrid `lane_wavelet` 0.768 | transformer `mixed` 0.667
  - `head_dist_bin` | hybrid `lane_wavelet` 0.538 | transformer `mixed` 0.421
- **Interpretation:** Linguistic probes suggest wavelet features align most with syntactic structure, while trig/poly lag on POS/dep compared to wavelet and mixed.

## 2026-01-24 — Wikitext-2 Linguistic Follow-ups (Phase 4, 1k Steps)
- **Setup:** 1k steps, seed 1, 8k samples; wavelet capacity up, wavelet-only, and lane diversity regularizer.
- **Outcome:** Changes are minor at this short horizon; wavelet capacity up slightly improves eval, wavelet-only and lane diversity are roughly flat to slightly worse.
- **Results:**
  - Baseline eval 4.5102 / holdout 4.3923.
  - Wavelet capacity eval 4.5002 / holdout 4.3935.
  - Wavelet-only eval 4.5274 / holdout 4.3997.
  - Lane diversity eval 4.5085 / holdout 4.3889.
- **Probe deltas (best per task vs baseline):**
  - Heuristic probes: lane diversity improves word count (+0.049), wavelet capacity improves capitalization (+0.039) and punctuation (+0.029), wavelet-only improves digit presence (+0.029) but hurts token length (‑0.068).
  - POS/dep probes: lane diversity and wavelet capacity show small gains (≈+0.003 to +0.008); wavelet-only drops on POS/head direction (‑0.023/‑0.010).
- **Interpretation:** Minor probe gains favor wavelet capacity and lane diversity; wavelet-only weakens syntactic signals.

## 2026-01-25 — Context Length Sweep (Wikitext-2, 1k Steps)
- **Setup:** 1k steps, seed 1, 8k samples; context lengths 128–4096; hybrid vs matched transformer.
- **Outcome:** Eval/holdout losses are relatively flat across context lengths at this short horizon; no sharp dropoff observed up to 4096.
- **Representative results (eval / holdout):**
  - Hybrid: `ctx128 4.5707 / 4.4446`, `ctx512 4.4946 / 4.4059`, `ctx4096 4.5102 / 4.3923`.
  - Transformer: `ctx128 4.9491 / 4.7459`, `ctx512 4.9146 / 4.7108`, `ctx4096 4.8966 / 4.7089`.
- **Interpretation:** At 1k steps, both models show minimal context-length sensitivity; deeper runs may be needed to reveal dropoff behavior.

## 2026-01-15 — Training-Step Scaling (MPS)
- **Setup:** Training step benchmark at seq lengths 128/256/512, batch size 4, 5 steps, 1 warmup.
- **Outcome:** Transformer is faster per step and higher throughput at these lengths; hybrid uses less memory.
- **Representative results:**
  - Hybrid timing `128:196.76ms/2602.1tps/586MB`, `256:179.12ms/5716.8tps/586MB`, `512:317.72ms/6445.9tps/587MB`.
  - Transformer timing `128:85.26ms/6005.4tps/777MB`, `256:117.64ms/8704.7tps/778MB`, `512:226.00ms/9061.9tps/778MB`.
- **Interpretation:** Training throughput favors transformer at short lengths; long‑seq training scaling still needs measurement.

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
