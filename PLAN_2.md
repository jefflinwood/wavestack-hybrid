# WaveStack Hybrid – Phase 2 Plan

This plan picks up after the initial expressivity/ablation work and focuses on stability, generalization, efficiency, and a clear written report.

## Phase 1 – Stability & Generalization
- **Longer Horizons:** Run 20k–50k steps on TinyStories for hybrid vs neural at 12m and/or 50m to confirm the gap persists.
- **Second Dataset:** Add a small non‑TinyStories corpus (e.g., MBPP or a small code/text mix) to validate transfer behavior. (completed: wikitext-2-raw-v1 runs)
- **Wikitext-2 Long Follow-up:** Run 10k-step Wikitext-2 baselines to test whether hybrid catches up with longer horizons. (completed)
- **Holdout Consistency:** Keep fixed‑seed train holdouts to monitor generalization drift.

## Phase 2 – Efficiency & Scaling
- **Runtime/Memory Sweep:** Use the new logging to capture tokens/s and peak memory at multiple sequence lengths.
- **Matched‑Param Baselines:** Compare hybrid vs a simple transformer at similar parameter counts, not just hidden_dim.
- **Cost Tables:** Report params/FLOPs/tokens‑per‑second side‑by‑side for all headline runs.

## Phase 3 – Targeted Ablations
- **Context Block Placement:** Compare pre vs post vs both (conv block) to isolate where context helps.
- **Lane Diversity at Scale:** Re-run cosine lane diversity at 50m to test scalability of the small gains.
- **Lane Pairing:** Evaluate best two‑lane combos (trig+wavelet) at 50m for efficiency/quality tradeoffs.

## Phase 4 – Findings Report
- **Summary:** Consolidate results into a concise report with charts + tables.
- **Decision Points:** State where WaveStack is ahead (quality vs cost), and where it needs optimization.
- **Next Hypotheses:** Identify what to try next (e.g., causal Fourier speedups, better recomposition depth).

## Execution Notes
- Use `EXPERIMENT_LOG.md` as the single source of truth for run metadata.
- Prefer 2–3 seeds for critical comparisons; 1 seed is acceptable for quick ablations.
- Capture runtime/memory metrics on the same machine for apples‑to‑apples comparisons.
