# WaveStack Hybrid – Phase 5 Plan (Wikitext-2 Linguistic Analysis)

This phase focuses on probing linguistic structure in Wikitext-2 to see whether WaveStack lanes capture interpretable linguistic signals.

## Phase 1 – Dataset + Baselines
- **Config:** Use Wikitext-2 long-seq configs already in exp1 (hybrid + transformer) to keep comparability.
- **Run:** Re-run a short, fixed-seed pass if needed to produce fresh checkpoints for probing.
- **Logging:** Record eval/holdout, runtime, params/FLOPs in `EXPERIMENT_LOG.md`.
- **Runner:** `uv run python scripts/run_wikitext2_linguistic_phase1.py --device auto --max-steps 1000 --samples 8000 --seed 1` (phase1 configs use batch size 8 to avoid MPS 4GB NDArray limit).

## Phase 2 – Linguistic Probes (Offline)
- **Token-level probes:** POS tagging and dependency head prediction using hidden states from each lane output and the mixed representation.
- **Span-level probes:** Noun-phrase boundary detection or chunking using pooled lane outputs.
- **Frequency/scale probes:** Correlate trig/wavelet lane activations with token frequency and sentence length.
- **Ablation probes:** Compare full hybrid vs lane-ablated variants on the same probe tasks.

## Phase 3 – Evaluation + Analysis
- **Comparisons:** Hybrid vs transformer probes; lane-wise comparisons within hybrid.
- **Metrics:** Accuracy/F1 on probes; correlation strength for frequency/scale tests.
- **Interpretation:** Summarize whether any lane consistently aligns with linguistic structure.
- **Write-up:** Capture findings in `FINDINGS.md` with a short summary table.

## Phase 4 – Follow-ups (If Signals Appear)
- **Targeted tweaks:** Adjust lane capacities or regularizers to emphasize the strongest linguistic signals.
- **Generalization:** Verify on a second corpus slice or different Wikitext-2 seed.

## Execution Notes
- Keep seeds, sample caps, and max steps consistent across probe extractions.
- Store probe outputs under `outputs/` with date-stamped filenames.
- Keep probe scripts in `scripts/` and log exact commands in `EXPERIMENT_LOG.md`.
