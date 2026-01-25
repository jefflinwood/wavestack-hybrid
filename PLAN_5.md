# WaveStack Hybrid – Phase 5 Plan (Wikitext-2 Linguistic Analysis)

This phase focuses on probing linguistic structure in Wikitext-2 to see whether WaveStack lanes capture interpretable linguistic signals.

## Phase 1 – Dataset + Baselines
- **Config:** Use Wikitext-2 long-seq configs already in exp1 (hybrid + transformer) to keep comparability.
- **Run:** Re-run a short, fixed-seed pass if needed to produce fresh checkpoints for probing.
- **Logging:** Record eval/holdout, runtime, params/FLOPs in `EXPERIMENT_LOG.md`.
- **Runner:** `uv run python scripts/run_wikitext2_linguistic_phase1.py --device auto --max-steps 1000 --samples 8000 --seed 1` (phase1 configs use batch size 8 to avoid MPS 4GB NDArray limit).
  - **Status:** Completed (Phase 1 Wikitext-2 long-seq runs logged on 2026-01-20).
  - **Note:** Hybrid and transformer checkpoints now write to `./checkpoints/phase5/hybrid` and `./checkpoints/phase5/transformer` to avoid collisions.

## Phase 2 – Linguistic Probes (Offline)
- **Token-level probes:** POS tagging and dependency head prediction using hidden states from each lane output and the mixed representation.
- **Span-level probes:** Noun-phrase boundary detection or chunking using pooled lane outputs.
- **Frequency/scale probes:** Correlate trig/wavelet lane activations with token frequency and sentence length.
- **Ablation probes:** Compare full hybrid vs lane-ablated variants on the same probe tasks.
  - **Extraction:** `uv run python scripts/extract_wikitext2_probe_reprs.py --config experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml --checkpoint checkpoints/phase5/hybrid/checkpoint_001000.pt --device auto --split validation --seq-len 512 --max-samples 512 --pool mean --output outputs/probes/wikitext2_hybrid_phase1.pt`.
  - **Probe run:** `uv run python scripts/run_wikitext2_linguistic_probes.py --hybrid outputs/probes/wikitext2_hybrid_phase1.pt --transformer outputs/probes/wikitext2_transformer_phase1.pt`.
  - **Status:** Completed (results in `outputs/probes/wikitext2_probe_results.jsonl`).

## Phase 3 – Evaluation + Analysis
- **Comparisons:** Hybrid vs transformer probes; lane-wise comparisons within hybrid.
- **Metrics:** Accuracy/F1 on probes; correlation strength for frequency/scale tests.
- **Interpretation:** Summarize whether any lane consistently aligns with linguistic structure.
- **Write-up:** Capture findings in `FINDINGS.md` with a short summary table.
  - **Status:** Completed (heuristic + POS/dep probes summarized in `FINDINGS.md`).
  - **POS/Dep extraction:** Re-extract with token-level reps (pool=none), e.g. `uv run python scripts/extract_wikitext2_probe_reprs.py --config experiments/exp1_expressivity/config_AU_hybrid_12m_wikitext2_longseq_phase1.yaml --checkpoint checkpoints/phase5/hybrid/checkpoint_001000.pt --device auto --split validation --seq-len 256 --max-samples 128 --pool none --output outputs/probes/wikitext2_hybrid_phase1_tokens.pt`.
  - **POS/Dep probes:** `uv run python scripts/run_wikitext2_pos_dependency_probes.py --hybrid outputs/probes/wikitext2_hybrid_phase1_tokens.pt --transformer outputs/probes/wikitext2_transformer_phase1_tokens.pt --spacy-model en_core_web_sm`.
  - **Status:** POS/dep probes completed; results summarized in `FINDINGS.md`.

## Phase 4 – Follow-ups (If Signals Appear)
- **Targeted tweaks:** Adjust lane capacities or regularizers to emphasize the strongest linguistic signals.
- **Generalization:** Verify on a second corpus slice or different Wikitext-2 seed.
  - **Experiments:** Wavelet capacity up, wavelet-only, lane-diversity regularizer.
  - **Runner:** `uv run python scripts/run_wikitext2_linguistic_phase4.py --device auto --max-steps 1000 --samples 8000 --seed 1`.
  - **Status:** Runs completed; compare probe signals for each variant.
  - **Probe extraction:** `uv run python scripts/run_wikitext2_phase4_probe_extraction.py --device auto --checkpoint-step 1000`.
  - **Heuristic probes:** `uv run python scripts/run_wikitext2_multi_linguistic_probes.py --inputs baseline=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_phase1_mean.pt wavelet_capacity=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_wavelet_capacity_mean.pt wavelet_only=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_wavelet_only_mean.pt lane_diversity=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_lane_diversity_mean.pt --output outputs/probes/wikitext2_phase4_probe_results.jsonl`.
  - **POS/dep probes:** `uv run python scripts/run_wikitext2_multi_pos_dependency_probes.py --inputs baseline=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_phase1_tokens.pt wavelet_capacity=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_wavelet_capacity_tokens.pt wavelet_only=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_wavelet_only_tokens.pt lane_diversity=outputs/probes/phase5/hybrid_12m_wikitext2_longseq_lane_diversity_tokens.pt --spacy-model en_core_web_sm --output outputs/probes/wikitext2_phase4_pos_dep_results.jsonl`.
  - **Status:** Probe comparisons completed; summarized in `FINDINGS.md`.

## Execution Notes
- Keep seeds, sample caps, and max steps consistent across probe extractions.
- Store probe outputs under `outputs/` with date-stamped filenames.
- Keep probe scripts in `scripts/` and log exact commands in `EXPERIMENT_LOG.md`.
