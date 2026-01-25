# WaveStack Hybrid – Phase 6 Plan (Context Size + Recall)

This phase explores how context size affects WaveStack performance and whether the model shows a context-window dropoff similar to attention.

## Context Handling (Explicit)
- **Input format:** The model consumes the full sequence of tokens (`input_ids` shape `batch x seq_len`) each forward pass.
- **Prediction target:** It predicts the next token at each position (causal shift), not a single-token classification.
- **Context window:** The effective context length is bounded by `model.max_seq_len` and the sequence length fed to the model.

## Phase 1 – Context Length Sweep (Training-Time Eval)
- **Goal:** Measure eval/holdout loss as a function of context length.
- **Protocol:** Train short runs at fixed steps, varying `max_seq_len` and evaluation sequence length.
- **Suggested lengths:** 128 / 256 / 512 / 1024 / 2048 / 4096 (where feasible).
- **Comparisons:** Hybrid vs transformer (matched params) on Wikitext-2; include a short-seq baseline (512).
- **Logging:** Record loss, runtime, tokens/s, and memory in `EXPERIMENT_LOG.md`.
  - **Runner:** `uv run python scripts/run_context_length_sweep.py --device auto --max-steps 1000 --samples 8000 --seed 1`.
  - **Status:** Completed (results summarized in `FINDINGS.md`).

## Phase 2 – Context Dropoff Test (Needle-in-Haystack)
- **Goal:** Test whether information recall degrades as the needle moves farther back in context.
- **Protocol:** Synthetic dataset with a key-value phrase early in context, query at end; measure exact match accuracy.
- **Sweep:** Place the key at multiple offsets (e.g., 64/128/256/512/1024/2048 tokens).
- **Outputs:** Accuracy vs distance plots for hybrid vs transformer.

## Phase 3 – Recall Probes on Natural Text
- **Goal:** Evaluate recall-like behavior on Wikitext-2 without synthetic cues.
- **Protocol:** Use masked cloze-style prompts from validation; measure whether prior noun phrases can be recovered.
- **Metrics:** Top-1/Top-5 accuracy for target tokens conditioned on varying prefix lengths.

## Phase 4 – Analysis + Findings
- **Summaries:** Plot loss vs context length and accuracy vs offset.
- **Interpretation:** Note whether dropoff is gradual or sharp, and whether lanes differ in recall signal.
- **Write-up:** Record findings in `FINDINGS.md` and link to key plots in `outputs/`.

## Execution Notes
- Keep seeds and sample caps fixed across runs.
- Use lower batch size on long contexts for MPS to avoid NDArray size errors.
- Store sweep results in `outputs/` with date-stamped filenames.
