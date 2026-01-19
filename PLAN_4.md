# WaveStack Hybrid – Phase 4 Plan (CodeSearchNet)

This phase focuses on running and analyzing a CodeSearchNet training pass to test how the hybrid model behaves on code data.

## Phase 1 – CodeSearchNet Setup
- **Config:** Use a dedicated exp1 config targeting `code_search_net:python`.
- **Run:** Execute a 10k‑step training pass with a fixed seed and sample cap for reproducibility. (completed: hybrid run logged)
- **Logging:** Capture eval/holdout, params/FLOPs, runtime, and tokens/s in `EXPERIMENT_LOG.md`.

## Phase 2 – Baselines
- **Neural Baseline:** Run the neural baseline at matched size. (completed)
- **Transformer Baseline:** Optional matched‑param transformer baseline if time allows. (completed)

## Phase 3 – Analysis
- **Compare:** Evaluate hybrid vs baseline losses and runtime.
- **Notes:** Record any training instabilities or dataset‑specific behavior in `FINDINGS.md`.

## Execution Notes
- CodeSearchNet uses legacy dataset scripts; if that fails, supply local data via `dataset_name: json:/path/to/csn.jsonl` or `parquet:/path/to/csn.parquet`.
- Preferred local path: download `python.zip` to `data/python.zip`, then run `scripts/prepare_codesearchnet_zip.py` to produce `data/codesearchnet/python_train.jsonl` and `data/codesearchnet/python_valid.jsonl`.
- Use `dataset_name: code_search_net:python` with `train` and `validation` splits if your datasets version supports scripts.
- Keep device, batch size, and sample cap consistent across runs.
