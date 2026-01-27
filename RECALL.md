# WaveStack Recall Lane + Memory Cache

This note documents the planned recall-focused changes to WaveStack Hybrid.

## Goal
Add explicit recall capability without full attention, using a linear-time, causal memory lane plus a lightweight cache.

## Summary of Changes
- **Recall lane:** A new analytical lane that performs kernelized causal retrieval.
- **Memory cache:** An optional exponential decay on prefix memory to emphasize recent context while preserving long-range signals.
- **Config knobs:** Feature width, decay, and lane capacity so we can tune recall strength.

## Core Idea (Math Sketch)
Let hidden states be `h_t ∈ R^D` for `t = 1..T`.

1) Project to queries/keys/values:
- `q_t = W_q h_t`, `k_t = W_k h_t`, `v_t = W_v h_t`

2) Apply a positive feature map (linear attention style):
- `φ(x) = elu(x) + 1`
- `q̄_t = φ(q_t)`, `k̄_t = φ(k_t)`

3) Causal prefix memory:
- `K_t = decay * K_{t-1} + k̄_t`
- `KV_t = decay * KV_{t-1} + k̄_t ⊗ v_t`

4) Retrieve:
- `r_t = (q̄_t · KV_t) / (q̄_t · K_t + ε)`

The result `r_t ∈ R^D` is a recall vector per position and is treated like other lanes (fed through recomposition + mixing).

## Config Additions
- `decomposition.recall_features`: size of the kernel feature space.
- `decomposition.recall_decay`: exponential decay factor (1.0 = no decay).
- `decomposition.recall_epsilon`: numerical stability for denominator.
- `recomposition.recall_capacity`: width multiplier for the recall lane MLP.
- `enabled_lanes`: include `"recall"` and update `num_lanes` accordingly.

## Notes
- This keeps complexity **O(T·D·F)** (linear in sequence length).
- The decay cache gives a controllable long/short context tradeoff.
- We expect wavelet + recall to be the most informative pair for syntactic/recall probes.
