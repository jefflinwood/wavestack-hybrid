# Recall Bias Head (Contextual Token Bias)

This note documents a recall mechanism that boosts logits for tokens already present in the context when a learned pattern says they are relevant.

## Goal
Provide explicit recall by injecting a context-conditioned bias into the token logits, without full attention.

## Core Idea
Add a **Recall Bias Head** that computes a per-token bias term `b_t` and adds it to the logits:

```
logits_t = lm_head(h_t) + beta * b_t
```

The bias should mainly affect tokens that have appeared in the prior context.

## Components
1) **Context summary** `c_t`  
   A causal summary derived from lane outputs (e.g., wavelet or recall lane).  
   Example: `c_t = f_lane(h_{<=t})` where `f_lane` is a causal filter.

2) **Context token sketch** `H_t`  
   A causal accumulator of previously seen tokens:
   - Exact: `H_t = decay * H_{t-1} + one_hot(x_t)`
   - Approximate: hashed or CountSketch to avoid full vocab-size vectors.

3) **Gating / alignment**  
   A learned gate from `c_t` that decides which tokens to boost:
   - `g_t = σ(W_g c_t)`  
   - `b_t = g_t ⊙ H_t` (exact)  
   - or `b_t = W_out (g_t ⊙ H_t)` for sketch variants.

## Why This Helps
- **Recall by construction:** only tokens in context can be boosted.
- **Math-friendly:** uses causal accumulation and linear projections.
- **Pattern-driven:** the model learns which context patterns warrant recall boosts.

## Config Knobs (Proposed)
- `recall_bias.enabled` (bool)
- `recall_bias.decay` (float, 0–1)
- `recall_bias.beta` (float, scaling for bias injection)
- `recall_bias.sketch_dim` (int, if using hashing)
- `recall_bias.use_wavelet_context` (bool, use wavelet lane to form `c_t`)

## Open Questions
- Exact vs sketch bias (memory vs precision tradeoff).
- Which lane features best predict recall gates (wavelet vs mixed).
- Whether to apply bias at all positions or only near the sequence end.
