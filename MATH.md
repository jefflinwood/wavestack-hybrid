# WaveStack Hybrid Math Notes

This file summarizes the mathematical ingredients behind the model. It is intentionally short and focused on the pieces that differ from standard Transformer-style baselines.

## Notation
- Sequence length: `S`
- Hidden size: `H`
- Input tokens: `x_1..x_S`
- Embeddings: `E in R^{S x H}`
- Lane index: `l in {poly, trig, wavelet}`

## Goal
Predict the next token at each position. The model produces logits `Y in R^{S x V}` and is trained with a causal shift (logits at position `t` are scored against the label at `t+1`).

## Chebyshev Lane (Polynomial Basis)
Chebyshev polynomials `T_k(z)` form an orthogonal basis on `[-1, 1]` with recurrence:

```
T_0(z) = 1
T_1(z) = z
T_k(z) = 2 z T_{k-1}(z) - T_{k-2}(z)
```

For sequence positions mapped to `z_t in [-1, 1]`, each hidden channel is approximated as:

```
E_t,h ~= sum_{k=0..K-1} c_{k,h} T_k(z_t)
```

The implementation estimates coefficients with a projection and reconstructs a smoothed signal. In causal mode, the coefficients at time `t` use only the prefix `1..t`.

## Fourier Lane (Trigonometric Basis)
Fourier bases decompose a sequence into frequency components. The non-causal path uses the rFFT to keep only the first `F` frequencies and reconstructs a band-limited signal.

The causal approximation uses prefix projections onto sine/cosine bases:

```
E_t,h ~= sum_{f=0..F-1} (a_{f,t,h} cos(2 pi f t / S) + b_{f,t,h} sin(2 pi f t / S))
```

with `a_{f,t,h}` and `b_{f,t,h}` computed from prefix averages so that time `t` only depends on `1..t`.

## Wavelet Lane (Multi-Scale Smoothing)
Wavelets capture local trends at multiple scales via low/high components. We approximate this with average pooling and residuals:

```
low_t = avg_pool(E, kernel=2^level)
detail_t = E_t - low_t
```

In causal mode, pooling only uses past context by left-padding and applying a one-sided average.

## Recomposition
Each lane output is passed through a small MLP:

```
R_l = MLP_l(Decomp_l(E))
```

These lane-specific representations return to `H` so they can be mixed.

## Mixing
Lane representations are combined by one of:
- **Gated sum**: weights per lane and per token position, then sum.
- **Lane attention**: lanes attend to each other at each position.
- **MLP**: concatenate lanes and map back to `H`.

The mixer produces a single sequence representation `M in R^{S x H}`.

## Output and Loss
Logits are:

```
Y = Linear(LayerNorm(M))
```

Loss uses a causal shift:

```
L_ce = CE(Y_{1..S-1}, labels_{2..S})
```

Optional auxiliary losses can be added for reconstruction or lane balance.
