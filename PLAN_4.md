# WaveStack Hybrid – Phase 4: Scaling Validation

This plan focuses on validating the core architectural hypothesis: **WaveStack scales linearly (O(n)) while Transformers scale quadratically (O(n²))** at long sequences (4K+ tokens).

## Hypothesis

At current benchmarked scales (64-512 tokens), both architectures appear linear:
- Transformer: exponent = 1.008
- WaveStack: exponent = 1.078

**Why this is misleading:**
- Quadratic attention cost is dominated by fixed overheads at short sequences
- Need to test at 2K-16K tokens where O(n²) becomes the bottleneck
- Memory scaling (also O(n²) for attention) should show clear divergence

**Expected outcomes at 4K+ tokens:**
- Transformer: time ~ n^1.8-2.0, memory ~ n²
- WaveStack: time ~ n^1.05-1.10, memory ~ n
- **Crossover point**: WaveStack becomes faster than Transformer at ~2-4K tokens

## Phase 1 – Extended Scaling Benchmarks

### 1.1 Long-Sequence Infrastructure
- **Goal:** Enable testing at 2K-16K token sequences
- **Tasks:**
  - Extend `max_seq_len` in model configs to 16384
  - Add memory-efficient data loading for long sequences
  - Implement gradient checkpointing for training at scale
  - Add OOM (out-of-memory) handling and fallback mechanisms

### 1.2 Comprehensive Benchmark Suite
- **Goal:** Measure time and memory scaling across full range
- **Script:** `scripts/benchmark_scaling_extended.py`
- **Test configurations:**
  - Sequence lengths: 512, 1K, 2K, 4K, 8K, 16K tokens
  - Batch sizes: 1, 2, 4 (reduce as sequence grows to fit memory)
  - Models: hybrid_12m, hybrid_50m, transformer_12m, transformer_50m
  - Metrics: time/token, peak memory, throughput (tokens/s)

### 1.3 Scaling Exponent Analysis
- **Goal:** Fit power laws to measure actual complexity
- **Analysis:**
  - Fit time ~ seq_len^α for each model
  - Fit memory ~ seq_len^β for each model
  - Compute crossover points: where WaveStack overtakes Transformer
  - Statistical significance: bootstrap 95% confidence intervals on exponents

### 1.4 Gradient Memory Profiling
- **Goal:** Understand memory scaling during training (not just inference)
- **Tasks:**
  - Measure peak memory during forward pass only
  - Measure peak memory during forward + backward pass
  - Profile activation memory vs parameter memory
  - Test gradient checkpointing impact on memory/speed tradeoff

## Phase 2 – Performance Optimization

### 2.1 Vectorized Causal Operations
- **Goal:** Eliminate O(n × k) loops in causal decompositions
- **Files:** `src/wavestack_hybrid/models/decomposition/fourier.py`, `chebyshev.py`
- **Current bottleneck:**
  ```python
  # fourier.py:68-74 - O(n × num_freqs) with sequential loop
  for freq_idx in range(num_freqs):
      coeff_cos = torch.cumsum(x * cos_vec, dim=1)
      coeff_sin = torch.cumsum(x * sin_vec, dim=1)
      reconstructed += gate * (coeff_cos * cos_vec + coeff_sin * sin_vec)
  ```
- **Optimization:**
  ```python
  # Vectorized version - O(n) with single cumsum
  cos_basis = cos_basis.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, num_freqs)
  sin_basis = sin_basis.unsqueeze(0).unsqueeze(2)
  x_expanded = x.unsqueeze(-1)  # (batch, seq, hidden, 1)
  weighted_cos = x_expanded * cos_basis  # (batch, seq, hidden, num_freqs)
  weighted_sin = x_expanded * sin_basis
  coeffs_cos = torch.cumsum(weighted_cos, dim=1) / denom
  coeffs_sin = torch.cumsum(weighted_sin, dim=1) / denom
  gates = freq_gates.view(1, 1, 1, num_freqs)
  reconstructed = (coeffs_cos * cos_basis + coeffs_sin * sin_basis) * gates
  reconstructed = reconstructed.sum(dim=-1)  # Reduce frequency dim
  ```
- **Expected speedup:** 2-3x for Fourier lane, 10-20% overall
- **Validation:** Re-run expressivity tests to ensure accuracy maintained

### 2.2 Adaptive Lane Complexity
- **Goal:** Reduce analytical lane cost at long sequences
- **Strategy:** Scale down decomposition parameters as sequence length grows
- **Implementation:**
  - `poly_order`: 4 at 512 tokens → 2 at 8K tokens
  - `num_freqs`: 64 at 512 tokens → 32 at 8K tokens
  - `wavelet_levels`: 3 at 512 tokens → 2 at 8K tokens
- **Config:** Add `adaptive_complexity: bool` and `complexity_schedule` parameters
- **Rationale:** At long sequences, lower-order patterns still capture structure while reducing compute

### 2.3 Mixed Precision Training
- **Goal:** Reduce memory footprint with float16/bfloat16
- **Tasks:**
  - Add `torch.amp.autocast` support to trainer
  - Test numerical stability of analytical decompositions in fp16
  - Compare memory reduction: fp32 vs fp16 vs bf16
  - Measure impact on final loss/perplexity

### 2.4 Flash Attention Baseline
- **Goal:** Compare against state-of-the-art optimized attention
- **Tasks:**
  - Integrate FlashAttention-2 into transformer baseline
  - Re-benchmark with optimized attention
  - Compare: naive attention vs FlashAttention vs WaveStack
  - Document: WaveStack advantage holds even vs optimized attention

## Phase 3 – Training at Scale

### 3.1 Long-Context Dataset Preparation
- **Goal:** Create datasets with naturally long sequences
- **Datasets:**
  - **Books (PG-19):** 2-8K token book excerpts
  - **Code (The Stack):** Full file contents, 1-4K tokens
  - **ArXiv papers:** Abstract + introduction, 1-3K tokens
  - **Long-form QA:** Natural Questions with full context
- **Preprocessing:**
  - Tokenize and pack to target lengths (2K, 4K, 8K)
  - Create train/eval/holdout splits
  - Compute dataset statistics (mean/median/95th percentile length)

### 3.2 Long-Context Training Experiments
- **Goal:** Train both architectures on long sequences, measure convergence
- **Configurations:**
  - **Baseline:** hybrid_12m vs transformer_12m on 2K token sequences
  - **Scale test:** hybrid_12m vs transformer_12m on 4K token sequences
  - **Stress test:** hybrid_50m vs transformer_50m on 8K token sequences (if memory allows)
- **Training setup:**
  - Max steps: 5000 (lightweight, focus on convergence speed)
  - Gradient accumulation to maintain effective batch size
  - Log: loss, perplexity, memory, tokens/s every 100 steps
- **Metrics:**
  - Final eval loss at 5000 steps
  - Wall-clock time to reach target perplexity
  - Peak memory usage during training
  - Tokens processed per second

### 3.3 Context Length Extrapolation
- **Goal:** Test generalization from short to long contexts
- **Experiment:**
  - Train hybrid_12m on 512-token sequences (3000 steps)
  - Evaluate on 512, 1K, 2K, 4K token sequences
  - Train transformer_12m on 512-token sequences
  - Evaluate on 512, 1K, 2K, 4K token sequences
- **Hypothesis:** Analytical decompositions should extrapolate better than attention
- **Analysis:** Plot perplexity vs context length for both models

### 3.4 Positional Encoding Analysis
- **Goal:** Understand if positional embeddings limit long-context performance
- **Current:** Learned position embeddings up to max_seq_len
- **Experiments:**
  - Test RoPE (Rotary Position Embeddings) for WaveStack
  - Test ALiBi (Attention with Linear Biases) for Transformer baseline
  - Compare extrapolation quality with different PE schemes
- **Implementation:** Add `position_encoding: Literal["learned", "rope", "alibi"]` to config

## Phase 4 – Memory Efficiency Analysis

### 4.1 Activation Memory Profiling
- **Goal:** Understand where memory is consumed
- **Tool:** PyTorch Memory Profiler
- **Analysis:**
  - Profile forward pass only: measure intermediate activations
  - Profile backward pass: measure gradient storage
  - Identify peak memory bottleneck (embedding, lanes, mixing, lm_head)
- **Compare:** Transformer attention vs WaveStack decomposition memory

### 4.2 KV Cache Simulation (Inference)
- **Goal:** Model inference memory for autoregressive generation
- **Transformer:** Requires O(n) KV cache per layer
- **WaveStack:** No KV cache needed (stateless decomposition)
- **Benchmark:**
  - Simulate generation of 100 tokens with varying context lengths
  - Measure cumulative memory as context grows
  - Plot: memory vs generated_tokens for both architectures
- **Expected result:** Transformer memory grows linearly, WaveStack stays constant

### 4.3 Gradient Checkpointing Trade-offs
- **Goal:** Quantify speed/memory tradeoff
- **Configurations:**
  - No checkpointing (baseline)
  - Checkpoint decomposition layers
  - Checkpoint recomposition layers
  - Checkpoint both
- **Metrics:**
  - Peak memory reduction (%)
  - Training speed slowdown (%)
  - Find optimal checkpointing strategy for long sequences

### 4.4 Batch Size Scaling
- **Goal:** Determine maximum throughput configuration
- **Experiment:**
  - For each sequence length (512, 1K, 2K, 4K), find max batch size that fits in memory
  - Measure tokens/s at max batch size
  - Plot: throughput vs sequence length vs batch size
- **Expected result:** WaveStack sustains higher batch sizes at long sequences

## Phase 5 – Findings & Publication

### 5.1 Scaling Law Visualization
- **Goal:** Create compelling plots showing scaling divergence
- **Plots:**
  - **Time scaling:** Log-log plot of time vs seq_len (lines for hybrid/transformer)
  - **Memory scaling:** Log-log plot of memory vs seq_len
  - **Throughput:** Tokens/s vs seq_len (show crossover point)
  - **Training efficiency:** Loss curves at 2K, 4K tokens (time to convergence)
  - **Cost analysis:** Projected training cost for 1B tokens at different context lengths

### 5.2 Scaling Report
- **Title:** "Linear Scaling in Language Models: Analytical Decomposition vs Quadratic Attention"
- **Sections:**
  1. Introduction: The quadratic attention bottleneck
  2. WaveStack Architecture: O(n) decomposition lanes
  3. Scaling Experiments: 512-16K tokens
  4. Results: Measured exponents, crossover analysis
  5. Training Efficiency: Long-context convergence
  6. Memory Analysis: Activation and inference memory
  7. Optimizations: Vectorized causal ops, adaptive complexity
  8. Discussion: Practical implications for long-context LLMs
  9. Limitations: Where analytical decomposition struggles

### 5.3 Optimization Guide
- **Document:** `docs/SCALING_GUIDE.md`
- **Contents:**
  - Recommended settings for different sequence lengths
  - Memory optimization checklist
  - Performance tuning tips
  - Common bottlenecks and solutions
  - Hardware-specific recommendations (GPU vs CPU vs MPS)

### 5.4 Benchmark Dashboard
- **Tool:** Interactive notebook or web dashboard
- **Features:**
  - Upload new benchmark results
  - Compare across models, sequence lengths, hardware
  - Automatically fit scaling exponents
  - Generate plots and tables
  - Export formatted results for papers

## Phase 6 – Validation & Stress Testing

### 6.1 Numerical Stability at Scale
- **Goal:** Ensure analytical decompositions remain stable at 16K tokens
- **Tests:**
  - Chebyshev: Check for numerical overflow/underflow
  - Fourier: Verify FFT accuracy at long sequences
  - Wavelet: Test pooling at extreme scales (kernel_size = 2^10 = 1024)
- **Metrics:** Reconstruction error, gradient norms, NaN detection

### 6.2 Edge Case Testing
- **Scenarios:**
  - Very short sequences (16 tokens) with high poly_order
  - Very long sequences (32K tokens) if memory allows
  - Batch size = 1 (no batching overhead)
  - Batch size = 64 (maximum parallelism)
- **Validation:** Model doesn't crash, produces reasonable outputs

### 6.3 Multi-GPU Scaling (Optional)
- **Goal:** Test if WaveStack advantages hold with model parallelism
- **Setup:**
  - Distribute layers across 2-4 GPUs
  - Measure communication overhead
  - Compare: single GPU vs multi-GPU throughput
- **Hypothesis:** Linear operations should parallelize better than attention

### 6.4 Real-World Task Evaluation
- **Goal:** Validate that long-context capability improves task performance
- **Tasks:**
  - Long-document QA: QASPER dataset (scientific papers)
  - Book summarization: BookSum (long-form summaries)
  - Code completion: HumanEval with full file context
- **Metrics:** Task-specific accuracy, not just perplexity
- **Compare:** hybrid vs transformer on same tasks

## Implementation Roadmap

### Week 1: Benchmarking Infrastructure
- **Days 1-2:** Extend max_seq_len configs, add long-sequence data loading
- **Days 3-4:** Implement `benchmark_scaling_extended.py` script
- **Day 5:** Run initial benchmarks at 512-4K tokens, verify setup

### Week 2: Optimization
- **Days 1-2:** Vectorize causal Fourier and Chebyshev operations
- **Days 3-4:** Implement adaptive complexity and mixed precision
- **Day 5:** Re-benchmark optimized WaveStack, measure speedup

### Week 3: Training Experiments
- **Days 1-2:** Prepare long-context datasets (PG-19, The Stack)
- **Days 3-5:** Run 2K and 4K token training experiments for both models

### Week 4: Analysis & Reporting
- **Days 1-2:** Memory profiling and KV cache simulation
- **Days 3-4:** Generate plots, fit scaling laws, write findings
- **Day 5:** Create scaling guide and benchmark dashboard

## Success Metrics

### Quantitative Goals
- **Scaling exponents:** Measure αₜ (time) and αₘ (memory) for both architectures
  - Target: Transformer αₜ > 1.8, WaveStack αₜ < 1.15
  - Target: Transformer αₘ > 1.7, WaveStack αₘ < 1.05
- **Crossover point:** Identify sequence length where WaveStack becomes faster
  - Hypothesis: 2K-4K tokens
- **Memory efficiency:** At 8K tokens, WaveStack uses <50% memory of Transformer
- **Training speed:** At 4K tokens, WaveStack trains ≥2x faster than Transformer

### Qualitative Goals
- Demonstrate WaveStack handles 16K tokens on single GPU (A100 40GB)
- Show Transformer struggles beyond 4K tokens without optimization
- Prove vectorized optimizations maintain accuracy (loss delta < 1%)
- Document clear practical advantage for long-context applications

## Technical Dependencies

### Software
- PyTorch ≥ 2.0 (for `torch.compile` and improved memory profiling)
- FlashAttention-2 (for optimized transformer baseline)
- `torch.profiler` for detailed performance analysis
- Weights & Biases or TensorBoard for experiment tracking

### Hardware
- **Minimum:** Single GPU with 24GB memory (RTX 3090, RTX 4090)
- **Recommended:** A100 40GB or H100 80GB for 8K+ token experiments
- **Ideal:** Multi-GPU setup for parallel experiments

### Data
- PG-19 books dataset (~11GB)
- The Stack (code) - filtered subset (~5GB)
- ArXiv papers - abstract+intro subset (~2GB)

## Open Questions

1. **Optimal hyperparameters at scale:** Does learning rate need adjustment for 4K+ tokens?
2. **Lane importance:** Do certain lanes (wavelet?) become more critical at long contexts?
3. **Recomposition depth:** Should recomposition networks be deeper for long sequences?
4. **Context blocks:** Do convolution-based context blocks help or hurt at scale?
5. **Batch size strategy:** Fixed tokens/batch vs fixed sequences/batch?
6. **Extrapolation limits:** What's the maximum context length WaveStack can handle before degrading?

## Risk Mitigation

### Risk: OOM at long sequences
- **Mitigation:** Implement gradient checkpointing, reduce batch size, use fp16
- **Fallback:** Test on H100 80GB if A100 40GB insufficient

### Risk: Vectorization breaks accuracy
- **Mitigation:** Unit tests comparing vectorized vs original implementation
- **Validation:** Re-run expressivity experiments, ensure loss within 1%

### Risk: Long-context training takes too long
- **Mitigation:** Use smaller models (12m only), reduce max_steps to 3000
- **Fallback:** Focus on inference benchmarks if training impractical

### Risk: Transformer with FlashAttention matches WaveStack
- **Response:** Emphasize memory advantages and interpretability
- **Analysis:** Compute projected cost at 32K+ tokens where even FlashAttention struggles

## Expected Outcomes & Impact

### If successful:
- **WaveStack demonstrates clear 2-5x speed advantage at 4K+ tokens**
- **Memory usage stays linear while Transformer explodes quadratically**
- **Training at 4K tokens is practical on consumer GPUs**
- **Establishes WaveStack as viable architecture for long-context applications**

### Applications unlocked:
- Long-document understanding (legal, medical, scientific papers)
- Full-file code completion and analysis
- Extended dialogue systems with long conversation history
- Book-length summarization and question answering

### Research contributions:
- Empirical validation of O(n) vs O(n²) scaling at relevant scales
- Optimization techniques for causal analytical decompositions
- Benchmark suite for long-context model evaluation
- Blueprint for future sub-quadratic architectures

## Notes

- **Prioritize 4K and 8K benchmarks** - these are most relevant for practical applications
- **Document failure modes** - be transparent about where WaveStack struggles
- **Compare fairly** - use FlashAttention for transformer, optimized code for WaveStack
- **Focus on wall-clock time** - theoretical complexity matters less than practical speed
- **Memory is key differentiator** - even if speed is similar, memory efficiency enables larger batch sizes
