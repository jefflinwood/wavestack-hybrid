# WaveStack Hybrid – Phase 3: Traceability & Interpretability

This plan focuses on the core hypothesis that analytical decompositions enable **traceable predictions**: the ability to map a predicted token back to specific patterns in the training data via interpretable mathematical decompositions.

## Vision

Enable WaveStack to answer:
- "Which training sequences contributed to this prediction?"
- "What mathematical patterns (polynomial trends, frequencies, wavelets) drove this output?"
- "How does this compare to memorization in attention-based models?"

## Phase 1 – Coefficient Extraction Infrastructure

### 1.1 Raw Coefficient Access
- **Goal:** Extract decomposition coefficients before recomposition networks.
- **Implementation:**
  - Add `return_coefficients=True` flag to `HybridWaveStack.forward()`
  - Return raw Chebyshev coefficients (poly_order values per position)
  - Return raw Fourier coefficients (complex amplitudes per frequency)
  - Return raw wavelet coefficients (values per scale/location)
  - Store in structured format: `{"poly": tensor, "trig": tensor, "wavelet": tensor}`

### 1.2 Coefficient Storage Module
- **Goal:** Index training data by coefficient patterns.
- **Files:** `src/wavestack_hybrid/analysis/coefficient_tracker.py`
- **Features:**
  - `CoefficientTracker` class for storing coefficient fingerprints
  - Efficient storage (sample every N steps to manage memory)
  - Per-lane coefficient statistics (mean, std, percentiles)
  - Optional compression for large-scale indexing
  - Save/load functionality for persistence across runs

### 1.3 Training Integration
- **Goal:** Collect coefficient data during training.
- **Files:** Extend `src/wavestack_hybrid/training/trainer.py`
- **Features:**
  - Add `track_coefficients=True` config option
  - Sample coefficient fingerprints periodically (e.g., every 100 steps)
  - Store mapping: `coefficient_pattern → (step, batch_idx, sequence_slice)`
  - Log coefficient distribution statistics to wandb/tensorboard
  - Create checkpoint with coefficient index

## Phase 2 – Pattern Matching & Retrieval

### 2.1 Similarity Metrics
- **Goal:** Define distance/similarity measures in coefficient space.
- **Files:** `src/wavestack_hybrid/analysis/similarity.py`
- **Metrics:**
  - Cosine similarity in coefficient space (fast, interpretable)
  - Euclidean distance with per-lane weighting
  - Correlation-based matching for periodic patterns (Fourier)
  - Multi-scale matching for wavelets (per-level similarity)
  - Combined metric: weighted sum across lanes

### 2.2 Training Data Index
- **Goal:** Build searchable index of training coefficient patterns.
- **Files:** `src/wavestack_hybrid/analysis/pattern_index.py`
- **Features:**
  - `PatternIndex` class wrapping coefficient tracker
  - KNN search in coefficient space (using FAISS or similar)
  - Top-K retrieval: find most similar training sequences
  - Lane-specific retrieval: "find sequences with similar Fourier patterns"
  - Filtering by training step/epoch range

### 2.3 Influence Quantification
- **Goal:** Measure training data influence on predictions.
- **Approach:**
  - At inference: extract test sequence coefficients
  - Search training index for similar patterns
  - Compute influence score = f(similarity, lane_contribution)
  - Return ranked list of influential training examples
  - Compare to gradient-based influence functions (optional)

## Phase 3 – Interpretability Tools

### 3.1 Coefficient Visualization
- **Goal:** Visualize what the model "sees" in coefficient space.
- **Files:** Extend `src/wavestack_hybrid/analysis/visualization.py`
- **Visualizations:**
  - Polynomial trends: plot fitted curves per token position
  - Frequency spectrum: bar chart of active Fourier components
  - Wavelet heatmaps: time-scale decomposition plots
  - Lane attribution over sequence: stacked area chart (already exists)
  - Coefficient evolution during training: line plots

### 3.2 Traceability Dashboard
- **Goal:** Interactive tool for exploring prediction provenance.
- **Files:** `notebooks/traceability_explorer.ipynb` or Streamlit app
- **Features:**
  - Input: test sequence
  - Output:
    - Top-K similar training sequences
    - Coefficient pattern breakdown per lane
    - Visual diff: test vs training patterns
    - Lane contribution percentages
    - "Why this prediction?" explanation in natural language

### 3.3 Memorization Analysis
- **Goal:** Quantify and characterize memorization behavior.
- **Experiments:**
  - Exact match detection: find training sequences that appear in test
  - Near-duplicate detection via coefficient similarity
  - Memorization score: % of test patterns with high-similarity training match
  - Compare: WaveStack memorization vs Transformer attention memorization
  - Analyze: which lanes memorize most? (hypothesis: wavelets > poly)

## Phase 4 – Validation & Experiments

### 4.1 Proof-of-Concept Study
- **Dataset:** TinyStories (small, known to work well)
- **Setup:**
  - Train hybrid_12m with coefficient tracking enabled
  - Index all training sequences (or sample 10k)
  - Run inference on 1000 test sequences
  - For each prediction, find top-5 similar training sequences
- **Metrics:**
  - Retrieval precision: % of retrieved sequences that are semantically relevant
  - Coefficient similarity vs prediction confidence correlation
  - Per-lane contribution to similarity matching

### 4.2 Cross-Architecture Comparison
- **Goal:** Compare traceability to attention-based models.
- **Baseline:** Transformer with activation-based retrieval
  - Extract hidden states from transformer
  - KNN search in activation space
  - Compare retrieval quality vs WaveStack coefficient retrieval
- **Hypothesis:** Coefficient-based retrieval is more interpretable (can explain *why* sequences match)

### 4.3 Ablation Studies
- **Lane importance:** Which lane coefficients are most predictive of influence?
- **Temporal scope:** How much context is needed for accurate pattern matching?
- **Recomposition opacity:** How much does the learned recomposition layer obscure traceability?
- **Scale testing:** Does traceability hold at 50m+ parameters?

## Phase 5 – Research Outputs

### 5.1 Technical Report
- **Title:** "Traceable Language Models via Analytical Decomposition"
- **Sections:**
  - Introduction: The interpretability challenge
  - Method: WaveStack architecture + coefficient indexing
  - Experiments: Proof-of-concept results
  - Analysis: Memorization patterns, lane contributions
  - Comparison: vs attention-based influence methods
  - Discussion: Implications for model auditing, copyright, debugging

### 5.2 Demos & Artifacts
- Interactive notebook showcasing traceability
- Video walkthrough: "Explaining a prediction step-by-step"
- Release coefficient index for pretrained models
- Open-source traceability toolkit

### 5.3 Future Directions
- Federated traceability: trace predictions without exposing training data
- Real-time traceability for production systems
- Copyright detection: identify training data sources
- Debugging tool: find training examples causing failures

## Implementation Roadmap

### Week 1: Infrastructure
- Day 1-2: Implement coefficient extraction (`return_coefficients=True`)
- Day 3-4: Build `CoefficientTracker` with storage/retrieval
- Day 5: Integrate coefficient tracking into trainer

### Week 2: Pattern Matching
- Day 1-2: Implement similarity metrics
- Day 3-4: Build `PatternIndex` with KNN search
- Day 5: Test on small dataset (100 sequences)

### Week 3: Validation
- Day 1-3: Run TinyStories proof-of-concept experiment
- Day 4-5: Build initial visualizations and notebook

### Week 4: Analysis & Comparison
- Day 1-2: Implement transformer baseline for comparison
- Day 3-4: Run ablation studies
- Day 5: Draft technical report outline

## Success Metrics

**Quantitative:**
- Retrieval precision@5 > 60% (semantically relevant matches)
- Coefficient similarity correlates with prediction confidence (r > 0.5)
- Traceability overhead < 20% memory, < 10% compute

**Qualitative:**
- Can explain predictions in interpretable terms (polynomial/frequency/wavelet)
- Human evaluators prefer WaveStack explanations over attention heatmaps
- Identify concrete use cases (debugging, copyright, auditing)

## Dependencies

**Code:**
- Existing decomposition modules (chebyshev.py, fourier.py, wavelet.py)
- Existing interpretability helpers (interpretability.py, visualization.py)
- Trainer infrastructure (trainer.py)

**External:**
- FAISS or similar for efficient KNN search (optional, can start with numpy)
- Matplotlib/Plotly for visualizations
- Jupyter for interactive exploration

**Data:**
- Pretrained model checkpoints (hybrid_12m from TinyStories experiments)
- Indexed training data (can start with 10k sequences)

## Open Questions

1. **Granularity:** Track coefficients per-token or per-sequence?
2. **Compression:** How to index millions of training sequences efficiently?
3. **Privacy:** Can traceability work without exposing raw training data?
4. **Lane weighting:** Should polynomial patterns count more/less than wavelets?
5. **Temporal evolution:** How do coefficient patterns change during training?

## Notes

- Traceability is a **differentiator** for WaveStack vs attention-based models
- Focus on interpretability, not just retrieval accuracy
- Start simple (TinyStories, 10k sequences) before scaling
- Document failure cases: when does traceability break down?
- Consider commercial applications: model auditing, copyright compliance, debugging tools
