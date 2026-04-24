# TRM 5-Seed Variance Test

**Date:** 2026-04-11
**Config:** `--max-steps 3200`, MPS (Apple Silicon), Mamba TRM 2.28M params
**Purpose:** Validate training reproducibility after RNG seeding + step-budget fixes

## Results

| Seed | NDCG@10  | Cosine Baseline | Delta (pts) |
|------|----------|-----------------|-------------|
| 42   | 0.882645 | 0.401927        | +480.7      |
| 123  | 0.817920 | 0.397368        | +420.6      |
| 456  | 0.777291 | 0.350319        | +427.0      |
| 789  | 0.783512 | 0.416976        | +366.5      |
| 1024 | 0.785123 | 0.376064        | +409.1      |

## Statistics

| Metric          | Value    |
|-----------------|----------|
| Mean            | 0.8093   |
| Std (σ, pop)    | 0.0393   |
| Std (σ, sample) | 0.0440   |
| Min             | 0.7773   |
| Max             | 0.8826   |
| Range           | 0.1054   |

## Gate Result

**FAIL — σ = 0.039 > threshold 0.020**

The gate criterion (σ < 0.02) is not met. Training is deterministic within a given seed
(step-budget and RNG seeding work correctly — seed 42 no longer produces the original
0.728 result), but performance varies meaningfully across different seeds.

## Analysis

The variance is driven primarily by seed 42 scoring ~10 pts higher than the other four
seeds (0.883 vs 0.777–0.818). The other four seeds cluster tightly (σ ≈ 0.016), which
would pass the gate on their own. Seed 42 appears to initialize into a particularly
favorable basin.

**Key observations:**
- The original reproducibility bug (0.728 vs 0.882 on *same* seed/config) is fixed:
  seed 42 now consistently produces ~0.883.
- Cross-seed variance (~4% σ) reflects genuine sensitivity to weight initialization,
  not a training pipeline bug.
- All 5 runs substantially outperform cosine baseline (+366 to +481 pts), confirming
  the model is learning.

## Recommendation

Consider one of:
1. **Accept the variance** — cross-seed σ of 0.04 is normal for small models on small
   datasets (162 sequences, 32 val). Use the best-checkpoint strategy already in place.
2. **Ensemble** — average predictions from multiple seeds for production inference.
3. **More data** — variance typically shrinks as training set grows.
