# Variance Test: InfoNCE vs Cross-Entropy Loss

**Date:** 2026-04-11  
**Training config:** `--max-steps 3200 --loss infonce`, device: MPS  
**Model:** LoRO Mamba TRM, 2,282,113 params

---

## Per-Seed Results

| Seed | CE NDCG@10 | InfoNCE NDCG@10 | Delta |
|------|------------|-----------------|-------|
| 42   | 0.882645   | 0.686820        | -0.195825 |
| 123  | 0.817920   | 0.609237        | -0.208683 |
| 456  | 0.777291   | 0.657541        | -0.119750 |
| 789  | 0.783512   | 0.619665        | -0.163847 |
| 1024 | 0.785123   | 0.691514        | -0.093609 |

---

## Summary Statistics

| Metric | Cross-Entropy | InfoNCE | Change |
|--------|--------------|---------|--------|
| Mean   | 0.8093       | 0.6530  | -0.1563 |
| σ (std) | 0.0440      | **0.0377** | **-0.0063** |
| Min    | 0.7773       | 0.6092  | -0.1681 |
| Max    | 0.8826       | 0.6915  | -0.1911 |

---

## Variance Hypothesis

**Hypothesis:** InfoNCE should reduce variance (σ) compared to cross-entropy.

**Result: PARTIALLY CONFIRMED — with a significant caveat.**

- σ did decrease: 0.0440 → 0.0377, a **14.3% reduction in standard deviation**
- However, **mean NDCG dropped sharply**: 0.8093 → 0.6530 (-0.156, or -19.3%)
- The reduced variance is likely a side effect of the lower absolute performance ceiling, not improved training stability per se

The seed-42 outlier pattern from CE (0.883 vs others ~0.78) is attenuated under InfoNCE (0.687 vs others 0.61–0.69), which drives the lower σ. But all seeds score worse under InfoNCE.

---

## Recommendation

**Do not adopt InfoNCE as the default loss at the current tau=0.02 configuration.**

InfoNCE reduces variance modestly (-14%) but at the cost of a large mean NDCG regression (-19%). The contrastive objective appears too aggressive for this small candidate pool (64) and low-data regime.

Suggested next steps:
1. **Tune tau**: Try tau=0.05, 0.1, 0.2 — the current tau=0.02 may be too sharp, collapsing the score distribution
2. **Hybrid loss**: Weight InfoNCE + CE (e.g., 0.5 each) to benefit from contrastive structure without sacrificing ranking signal
3. **Larger candidate pool**: InfoNCE with in-batch negatives benefits from larger N; current CANDIDATE_POOL=64 may be too small
4. **Revert to CE for now**: CE (mean=0.8093, σ=0.0440) remains the better performing configuration until InfoNCE hyperparameters are tuned
