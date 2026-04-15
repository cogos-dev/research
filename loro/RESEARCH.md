# LoRO Research: Learned Document Retrieval for Foveated Context Assembly

## What This Is

A learned retrieval model (the "TRM" -- Tiny Recursive Model) that scores which workspace documents should appear in an AI agent's context window. It runs inside the CogOS kernel as one scorer in a pipeline that also includes git-derived salience and keyword matching.

The system works without the TRM (salience fallback), but the TRM improves ranking quality when it works.

## Data Privacy and What We Release

**Model weights are NOT released.** The TRM is trained on personal Claude Code session traces -- what documents you read, in what order, which ones triggered new writing, how your attention moved through the workspace over months. The weights are a compressed representation of one person's cognitive patterns. Releasing them would be releasing personal behavioral data.

**What we release is the method:**

1. **Signal extraction** -- How to mine Claude Code session transcripts for retrieval supervision signals. Which tool-call patterns indicate successful retrieval (crystallization: read docs → wrote new document). How to detect cognitive phases (probe → intake → cascade → crystallize → consolidate) from behavioral features alone.

2. **Training data preparation** -- How to convert raw signals into weighted training pairs. The intensity weighting scheme (crystallization 3.0, cascade 2.5, provenance 2.0, etc). How to generate synthetic negatives alongside real behavioral signals.

3. **Model architecture** -- Full architecture specification (Mamba SSM, 2.28M params). Why temporal SSM beats spatial transformer for this task (639 experiments proving it). The complete hyperparameter space explored.

4. **Evaluation methodology** -- Four-layer eval (stock LLM, RAG, cosine, foveated). NDCG@10 with honest variance reporting (mean 0.878, not peak 0.900). How to construct evaluation sets from CogDoc ideal sources.

5. **Deployment pipeline** -- How to export weights to Go for microsecond inference. The closed-loop extract → train → deploy nightly consolidation.

Anyone can reproduce this on their own workspace with their own traces. The recipe is the contribution, not the trained model.

## How It Connects to CogOS

```
User types prompt in Claude Code
  → UserPromptSubmit hook fires
    → foveated-context.py calls POST /v1/context/foveated on the Go kernel
      → serve_foveated.go checks: is a TRM loaded?
        → YES: embed query → cosine top-100 → TRM re-rank → sigmoid blend → return docs
        → NO:  keyword + salience scoring (no ML, still works)
      → Response rendered as stability-zone context blocks
        → Injected into Claude's context before it sees the message
```

The TRM is the learned layer in step 4. Everything else (hooks, salience, zones, injection) runs without it.

**Key files in the kernel:**
- `internal/engine/serve_foveated.go` -- HTTP endpoint, orchestrates scoring
- `internal/engine/trm_context.go` -- TRM loading, embedding, cosine pre-filter, score blending
- `internal/engine/trm.go` -- MambaTRM Go inference (ported from Python)
- `internal/engine/context_assembly.go` -- Stability zone assembly, budget eviction

**Key files in this directory:**
- `train_mamba.py` -- Active model: Mamba SSM training (**this is the one that ships**)
- `train.py` -- Superseded: cross-attention transformer (kept as historical record)
- `prepare.py` -- Data pipeline: signal extraction → training pair generation
- `prepare_sequences.py` -- Temporal sequence preparation for Mamba
- `embed_index.py` -- Incremental embedding index (Ollama nomic-embed-text)
- `trm_export.py` -- Export trained weights to Go-compatible format

## Architecture Evolution

### Phase 1-3: Cross-Attention Transformer (train.py)

**Problem framing:** Given a query embedding and N candidate document embeddings, score each candidate's relevance.

**Architecture:** Query → cross-attention with candidates → iterative refinement (K steps) → relevance scores. 3.25M parameters.

**Results:** 196 logged experiments in `results.tsv`.
- Started at 0.604 NDCG@10 (cosine baseline: 0.654)
- Peaked at 0.749 NDCG after extensive hyperparameter search
- Hit a hard plateau: no architecture or training change could break 0.75
- High seed sensitivity: same config varied by 0.022 NDCG across seeds

**Why it failed:** The transformer treats each (query, candidate) pair independently -- a spatial metric. But retrieval patterns are temporal: what you need next depends on what you've been doing, not just what you're asking right now. The model was solving the wrong problem.

**Status:** `train.py` is kept as historical record. `best_model.pt` contains the last checkpoint. Not deployed.

### Phase 4: Mamba SSM Pivot (train_mamba.py)

**Problem reframing:** Given a temporal sequence of workspace events (queries, reads, edits), predict which documents will be needed next.

**Architecture:** Event sequence → type embedding → Mamba SSM blocks → hidden state ("light cone") → score head → candidate scores. 2.28M parameters.

The key insight: the SSM maintains a hidden state that compresses the user's trajectory through the workspace. This state is the "light cone" -- it captures what the user has been doing and naturally decays older context, similar to how SSMs handle long sequences.

**Results:** 443 logged experiments in `results_mamba.tsv`.
- Started at 0.424 NDCG with naive architecture (D_STATE=64, 4 layers)
- Discovery: smaller state spaces work better. D_STATE=4 outperformed D_STATE=64.
- Converged to: D_MODEL=384, D_STATE=4, D_CONV=2, N_LAYERS=2, EXPAND=1
- **Final: 0.878 mean NDCG@10 across 183 variance reruns (peak 0.900, +4.4σ)**
- Cosine baseline on same task: 0.773

**Why it works:** Retrieval is a temporal prediction problem. The SSM's recurrence naturally captures session dynamics -- early queries set context, mid-session queries narrow focus, late queries often revisit fundamentals. A 6KB hidden state encodes this entire trajectory.

### Phase 5: Signal-Weighted Training (current)

**Addition:** Train on behavioral signals extracted from real Claude Code sessions, not just synthetic pairs.

**Signal types and weights:**
| Signal | Count | Weight | Meaning |
|--------|-------|--------|---------|
| crystallization | 237 | 3.0 | Read docs → wrote new CogDoc (gold retrieval label) |
| cascade | 40 | 2.5 | High R/W ratio flow state |
| provenance | 317 | 2.0 | Reverse-mapped CogDoc to authoring session |
| correct | 41 | 1.5 | Agent self-correction after retrieval |
| accept | 100 | 1.0 | User accepted agent output |
| continue | 1,333 | 0.5 | Conversation continued (weak positive) |
| last | 230 | 0.3 | Final exchange (likely noise) |

**Total:** 2,298 signals from 805 sessions. Generates 659 signal-based + 1,818 synthetic = 2,477 weighted training pairs.

**Result:** NDCG improved from 0.882 to 0.891 with weighted signals. Cosine delta improved from +108 to +489 points.

## Current Architecture (Deployed)

```
MambaTRM (2.28M params)
├── Input projection: Linear(384 → 384)
├── Event type embedding: Embedding(4, 384)  [query, retrieval, search, edit]
├── Mamba Block 1: SSM(d_model=384, d_state=4, d_conv=2, expand=1)
├── Mamba Block 2: SSM(d_model=384, d_state=4, d_conv=2, expand=1)
├── LayerNorm(384)
└── Score head: Linear(384 → 1)
```

**Inference path (Go):**
1. Query embedded via Ollama (nomic-embed-text, 768-dim → truncated to 384)
2. Cosine pre-filter: top-100 candidates from 46,165 chunk index
3. TRM step: process query through SSM, update light cone state
4. Score candidates: dot product of context vector with candidate embeddings
5. Sigmoid normalize TRM scores (prevents unbounded negatives from destroying cosine signal)
6. Blend: 0.7 × cosine + 0.3 × sigmoid(TRM)
7. Path-level dedup: keep highest-scoring chunk per unique file
8. Return top-10 with scores

**Latency:** ~6ms total (embedding dominates; TRM inference is microseconds)

## Known Issues and Limitations

### Reproducibility (identified 2026-04-11)
- Training is NOT reproducible across runs. Same config produced 0.882 and 0.728.
- **Root causes:** Incomplete RNG seeding (MPS, numpy, random unseeded), time-budget training (120s wall clock, not fixed steps), embedding index drift between runs.
- **Fixes applied:** Complete seeding (torch/MPS/numpy/random to 42), `--max-steps` flag, data.pt hash logging.
- **Status:** Fixes in code, not yet re-validated with 5-run variance test.

### Signal Waste
- 59% of extracted signals (1,352 of 2,298) are skipped because the embedding index only covers `.md` files, but many signals reference `.go`, `.py`, `.ts`, etc.
- Path resolution was also buggy (fixed 2026-04-11: 1,807x faster, zero false positives).
- To improve: either expand the embedding index to cover source code, or accept that the TRM only learns from documentation retrieval patterns.

### Evaluation Validity
- Train/val split is random at pair level, not session level. Same-session signals can leak between sets, potentially inflating NDCG.
- Recommended: session-held-out evaluation as prerequisite for future experiments.

### Score Blending
- Sigmoid normalization + 70/30 cosine/TRM blend was chosen empirically (2026-04-10 debug session), not derived.
- The TRM's contribution is modest -- it's a tiebreaker and recency boost on top of cosine, not a dominant signal.
- With a small model (2 layers, d_state=4) and limited training data, cosine must dominate to preserve semantic relevance.

### D_STATE=4
- Chosen through empirical search, not theoretical derivation.
- The "3+1 decomposition" interpretation (content/recency/structure + query) was named post-hoc.
- May be a dataset artifact (optimal bias-variance tradeoff for ~130 validation samples).
- Needs: d_state scaling study on larger corpus, state probing analysis.

## Experiment Log Summary

| Phase | Experiments | Architecture | Best NDCG | Cosine Baseline |
|-------|------------|-------------|-----------|-----------------|
| 1-3 | 196 | Cross-attention transformer | 0.749 | 0.654 |
| 4 | 443 | Mamba SSM | 0.878 (mean) | 0.773 |
| 5 | ~10 | Mamba + weighted signals | 0.891 | 0.773 |

Full logs: `results.tsv` (transformer), `results_mamba.tsv` (Mamba)

## Prior Art Positioning

**What's industry standard (can't claim as novel):**
- Content-addressed KV blocks via hash (vLLM, LMCache, Everpure)
- Cosine similarity retrieval for context assembly (RAG)

**What's genuinely novel:**
- Learned temporal retrieval model (SSM) trained on agent behavioral traces
- Crystallization events as gold retrieval labels
- Cognitive phase detection from tool-call patterns
- Closed-loop extract → train → deploy pipeline from co-creation traces

**Closest prior art:** LRAT (arxiv 2604.04949) -- agent trajectory retrieval with similar core insight. TRM extends with crystallization-as-label, cascade detection from behavioral features, and SSM architecture for temporal modeling.

See: `.cog/mem/semantic/projects/portfolio/overnight-runs/run-2026-04-11-prior-art-deep.md` for full survey of 25+ papers.

## Planned Experiments

Five experiments designed (see `run-2026-04-11-training-next-design.md`):
1. Reproducibility baseline (5 seeded runs, ~40 min)
2. Signal quality ablation (no-last, gold-only, no-continue conditions)
3. Session-held-out evaluation (split by session, not pair)
4. D_STATE scaling {2, 4, 8, 16} on current corpus
5. Behavioral hard negative mining

## File Inventory

### Active (ships to production)
| File | Purpose |
|------|---------|
| `train_mamba.py` | Mamba TRM training |
| `prepare.py` | Training data pipeline (signals → pairs) |
| `prepare_sequences.py` | Temporal sequence preparation |
| `embed_index.py` | Incremental embedding index |
| `trm_export.py` | Weight export to Go format |

### Supporting
| File | Purpose |
|------|---------|
| `program_mamba.md` | Ralph agent program (Mamba) |
| `mine_sessions.py` | Session data mining |
| `collect_judge_data.py` | A/B judge data collection |
| `eval_downstream.py` | Downstream evaluation |
| `dashboard.py` | Results visualization |

### Historical (not deployed)
| File | Purpose | Status |
|------|---------|--------|
| `train.py` | Cross-attention transformer | Superseded by Mamba |
| `program.md` | Ralph agent program (transformer) | Historical |
| `finetune_judge.py` | Judge model fine-tuning | Historical |
| `make_judge_labels.py` | Judge label generation | Historical |
| `integrate_retrospective.py` | Retrospective integration | Historical |
| `retrospective_training_data.py` | Retro training data | Historical |
| `mine_attention.py` | Attention pattern mining | Historical |
| `shadow_trm.py` | Shadow scoring comparison | Historical |
| `eval_response.py` | Response quality evaluation | Historical |

### Data Files
| File | Purpose |
|------|---------|
| `results_mamba.tsv` | 443 Mamba experiment results |
| `results.tsv` | 196 transformer experiment results |
| `best_mamba_model.pt` | Best Mamba checkpoint |
| `best_model.pt` | Best transformer checkpoint (historical) |
| `ralph_mamba.log` | Mamba training log |
| `ralph.log` | Transformer training log |
| `training-signals/` | Extracted behavioral signals |
