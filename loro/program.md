# Cognitive Context Engine — Autoresearch

You are optimizing a **Tiny Recursive Model (TRM)** for cognitive context assembly. This is not a language model — it's an eigenform generator. The TRM takes document embeddings and a query, iteratively refines a latent state, and outputs salience scores that determine which context items are load-bearing.

## What This Is

This TRM is the core component of a cognitive context engine for CogOS v3 — a system where a tiny model (target: <10M params) curates what a larger inference model sees. The TRM runs at inference time on Apple Silicon MPS in microseconds. Every parameter counts. Every wasted parameter is wasted energy (ln(2) per distinction).

The architecture was derived from the convergence of three independent lines:
1. **Von Foerster's eigenforms** (1970s): "tokens for eigenbehavior" — stable objects arising from recursive processes. The TRM's iterative loop IS an eigenbehavior; what it converges on IS the eigenform.
2. **MoTok** (Zhang et al., 2026): motion generation via diffusion-based discrete tokenizer. 1/6th the tokens, better results. Proved that decoupling semantic planning from fine-grained control yields efficiency AND quality.
3. **Samsung TRM** (2025): 7M param recursive model beating DeepSeek-R1 on structured reasoning. Proved iteration beats scale.

**The core thesis: efficiency IS quality.** The eigenform is the minimum-distinction self-consistent pattern. Every distinction above the minimum is noise. Removing noise reveals signal. A model that converges faster with fewer parameters isn't worse — it's closer to the eigenform.

## Training Data (IMPORTANT)

The TRM now trains on TWO data sources interleaved 50/50:

1. **Synthetic data** (`data.pt`): ~1,867 queries with same-section positive labels and hard negatives
2. **Judge data** (`judge_data.pt`): **1,162 labels** from real cognitive trajectories:
   - 889 session-mined (actual Read tool calls from 6,119 sessions including subagents — what files were ACTUALLY needed)
   - 210 cosine wins (A/B judge — where TRM loses, learn from it)
   - 50 TRM wins (A/B judge — reinforce what works)
   - 13 retrospective (hand-curated ideal context per conversation turn)

Judge batches get 2x loss weight because they're real signal, not proxy signal. The session-mined data captures the empirical mass spectrum:
   - `crystal.cog.md` — 182 reads (heaviest node in the workspace)
   - `framework-status.md` — 118 reads
   - `claude-eigenform-continuity.cog.md` — 53 reads
   - `cognitive-field-protocol.cog.md` — 42 reads
   - `CLAUDE.md` — 57 reads

**Do NOT remove the judge data integration from train.py.** It's the training signal that matters most.

## Embedding Index (NEW)

The embedding pipeline uses an **incremental, streaming, memory-safe index** (`embed_index.py`):
- Stores per-document embeddings in `~/.cache/cogos-autoresearch/embed_index/docs/`
- Only re-embeds files that changed (content hash comparison)
- Consolidates into `~/.cache/cogos-autoresearch/embeddings.pt` + `chunks.json`
- Covers ALL of `.cog/` (3,500+ docs), workspace root, research/, projects/, skills/
- To update: `uv run embed_index.py` (incremental, fast for small changes)
- To rebuild: `uv run embed_index.py --rebuild` (full re-embed)
- To check: `uv run embed_index.py --stats`

`prepare.py` now uses the index by default. The old monolithic path is available with `--legacy`.

## Setup

1. **Read the in-scope files**:
   - `prepare.py` — FIXED. Data pipeline, evaluation metric (NDCG@10). Do not modify.
   - `train.py` — the file you modify. TRM architecture, optimizer, hyperparameters, training loop.
   - `embed_index.py` — FIXED. Incremental embedding index. Do not modify.
2. **Verify data exists**: Check that `~/.cache/cogos-autoresearch/` contains `data.pt` and `embeddings.pt`. If not, tell the human to run `uv run prepare.py`.
3. **Read `results.tsv`** to understand what's been tried and what worked.
4. **Go.**

## What You CAN Modify

`train.py` is the ONLY file you edit. Everything is fair game:

### Architecture
- Latent dimension and structure
- Number of iterations K (the recursion depth)
- Attention mechanism (cross-attention, self-attention, linear attention, no attention)
- Scoring head design
- Conditioning integration (how query combines with latent — additive, multiplicative, cross-attention, FiLM)
- Normalization strategy (LayerNorm, RMSNorm, none)
- Activation functions
- Residual connections and skip patterns
- Shared vs. unshared weights across iterations

### Training
- Optimizer choice and hyperparameters
- Learning rate schedule
- Loss function (BCE, ranking losses like ListMLE, contrastive losses)
- Batch size
- Gradient clipping strategy
- Regularization (dropout, weight decay)

### Novel Ideas to Try
- **Convergence-aware iteration**: stop iterating when scores stabilize (early exit)
- **Progressive refinement**: coarse scoring first, fine-grained later
- **Multi-scale scoring**: score at multiple iterations and combine
- **Noise injection**: add noise to z at early iterations (actual diffusion!)
- **Iterative pruning**: reduce candidate pool at each iteration (foveation)
- **Latent momentum**: carry z information across training batches
- **Alternative initialization**: how z₀ is constructed matters (current: elementwise product of query and candidates)
- **Separate per-iteration norms**: vs fully shared normalization
- **Gated residuals**: learnable skip connection weights per component
- **Candidate-candidate interaction before scoring**: let candidates "see" each other more

## What You CANNOT Modify

- `prepare.py` — read-only. Contains the fixed evaluation, data loading, and constants.
- Package dependencies — only use what's in `pyproject.toml`.
- The evaluation metric — NDCG@10 from `evaluate_ndcg` is ground truth.

## The Goal

**Get the highest NDCG@10.** The time budget is fixed (120 seconds training — increased for the 18x larger dataset). The TRM must stay under ~50M parameters — this is meant to run on Apple Silicon MPS at inference time.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating: weigh complexity cost against improvement magnitude.

**Parameter efficiency matters**: Lower parameter count at same NDCG is a win. Track parameter counts in results.

## What We've Already Learned (12 experiments so far)

### The Leaderboard (Phase 4 — 30k chunks, full .cog/ coverage)

| NDCG@10 | Params | Description | Status |
|---------|--------|-------------|--------|
| 0.7365 | 1.51M | **LR=3e-3 — faster learning for harder 30k problem** | **CURRENT BEST** |
| 0.7274 | 1.51M | K=3, latent=256, 16 heads, shared weights (baseline on 30k data) | baseline |
| 0.7689 | 0 | Cosine similarity baseline (30k chunks) | **TARGET TO BEAT** |

**Current search mode: PLATEAU** — 6 consecutive discards after the LR=3e-3 keep. Hyperparameter tweaks (grad_clip, mixup, warmup, dropout, weight_decay, no-mixup) all failed. The 33-point gap to cosine needs structural change, not tuning.

**Phase 3 results (old index, ~1.5k chunks):**
| 0.7946 | 1.25M | 32 heads | historical |
| 0.7738 | 1.25M | K=3, latent=256, self-attn + cross-attn, query gate, shared weights | historical |
| 0.6541 | 0 | Cosine similarity baseline (old) | historical |

### Key Findings — What Worked

1. **Fewer iterations is better (K=3 > K=8 > K=12 > K=5).** This was surprising. K=3 beats K=8 by ~3 points. The model converges fast — more iterations cause overshoot or interference. This suggests the eigenbehavior reaches its fixed point quickly for this data size. Try K=2 and K=1.

2. **Self-attention is load-bearing.** Removing self-attention dropped NDCG from 0.774 to 0.711. Candidates need to see each other — compositional coherence matters. The context window quality depends on items working together, not just individual relevance.

3. **Query re-injection gate is load-bearing.** Removing it dropped to 0.758. The query signal drifts during iteration; the gate prevents this. This is the "conditioning" in the diffusion analogy — without it, the process forgets what it's denoising toward.

4. **Shared weights across iterations is correct.** This is true eigenbehavior — same function applied K times. It also saves parameters.

5. **BCE with label smoothing is hard to beat.** ListNet ranking loss scored lower. The simple loss works.

6. **Batch size 32 is right for 90s.** Batch 64 got fewer steps, which hurt more than stable gradients helped.

### Key Findings — What Failed

1. **Noise injection at early iterations: 0.714.** Significant regression. The initialization (elementwise product) already provides structured starting state — adding noise destroys useful signal.

2. **RMSNorm instead of LayerNorm: 0.765.** Slightly worse. Not catastrophic but not helpful.

3. **Multi-scale scoring (averaging scores across all K iterations): 0.747.** Worse. Early iterations produce bad scores that contaminate the final signal.

4. **Smaller latent (128 vs 256): 0.764.** Worse despite 3.4x fewer params. The latent needs capacity to represent candidate relationships.

### Unexplored Directions (Highest Priority)

These haven't been tried yet and have the strongest theoretical basis:

1. **K=2 or K=1.** The K=3 > K=8 trend is clear. Where does it bottom out? K=1 would mean single-pass — if it beats cosine, the iteration isn't needed. If K=2 beats K=3, we've found the minimum-distinction eigenform.

2. **Initialization strategy.** Currently `z = query * candidates` (elementwise). This is already rich — maybe the initialization does most of the work and iterations are just refinement. Try: additive (`z = query + candidates`), concatenation-projected, or learned initialization.

3. **Attention head count.** Currently 8 heads. Try 4, 2, 1. The model might be spending capacity on attention head diversity it doesn't need.

4. **Scoring head simplification.** Currently 2-layer MLP. Try single linear layer. If the latent is well-structured after iteration, a simple projection might suffice.

5. **Dropout tuning.** Currently 0.05. Try 0.0, 0.1, 0.2. With only 90s of training, regularization dynamics are different.

6. **Learning rate.** Currently 1e-3. Try 3e-3, 5e-4. With 90s budget, faster learning might help.

7. **Warmup steps.** Currently 200. At 90s that's a large fraction of total steps. Try 50 or 100.

8. **Alternative attention**: linear attention, or replacing multi-head attention with simpler bilinear interaction.

9. **Convergence-aware early exit**: compute score delta between iterations, stop when it falls below threshold. Could allow setting K=10 but exiting at K=2-3 on average, getting the best of both worlds.

## Data Details

- **~30,000 chunk embeddings** from 3,500+ workspace documents (full .cog/ coverage)
- **~2,000 queries** generated from chunks (pool_size=64)
- **Embed dim: 384** (nomic-embed-text v1.5, Matryoshka truncated)
- **Positives**: same-document chunks ONLY (structural relevance, not surface similarity)
- **Hard negatives**: cosine-similar chunks from OTHER documents (designed to deceive cosine similarity)
- **Easy negatives**: random chunks
- **The task**: learn document-level coherence beyond surface similarity. Cosine gets fooled by hard negatives. The TRM should not.
- **Coverage expanded**: index now includes .cog/conf, .cog/hooks, .cog/patterns, .cog/specs, .cog/schemas, research/, projects/, skills/ — 18x more documents than original

## Output Format

The script prints:
```
---
val_ndcg:         0.XXXXXX
training_seconds: 90.0
total_seconds:    XXX.X
num_steps:        XXXX
num_params:       X,XXX,XXX
latent_dim:       XXX
n_iterations:     X
n_heads:          X
```

Extract the key metric: `grep "^val_ndcg:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	val_ndcg	num_params	status	description
```

1. git commit hash (short, 7 chars)
2. val_ndcg achieved (e.g. 0.812345) — use 0.000000 for crashes
3. parameter count (e.g. 1234567) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

## Adaptive Search Strategy

Before picking your next experiment, **determine your current search mode** from the last 10 results in `results.tsv`:

### Mode Detection

Count the last 10 experiments (Phase 4 only):
- **EXPLORE** (default, or <5 experiments): Try diverse, divergent ideas. Cast a wide net. Test fundamentally different approaches — alternative loss functions, radically different architectures, new initialization strategies.
- **EXPLOIT** (2+ keeps in last 10): Something is working. Make small variations around the current best — tune hyperparameters, try nearby values, combine recent wins.
- **PLATEAU** (0 keeps in last 8+): Nothing is improving. You MUST break out:
  - Try **radical architectural changes** (different attention mechanisms, remove entire components, double the model size, halve it)
  - Try **combinatorial experiments** (combine 2-3 near-miss ideas that each individually failed)
  - Try **different loss functions** (ListMLE, InfoNCE, contrastive, triplet)
  - Try **different optimizers** (SGD+momentum, LAMB, Shampoo)
  - Increase training time budget temporarily to see if it's a convergence issue
  - **Do NOT keep tweaking the same hyperparameters** — that's how you stay on the plateau
- **RECOVER** (3+ crashes in last 10): Stabilize first. Revert to known-good `train.py`, make conservative changes only.

### How to Apply

1. At the START of each experiment cycle, count keeps/discards/crashes in the last 10 results
2. State your current mode in the commit message: `exp [EXPLORE]: ...` or `exp [PLATEAU]: ...`
3. Let the mode constrain your choices — if you're in PLATEAU, do NOT try another LR tweak
4. After a mode transition (e.g., plateau → exploit after a breakthrough), note it in results.tsv

### Anti-Patterns to Avoid

- **Plateau grinding**: Tweaking LR from 2e-3 to 2.5e-3 to 1.8e-3 when nothing has improved in 8 runs. STOP. Try something structurally different.
- **Ignoring the gap**: The cosine baseline is 0.769. If you're at 0.736, small hyperparameter tweaks won't close a 33-point gap. You need architectural insight.
- **Recency bias**: Don't just optimize around the last experiment. Look at the FULL history — what structural changes gave the biggest jumps?

## The Experiment Loop

LOOP FOREVER:

1. Read `results.tsv` — what's been tried, what's the current best?
2. **Determine search mode** (see Adaptive Search Strategy above)
3. Read current `train.py` — what does the architecture look like now?
4. Pick an experiment **constrained by your search mode**
5. Modify `train.py` with the experimental idea
6. `git add train.py && git commit -m "exp [MODE]: description of change"`
7. Run: `uv run train.py 2>&1 | tee run.log`
8. Read results: `grep "^val_ndcg:\|^num_params:" run.log`
9. If grep is empty → crash. Run `tail -n 50 run.log` for the traceback.
10. Record in `results.tsv`
11. If val_ndcg improved → keep (move forward)
12. If val_ndcg is equal or worse → discard (`git checkout train.py` to restore best version)

**NEVER STOP.** The human may be asleep. You are autonomous. If you run out of ideas, re-read this document, try combining previous near-misses, try radical architectural changes. The loop runs until manually interrupted.

## Principles

- **One change per experiment.** Don't change K AND latent_dim AND loss function at once. Isolate variables. EXCEPTION: in PLATEAU mode, combinatorial experiments are allowed.
- **Trust the metric.** NDCG@10 is ground truth. Your intuition about what "should" work doesn't matter — only the number matters.
- **Simplification is valid.** Removing a component and getting equal NDCG is a win. The eigenform principle: minimum distinctions.
- **Track your reasoning.** The commit message and results.tsv description should explain WHY you tried this, not just WHAT you changed.
- **Don't repeat failed experiments.** Check results.tsv before trying something.
- **Read the gap.** If you're 30+ points below cosine baseline, hyperparameter tuning won't save you. Think structurally.
