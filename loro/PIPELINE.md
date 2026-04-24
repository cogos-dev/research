# Training Data Pipeline: End-to-End Walkthrough

How to train a personalized document retrieval model from your own Claude Code session traces.

## Overview

The pipeline has five stages. Each stage takes the output of the previous one. You can run them individually or all at once via the nightly consolidation.

```
Stage 1: Signal Extraction
    Claude Code session transcripts (.jsonl)
      → mine_sessions.py
        → training-signals/signals/*.json (2,298 signals)

Stage 2: Data Preparation
    Signals + workspace CogDocs
      → prepare.py
        → data.pt (training pairs with embeddings and labels)

Stage 3: Training
    data.pt
      → train_mamba.py
        → best_mamba_model.pt (trained Mamba SSM weights)

Stage 4: Export
    best_mamba_model.pt
      → trm_export.py
        → trm_weights.bin + trm_chunks.json (Go-compatible format)

Stage 5: Deployment
    trm_weights.bin → CogOS kernel restart → live inference
```

## Prerequisites

```sh
cd apps/cogos-v3/autoresearch
python3 -m venv .venv
source .venv/bin/activate
pip install torch sentence-transformers pyyaml numpy
```

You also need:
- Ollama running locally with `nomic-embed-text` model (for the kernel's embedding path)
- Claude Code session transcripts at `~/.claude/projects/*/` (generated automatically by Claude Code)
- A workspace with CogDocs (`.cog/mem/**/*.md` files)

## Stage 1: Signal Extraction

**What it does:** Reads Claude Code session transcripts and extracts behavioral signals -- patterns that indicate whether a retrieval was useful.

**Script:** `mine_sessions.py`

**Input:** Claude Code session JSON lines at `~/.claude/projects/-Users-slowbro-workspaces-cog/*.jsonl`

**Output:** Individual JSON files at `training-signals/signals/*.json`

### Signal Schema

Each signal is a JSON file with this structure:

```json
{
  "id": "0006dbd10af8",
  "session": "70b70766-8dc6-4e81-96cb-50ce9c54fa4c",
  "type": "cascade",
  "query": "the user's message text...",
  "positives": ["path/to/doc/that/was/read.md"],
  "negatives": [],
  "edits": ["path/to/doc/that/was/written.md"],
  "outcome": "cascade",
  "density": 1.0,
  "n_turns": 3,
  "timestamp": "2026-04-08T03:19:21.045Z"
}
```

### Signal Types

Signals are classified by the behavioral pattern that produced them:

| Type | What happened | How detected |
|------|--------------|-------------|
| **crystallization** | Read documents, then wrote a new CogDoc | Read tool calls followed by Write to `.cog/mem/` |
| **cascade** | Flow-state session with high tool density | R/W ratio > threshold, high tools-per-turn |
| **provenance** | Session authored a CogDoc | Reverse-map: find which session created each CogDoc via Write tool calls |
| **correct** | Agent self-corrected after retrieval | Error → retry → success pattern in tool calls |
| **accept** | User accepted agent output | Positive user response after tool use |
| **continue** | Conversation continued normally | Default: message exchange without special pattern |
| **last** | Final exchange in session | Last message pair (dropped from training: weight=0.0) |

**Crystallization is the gold signal.** When someone reads documents and then synthesizes a new document, that's direct evidence the read documents were useful for the task. This is the closest thing to a ground-truth retrieval label you can get from natural interaction traces.

### Running Signal Extraction

```sh
python3 mine_sessions.py
```

This is idempotent -- it skips sessions already extracted. New sessions are appended to `training-signals/signals/`.

### Signal Statistics (as of 2026-04-10)

| Type | Count | Weight | % of Total |
|------|-------|--------|------------|
| continue | 1,333 | 0.5 | 58.0% |
| provenance | 317 | 2.0 | 13.8% |
| crystallization | 237 | 3.0 | 10.3% |
| last | 230 | 0.0 | 10.0% (dropped) |
| accept | 100 | 1.0 | 4.4% |
| correct | 41 | 1.5 | 1.8% |
| cascade | 40 | 2.5 | 1.7% |

**Total: 2,298 signals from 805 sessions.**

## Stage 2: Data Preparation

**What it does:** Converts signals + workspace documents into training pairs (query embedding, candidate embeddings, relevance labels, sample weights).

**Script:** `prepare.py`

**Input:**
1. Training signals from `training-signals/signals/*.json`
2. All CogDocs in the workspace (`.cog/mem/**/*.md`, research docs, etc.)

**Output:** `~/.cache/cogos-autoresearch/data.pt` (PyTorch tensor dict)

### How Pairs Are Generated

For each signal:

1. **Query embedding:** The signal's query text is embedded using `nomic-embed-text` (768-dim Matryoshka, truncated to 384-dim)

2. **Positive candidates:** The signal's `positives` list (file paths) is resolved to chunk indices in the embedding index. This is where most signals get dropped -- the file path must match a chunk in the index.

3. **Negative candidates:** Three types, mixed:
   - **Explicit negatives** from the signal (files retrieved but not useful)
   - **Hard negatives** by cosine similarity (similar but wrong)
   - **Easy negatives** by random sampling

4. **Candidate pool:** 64 candidates per query (mix of positives + all negative types)

5. **Labels:** 1.0 for positive chunks, 0.0 for everything else

6. **Weights:** LRAT-inspired intensity weighting per signal type (see table above), with density and turn-count modifiers

### Session-Level Split

The train/val split is done at the **session level**, not the pair level. All pairs from a given Claude Code session go into either train OR validation, never both. This prevents the model from memorizing session-specific patterns and inflating validation scores.

- 80% of unique sessions → training set
- 20% of unique sessions → validation set

### Running Data Preparation

```sh
python3 prepare.py
# or with custom workspace:
python3 prepare.py --workspace ~/my-project
```

**Output includes:**
- Number of signal-based vs synthetic pairs
- Number of signals skipped (path resolution failures)
- Weight distribution statistics
- Cosine baseline NDCG@10 (the target to beat)
- SHA-256 hash of data.pt (for reproducibility tracking)

### Known Limitation: 59% Signal Skip Rate

Most skipped signals reference non-markdown files (`.go`, `.py`, `.ts`, `.json`) that aren't in the embedding index. The index currently only covers `.md` files. Expanding to source code would require changes to `embed_index.py` and the chunking strategy.

## Stage 3: Training

**What it does:** Trains a Mamba SSM (2.28M parameters) to predict which documents will be useful given a query and the user's historical access patterns.

**Script:** `train_mamba.py`

**Input:** `~/.cache/cogos-autoresearch/data.pt`

**Output:** `best_mamba_model.pt` (PyTorch checkpoint)

### Architecture

```
MambaTRM (2.28M params)
├── Input projection: Linear(384 → 384)
├── Event type embedding: Embedding(4, 384)  [query, retrieval, search, edit]
├── Mamba Block 1: SSM(d_model=384, d_state=4, d_conv=2, expand=1)
├── Mamba Block 2: SSM(d_model=384, d_state=4, d_conv=2, expand=1)
├── LayerNorm(384)
└── Score head: Linear(384 → 1)
```

**Key hyperparameters:**
- D_MODEL=384 (matches embedding dimension)
- D_STATE=4 (SSM state dimension)
- D_CONV=2 (local convolution width)
- N_LAYERS=2 (Mamba blocks)
- Learning rate: 0.0012 (AdamW)

### Reproducibility

All RNGs are seeded (torch, MPS, numpy, random) to seed=42. Use `--max-steps N` for deterministic step counts instead of the default 120-second time budget.

```sh
# Default: time-budget training (120 seconds)
python3 train_mamba.py

# Reproducible: fixed step count
python3 train_mamba.py --max-steps 5000
```

### What to Expect

- Cosine baseline: ~0.77 NDCG@10
- Trained model: ~0.88 NDCG@10 (with session-level split)
- Training time: ~2 minutes on Apple Silicon (MPS)
- The model should beat cosine baseline. If it doesn't, something is wrong with the data.

## Stage 4: Export

**What it does:** Converts PyTorch weights to a format the Go kernel can load for microsecond inference.

**Script:** `trm_export.py`

**Input:** `best_mamba_model.pt`

**Output:**
- `trm_weights.bin` (binary weight tensor)
- `trm_chunks.json` (chunk metadata for the embedding index)

```sh
python3 trm_export.py
```

The export writes to the kernel's expected paths (configured in the CogOS workspace).

## Stage 5: Deployment

The CogOS kernel loads TRM weights at startup. After exporting new weights:

```sh
# Restart the kernel to pick up new weights
launchctl kickstart -k gui/$(id -u)/com.cogos.kernel
```

The kernel logs TRM loading status at startup:
```
trm: loaded weights (46165 chunks, 384 dim, 4 state)
```

## Running the Full Pipeline

All stages in sequence:

```sh
cd apps/cogos-v3/autoresearch
source .venv/bin/activate

# Stage 1: Extract signals (idempotent)
python3 mine_sessions.py

# Stage 2: Prepare training data
python3 prepare.py

# Stage 3: Train
python3 train_mamba.py --max-steps 5000

# Stage 4: Export
python3 trm_export.py

# Stage 5: Restart kernel
launchctl kickstart -k gui/$(id -u)/com.cogos.kernel
```

## Evaluation

### NDCG@10

The primary metric. Measures ranking quality: does the model put the right documents at the top?

- 1.0 = perfect ranking
- Cosine baseline = ~0.77 (what you get without the TRM)
- Current model = ~0.88 mean (across variance reruns)

### Honest Numbers

The peak NDCG of 0.900 is a +4.4σ outlier. The mean across 183 variance reruns is 0.878±0.005. We report the mean, not the peak. See `docs/EVALUATION.md` in the cogos repo for full methodology.

### Verifying the Deployed Model

```sh
# Test the foveated endpoint
curl -s -X POST http://localhost:5100/v1/context/foveated \
  -H "Content-Type: application/json" \
  -d '{"prompt": "your test query here", "budget": 12000}' | jq '.meta'
```

The response includes `trm_scored` count -- should be > 0 if the TRM is loaded and working.

## Adapting This for Your Own Workspace

The pipeline is designed around Claude Code session traces, but the core pattern works with any agent trace format:

1. **Define your signals:** What behavioral patterns indicate successful retrieval in your tool? For Claude Code, crystallization (read → write) is gold. For your tool, it might be different.

2. **Write a signal extractor:** Mine your session logs for those patterns. Output the same JSON schema (query, positives, negatives, type, session ID).

3. **Run prepare.py:** It handles embedding, negative mining, and pair generation. Point it at your workspace with `--workspace`.

4. **Train:** The Mamba architecture and hyperparameters were validated across 639 experiments. They should generalize to similar workspace sizes (10K-100K chunks).

5. **Deploy:** Export weights and plug into your context assembly pipeline.

The model learns *your* retrieval patterns from *your* traces. Different people working on different projects will produce different models. That's the point.

## Why CogOS Is the Coordination Layer

The pipeline isn't a standalone ML project bolted onto a workspace. CogOS is simultaneously:

1. **The trace source.** It integrates with Claude Code (via UserPromptSubmit hooks), OpenClaw (via plugin), Cursor (via session import), and Codex (via adapter). Every agent interaction generates traces that feed the pipeline.

2. **The document corpus.** The CogDocs that the model learns to score are the same CogDocs it will rank at inference time. The embedding index covers the live workspace, not a frozen snapshot.

3. **The deployment target.** The trained model runs inside the same kernel that generated the training data. Weights export directly to the Go binary that serves the foveated endpoint.

4. **The feedback loop.** When the model helps you find the right document, and you use it to write a new insight, that crystallization event becomes a gold training label for the next training run. The system literally improves from being used.

This means the training corpus and the retrieval corpus are always in sync. You don't train on one dataset and deploy against another. When a new CogDoc is created, it enters the retrieval corpus immediately and the training data on the next consolidation cycle.

It also means the model learns from your full cross-tool behavior. Use Claude Code in the morning, Cursor in the afternoon, Codex for overnight runs -- the traces all flow into the same signal store, and the model learns from the aggregate pattern.
