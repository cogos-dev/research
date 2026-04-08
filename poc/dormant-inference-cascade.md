# Dormant Inference Cascade — Autonomous Research via Idle GPU Cycles

> Design doc for CogOS kernel capability: run supervised multi-agent inference
> cascades during Dormant state using idle local GPU capacity.

## The Insight

A 26B parameter model cold-starts in 30 seconds on Apple Silicon and generates
at ~100 tok/s. If the kernel loads it hourly during Dormant state, that's 3-5
minutes of high-quality inference per cycle with zero user disruption.

The kernel already has the Dormant state. It already runs memory consolidation
there. This extends Dormant to run **supervised inference cascades** — a
supervisor model designs experiments, spawns parallel agents, collects their
observations, updates hypotheses, and records everything to the ledger.

## Architecture

### The Cascade

```
Dormant state triggers (hourly, or on-idle threshold)
  │
  ├─ 1. LOAD: ollama pull/warm gemma4:26b (~30s cold, instant if warm)
  │
  ├─ 2. SUPERVISOR TURN: single inference call
  │    Input:  previous cycle's hypotheses + chain deltas
  │    Output: experiment spec + N agent prompts
  │    Cost:   ~500 tokens, ~5s
  │
  ├─ 3. CASCADE: supervisor issues blocking tool call → run_cascade(agents)
  │    │
  │    │  Spawns N parallel Ralph agents, each:
  │    │    - Same prompt (or variations)
  │    │    - READ-WIDE: full workspace visibility
  │    │    - WRITE-NARROW: can ONLY append to its chain file
  │    │    - Runs until observation recorded
  │    │
  │    ├── Ralph-1 (gemma4:26b) → chain-1.jsonl
  │    ├── Ralph-2 (gemma4:e4b)  → chain-2.jsonl
  │    ├── Ralph-3 (qwen3.5:9b)  → chain-3.jsonl
  │    └── Ralph-N (custom LoRO)  → chain-N.jsonl
  │    │
  │    │  Each agent appends a delta:
  │    │  {"timestamp", "model", "prompt_hash", "observation", "confidence", "sources"}
  │    │
  │    └── All resolve → tool returns collected deltas
  │
  ├─ 4. ANALYSIS: supervisor inference call
  │    Input:  all deltas from this cycle
  │    Output: pattern analysis + updated hypotheses + reconciliation actions
  │    Looks for:
  │      - Agreement across models (high-confidence signal)
  │      - Disagreement (interesting — investigate next cycle)
  │      - Novel observation from one model only (hallucination or discovery?)
  │      - LoRO model divergence from base (training signal)
  │
  ├─ 5. RECORD: append to ledger
  │    - Cycle metadata (models used, prompts, timing)
  │    - All agent observations
  │    - Supervisor analysis
  │    - Updated hypotheses
  │    - Reconciliation actions (if any)
  │
  ├─ 6. RECONCILE: if novel salient observations found
  │    - Priority: train on novel experiences first
  │    - Propagate changed understanding through workspace
  │    - Update cogdocs that reference changed knowledge
  │    - Resolve when all docs are coherent with new state
  │
  └─ 7. UNLOAD: free VRAM, return to Dormant/Receptive
```

### Cost Per Cycle

| Step | Tokens | Time | GPU |
|------|--------|------|-----|
| Load model | 0 | 30s (cold) / 0s (warm) | VRAM allocation |
| Supervisor turn 1 | ~500 | ~5s | Inference |
| N=3 Ralph agents | ~1500 | ~15s (parallel) | Inference |
| Supervisor turn 2 | ~800 | ~8s | Inference |
| Record + reconcile | 0 | ~2s | CPU only |
| Unload | 0 | instant | VRAM free |
| **Total** | **~2800** | **~60s** | **~30s GPU** |

One cycle costs less than a single user conversation turn. 24 cycles/day = 67K tokens of autonomous research. Free.

## The Chain File Format

Each Ralph appends observations to a JSONL chain file. The chain is append-only
and hash-linked (like the ledger):

```jsonl
{"seq":1,"prior_hash":"","timestamp":"...","model":"gemma4:26b","prompt_hash":"abc123","observation":"The foveated engine's zone ordering appears to over-weight recency vs semantic relevance for long-running sessions","confidence":0.7,"sources":["internal/engine/context_assembly.go:245","internal/engine/trm_context.go:89"],"hash":"def456"}
{"seq":2,"prior_hash":"def456","timestamp":"...","model":"gemma4:26b","prompt_hash":"abc123","observation":"Zone 2 budget grows linearly with conversation length but salience scoring doesn't account for topic drift","confidence":0.8,"sources":["internal/engine/context_assembly.go:180"],"hash":"ghi789"}
```

The supervisor reads ALL chains and treats them as **conflicting commits** —
multiple observers looking at the same workspace from different angles. Where
they agree, that's signal. Where they diverge, that's interesting.

## Read-Wide Write-Narrow Sandbox

The key sandbox primitive for Ralph agents:

```
Permissions:
  READ:  entire .cog/ workspace, all source code, all cogdocs
  WRITE: ONLY append to assigned chain file (one file per agent)
  EXEC:  NONE (no shell, no tools except read + append)
```

This is maximally safe:
- Agents can't modify any workspace files
- Agents can't execute arbitrary commands
- Agents can't communicate with each other (only through chains, read by supervisor)
- All output is append-only and hash-chained (tamper-evident)

The supervisor has broader permissions:
- Read all chains
- Write hypotheses file
- Trigger reconciliation actions (via kernel API)
- Load/unload models (via Ollama API)

## LoRO Model Evaluation

Running custom LoRO-trained models alongside base models in the same cascade
produces **live evaluation data**:

```
Cycle output:
  Ralph-1 (gemma4:26b base): "Zone ordering looks correct for this query"
  Ralph-2 (gemma4:e4b base):  "Zone ordering looks correct for this query"
  Ralph-3 (custom LoRO):      "Zone 2 should prioritize the TRM cogdoc over the config file"

Supervisor inference:
  "Ralph-3 (LoRO) disagrees with base models on zone ordering.
   Ralph-3 may have learned domain-specific relevance from training data.
   Hypothesis: LoRO model has better workspace-specific salience.
   Test: compare NDCG of LoRO vs base on this query class."
```

This creates a **continuous evaluation loop**:
1. LoRO models are trained on workspace data
2. They run in cascades alongside base models
3. Disagreements are recorded and analyzed
4. High-quality disagreements (where LoRO is right and base is wrong) confirm training is working
5. Low-quality disagreements (where LoRO is wrong) identify training data gaps

The training pipeline feeds the evaluation cascade feeds the training pipeline.

## Novel Experience Prioritization

When a cascade discovers something genuinely new (a cogdoc that's wrong, a code
pattern that contradicts documentation, a concept that's missing from the
workspace), the reconciliation step prioritizes it:

```
Priority queue for reconciliation:
  1. NOVEL + HIGH AGREEMENT: Multiple models found it. High confidence.
     → Immediate workspace update
  2. NOVEL + SINGLE MODEL: Only one model found it.
     → Queue for next cycle's focused investigation
  3. KNOWN + CONFIRMED: Models agree with existing docs.
     → No action (workspace is coherent)
  4. KNOWN + CONTRADICTED: Models disagree with existing docs.
     → Flag for human review
```

Reconciliation propagates changes through the workspace until everything
resolves — like a git rebase that touches every file referencing the changed
concept.

## Integration with Kernel Process Loop

```go
// In process.go, during Dormant state:
func (p *Process) maybeDormantCascade() {
    if p.state != StateDormant { return }
    if time.Since(p.lastCascade) < p.cfg.CascadeInterval { return }
    
    // Load model
    p.loadCascadeModel()
    defer p.unloadCascadeModel()
    
    // Run supervisor + agents
    results := p.runInferenceCascade(p.cascadeHypotheses)
    
    // Record to ledger
    p.emitEvent("cascade_complete", results)
    
    // Reconcile if needed
    if results.HasNovelObservations() {
        p.reconcileWorkspace(results.NovelObservations)
    }
    
    p.lastCascade = time.Now()
    p.cascadeHypotheses = results.UpdatedHypotheses
}
```

## Relationship to Existing Architecture

| Component | Role in Cascade |
|-----------|----------------|
| Process state machine | Triggers cascade during Dormant |
| Foveated engine | Agents read workspace through foveated context |
| Tool-call gate | Validates cascade tool calls (run_cascade) |
| Ledger | Records all observations, hash-chained |
| TRM | Benefits from cascade observations as training data |
| Semantic coherence | Cascade IS continuous semantic validation |
| LoRO evaluation | Cascade IS the live model comparison framework |

## Next Steps

1. Implement `run_cascade` as a kernel tool (blocking, spawns N agents)
2. Implement Ralph agent with read-wide write-narrow sandbox
3. Implement chain file format (append-only JSONL with hash linking)
4. Wire into Dormant state with configurable interval
5. Build supervisor prompt template
6. Test with gemma4:26b + gemma4:e4b side by side
7. Add reconciliation mode (propagate novel observations)
