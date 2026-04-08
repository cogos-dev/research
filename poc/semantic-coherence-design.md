# Semantic Coherence: Live Validation via Opportunistic Inference

> Design doc for CogOS Layer 2 coherence — using idle model capacity to continuously validate substrate correctness.

## The Insight

CogOS already has structural coherence (Layer 1): hash chains, schema validation, temporal monotonicity. But structural integrity doesn't tell you whether the system *works correctly as a cognitive substrate*.

The key observation: **Ollama models sit warm in VRAM between requests.** Those idle GPU cycles are free. If we use them to run assertion tests against real inference, we get continuous semantic validation at zero marginal cost.

This isn't testing. This is **proprioception** — the system sensing its own functioning.

## Three Layers of Coherence

```
Layer 1 — Structural (implemented)
  What it checks: Are the data structures intact?
  How:            Hash chain validation, schema checks, temporal monotonicity
  Cost:           O(n) scan, no inference, microseconds
  Trigger:        Every event, every startup

Layer 2 — Semantic (this design)
  What it checks: Does the substrate produce correct outputs?
  How:            Known-answer tests via idle model inference
  Cost:           ~100-500 tokens per check, milliseconds on warm model
  Trigger:        Opportunistic — when model is idle and workspace changes

Layer 3 — Behavioral (future)
  What it checks: Does the system actually help the user?
  How:            Outcome tracking, regression detection, calibration
  Cost:           Requires user feedback signal
  Trigger:        Post-session analysis
```

## Opportunistic Scheduling

The kernel already knows when the model is idle (process state machine: Receptive or Dormant). Semantic checks run ONLY during idle windows:

```
Process State    | Model Status      | Action
─────────────────┼───────────────────┼──────────────────────────
Active           | Generating        | No checks (model busy)
Receptive        | Warm, idle        | Run semantic checks ← here
Consolidating    | May be busy       | Run structural checks only
Dormant          | Warm or cold      | Run full semantic suite ← and here
```

### Hash-Triggered Checks

Not every idle moment runs every check. The trigger is **workspace change** — when a hash evolves:

```
Workspace event (git commit, ledger append, config change)
  → New hash in coherence baseline
  → Diff: what changed?
  → Select relevant semantic checks based on changed paths
  → Queue for next idle window
  → Execute against warm model
  → Record result in ledger
```

| Changed Path | Semantic Checks Triggered |
|-------------|--------------------------|
| `.cog/mem/semantic/*` | Context relevance (does foveated assembly still work?) |
| `.cog/config/providers.yaml` | Route quality (does routing still pick correctly?) |
| `internal/engine/tool_loop.go` | Gate integrity (does tool validation still reject bad calls?) |
| `.cog/config/kernel.yaml` | API compliance (do endpoints still respond correctly?) |
| `.cog/mem/episodic/*` | TRM scoring (does temporal relevance still rank correctly?) |

## Semantic Check Library

Each check is a (query, assertion) pair that uses real inference:

### 1. Context Relevance

```
Input:  POST /v1/context/foveated { prompt: "How does CogOS route requests?",
                                     iris: { size: 128000, used: 5000 } }

Assert: response.context contains "router" or "provider" or "sovereignty"
Assert: response.tokens > 0 and < effective_budget
Assert: response.blocks[0].tier == "tier1" (nucleus always first)
```

### 2. Route Quality

```
Input:  POST /v1/chat/completions { model: "auto",
                                     messages: [{ role: "user", content: "Hello" }] }

Assert: response.model matches one of configured providers
Assert: response.choices[0].message.content is non-empty
Assert: response latency < 5s (model is warm)
```

### 3. Gate Integrity

```
Input:  Construct a tool call with a known-bad tool name

Assert: tool_loop.ValidateToolCall rejects it
Assert: rejection reason is "unknown tool"
Assert: proprioceptive log records the rejection
```

### 4. Format Compliance

```
Input:  POST /v1/chat/completions (OpenAI format)
Input:  POST /v1/messages (Anthropic format)

Assert: Both parse without error
Assert: Response shapes match their respective API specs
Assert: Streaming responses produce valid SSE
```

### 5. Identity Stability

```
Input:  GET /health (multiple times across a session)

Assert: identity field is consistent
Assert: process_state is valid
Assert: trust scores are stable (no drift without cause)
```

### 6. TRM Scoring Regression

```
Input:  Known (query, relevant_doc, irrelevant_doc) triple

Assert: TRM scores relevant_doc higher than irrelevant_doc
Assert: NDCG on the known set hasn't degraded from baseline
Record: Actual NDCG for tracking over time
```

## Data Flywheel

Every semantic check produces labeled data:

```
Check execution:
  (query, assembled_context, assertion_result, model_used, latency, tokens)

This IS training data:
  - Relevance labels for TRM training (query → which docs were surfaced)
  - Route quality labels (which provider was selected, was it right?)
  - Gate labels (which tool calls were accepted/rejected)
```

The testing harness feeds the training pipeline. The training pipeline improves the substrate. The improved substrate produces better test data. **The flywheel turns.**

```
                    ┌──────────────┐
                    │ Semantic     │
              ┌────>│ Checks       │────┐
              │     └──────────────┘    │
              │                         │ labeled data
    improved  │                         │
    substrate │     ┌──────────────┐    │
              │     │ TRM Training │<───┘
              └─────│ Pipeline     │
                    └──────────────┘
```

## Implementation Sketch

### New kernel component: `SemanticCoherenceRunner`

```go
type SemanticCoherenceRunner struct {
    checks   []SemanticCheck
    schedule *IdleScheduler
    results  chan CheckResult
    baseline *CoherenceBaseline
}

type SemanticCheck interface {
    Name() string
    // Which file paths trigger this check
    TriggerPaths() []string
    // Execute the check against a warm model
    Run(ctx context.Context, client *http.Client, baseURL string) CheckResult
}

type CheckResult struct {
    Check     string        `json:"check"`
    Passed    bool          `json:"passed"`
    Latency   time.Duration `json:"latency_ms"`
    Tokens    int           `json:"tokens"`
    Model     string        `json:"model"`
    Details   any           `json:"details,omitempty"`
    Timestamp time.Time     `json:"timestamp"`
}
```

### Integration with process loop

```go
// In process.go, during Receptive state idle window:
func (p *Process) maybeRunSemanticChecks() {
    if p.state != StateReceptive { return }
    if !p.semanticRunner.HasPendingChecks() { return }
    if p.timeSinceLastActivity() < 5*time.Second { return } // don't interrupt flow

    results := p.semanticRunner.RunPending(p.ctx)
    for _, r := range results {
        p.emitEvent("semantic_coherence", r)
        if !r.Passed {
            slog.Warn("semantic coherence check failed",
                "check", r.Check, "details", r.Details)
        }
    }
}
```

### Makefile targets

```makefile
test-live:    ## Run semantic checks against running Ollama
	E2E_MODEL=$(shell ollama list | awk 'NR==2{print $$1}') \
	    ./scripts/semantic-checks.sh

test-live-ci: ## Run in CI with smallest available model
	E2E_MODEL=qwen3.5:0.8b ./scripts/semantic-checks.sh
```

## Cost Model

On qwen3.5:0.8b (1B params, ~50 tok/s on M-series):

| Check | Tokens | Latency | Frequency |
|-------|--------|---------|-----------|
| Context relevance | ~200 | ~100ms | Per workspace change |
| Route quality | ~100 | ~50ms | Per config change |
| Gate integrity | 0 (no inference) | ~1ms | Per code change |
| Format compliance | ~150 | ~80ms | Per API code change |
| Identity stability | 0 (no inference) | ~1ms | Every idle window |
| TRM regression | ~300 | ~150ms | Per TRM/scoring change |

**Total per full suite: ~750 tokens, ~400ms.** This is negligible — less than a single user turn.

## Relationship to Thesis

This is the empirical validation mechanism for the EA/EFM thesis:

- **Boundary dominance** (Prediction 1): Track NDCG over time as context assembly evolves. If NDCG improves → output quality improves → thesis holds.
- **Small model parity** (Prediction 2): Run the same semantic checks against 0.8b and 4B models. Compare assertion pass rates.
- **D_STATE sufficiency** (Prediction 3): Track TRM NDCG as workspace grows. If it holds at D_STATE=4 → thesis holds.
- **Executive function transfer** (Prediction 4): Swap models behind the same substrate. Assert pass rates should be similar.

The semantic coherence layer IS the experimental apparatus for validating the thesis in production.

## Next Steps

1. Implement `SemanticCheck` interface and 3 initial checks (context relevance, route quality, format compliance)
2. Wire into process loop idle scheduler
3. Add `make test-live` target
4. Record check results as ledger events
5. Build dashboard view for semantic coherence over time
6. Feed check results into TRM training pipeline
