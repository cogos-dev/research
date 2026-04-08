# Externalized Attention and Executive Function Modulation for Intelligent Systems

> This document extends the architectural overview in the [cogos README](https://github.com/cogos-dev/cogos). That document describes *what* CogOS does. This one argues *why it works* and *what follows* from the design.

## The Claim

Quality of a cognitive system's output is a function of its **boundary quality**, not its model size. If you assemble the right context with the right conditioning signals, a 4B model on a phone produces quality comparable to a 100B+ model in a standard agent loop — because the cognitive overhead is externalized into the substrate rather than consumed as tokens.

This is a testable, falsifiable claim. The evidence is both theoretical (information-theoretic arguments about where cognitive work actually happens) and empirical (a TRM — Tiny Recursive Model inspired by [Jolicoeur-Martineau 2024](https://arxiv.org/abs/2510.04871), implemented as a 2.28M-parameter Mamba SSM (Selective State Space Model) — scoring context relevance at D_STATE (the Mamba state dimension — the compressed representation size), set to 4, with preliminary NDCG (Normalized Discounted Cumulative Gain, a standard ranking quality metric) around 0.9 on workspace context selection).

## The Holographic Principle for Cognitive Systems

In physics, the holographic principle describes how the information content of a volume can be encoded on its boundary surface. While the mechanism is different (gravitational entropy vs. information selection), the same *structural pattern* — boundary encodes bulk — holds for cognitive architectures:

```
Bulk:       30,000+ chunks in .cog/ — full ledger, all memory, complete history
            Never seen by the model. Stored in persistent workspace.

Boundary:   ~10 documents, scored by TRM + salience, zone-ordered
            The ONLY information the model receives.
            Assembled per-request in milliseconds.

Generation: Structured output consistent with the boundary.
            Quality = f(boundary quality), not f(model size).
```

The information-theoretic argument is straightforward:

1. The model's output is conditioned entirely on its input (the boundary)
2. No amount of model capacity can recover information absent from the input
3. Therefore, output quality is bounded above by boundary quality
4. The marginal return on model size is limited by the boundary's information content

The standard scaling approach invests in the model (the generator). The substrate approach invests in the boundary (the assembled context). These are complementary but asymmetric — the boundary has the larger effect because it's the constraining variable.

### Empirical signature

If the holographic principle holds, then:
- Improving boundary quality (better TRM, better salience) should improve output more than upgrading the model
- A small model with good boundary should beat a large model with poor boundary
- Context assembly time should dominate generation quality, not model inference time

Preliminary TRM evaluation at D_STATE=4 shows NDCG ~0.9 on workspace context selection (formal benchmarking with cosine similarity and BM25 baselines is in progress). If this holds, it means the entire coupling between an observer and a workspace with thousands of chunks compresses to 4 dimensions. The viable manifold of relevance is low-dimensional.

## Structured Hallucination

"Hallucination" is typically framed as a failure mode. But generation IS hallucination — the model produces structured outputs consistent with statistical regularities in its training data, conditioned on its input.

The question isn't "does the model hallucinate?" (it always does). The question is "does the hallucination align with the task?"

- Well-formed boundary → aligned hallucination → useful output
- Malformed boundary → misaligned hallucination → garbage output
- No boundary (zero-shot) → unconstrained hallucination → model must use its own parameters as the only source of structure

This reframes the entire scaling debate. "Making models less hallucinate-y" (RLHF, constitutional AI, etc.) is model-side optimization. "Making the boundary better so hallucinations are well-aimed" is substrate-side optimization. Both matter, but the substrate side has been systematically underinvested.

The CogOS architecture bets that boundary quality is the higher-leverage variable. The model is a **dependent variable** — its quality follows from what it's given.

## Four Nested Attention Mechanisms

The foveated context engine is commonly misidentified as RAG. It is not retrieval. It is the system-level implementation of attention — the same computational operation that attention heads perform inside transformers, applied at the workspace scale.

| Scale | Mechanism | Input | Bottleneck | Time scale | What it selects |
|-------|-----------|-------|------------|------------|-----------------|
| Token | PLE (Per-Layer Embeddings, Gemma 4) | Token ID + context | 256 dim | Nanoseconds | Per-layer behavior modification |
| Sequence | Transformer attention | Token positions | Head dim 64-128 | Nanoseconds | Which tokens attend to which |
| Workspace | Foveated engine | All available chunks | TRM D_STATE=4 | Milliseconds | Which documents enter the boundary |
| Session | TRM light cone | Interaction events | D_STATE=4 | Seconds-minutes | How relevance evolves over time |

Four mechanisms, each operating through a low-rank bottleneck, each conditioning the layer above it. This is not a coincidence — it reflects the empirical finding that **meaningful modulation of high-dimensional systems lives in a low-dimensional subspace** (see the [LoRO (Low-Rank Observer) framework](../loro/framework.md) for the mathematical treatment).

The zone structure maps onto attention directly:
- **Zone 0 (Nucleus)** = positional anchor — always present, shapes all subsequent processing. Analogous to the `[CLS]` token or system prompt.
- **Zone 1 (Knowledge)** = slow-moving context — high KV cache reuse, shifts across sessions not turns. Analogous to few-shot examples.
- **Zone 2 (History)** = conversation memory — scored by TRM, evictable under pressure. Analogous to the attention window's middle region.
- **Zone 3 (Current)** = immediate input — always present, highest volatility. Analogous to the most recent tokens.

**Iris pressure** is attention budget management. High pressure contracts the fovea (fewer, more relevant documents, tighter zone budgets). Low pressure dilates it (broader context, more exploratory). This mirrors pupil response in biological vision — and the gating mechanism in Mamba's selective state space.

## Why Externalized and Why Not Learned

A natural objection: if the substrate's attention and executive function are so important, why not train the model to do them?

Three reasons:

**1. Thermodynamic cost.** Autoregressive generation costs O(n) per token, where n is the sequence length. The substrate's attention (TRM scoring, zone ordering, iris adjustment) is O(1) per candidate chunk — a fast deterministic scan, not a stochastic generation process. Using the model to manage its own context burns tokens on overhead that could be spent on generation.

**2. The self-reference problem.** A model cannot attend to information it hasn't seen. It cannot decide what context is relevant without first loading all context to evaluate. This is circular — you need the answer to form the question. The substrate breaks this circularity by maintaining state across sessions (the TRM light cone) that the model never sees but that shapes what the model receives.

**3. Combinatorial fragility.** Fine-tuning a model to manage its own attention + tool use + memory + routing creates a coupled system where improving one capability can degrade others. Externalizing these functions into separate, independently testable components (foveated engine, tool-call gate, router, consolidation policy) creates a modular system where each component can be improved without side effects.

The parallel to human cognition is exact: executive function disorders (ADHD, frontal lobe damage) impair the *management* of intelligence, not intelligence itself. The deficit is in attention direction, impulse inhibition, task switching, and working memory maintenance — precisely the functions CogOS externalizes. A brilliant person with executive dysfunction isn't lacking capability. They're lacking the environmental scaffolding that neurotypical brains build automatically. CogOS is that scaffolding, made explicit and available to any observer.

## Substrate Independence

The architecture says "intelligent systems," not "AI," because the substrate doesn't care what the observer is:

- A transformer reading a foveated context package
- A Mamba SSM processing the same zones
- A human reading a dashboard with the same documents
- A future architecture consuming the same MCP endpoint

The foveated engine assembles context for *any observer* based on workspace state and the observer's trajectory (tracked by TRM). The coupling function is the same regardless of what's coupled.

This is why the TRM works at D_STATE=4. It's not modeling intelligence — it's modeling the **coupling** between an observer and an information environment. That coupling is low-dimensional because the relevant relationship between observer and environment is simple, even when both sides are individually complex.

## Comparison with Existing Cognitive Architectures

| Architecture | Approach | Attention mechanism | Executive function | Substrate independence |
|-------------|----------|--------------------|--------------------|----------------------|
| **ACT-R** (Anderson) | Production system with declarative/procedural memory | Activation-based spreading | Built into production matching | Tightly coupled to ACT-R runtime |
| **SOAR** (Laird) | Universal subgoaling with working memory | Recency + preference rules | Impasse-driven subgoaling | SOAR-specific representations |
| **Global Workspace** (Baars) | Broadcast to specialized processors | Competition for global workspace | Implicit in broadcast selection | Theoretical, not implemented |
| **RAG systems** | Retrieve-then-generate | Embedding similarity (static) | None (model does everything) | Model-agnostic but attention is primitive |
| **CogOS** | Externalized EA + EFM | Foveated assembly (TRM + salience + zones) | Process state machine + sovereignty gradient + tool gate | Any observer via standard protocols |

The key difference: ACT-R and SOAR build intelligence into the architecture. CogOS builds the *management infrastructure* and leaves intelligence to the observer. This is why it's substrate-independent — it doesn't implement cognition, it implements the scaffolding that makes cognition effective.

## Falsifiability

The thesis makes specific predictions:

1. **Boundary dominance:** Improving context assembly quality (better TRM, better salience scoring) should improve output quality more than upgrading the model by one size class. Testable by A/B comparison.

2. **Small model parity:** A 4B model with CogOS substrate should match a 70B model without it on workspace-specific tasks where context is the bottleneck. Testable by benchmark comparison.

3. **D_STATE sufficiency:** The TRM's D_STATE=4 should remain sufficient as workspace size grows (up to the point where the observer-environment coupling becomes genuinely high-dimensional). Testable by scaling experiments.

4. **Executive function transfer:** Adding CogOS to a new model should immediately improve its task completion rate without any model fine-tuning. Testable by swapping models behind the same substrate.

5. **Diminishing model returns:** Within a CogOS-managed workspace, doubling model size should produce less improvement than doubling context assembly quality. Testable by cost-benefit analysis.

If boundary quality turns out NOT to be the dominant variable — if model size always wins regardless of context quality — the thesis is wrong.

## Connection to the LoRO Framework

The [LoRO framework](../loro/framework.md) provides the mathematical foundation for why low-rank conditioning works at every scale. PLE (token-level), LoRA (Low-Rank Adaptation, task-level), and TRM (session-level) are all instances of the same pattern: a compact state modulates a much larger system through a bottleneck.

The connection to EA/EFM: the substrate IS a LoRO — a low-rank observer that conditions the generator. The foveated boundary is the bottleneck. The workspace bulk is the high-dimensional system being modulated. The TRM light cone is the compact state.

```
LoRO at substrate scale:
  state       = TRM light cone (D_STATE=4)
  update      = Session events → selective scan
  gate        = Iris pressure + salience scoring
  project     = Foveated assembly (workspace → boundary)
  base        = Model's pre-trained generation
  inject      = Context window (the boundary IS the injection)
```

The substrate principle and LoRO are the same insight at different levels of abstraction.
