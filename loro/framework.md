# LoRO: Low-Rank Observer -- A Unified Abstraction

## Executive Summary

Per-Layer Embeddings (PLE, Gemma 4), LoRA adapters, and the Mamba TRM are structurally convergent mechanisms. All three:

1. Maintain a compact state (PLE: 256-dim, LoRA: rank r, TRM: D_STATE=4)
2. Use it to gate/modulate a larger system's computation through a bottleneck
3. Inject via residual connection
4. Are dramatically smaller than the systems they condition
5. Exploit the low intrinsic dimensionality of their target computation

While individual mechanisms are well-studied (LoRA variants, PLE in Gemma, SSMs for sequence modeling), the structural convergence across all three — and especially the connection to SSM-based context scoring — appears uncharted as of April 2026. We propose a unified abstraction -- **LoRO (Low-Rank Observer)** -- that captures all three as instances of a self-referential loop where a low-rank state shapes the computation that produces the next observation that updates the state.

## LoRA Mathematical Foundation

**Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)

For pre-trained weight W_0 in R^{d x k}:

```
h = W_0 * x + delta_W * x = W_0 * x + B * A * x
B in R^{d x r}, A in R^{r x k}, r << min(d, k)
```

Scaled: `h = W_0 * x + (alpha/r) * B * A * x`

**Key finding:** delta_W has low intrinsic rank. r=1 achieves competitive performance. Different random seeds converge to overlapping subspaces. delta_W amplifies directions not emphasized by W_0 with amplification factor ~21.5x for r=4.

### Intrinsic Dimensionality

Aghajanyan et al. (2020) showed that optimizing only ~200 randomly-projected parameters, RoBERTa achieves 90% of full fine-tuning on MRPC. Pre-training implicitly minimizes intrinsic dimensionality. Hu et al. extend this: the update space is overwhelmingly redundant, and effective degrees of freedom for task adaptation are orders of magnitude smaller than the parameter count.

## PLE Mathematical Foundation

**Source:** Gemma 4 architecture (HuggingFace transformers, `models/gemma3n/modeling_gemma3n.py`)

### Stage 1 -- Token Identity Lookup

```
E_PLE in R^{V x (L * p)}     # V=262144, L=num_layers, p=256
e(token) = sqrt(p) * E_PLE[token_id]    # reshaped to [B, S, L, p]
```

### Stage 2 -- Context Projection

```
W_proj in R^{d x (L * p)}
c = RMSNorm(W_proj * x_embed / sqrt(d))    # reshaped to [B, S, L, p]
PLE_combined = (e + c) / sqrt(2)
```

Combines token identity (context-independent) with context awareness (position/content dependent).

### Stage 3 -- Per-Layer Gated Injection

At each decoder layer l, AFTER attention + FFN:

```
g_l = GELU(W_gate_l * h_l)                    # R^d -> R^p
z_l = g_l (element-wise) PLE_combined[:,:,l,:] # Hadamard product in R^p
out_l = RMSNorm(W_up_l * z_l)                 # R^p -> R^d
h_{l+1} += out_l                               # residual injection
```

## Structural Comparison

Both PLE and LoRA are **low-rank conditioning of the residual stream**:

```
h' = h + f(h, theta)    where f produces a rank-bounded perturbation
```

| Property | LoRA | PLE |
|----------|------|-----|
| Core operation | h + B * A * x | h + W_up * (GELU(W_gate * h) . lookup[token, layer]) |
| Bottleneck dimension | r (typically 4-64) | p = 256 |
| What is adapted | Weight matrices (W_q, W_k, W_v, etc.) | Residual stream directly |
| Conditioning signal | Input x (same for all tasks) | Token identity + context (per-layer) |
| When computed | During linear projection | After attention+FFN block |
| Input-dependent? | Linear in x | Nonlinear (GELU + Hadamard) |
| Layer-specific? | Separate A, B per layer | Separate gate/proj per layer + shared lookup |
| Training paradigm | Post-hoc; freeze W_0, train A, B | End-to-end during pre-training |

### Mathematical Equivalences

**Without the nonlinearity and lookup table**, PLE injection reduces to:
```
out = W_up * W_gate * h    (rank-p linear residual)
```
This is precisely LoRA with rank=p applied to the identity mapping: W_0 = I, B = W_up, A = W_gate.

**The Hadamard product with the PLE lookup** makes the effective rank input-dependent: different tokens activate different bottleneck dimensions. PLE is a **conditional adapter** -- a generalization of static LoRA.

### Where They Diverge

1. **PLE is nonlinear; LoRA is linear.** GELU + Hadamard gives PLE strictly more expressive power per bottleneck parameter.
2. **PLE has per-token conditioning; LoRA does not.** The lookup table provides per-token, per-layer "personality." LoRA applies the same subspace regardless of token.
3. **PLE operates post-block; LoRA operates within projections.** LoRA modifies how a weight matrix transforms input. PLE modifies the residual stream after the entire block.
4. **PLE separates storage from compute.** The lookup table lives off-accelerator. LoRA adapters are small enough for GPU but don't leverage this asymmetry.

## LAUREL Is LoRA

LAUREL (Learned Augmented Residual Layer), part of Gemma 3n/4, computes:
```
h + RMSNorm(W_right * W_left * h)
W_left in R^{64 x d}, W_right in R^{d x 64}
```

This is **LoRA with rank=64 applied to the identity mapping**, with RMSNorm replacing zero initialization as the mechanism for ensuring small perturbations at start. Google built LoRA into the Gemma architecture under a different name.

## The LoRA Family Converging Toward PLE

Recent LoRA variants independently rediscover pieces of PLE:

| Variant | Key Innovation | PLE Connection |
|---------|---------------|----------------|
| **VeRA** (ICLR 2024) | Shared frozen random matrices + per-layer diagonal scaling | Shared structure + per-layer modulation = PLE's shared lookup + per-layer gates |
| **DoRA** (ICML 2024) | Magnitude/direction decomposition | Separating "what" from "how much" parallels PLE's lookup + gating split |
| **LoRA-XS** (2024) | Frozen SVD projectors + small trainable core | Frozen structure + learnable core = PLE's frozen lookup + learnable gates |
| **LoRA-FA** (2024) | Frozen random A, only B trained | Frozen random projector the model learns to use = simplest PLE table analog |
| **GaLore** (2024) | Low-rank gradient projection with periodic SVD | Training-time analog of PLE's inference-time insight: active subspace is small |
| **MoLoRA** (2026) | Per-token adapter routing via learned router | Architecturally convergent with PLE: different low-rank modifications per token |
| **Brainstacks** (2026) | Frozen MoE-LoRA stacks composing additively | Frozen stacks adding to residual = PLE lookup entries adding to residual stream |

MoLoRA on Qwen3-1.7B exceeds Qwen3-8B across four reasoning benchmarks while being 4.7x smaller. Per-token routing achieves O(N) work vs K*N for per-sequence routing.

## The TRM Connection

The CogOS TRM applies the Tiny Recursive Model technique ([Jolicoeur-Martineau 2024](https://arxiv.org/abs/2510.04871)) using a Mamba SSM backbone instead of the original paper's attention-based architecture. The result is a 2.28M-parameter model (D_STATE=4, N_LAYERS=2, EXPAND_FACTOR=1) that maintains a "light cone" -- a compressed hidden state representing the observer's trajectory through the workspace. The Mamba SSM choice is deliberate: its O(1) state update and persistent hidden state naturally track temporal evolution, whereas an attention-based TRM would need to re-attend to the full history each cycle. The SSM state IS the compressed history.

```
PLE:       token_id  -> lookup_table -> 256-dim conditioning -> gates layer behavior
Mamba TRM: sequence  -> SSM state    -> D_STATE=4 hidden     -> gates context selection
```

Both are **low-rank conditioning mechanisms that modulate a larger system through a compressed state.**

### Shared Properties

| Property | PLE | LoRA | Mamba TRM |
|----------|-----|------|-----------|
| State dimension | 256 | r (4-64) | 4 |
| Update per | Token | (static per task) | Attention event |
| Conditions | Decoder layer computation | Weight matrix output | Context candidate selection |
| Residual injection | h += gated_projection | h += B * A * x | score += SSM_output |
| Storage model | Flash-mapped lookup | GPU-resident adapter | In-memory SSM state |
| Size vs system | ~69M vs ~2.3B active | ~35MB vs ~350GB | 2.28M vs 30K chunks |

## Intrinsic Dimensionality

All these mechanisms point at the same empirical finding:

```
Full parameter space:    R^{d x d}    (~4M params per weight matrix)
LoRA subspace:           R^{d x r}    (~4K-64K params, static per task)
PLE conditioning:        R^{256}      (~256 params, dynamic per token per layer)
Intrinsic dimension:     R^{~200}     (Aghajanyan: 200 params for 90% performance)
TRM light cone:          R^{4}        (D_STATE=4, dynamic per event)
```

**The viable manifold of model behavior is low-dimensional.** The TRM achieving NDCG 0.900 with D_STATE=4 is the most extreme demonstration -- the entire observer trajectory through a workspace with thousands of chunks can be compressed to 4 dimensions and still predict relevance better than any static similarity measure.

### The Matryoshka Connection

Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022) demonstrates embeddings have nested structure -- first m dimensions contain a valid representation for any m < d. LoRA's rank r selects how many nesting levels of the update to capture. PLE's p=256 is a fixed nesting level. GaLore's periodic SVD recomputes optimal nesting at each interval.

Same fundamental principle: **meaningful model modulation lives in a low-dimensional subspace, and this subspace may vary by layer, by token, and over training.**

## LoRO -- The Unified Abstraction

We propose **LoRO (Low-Rank Observer)** as the unified abstraction:

```
state_t = update(state_{t-1}, observation_t)              # State update
conditioning_t = gate(project(state_t))                    # Bottleneck projection
system_output_t = base_computation + inject(conditioning_t) # Residual modulation
```

### Instantiations

| | PLE | LoRA | TRM |
|---|---|---|---|
| **state** | PLE_combined[token, layer] | Adapter weights (A, B) | SSM hidden state h_t |
| **update** | Lookup + context projection | Gradient descent on A, B | Mamba selective scan |
| **gate** | GELU(W_gate * h) . state | alpha/r scaling | Sigmoid output gate |
| **project** | W_up (R^256 -> R^d) | B (R^r -> R^d) | Score projection |
| **base_computation** | Decoder layer output | Pre-trained weight output | Cosine similarity baseline |
| **inject** | Residual add to hidden state | Residual add to projection | Score addition |

### Time Scales

The three mechanisms operate at different time scales of the same pattern:

- **PLE:** Per-token (nanoseconds). Conditions each layer within a single forward pass.
- **LoRA:** Per-task (hours-days). Adapts weight matrices across a fine-tuning run.
- **TRM:** Per-session (minutes-hours). Updates light cone state across interaction events.

These are nested temporal loops of the same low-rank observer pattern, each feeding into the next:

```
TRM (session) -> selects context -> PLE (token) -> conditions generation -> user interaction -> TRM update
                                     ^
                                     |
                                     LoRA (task) -> adapts base weights for domain
```

## The Substrate Principle

CogOS fills a specific architectural niche: **the gap between the generative model and the observer**. The substrate handles what's relevant, what zone it goes in, what tools are safe, what the intent is, and what the momentum is -- all before the model generates.

The model is a **pure generation engine**. It doesn't reason about its own context. The substrate already did that with specialized, fast, deterministic code.

This maps onto the LoRO framework at every scale:
- **PLE** is this principle applied *inside* the model (per-layer conditioning from lookup tables, not from attention)
- **TRM** is this principle applied *outside* the model (context selection from SSM state, not from model reasoning)
- **Foveated assembly** is this principle applied *between* model and user (zone ordering, not prompt stuffing)

The substrate is the observer's coupling function. The model is the generator. CogOS keeps them cleanly separated.

## Practical Implications

### Architecture

```
Base model weights (shared across all devices, resident in VRAM)
  +-- PLE table A: "reasoning agent" (flash-mapped, ~2GB)
  +-- PLE table B: "tool dispatcher" (flash-mapped, ~2GB)
  +-- PLE table C: "voice interface" (flash-mapped, ~2GB)
  +-- LoRA adapter D: "domain-specific" (GPU, ~100MB)
  +-- TRM light cone: per-conversation SSM state (memory, ~10KB)
```

Same base weights across all modes. Swap behavior by swapping conditioning signals.

### Novel Directions (Unexplored)

1. **PLE-tuning:** Fine-tune PLE tables instead of (or alongside) LoRA. Different tables = different behaviors from same base model. Flash-mappable = hot-swappable.

2. **TRM-conditioned PLE:** Feed TRM light cone state into PLE conditioning. The observer's trajectory directly modulates per-layer token processing.

3. **Speculative decoding across LoRO scales:** Smallest device uses TRM for fast context scoring (microseconds), proposes context candidates; larger device verifies with full PLE-conditioned generation.

4. **Stacked LoRO:** Two layers of low-rank modulation -- PLE (pre-trained token-level) + LoRA (task-level). Both leaving base weights untouched. Both operating through bottleneck mechanisms. Potentially multiplicative rather than additive.

5. **PLE table distillation from larger models:** Train small-model PLE tables to approximate large-model behavior on specific domains. The PLE table becomes a "compressed expert."

## Literature Gap

While individual mechanisms are well-studied (LoRA variants, PLE in Gemma, SSMs for sequence modeling), the structural convergence across all three — and especially the connection to SSM-based context scoring — appears uncharted as of April 2026.

- The LoRA comprehensive review (arXiv 2501.00365) does not include PLE in its taxonomy
- MoLoRA (2026) moves LoRA toward PLE-like per-token conditioning but doesn't cite PLE
- Brainstacks (2026) resembles PLE operationally but frames entirely in LoRA/MoE tradition
- No one has connected PLE or LoRA to SSM-based context conditioning (the TRM pattern)

The structural parallel -- PLE, LoRA, and compact SSMs as instances of low-rank residual conditioning with self-referential feedback -- appears to be an original observation.

## Sources

### Foundational Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) -- Hu et al., 2021
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) -- Aghajanyan et al., ACL 2021
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) -- Kusupati et al., NeurIPS 2022
- [MatFormer: Nested Transformer for Elastic Inference](https://arxiv.org/abs/2310.07707) -- NeurIPS 2024
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) -- Perez et al., 2018
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) -- Gu & Dao, 2023

### LoRA Variants

- [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454) -- Kopiczko et al., ICLR 2024
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) -- Liu et al., ICML 2024
- [LoRA-XS: Low-Rank Adaptation with Extremely Small Parameters](https://arxiv.org/abs/2405.17604) -- Banaei et al., 2024
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) -- Zhao et al., 2024
- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965) -- 2026
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285) -- 2023
- [Brainstacks: Cross-Domain via Frozen MoE-LoRA Stacks](https://arxiv.org/abs/2604.01152) -- 2026
- [Low-Rank Adaptation for Foundation Models: A Comprehensive Review](https://arxiv.org/abs/2501.00365) -- 2025

### Gemma Architecture

- [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)
- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4) -- Grootendorst
- [Reverse Engineering Gemma 3n](https://github.com/antimatter15/reverse-engineering-gemma-3n) -- antimatter15
