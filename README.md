# CogOS Research

**Part of the [CogOS](https://github.com/cogos-dev) ecosystem -- why it WORKS.**

Theoretical foundations, architecture research, and proof-of-concept experiments for cognitive operating system design.

## Contents

| Document | Description |
|----------|-------------|
| [eaefm/thesis.md](eaefm/thesis.md) | **EA/EFM Thesis** -- Externalized Attention and Executive Function Modulation. The core argument: the substrate thinks, the model generates, and quality is a function of boundary quality. |
| [loro/framework.md](loro/framework.md) | **LoRO Framework** -- Low-Rank Observer as a unified abstraction connecting PLE (Per-Layer Embeddings), LoRA (Low-Rank Adaptation), and TRM (Tiny Recursive Model). Three mechanisms, one pattern, operating at different time scales. |
| `poc/` | Proof-of-concept experiments (coming soon) |
| `papers/` | Academic papers and preprints (coming soon) |

## What this is

This repo contains the public research that underpins CogOS -- the ideas about *why* externalizing attention and executive function into a substrate produces better outcomes than scaling model size alone.

The key claims:

1. **EA (Externalized Attention):** Deciding what information is relevant *before* the model sees it -- not retrieval, not augmentation, but selective amplification of what matters.
2. **EFM (Executive Function Modulation):** Deciding how the model should behave *before* it generates -- not prompting, but shaping the computational trajectory through conditioning signals.
3. **LoRO (Low-Rank Observer):** PLE, LoRA, and TRM are structurally convergent mechanisms -- all low-rank conditioning of a larger system through a bottleneck. This convergence is not noted in published literature as of April 2026.

## What this is not

This is public architecture research related to CogOS. It does not contain the full theoretical framework, fundamental physics, or private workspace internals.

## Related projects

- [cogos](https://github.com/cogos-dev/cogos) -- The kernel — continuous process daemon with foveated context and multi-provider routing
- [constellation](https://github.com/cogos-dev/constellation) -- Distributed trust — identity as temporal coherence, O(1) mutual verification
- [mod3](https://github.com/cogos-dev/mod3) -- Modality bus — translates between thinking and acting, voice-first
- [skills](https://github.com/cogos-dev/skills) -- Plugin marketplace — Agent Skills across workflow, research, voice, and dev tools
- [charts](https://github.com/cogos-dev/charts) -- Deployment — Helm charts + Docker Compose
- [desktop](https://github.com/cogos-dev/desktop) -- Native macOS app — kernel management, terminal, dashboard
- [openclaw-plugin](https://github.com/cogos-dev/openclaw-plugin) -- OpenClaw integration (how it CONNECTS)

## License

MIT
