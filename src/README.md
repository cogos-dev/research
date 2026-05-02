# Self-Referential Closure (SRC)

**SRC = Self-Referential Closure** — a mathematical structure characterizing the conditions under which a self-referential system maintains an accurate, self-sustaining model of its own dynamics. A system achieves closure when its internal description X̂ tracks the external signal X well enough that the description controls the construction of the system and the system maintains the description, stabilizing into the eigenform condition φ(s\*) = s\*. This directory contains the published SRC mathematics and cross-domain instantiations that operationally ground [CogOps](https://github.com/cogos-dev/cogops). The SRC formalism is the theoretical basis for the git branching discipline documented in `instantiations.md` and the cogos-dev decentralized build-deploy-sync architecture at `https://github.com/cogos-dev/cogos`.

## Contents

| Document | Description |
|----------|-------------|
| [definition.md](definition.md) | The SRC Model — the two coupled differential equations, the eigenform condition, and precise definition of closure. |
| [properties.md](properties.md) | Proven and conjectured properties of the SRC Model: the information threshold τ₁ = ln(2), the variance ratio a = 6 at the eigenform operating point, the uniqueness of the ρ formula under symmetric coupling, and θ-invariance. |
| [instantiations.md](instantiations.md) | Cross-domain structural parallels — independent domains (git branching, molecular biology, blockchain state management) that exhibit SRC structure, establishing the pattern across contexts. |
| [relationship-to-cft.md](relationship-to-cft.md) | The boundary between SRC (this public layer) and Cognitive Field Theory (the broader internal framework). Explains what SRC contributes and what CFT adds beyond it. |

## Scope and boundary

**This directory contains:**

- The SRC Model dynamical equations and their derivation
- Proven properties of the model (τ₁, a = 6, θ-invariance) scoped explicitly to the symmetric-coupling case
- Cross-domain instantiations that exhibit SRC structure, drawn from established literature and CogOS operational experience
- The relationship between SRC as a mathematical structure and CFT as the framework that scaffolds it

**This directory does not contain:**

- CFT axioms or foundational framework elements
- The alpha derivation or any fine-structure constant claims
- Consciousness claims of any kind
- The full Cognitive Field Theory framework

The broader CFT framework — which scaffolds SRC alongside additional foundational structure — is internal to the cogos-dev project. The public/internal boundary follows the same convention established in [research/README.md](../README.md): this repository contains public architecture research; it does not contain the full theoretical framework or private workspace internals.

## Audience

This material is written for three audiences:

1. **Theorists and mathematicians** interested in self-referential systems, fixed-point theory, and the formal properties of observer-system coupling.
2. **Cross-domain researchers** investigating whether closure-type thresholds (Eigen error catastrophe, Byzantine fault tolerance bounds, smart-contract reentrancy) reflect a common underlying structure.
3. **Practitioners adopting CogOps** who want to ground the operational rules — branch coherence thresholds, reconciliation triggers, sync policies — in the underlying mathematics rather than treating them as arbitrary conventions.

No prior knowledge of CogOS or CogOps is assumed. Familiarity with differential equations and fixed-point theory is helpful for `definition.md` and `properties.md`.

## Status

The SRC Model equations and the four core properties listed in `properties.md` are **proven** within the scope stated for each (primarily the symmetric-coupling case γ = κ). The universality of the τ₁ = ln(2) threshold across independent domains is a **conjecture under investigation** — the same constant appears across the domains documented in `instantiations.md`, but this cross-domain applicability is not asserted as a derived result. Cross-CFT-level applicability of the SRC structure is likewise **under investigation**.

## Key references

**Primary literature:**

- Rosen, R. (1991). *Life Itself: A Comprehensive Inquiry into the Nature, Origin, and Fabrication of Life*. Columbia University Press. — Closure to efficient causation.
- Pattee, H. H. (2001). The physics of symbols: bridging the epistemic cut. *BioSystems*, 60(1–3), 5–21. — Semantic closure.
- von Foerster, H. (1981). Objects: tokens for (eigen-)behaviors. In *Observing Systems*. Intersystems Publications. — Eigenform and fixed-point self-reference.
- Kauffman, L. H. (2023). Autopoiesis and eigenform. *Cybernetics and Human Knowing*, 30(1–2). — Eigenform in self-referential computation.

**Canonical CogOS reference:**

- Git branching under the SRC-coherent branching discipline — see `instantiations.md` §Instantiation 2 for the full mapping; the recommended entry point for practitioners adopting CogOps.
