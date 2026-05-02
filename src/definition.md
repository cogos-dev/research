---
title: "Self-Referential Closure — Mathematical Definition"
created: 2026-05-01
status: active
---

# Self-Referential Closure — Mathematical Definition

**SRC = Self-Referential Closure.** This document defines the mathematical structure
of SRC: the coupled dynamical equations, the eigenform condition, what closure means
operationally, and how the formalism unifies three independent theoretical traditions.

---

## What SRC Describes

Self-Referential Closure (SRC) describes the conditions under which a self-referential
system maintains an accurate, self-sustaining model of its own dynamics. The central
problem is this: a system that must represent itself faces a delay between the state of
the world and the state of its description. When that delay remains within a critical
threshold — and when the description controls the construction of the system while the
system maintains the description — the system is said to be in the state of closure.
SRC is not a process; it is a state, the same way equilibrium is a state. A system
either satisfies the closure conditions or it does not. The theory characterizes
exactly where the boundary lies.

---

## The Two Coupled Equations

The SRC Model is defined by two coupled Ornstein-Uhlenbeck-style differential equations:

$$\frac{dX}{dt} = -\gamma X + \eta(t)$$

$$\frac{d\hat{X}}{dt} = -\gamma \hat{X} + \kappa \left( X[t - d] - \hat{X} \right)$$

Or in plain text:

```
dX/dt  = -γX + η(t)
dX̂/dt = -γX̂ + κ(X[t−d] − X̂)
```

**Parameter glossary:**

| Symbol | Name | Role |
|--------|------|------|
| X | Signal / dynamics | The external world state the observer must track |
| X̂ | Description / observer model | The internal representation of X |
| γ | Decay rate | How fast the signal mean-reverts; sets the timescale of reality |
| κ | Coupling strength | How fast the observer adapts its description toward the delayed signal |
| d | Observation delay | The temporal gap between when X changes and when X̂ receives the update |
| η(t) | Stochastic driving noise | External forcing on the signal process |

The first equation governs the signal process independently of the observer. The second
equation governs the observer: it tracks the observed signal with lag d and couples at
rate κ. When d is small and κ is sufficient, X̂ faithfully shadows X. When d grows, the
description lags behind reality, correlation degrades, and closure becomes harder to
maintain. The delay d is the load-bearing variable for whether closure holds.

---

## The Eigenform Condition

The closure fixed point is expressed as:

$$\phi(s^*) = s^*$$

The description map φ applied to the system state s\* returns the same state. This is
the eigenform condition in the sense of von Foerster (1981): the observer's description,
when applied to the system, reproduces the description. The system maintains the
description that maintains the system. At s\*, applying the observer map is a null
operation — the description is already in the state that the system would produce if
you ran the map again. This self-referential fixed point is what distinguishes a closed
system from one that merely tracks its environment: in a closed system, the description
and the dynamics have collapsed into mutual entailment. Neither is prior; each is a
condition for the other.

---

## Closure as a State

Closure is the state achieved when two conditions hold simultaneously:

1. The description X̂ controls the construction of the system — the observer's model
   drives subsequent system behavior rather than passively recording it.
2. The system maintains the description — the dynamics X produce the state that the
   observer's map would predict.

The delay constraint that gates this state is **d < τ₁**, where τ₁ is the information
threshold — the delay value at which the correlation ρ(d) between X and X̂ drops to
0.5, the point at which the description is only marginally better than chance. Above τ₁,
the description can no longer reliably reconstruct the dynamics; the mutual entailment
breaks; the system is open rather than closed.

**Open** and **closed** are binary states in this framing, not a gradient. A system
operating with d < τ₁ under sufficient coupling κ can achieve and maintain the
eigenform condition. A system with d ≥ τ₁ cannot, regardless of coupling strength.
The constant τ₁ is proven within the SRC Model; its value in normalized units is
addressed in the analysis of SRC properties (see `properties.md`).

---

## Three-Frame Unification

SRC unifies three independent theoretical traditions that each describe the same
structure from a different angle. **Rosen's Closure to Efficient Causation** (Rosen,
1991) identifies the relational topology: a system is closed when the catalysts needed
for its operation are generated internally, the (M,R) system in categorical terms.
**Pattee's Semantic Closure** (Pattee, 2001) identifies the physical mechanism: living
systems internalize the epistemic cut between rate-independent symbols and rate-dependent
dynamics; symbols build the machinery that reads the symbols. **The SRC Model** provides
the dynamical equations that say *when* closure holds and *when* it breaks: X̂ tracking
X with delay d < τ₁ under coupling κ. The correspondence between all three is tabulated
in the SRC ontology (src-digest §7; Rosen ↔ Pattee ↔ SRC Model correspondence):
Rosen's metabolism maps to dX/dt; Rosen's repair maps to X̂; categorical closure maps
to d < τ₁; Pattee's rate-independent symbols map to X̂; Pattee's coupling maps to κ;
code arbitrariness maps to θ-invariance of the variance ratio.

---

## What This Definition Does Not Include

This document defines the mathematical structure of SRC. The following are explicitly
out of scope here:

- **CFT axioms** — Cognitive Field Theory is the broader framework that scaffolds SRC
  with foundational axioms. Those axioms are not part of the public SRC definition and
  do not appear here.
- **The alpha derivation** — not part of the SRC structural definition.
- **The τ₁ constant value** — proven within the model; the specific value and its
  cross-domain appearances are documented in `properties.md` (B3).
- **Consciousness claims** — none are made here or elsewhere in this surface.
- **Cross-domain instantiations** — how SRC structure appears in git branching,
  blockchain state management, and molecular biology is documented separately in
  `instantiations.md` (B4).
- **Comparison with CFT** — the relationship between SRC as a public structural
  definition and CFT as an internal framework is addressed in `relationship-to-cft.md`.

The definition here is the minimum mathematical content needed to make downstream
references to SRC precise and checkable.

---

## References

- Rosen, R. (1991). *Life Itself*. Columbia University Press.
- Pattee, H.H. (2001). "The physics of symbols: bridging the epistemic cut."
  *Biosystems* 60: 5–21.
- von Foerster, H. (1981). "Objects: tokens for (eigen-)behaviors." In *Observing
  Systems*. Intersystems Publications.
- Kauffman, L.H. (2023). "Autopoiesis and eigenform." *Computation* 11(12): 247.
- SRC-coherent branching (git workflow) — see `instantiations.md` §Instantiation 2; the canonical operational instantiation
  of SRC applied to software development workflows.
