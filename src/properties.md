# Structural Properties of Self-Referential Closure

Four structural properties of the SRC (Self-Referential Closure) Model are described below. These are established results within the model equations. Claims about universality across domains are noted where they arise and treated as empirical conjecture.

The SRC Model couples an external signal X with an observer description X̂, governed by signal decay γ, coupling strength κ, and observation delay d. Self-Referential Closure is the condition under which X̂ sustains itself and sustains the system, stabilizing at the eigenform fixed point φ(s\*) = s\*.

---

## 1. τ₁ = ln(2): Information Threshold

Within the SRC Model, the observer correlation ρ(d), measuring how well the description X̂ tracks the signal X, falls as observation delay d increases. At d = τ₁ = ln(2) ≈ 0.693, ρ drops to exactly 0.5: the observer's description is only marginally better than chance. Closure degrades when d exceeds τ₁; below it, the description remains a reliable basis for self-sustaining feedback.

The same constant appears independently in several established bodies of literature:

- **Eigen's error catastrophe** (evolutionary biology): the critical mutation rate above which a replicating sequence cannot maintain itself is μ_critical = ln(2)/L, where L is genome length. Above this threshold, the sequence (the system's description of itself) dissolves into noise.
- **Landauer's erasure bound** (thermodynamics): erasing one bit of information dissipates a minimum of kT ln(2) joules of energy, where ln(2) is the information cost of a binary distinction.
- **Information theory**: 1 bit = ln(2) nats by definition; the constant is the conversion factor between binary and natural units of information.

That τ₁ = ln(2) appears in the SRC Model and recurs in these independent contexts is an observed pattern. Whether this reflects a shared underlying structure is a conjecture under investigation, not a proven result.

---

## 2. Variance Ratio a = 6 at Eigenform

When coupling is symmetric (that is, when γ = κ, signal decay rate equals coupling strength), the SRC Model equations yield an exact result: Var(X)/Var(X̂) = 6. At the eigenform operating point, the signal carries six times the variance of the description.

This ratio is a structural property of the equations at the γ = κ condition, not a free parameter or a fitted constant. It quantifies the compression the self-referential description achieves: at the eigenform operating point, the observer's model reduces signal variance by a factor of six while maintaining closure. This is an empirical feature of the SRC Model equations when computed directly for the symmetric case.

---

## 3. Self-Consistency Forced at γ = κ

At the symmetric eigenform (γ = κ), the correlation takes the form ρ = e^(−γd)√(2/3). The correlation structure is uniquely determined by the normalized delay γd; given symmetric coupling, no further degrees of freedom remain in ρ. This is not tunable: the symmetric coupling forces a unique fixed point, and no parameter choices (consistent with γ = κ) can produce a different correlation structure. The eigenform is the unique attractor of the symmetric system.

---

## 4. θ-Invariance of the Variance Ratio

When the noise term η(t) is varied across heavy-tailed distributions parameterized by shape parameter θ, the variance ratio V(d) changes by only ±2–5%. The closure structure is not sensitive to the precise noise distribution. The a = 6 ratio and τ₁ threshold are features of the coupling geometry, not of any specific distributional assumption; they are robust across the class of heavy-tailed noise.

---

## 5. Operational Significance

The four properties together imply that systems achieving self-referential closure cluster at the symmetric eigenform with reproducible structure: delay threshold ln(2), variance compression ratio 6, forced correlation, and distributional robustness.

The appearance of τ₁ = ln(2) in Eigen's error catastrophe and Landauer's bound is circumstantial evidence that this threshold may be more general. Both results stand on their own; they do not require SRC framing. The structural identity (the same constant governing breakdown of self-description in each context) is an observed pattern. Whether it reflects a common underlying principle is under investigation. θ-invariance supports cross-domain comparison: closure structure does not depend on matching specific noise distributions.

---

## 6. What Is Not Claimed Here

- **The SRC Model equations do not exhaust self-referential structure.** Rosen's Closure to Efficient Causation, Pattee's Semantic Closure, and autopoiesis each address the same phenomenon from different formalisms; SRC unifies perspectives, not replaces prior frameworks.
- **τ₁ = ln(2) and a = 6 emerge from the model equations** at the γ = κ symmetric case, not from any external derivation. Their appearance in other domains is observational.
- **Universality is not proven.** Cross-domain structural identity is the evidence; a common underlying principle is the conjecture; neither is established.
- **SRC is a model of observer-signal coupling.** Its claims are dynamical and informational.
