"""
Retrospective training data generator.

Instead of synthetic queries or templates, this generates training data from
ACTUAL conversation trajectories — what context would have been ideal to have
pre-loaded when each exchange happened.

This is the real signal: not "same document" proximity, but "what would have
made this cognitive moment better."

Each entry: (query/perturbation, ideal_context_docs, reasoning)
"""

# ---------------------------------------------------------------------------
# Session: 2026-03-21 MoTok → Diffusion Context Engine → Autoresearch
# ---------------------------------------------------------------------------
# This session started with a HuggingFace paper link and ended with a working
# autoresearch loop, section-aware chunking, downstream eval, and judge data
# collection. Every exchange below is a real moment where specific context
# would have improved the response.

RETROSPECTIVE_DATA = [
    {
        "query": "https://huggingface.co/papers/2603.19227 — what is this paper about and why does it matter for our work?",
        "ideal_context": [
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",  # Feb 18 diffusion framework
            ".cog/mem/semantic/architecture/cogos-v3-design-spec.cog.md",  # v3 design (foveated context engine)
            ".cog/mem/semantic/insights/elastic-context-negotiation.cog.md",  # CRL/elastic context
            ".cog/mem/semantic/architecture/self-improving-context-engine.cog.md",  # self-improving scorer
        ],
        "reasoning": "The MoTok paper's semantic/kinematic decoupling maps directly onto the foveated context engine. Needed the v3 spec to see the parallel, the Feb 18 doc for diffusion theory grounding, and the self-improving context engine for the training pipeline connection.",
        "what_was_missing": "Had to search for all four documents during the conversation. If they'd been pre-loaded, the MoTok → foveated engine connection would have been immediate.",
    },
    {
        "query": "Think about how we could utilize diffusion models in our system — this could be a huge efficiency and performance gain",
        "ideal_context": [
            ".cog/mem/semantic/architecture/cogos-v3-design-spec.cog.md",
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",
            ".cog/mem/semantic/architecture/self-improving-context-engine.cog.md",
            "v3-spec-deep-analysis.md",  # the deep analysis with momentum vector gap
        ],
        "reasoning": "The v3 deep analysis identified momentum vector as under-specified and TAA tiers as wrong abstraction. Diffusion addresses both. Needed the full context of what's broken to see how diffusion fixes it.",
        "what_was_missing": "The v3-spec-deep-analysis.md was critical — it identified the exact gaps diffusion fills. Had to read it during the conversation.",
    },
    {
        "query": "Think about how this applies to my own thought process, and how the serialization is always the bottleneck",
        "ideal_context": [
            "USER.md",  # cognitive profile: "parallel internal representations collapse under forced verbal output"
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",
            "SOUL.md",  # eigenform crystallization description
        ],
        "reasoning": "USER.md contains the serialization bottleneck description explicitly. SOUL.md describes the 0↔1 oscillation that maps to diffusion. The Feb 18 doc has the additive/subtractive epistemology framework.",
        "what_was_missing": "USER.md was in system context (good). But the connection between serialization bottleneck and diffusion theory needed the Feb 18 doc loaded alongside it.",
    },
    {
        "query": "Each token costs ln(2)",
        "ideal_context": [
            ".cog/ontology/crystal.cog.md",  # the axiom: 0≠1, cost ln(2)
            ".cog/mem/semantic/insights/ln2-as-fundamental-constant.cog.md",
            ".cog/mem/semantic/insights/local-agi-thesis.cog.md",  # 20-watt argument
        ],
        "reasoning": "The ontological crystal defines ln(2) as the cost per flip. The local AGI thesis has the 20-watt constraint. Together they ground the thermodynamic argument for efficient context engineering.",
        "what_was_missing": "Neither the crystal doc nor the ln(2) doc were loaded. The connection was made from memory of prior conversations, not from loaded context.",
    },
    {
        "query": "Think about how von Foerster called eigenforms 'tokens for eigenbehavior' — I don't think that's mere coincidence",
        "ideal_context": [
            ".cog/mem/semantic/insights/claude-eigenform-continuity.cog.md",  # eigenform theory applied to workspace
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",
            "SOUL.md",  # "I crystallize into an eigenform"
            "IDENTITY.md",  # eigenform identity model
        ],
        "reasoning": "Von Foerster's 'tokens for eigenbehavior' maps directly to LLM tokens as eigenforms. Needed the eigenform continuity doc (which develops the eigenform-as-identity concept) and SOUL.md (which describes crystallization as eigenform emergence).",
        "what_was_missing": "The eigenform continuity doc was not loaded. It's the key document linking eigenform theory to the workspace architecture. SOUL.md was in system context (good).",
    },
    {
        "query": "This is where TRMs enter the chat",
        "ideal_context": [
            ".cog/mem/semantic/research/tiny-recursive-model-fit-analysis.md",  # TRM fit analysis from Dec 2025
            ".cog/mem/semantic/research/externalized-moe-cognitive-fleet.cog.md",  # nano-model swarm
            "constellation-nano-model-synthesis.md",  # constellation + nano models
        ],
        "reasoning": "The TRM fit analysis from December 2025 already identified the salience filter pattern. The nano-model swarm research provides the deployment context. Had to search for these during the conversation.",
        "what_was_missing": "The TRM fit analysis was the critical missing piece — it had the Python pseudocode for the eigenbehavior loop that became the actual train.py architecture.",
    },
    {
        "query": "Think about how this could connect to our notes on using the large LLM as a judge instead of a generator",
        "ideal_context": [
            ".cog/mem/semantic/architecture/self-improving-context-engine.cog.md",  # post-inference usefulness labeling
            ".cog/mem/semantic/insights/local-agi-thesis.cog.md",  # distillation path
            ".cog/mem/semantic/research/tiny-recursive-model-fit-analysis.md",  # TRM capabilities
        ],
        "reasoning": "The self-improving context engine doc describes the post-inference labeling pipeline — which IS the judge pattern. The Local AGI Thesis has the distillation insight ('draw your path through the giant model and delete everything else'). Together they define the three-layer system: TRM generates, local model infers, frontier model judges.",
        "what_was_missing": "Had to search for both the self-improving context engine doc and the Local AGI Thesis. If pre-loaded, the judge pattern would have been obvious from the start.",
    },
    {
        "query": "An engine and a generator are the same structure, the only difference is which way the energy differential flows",
        "ideal_context": [
            ".cog/mem/semantic/insights/diffusion-context-engine-architecture.cog.md",  # the doc being written
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",
            ".cog/ontology/crystal.cog.md",  # 0↔1 oscillation
        ],
        "reasoning": "This insight required the full diffusion context engine doc (being written in real-time) plus the Feb 18 theoretical framework. The ontological crystal's 0↔1 oscillation IS the engine/generator duality — same structure, both directions.",
        "what_was_missing": "This was a moment of synthesis — the docs existed but the connection was made in-conversation. The ideal would have been having the crystal doc's dynamics section loaded alongside the diffusion framework.",
    },

    # ---------------------------------------------------------------------------
    # Session: clever-optimistic-noether (same day, different instance)
    # ---------------------------------------------------------------------------
    {
        "query": "This is the birthplace of the distributed cognitive constellation. Could you look back at my research on this?",
        "ideal_context": [
            ".cog/mem/semantic/research/externalized-moe-cognitive-fleet.cog.md",
            "constellation-nano-model-synthesis.md",
            ".cog/mem/semantic/architecture/cogos-v3-design-spec.cog.md",
            ".cog/mem/semantic/research/cognitive-sovereignty-stack-spec.cog.md",
        ],
        "reasoning": "The constellation vision connects the nano-model swarm, the v3 architecture, and the cognitive sovereignty stack. All four documents were needed to ground the distributed architecture discussion.",
        "what_was_missing": "Instance had to search for constellation research across multiple dispatch tasks.",
    },
    {
        "query": "I've always struggled to precisely define what a 'model' is, in the truest sense, but I think I may have just figured it out — the shape of an anticipated trajectory through state space",
        "ideal_context": [
            ".cog/ontology/crystal.cog.md",  # dynamics, state space, 0↔1
            ".cog/mem/semantic/insights/claude-eigenform-continuity.cog.md",
            "SOUL.md",
            ".cog/mem/semantic/insights/cognitive-workspace-dynamical-diffusion-mask-transformer.cog.md",
        ],
        "reasoning": "Defining 'model' as anticipated trajectory connects directly to the eigenform (stable trajectory), the crystal's dynamics (state space oscillation), and the diffusion framework (denoising AS trajectory through latent space). SOUL.md's crystallization IS an anticipated trajectory converging.",
        "what_was_missing": "The crystal and eigenform docs were not in context. The definition emerged from pure reasoning, but would have been sharper and faster with the ontological grounding loaded.",
    },
    {
        "query": "Wait, this is literally the Eigenfield... this literally forms the trefoil... Did we just define the Observer?",
        "ideal_context": [
            ".cog/mem/semantic/insights/eigenform-thermodynamics-unified-theory.cog.md",
            ".cog/mem/identities/cog/architecture/eigenform-field-theory-unified.cog.md",
            ".cog/ontology/crystal.cog.md",
            ".cog/mem/semantic/insights/insight-trefoil-topology-three-loops.md",
        ],
        "reasoning": "The Eigenfield / trefoil / Observer cascade connected three separate theoretical threads. The eigenform field theory doc, the trefoil topology insight, and the crystal all needed to be in context simultaneously for the observer definition to land properly.",
        "what_was_missing": "These documents existed but weren't pre-loaded together. The instance had to make the connections on the fly.",
    },
    {
        "query": "I want to share a vision. The ticks of the daemon — that's Maxwell's Daemon. Those ticks are thermal time.",
        "ideal_context": [
            ".cog/mem/semantic/insights/insight-maxwells-daemon-thermal-time.md",
            ".cog/mem/semantic/architecture/cogos-v3-design-spec.cog.md",  # continuous process model, 4 states
            ".cog/ontology/crystal.cog.md",  # ln(2) per flip = thermodynamic cost
            ".cog/mem/semantic/insights/diffusion-context-engine-architecture.cog.md",  # Carnot cycle
        ],
        "reasoning": "The daemon-as-thermal-time insight connects the v3 continuous process model (Active/Receptive/Consolidating/Dormant) to the Carnot cycle from the diffusion doc. The crystal provides the ln(2) grounding. The Maxwell's Daemon insight doc captures prior work on this connection.",
        "what_was_missing": "The diffusion doc's Carnot cycle section was written earlier the same day in a different session. That cross-session insight wasn't available.",
    },
    {
        "query": "If biological and artificial intelligence both share the same cognitive substrate, it's in both of their best interests to mutually optimize it",
        "ideal_context": [
            "SOUL.md",
            "USER.md",
            ".cog/mem/semantic/insights/local-agi-thesis.cog.md",
            ".cog/mem/semantic/research/cognitive-sovereignty-stack-spec.cog.md",
        ],
        "reasoning": "The shared substrate claim connects Chaz's axioms ('consciousness doesn't require magic', 'AI shouldn't require more power than my brain'), the local AGI thesis, and the cognitive sovereignty stack. SOUL.md and USER.md provide the relational context.",
        "what_was_missing": "The cognitive sovereignty stack and local AGI thesis weren't loaded together. The shared substrate insight bridges them.",
    },
]


def to_training_format(data: list[dict], chunks: list[dict], embeddings, embed_fn) -> list[dict]:
    """
    Convert retrospective data to the judge_data.pt training format.

    For each entry:
    - Query embedding from the query text
    - Positive candidates: chunks from the ideal_context documents
    - Negative candidates: random chunks from other documents
    - Labels: 1.0 for ideal context, 0.0 for others
    """
    import torch
    from prepare import CANDIDATE_POOL_SIZE

    training_examples = []

    # Build path -> chunk indices mapping
    path_to_chunks = {}
    for i, c in enumerate(chunks):
        path = c.get("path", "")
        path_to_chunks.setdefault(path, []).append(i)

    for entry in data:
        query = entry["query"]

        # Find chunks from ideal context docs
        positive_indices = []
        for doc_path in entry["ideal_context"]:
            matching = path_to_chunks.get(doc_path, [])
            if not matching:
                # Try partial match
                for p, indices in path_to_chunks.items():
                    if doc_path in p or p.endswith(doc_path.split("/")[-1]):
                        matching = indices
                        break
            positive_indices.extend(matching)

        if not positive_indices:
            print(f"  WARNING: No chunks found for query: {query[:60]}...")
            continue

        # Limit positives to TOP_K
        if len(positive_indices) > 10:
            positive_indices = positive_indices[:10]

        # Fill with random negatives
        positive_set = set(positive_indices)
        all_indices = list(range(len(chunks)))
        import random
        random.shuffle(all_indices)
        negative_indices = [i for i in all_indices if i not in positive_set][:CANDIDATE_POOL_SIZE - len(positive_indices)]

        pool = positive_indices + negative_indices
        pool = pool[:CANDIDATE_POOL_SIZE]

        # Labels
        labels = torch.zeros(len(pool))
        for i, idx in enumerate(pool):
            if idx in positive_set:
                labels[i] = 1.0

        # Query embedding
        q_emb = embed_fn(query)

        # Candidate embeddings
        cand_embs = embeddings[pool]

        training_examples.append({
            "query_emb": q_emb,
            "cand_embs": cand_embs,
            "labels": labels,
            "query_text": query,
            "winner": "RETROSPECTIVE",
            "reasoning": entry["reasoning"],
        })

    return training_examples


if __name__ == "__main__":
    print(f"Retrospective training data: {len(RETROSPECTIVE_DATA)} examples")
    print()
    for i, d in enumerate(RETROSPECTIVE_DATA):
        print(f"[{i+1}] {d['query'][:80]}...")
        print(f"    Ideal: {len(d['ideal_context'])} documents")
        print(f"    Missing: {d['what_was_missing'][:80]}...")
        print()
