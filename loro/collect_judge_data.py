"""
Automated judge data collector for TRM training.

Generates realistic workspace queries from CogDoc topics, runs TRM vs cosine
context selection, has a frontier model judge which set is better, and saves
labels for TRM training.

This is the judge loop — the selection pressure that teaches the TRM what
"load-bearing context" actually means.

Usage:
    uv run collect_judge_data.py                    # generate + judge 50 queries
    uv run collect_judge_data.py --n-queries 100    # more queries
    uv run collect_judge_data.py --generate-only    # just generate queries, don't judge
"""

import os
import sys
import json
import glob
import hashlib
import argparse
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from prepare import (
    EMBED_DIM, TOP_K, CANDIDATE_POOL_SIZE,
    find_cogdocs, chunk_document, compute_embeddings,
    CACHE_DIR, EMBED_FILE,
)
from train import TRM

# ---------------------------------------------------------------------------
# Query generation from workspace content
# ---------------------------------------------------------------------------

QUERY_TEMPLATES = [
    # Architecture / design questions (researcher's primary pattern)
    "What is {concept} and how does it work?",
    "How does {concept_a} relate to {concept_b}?",
    "What is the connection between {concept_a} and {concept_b}?",
    "How is {concept} implemented in the current architecture?",
    "What was the decision behind {concept}?",

    # Status / recall questions
    "What do we know about {concept}?",
    "What's the current status of {concept}?",
    "What did we decide about {concept}?",

    # Theoretical questions
    "How does {concept} derive from first principles?",
    "What is the theoretical basis for {concept}?",
    "How does {concept} connect to the SRC framework?",
    "What does {concept} mean in terms of eigenforms?",

    # Cross-domain connection questions (researcher's signature pattern)
    "How does {concept_a} from {domain_a} connect to {concept_b} from {domain_b}?",
    "What structural parallel exists between {concept_a} and {concept_b}?",
]


def extract_concepts_from_docs(docs: list[dict]) -> list[dict]:
    """
    Extract concepts, topics, and domains from document titles and sections.
    These become the seeds for query generation.
    """
    concepts = []
    domains = set()

    for doc in docs:
        title = doc.get("title", "")
        path = doc.get("path", "")

        # Extract domain from path
        parts = path.split("/")
        for p in parts:
            if p in ("semantic", "episodic", "procedural", "reflective"):
                continue
            if p in ("insights", "architecture", "research", "decisions", "sessions"):
                domains.add(p)

        # Title as concept
        if title and len(title) > 5 and len(title) < 100:
            concepts.append({
                "text": title,
                "path": path,
                "doc_id": doc.get("doc_id", ""),
            })

        # Section titles as concepts
        sections = doc.get("sections", [])
        if sections:
            for sec in sections:
                if isinstance(sec, dict):
                    sec_title = sec.get("title", "")
                    if sec_title and len(sec_title) > 5 and len(sec_title) < 80:
                        concepts.append({
                            "text": sec_title,
                            "path": path,
                            "doc_id": doc.get("doc_id", ""),
                        })

    return concepts, list(domains)


def generate_queries(concepts: list[dict], domains: list[str], n_queries: int = 50, seed: int = None) -> list[str]:
    """
    Generate realistic workspace queries by filling templates with extracted concepts.
    """
    rng = random.Random(seed or random.randint(0, 2**32))
    queries = set()

    # Single-concept templates
    single_templates = [t for t in QUERY_TEMPLATES if "{concept_b}" not in t and "{concept_a}" not in t]
    # Dual-concept templates
    dual_templates = [t for t in QUERY_TEMPLATES if "{concept_a}" in t and "{concept_b}" in t]

    attempts = 0
    while len(queries) < n_queries and attempts < n_queries * 10:
        attempts += 1

        if rng.random() < 0.6 and dual_templates and len(concepts) >= 2:
            # Cross-concept query
            template = rng.choice(dual_templates)
            c1, c2 = rng.sample(concepts, 2)

            q = template.replace("{concept_a}", c1["text"]).replace("{concept_b}", c2["text"])
            if "{domain_a}" in q:
                d1 = rng.choice(domains) if domains else "architecture"
                d2 = rng.choice(domains) if domains else "research"
                q = q.replace("{domain_a}", d1).replace("{domain_b}", d2)
        else:
            # Single-concept query
            template = rng.choice(single_templates)
            c = rng.choice(concepts)
            q = template.replace("{concept}", c["text"])

        if len(q) > 20 and len(q) < 300:
            queries.add(q)

    return list(queries)[:n_queries]


# ---------------------------------------------------------------------------
# Context selection (TRM vs cosine)
# ---------------------------------------------------------------------------

def load_workspace(workspace_root: str):
    """Load docs, chunks, embeddings, and TRM model."""
    print("Loading workspace documents...")
    docs = find_cogdocs(workspace_root)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} docs -> {len(all_chunks)} chunks")

    print("Loading embeddings...")
    if os.path.exists(EMBED_FILE):
        embeddings = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)
        if embeddings.size(0) != len(all_chunks):
            print(f"  Embedding count mismatch ({embeddings.size(0)} vs {len(all_chunks)}), recomputing...")
            embeddings = compute_embeddings(all_chunks)
            torch.save(embeddings, EMBED_FILE)
    else:
        embeddings = compute_embeddings(all_chunks)
        torch.save(embeddings, EMBED_FILE)

    # Load TRM
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    model = None
    if os.path.exists(model_path):
        print(f"Loading TRM from {model_path}...")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model = TRM(
            embed_dim=config.get("embed_dim", EMBED_DIM),
            latent_dim=config.get("latent_dim", 256),
            n_iterations=config.get("n_iterations", 3),
            n_heads=config.get("n_heads", 16),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Config: latent={config.get('latent_dim')}, K={config.get('n_iterations')}, heads={config.get('n_heads')}")
    else:
        print("  WARNING: No TRM model found — will only collect cosine baseline")

    return docs, all_chunks, embeddings, model


def embed_query(query: str) -> torch.Tensor:
    """Embed a single query using nomic-embed-text."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    emb = model.encode(
        [f"search_query: {query}"],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    emb = emb[:, :EMBED_DIM]
    return F.normalize(emb, p=2, dim=1).cpu()[0]


def embed_queries_batch(queries: list[str]) -> torch.Tensor:
    """Embed multiple queries at once."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    texts = [f"search_query: {q}" for q in queries]
    embs = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embs = embs[:, :EMBED_DIM]
    return F.normalize(embs, p=2, dim=1).cpu()


def select_context(query_emb, embeddings, chunks, model=None, k=TOP_K):
    """Select top-k context via TRM and cosine. Returns both selections."""
    # Cosine selection
    sims = F.cosine_similarity(query_emb.unsqueeze(0), embeddings, dim=-1)
    cosine_topk = sims.topk(k).indices.tolist()

    # TRM selection
    trm_topk = None
    if model is not None:
        with torch.no_grad():
            q = query_emb.unsqueeze(0)  # (1, D)
            c = embeddings.unsqueeze(0)  # (1, N, D)
            scores = model(q, c)  # (1, N)
            trm_topk = scores[0].topk(k).indices.tolist()

    return {
        "cosine": cosine_topk,
        "trm": trm_topk,
        "cosine_chunks": [chunks[i] for i in cosine_topk],
        "trm_chunks": [chunks[i] for i in trm_topk] if trm_topk else None,
    }


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def format_context_set(chunks: list[dict], label: str) -> str:
    """Format chunks for the judge prompt."""
    parts = []
    for i, c in enumerate(chunks):
        path = c.get("path", "unknown")
        sec = c.get("section_title", "")
        header = f"[{i+1}] {path}"
        if sec:
            header += f" § {sec}"
        text = c["text"][:500]
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def judge_context_pair(query, context_a, context_b):
    """Ask frontier model which context set better serves the query."""
    prompt = f"""You are evaluating context quality for a knowledge retrieval system.

QUERY: {query}

Below are two sets of retrieved context documents. Judge which set would better help answer the query. Consider:
- Relevance: Do the documents actually address the query?
- Coverage: Do they cover different aspects of the answer?
- Coherence: Do the documents work together to provide a complete picture?
- Signal-to-noise: Is there irrelevant content that would distract?

CONTEXT SET A:
{context_a}

---

CONTEXT SET B:
{context_b}

---

Which context set better serves the query? Reply with EXACTLY one of:
- WINNER: A
- WINNER: B
- WINNER: TIE

Then explain your reasoning in 2-3 sentences."""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.expanduser("~"),
        )
        response = result.stdout.strip()

        winner = "tie"
        for line in response.split("\n"):
            line = line.strip().upper()
            if "WINNER: A" in line or "WINNER:A" in line:
                winner = "A"
                break
            elif "WINNER: B" in line or "WINNER:B" in line:
                winner = "B"
                break
            elif "WINNER: TIE" in line or "WINNER:TIE" in line:
                winner = "tie"
                break

        return {"winner": winner, "reasoning": response}
    except Exception as e:
        return {"winner": "error", "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Judge data format
# ---------------------------------------------------------------------------

def create_judge_label(query_emb, winner_chunks, loser_chunks, embeddings, chunks):
    """
    Create a training label from a judge verdict.

    Winner chunks get label 1.0, loser chunks get 0.0.
    Pool includes both sets + random fillers to reach CANDIDATE_POOL_SIZE.
    """
    winner_indices = []
    loser_indices = []

    # Find chunk indices
    for wc in winner_chunks:
        for i, c in enumerate(chunks):
            if c["chunk_id"] == wc["chunk_id"]:
                winner_indices.append(i)
                break

    for lc in loser_chunks:
        for i, c in enumerate(chunks):
            if c["chunk_id"] == lc["chunk_id"]:
                loser_indices.append(i)
                break

    # Build pool
    pool_indices = winner_indices + loser_indices
    used = set(pool_indices)

    # Fill remaining with random negatives
    all_indices = list(range(len(chunks)))
    random.shuffle(all_indices)
    for idx in all_indices:
        if len(pool_indices) >= CANDIDATE_POOL_SIZE:
            break
        if idx not in used:
            pool_indices.append(idx)
            used.add(idx)

    pool_indices = pool_indices[:CANDIDATE_POOL_SIZE]

    # Labels: winner=1, loser=0, filler=0
    labels = torch.zeros(len(pool_indices))
    winner_set = set(winner_indices)
    for i, idx in enumerate(pool_indices):
        if idx in winner_set:
            labels[i] = 1.0

    # Embeddings
    cand_embs = embeddings[pool_indices]

    return {
        "query_emb": query_emb,
        "cand_embs": cand_embs,
        "labels": labels,
        "query_text": None,  # filled in by caller
        "winner": None,      # filled in by caller
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect judge training data for TRM")
    parser.add_argument("--workspace", type=str, default=os.path.expanduser("~/cog-workspace"))
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate-only", action="store_true", help="Just generate queries, don't judge")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "judge_data.pt"))
    args = parser.parse_args()

    # Load workspace
    docs, all_chunks, embeddings, trm_model = load_workspace(args.workspace)

    if trm_model is None:
        print("ERROR: No TRM model found. Train one first with: uv run train.py")
        return

    # Extract concepts and generate queries
    print(f"\nExtracting concepts from {len(docs)} documents...")
    concepts, domains = extract_concepts_from_docs(docs)
    print(f"  {len(concepts)} concepts, {len(domains)} domains")

    print(f"\nGenerating {args.n_queries} queries...")
    queries = generate_queries(concepts, domains, n_queries=args.n_queries, seed=args.seed)
    for i, q in enumerate(queries[:5]):
        print(f"  [{i+1}] {q}")
    if len(queries) > 5:
        print(f"  ... and {len(queries) - 5} more")

    if args.generate_only:
        for q in queries:
            print(q)
        return

    # Embed queries
    print(f"\nEmbedding {len(queries)} queries...")
    query_embs = embed_queries_batch(queries)

    # Load existing judge data if any
    existing_labels = []
    if os.path.exists(args.output):
        existing_labels = torch.load(args.output, map_location="cpu", weights_only=False)
        print(f"Loaded {len(existing_labels)} existing judge labels")

    # Run judge loop
    print(f"\n{'='*60}")
    print(f"JUDGE LOOP: {len(queries)} queries")
    print(f"{'='*60}\n")

    new_labels = []
    trm_wins = 0
    cosine_wins = 0
    ties = 0
    errors = 0

    for i, (query, q_emb) in enumerate(zip(queries, query_embs)):
        print(f"\nQuery {i+1}/{len(queries)}: {query[:80]}...")

        # Select context
        selection = select_context(q_emb, embeddings, all_chunks, model=trm_model)

        # Format for judge (randomize A/B to prevent position bias)
        if random.random() < 0.5:
            ctx_a = format_context_set(selection["trm_chunks"], "A")
            ctx_b = format_context_set(selection["cosine_chunks"], "B")
            trm_is = "A"
        else:
            ctx_a = format_context_set(selection["cosine_chunks"], "A")
            ctx_b = format_context_set(selection["trm_chunks"], "B")
            trm_is = "B"

        # Judge
        verdict = judge_context_pair(query, ctx_a, ctx_b)
        print(f"  Judge verdict: {verdict['winner']} (TRM was {trm_is})")

        # Map verdict to TRM win/lose
        if verdict["winner"] == "error":
            errors += 1
            print(f"  ERROR: {verdict['reasoning'][:100]}")
            continue

        if verdict["winner"] == trm_is:
            trm_wins += 1
            winner_chunks = selection["trm_chunks"]
            loser_chunks = selection["cosine_chunks"]
            label_winner = "TRM"
        elif verdict["winner"] == "tie":
            ties += 1
            # For ties, we still create labels — TRM chunks as slightly positive
            winner_chunks = selection["trm_chunks"]
            loser_chunks = selection["cosine_chunks"]
            label_winner = "TIE"
        else:
            cosine_wins += 1
            winner_chunks = selection["cosine_chunks"]
            loser_chunks = selection["trm_chunks"]
            label_winner = "COSINE"

        # Create training label (skip ties for clean signal)
        if verdict["winner"] != "tie":
            label = create_judge_label(q_emb, winner_chunks, loser_chunks, embeddings, all_chunks)
            label["query_text"] = query
            label["winner"] = label_winner
            new_labels.append(label)

        # Running score
        total = trm_wins + cosine_wins + ties
        print(f"  Running: TRM {trm_wins}/{total} ({100*trm_wins/total:.0f}%) | "
              f"Cosine {cosine_wins}/{total} ({100*cosine_wins/total:.0f}%) | "
              f"Ties {ties}/{total}")

    # Save combined labels
    all_labels = existing_labels + new_labels
    torch.save(all_labels, args.output)
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"TRM wins:    {trm_wins}")
    print(f"Cosine wins: {cosine_wins}")
    print(f"Ties:        {ties}")
    print(f"Errors:      {errors}")
    print(f"New labels:  {len(new_labels)}")
    print(f"Total labels: {len(all_labels)} (saved to {args.output})")


if __name__ == "__main__":
    main()
