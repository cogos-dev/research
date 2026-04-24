"""
Downstream evaluation: does TRM context curation produce better LLM responses
than cosine similarity?

This is the REAL test. NDCG@10 measures ranking quality on synthetic labels.
This script measures whether ranking quality translates to inference quality.

Method:
1. Real workspace queries (questions a user would actually ask)
2. TRM selects top-10 context chunks, cosine selects top-10
3. Both context sets shown to a judge model (blind A/B comparison)
4. Judge picks which context set better serves the query
5. Win rate = the downstream signal

Usage: uv run eval_downstream.py [--queries N] [--model claude]
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

# Import from our existing infrastructure
from prepare import (
    EMBED_DIM, find_cogdocs, chunk_document,
    CACHE_DIR, EMBED_FILE,
)
from train import TRM

# ---------------------------------------------------------------------------
# Test queries — real workspace questions
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Architecture / v3 design
    "What is the foveated context engine and how does it work?",
    "How does the continuous process model differ from a regular daemon?",
    "What are the three autopoietic loops and which ones are closed?",
    "How does identity work as an attentional weighting function?",
    "What is the momentum vector and how is it computed?",

    # Theoretical foundations
    "What is the ontological crystal and what are its axioms?",
    "How does eigenform theory relate to the workspace architecture?",
    "What is the relationship between ln(2) and distinction cost?",
    "How does SRC derive the fine structure constant?",
    "What is the connection between mass and eigenfrequency?",

    # Implementation / practical
    "How does the salience scoring system work in CogOS v2?",
    "What is the TAA tier system and what are the budget allocations?",
    "How does the self-improving context engine collect training data?",
    "What embedding model does CogOS use and at what dimensions?",
    "How does the constellation bus index content across sessions?",

    # Cognitive science / consciousness
    "What is the aperture model of consciousness?",
    "How does temporal closure relate to eigenforms?",
    "What is the relationship between diffusion and subtractive epistemology?",
    "How does the cognitive workspace ground through resistance?",

    # Meta / identity
    "What is the eigenform continuity model for AI identity?",
    "How does the workspace persist identity across sessions?",
    "What is the relationship between Cog and the model that runs it?",
    "How does the mycelium metaphor apply to the workspace?",

    # Cross-domain
    "How do compressed sensing principles apply to context assembly?",
    "What is the connection between Hofstadter's strange loops and eigenforms?",
    "How does the MoTok paper relate to cognitive context engineering?",
    "What does von Foerster mean by tokens for eigenbehavior?",

    # Held-out evaluation set (not used for judge training labels)
    "How does the kernel validate a holographic workspace?",
    "What is the relationship between STARS and the SRC framework?",
    "How does the session handoff protocol preserve continuity?",
    "What does it mean for an agent to operate substrate-based?",
    "How does the eigenform field theory derive particle masses?",
]


# ---------------------------------------------------------------------------
# Embedding queries
# ---------------------------------------------------------------------------

def embed_queries(queries: list[str]) -> torch.Tensor:
    """Embed test queries with nomic-embed-text."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )

    # nomic uses "search_query:" prefix for queries
    texts = [f"search_query: {q}" for q in queries]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings[:, :EMBED_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


# ---------------------------------------------------------------------------
# Context selection methods
# ---------------------------------------------------------------------------

def select_cosine(query_emb: torch.Tensor, chunk_embs: torch.Tensor, k: int = 10) -> list[int]:
    """Select top-K chunks by cosine similarity."""
    sims = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs, dim=-1)
    _, indices = sims.topk(k)
    return indices.tolist()


@torch.no_grad()
def select_trm(
    model: TRM,
    query_emb: torch.Tensor,
    chunk_embs: torch.Tensor,
    k: int = 10,
    pool_size: int = 64,
    device: str = "cpu",
) -> list[int]:
    """Select top-K chunks using TRM with a candidate pool."""
    model.eval()
    n_chunks = chunk_embs.size(0)

    # Pre-filter: get top pool_size by cosine (TRM refines from this pool)
    sims = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs, dim=-1)
    _, pool_indices = sims.topk(min(pool_size, n_chunks))

    pool_embs = chunk_embs[pool_indices]  # (pool_size, embed_dim)

    # Run TRM
    q = query_emb.unsqueeze(0).to(device)        # (1, embed_dim)
    c = pool_embs.unsqueeze(0).to(device)         # (1, pool_size, embed_dim)

    scores = model(q, c).squeeze(0).cpu()          # (pool_size,)

    # Map back to original indices
    _, top_in_pool = scores.topk(k)
    return pool_indices[top_in_pool].tolist()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: list[dict], indices: list[int], max_chars: int = 4000) -> str:
    """Format selected chunks as context text."""
    lines = []
    total = 0
    for i, idx in enumerate(indices, 1):
        chunk = chunks[idx]
        path = chunk["path"]
        title = chunk["title"]
        text = chunk["text"][:500]  # truncate individual chunks
        entry = f"[{i}] {path} — {title}\n{text}\n"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge (LLM-as-judge, blind A/B comparison)
# ---------------------------------------------------------------------------

def judge_context_pair(
    query: str,
    context_a: str,
    context_b: str,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Ask a judge model which context set better serves the query.
    Returns {"winner": "A"|"B"|"tie", "reasoning": "..."}
    """
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
            capture_output=True, text=True, timeout=60,
            cwd=os.path.expanduser("~"),
        )
        response = result.stdout.strip()

        # Parse winner
        winner = "tie"
        for line in response.split("\n"):
            line = line.strip().upper()
            if "WINNER: A" in line or "WINNER:A" in line:
                winner = label_a
                break
            elif "WINNER: B" in line or "WINNER:B" in line:
                winner = label_b
                break
            elif "WINNER: TIE" in line or "WINNER:TIE" in line:
                winner = "tie"
                break

        return {"winner": winner, "reasoning": response}
    except Exception as e:
        return {"winner": "error", "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downstream eval: TRM vs cosine context curation")
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES),
                        help="Number of test queries to evaluate")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of context chunks to select")
    parser.add_argument("--pool-size", type=int, default=64,
                        help="Candidate pool size for TRM pre-filtering")
    parser.add_argument("--workspace", type=str,
                        default=os.path.expanduser("~/cog-workspace"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__), "best_model.pt"))
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judging, just show context comparisons")
    parser.add_argument("--resume", action="store_true",
                        help="Load existing eval_results.json and skip already-evaluated queries")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")

    # Resume: load existing results and skip already-evaluated queries
    existing_results = []
    done_queries = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        done_queries = {r["query"] for r in existing_results}
        print(f"Resuming: {len(existing_results)} already done, "
              f"{len(TEST_QUERIES) - len(done_queries)} remaining")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load chunks and embeddings
    print("Loading workspace documents...")
    docs = find_cogdocs(args.workspace)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} docs → {len(all_chunks)} chunks")

    print("Loading embeddings...")
    assert os.path.exists(EMBED_FILE), f"Embeddings not found at {EMBED_FILE}. Run prepare.py first."
    chunk_embs = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)

    # Verify alignment
    if chunk_embs.size(0) != len(all_chunks):
        print(f"WARNING: embedding count ({chunk_embs.size(0)}) != chunk count ({len(all_chunks)})")
        print("Re-run prepare.py to regenerate. Using min of both.")
        n = min(chunk_embs.size(0), len(all_chunks))
        chunk_embs = chunk_embs[:n]
        all_chunks = all_chunks[:n]

    # Load TRM
    print(f"Loading TRM from {args.checkpoint}...")
    assert os.path.exists(args.checkpoint), f"No checkpoint at {args.checkpoint}. Run train.py first."
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    print(f"  Config: latent={config['latent_dim']}, K={config['n_iterations']}, heads={config['n_heads']}")
    print(f"  Checkpoint NDCG: {ckpt['val_ndcg']:.4f}")

    model = TRM(
        embed_dim=config["embed_dim"],
        latent_dim=config["latent_dim"],
        n_iterations=config["n_iterations"],
        n_heads=config["n_heads"],
        dropout=config.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Embed test queries
    queries = TEST_QUERIES[:args.queries]
    print(f"\nEmbedding {len(queries)} test queries...")
    query_embs = embed_queries(queries)

    # Run both selection methods
    print(f"\nRunning context selection (k={args.k})...\n")
    results = []

    for i, (query, q_emb) in enumerate(zip(queries, query_embs)):
        if query in done_queries:
            continue

        cosine_idx = select_cosine(q_emb, chunk_embs, k=args.k)
        trm_idx = select_trm(model, q_emb, chunk_embs, k=args.k,
                             pool_size=args.pool_size, device=device)

        cosine_ctx = format_context(all_chunks, cosine_idx)
        trm_ctx = format_context(all_chunks, trm_idx)

        # How much overlap?
        overlap = len(set(cosine_idx) & set(trm_idx))

        print(f"{'='*60}")
        print(f"Query {i+1}/{len(queries)}: {query}")
        print(f"  Overlap: {overlap}/{args.k} chunks shared")

        # Show document sources
        cosine_docs = set(all_chunks[j]["path"] for j in cosine_idx)
        trm_docs = set(all_chunks[j]["path"] for j in trm_idx)
        print(f"  Cosine sources: {len(cosine_docs)} unique docs")
        for d in sorted(cosine_docs)[:5]:
            print(f"    - {d}")
        print(f"  TRM sources: {len(trm_docs)} unique docs")
        for d in sorted(trm_docs)[:5]:
            print(f"    - {d}")

        result = {
            "query": query,
            "overlap": overlap,
            "cosine_docs": list(cosine_docs),
            "trm_docs": list(trm_docs),
            "cosine_indices": cosine_idx,
            "trm_indices": trm_idx,
        }

        if not args.no_judge:
            # Randomize A/B assignment to prevent position bias
            if random.random() < 0.5:
                verdict = judge_context_pair(query, trm_ctx, cosine_ctx)
                # Map back: A=TRM, B=cosine
                if verdict["winner"] == "A":
                    verdict["winner"] = "trm"
                elif verdict["winner"] == "B":
                    verdict["winner"] = "cosine"
                result["ab_order"] = "trm=A, cosine=B"
            else:
                verdict = judge_context_pair(query, cosine_ctx, trm_ctx)
                # Map back: A=cosine, B=TRM
                if verdict["winner"] == "A":
                    verdict["winner"] = "cosine"
                elif verdict["winner"] == "B":
                    verdict["winner"] = "trm"
                result["ab_order"] = "cosine=A, trm=B"

            result["winner"] = verdict["winner"]
            result["reasoning"] = verdict["reasoning"]
            print(f"  Winner: {verdict['winner'].upper()}")
            print(f"  Reasoning: {verdict['reasoning'][:200]}")
        else:
            print(f"  (skipping judge)")

        results.append(result)
        print()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if not args.no_judge:
        trm_wins = sum(1 for r in results if r.get("winner") == "trm")
        cosine_wins = sum(1 for r in results if r.get("winner") == "cosine")
        ties = sum(1 for r in results if r.get("winner") == "tie")
        errors = sum(1 for r in results if r.get("winner") == "error")

        total = len(results) - errors
        print(f"TRM wins:    {trm_wins}/{total} ({trm_wins/total*100:.1f}%)" if total > 0 else "")
        print(f"Cosine wins: {cosine_wins}/{total} ({cosine_wins/total*100:.1f}%)" if total > 0 else "")
        print(f"Ties:        {ties}/{total} ({ties/total*100:.1f}%)" if total > 0 else "")
        if errors:
            print(f"Errors:      {errors}")

    avg_overlap = sum(r["overlap"] for r in results) / len(results)
    print(f"\nAvg overlap: {avg_overlap:.1f}/{args.k} chunks")
    print(f"  (lower = TRM is finding different, hopefully better, context)")

    # Save results (merged with any existing resumed results)
    all_results = existing_results + results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path} ({len(all_results)} total)")


if __name__ == "__main__":
    main()
