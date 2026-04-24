"""
Shadow TRM — run alongside live conversations to collect training data.

On every message:
  1. Embed the query
  2. Run TRM to select top-10 context candidates
  3. Run cosine to select top-10 (baseline comparison)
  4. Log both selections with timestamp

During downtime:
  - Compare TRM vs cosine selections against what was ACTUALLY used
  - Generate new judge labels from the comparison
  - Retrain the TRM on enriched data

Usage:
    # Single query (returns TRM's context picks)
    uv run shadow_trm.py "What is the foveated context engine?"

    # Log mode (appends to shadow_log.jsonl)
    uv run shadow_trm.py --log "What is the foveated context engine?"

    # Batch process a conversation transcript
    uv run shadow_trm.py --batch transcript.jsonl
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

from prepare import EMBED_DIM, TOP_K, CACHE_DIR, EMBED_FILE
from train import TRM
from embed_index import load_index


# ---------------------------------------------------------------------------
# Globals (lazy-loaded for speed on repeated calls)
# ---------------------------------------------------------------------------

_model = None
_embeddings = None
_chunks = None
_embed_model = None

SHADOW_LOG = os.path.join(os.path.dirname(__file__), "shadow_log.jsonl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pt")
WORKSPACE = os.path.expanduser("~/cog-workspace")


def _load_all():
    """Lazy-load TRM, embeddings, chunks, and embedding model."""
    global _model, _embeddings, _chunks, _embed_model

    if _model is not None:
        return

    # Load embeddings + chunk metadata from incremental index
    _embeddings, chunk_meta = load_index()
    _chunks = chunk_meta  # list of dicts with path, title, section_title, chunk_id, etc.

    # Load TRM
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        _model = TRM(
            embed_dim=config.get("embed_dim", EMBED_DIM),
            latent_dim=config.get("latent_dim", 256),
            n_iterations=config.get("n_iterations", 3),
            n_heads=config.get("n_heads", 16),
        )
        _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()

    # Load embedding model (for query embedding)
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def embed_query(text: str) -> torch.Tensor:
    """Embed a query string."""
    _load_all()
    emb = _embed_model.encode(
        [f"search_query: {text[:500]}"],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    emb = emb[:, :EMBED_DIM]
    return F.normalize(emb, p=2, dim=1).cpu()[0]


def select_context(query_text: str, k: int = TOP_K) -> dict:
    """
    Run TRM and cosine context selection on a query.

    Returns:
        {
            "query": str,
            "trm_picks": [{"path": str, "title": str, "section": str, "score": float}, ...],
            "cosine_picks": [{"path": str, "title": str, "section": str, "score": float}, ...],
            "overlap": int,  # how many picks are in both sets
            "trm_unique": [...],  # TRM found but cosine didn't
            "cosine_unique": [...],  # cosine found but TRM didn't
        }
    """
    _load_all()
    q_emb = embed_query(query_text)

    # Cosine selection
    sims = F.cosine_similarity(q_emb.unsqueeze(0), _embeddings, dim=-1)
    cosine_topk = sims.topk(k)

    cosine_picks = []
    for idx, score in zip(cosine_topk.indices.tolist(), cosine_topk.values.tolist()):
        c = _chunks[idx]
        cosine_picks.append({
            "path": c.get("path", ""),
            "title": c.get("title", ""),
            "section": c.get("section_title", ""),
            "chunk_id": c.get("chunk_id", ""),
            "score": round(score, 4),
        })

    # TRM selection (two-stage: cosine pre-filter → TRM refinement)
    # Can't run TRM on 23K+ candidates (O(N²) attention). Pre-filter to top 100.
    PRE_FILTER_K = 100
    trm_picks = []
    if _model is not None:
        # Stage 1: cosine pre-filter
        pre_topk = sims.topk(PRE_FILTER_K)
        pre_indices = pre_topk.indices
        pre_embeddings = _embeddings[pre_indices]  # (100, D)

        # Stage 2: TRM refinement on pre-filtered candidates
        with torch.no_grad():
            q = q_emb.unsqueeze(0)  # (1, D)
            c = pre_embeddings.unsqueeze(0)  # (1, 100, D)
            scores = _model(q, c)[0]  # (100,)
            trm_topk = scores.topk(k)

            for local_idx, score in zip(trm_topk.indices.tolist(), trm_topk.values.tolist()):
                idx = pre_indices[local_idx].item()  # map back to global index
                ch = _chunks[idx]
                trm_picks.append({
                    "path": ch.get("path", ""),
                    "title": ch.get("title", ""),
                    "section": ch.get("section_title", ""),
                    "chunk_id": ch.get("chunk_id", ""),
                    "score": round(score, 4),
                })

    # Compute overlap
    trm_ids = {p["chunk_id"] for p in trm_picks}
    cosine_ids = {p["chunk_id"] for p in cosine_picks}
    overlap = len(trm_ids & cosine_ids)

    return {
        "query": query_text[:200],
        "timestamp": datetime.now().isoformat(),
        "trm_picks": trm_picks,
        "cosine_picks": cosine_picks,
        "overlap": overlap,
        "trm_unique": [p for p in trm_picks if p["chunk_id"] not in cosine_ids],
        "cosine_unique": [p for p in cosine_picks if p["chunk_id"] not in trm_ids],
    }


def log_selection(result: dict):
    """Append a selection result to the shadow log."""
    with open(SHADOW_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")


def format_picks(picks: list[dict], label: str) -> str:
    """Pretty-print context selections."""
    lines = [f"\n{label} ({len(picks)} items):"]
    seen_docs = set()
    for i, p in enumerate(picks):
        doc = p["path"]
        sec = p.get("section") or ""
        score = p.get("score", 0)
        marker = " (NEW)" if doc not in seen_docs else ""
        seen_docs.add(doc)
        sec_str = f" § {sec}" if sec else ""
        lines.append(f"  [{i+1}] {doc}{sec_str} ({score:.3f}){marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shadow TRM context selection")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("--log", action="store_true", help="Append results to shadow_log.jsonl")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of context items to select")
    args = parser.parse_args()

    if not args.query:
        print("Usage: uv run shadow_trm.py 'your query here'")
        return

    t0 = time.time()
    result = select_context(args.query, k=args.k)
    dt = time.time() - t0

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Query: {result['query']}")
        print(f"Time: {dt:.1f}s (includes model loading)")
        print(f"Overlap: {result['overlap']}/{args.k} shared picks")
        print(format_picks(result["trm_picks"], "TRM"))
        print(format_picks(result["cosine_picks"], "Cosine"))

        if result["trm_unique"]:
            print(f"\nTRM found (cosine missed):")
            for p in result["trm_unique"]:
                print(f"  - {p['path']} § {p.get('section', '')}")

        if result["cosine_unique"]:
            print(f"\nCosine found (TRM missed):")
            for p in result["cosine_unique"]:
                print(f"  - {p['path']} § {p.get('section', '')}")

    if args.log:
        log_selection(result)
        print(f"\nLogged to {SHADOW_LOG}")


if __name__ == "__main__":
    main()
