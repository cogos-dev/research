"""
Convert eval_results.json judge verdicts into TRM training signal.

For each query where the judge preferred one context set:
  - Winner's unique chunks  → positive label (1.0)
  - Shared chunks           → positive label (1.0)
  - Loser's unique chunks   → negative label (0.0)

Output: judge_data.pt — same structure as data.pt, usable by train.py
        as supplementary high-quality training data.

Usage:
    uv run make_judge_labels.py [--weight 10.0] [--out judge_data.pt]

Integration with train.py:
    Pass --judge-data judge_data.pt to train.py (requires the flag in train.py).
    Each judge example is replicated --weight times to compensate for the
    small dataset size vs the ~1600-query main training set.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers to resolve chunk indices for old results (no stored indices)
# ---------------------------------------------------------------------------

def resolve_indices_from_docs(
    result: dict,
    all_chunks: list[dict],
    chunk_embs: torch.Tensor,
    query_emb: torch.Tensor,
    k: int = 10,
    pool_size: int = 64,
    device: str = "cpu",
) -> tuple[list[int], list[int]]:
    """
    Re-derive cosine_indices and trm_indices for results that predate
    index storage. Falls back to doc-path matching (approximate).
    """
    # If the result already has indices, use them
    if "cosine_indices" in result and "trm_indices" in result:
        return result["cosine_indices"], result["trm_indices"]

    # Fall back: match stored doc paths to chunk indices
    cosine_paths = set(result.get("cosine_docs", []))
    trm_paths = set(result.get("trm_docs", []))

    # Get cosine top-k by similarity, then filter to matching doc paths
    sims = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs, dim=-1)
    _, top_indices = sims.topk(min(pool_size * 4, chunk_embs.size(0)))
    top_indices = top_indices.tolist()

    cosine_idx = [i for i in top_indices if all_chunks[i]["path"] in cosine_paths][:k]
    trm_idx = [i for i in top_indices if all_chunks[i]["path"] in trm_paths][:k]

    return cosine_idx, trm_idx


# ---------------------------------------------------------------------------
# Build judge-labeled training examples
# ---------------------------------------------------------------------------

def build_judge_examples(
    eval_results: list[dict],
    all_chunks: list[dict],
    chunk_embs: torch.Tensor,
    query_embs: torch.Tensor,
    test_queries: list[str],
    k: int = 10,
    pool_size: int = 64,
    device: str = "cpu",
) -> list[dict]:
    """
    For each non-tie, non-error verdict, build a training example:
        {
            "query_emb": tensor (embed_dim,),
            "cand_embs": tensor (n_cands, embed_dim),
            "labels":    tensor (n_cands,),   # 1.0 = positive, 0.0 = negative
            "query_text": str,
            "winner": str,
        }
    """
    # Map query text -> embedding
    query_text_to_emb = {q: query_embs[i] for i, q in enumerate(test_queries)}

    examples = []
    skipped = 0

    for result in eval_results:
        winner = result.get("winner", "error")
        if winner in ("tie", "error", ""):
            skipped += 1
            continue

        query_text = result["query"]
        if query_text not in query_text_to_emb:
            print(f"  WARNING: query not in embedding set, skipping: {query_text[:50]}")
            skipped += 1
            continue

        q_emb = query_text_to_emb[query_text]

        cosine_idx, trm_idx = resolve_indices_from_docs(
            result, all_chunks, chunk_embs, q_emb, k=k, pool_size=pool_size, device=device
        )

        if not cosine_idx or not trm_idx:
            print(f"  WARNING: could not resolve indices for: {query_text[:50]}")
            skipped += 1
            continue

        # Determine positive / negative sets
        cosine_set = set(cosine_idx)
        trm_set = set(trm_idx)
        all_candidate_idx = list(cosine_set | trm_set)

        if winner == "trm":
            pos_set = trm_set
            neg_set = cosine_set - trm_set   # cosine-only chunks are noise
        else:  # winner == "cosine"
            pos_set = cosine_set
            neg_set = trm_set - cosine_set   # trm-only chunks are noise

        # Shared chunks are positive regardless of winner
        shared = cosine_set & trm_set
        pos_set = pos_set | shared

        labels = torch.tensor(
            [1.0 if idx in pos_set else 0.0 for idx in all_candidate_idx],
            dtype=torch.float32,
        )
        cand_embs = chunk_embs[all_candidate_idx]  # (n_cands, embed_dim)

        examples.append({
            "query_emb": q_emb.clone(),
            "cand_embs": cand_embs.clone(),
            "labels": labels,
            "query_text": query_text,
            "winner": winner,
            "n_pos": int(labels.sum().item()),
            "n_neg": int((labels == 0).sum().item()),
        })

    print(f"  Built {len(examples)} training examples, skipped {skipped} (ties/errors/missing)")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert judge verdicts to TRM training data")
    parser.add_argument("--eval-results", type=str,
                        default=os.path.join(os.path.dirname(__file__), "eval_results.json"))
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__), "judge_data.pt"))
    parser.add_argument("--weight", type=float, default=10.0,
                        help="Replicate each judge example N times to upweight vs main dataset")
    parser.add_argument("--workspace", type=str,
                        default=os.path.expanduser("~/cog-workspace"))
    args = parser.parse_args()

    # Import workspace infrastructure
    sys.path.insert(0, os.path.dirname(__file__))
    from prepare import (
        EMBED_DIM, find_cogdocs, chunk_document,
        CACHE_DIR, EMBED_FILE,
    )
    from eval_downstream import TEST_QUERIES, embed_queries

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load eval results
    assert os.path.exists(args.eval_results), f"Not found: {args.eval_results}"
    with open(args.eval_results) as f:
        eval_results = json.load(f)

    trm_wins = sum(1 for r in eval_results if r.get("winner") == "trm")
    cosine_wins = sum(1 for r in eval_results if r.get("winner") == "cosine")
    ties = sum(1 for r in eval_results if r.get("winner") == "tie")
    print(f"Eval results: {len(eval_results)} total — "
          f"TRM {trm_wins}, cosine {cosine_wins}, tie {ties}")

    # Load chunks and embeddings
    print("Loading workspace documents...")
    docs = find_cogdocs(args.workspace)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} docs → {len(all_chunks)} chunks")

    print("Loading chunk embeddings...")
    assert os.path.exists(EMBED_FILE), f"Run prepare.py first: {EMBED_FILE}"
    chunk_embs = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)
    if chunk_embs.size(0) != len(all_chunks):
        n = min(chunk_embs.size(0), len(all_chunks))
        chunk_embs = chunk_embs[:n]
        all_chunks = all_chunks[:n]

    # Embed test queries
    print(f"Embedding {len(TEST_QUERIES)} test queries...")
    query_embs = embed_queries(TEST_QUERIES)

    # Build judge examples
    print("Building judge-labeled training examples...")
    examples = build_judge_examples(
        eval_results, all_chunks, chunk_embs, query_embs, TEST_QUERIES,
        device=device,
    )

    if not examples:
        print("No examples built. Check eval_results.json has non-tie verdicts.")
        return

    # Print summary
    print(f"\nExample breakdown:")
    for ex in examples:
        print(f"  [{ex['winner']:>6}] +{ex['n_pos']} pos / -{ex['n_neg']} neg | {ex['query_text'][:60]}")

    # Save as judge_data.pt
    # Format: list of dicts, each with query_emb, cand_embs, labels
    # Replicate by weight factor to compensate for small dataset size
    weight_int = max(1, int(args.weight))
    weighted_examples = examples * weight_int
    print(f"\nReplicating {len(examples)} examples × {weight_int} = {len(weighted_examples)} total")

    torch.save(weighted_examples, args.out)
    print(f"Saved to {args.out}")

    # Stats
    total_pos = sum(ex["n_pos"] for ex in examples)
    total_neg = sum(ex["n_neg"] for ex in examples)
    print(f"\nLabel stats: {total_pos} positives, {total_neg} negatives "
          f"({total_pos / (total_pos + total_neg) * 100:.1f}% positive rate)")
    print(f"\nTo use in training, add to train.py:")
    print(f"  judge_data = torch.load('judge_data.pt')")
    print(f"  # Prepend judge examples to training batches with high weight")


if __name__ == "__main__":
    main()
