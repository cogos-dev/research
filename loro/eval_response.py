"""
Response-based downstream evaluation: does TRM context produce better RESPONSES?

The context-judge eval (eval_downstream.py) has a structural flaw: the judge
can only see tokens, not document structure. It rewards surface similarity,
which is exactly what cosine already maximizes — making the eval circular.

This script fixes that by evaluating the downstream *effect* of context selection:
  1. For each query, assemble context two ways (TRM top-10 vs cosine top-10)
  2. Feed each context to `claude -p` to generate an actual response
  3. Show the judge ONLY the two responses (never the context)
  4. Judge evaluates: accuracy, completeness, coherence, insight depth

Structural coherence becomes visible through its downstream effects even when
the judge can't directly observe document structure.

Flow:
    query → TRM context → claude -p → response_trm
    query → cosine context → claude -p → response_cosine
    judge(query, response_trm, response_cosine) → verdict

Usage:
    uv run eval_response.py [--queries N] [--resume]
    uv run eval_response.py --resume --checkpoint judge_model.pt
"""

import os
import sys
import json
import random
import subprocess
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from prepare import (
    EMBED_DIM, find_cogdocs, chunk_document,
    CACHE_DIR, EMBED_FILE,
)
from train import TRM
from eval_downstream import (
    TEST_QUERIES,
    embed_queries,
    select_cosine,
    select_trm,
    format_context,
)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

RESPONSE_PROMPT = """\
You are answering a question using the provided context documents from a knowledge workspace.
Use only what is in the context. Be specific and accurate. 3-5 paragraphs.

CONTEXT:
{context}

QUESTION: {query}

Answer:"""

JUDGE_PROMPT = """\
You are comparing two AI responses to the same question. The responses were generated from \
different context selections — you will NOT see the context, only the responses.

QUESTION: {query}

RESPONSE A:
{response_a}

---

RESPONSE B:
{response_b}

---

Which response better answers the question? Evaluate on:
- Accuracy: Does it correctly address the question?
- Completeness: Does it cover the key aspects?
- Coherence: Is it logically consistent and well-structured?
- Insight depth: Does it show genuine understanding vs surface-level matching?

Reply with EXACTLY one of:
- WINNER: A
- WINNER: B
- WINNER: TIE

Then explain your reasoning in 2-3 sentences."""


def generate_response(query: str, context: str, timeout: int = 90) -> str:
    """Generate a response to query given context, using claude -p."""
    prompt = RESPONSE_PROMPT.format(context=context, query=query)
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expanduser("~"),
        )
        response = result.stdout.strip()
        if not response and result.stderr:
            return f"[ERROR: {result.stderr[:200]}]"
        return response if response else "[ERROR: empty response]"
    except subprocess.TimeoutExpired:
        return "[ERROR: timeout]"
    except Exception as e:
        return f"[ERROR: {e}]"


def judge_response_pair(
    query: str,
    response_a: str,
    response_b: str,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Ask judge which response better answers the query.
    Judge sees ONLY responses — never the source context.
    Returns {"winner": "trm"|"cosine"|"tie"|"error", "reasoning": "..."}
    """
    prompt = JUDGE_PROMPT.format(
        query=query,
        response_a=response_a[:3000],  # cap to avoid huge prompts
        response_b=response_b[:3000],
    )
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=60,
            cwd=os.path.expanduser("~"),
        )
        response = result.stdout.strip()

        winner = "tie"
        for line in response.split("\n"):
            line_upper = line.strip().upper()
            if "WINNER: A" in line_upper or "WINNER:A" in line_upper:
                winner = label_a
                break
            elif "WINNER: B" in line_upper or "WINNER:B" in line_upper:
                winner = label_b
                break
            elif "WINNER: TIE" in line_upper or "WINNER:TIE" in line_upper:
                winner = "tie"
                break

        return {"winner": winner, "reasoning": response}
    except Exception as e:
        return {"winner": "error", "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Response-based eval: judge sees responses, not context"
    )
    parser.add_argument("--queries", type=int, default=len(TEST_QUERIES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--workspace", type=str,
                        default=os.path.expanduser("~/cog-workspace"))
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__), "best_model.pt"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries already in eval_response_results.json")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip response generation, only show context comparisons")
    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "eval_response_results.json")

    # Resume: load existing results
    existing_results = []
    done_queries = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        done_queries = {r["query"] for r in existing_results}
        print(f"Resuming: {len(existing_results)} already done, "
              f"{args.queries - len(done_queries)} remaining")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load workspace chunks + embeddings
    print("Loading workspace documents...")
    docs = find_cogdocs(args.workspace)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} docs → {len(all_chunks)} chunks")

    print("Loading embeddings...")
    assert os.path.exists(EMBED_FILE), f"Run prepare.py first: {EMBED_FILE}"
    chunk_embs = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)
    if chunk_embs.size(0) != len(all_chunks):
        n = min(chunk_embs.size(0), len(all_chunks))
        chunk_embs = chunk_embs[:n]
        all_chunks = all_chunks[:n]

    # Load TRM
    print(f"Loading TRM from {os.path.basename(args.checkpoint)}...")
    assert os.path.exists(args.checkpoint), f"No checkpoint at {args.checkpoint}"
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    _v = ckpt.get('val_ndcg'); print(f"  NDCG: {f'{_v:.4f}' if isinstance(_v, (int, float)) else '?'} | "
          f"judge_trained: {ckpt.get('judge_trained', False)}")

    model = TRM(
        embed_dim=cfg["embed_dim"],
        latent_dim=cfg["latent_dim"],
        n_iterations=cfg["n_iterations"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.05),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Embed queries
    queries = TEST_QUERIES[:args.queries]
    print(f"\nEmbedding {len(queries)} queries...")
    query_embs = embed_queries(queries)

    # Evaluation loop
    print(f"\nRunning response-based evaluation (k={args.k})...\n")
    results = []

    for i, (query, q_emb) in enumerate(zip(queries, query_embs)):
        if query in done_queries:
            continue

        cosine_idx = select_cosine(q_emb, chunk_embs, k=args.k)
        trm_idx = select_trm(model, q_emb, chunk_embs, k=args.k,
                              pool_size=args.pool_size, device=device)

        cosine_ctx = format_context(all_chunks, cosine_idx)
        trm_ctx = format_context(all_chunks, trm_idx)
        overlap = len(set(cosine_idx) & set(trm_idx))

        cosine_docs = set(all_chunks[j]["path"] for j in cosine_idx)
        trm_docs = set(all_chunks[j]["path"] for j in trm_idx)

        print(f"{'='*60}")
        print(f"Query {i+1}/{len(queries)}: {query}")
        print(f"  Overlap: {overlap}/{args.k}")

        result = {
            "query": query,
            "overlap": overlap,
            "cosine_docs": list(cosine_docs),
            "trm_docs": list(trm_docs),
            "cosine_indices": cosine_idx,
            "trm_indices": trm_idx,
        }

        if not args.no_generate:
            # Generate responses
            print(f"  Generating TRM response...", end=" ", flush=True)
            trm_response = generate_response(query, trm_ctx)
            trm_ok = not trm_response.startswith("[ERROR")
            print(f"{'ok' if trm_ok else 'FAILED'} ({len(trm_response)} chars)")

            print(f"  Generating cosine response...", end=" ", flush=True)
            cosine_response = generate_response(query, cosine_ctx)
            cosine_ok = not cosine_response.startswith("[ERROR")
            print(f"{'ok' if cosine_ok else 'FAILED'} ({len(cosine_response)} chars)")

            result["trm_response"] = trm_response
            result["cosine_response"] = cosine_response

            if trm_ok and cosine_ok:
                # Randomize A/B
                if random.random() < 0.5:
                    verdict = judge_response_pair(query, trm_response, cosine_response)
                    if verdict["winner"] == "A":
                        verdict["winner"] = "trm"
                    elif verdict["winner"] == "B":
                        verdict["winner"] = "cosine"
                    result["ab_order"] = "trm=A, cosine=B"
                else:
                    verdict = judge_response_pair(query, cosine_response, trm_response)
                    if verdict["winner"] == "A":
                        verdict["winner"] = "cosine"
                    elif verdict["winner"] == "B":
                        verdict["winner"] = "trm"
                    result["ab_order"] = "cosine=A, trm=B"

                result["winner"] = verdict["winner"]
                result["reasoning"] = verdict["reasoning"]
                print(f"  Winner: {verdict['winner'].upper()}")
                print(f"  Reason: {verdict['reasoning'][:200]}")
            else:
                result["winner"] = "error"
                result["reasoning"] = "response generation failed"
                print(f"  Skipping judge (generation failed)")

        results.append(result)
        print()

        # Save incrementally (resume-safe)
        all_results = existing_results + results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (response-based eval)")
    print(f"{'='*60}")

    all_results = existing_results + results
    evaluated = [r for r in all_results if r.get("winner") not in (None, "error")]
    trm_wins = sum(1 for r in evaluated if r.get("winner") == "trm")
    cosine_wins = sum(1 for r in evaluated if r.get("winner") == "cosine")
    ties = sum(1 for r in evaluated if r.get("winner") == "tie")
    errors = sum(1 for r in all_results if r.get("winner") == "error")

    total = len(evaluated)
    if total > 0:
        print(f"TRM wins:    {trm_wins}/{total} ({trm_wins/total*100:.1f}%)")
        print(f"Cosine wins: {cosine_wins}/{total} ({cosine_wins/total*100:.1f}%)")
        print(f"Ties:        {ties}/{total} ({ties/total*100:.1f}%)")
    if errors:
        print(f"Errors:      {errors}")

    avg_overlap = sum(r["overlap"] for r in all_results) / len(all_results) if all_results else 0
    print(f"\nAvg overlap: {avg_overlap:.1f}/{args.k} chunks")
    print(f"Results saved to {output_path} ({len(all_results)} total)")

    # Compare with context-based eval if available
    ctx_eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if os.path.exists(ctx_eval_path):
        with open(ctx_eval_path) as f:
            ctx_results = json.load(f)
        ctx_eval = [r for r in ctx_results if r.get("winner") not in (None, "error")]
        ctx_trm = sum(1 for r in ctx_eval if r.get("winner") == "trm")
        print(f"\nComparison with context-based eval ({len(ctx_eval)} queries):")
        print(f"  Context judge:  TRM {ctx_trm}/{len(ctx_eval)} "
              f"({ctx_trm/len(ctx_eval)*100:.1f}%)")
        if total > 0:
            print(f"  Response judge: TRM {trm_wins}/{total} "
                  f"({trm_wins/total*100:.1f}%)")


if __name__ == "__main__":
    main()
