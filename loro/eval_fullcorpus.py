#!/usr/bin/env python3
"""
Full-corpus TRM evaluation — honest numbers against 46K chunks.

Instead of scoring 64 pre-filtered candidates (easy mode), this scores
ALL chunks in the embedding index for each eval query. Reports NDCG@10,
@50, @100 for TRM, cosine baseline, and random baseline.

Usage:
    uv run eval_fullcorpus.py
    uv run eval_fullcorpus.py --top-k 10 --max-queries 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = Path(os.path.expanduser("~/.cache/cogos-autoresearch"))
MODEL_PATH = SCRIPT_DIR / "best_model_mamba.pt"
SEQUENCES_PATH = CACHE_DIR / "sequences.pt"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.pt"
CHUNKS_PATH = CACHE_DIR / "embed_index" / "chunks.json"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute NDCG@k for a single query."""
    n = min(k, scores.shape[0])
    _, top_idx = scores.topk(n)
    gains = labels[top_idx]
    discounts = torch.log2(torch.arange(2, n + 2, device=scores.device, dtype=torch.float32))
    dcg = (gains / discounts).sum().item()

    _, ideal_idx = labels.topk(min(k, labels.shape[0]))
    ideal_gains = labels[ideal_idx[:n]]
    idcg = (ideal_gains / discounts[:len(ideal_gains)]).sum().item()

    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Full-corpus TRM evaluation")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of eval queries")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Candidate scoring batch size")
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────
    print("Loading model...")
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from train_mamba import MambaTRM

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = MambaTRM(
        d_model=cfg.get("d_model", 384),
        d_state=cfg.get("d_state", 4),
        d_conv=cfg.get("d_conv", 2),
        n_layers=cfg.get("n_layers", 2),
        expand=cfg.get("expand", 1),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"  Model NDCG (training eval): {ckpt.get('ndcg', 'N/A')}")
    print(f"  Config: {cfg}")

    # ── Load full embedding index ────────────────────────────────────
    print("\nLoading full corpus...")
    all_embeddings = torch.load(EMBEDDINGS_PATH, map_location=DEVICE, weights_only=True)
    n_chunks, dim = all_embeddings.shape
    print(f"  {n_chunks:,} chunks x {dim} dims")

    with open(CHUNKS_PATH) as f:
        chunks_meta = json.load(f)
    print(f"  {len(chunks_meta):,} chunk metadata entries")

    # ── Load eval sequences ──────────────────────────────────────────
    print("\nLoading eval sequences...")
    sequences = torch.load(SEQUENCES_PATH, weights_only=False)
    print(f"  {len(sequences)} total sequences")

    # Use the same val split as training (seed 42, 20%)
    n_val = max(1, len(sequences) // 5)
    torch.manual_seed(42)
    perm = torch.randperm(len(sequences))
    val_indices = perm[:n_val]
    val_data = [sequences[i] for i in val_indices]

    if args.max_queries and args.max_queries < len(val_data):
        val_data = val_data[:args.max_queries]

    print(f"  Eval queries: {len(val_data)}")

    # ── Build ground truth by remapping positive embeddings ─────────
    # The training sequences were built against an older index. The candidate
    # indices are stale but the EMBEDDINGS are still valid. We find each
    # positive's nearest neighbor in the current index to remap ground truth.
    print("\nRemapping ground truth to current index via embedding similarity...")

    eval_queries = []
    skipped = 0
    for seq in val_data:
        labels = seq["labels"]
        stored_candidates = seq["candidates"]  # (64, 384)

        # Find positive candidate embeddings
        pos_embeddings = []
        for local_idx in range(len(labels)):
            if labels[local_idx] > 0:
                pos_embeddings.append(stored_candidates[local_idx])

        if not pos_embeddings:
            skipped += 1
            continue

        # For each positive embedding, find its nearest neighbor in current index
        full_labels = torch.zeros(n_chunks, dtype=torch.float32)
        positive_global_indices = []
        for pos_emb in pos_embeddings:
            # Cosine similarity against full index
            sims = F.cosine_similarity(
                pos_emb.unsqueeze(0).to(all_embeddings.device),
                all_embeddings, dim=-1
            )
            best_idx = int(sims.argmax().item())
            best_sim = sims[best_idx].item()
            # Only accept if similarity is very high (same content, possibly re-embedded)
            if best_sim > 0.85:
                full_labels[best_idx] = 1.0
                positive_global_indices.append(best_idx)

        if not positive_global_indices:
            skipped += 1
            continue

        eval_queries.append({
            "prefix_events": seq["prefix_events"],
            "prefix_types": seq["prefix_types"],
            "prefix_len": seq["prefix_len"],
            "full_labels": full_labels,
            "n_positives": len(positive_global_indices),
            "positive_indices": positive_global_indices,
        })

    print(f"  Valid eval queries: {len(eval_queries)} (skipped {skipped})")
    if not eval_queries:
        print("ERROR: No valid eval queries. Check candidate_indices in sequences.pt")
        return

    avg_positives = np.mean([q["n_positives"] for q in eval_queries])
    print(f"  Average positives per query: {avg_positives:.1f}")
    print(f"  Corpus size: {n_chunks:,} (that's {n_chunks / avg_positives:.0f}x harder than 64-candidate eval)")

    # ── Evaluate: Full-corpus TRM scoring ────────────────────────────
    print(f"\n{'='*60}")
    print("FULL-CORPUS EVALUATION")
    print(f"{'='*60}")

    ks = [10, 50, 100]
    trm_ndcgs = {k: [] for k in ks}
    cosine_ndcgs = {k: [] for k in ks}
    random_ndcgs = {k: [] for k in ks}

    total_time_trm = 0
    total_time_cos = 0

    with torch.no_grad():
        for qi, query in enumerate(eval_queries):
            ev = query["prefix_events"][:query["prefix_len"]].unsqueeze(0).to(DEVICE)
            ty = query["prefix_types"][:query["prefix_len"]].unsqueeze(0).to(DEVICE)
            ln = torch.tensor([query["prefix_len"]], device=DEVICE)
            full_labels = query["full_labels"].to(DEVICE)

            # ── TRM scoring (batch over candidates) ──────────────
            t0 = time.time()

            # Run trajectory through Mamba to get context
            from train_mamba import D_MODEL
            type_emb = model.type_embed(ty)
            x = model.input_proj(torch.cat([ev, type_emb], dim=-1))
            for layer in model.layers:
                x = layer(x)
            x = model.final_norm(x)

            # Pool context
            mask = torch.arange(x.shape[1], device=DEVICE).unsqueeze(0) < ln.unsqueeze(1)
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            max_ctx = x_masked.max(dim=1)[0]
            mean_ctx = (x * mask.unsqueeze(-1).float()).sum(1) / ln.unsqueeze(1).float()
            context = max_ctx + mean_ctx  # (1, D)

            # Score ALL candidates in batches
            all_scores = []
            batch_size = args.batch_size
            for start in range(0, n_chunks, batch_size):
                end = min(start + batch_size, n_chunks)
                cand_batch = all_embeddings[start:end].unsqueeze(0)  # (1, batch, D)

                # Attention probes need candidates
                ctx_for_batch = context.clone()
                for probe, norm in zip(model.attn_probes, model.probe_norms):
                    ctx_for_batch = ctx_for_batch + probe(norm(ctx_for_batch), cand_batch)

                ctx_exp = ctx_for_batch.unsqueeze(1).expand(-1, cand_batch.shape[1], -1)
                combined = torch.cat([ctx_exp, cand_batch], dim=-1)
                batch_scores = model.score_head(combined).squeeze(-1)  # (1, batch)
                all_scores.append(batch_scores)

            trm_scores = torch.cat(all_scores, dim=1).squeeze(0)  # (n_chunks,)
            total_time_trm += time.time() - t0

            # ── Cosine baseline ──────────────────────────────────
            t0 = time.time()
            query_emb = ev[0, -1]  # last event embedding as query
            cosine_scores = F.cosine_similarity(
                query_emb.unsqueeze(0), all_embeddings, dim=-1
            )
            total_time_cos += time.time() - t0

            # ── Random baseline ──────────────────────────────────
            random_scores = torch.randn(n_chunks, device=DEVICE)

            # ── Compute NDCG at each k ───────────────────────────
            for k in ks:
                trm_ndcgs[k].append(ndcg_at_k(trm_scores, full_labels, k))
                cosine_ndcgs[k].append(ndcg_at_k(cosine_scores, full_labels, k))
                random_ndcgs[k].append(ndcg_at_k(random_scores, full_labels, k))

            if (qi + 1) % 5 == 0 or qi == 0:
                print(f"  Query {qi+1}/{len(eval_queries)}: "
                      f"TRM@10={trm_ndcgs[10][-1]:.4f}  "
                      f"cos@10={cosine_ndcgs[10][-1]:.4f}  "
                      f"({total_time_trm/(qi+1)*1000:.0f}ms/query TRM, "
                      f"{total_time_cos/(qi+1)*1000:.0f}ms/query cosine)")

    # ── Report ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS — {len(eval_queries)} queries, {n_chunks:,} candidates")
    print(f"{'='*60}")
    print(f"\n{'Method':<15} {'NDCG@10':>10} {'NDCG@50':>10} {'NDCG@100':>10}")
    print("-" * 50)

    for name, results in [("TRM", trm_ndcgs), ("Cosine", cosine_ndcgs), ("Random", random_ndcgs)]:
        means = {k: np.mean(v) for k, v in results.items()}
        stds = {k: np.std(v) for k, v in results.items()}
        print(f"{name:<15} {means[10]:>9.4f}  {means[50]:>9.4f}  {means[100]:>9.4f}")

    print()
    print(f"TRM latency:    {total_time_trm/len(eval_queries)*1000:.1f}ms/query (full {n_chunks:,} chunks)")
    print(f"Cosine latency: {total_time_cos/len(eval_queries)*1000:.1f}ms/query")

    # Per-query detail
    print(f"\n{'='*60}")
    print("PER-QUERY BREAKDOWN (TRM vs Cosine @ k=10)")
    print(f"{'='*60}")
    print(f"{'Query':<8} {'Positives':>10} {'TRM@10':>10} {'Cos@10':>10} {'Delta':>10}")
    print("-" * 50)
    for i, q in enumerate(eval_queries):
        delta = trm_ndcgs[10][i] - cosine_ndcgs[10][i]
        marker = "  ✓" if delta > 0 else "  ✗" if delta < -0.01 else ""
        print(f"{i+1:<8} {q['n_positives']:>10} {trm_ndcgs[10][i]:>10.4f} "
              f"{cosine_ndcgs[10][i]:>10.4f} {delta:>+10.4f}{marker}")

    trm_wins = sum(1 for i in range(len(eval_queries))
                   if trm_ndcgs[10][i] > cosine_ndcgs[10][i] + 0.001)
    cos_wins = sum(1 for i in range(len(eval_queries))
                   if cosine_ndcgs[10][i] > trm_ndcgs[10][i] + 0.001)
    ties = len(eval_queries) - trm_wins - cos_wins

    print(f"\nTRM wins: {trm_wins}  Cosine wins: {cos_wins}  Ties: {ties}")

    # ── Honest assessment ────────────────────────────────────────────
    trm_mean = np.mean(trm_ndcgs[10])
    cos_mean = np.mean(cosine_ndcgs[10])
    print(f"\n{'='*60}")
    print("HONEST NUMBERS")
    print(f"{'='*60}")
    print(f"Training eval NDCG@10 (64 candidates):   {ckpt.get('ndcg', 0):.4f}")
    print(f"Full-corpus NDCG@10 ({n_chunks:,} candidates): {trm_mean:.4f}")
    print(f"Cosine baseline ({n_chunks:,} candidates):     {cos_mean:.4f}")
    print(f"TRM lift over cosine:                     {(trm_mean - cos_mean)*1000:+.1f} pts")
    if cos_mean > 0:
        print(f"Relative improvement:                     {(trm_mean / cos_mean - 1)*100:+.1f}%")


if __name__ == "__main__":
    main()
