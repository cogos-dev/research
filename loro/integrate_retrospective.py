"""
Integrate retrospective training data into judge_data.pt.

Takes the hand-curated conversation trajectory data from retrospective_training_data.py
and converts it into the same format used by the judge training pipeline.
"""

import os
import sys
import random

import torch
import torch.nn.functional as F

from prepare import (
    EMBED_DIM, CANDIDATE_POOL_SIZE, TOP_K,
    find_cogdocs, chunk_document,
    CACHE_DIR, EMBED_FILE,
)
from retrospective_training_data import RETROSPECTIVE_DATA


def main():
    workspace = os.path.expanduser("~/cog-workspace")
    output_path = os.path.join(os.path.dirname(__file__), "judge_data.pt")

    # Load workspace
    print("Loading workspace...")
    docs = find_cogdocs(workspace)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  {len(docs)} docs → {len(all_chunks)} chunks")

    # Load embeddings
    print("Loading embeddings...")
    embeddings = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)
    if embeddings.size(0) != len(all_chunks):
        n = min(embeddings.size(0), len(all_chunks))
        print(f"  Trimming to {n} (embeddings: {embeddings.size(0)}, chunks: {len(all_chunks)})")
        embeddings = embeddings[:n]
        all_chunks = all_chunks[:n]

    # Build path → chunk indices mapping
    path_to_chunks = {}
    for i, c in enumerate(all_chunks):
        path = c.get("path", "")
        path_to_chunks.setdefault(path, []).append(i)

    # Load embedding model for queries
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    def embed_query(text):
        emb = model.encode(
            [f"search_query: {text}"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        emb = emb[:, :EMBED_DIM]
        return F.normalize(emb, p=2, dim=1).cpu()[0]

    # Process retrospective data
    print(f"\nProcessing {len(RETROSPECTIVE_DATA)} retrospective examples...")
    new_labels = []

    for i, entry in enumerate(RETROSPECTIVE_DATA):
        query = entry["query"]
        ideal_paths = entry["ideal_context"]

        print(f"\n[{i+1}/{len(RETROSPECTIVE_DATA)}] {query[:70]}...")

        # Find positive chunk indices from ideal context docs
        positive_indices = []
        matched_paths = []
        for doc_path in ideal_paths:
            # Try exact match first
            matching = path_to_chunks.get(doc_path, [])

            # Try partial match (filename)
            if not matching:
                target = doc_path.split("/")[-1]
                for p, indices in path_to_chunks.items():
                    if p.endswith(target):
                        matching = indices
                        break

            # Try broader partial match
            if not matching:
                for p, indices in path_to_chunks.items():
                    if doc_path in p or any(part in p for part in doc_path.split("/") if len(part) > 8):
                        matching = indices
                        break

            if matching:
                positive_indices.extend(matching[:5])  # Cap at 5 chunks per doc
                matched_paths.append(doc_path)
            else:
                print(f"  WARNING: No chunks found for {doc_path}")

        if not positive_indices:
            print(f"  SKIP: No positive chunks found")
            continue

        print(f"  Matched {len(matched_paths)}/{len(ideal_paths)} docs, {len(positive_indices)} positive chunks")

        # Build candidate pool
        positive_set = set(positive_indices)
        all_indices = list(range(len(all_chunks)))
        random.shuffle(all_indices)

        # Hard negatives: cosine-similar but NOT in ideal set
        q_emb = embed_query(query)
        sims = F.cosine_similarity(q_emb.unsqueeze(0), embeddings, dim=-1)
        hard_neg_candidates = sims.topk(min(50, len(all_chunks))).indices.tolist()
        hard_negatives = [idx for idx in hard_neg_candidates if idx not in positive_set][:20]

        # Easy negatives: random
        used = positive_set | set(hard_negatives)
        easy_negatives = [idx for idx in all_indices if idx not in used][:CANDIDATE_POOL_SIZE - len(positive_indices) - len(hard_negatives)]

        pool = list(positive_indices) + hard_negatives + easy_negatives
        pool = pool[:CANDIDATE_POOL_SIZE]
        random.shuffle(pool)

        # Labels
        labels = torch.zeros(len(pool))
        for j, idx in enumerate(pool):
            if idx in positive_set:
                labels[j] = 1.0

        # Candidate embeddings
        cand_embs = embeddings[pool]

        new_labels.append({
            "query_emb": q_emb,
            "cand_embs": cand_embs,
            "labels": labels,
            "query_text": query,
            "winner": "RETROSPECTIVE",
            "reasoning": entry["reasoning"],
        })

        n_pos = int(labels.sum().item())
        n_neg = len(labels) - n_pos
        print(f"  Pool: {n_pos} positive, {n_neg} negative")

    # Load existing judge data and merge
    existing = []
    if os.path.exists(output_path):
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        # Filter out any previous retrospective entries to avoid duplicates
        existing = [e for e in existing if e.get("winner") != "RETROSPECTIVE"]
        print(f"\nExisting judge labels: {len(existing)}")

    combined = existing + new_labels
    torch.save(combined, output_path)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Retrospective examples:  {len(new_labels)}")
    print(f"Existing judge labels:   {len(existing)}")
    print(f"Total labels saved:      {len(combined)}")
    print(f"Output: {output_path}")

    # Breakdown
    retro = sum(1 for e in combined if e.get("winner") == "RETROSPECTIVE")
    trm = sum(1 for e in combined if e.get("winner") == "TRM")
    cosine = sum(1 for e in combined if e.get("winner") == "COSINE")
    other = len(combined) - retro - trm - cosine
    print(f"\nBy source:")
    print(f"  Retrospective (conversation trajectory): {retro}")
    print(f"  TRM wins (judge-verified):               {trm}")
    print(f"  Cosine wins (judge-verified):             {cosine}")
    print(f"  Other:                                    {other}")


if __name__ == "__main__":
    main()
