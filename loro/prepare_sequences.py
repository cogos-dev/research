"""
Temporal sequence preparation for LoRO-oriented Mamba TRM.

Extracts temporally ordered session sequences from Claude Code / CogOS
session transcripts. Each sequence is a series of events:
  - query (user message)
  - retrieval (Read tool call)
  - search (Grep/Glob)
  - edit (Edit/Write)

Training objective: given events[:t], predict which chunks are retrieved at t+1.
The temporal ordering IS the training signal — the coupling function lives in
trajectory space, not embedding space.

Usage:
    uv run prepare_sequences.py                     # build from all sessions
    uv run prepare_sequences.py --stats             # show stats without building
    uv run prepare_sequences.py --max-sessions 100  # limit sessions processed
"""

import gc
import os
import json
import glob
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from mine_sessions import mine_session as _mine_session_raw


def parse_session(jsonl_path: str) -> list[dict]:
    """Adapt mine_session triples into exchange format for sequence building.

    mine_session returns: [{query, file_path, reasoning_tokens, outcome}, ...]
    We need: [{user_message, reads: [path, ...]}, ...]
    Group consecutive triples with the same query into one exchange.
    """
    triples = _mine_session_raw(jsonl_path)
    if not triples:
        return []

    exchanges = []
    current_query = None
    current_reads = []

    for t in triples:
        query = t.get("query", "")
        fpath = t.get("file_path", "")
        if not query or not fpath:
            continue

        if query != current_query:
            if current_query and current_reads:
                exchanges.append({
                    "user_message": current_query,
                    "reads": current_reads,
                })
            current_query = query
            current_reads = [fpath]
        else:
            current_reads.append(fpath)

    if current_query and current_reads:
        exchanges.append({
            "user_message": current_query,
            "reads": current_reads,
        })

    return exchanges

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 384
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cogos-autoresearch")
SEQUENCE_FILE = os.path.join(CACHE_DIR, "sequences.pt")
MAX_SEQ_LEN = 200       # max events per sequence
MIN_SEQ_LEN = 4         # minimum events to be useful (at least 2 query-retrieval pairs)
CANDIDATE_POOL = 64     # candidates per prediction step
TOP_K = 10              # evaluation metric

# Event types
EVENT_QUERY = 0
EVENT_RETRIEVAL = 1
EVENT_SEARCH = 2
EVENT_EDIT = 3

# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            device="cpu",  # force CPU to avoid MPS OOM
        )
    return _embedder


def embed_texts(texts: list[str], prefix: str = "search_query") -> torch.Tensor:
    """Embed a list of texts using nomic-embed-text, truncated to 384 dims."""
    model = get_embedder()
    prefixed = [f"{prefix}: {t}" for t in texts]
    embs = model.encode(
        prefixed, convert_to_tensor=True, show_progress_bar=False,
        device="cpu", batch_size=16,
    )
    embs = embs[:, :EMBED_DIM]  # Matryoshka truncation
    return F.normalize(embs, dim=-1)


# ---------------------------------------------------------------------------
# Chunk index loading
# ---------------------------------------------------------------------------

def load_chunk_index():
    """Load the 30K chunk embedding index."""
    # Try embed_index subdirectory first, then root
    embed_path = os.path.join(CACHE_DIR, "embed_index", "embeddings.pt")
    chunks_path = os.path.join(CACHE_DIR, "embed_index", "chunks.json")

    if not os.path.exists(embed_path):
        embed_path = os.path.join(CACHE_DIR, "embeddings.pt")

    if not os.path.exists(chunks_path):
        chunks_path = os.path.join(CACHE_DIR, "chunks.json")

    embeddings = torch.load(embed_path, weights_only=True, map_location="cpu")
    with open(chunks_path) as f:
        chunks = json.load(f)

    return embeddings, chunks


def find_chunk_indices(file_path: str, chunks: list[dict]) -> list[int]:
    """Find chunk indices in the index that match a given file path."""
    # Normalize path
    path = file_path.replace(os.path.expanduser("~"), "~")
    home = os.path.expanduser("~")
    path_variants = [
        file_path,
        path,
        file_path.replace(os.path.join(home, "cog-workspace") + "/", ""),
        file_path.replace(os.path.join(home, "cog-workspace", ".cog", "mem") + "/", ""),
    ]

    matches = []
    for i, chunk in enumerate(chunks):
        chunk_path = chunk.get("path", "")
        for variant in path_variants:
            if variant and (variant in chunk_path or chunk_path in variant):
                matches.append(i)
                break

    return matches


# ---------------------------------------------------------------------------
# Session → Sequence conversion
# ---------------------------------------------------------------------------

def collect_queries_from_sessions(
    session_files: list[str],
) -> tuple[dict[str, list[dict]], list[str]]:
    """
    First pass: parse all sessions and collect unique query texts.
    Only keeps sessions that have at least MIN_SEQ_LEN events worth of data.
    Returns (session_map {filepath: exchanges}, unique_query_texts).
    """
    session_map = {}
    query_texts = []
    query_set = set()

    for i, sf in enumerate(session_files):
        if (i + 1) % 1000 == 0:
            print(f"  Parsed {i + 1}/{len(session_files)} session files "
                  f"({len(session_map)} usable, {len(query_texts)} queries)...")
        try:
            exchanges = parse_session(sf)
        except Exception:
            continue

        # Only keep sessions with enough data to be useful
        # Need at least 1 exchange with 2+ reads to build a training sample
        total_reads = sum(len(e.get("reads", [])) for e in exchanges)
        if len(exchanges) < 1 or total_reads < 2:
            continue

        session_map[sf] = exchanges
        for ex in exchanges:
            msg = ex.get("user_message", "")
            if msg and len(msg) >= 10 and msg not in query_set:
                query_set.add(msg)
                query_texts.append(msg)

    return session_map, query_texts


def session_to_sequence(
    exchanges: list[dict],
    chunk_embeddings: torch.Tensor,
    chunks: list[dict],
    query_embedding_map: dict[str, torch.Tensor],
) -> dict | None:
    """
    Convert a parsed session (list of exchanges) into a temporal sequence
    suitable for Mamba TRM training.

    Args:
        exchanges: parsed session exchanges
        chunk_embeddings: (N, 384) chunk embedding tensor
        chunks: chunk metadata list
        query_embedding_map: pre-computed {query_text: embedding} map

    Returns dict with:
      - events: (L, EMBED_DIM) tensor of event embeddings
      - event_types: (L,) tensor of event type IDs
      - targets: list of L lists, each containing chunk indices retrieved at that step
      - metadata: list of dicts with event details
    """
    events = []
    event_types = []
    targets = []
    metadata = []

    for exchange in exchanges:
        user_msg = exchange.get("user_message", "")
        reads = exchange.get("reads", [])

        if not user_msg or len(user_msg) < 10:
            continue

        # Query event — use pre-computed embedding
        query_emb = query_embedding_map.get(user_msg)
        if query_emb is None:
            continue
        events.append(query_emb)
        event_types.append(EVENT_QUERY)
        targets.append([])  # no target at query step
        metadata.append({"type": "query", "text": user_msg[:200]})

        # Retrieval events (each read file is a separate event)
        for read_path in reads:
            chunk_indices = find_chunk_indices(read_path, chunks)
            if not chunk_indices:
                continue

            # Use the first chunk's embedding as the event embedding
            chunk_emb = chunk_embeddings[chunk_indices[0]]
            events.append(chunk_emb)
            event_types.append(EVENT_RETRIEVAL)
            targets.append(chunk_indices)
            metadata.append({
                "type": "retrieval",
                "path": read_path,
                "n_chunks": len(chunk_indices),
            })

    if len(events) < MIN_SEQ_LEN:
        return None

    # Truncate to max length
    if len(events) > MAX_SEQ_LEN:
        events = events[:MAX_SEQ_LEN]
        event_types = event_types[:MAX_SEQ_LEN]
        targets = targets[:MAX_SEQ_LEN]
        metadata = metadata[:MAX_SEQ_LEN]

    return {
        "events": torch.stack(events),           # (L, 384)
        "event_types": torch.tensor(event_types), # (L,)
        "targets": targets,                       # list of lists
        "metadata": metadata,
        "length": len(events),
    }


# ---------------------------------------------------------------------------
# Build training sequences
# ---------------------------------------------------------------------------

def build_next_retrieval_samples(
    sequence: dict,
    chunk_embeddings: torch.Tensor,
    n_chunks: int,
) -> list[dict]:
    """
    From a single sequence, extract (prefix, candidates, labels) samples
    for next-retrieval prediction.

    For each retrieval event at position t, create a sample:
      - prefix: events[:t] (the trajectory so far)
      - candidates: pool of CANDIDATE_POOL chunks (including the actual retrievals)
      - labels: 1.0 for actually-retrieved chunks, 0.0 for others
    """
    samples = []
    events = sequence["events"]
    event_types = sequence["event_types"]
    targets = sequence["targets"]
    L = sequence["length"]

    for t in range(1, L):
        if event_types[t] != EVENT_RETRIEVAL:
            continue
        if not targets[t]:
            continue

        # Prefix: everything before this retrieval
        prefix_events = events[:t]          # (t, 384)
        prefix_types = event_types[:t]      # (t,)

        # Positive indices (what was actually retrieved)
        pos_indices = set(targets[t])

        # Build candidate pool
        pos_list = list(pos_indices)
        n_pos = min(len(pos_list), CANDIDATE_POOL // 4)
        selected_pos = random.sample(pos_list, n_pos) if len(pos_list) > n_pos else pos_list

        # Hard negatives: cosine-similar to query (the event before this retrieval)
        query_emb = events[t - 1] if event_types[t - 1] == EVENT_QUERY else events[t]
        sims = F.cosine_similarity(
            query_emb.unsqueeze(0),
            chunk_embeddings,
            dim=-1,
        )
        # Exclude positives from negative candidates
        for idx in pos_indices:
            sims[idx] = -1.0
        n_hard = CANDIDATE_POOL // 3
        hard_neg_indices = sims.topk(n_hard).indices.tolist()

        # Easy negatives: random
        n_easy = CANDIDATE_POOL - n_pos - len(hard_neg_indices)
        all_indices = set(range(n_chunks))
        available = list(all_indices - pos_indices - set(hard_neg_indices))
        easy_neg_indices = random.sample(available, min(n_easy, len(available)))

        # Assemble candidate pool
        candidate_indices = selected_pos + hard_neg_indices + easy_neg_indices
        random.shuffle(candidate_indices)
        candidate_indices = candidate_indices[:CANDIDATE_POOL]

        candidate_embs = chunk_embeddings[candidate_indices]  # (POOL, 384)
        labels = torch.tensor(
            [1.0 if idx in pos_indices else 0.0 for idx in candidate_indices]
        )

        samples.append({
            "prefix_events": prefix_events,
            "prefix_types": prefix_types,
            "prefix_len": t,
            "candidates": candidate_embs,
            "labels": labels,
            "candidate_indices": candidate_indices,
        })

    return samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ndcg(scores: torch.Tensor, labels: torch.Tensor, k: int = TOP_K) -> float:
    """Compute NDCG@k for a batch of (scores, labels)."""
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        labels = labels.unsqueeze(0)

    B = scores.shape[0]
    ndcgs = []

    for i in range(B):
        # Predicted ranking
        _, pred_order = scores[i].sort(descending=True)
        pred_labels = labels[i][pred_order[:k]]
        dcg = (pred_labels / torch.log2(torch.arange(2, k + 2, dtype=torch.float))).sum()

        # Ideal ranking
        ideal_labels, _ = labels[i].sort(descending=True)
        ideal_labels = ideal_labels[:k]
        idcg = (ideal_labels / torch.log2(torch.arange(2, k + 2, dtype=torch.float))).sum()

        ndcgs.append((dcg / idcg).item() if idcg > 0 else 0.0)

    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_sessions(workspace: str = None) -> list[str]:
    """Find all session transcript files."""
    if workspace is None:
        workspace = os.path.expanduser("~/cog-workspace")

    search_dirs = []
    claude_projects = os.path.expanduser("~/.claude/projects")
    if os.path.exists(claude_projects):
        for d in os.listdir(claude_projects):
            full = os.path.join(claude_projects, d)
            if os.path.isdir(full):
                search_dirs.append(full)

    threads_dir = os.path.join(workspace, ".cog", "mem", "episodic", "threads")
    if os.path.exists(threads_dir):
        search_dirs.append(threads_dir)

    session_files = []
    for sd in search_dirs:
        session_files.extend(glob.glob(os.path.join(sd, "*.jsonl")))
        session_files.extend(glob.glob(os.path.join(sd, "*", "*.jsonl")))
        session_files.extend(glob.glob(os.path.join(sd, "*/subagents/*.jsonl")))

    # Deduplicate
    seen = set()
    filtered = []
    for sf in sorted(session_files):
        real = os.path.realpath(sf)
        if real not in seen:
            seen.add(real)
            filtered.append(sf)

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Build temporal training sequences")
    parser.add_argument("--workspace", default=os.path.expanduser("~/cog-workspace"))
    parser.add_argument("--max-sessions", type=int, default=0, help="Limit sessions (0=all)")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--output", default=SEQUENCE_FILE)
    args = parser.parse_args()

    # Load chunk index
    print("Loading chunk index...")
    chunk_embeddings, chunks = load_chunk_index()
    n_chunks = len(chunks)
    print(f"  {n_chunks} chunks, {chunk_embeddings.shape}")

    # Discover sessions
    session_files = discover_sessions(args.workspace)
    if args.max_sessions > 0:
        session_files = session_files[:args.max_sessions]
    print(f"  {len(session_files)} session files")

    if args.stats:
        # Quick stats: parse a few sessions and show structure
        n_sample = min(20, len(session_files))
        total_exchanges = 0
        total_reads = 0
        for sf in random.sample(session_files, n_sample):
            exchanges = parse_session(sf)
            total_exchanges += len(exchanges)
            total_reads += sum(len(e.get("reads", [])) for e in exchanges)
        print(f"\nSample of {n_sample} sessions:")
        print(f"  Avg exchanges per session: {total_exchanges / n_sample:.1f}")
        print(f"  Avg reads per session: {total_reads / n_sample:.1f}")
        return

    # Pass 1: Parse all sessions and collect unique query texts
    print("\nPass 1: Parsing sessions and collecting queries...")
    session_map, query_texts = collect_queries_from_sessions(session_files)
    print(f"  {len(session_map)} usable sessions (with 2+ exchanges and 2+ reads)")
    print(f"  {len(query_texts)} unique query texts to embed")

    # Batch embed all queries at once (avoids repeated model loads)
    print("\nEmbedding queries...")
    query_embedding_map = {}
    if query_texts:
        EMBED_BATCH = 64
        all_query_embs = []
        for start in range(0, len(query_texts), EMBED_BATCH):
            batch = query_texts[start:start + EMBED_BATCH]
            embs = embed_texts(batch, prefix="search_query")
            all_query_embs.append(embs.cpu())
            if (start + EMBED_BATCH) % 256 == 0:
                print(f"  Embedded {start + len(batch)}/{len(query_texts)}...")
        all_query_embs = torch.cat(all_query_embs, dim=0)
        for text, emb in zip(query_texts, all_query_embs):
            query_embedding_map[text] = emb
        print(f"  Done: {len(query_embedding_map)} query embeddings")

    # Unload embedder to free memory
    global _embedder
    _embedder = None
    gc.collect()

    # Pass 2: Build temporal sequences and training samples
    print("\nPass 2: Building temporal sequences...")
    all_samples = []
    n_sessions_used = 0
    n_sessions_skipped = 0

    for i, (sf, exchanges) in enumerate(session_map.items()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(session_map)} sessions "
                  f"({n_sessions_used} used, {len(all_samples)} samples)...")

        try:
            seq = session_to_sequence(
                exchanges, chunk_embeddings, chunks, query_embedding_map
            )
            if seq is None:
                n_sessions_skipped += 1
                continue

            samples = build_next_retrieval_samples(seq, chunk_embeddings, n_chunks)
            if samples:
                all_samples.extend(samples)
                n_sessions_used += 1
            else:
                n_sessions_skipped += 1
        except Exception as e:
            print(f"  WARNING: Error processing {sf}: {e}")
            n_sessions_skipped += 1
            continue

    print(f"\nResults:")
    print(f"  Sessions used: {n_sessions_used}")
    print(f"  Sessions skipped: {n_sessions_skipped}")
    print(f"  Training samples: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No training samples generated!")
        return

    # Compute prefix length distribution
    prefix_lens = [s["prefix_len"] for s in all_samples]
    print(f"  Prefix lengths: min={min(prefix_lens)}, max={max(prefix_lens)}, "
          f"median={sorted(prefix_lens)[len(prefix_lens)//2]}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(all_samples, args.output)
    print(f"\nSaved {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
