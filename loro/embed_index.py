"""
Incremental, streaming, memory-safe embedding index for TRM.

Design principles:
  - Only re-embeds files that changed (content hash comparison)
  - Processes one document at a time (never holds all docs in RAM)
  - Stores per-document embeddings on disk, consolidates on demand
  - Covers all of .cog/ plus workspace root docs

Usage:
    # Build/update the index (incremental — only embeds changes)
    uv run embed_index.py

    # Force full rebuild
    uv run embed_index.py --rebuild

    # Show stats without updating
    uv run embed_index.py --stats

    # Use from other scripts:
    from embed_index import load_index
    embeddings, chunks = load_index()
"""

import os
import gc
import json
import glob
import hashlib
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prepare import EMBED_DIM, parse_frontmatter, chunk_document, extract_sections_from_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cogos-autoresearch")
INDEX_DIR = os.path.join(CACHE_DIR, "embed_index")
DOC_EMBEDS_DIR = os.path.join(INDEX_DIR, "docs")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")
CONSOLIDATED_EMBEDS = os.path.join(CACHE_DIR, "embeddings.pt")
CONSOLIDATED_CHUNKS = os.path.join(INDEX_DIR, "chunks.json")

WORKSPACE = os.path.expanduser("~/cog-workspace")

# Embedding batch size — per-document, so safe even at 32
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Document discovery — expanded coverage
# ---------------------------------------------------------------------------

# Directories to exclude entirely
EXCLUDE_DIRS = {
    "node_modules", "__pycache__", ".venv", ".git",
    "ledger",       # too volatile, event logs
    "sessions",     # session transcripts (mined separately)
    "logs",         # runtime logs
    ".state",       # internal state
    "data",         # binary data
    "oci",          # container images
    "training",     # training artifacts
    "internal",     # internal state
    "plugins",      # plugin binaries
    "tools",        # tool binaries
}

# Files to exclude by name
EXCLUDE_FILES = {
    "CHANGELOG.md",
    "BACKLOG.md",
    "uv.lock",
}


def find_all_cogdocs(workspace_root: str) -> list[dict]:
    """
    Find all markdown files in the workspace.
    Broader than prepare.py's find_cogdocs — covers all of .cog/.
    """
    docs = []
    seen = set()

    search_roots = [
        # All of .cog/ (except excluded dirs)
        os.path.join(workspace_root, ".cog"),
        # Root workspace files
        workspace_root,
        # Research
        os.path.join(workspace_root, "research"),
        # Projects
        os.path.join(workspace_root, "projects"),
        # Skills
        os.path.join(workspace_root, "skills"),
        os.path.join(workspace_root, ".claude", "skills"),
    ]

    for root_path in search_roots:
        if not os.path.isdir(root_path):
            continue

        if root_path == workspace_root:
            # Only root-level .md files, no recursion
            patterns = [os.path.join(root_path, "*.md")]
        else:
            patterns = [
                os.path.join(root_path, "**", "*.md"),
                os.path.join(root_path, "**", "*.cog.md"),
            ]

        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                if path in seen:
                    continue

                # Check exclusions
                parts = path.split(os.sep)
                if any(ex in parts for ex in EXCLUDE_DIRS):
                    continue
                basename = os.path.basename(path)
                if basename in EXCLUDE_FILES:
                    continue

                seen.add(path)

                try:
                    text = Path(path).read_text(encoding="utf-8", errors="ignore")
                    if len(text.strip()) < 100:
                        continue

                    fm, body = parse_frontmatter(text)
                    title = basename
                    if fm and "title" in fm:
                        title = fm["title"]
                    else:
                        for line in text.split("\n"):
                            line = line.strip()
                            if line.startswith("# ") and not line.startswith("# ---"):
                                title = line[2:].strip()
                                break

                    sections = None
                    if fm and "sections" in fm and isinstance(fm["sections"], list):
                        sections = fm["sections"]

                    rel_path = os.path.relpath(path, workspace_root)
                    doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:8]

                    docs.append({
                        "path": rel_path,
                        "abs_path": path,
                        "title": title,
                        "text": text,
                        "body": body,
                        "sections": sections,
                        "doc_id": doc_id,
                    })
                except Exception:
                    continue

    return docs


def content_hash(text: str) -> str:
    """SHA-256 of file content, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Embedding — streaming, one doc at a time
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    """Lazy-load the embedding model. Loaded once, reused."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading nomic-embed-text model...")
        _embed_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )
    return _embed_model


def embed_chunks(chunks: list[dict]) -> torch.Tensor:
    """
    Embed a list of chunks. Small batches for memory safety.
    Returns (n_chunks, EMBED_DIM) tensor.
    """
    model = _get_embed_model()

    def make_text(c):
        sec = c.get("section_title")
        if sec:
            return f"search_document: {c['title']} — {sec}. {c['text'][:1000]}"
        return f"search_document: {c['title']}. {c['text'][:1000]}"

    texts = [make_text(c) for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # Matryoshka truncation
    embeddings = embeddings[:, :EMBED_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


def embed_single_doc(doc: dict) -> tuple[torch.Tensor, list[dict]]:
    """
    Chunk and embed a single document.
    Returns (embeddings, chunk_metadata) — embeddings are (n_chunks, EMBED_DIM).
    """
    chunks = chunk_document(doc)
    if not chunks:
        return torch.zeros(0, EMBED_DIM), []

    embeddings = embed_chunks(chunks)

    # Strip text from chunk metadata to save disk (keep path, title, section, ids)
    meta = []
    for c in chunks:
        meta.append({
            "doc_id": c["doc_id"],
            "path": c["path"],
            "title": c["title"],
            "section_title": c.get("section_title"),
            "chunk_idx": c["chunk_idx"],
            "chunk_id": c["chunk_id"],
            "text_preview": c["text"][:200],  # enough for debugging
        })

    return embeddings, meta


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    """Load the index manifest. Returns {rel_path: {hash, n_chunks, mtime}}."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    """Save the index manifest."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def update_index(workspace_root: str = WORKSPACE, force_rebuild: bool = False) -> dict:
    """
    Incrementally update the embedding index.

    Returns stats: {total_docs, new, updated, unchanged, removed, total_chunks}
    """
    os.makedirs(DOC_EMBEDS_DIR, exist_ok=True)

    manifest = {} if force_rebuild else load_manifest()
    docs = find_all_cogdocs(workspace_root)

    # Build lookup of current docs
    current_paths = {d["path"]: d for d in docs}

    stats = {"total_docs": len(docs), "new": 0, "updated": 0, "unchanged": 0, "removed": 0}

    # Remove entries for deleted files
    removed_paths = [p for p in manifest if p not in current_paths]
    for p in removed_paths:
        entry = manifest.pop(p)
        doc_file = os.path.join(DOC_EMBEDS_DIR, f"{entry['doc_id']}.pt")
        if os.path.exists(doc_file):
            os.remove(doc_file)
        stats["removed"] += 1

    if removed_paths:
        print(f"  Removed {len(removed_paths)} deleted documents")

    # Process each document
    model_loaded = False
    processed = 0

    for doc in docs:
        rel_path = doc["path"]
        new_hash = content_hash(doc["text"])

        # Check if unchanged
        if rel_path in manifest and manifest[rel_path].get("hash") == new_hash:
            stats["unchanged"] += 1
            continue

        # Needs embedding — load model on first use
        if not model_loaded:
            _get_embed_model()
            model_loaded = True

        is_new = rel_path not in manifest
        action = "new" if is_new else "updated"

        embeddings, meta = embed_single_doc(doc)

        if embeddings.size(0) == 0:
            continue

        # Save per-document embeddings
        doc_file = os.path.join(DOC_EMBEDS_DIR, f"{doc['doc_id']}.pt")
        torch.save({
            "embeddings": embeddings,
            "chunks": meta,
            "path": rel_path,
            "hash": new_hash,
        }, doc_file)

        # Update manifest
        manifest[rel_path] = {
            "doc_id": doc["doc_id"],
            "hash": new_hash,
            "n_chunks": embeddings.size(0),
            "mtime": os.path.getmtime(doc["abs_path"]) if os.path.exists(doc["abs_path"]) else 0,
        }

        stats[action] = stats.get(action, 0) + 1
        processed += 1

        if processed % 50 == 0:
            print(f"  Embedded {processed} documents...")
            save_manifest(manifest)  # checkpoint every 50 docs
            gc.collect()

        # Free memory for this doc
        del embeddings, meta
        if "text" in doc:
            del doc["text"]
        if "body" in doc:
            del doc["body"]
        if processed % 100 == 0:
            gc.collect()

    save_manifest(manifest)

    total_chunks = sum(e["n_chunks"] for e in manifest.values())
    stats["total_chunks"] = total_chunks

    print(f"  Index: {len(manifest)} docs, {total_chunks} chunks")
    print(f"  Changes: {stats['new']} new, {stats['updated']} updated, "
          f"{stats['unchanged']} unchanged, {stats['removed']} removed")

    return stats


# ---------------------------------------------------------------------------
# Consolidation — build single embeddings.pt + chunks.json
# ---------------------------------------------------------------------------

def consolidate_index() -> tuple[torch.Tensor, list[dict]]:
    """
    Build consolidated embeddings tensor + chunk metadata from per-doc files.
    Streams through files in batches — low memory overhead.
    Returns (embeddings, chunks).
    """
    manifest = load_manifest()
    if not manifest:
        print("No index found. Run update_index() first.")
        return torch.zeros(0, EMBED_DIM), []

    total_chunks = sum(e["n_chunks"] for e in manifest.values())
    print(f"Consolidating {len(manifest)} docs, {total_chunks} chunks...")

    # Pre-allocate output tensor (~44MB for 30k chunks)
    all_embeddings = torch.zeros(total_chunks, EMBED_DIM)
    all_chunks = []
    offset = 0
    loaded = 0

    for rel_path, entry in manifest.items():
        doc_file = os.path.join(DOC_EMBEDS_DIR, f"{entry['doc_id']}.pt")
        if not os.path.exists(doc_file):
            continue

        data = torch.load(doc_file, map_location="cpu", weights_only=False)
        embs = data["embeddings"]
        n = embs.size(0)

        if offset + n > all_embeddings.size(0):
            extra = torch.zeros(n, EMBED_DIM)
            all_embeddings = torch.cat([all_embeddings, extra])

        all_embeddings[offset:offset + n] = embs
        all_chunks.extend(data["chunks"])
        offset += n
        loaded += 1

        del data, embs
        # GC every 500 docs, not every doc
        if loaded % 500 == 0:
            gc.collect()
            print(f"  Consolidated {loaded}/{len(manifest)} docs ({offset} chunks)...")

    all_embeddings = all_embeddings[:offset]

    # Save consolidated files
    torch.save(all_embeddings, CONSOLIDATED_EMBEDS)
    with open(CONSOLIDATED_CHUNKS, "w") as f:
        json.dump(all_chunks, f)

    print(f"Saved: {CONSOLIDATED_EMBEDS} ({all_embeddings.size(0)} x {EMBED_DIM})")
    return all_embeddings, all_chunks


# ---------------------------------------------------------------------------
# Loading (used by prepare.py, shadow_trm.py, train.py)
# ---------------------------------------------------------------------------

def load_index() -> tuple[torch.Tensor, list[dict]]:
    """
    Load the consolidated embedding index.
    If consolidated file doesn't exist, builds it from per-doc files.
    """
    if os.path.exists(CONSOLIDATED_EMBEDS) and os.path.exists(CONSOLIDATED_CHUNKS):
        embeddings = torch.load(CONSOLIDATED_EMBEDS, map_location="cpu", weights_only=True)
        with open(CONSOLIDATED_CHUNKS) as f:
            chunks = json.load(f)

        # Verify alignment
        if embeddings.size(0) == len(chunks):
            return embeddings, chunks
        print(f"WARNING: embedding/chunk count mismatch ({embeddings.size(0)} vs {len(chunks)}), reconsolidating...")

    return consolidate_index()


def index_stats() -> dict:
    """Return stats about the current index without loading embeddings."""
    manifest = load_manifest()
    total_chunks = sum(e["n_chunks"] for e in manifest.values())

    # Count by directory
    by_dir = {}
    for path in manifest:
        top = path.split(os.sep)[0]
        if top == ".cog" and len(path.split(os.sep)) > 1:
            top = os.sep.join(path.split(os.sep)[:2])
        by_dir[top] = by_dir.get(top, 0) + 1

    consolidated_exists = os.path.exists(CONSOLIDATED_EMBEDS)
    consolidated_size = os.path.getsize(CONSOLIDATED_EMBEDS) if consolidated_exists else 0

    return {
        "total_docs": len(manifest),
        "total_chunks": total_chunks,
        "by_directory": dict(sorted(by_dir.items(), key=lambda x: -x[1])),
        "consolidated": consolidated_exists,
        "consolidated_size_mb": consolidated_size / 1024 / 1024,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incremental embedding index for TRM")
    parser.add_argument("--workspace", type=str, default=WORKSPACE)
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--no-consolidate", action="store_true", help="Skip consolidation step")
    args = parser.parse_args()

    if args.stats:
        stats = index_stats()
        print(f"Embedding Index Stats:")
        print(f"  Documents: {stats['total_docs']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Consolidated: {stats['consolidated']} ({stats['consolidated_size_mb']:.1f} MB)")
        print(f"  By directory:")
        for d, n in stats["by_directory"].items():
            print(f"    {d}: {n}")
    else:
        t0 = time.time()
        print(f"Updating embedding index (workspace: {args.workspace})...")
        update_stats = update_index(args.workspace, force_rebuild=args.rebuild)

        if not args.no_consolidate and (update_stats["new"] > 0 or update_stats["updated"] > 0 or update_stats["removed"] > 0 or args.rebuild):
            consolidate_index()
        elif not args.no_consolidate and not os.path.exists(CONSOLIDATED_EMBEDS):
            consolidate_index()
        else:
            print("No changes — skipping consolidation.")

        dt = time.time() - t0
        print(f"Done in {dt:.1f}s")
