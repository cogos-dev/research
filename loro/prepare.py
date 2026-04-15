"""
Data preparation and evaluation for TRM context engine autoresearch.
This file is FIXED — the autoresearch agent must NOT modify it.

Generates training data from workspace CogDocs:
  - Embeds all documents with nomic-embed-text
  - Creates query-candidate-label triples from both structural analysis
    and real interaction signals (training-signals store)
  - Provides the fixed evaluation metric

Signal weighting (LRAT-inspired):
  Crystallization > Cascade > Provenance > Correct > Accept > Continue > Last
  Real signals from the training-signals store are weighted by type, with
  stronger behavioral signals receiving higher weight in the training loss.

Usage:
    uv run prepare.py                    # full prep
    uv run prepare.py --workspace /path  # custom workspace
    uv run prepare.py --validate-split   # validate train/val split integrity (CI gate)
"""

import os
import json
import math
import glob
import hashlib
import argparse
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import yaml
import torch
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

EMBED_DIM = 384             # nomic-embed-text Matryoshka truncation
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # Override with --embed-model
CANDIDATE_POOL_SIZE = 64    # candidates per query during training
TOP_K = 10                  # how many to select
TIME_BUDGET = 120          # 2 min: 1162 judge labels, converges by ~5000 steps
EVAL_QUERIES = 200          # number of evaluation queries
MIN_DOCS = 50               # minimum documents needed

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cogos-autoresearch")
DATA_FILE = os.path.join(CACHE_DIR, "data.pt")
DELTA_DATA_FILE = os.path.join(CACHE_DIR, "delta.pt")
LAST_TRAINED_FILE = os.path.join(CACHE_DIR, "last_trained.json")
EMBED_FILE = os.path.join(CACHE_DIR, "embeddings.pt")  # legacy — prefer embed_index
SESSION_SIGNALS_FILE = os.path.join(CACHE_DIR, "session_signals.json")

# ---------------------------------------------------------------------------
# LRAT-inspired signal weights (by outcome/type)
# ---------------------------------------------------------------------------
# Crystallization = read docs -> wrote new CogDoc (strongest retrieval signal)
# Cascade = flow-state sessions with high tool density
# Provenance = session authored a CogDoc
# Correct = hard negatives are valuable
# Accept = user accepted result
# Continue/Last = weakest signals (conversation bookkeeping)

SIGNAL_WEIGHTS = {
    "crystallization": 3.0,
    "cascade": 2.5,
    "provenance": 2.0,
    "correct": 1.5,
    "accept": 1.0,
    "continue": 0.5,
    "last": 0.0,       # Dropped: no causal relationship with retrieval quality
}


# ---------------------------------------------------------------------------
# LRAT-style intensity weighting (Section 5.2.1, Eq. 3)
# ---------------------------------------------------------------------------

def compute_lrat_intensity(durations: torch.Tensor, beta: float = None) -> torch.Tensor:
    """
    LRAT-style intensity weighting using exponential saturation (Eq. 3).

    Maps session duration to a bounded [0, 1] utility score.
    Longer sessions = higher intensity, with diminishing returns.

    Args:
        durations: (N,) session durations in seconds
        beta: half-life parameter. If None, uses median duration.
    Returns:
        (N,) intensity weights, normalized so mean ~ 1.0
    """
    if durations.numel() == 0:
        return torch.ones_like(durations)

    if beta is None:
        # Use median as half-life (LRAT recommendation)
        beta = durations.median().item()
        if beta <= 0:
            beta = 300.0  # 5-minute fallback

    raw = 1.0 - torch.exp(-torch.log(torch.tensor(2.0)) * durations / beta)

    # Normalize so E[w] ~ 1 (LRAT Eq. 3)
    mu = raw.mean()
    if mu > 0:
        return raw / mu
    return torch.ones_like(raw)


def combine_weights(type_weights: torch.Tensor, intensity_weights: torch.Tensor) -> torch.Tensor:
    """Combine discrete signal type weights with continuous LRAT intensity."""
    # Multiplicative combination: strong signal type x long session = strongest
    combined = type_weights * intensity_weights
    # Re-normalize to preserve original weight scale
    if combined.mean() > 0:
        combined = combined * (type_weights.mean() / combined.mean())
    return combined


def filter_weak_positives(signals: list[dict], min_weight: float = 0.3) -> list[dict]:
    """
    LRAT-style positive filtering: remove signals that are likely noise.

    Filters:
    1. Signals with weight below min_weight (very weak signals)
    2. Signals where outcome suggests the retrieval wasn't productive:
       - "pivot" outcomes (user immediately changed direction)
       - "last" signals (already weight=0.0, but belt-and-suspenders)
    3. Signals with very short queries (< 10 chars, likely navigation not search)

    Args:
        signals: list of signal dicts with 'weight', 'outcome', 'query' fields
        min_weight: minimum weight threshold (default 0.3)
    Returns:
        filtered list of signals
    """
    filtered = []
    n_dropped = {"low_weight": 0, "pivot": 0, "short_query": 0}

    for sig in signals:
        weight = sig.get("weight", 0)
        outcome = sig.get("outcome", "")
        query = sig.get("query", "")

        if weight < min_weight:
            n_dropped["low_weight"] += 1
            continue
        if outcome in ("pivot", "last"):
            n_dropped["pivot"] += 1
            continue
        if len(query) < 10:
            n_dropped["short_query"] += 1
            continue

        filtered.append(sig)

    total_dropped = sum(n_dropped.values())
    if total_dropped > 0:
        print(f"  Positive filtering: kept {len(filtered)}/{len(signals)} signals "
              f"(dropped: {n_dropped})")

    return filtered


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO timestamps (including Z suffix) into timezone-aware UTC datetimes."""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_last_trained_state(path: str) -> dict | None:
    """Load last training state JSON if present and valid."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            state = json.load(fh)
        if isinstance(state, dict):
            return state
    except (json.JSONDecodeError, IOError):
        pass
    return None


def filter_signals_since(signals: list[dict], last_timestamp: str | None) -> list[dict]:
    """Return signals with timestamp strictly newer than last_timestamp."""
    last_dt = _parse_iso_timestamp(last_timestamp)
    if last_dt is None:
        return signals

    filtered = []
    for sig in signals:
        sig_dt = _parse_iso_timestamp(sig.get("timestamp"))
        if sig_dt is not None and sig_dt > last_dt:
            filtered.append(sig)
    return filtered

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict | None, str]:
    """Extract YAML frontmatter and body from a markdown document."""
    if not text.startswith("---"):
        return None, text
    try:
        end = text.index("---", 3)
        fm = yaml.safe_load(text[3:end])
        body = text[end + 3:].lstrip("\n")
        return fm, body
    except (ValueError, yaml.YAMLError):
        return None, text


def find_cogdocs(workspace_root: str) -> list[dict]:
    """Find all markdown files in the workspace — mem, ontology, docs, ADRs, identities, root."""
    docs = []
    search_paths = [
        # CogDocs (memory)
        os.path.join(workspace_root, ".cog", "mem", "**", "*.md"),
        os.path.join(workspace_root, ".cog", "mem", "**", "*.cog.md"),
        # Ontology (crystal, SRC, foundations — heaviest nodes)
        os.path.join(workspace_root, ".cog", "ontology", "**", "*.md"),
        os.path.join(workspace_root, ".cog", "ontology", "**", "*.cog.md"),
        # Docs (framework-status, workspace-tools)
        os.path.join(workspace_root, ".cog", "docs", "**", "*.md"),
        # ADRs (architectural decisions)
        os.path.join(workspace_root, ".cog", "adr", "**", "*.md"),
        os.path.join(workspace_root, ".cog", "adr", "**", "*.cog.md"),
        # Agent identities (SOUL.md equivalents)
        os.path.join(workspace_root, ".cog", "bin", "agents", "**", "*.md"),
        # Work items and plans
        os.path.join(workspace_root, ".cog", "work", "**", "*.md"),
        os.path.join(workspace_root, ".cog", "coordination", "**", "*.md"),
        # Archived ADRs
        os.path.join(workspace_root, ".cog", "var", "archive", "**", "*.md"),
        # Root workspace files (CLAUDE.md, AGENTS.md, SOUL.md, etc.)
        os.path.join(workspace_root, "*.md"),
    ]
    # Exclude node_modules and other noise
    exclude_patterns = ["node_modules", "__pycache__", ".venv"]
    seen = set()
    for pattern in search_paths:
        for path in glob.glob(pattern, recursive=True):
            if path in seen:
                continue
            # Skip noise directories
            if any(ex in path for ex in exclude_patterns):
                continue
            seen.add(path)
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) < 100:
                    continue
                # Parse frontmatter for section metadata
                fm, body = parse_frontmatter(text)
                # Extract title from frontmatter, first heading, or filename
                title = os.path.basename(path)
                if fm and "title" in fm:
                    title = fm["title"]
                else:
                    for line in text.split("\n"):
                        line = line.strip()
                        if line.startswith("# ") and not line.startswith("# ---"):
                            title = line[2:].strip()
                            break
                # Extract sections metadata if available
                sections = None
                if fm and "sections" in fm and isinstance(fm["sections"], list):
                    sections = fm["sections"]
                rel_path = os.path.relpath(path, workspace_root)
                docs.append({
                    "path": rel_path,
                    "title": title,
                    "text": text,
                    "body": body,
                    "sections": sections,
                    "doc_id": hashlib.md5(rel_path.encode()).hexdigest()[:8],
                })
            except Exception:
                continue
    return docs


def extract_sections_from_text(body: str, sections_meta: list[dict]) -> list[dict]:
    """
    Extract section content using frontmatter line/size hints.
    Falls back to heading-based splitting if hints are unreliable.
    """
    lines = body.split("\n")
    total_lines = len(lines)
    extracted = []

    for i, sec in enumerate(sections_meta):
        # Handle both dict sections and string sections
        if isinstance(sec, str):
            continue  # Skip string-only entries (no line info)
        if not isinstance(sec, dict):
            continue
        title = sec.get("title", f"section_{i}")
        line_start = sec.get("line")

        if line_start is not None and isinstance(line_start, int):
            # Use line number hint (1-indexed in frontmatter)
            start_idx = max(0, line_start - 1)
            # End is either next section's start or end of file
            if i + 1 < len(sections_meta):
                next_line = sections_meta[i + 1].get("line")
                if next_line is not None and isinstance(next_line, int):
                    end_idx = max(start_idx, next_line - 1)
                else:
                    end_idx = total_lines
            else:
                end_idx = total_lines
            content = "\n".join(lines[start_idx:end_idx]).strip()
        else:
            # No line hint — skip this section
            continue

        if len(content) > 50:
            extracted.append({"title": title, "content": content})

    return extracted


def chunk_document(doc: dict, chunk_size: int = 512, overlap: int = 128) -> list[dict]:
    """
    Split a document into chunks. Section-aware for docs with section metadata,
    word-count-based fallback for docs without.

    Section-aware chunking produces semantically coherent chunks that respect
    document structure. Large sections are sub-chunked by word count.
    """
    chunks = []
    chunk_idx = 0

    if doc.get("sections"):
        # Section-aware chunking
        body = doc.get("body", doc["text"])
        sections = extract_sections_from_text(body, doc["sections"])

        for sec in sections:
            sec_text = sec["content"]
            sec_title = sec["title"]
            sec_words = sec_text.split()

            if len(sec_words) <= chunk_size:
                # Section fits in one chunk — keep it whole
                if len(sec_text.strip()) > 50:
                    chunks.append({
                        "text": sec_text,
                        "doc_id": doc["doc_id"],
                        "path": doc["path"],
                        "title": doc["title"],
                        "section_title": sec_title,
                        "chunk_idx": chunk_idx,
                        "chunk_id": f"{doc['doc_id']}_{chunk_idx}",
                    })
                    chunk_idx += 1
            else:
                # Large section — sub-chunk with overlap
                start = 0
                while start < len(sec_words):
                    end = min(start + chunk_size, len(sec_words))
                    chunk_text = " ".join(sec_words[start:end])
                    if len(chunk_text.strip()) > 50:
                        chunks.append({
                            "text": chunk_text,
                            "doc_id": doc["doc_id"],
                            "path": doc["path"],
                            "title": doc["title"],
                            "section_title": sec_title,
                            "chunk_idx": chunk_idx,
                            "chunk_id": f"{doc['doc_id']}_{chunk_idx}",
                        })
                        chunk_idx += 1
                    start = end - overlap if end < len(sec_words) else len(sec_words)

        # If no sections extracted (bad metadata), fall through to word-count
        if chunks:
            return chunks

    # Fallback: word-count-based chunking
    text = doc["text"]
    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        if len(chunk_text.strip()) > 50:
            chunks.append({
                "text": chunk_text,
                "doc_id": doc["doc_id"],
                "path": doc["path"],
                "title": doc["title"],
                "section_title": None,
                "chunk_idx": chunk_idx,
                "chunk_id": f"{doc['doc_id']}_{chunk_idx}",
            })
            chunk_idx += 1
        start = end - overlap if end < len(words) else len(words)
    return chunks


# ---------------------------------------------------------------------------
# Signal loading (training-signals store)
# ---------------------------------------------------------------------------

def load_signals(workspace_root: str) -> list[dict]:
    """
    Load training signals from the rich signal store (2K+ signals).
    Falls back to the legacy JSONL if the new store doesn't exist.

    Each signal has: query, positives, negatives, outcome, and optionally
    type, density, tool_chain, n_turns, searches, etc.

    Returns list of signal dicts, each augmented with a 'weight' field
    derived from LRAT-inspired signal type weighting.
    """
    autoresearch_dir = os.path.join(workspace_root, "apps", "cogos-v3", "autoresearch")
    signals_dir = os.path.join(autoresearch_dir, "training-signals", "signals")
    legacy_jsonl = os.path.join(autoresearch_dir, "training-signals.jsonl")

    signals = []

    if os.path.isdir(signals_dir):
        # New format: individual JSON files with rich schema
        json_files = glob.glob(os.path.join(signals_dir, "*.json"))
        for jf in json_files:
            try:
                with open(jf, "r") as fh:
                    sig = json.load(fh)
                signals.append(sig)
            except (json.JSONDecodeError, IOError):
                continue
        print(f"Loaded {len(signals)} signals from new store ({signals_dir})")
    elif os.path.isfile(legacy_jsonl):
        # Legacy format: one JSON per line
        with open(legacy_jsonl, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                    signals.append(sig)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(signals)} signals from legacy JSONL ({legacy_jsonl})")
    else:
        print("No signal store found — using synthetic pairs only")
        return []

    # Resolve signal type: use 'type' field if present, else fall back to 'outcome'
    for sig in signals:
        sig_type = sig.get("type") or sig.get("outcome", "continue")
        sig["resolved_type"] = sig_type
        sig["weight"] = SIGNAL_WEIGHTS.get(sig_type, 0.5)

        # Density modifier: high-density signals (lots of tools per turn) get a boost
        density = sig.get("density", 0.0)
        if density > 0.5:
            sig["weight"] *= 1.0 + 0.2 * density  # up to 20% boost

        # n_turns modifier: longer exchanges (more context) slightly boost weight
        n_turns = sig.get("n_turns", 1)
        if n_turns > 10:
            sig["weight"] *= 1.0 + 0.1 * min(n_turns / 50.0, 1.0)  # up to 10% boost

    return signals


def load_session_signals() -> list[dict]:
    """Load training signals mined from Claude Code session transcripts."""
    if not os.path.exists(SESSION_SIGNALS_FILE):
        return []

    try:
        with open(SESSION_SIGNALS_FILE) as f:
            signals = json.load(f)
        print(f"  Loaded {len(signals)} session-mined signals from {SESSION_SIGNALS_FILE}")
        return signals
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: could not load session signals: {e}")
        return []


def print_signal_stats(signals: list[dict]):
    """Print summary statistics about loaded signals."""
    if not signals:
        print("  No signals to summarize")
        return

    type_counts = Counter()
    type_weights = {}
    total_positives = 0
    total_negatives = 0

    for sig in signals:
        rt = sig["resolved_type"]
        type_counts[rt] += 1
        type_weights.setdefault(rt, []).append(sig["weight"])
        total_positives += len(sig.get("positives", []))
        total_negatives += len(sig.get("negatives", []))

    print(f"\n  Signal store summary:")
    print(f"  {'Type':<20s} {'Count':>6s} {'Avg Weight':>10s}")
    print(f"  {'-'*20} {'-'*6} {'-'*10}")
    for t, c in type_counts.most_common():
        avg_w = sum(type_weights[t]) / len(type_weights[t])
        print(f"  {t:<20s} {c:>6d} {avg_w:>10.2f}")
    print(f"  {'-'*20} {'-'*6} {'-'*10}")
    print(f"  {'TOTAL':<20s} {len(signals):>6d}")
    print(f"  Total positive refs: {total_positives}")
    print(f"  Total negative refs: {total_negatives}")


def _build_chunk_path_index(chunks: list[dict]) -> dict:
    """
    Build an index mapping normalized paths to chunk indices for O(1) lookup.

    Returns a dict with:
      - 'exact': {path_str -> [chunk_indices]}  for exact path matches
      - 'basename': {basename -> [chunk_indices]}  for fallback matching
      - 'suffix': {suffix -> [chunk_indices]}  for path-suffix matching
    """
    exact = {}
    basename_idx = {}
    suffix_idx = {}

    for i, chunk in enumerate(chunks):
        path = chunk.get("path", "")
        if not path:
            continue

        # Exact path
        exact.setdefault(path, []).append(i)

        # Basename (e.g. "crystal.cog.md")
        bn = os.path.basename(path)
        basename_idx.setdefault(bn, []).append(i)

        # All path suffixes (for partial matching)
        # e.g. ".cog/mem/semantic/insights/foo.md" -> also index
        # "mem/semantic/insights/foo.md", "semantic/insights/foo.md", etc.
        parts = path.split("/")
        for j in range(1, len(parts)):
            suffix = "/".join(parts[j:])
            suffix_idx.setdefault(suffix, []).append(i)

    return {"exact": exact, "basename": basename_idx, "suffix": suffix_idx}


# Module-level cache so the index is built once per run
_chunk_path_index_cache = None
_chunk_path_index_size = 0


def _get_chunk_path_index(chunks: list[dict]) -> dict:
    """Return cached chunk path index, building it on first call."""
    global _chunk_path_index_cache, _chunk_path_index_size
    if _chunk_path_index_cache is None or _chunk_path_index_size != len(chunks):
        _chunk_path_index_cache = _build_chunk_path_index(chunks)
        _chunk_path_index_size = len(chunks)
    return _chunk_path_index_cache


def _normalize_signal_path(file_path: str, workspace_root: str) -> set[str]:
    """
    Generate all candidate normalized paths from a signal file path.

    Handles:
      - Absolute paths from multiple workspace roots (cog, cogos, cogos-dev, etc.)
      - Worktree paths (.claude/worktrees/<name>/X -> X)
      - .agents/skills/ -> .claude/skills/ rename
      - memory/X -> .cog/mem/ basename search
      - .cog/mem/ prefix stripping
      - /private/tmp/ claude task sandbox paths
    """
    candidates = set()
    candidates.add(file_path)

    # Ensure workspace_root has trailing slash for safe prefix stripping
    ws = workspace_root.rstrip("/") + "/"
    if file_path.startswith(ws):
        candidates.add(file_path[len(ws):])

    # Strip common absolute prefixes (with trailing slash to avoid partial matches)
    abs_prefixes = [
        "/Users/slowbro/cog-workspace/",
        "/Users/slowbro/workspaces/cog/",
        "/Users/slowbro/workspaces/cogos/",
        "/Users/slowbro/workspaces/cogos-dev/cogos/",
        "/Users/slowbro/workspaces/cogos-dev/",
        "/Users/slowbro/claw-workspace/",
        "/Users/slowbro/.claude/projects/-Users-slowbro-workspaces-cog/",
        "/Users/slowbro/.claude/projects/-Users-slowbro-cog-workspace/",
        "/Users/slowbro/.openclaw/workspace/",
    ]
    for prefix in abs_prefixes:
        if file_path.startswith(prefix):
            candidates.add(file_path[len(prefix):])

    # Strip .cog/mem/ prefix (some signals use bare memory-relative paths)
    if ".cog/mem/" in file_path:
        idx = file_path.index(".cog/mem/")
        candidates.add(file_path[idx:])
        candidates.add(file_path[idx + len(".cog/mem/"):])

    # Worktree paths: .claude/worktrees/<name>/X -> X
    wt_match = re.match(r"(?:.*?)\.claude/worktrees/[^/]+/(.*)", file_path)
    if wt_match:
        candidates.add(wt_match.group(1))

    # .agents/skills/ -> .claude/skills/ (directory was renamed)
    if ".agents/skills/" in file_path:
        candidates.add(file_path.replace(".agents/skills/", ".claude/skills/"))
        # Also try with the path stripped to just the skills-relative part
        idx = file_path.index(".agents/skills/")
        candidates.add(".claude/skills/" + file_path[idx + len(".agents/skills/"):])

    # /private/tmp/claude-501/ sandbox paths — try to extract workspace-relative path
    if file_path.startswith("/private/tmp/claude-501/"):
        # Pattern: /private/tmp/claude-501/-Users-slowbro-workspaces-cog/<session>/...
        # The session ID and task dirs are not useful, but sometimes the path
        # contains workspace-relative files. Skip these — they are task outputs.
        pass

    return candidates


def find_chunks_for_path(
    file_path: str, chunks: list[dict], workspace_root: str
) -> list[int]:
    """
    Find chunk indices matching a file path from a signal.
    Handles various path formats: absolute, relative, workspace-relative.

    Uses a pre-built path index for O(1) lookups instead of linear scan.
    """
    idx = _get_chunk_path_index(chunks)
    candidates = _normalize_signal_path(file_path, workspace_root)

    matches = set()

    for variant in candidates:
        if not variant:
            continue
        # Skip very short variants (just a filename) for suffix/exact matching
        # to avoid false positives like "README.md" matching every README
        n_components = variant.count("/") + 1

        # 1. Exact match against chunk paths
        if variant in idx["exact"]:
            matches.update(idx["exact"][variant])

        # 2. Check if variant is a known suffix of a chunk path
        #    (e.g. signal has "semantic/insights/foo.md", chunk has ".cog/mem/semantic/insights/foo.md")
        #    Require at least 2 path components to avoid spurious matches
        if n_components >= 2 and variant in idx["suffix"]:
            matches.update(idx["suffix"][variant])

        # 3. Check if any chunk path is a suffix of variant
        #    (e.g. signal has ".cog/mem/semantic/insights/foo.md", chunk has "semantic/insights/foo.md")
        #    We do this by trying suffixes of the variant against the exact index.
        #    Only try suffixes that retain at least 2 path components.
        if n_components >= 3:
            parts = variant.split("/")
            for j in range(1, len(parts) - 1):  # keep at least 2 components
                suffix = "/".join(parts[j:])
                if suffix in idx["exact"]:
                    matches.update(idx["exact"][suffix])

    # 4. Basename fallback: if no matches yet and basename is unique, use it
    #    Only for .md files to avoid spurious matches on common filenames
    if not matches and file_path.endswith(".md"):
        bn = os.path.basename(file_path)
        # Skip very common basenames that would produce false positives
        if bn not in ("README.md", "SKILL.md", "index.md", "CHANGELOG.md"):
            if bn in idx["basename"]:
                basename_matches = idx["basename"][bn]
                # Only use basename match if the basename maps to a single document
                unique_docs = set()
                for ci in basename_matches:
                    unique_docs.add(chunks[ci].get("doc_id"))
                if len(unique_docs) == 1:
                    matches.update(basename_matches)

    return list(matches)


def generate_signal_pairs(
    signals: list[dict],
    chunks: list[dict],
    embeddings: torch.Tensor,
    workspace_root: str,
    pool_size: int = CANDIDATE_POOL_SIZE,
) -> tuple[dict, int, int]:
    """
    Generate query-candidate pairs from real interaction signals.

    Each signal provides:
      - query text -> embedded as the query
      - positives: file paths that were actually used
      - negatives: file paths that were retrieved but not useful

    Returns:
      (data_dict, n_signal_pairs, n_skipped)
    where data_dict has:
      query_embeddings, candidate_embeddings, labels, weights, n_queries, pool_size, embed_dim
    """
    from sentence_transformers import SentenceTransformer

    n_chunks = len(chunks)
    if n_chunks < MIN_DOCS:
        return None, 0, len(signals)

    rng = random.Random(42)
    all_indices = list(range(n_chunks))

    # Pre-compute similarity for hard negative mining
    # Only compute if we have enough signals to justify the cost
    sim_matrix = None

    # Batch-embed all signal queries
    valid_signals = [s for s in signals
                     if s.get("query") and len(s["query"]) >= 10
                     and (s.get("positives") or s.get("negatives"))]

    if not valid_signals:
        return None, 0, len(signals)

    embed_model = os.environ.get("TRM_EMBED_MODEL", EMBED_MODEL)
    print(f"  Embedding {len(valid_signals)} signal queries with {embed_model}...")
    model = SentenceTransformer(
        embed_model,
        trust_remote_code=True,
    )

    query_texts = [f"search_query: {s['query'][:500]}" for s in valid_signals]
    query_embs = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    query_embs = query_embs[:, :EMBED_DIM]
    query_embs = F.normalize(query_embs, p=2, dim=1).cpu()

    # Free the model
    del model

    queries_out = []
    candidates_out = []
    labels_out = []
    weights_out = []
    session_ids_out = []  # Track session IDs for session-level splits
    session_durations_out = []  # For LRAT-style intensity weighting
    tool_call_counts_out = []   # For LRAT-style intensity weighting
    n_skipped = 0

    for sig_idx, sig in enumerate(valid_signals):
        query_emb = query_embs[sig_idx]  # (EMBED_DIM,)

        # Resolve positive file paths to chunk indices
        pos_chunks = set()
        for fp in sig.get("positives", []):
            matched = find_chunks_for_path(fp, chunks, workspace_root)
            pos_chunks.update(matched)

        # Resolve negative file paths to chunk indices
        neg_chunks = set()
        for fp in sig.get("negatives", []):
            matched = find_chunks_for_path(fp, chunks, workspace_root)
            neg_chunks.update(matched)
        # Remove any overlap (positive takes precedence)
        neg_chunks -= pos_chunks

        # Need at least 1 positive to create a training pair
        if not pos_chunks:
            n_skipped += 1
            continue

        pos_list = list(pos_chunks)
        neg_list = list(neg_chunks)

        # Build candidate pool:
        # 1. All positives (up to pool_size // 3)
        n_pos = min(len(pos_list), pool_size // 3)
        selected_pos = rng.sample(pos_list, n_pos) if len(pos_list) > n_pos else pos_list

        # 2. Explicit negatives from the signal (these are "correct" hard negatives)
        n_explicit_neg = min(len(neg_list), pool_size // 3)
        selected_neg = rng.sample(neg_list, n_explicit_neg) if len(neg_list) > n_explicit_neg else neg_list

        # 3. Hard negatives by cosine similarity to query
        used = set(selected_pos) | set(selected_neg)
        query_sims = F.cosine_similarity(query_emb.unsqueeze(0), embeddings, dim=-1)
        for idx in used:
            query_sims[idx] = -1.0
        n_hard = min(pool_size // 4, n_chunks - len(used))
        hard_neg_indices = query_sims.topk(max(n_hard, 1)).indices.tolist()
        hard_neg_indices = [i for i in hard_neg_indices if i not in used]

        # 4. Easy negatives: random fill
        used.update(hard_neg_indices)
        n_remaining = pool_size - len(selected_pos) - len(selected_neg) - len(hard_neg_indices)
        available = [i for i in all_indices if i not in used]
        easy_neg = rng.sample(available, min(max(n_remaining, 0), len(available))) if n_remaining > 0 and available else []

        # Assemble pool
        pool = list(selected_pos) + list(selected_neg) + hard_neg_indices + easy_neg
        rng.shuffle(pool)
        pool = pool[:pool_size]

        # Pad if needed
        while len(pool) < pool_size:
            extra = rng.randint(0, n_chunks - 1)
            if extra not in set(pool):
                pool.append(extra)

        # Labels: 1.0 for positives, 0.0 for everything else
        pool_labels = [1.0 if idx in pos_chunks else 0.0 for idx in pool]

        queries_out.append(query_emb)
        candidates_out.append(pool)
        labels_out.append(pool_labels)
        weights_out.append(sig["weight"])
        session_ids_out.append(sig.get("session", "unknown"))
        session_durations_out.append(sig.get("session_duration", 0))
        tool_call_counts_out.append(sig.get("tool_calls", 0))

    n_pairs = len(queries_out)
    if n_pairs == 0:
        return None, 0, n_skipped

    print(f"  Generated {n_pairs} signal-based pairs ({n_skipped} signals skipped)")

    query_embeddings = torch.stack(queries_out)  # (N, D)
    cand_idx_tensor = torch.tensor(candidates_out, dtype=torch.long)  # (N, pool_size)
    label_tensor = torch.tensor(labels_out, dtype=torch.float32)  # (N, pool_size)
    weight_tensor = torch.tensor(weights_out, dtype=torch.float32)  # (N,)

    # Gather candidate embeddings
    cand_embeddings = embeddings[cand_idx_tensor.view(-1)].view(
        n_pairs, pool_size, EMBED_DIM
    )

    return {
        "query_embeddings": query_embeddings,
        "candidate_embeddings": cand_embeddings,
        "labels": label_tensor,
        "weights": weight_tensor,
        "session_ids": session_ids_out,
        "session_durations": torch.tensor(session_durations_out, dtype=torch.float32),
        "tool_call_counts": torch.tensor(tool_call_counts_out, dtype=torch.float32),
        "n_queries": n_pairs,
        "pool_size": pool_size,
        "embed_dim": EMBED_DIM,
    }, n_pairs, n_skipped


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def compute_embeddings(chunks: list[dict], batch_size: int = 32) -> torch.Tensor:
    """Compute embeddings for all chunks using the configured model."""
    from sentence_transformers import SentenceTransformer

    embed_model = os.environ.get("TRM_EMBED_MODEL", EMBED_MODEL)
    print(f"Loading embedding model: {embed_model}")
    model = SentenceTransformer(
        embed_model,
        trust_remote_code=True,
    )

    def make_embed_text(c):
        sec = c.get("section_title")
        if sec:
            return f"search_document: {c['title']} — {sec}. {c['text'][:1000]}"
        return f"search_document: {c['title']}. {c['text'][:1000]}"

    texts = [make_embed_text(c) for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # Matryoshka truncation to EMBED_DIM
    embeddings = embeddings[:, :EMBED_DIM]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_query_candidate_pairs(
    chunks: list[dict],
    embeddings: torch.Tensor,
    n_queries: int = 2000,
    pool_size: int = CANDIDATE_POOL_SIZE,
    n_positive: int = 5,
) -> dict:
    """
    Generate query-candidate pairs with relevance labels.

    Designed to be HARD for cosine similarity:
    - Positives: same-document chunks ONLY (structural relevance)
    - Hard negatives: cosine-similar chunks from OTHER documents (surface similar, structurally irrelevant)
    - Easy negatives: random chunks (to ensure basic discrimination)

    The TRM must learn document-level coherence beyond surface similarity.
    Cosine similarity gets deceived by hard negatives; the TRM should not.
    """
    n_chunks = len(chunks)
    assert n_chunks >= MIN_DOCS, f"Need at least {MIN_DOCS} chunks, got {n_chunks}"

    # Pre-compute doc_id -> chunk indices mapping
    doc_to_indices = {}
    for i, c in enumerate(chunks):
        doc_to_indices.setdefault(c["doc_id"], []).append(i)

    # Pre-compute full similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = embeddings @ embeddings.T

    queries = []
    candidate_indices = []
    labels = []

    rng = random.Random(42)
    all_indices = list(range(n_chunks))

    n_hard_neg = pool_size // 3       # ~21 hard negatives (cosine-similar but wrong doc)
    n_easy_neg_target = pool_size // 3 # ~21 easy negatives (random)

    for q_idx in range(n_queries):
        # Pick a random chunk as query
        query_idx = rng.randint(0, n_chunks - 1)
        query_doc_id = chunks[query_idx]["doc_id"]

        # Positives: ONLY same-document chunks (structural relevance)
        same_doc = [i for i in doc_to_indices[query_doc_id] if i != query_idx]

        # Need at least 2 same-doc chunks
        if len(same_doc) < 2:
            continue

        positives = set(same_doc)

        # Hard negatives: top cosine-similar from OTHER documents
        sims = sim_matrix[query_idx].clone()
        for i in doc_to_indices[query_doc_id]:
            sims[i] = -1  # mask same-doc
        hard_neg_indices = sims.topk(min(n_hard_neg, n_chunks)).indices.tolist()
        hard_negatives = set(hard_neg_indices)

        # Easy negatives: random from other documents
        available_easy = [i for i in all_indices
                         if i not in positives and i not in hard_negatives and i != query_idx]
        n_easy = min(n_easy_neg_target, len(available_easy))
        easy_negatives = set(rng.sample(available_easy, n_easy)) if n_easy > 0 else set()

        # Build candidate pool
        all_candidates = list(positives | hard_negatives | easy_negatives)
        rng.shuffle(all_candidates)

        # Truncate to pool_size, keeping all positives
        if len(all_candidates) > pool_size:
            pool_pos = [i for i in all_candidates if i in positives]
            pool_neg = [i for i in all_candidates if i not in positives]
            all_candidates = pool_pos + pool_neg[:pool_size - len(pool_pos)]
            rng.shuffle(all_candidates)
        elif len(all_candidates) < pool_size:
            extra = [i for i in all_indices
                     if i not in set(all_candidates) and i != query_idx]
            extra = rng.sample(extra, min(pool_size - len(all_candidates), len(extra)))
            all_candidates.extend(extra)

        pool = all_candidates[:pool_size]

        # Labels: only same-document = positive
        pool_labels = [1.0 if i in positives else 0.0 for i in pool]

        queries.append(query_idx)
        candidate_indices.append(pool)
        labels.append(pool_labels)

    print(f"Generated {len(queries)} synthetic query-candidate pairs")

    # Convert to tensors
    query_embeddings = embeddings[queries]  # (N, D)
    cand_idx_tensor = torch.tensor(candidate_indices, dtype=torch.long)  # (N, pool_size)
    label_tensor = torch.tensor(labels, dtype=torch.float32)  # (N, pool_size)
    weight_tensor = torch.ones(len(queries), dtype=torch.float32)  # uniform weight for synthetic

    # Gather candidate embeddings: (N, pool_size, D)
    cand_embeddings = embeddings[cand_idx_tensor.view(-1)].view(
        len(queries), pool_size, EMBED_DIM
    )

    return {
        "query_embeddings": query_embeddings,
        "candidate_embeddings": cand_embeddings,
        "labels": label_tensor,
        "weights": weight_tensor,
        "n_queries": len(queries),
        "pool_size": pool_size,
        "embed_dim": EMBED_DIM,
    }


# ---------------------------------------------------------------------------
# Data loading (used by train.py)
# ---------------------------------------------------------------------------

def load_data() -> dict:
    """Load pre-computed training data."""
    assert os.path.exists(DATA_FILE), f"Data not found at {DATA_FILE}. Run prepare.py first."
    return torch.load(DATA_FILE, map_location="cpu", weights_only=True)


def make_dataloader(data: dict, batch_size: int, split: str = "train"):
    """
    Infinite dataloader yielding (query_emb, candidate_embs, labels, epoch) batches.
    If data contains 'weights', also yields sample weights as a 5th element.

    Split strategy:
      - If data has 'session_ids': split by session (80% train sessions, 20% val sessions).
        This prevents same-session leakage between train/val.
      - Otherwise: fall back to 80/20 index split.
    """
    n = data["n_queries"]
    session_ids = data.get("session_ids")

    if session_ids and len(session_ids) == n:
        # Session-level split: group indices by session, then split sessions
        from collections import defaultdict
        session_to_indices = defaultdict(list)
        for i, sid in enumerate(session_ids):
            session_to_indices[sid].append(i)
        sessions = sorted(session_to_indices.keys())
        split_idx = int(len(sessions) * 0.8)
        train_sessions = set(sessions[:split_idx])
        if split == "train":
            indices = [i for sid in sessions[:split_idx] for i in session_to_indices[sid]]
        else:
            indices = [i for sid in sessions[split_idx:] for i in session_to_indices[sid]]
    else:
        split_idx = int(n * 0.8)
        if split == "train":
            indices = list(range(split_idx))
        else:
            indices = list(range(split_idx, n))

    q = data["query_embeddings"]
    c = data["candidate_embeddings"]
    l = data["labels"]
    w = data.get("weights")  # may be None for old data files

    epoch = 1
    while True:
        perm = torch.randperm(len(indices)).tolist() if split == "train" else list(range(len(indices)))
        for i in range(0, len(perm), batch_size):
            batch_idx = [indices[perm[j]] for j in range(i, min(i + batch_size, len(perm)))]
            if w is not None:
                yield q[batch_idx], c[batch_idx], l[batch_idx], epoch, w[batch_idx]
            else:
                yield q[batch_idx], c[batch_idx], l[batch_idx], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Split integrity validation (used as CI gate)
# ---------------------------------------------------------------------------

def validate_split_integrity(data: dict) -> dict:
    """
    Validate that the train/val session-level split has no leakage.

    Checks:
      1. No session_id appears in both train and val index sets
      2. Val set has >= 20 pairs (enough for meaningful NDCG)
      3. Train and val index sets are completely disjoint
      4. Reports session counts for train vs val

    Returns a dict with stats and a leakage_detected flag.
    Works with both old (no session_ids) and new data.pt formats.
    """
    n = data["n_queries"]
    session_ids = data.get("session_ids")

    result = {
        "n_train_sessions": 0,
        "n_val_sessions": 0,
        "n_train_pairs": 0,
        "n_val_pairs": 0,
        "leakage_detected": False,
    }

    if session_ids and len(session_ids) == n:
        # Session-level split (mirrors make_dataloader logic)
        from collections import defaultdict
        session_to_indices = defaultdict(list)
        for i, sid in enumerate(session_ids):
            session_to_indices[sid].append(i)
        sessions = sorted(session_to_indices.keys())
        split_idx = int(len(sessions) * 0.8)

        train_sessions = set(sessions[:split_idx])
        val_sessions = set(sessions[split_idx:])

        train_indices = set()
        for sid in sessions[:split_idx]:
            train_indices.update(session_to_indices[sid])
        val_indices = set()
        for sid in sessions[split_idx:]:
            val_indices.update(session_to_indices[sid])

        result["n_train_sessions"] = len(train_sessions)
        result["n_val_sessions"] = len(val_sessions)
        result["n_train_pairs"] = len(train_indices)
        result["n_val_pairs"] = len(val_indices)

        # Check 1: no session appears in both splits
        session_overlap = train_sessions & val_sessions
        if session_overlap:
            result["leakage_detected"] = True
            print(f"FAIL: {len(session_overlap)} session(s) appear in both train and val splits")

        # Check 4: index disjointness
        index_overlap = train_indices & val_indices
        if index_overlap:
            result["leakage_detected"] = True
            print(f"FAIL: {len(index_overlap)} index(es) appear in both train and val sets")

    else:
        # Old format: no session_ids — fall back to index split
        print("WARNING: data.pt lacks session_ids — skipping session-level checks")
        split_idx = int(n * 0.8)
        train_indices = set(range(split_idx))
        val_indices = set(range(split_idx, n))

        result["n_train_pairs"] = len(train_indices)
        result["n_val_pairs"] = len(val_indices)

        # Check 4: index disjointness (still valid for old format)
        index_overlap = train_indices & val_indices
        if index_overlap:
            result["leakage_detected"] = True
            print(f"FAIL: {len(index_overlap)} index(es) appear in both train and val sets")

    # Check 2: minimum val size
    if result["n_val_pairs"] < 20:
        result["leakage_detected"] = True
        print(f"FAIL: val set has only {result['n_val_pairs']} pairs (need >= 20 for meaningful NDCG)")

    return result


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ndcg(model, data: dict, batch_size: int = 32, device: str = "cpu") -> float:
    """
    NDCG@10 (Normalized Discounted Cumulative Gain at 10).

    Measures how well the TRM's ranking matches the oracle relevance labels.
    Score of 1.0 = perfect ranking. Score of 0.0 = completely wrong.

    Uses session-level split when session_ids are available to prevent
    same-session leakage between train and validation sets.
    """
    model.eval()
    n = data["n_queries"]
    session_ids = data.get("session_ids")

    if session_ids and len(session_ids) == n:
        from collections import defaultdict
        session_to_indices = defaultdict(list)
        for i, sid in enumerate(session_ids):
            session_to_indices[sid].append(i)
        sessions = sorted(session_to_indices.keys())
        split_idx = int(len(sessions) * 0.8)
        val_indices = [i for sid in sessions[split_idx:] for i in session_to_indices[sid]]
    else:
        split_idx = int(n * 0.8)
        val_indices = list(range(split_idx, n))

    val_indices_t = torch.tensor(val_indices, dtype=torch.long)
    q = data["query_embeddings"][val_indices_t].to(device)
    c = data["candidate_embeddings"][val_indices_t].to(device)
    labels = data["labels"][val_indices_t]

    n_val = q.size(0)
    if n_val == 0:
        return 0.0

    all_ndcg = []

    for i in range(0, n_val, batch_size):
        end = min(i + batch_size, n_val)
        q_batch = q[i:end]
        c_batch = c[i:end]
        l_batch = labels[i:end]

        # Get model scores
        scores = model(q_batch, c_batch)  # (B, pool_size)
        scores = scores.cpu()

        # Compute NDCG@10 for each query
        for j in range(scores.size(0)):
            s = scores[j]
            l = l_batch[j]

            # Get top-K by model score
            _, top_indices = s.topk(min(TOP_K, s.size(0)))
            relevances = l[top_indices]

            # DCG
            positions = torch.arange(1, len(relevances) + 1, dtype=torch.float32)
            dcg = (relevances / torch.log2(positions + 1)).sum().item()

            # Ideal DCG (sort labels descending)
            ideal_rel, _ = l.sort(descending=True)
            ideal_rel = ideal_rel[:TOP_K]
            ideal_positions = torch.arange(1, len(ideal_rel) + 1, dtype=torch.float32)
            idcg = (ideal_rel / torch.log2(ideal_positions + 1)).sum().item()

            ndcg = dcg / idcg if idcg > 0 else 0.0
            all_ndcg.append(ndcg)

    return sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0


# ---------------------------------------------------------------------------
# BM25+recency baseline
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer with basic punctuation stripping."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _build_bm25_scorer(corpus_texts: list[str], k1: float = 1.5, b: float = 0.75):
    """
    Build a simple BM25 scoring function from corpus texts.
    Falls back to this when rank_bm25 is not installed.

    Returns a callable: score(query_tokens) -> np.array of scores per document.
    """
    # Tokenize corpus
    corpus_tokens = [_simple_tokenize(t) for t in corpus_texts]
    n_docs = len(corpus_tokens)
    if n_docs == 0:
        return lambda q: np.zeros(0)

    # Average document length
    doc_lens = np.array([len(d) for d in corpus_tokens])
    avgdl = doc_lens.mean() if n_docs > 0 else 1.0

    # Document frequency for each term
    df = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))

    # Term frequency per document (sparse: list of Counters)
    tfs = [Counter(doc) for doc in corpus_tokens]

    def score(query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(n_docs)
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            for j in range(n_docs):
                tf = tfs[j].get(term, 0)
                if tf == 0:
                    continue
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lens[j] / avgdl))
                scores[j] += idf * tf_norm
        return scores

    return score


def compute_bm25_recency_baseline(
    q_val: torch.Tensor,
    c_val: torch.Tensor,
    l_val: torch.Tensor,
    embeddings: torch.Tensor,
    all_chunks: list[dict],
    workspace_root: str,
    bm25_alpha: float = 0.7,
    recency_lambda: float = 0.01,
) -> float | None:
    """
    Compute BM25+recency baseline NDCG@10 on the validation set.

    Strategy:
    1. Reverse-map each candidate embedding back to its source chunk via
       nearest-neighbor lookup against the full embedding matrix.
    2. If chunk text is available, compute real BM25 scores. Otherwise use
       embedding dot product as a TF-IDF proxy.
    3. Compute recency scores from file modification times.
    4. Combine: score = alpha * bm25 + (1-alpha) * recency

    Returns NDCG@10 or None if reverse mapping fails for too many candidates.
    """
    import time as _time

    print(f"\nBaseline: BM25+recency proxy ranking...")
    n_chunks = embeddings.size(0)
    n_val = q_val.size(0)
    pool_size = c_val.size(1)

    # --- Step 1: Reverse-map candidate embeddings to chunk indices ---
    # Each candidate embedding was gathered from `embeddings[idx]`, so we can
    # find the original chunk by cosine similarity (exact match for gathered vecs).
    # To avoid O(n_val * pool_size * n_chunks) cost, batch the lookups.

    # Flatten all candidate embeddings: (n_val * pool_size, D)
    c_flat = c_val.view(-1, c_val.size(-1))

    # Batch cosine similarity against all chunk embeddings
    # This is O(n_val * pool_size * n_chunks) but with batched matmul it's fast
    # Process in batches to avoid OOM
    batch_sz = 512
    chunk_indices = torch.zeros(c_flat.size(0), dtype=torch.long)
    n_exact_matches = 0

    for start in range(0, c_flat.size(0), batch_sz):
        end = min(start + batch_sz, c_flat.size(0))
        # (batch, D) @ (D, n_chunks) -> (batch, n_chunks)
        sims = c_flat[start:end] @ embeddings.t()
        best_sim, best_idx = sims.max(dim=1)
        chunk_indices[start:end] = best_idx
        # Count exact matches (cosine > 0.9999 means it was gathered, not interpolated)
        n_exact_matches += (best_sim > 0.9999).sum().item()

    match_rate = n_exact_matches / c_flat.size(0) if c_flat.size(0) > 0 else 0
    print(f"  Candidate->chunk mapping: {match_rate:.1%} exact matches ({n_exact_matches}/{c_flat.size(0)})")

    if match_rate < 0.5:
        print(f"  WARNING: Low match rate — BM25+recency scores may be unreliable")

    chunk_indices = chunk_indices.view(n_val, pool_size)

    # --- Step 2: Gather chunk texts and compute BM25 ---
    # Check if chunks have usable text
    has_text = any(len(c.get("text", "")) > 20 for c in all_chunks[:10])

    if has_text:
        # Try to use rank_bm25 for proper BM25 scoring
        try:
            from rank_bm25 import BM25Okapi
            _use_rank_bm25 = True
        except ImportError:
            _use_rank_bm25 = False

        print(f"  Using {'rank_bm25' if _use_rank_bm25 else 'built-in'} BM25 with chunk text")
    else:
        _use_rank_bm25 = False
        print(f"  No chunk text available — using dot-product as BM25 proxy")

    # --- Step 3: Compute per-chunk recency scores ---
    # Build recency scores for all chunks based on file mtime
    now = _time.time()
    chunk_recency = np.zeros(n_chunks, dtype=np.float32)
    n_recency_valid = 0

    for idx, chunk in enumerate(all_chunks):
        chunk_path = chunk.get("path", "")
        if chunk_path:
            abs_path = os.path.join(workspace_root, chunk_path)
            try:
                mtime = os.path.getmtime(abs_path)
                age_days = (now - mtime) / 86400.0
                chunk_recency[idx] = math.exp(-recency_lambda * age_days)
                n_recency_valid += 1
            except OSError:
                chunk_recency[idx] = 0.5  # neutral fallback
        else:
            chunk_recency[idx] = 0.5

    print(f"  Recency scores: {n_recency_valid}/{n_chunks} chunks have valid mtime")

    # --- Step 4: Score each validation query ---
    bm25_recency_ndcg = []

    # Pre-tokenize all chunks if we have text (for BM25)
    if has_text:
        all_chunk_texts = [c.get("text", "") for c in all_chunks]

        if _use_rank_bm25:
            tokenized_corpus = [_simple_tokenize(t) for t in all_chunk_texts]
            bm25_model = BM25Okapi(tokenized_corpus)
        else:
            bm25_scorer = _build_bm25_scorer(all_chunk_texts)

    for i in range(n_val):
        pool_chunk_idx = chunk_indices[i]  # (pool_size,) indices into all_chunks

        if has_text:
            # Build a "query" from the query embedding's nearest chunk text
            # (since we don't have the actual query text)
            q_sims = q_val[i] @ embeddings.t()  # (n_chunks,)
            q_nearest = q_sims.argmax().item()
            query_text = all_chunks[q_nearest].get("text", "")
            query_tokens = _simple_tokenize(query_text)

            if _use_rank_bm25:
                # Get BM25 scores for all chunks, then gather for our pool
                all_scores = bm25_model.get_scores(query_tokens)
                bm25_scores = np.array([all_scores[idx] for idx in pool_chunk_idx.tolist()])
            else:
                all_scores = bm25_scorer(query_tokens)
                bm25_scores = np.array([all_scores[idx] for idx in pool_chunk_idx.tolist()])
        else:
            # Dot-product proxy (= cosine for normalized embeddings)
            bm25_scores = (q_val[i].unsqueeze(0) @ c_val[i].t()).squeeze(0).numpy()

        # Normalize BM25 scores to [0, 1]
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.ones_like(bm25_scores) * 0.5

        # Gather recency scores for pool
        recency_scores = np.array([chunk_recency[idx] for idx in pool_chunk_idx.tolist()])

        # Combined score
        combined = bm25_alpha * bm25_norm + (1 - bm25_alpha) * recency_scores

        # NDCG@10 using combined scores
        combined_t = torch.from_numpy(combined).float()
        _, top_idx = combined_t.topk(min(TOP_K, combined_t.size(0)))
        rels = l_val[i][top_idx]
        positions = torch.arange(1, len(rels) + 1, dtype=torch.float32)
        dcg = (rels / torch.log2(positions + 1)).sum().item()
        ideal_rel, _ = l_val[i].sort(descending=True)
        ideal_rel = ideal_rel[:TOP_K]
        ideal_pos = torch.arange(1, len(ideal_rel) + 1, dtype=torch.float32)
        idcg = (ideal_rel / torch.log2(ideal_pos + 1)).sum().item()
        bm25_recency_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(bm25_recency_ndcg) / len(bm25_recency_ndcg) if bm25_recency_ndcg else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for TRM context engine autoresearch")
    parser.add_argument("--workspace", type=str, default=os.path.expanduser("~/cog-workspace"),
                        help="Path to cog-workspace root")
    parser.add_argument("--n-queries", type=int, default=2000,
                        help="Number of synthetic query-candidate pairs to generate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (only used for legacy path)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy monolithic embedding (not recommended)")
    parser.add_argument("--no-signals", action="store_true",
                        help="Skip loading real interaction signals (synthetic only)")
    parser.add_argument("--validate-split", action="store_true",
                        help="Load data.pt, validate train/val split integrity, and exit")
    parser.add_argument("--delta", action="store_true",
                        help="Delta mode: only process signals newer than last training run.")
    args = parser.parse_args()

    # --validate-split: load, validate, print report, exit 0 or 1
    if args.validate_split:
        if not os.path.exists(DATA_FILE):
            print(f"FAIL: data.pt not found at {DATA_FILE}")
            exit(1)
        data = torch.load(DATA_FILE, map_location="cpu", weights_only=True)
        print(f"Validating split integrity for {DATA_FILE}")
        print(f"Total pairs: {data['n_queries']}")
        result = validate_split_integrity(data)
        print(f"\n--- Split Integrity Report ---")
        print(f"  Train sessions:  {result['n_train_sessions']}")
        print(f"  Val sessions:    {result['n_val_sessions']}")
        print(f"  Train pairs:     {result['n_train_pairs']}")
        print(f"  Val pairs:       {result['n_val_pairs']}")
        print(f"  Leakage:         {'YES' if result['leakage_detected'] else 'none'}")
        if result["leakage_detected"]:
            print("\nFAIL: split integrity check failed")
            exit(1)
        else:
            print("\nPASS: split integrity verified")
            exit(0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")

    if args.delta and args.no_signals:
        print("ERROR: --delta cannot be combined with --no-signals")
        exit(1)

    run_mode = "delta" if args.delta else "full"
    output_data_file = DELTA_DATA_FILE if args.delta else DATA_FILE
    last_trained_state = None
    last_trained_timestamp = None
    n_signals_for_run = 0

    if args.legacy:
        # Legacy path: monolithic embedding (high memory usage)
        print(f"\nScanning workspace: {args.workspace}")
        docs = find_cogdocs(args.workspace)
        print(f"Found {len(docs)} documents")

        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc))
        print(f"Created {len(all_chunks)} chunks")

        if len(all_chunks) < MIN_DOCS:
            print(f"ERROR: Need at least {MIN_DOCS} chunks. Found {len(all_chunks)}.")
            exit(1)

        if os.path.exists(EMBED_FILE):
            embeddings = torch.load(EMBED_FILE, map_location="cpu", weights_only=True)
            if embeddings.size(0) != len(all_chunks):
                embeddings = compute_embeddings(all_chunks, batch_size=args.batch_size)
                torch.save(embeddings, EMBED_FILE)
        else:
            embeddings = compute_embeddings(all_chunks, batch_size=args.batch_size)
            torch.save(embeddings, EMBED_FILE)
    else:
        # New path: incremental embedding index (memory-safe)
        from embed_index import update_index, consolidate_index, load_index

        print(f"\nUpdating embedding index (incremental)...")
        update_index(args.workspace)
        embeddings, chunk_meta = load_index()

        # Reconstruct chunks with text for training data generation
        # We need doc_id for same-document matching, which chunk_meta has
        print(f"Loaded {embeddings.size(0)} chunk embeddings from index")

        # Build minimal chunk objects for generate_query_candidate_pairs
        all_chunks = []
        for cm in chunk_meta:
            all_chunks.append({
                "doc_id": cm["doc_id"],
                "path": cm["path"],
                "title": cm["title"],
                "section_title": cm.get("section_title"),
                "chunk_idx": cm["chunk_idx"],
                "chunk_id": cm["chunk_id"],
                "text": cm.get("text_preview", ""),
            })

    if len(all_chunks) < MIN_DOCS:
        print(f"ERROR: Need at least {MIN_DOCS} chunks. Found {len(all_chunks)}.")
        exit(1)

    # -----------------------------------------------------------------------
    # Phase 1: Load real interaction signals
    # -----------------------------------------------------------------------
    signals = []
    signal_data = None
    n_signal_pairs = 0
    n_signal_skipped = 0

    if not args.no_signals:
        print(f"\n{'='*60}")
        print("Phase 1: Loading real interaction signals")
        print(f"{'='*60}")
        existing_signals = load_signals(args.workspace)

        # Load session-mined signals (from mine_sessions.py)
        session_signals = load_session_signals()
        if session_signals:
            signals = existing_signals + session_signals
            print(f"  Combined: {len(existing_signals)} store + {len(session_signals)} session-mined = {len(signals)} total signals")
        else:
            signals = existing_signals

        if args.delta:
            last_trained_state = load_last_trained_state(LAST_TRAINED_FILE)
            if last_trained_state:
                last_trained_timestamp = last_trained_state.get("timestamp")
            signals = filter_signals_since(signals, last_trained_timestamp)
            if not signals:
                print("No new signals since last training")
                exit(0)

        n_signals_for_run = len(signals)
        if signals:
            print_signal_stats(signals)
            signals = filter_weak_positives(signals)

            print(f"\nGenerating signal-based training pairs...")
            signal_result = generate_signal_pairs(
                signals, all_chunks, embeddings, args.workspace
            )
            signal_data, n_signal_pairs, n_signal_skipped = signal_result

    # -----------------------------------------------------------------------
    # Phase 2: Generate synthetic training data
    # -----------------------------------------------------------------------
    if args.delta:
        if signal_data is None or signal_data["n_queries"] == 0:
            print("No new signal-based pairs could be generated from new signals")
            exit(0)
        data = signal_data
        since_timestamp = last_trained_timestamp or "start (no previous training state)"
        print(f"Delta: {len(signals)} new signals since {since_timestamp}, {data['n_queries']} new pairs generated")
    else:
        print(f"\n{'='*60}")
        print("Phase 2: Generating synthetic query-candidate pairs")
        print(f"{'='*60}")
        print(f"Generating {args.n_queries} synthetic pairs...")
        synthetic_data = generate_query_candidate_pairs(
            all_chunks, embeddings, n_queries=args.n_queries
        )

        # -------------------------------------------------------------------
        # Phase 3: Merge signal + synthetic data
        # -------------------------------------------------------------------
        if signal_data is not None and signal_data["n_queries"] > 0:
            print(f"\n{'='*60}")
            print("Phase 3: Merging signal + synthetic training data")
            print(f"{'='*60}")

            data = {
                "query_embeddings": torch.cat([
                    signal_data["query_embeddings"],
                    synthetic_data["query_embeddings"],
                ], dim=0),
                "candidate_embeddings": torch.cat([
                    signal_data["candidate_embeddings"],
                    synthetic_data["candidate_embeddings"],
                ], dim=0),
                "labels": torch.cat([
                    signal_data["labels"],
                    synthetic_data["labels"],
                ], dim=0),
                "weights": torch.cat([
                    signal_data["weights"],
                    synthetic_data["weights"],
                ], dim=0),
                "pool_size": synthetic_data["pool_size"],
                "embed_dim": EMBED_DIM,
            }
            data["n_queries"] = data["query_embeddings"].size(0)

            print(f"  Signal pairs:    {signal_data['n_queries']:>6d} (real interactions)")
            print(f"  Synthetic pairs: {synthetic_data['n_queries']:>6d} (structural)")
            print(f"  Total pairs:     {data['n_queries']:>6d}")
            print(f"  Signals skipped: {n_signal_skipped:>6d} (no matching chunks)")

            # Weight distribution stats
            w = data["weights"]
            print(f"\n  Weight stats:")
            print(f"    Min:    {w.min().item():.2f}")
            print(f"    Max:    {w.max().item():.2f}")
            print(f"    Mean:   {w.mean().item():.2f}")
            print(f"    Median: {w.median().item():.2f}")

            # Effective samples (sum of weights normalized)
            w_norm = w / w.sum()
            eff_samples = torch.exp(-torch.sum(w_norm * torch.log(w_norm + 1e-10)))
            print(f"    Effective samples: {eff_samples.item():.0f} / {len(w)}")
        else:
            data = synthetic_data
            print(f"\nUsing synthetic pairs only ({data['n_queries']} pairs)")

    torch.save(data, output_data_file)
    print(f"\nSaved training data to {output_data_file}")

    # Log data hash for reproducibility tracking
    data_hash = hashlib.sha256(open(output_data_file, "rb").read()).hexdigest()
    print(f"{os.path.basename(output_data_file)} hash: {data_hash}")

    # Update last training state after successful data generation
    state_payload = {
        "data_hash": data_hash,
        "timestamp": _utc_now_iso(),
        "n_signals": n_signals_for_run,
    }
    with open(LAST_TRAINED_FILE, "w") as fh:
        json.dump(state_payload, fh, indent=2)
    print(f"Updated training state: {LAST_TRAINED_FILE}")

    # -----------------------------------------------------------------------
    # Baseline evaluation (cosine similarity)
    # -----------------------------------------------------------------------
    print(f"\nBaseline: cosine similarity ranking...")
    n = data["n_queries"]
    session_ids = data.get("session_ids")

    if session_ids and len(session_ids) == n:
        from collections import defaultdict
        _s2i = defaultdict(list)
        for _i, _sid in enumerate(session_ids):
            _s2i[_sid].append(_i)
        _sessions = sorted(_s2i.keys())
        _split = int(len(_sessions) * 0.8)
        _val_idx = torch.tensor([i for sid in _sessions[_split:] for i in _s2i[sid]], dtype=torch.long)
        n_unique_sessions = len(_sessions)
        n_val_sessions = len(_sessions) - _split
        print(f"  Session-level split: {n_unique_sessions} sessions, {n_val_sessions} held out for validation")
    else:
        _split = int(n * 0.8)
        _val_idx = torch.arange(_split, n, dtype=torch.long)

    q_val = data["query_embeddings"][_val_idx]
    c_val = data["candidate_embeddings"][_val_idx]
    l_val = data["labels"][_val_idx]

    # Compute LRAT intensity weights if session duration data is available
    if 'session_durations' in data and data['session_durations'].sum() > 0:
        intensity_w = compute_lrat_intensity(data['session_durations'])
        combined_w = combine_weights(data['weights'], intensity_w)
        print(f"  LRAT intensity: median_duration={data['session_durations'].median():.0f}s, "
              f"intensity_range=[{intensity_w.min():.2f}, {intensity_w.max():.2f}]")
    else:
        combined_w = data['weights']
        print(f"  LRAT intensity: no session duration data, using type weights only")

    cosine_ndcg = []
    for i in range(q_val.size(0)):
        sims = F.cosine_similarity(q_val[i].unsqueeze(0), c_val[i], dim=-1)
        _, top_idx = sims.topk(min(TOP_K, sims.size(0)))
        rels = l_val[i][top_idx]
        positions = torch.arange(1, len(rels) + 1, dtype=torch.float32)
        dcg = (rels / torch.log2(positions + 1)).sum().item()
        ideal_rel, _ = l_val[i].sort(descending=True)
        ideal_rel = ideal_rel[:TOP_K]
        ideal_pos = torch.arange(1, len(ideal_rel) + 1, dtype=torch.float32)
        idcg = (ideal_rel / torch.log2(ideal_pos + 1)).sum().item()
        cosine_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    baseline = sum(cosine_ndcg) / len(cosine_ndcg)

    # Weight-aware baseline for comparison
    w_val = data.get("weights")
    if w_val is not None:
        w_val = w_val[_val_idx]
        weighted_ndcg = sum(n * w for n, w in zip(cosine_ndcg, w_val.tolist())) / w_val.sum().item()
        print(f"Cosine baseline NDCG@{TOP_K} (unweighted): {baseline:.6f}")
        print(f"Cosine baseline NDCG@{TOP_K} (weighted):   {weighted_ndcg:.6f}")
    else:
        print(f"Cosine similarity baseline NDCG@{TOP_K}: {baseline:.6f}")

    # -----------------------------------------------------------------------
    # Baseline evaluation (BM25+recency proxy)
    # -----------------------------------------------------------------------
    bm25_recency_baseline = compute_bm25_recency_baseline(
        q_val, c_val, l_val, embeddings, all_chunks, args.workspace,
    )
    if bm25_recency_baseline is not None:
        print(f"BM25+recency baseline NDCG@{TOP_K}: {bm25_recency_baseline:.6f}")
    else:
        print(f"BM25+recency baseline: skipped (could not compute scores)")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Run mode:              {run_mode}")
    print(f"  Total training pairs:  {data['n_queries']}")
    print(f"  Signal-based pairs:    {n_signal_pairs}")
    print(f"  Synthetic pairs:       {data['n_queries'] - n_signal_pairs}")
    print(f"  Has sample weights:    {'weights' in data}")
    print(f"  Cosine baseline:       {baseline:.6f}")
    if bm25_recency_baseline is not None:
        print(f"  BM25+recency baseline: {bm25_recency_baseline:.6f}")
    print(f"{'='*60}")
    print(f"\nDone! Ready to train. The TRM should beat {baseline:.6f}")
