"""
Mine Claude Code session transcripts for LRAT-style training triples.

Processes ~/.claude/projects session JSONLs to extract search->browse->reason
triples that capture actual cognitive trajectories through the workspace.

Each triple records:
  - query: the user message that drove the tool call
  - file_path: the file that was browsed (Read/Grep/Glob)
  - reasoning_tokens: how much the assistant reasoned after reading
  - outcome: what happened next (write/edit/continue/pivot)

This is the real training signal -- actual agent behavior recorded in
tool call history. Output is consumed by prepare.py (task C3).

Usage:
    python mine_sessions.py --sessions-dir ~/.claude/projects/-Users-slowbro-workspaces-cog/
    python mine_sessions.py --sessions-dir ~/.claude/projects/ --recursive
    python mine_sessions.py --help
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Session parsing helpers
# ---------------------------------------------------------------------------

# User messages that are not real queries (approvals, confirmations, etc.)
SKIP_PATTERNS = frozenset([
    "yes", "no", "ok", "sure", "thanks", "continue",
    "go ahead", "do it", "looks good", "approved",
    "y", "n", "heartbeat", "no_reply", "ok!", "yep",
    "nope", "agreed", "lgtm", "ship it",
])

# Tool names that count as "browsing" (search/read operations)
BROWSE_TOOLS = frozenset(["Read", "Grep", "Glob"])

# Tool names that count as "write" outcomes
WRITE_TOOLS = frozenset(["Edit", "Write"])

# MCP variants of browse/write tools
MCP_BROWSE_TOOLS = frozenset([
    "mcp__cogos-bridge__openclaw_read",
    "mcp__cogos-bridge__cogos_memory_read",
    "mcp__cogos-bridge__openclaw_memory_get",
    "mcp__cogos-bridge__openclaw_memory_search",
    "mcp__cogos-bridge__cogos_memory_search",
])

MCP_WRITE_TOOLS = frozenset([
    "mcp__cogos-bridge__openclaw_edit",
    "mcp__cogos-bridge__openclaw_write",
])


def _extract_user_text(content) -> str | None:
    """Extract the actual user text from a message content field.

    Content may be a plain string or a list of content blocks.
    Returns None if the content is a tool_result (not a real user message).
    """
    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        # Check if this is a tool_result message (not a real user query)
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return None  # tool_result, not a user message

        # Extract text from content blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])
        text = " ".join(texts).strip()
        return text or None

    return None


def _is_skip_message(text: str) -> bool:
    """Return True if the message is too short or is a confirmation/approval."""
    if not text or len(text) < 15:
        return True
    return text.lower().strip().rstrip("!?.") in SKIP_PATTERNS


def _extract_file_path_from_tool(tool_name: str, tool_input: dict) -> str | None:
    """Extract the browsed file path from a tool_use block."""
    if tool_name == "Read":
        return tool_input.get("file_path")
    elif tool_name == "Grep":
        # Grep targets a path (directory or file)
        path = tool_input.get("path", "")
        pattern = tool_input.get("pattern", "")
        if path:
            return path
        # If no explicit path, record the pattern as context
        if pattern:
            return f"grep:{pattern}"
        return None
    elif tool_name == "Glob":
        return tool_input.get("pattern")
    elif tool_name in MCP_BROWSE_TOOLS:
        return tool_input.get("path") or tool_input.get("file_path") or tool_input.get("query")
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        # Extract file reads from bash commands
        mem_reads = re.findall(r'cog\s+memory\s+read\s+([\w/.-]+)', cmd)
        if mem_reads:
            return mem_reads[0]
        # cat with an explicit file path (must start with / or . or ~, not a flag)
        cat_reads = re.findall(r'\bcat\s+["\']?(/[^\s"\'|;]+|\./?[^\s"\'|;]+|~/[^\s"\'|;]+)', cmd)
        if cat_reads:
            return cat_reads[0]
        return None
    return None


def _approximate_tokens(text: str) -> int:
    """Approximate token count from text (roughly 4 chars per token)."""
    if not text:
        return 0
    return len(text) // 4


def _is_valid_file_path(path: str) -> bool:
    """Return True if the path looks like a real file path, not a flag or noise."""
    if not path or len(path) < 3:
        return False
    # Reject flags (-5, -20, etc.) and heredoc markers (<<)
    if path.startswith("-") or path.startswith("<<"):
        return False
    # Reject bare grep patterns (no path separator)
    if path.startswith("grep:"):
        # Keep grep patterns as they encode search intent
        return True
    # Must contain at least one path separator or look like a relative path
    if "/" not in path and "." not in path and "*" not in path:
        return False
    return True


# ---------------------------------------------------------------------------
# Core parser: Claude Code session format
# ---------------------------------------------------------------------------

def _parse_claude_code_session(messages: list[dict]) -> list[dict]:
    """Parse Claude Code session format into LRAT triples.

    Walks the message stream and for each browse tool call (Read/Grep/Glob),
    records:
      - The most recent real user query
      - The file path from the tool call
      - The reasoning tokens in the assistant's response after the tool result
      - The outcome: whether the session went on to write/edit, pivoted to a
        new query, or just continued browsing
    """
    triples = []

    # State tracking
    current_query = None
    current_query_ts = None
    pending_browse = None  # (file_path, tool_name, line_index)
    assistant_text_after_browse = []
    browse_calls_in_turn = []  # all browse calls since last user message

    for i, msg in enumerate(messages):
        msg_type = msg.get("type", "")
        timestamp = msg.get("timestamp")

        if msg_type == "user":
            content = msg.get("message", {}).get("content", "")
            user_text = _extract_user_text(content)

            if user_text is not None and not _is_skip_message(user_text):
                # This is a real user query -- flush any pending browse calls
                # with "continue" or "pivot" outcome
                for browse in browse_calls_in_turn:
                    triples.append(browse)

                # New query
                current_query = user_text[:500]
                current_query_ts = timestamp
                browse_calls_in_turn = []
                assistant_text_after_browse = []

            elif user_text is None:
                # tool_result -- the response to a tool call
                # Don't change query state, just continue
                pass

        elif msg_type == "assistant" and current_query:
            content = msg.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            for block in content:
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    # This is reasoning text -- accumulate for the most recent browse
                    if browse_calls_in_turn:
                        browse_calls_in_turn[-1]["reasoning_tokens"] += _approximate_tokens(text)

                elif block_type == "tool_use":
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    # Check if this is a browse tool
                    is_browse = (
                        tool_name in BROWSE_TOOLS
                        or tool_name in MCP_BROWSE_TOOLS
                        or (tool_name == "Bash" and _extract_file_path_from_tool("Bash", tool_input))
                    )

                    # Check if this is a write tool
                    is_write = tool_name in WRITE_TOOLS or tool_name in MCP_WRITE_TOOLS

                    if is_write:
                        # Mark all pending browse calls as having "write" or "edit" outcome
                        outcome = "edit" if tool_name == "Edit" else "write"
                        for browse in browse_calls_in_turn:
                            if browse["outcome"] == "continue":
                                browse["outcome"] = outcome
                        # Flush
                        for browse in browse_calls_in_turn:
                            triples.append(browse)
                        browse_calls_in_turn = []

                    elif is_browse:
                        file_path = _extract_file_path_from_tool(tool_name, tool_input)
                        if file_path and _is_valid_file_path(file_path):
                            triple = {
                                "query": current_query,
                                "file_path": file_path,
                                "reasoning_tokens": 0,
                                "outcome": "continue",
                                "timestamp": timestamp or current_query_ts,
                            }
                            browse_calls_in_turn.append(triple)

    # Flush remaining browse calls
    for browse in browse_calls_in_turn:
        triples.append(browse)

    return triples


# ---------------------------------------------------------------------------
# Core parser: CogOS thread format
# ---------------------------------------------------------------------------

def _parse_cogos_thread(messages: list[dict]) -> list[dict]:
    """Parse CogOS thread format: {role, content, timestamp}.

    CogOS threads embed tool usage in the content text rather than using
    structured tool_use blocks. We extract file references from the text.
    """
    triples = []
    current_query = None
    current_query_ts = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp")

        if role == "user" and isinstance(content, str):
            # Strip metadata from user messages
            text = content
            text = re.sub(
                r'<<<EXTERNAL_UNTRUSTED_CONTENT.*?<<<END_EXTERNAL_UNTRUSTED_CONTENT.*?>>>',
                '', text, flags=re.DOTALL
            )
            text = re.sub(r'```json\n\{[^}]+\}\n```', '', text, flags=re.DOTALL)
            text = re.sub(r'A new session was started.*?reasoning\.', '', text, flags=re.DOTALL)
            text = re.sub(r'Conversation info.*?Sender.*?```', '', text, flags=re.DOTALL)
            text = text.strip()

            if text and not _is_skip_message(text):
                current_query = text[:500]
                current_query_ts = timestamp

        elif role == "assistant" and isinstance(content, str) and current_query:
            # Look for file path references that indicate reads
            paths = re.findall(r'(?:cog://mem/|\.cog/mem/)[\w/.-]+\.(?:cog\.)?md', content)
            for p in paths:
                p = p.replace("cog://mem/", ".cog/mem/")
                if not p.endswith(".md"):
                    p += ".md"
                triples.append({
                    "query": current_query,
                    "file_path": p,
                    "reasoning_tokens": _approximate_tokens(content),
                    "outcome": "continue",
                    "timestamp": timestamp or current_query_ts,
                })

    return triples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mine_session(jsonl_path: str) -> list[dict]:
    """
    Extract LRAT-style training triples from a Claude Code session transcript.

    Returns list of:
    {
        "session_id": str,        # from filename
        "query": str,             # user message text (truncated to 500 chars)
        "file_path": str,         # the file that was read
        "reasoning_tokens": int,  # length of assistant response after read
        "outcome": str,           # "write" | "edit" | "continue" | "pivot"
        "timestamp": str,         # ISO timestamp if available
    }
    """
    session_id = Path(jsonl_path).stem

    # Parse JSONL
    messages = []
    parse_errors = 0
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
    except (OSError, IOError) as e:
        print(f"  Warning: could not read {jsonl_path}: {e}", file=sys.stderr)
        return []

    if not messages:
        return []

    # Skip sessions shorter than 5 turns (not enough signal)
    turn_count = sum(1 for m in messages if m.get("type") in ("user", "assistant")
                     or m.get("role") in ("user", "assistant"))
    if turn_count < 5:
        return []

    # Detect format: Claude Code uses "type" field, CogOS threads use "role" field
    first = messages[0]
    is_cogos_thread = "role" in first and "type" not in first

    if is_cogos_thread:
        raw_triples = _parse_cogos_thread(messages)
    else:
        raw_triples = _parse_claude_code_session(messages)

    # Annotate with session_id and handle pivot detection
    triples = []
    seen_queries = set()
    for triple in raw_triples:
        query = triple["query"]

        # Detect pivot: if the query changed between consecutive browse calls
        # that haven't led to a write, mark earlier ones as "pivot"
        # (We handle this simply: the flush logic in the parser already sets
        # "continue" for non-write outcomes; we refine here)

        triple["session_id"] = session_id
        triples.append(triple)

    if parse_errors > 0:
        print(f"  Warning: {parse_errors} malformed lines in {jsonl_path}", file=sys.stderr)

    return triples


def mine_all_sessions(sessions_dir: str, recursive: bool = False) -> list[dict]:
    """Mine all JSONL session files in a directory.

    Args:
        sessions_dir: Path to directory containing session JSONL files
        recursive: If True, also scan subdirectories (one level deep)

    Returns:
        Combined list of triples from all sessions
    """
    import glob as glob_mod

    session_files = []

    if not os.path.isdir(sessions_dir):
        print(f"Warning: sessions directory not found: {sessions_dir}", file=sys.stderr)
        return []

    # Direct JSONL files
    session_files.extend(glob_mod.glob(os.path.join(sessions_dir, "*.jsonl")))

    if recursive:
        # One level deep
        session_files.extend(glob_mod.glob(os.path.join(sessions_dir, "*", "*.jsonl")))
        # Subagent sessions
        session_files.extend(glob_mod.glob(os.path.join(sessions_dir, "*/subagents/*.jsonl")))

    # Deduplicate by realpath
    seen = set()
    unique_files = []
    for sf in sorted(session_files):
        real = os.path.realpath(sf)
        if real not in seen:
            seen.add(real)
            unique_files.append(sf)

    all_triples = []
    sessions_processed = 0
    sessions_with_data = 0
    total_parse_errors = 0

    for sf in unique_files:
        sessions_processed += 1
        triples = mine_session(sf)
        if triples:
            sessions_with_data += 1
            all_triples.extend(triples)

    return all_triples


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

# Map mine_sessions outcome strings to signal weights for prepare.py
# These map to the LRAT-inspired SIGNAL_WEIGHTS in prepare.py:
#   crystallization=3.0, cascade=2.5, provenance=2.0, correct=1.5,
#   accept=1.0, continue=0.5, last=0.0
# We emit these as "outcome" values so prepare.py can apply its own weighting.
OUTCOME_TO_SIGNAL_TYPE = {
    "write":    "crystallization",  # new file written → strongest retrieval signal
    "edit":     "cascade",          # existing file edited → high-value interaction
    "continue": "accept",           # browsed and continued → accepted as useful
    "pivot":    "continue",         # browsed then changed direction → weak signal
}

# Local weight mapping for pre-computed weight field (matches prepare.py SIGNAL_WEIGHTS)
SIGNAL_WEIGHTS_LOCAL = {
    "write":    3.0,
    "edit":     2.0,
    "continue": 1.0,
    "pivot":    0.5,
}


def _parse_timestamp(ts: str | None):
    """Parse ISO timestamp string to datetime, or None on failure."""
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def generate_training_pairs(triples: list[dict], workspace_root: str) -> list[dict]:
    """
    Convert raw session triples into training signal format compatible with prepare.py.

    Each triple becomes a signal dict matching the format in prepare.py's
    generate_signal_pairs():
    {
        "query": str,              # the user's prompt
        "positives": [str],        # file paths that were read (workspace-relative)
        "negatives": [],           # empty (prepare.py mines its own negatives)
        "weight": float,           # from SIGNAL_WEIGHTS_LOCAL mapping
        "outcome": str,            # prepare.py signal type (e.g. "crystallization")
        "session": str,            # session ID
        "session_duration": float, # seconds from first-to-last timestamp in session
        "tool_calls": int,         # count of triples in this (session, query) group
        "timestamp": str,          # ISO timestamp of earliest triple in group
    }

    Steps:
    1. Group triples by (session_id, query) — multiple file reads for the same
       query become one signal with all paths as positives.
    2. For each group:
       - positives = all valid file_paths in the group (workspace-relative)
       - weight = max weight across all outcomes in the group
       - outcome = signal type for the max-weight outcome
    3. Filter: skip signals where query < 10 chars or no valid positives remain.
    4. Convert absolute file paths to workspace-relative paths.
    """
    workspace_root = workspace_root.rstrip("/")

    # ---------------------------------------------------------------
    # Step 1: Group by (session_id, query)
    # ---------------------------------------------------------------
    from collections import OrderedDict

    # key: (session_id, query) → list of triples
    groups: dict[tuple, list[dict]] = OrderedDict()
    for triple in triples:
        key = (triple.get("session_id", ""), triple.get("query", ""))
        groups.setdefault(key, []).append(triple)

    # ---------------------------------------------------------------
    # Step 2: Compute per-session timestamp spreads for duration
    # ---------------------------------------------------------------
    # session_id → (first_ts, last_ts) as datetime objects
    session_ts: dict[str, list] = {}
    for triple in triples:
        sid = triple.get("session_id", "")
        dt = _parse_timestamp(triple.get("timestamp"))
        if dt is None:
            continue
        if sid not in session_ts:
            session_ts[sid] = [dt, dt]
        else:
            if dt < session_ts[sid][0]:
                session_ts[sid][0] = dt
            if dt > session_ts[sid][1]:
                session_ts[sid][1] = dt

    def _session_duration(sid: str) -> float:
        if sid not in session_ts:
            return 0.0
        first, last = session_ts[sid]
        return max(0.0, (last - first).total_seconds())

    # ---------------------------------------------------------------
    # Step 3: Build signal dicts
    # ---------------------------------------------------------------
    signals = []

    for (session_id, query), group_triples in groups.items():
        # Filter: skip short queries
        if not query or len(query) < 10:
            continue

        # Collect file paths and make them workspace-relative
        positives = []
        seen_paths: set[str] = set()
        for t in group_triples:
            fp = t.get("file_path", "")
            if not fp or not _is_valid_file_path(fp):
                continue
            # Make workspace-relative
            if fp.startswith(workspace_root + "/"):
                fp = fp[len(workspace_root) + 1:]
            elif fp.startswith(workspace_root):
                fp = fp[len(workspace_root):]
            # Skip grep: patterns and glob wildcards as positives
            # (they don't map to files prepare.py can chunk-match)
            if fp.startswith("grep:") or "*" in fp:
                continue
            if fp not in seen_paths:
                seen_paths.add(fp)
                positives.append(fp)

        # Filter: must have at least one usable file path
        if not positives:
            continue

        # Determine best outcome weight for this group
        best_weight = 0.0
        best_outcome_raw = "continue"
        for t in group_triples:
            raw_outcome = t.get("outcome", "continue")
            w = SIGNAL_WEIGHTS_LOCAL.get(raw_outcome, 0.5)
            if w > best_weight:
                best_weight = w
                best_outcome_raw = raw_outcome

        signal_type = OUTCOME_TO_SIGNAL_TYPE.get(best_outcome_raw, "continue")

        # Earliest timestamp in this group
        group_timestamps = [t.get("timestamp") for t in group_triples if t.get("timestamp")]
        group_dts = [_parse_timestamp(ts) for ts in group_timestamps]
        group_dts = [dt for dt in group_dts if dt is not None]
        earliest_ts = min(group_dts).isoformat().replace("+00:00", "Z") if group_dts else None

        signals.append({
            "query": query,
            "positives": positives,
            "negatives": [],
            "weight": best_weight,
            "outcome": signal_type,
            "session": session_id,
            "session_duration": _session_duration(session_id),
            "tool_calls": len(group_triples),
            "timestamp": earliest_ts,
        })

    return signals


def compute_stats(triples: list[dict]) -> dict:
    """Compute summary statistics over extracted triples."""
    if not triples:
        return {
            "total_triples": 0,
            "sessions": 0,
            "outcome_counts": {},
            "avg_reasoning_tokens": 0,
            "top_files": [],
        }

    sessions = set(t["session_id"] for t in triples)
    outcome_counts = defaultdict(int)
    file_counts = defaultdict(int)
    total_reasoning = 0

    for t in triples:
        outcome_counts[t["outcome"]] += 1
        file_counts[t["file_path"]] += 1
        total_reasoning += t["reasoning_tokens"]

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:20]

    return {
        "total_triples": len(triples),
        "sessions": len(sessions),
        "outcome_counts": dict(outcome_counts),
        "avg_reasoning_tokens": total_reasoning / len(triples),
        "top_files": top_files,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mine Claude Code session transcripts for LRAT-style training triples"
    )
    parser.add_argument(
        "--sessions-dir",
        type=str,
        default=os.path.expanduser("~/.claude/projects/-Users-slowbro-workspaces-cog/"),
        help="Directory containing session JSONL files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subdirectories (one level deep)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: ~/.cache/cogos-autoresearch/session_triples.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only, don't write output",
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help=(
            "Convert mined triples into training signal format for prepare.py. "
            "Reads session_triples.json from cache and writes session_signals.json."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(pathlib.Path.home() / "workspaces" / "cog"),
        help="Workspace root for making file paths relative (used with --generate-pairs)",
    )
    args = parser.parse_args()

    # Determine output path
    output_dir = os.path.expanduser("~/.cache/cogos-autoresearch")
    output_path = args.output or os.path.join(output_dir, "session_triples.json")

    # -----------------------------------------------------------------------
    # --generate-pairs: convert existing triples → training signals
    # -----------------------------------------------------------------------
    if args.generate_pairs:
        triples_path = os.path.join(output_dir, "session_triples.json")
        if not os.path.isfile(triples_path):
            print(
                f"Error: {triples_path} not found.\n"
                f"Run without --generate-pairs first to mine session triples.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Loading triples from: {triples_path}")
        with open(triples_path, "r") as f:
            triples = json.load(f)
        print(f"Loaded {len(triples)} triples")

        workspace_root = os.path.expanduser(args.workspace)
        signals = generate_training_pairs(triples, workspace_root)

        signals_path = os.path.join(output_dir, "session_signals.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(signals_path, "w") as f:
            json.dump(signals, f, indent=2, default=str)

        n_sessions = len(set(s["session"] for s in signals))
        print(
            f"Generated {len(signals)} signals from {len(triples)} triples "
            f"({n_sessions} sessions)"
        )
        print(f"Wrote: {signals_path}")
        return

    print(f"Mining sessions from: {args.sessions_dir}")
    if args.recursive:
        print(f"  (recursive scan enabled)")

    # Mine all sessions
    triples = mine_all_sessions(args.sessions_dir, recursive=args.recursive)

    # Compute and print stats
    stats = compute_stats(triples)

    print(f"\nProcessed {stats['sessions']} sessions, extracted {stats['total_triples']} triples")
    print(f"\nOutcome breakdown:")
    for outcome, count in sorted(stats["outcome_counts"].items(), key=lambda x: -x[1]):
        print(f"  {outcome:12s}  {count:6d}")

    print(f"\nAvg reasoning tokens per triple: {stats['avg_reasoning_tokens']:.0f}")

    if stats["top_files"]:
        print(f"\nTop 20 most-browsed files:")
        for path, count in stats["top_files"]:
            print(f"  {count:4d}  {path}")

    # Write output
    if not args.dry_run and triples:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(triples, f, indent=2, default=str)
        print(f"\nWrote {len(triples)} triples to {output_path}")
    elif args.dry_run:
        print(f"\nDry run -- {len(triples)} triples ready (not written)")
    else:
        print(f"\nNo triples extracted.")


if __name__ == "__main__":
    main()
