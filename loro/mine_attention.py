"""
Mine attention signals (.cog/run/attention.jsonl) for TRM training data.

Converts the attention signal log into the same (query -> retrievals)
exchange format that mine_sessions.py produces. This closes the digestion
loop: attention signals from live conversations feed back into TRM training.

Attention signals are emitted by the post-tool-use hook
(.cog/hooks/post-tool-use.d/16-attention-signal.py) and record every
Read, Grep, Glob, Edit, and Write tool call with a cog:// URI target.

The key insight: attention signals don't have user messages (queries)
attached. We reconstruct query-retrieval pairs by:
  1. Grouping signals by session (participant_id + time window)
  2. Using 'search' signals as implicit queries
  3. Using 'read' signals as retrievals (the same as Read tool calls)
  4. Treating each session window as one exchange

Output format matches mine_sessions.parse_session() return value, so
prepare_sequences.py can consume it without changes.

Usage:
    python3 mine_attention.py                              # process default log
    python3 mine_attention.py --log /path/to/attention.jsonl
    python3 mine_attention.py --dry-run                    # preview without embedding
    python3 mine_attention.py --output exchanges.json      # save exchanges as JSON
    python3 mine_attention.py --format sequences           # output for prepare_sequences
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LOG = os.path.join(
    os.path.expanduser("~/cog-workspace"),
    ".cog", "run", "attention.jsonl",
)

# Time window for grouping signals into sessions (seconds).
# Signals within this window from the same participant are one "session".
SESSION_GAP_SECONDS = 300  # 5 minutes

# Signal types that count as retrievals (positive training signal)
RETRIEVAL_SIGNALS = {"read"}

# Signal types that count as searches (query-like)
SEARCH_SIGNALS = {"search", "traverse"}

# Signal types that count as edits
EDIT_SIGNALS = {"write"}

# Minimum signals per session to be useful
MIN_SIGNALS_PER_SESSION = 3

# Minimum reads per session to produce a training exchange
MIN_READS_PER_SESSION = 1

# URI prefixes we care about (workspace files, memory docs)
USEFUL_URI_PREFIXES = (
    "cog://workspace/",
    "cog://mem/",
    "cog://search/",
    "cog://glob/",
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_attention_log(log_path: str) -> list[dict]:
    """
    Parse attention.jsonl into a list of signal dicts.

    Each signal has:
      - participant_id: str (e.g., "claude-code:unknown", "human:Mac-...")
      - target_uri: str (e.g., "cog://workspace/apps/cogos-v3/main.go")
      - signal_type: str (read, search, write, visit, traverse)
      - context: dict (optional, e.g., {"tool": "Read"})
      - occurred_at: str (ISO timestamp)
    """
    signals = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                    # Basic validation
                    if "target_uri" in sig and "signal_type" in sig:
                        signals.append(sig)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Warning: attention log not found: {log_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: error reading attention log: {e}", file=sys.stderr)

    return signals


def parse_timestamp(ts: str) -> datetime:
    """Parse an ISO timestamp, handling both Z-suffix and +00:00 formats."""
    ts = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        # Fallback: strip fractional seconds
        if "." in ts:
            base, rest = ts.rsplit(".", 1)
            # Find the timezone part
            for sep in ("+", "-"):
                if sep in rest[1:]:
                    tz_idx = rest[1:].index(sep) + 1
                    tz = rest[tz_idx:]
                    ts = f"{base}{sep}{tz}"
                    break
            else:
                ts = base + "+00:00"
        return datetime.fromisoformat(ts)


def group_into_sessions(
    signals: list[dict],
    gap_seconds: int = SESSION_GAP_SECONDS,
) -> list[list[dict]]:
    """
    Group signals into sessions by participant + time proximity.

    A new session starts when there's a gap of > gap_seconds between
    consecutive signals from the same participant.
    """
    if not signals:
        return []

    # Sort by timestamp
    def sort_key(s):
        try:
            return parse_timestamp(s.get("occurred_at", ""))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    sorted_signals = sorted(signals, key=sort_key)

    # Group by participant, then split by time gaps
    by_participant = defaultdict(list)
    for sig in sorted_signals:
        pid = sig.get("participant_id", "unknown")
        by_participant[pid].append(sig)

    sessions = []
    for pid, sigs in by_participant.items():
        if not sigs:
            continue

        current_session = [sigs[0]]
        for sig in sigs[1:]:
            try:
                prev_time = parse_timestamp(current_session[-1]["occurred_at"])
                curr_time = parse_timestamp(sig["occurred_at"])
                gap = (curr_time - prev_time).total_seconds()
            except Exception:
                gap = 0

            if gap > gap_seconds:
                # Start new session
                if len(current_session) >= MIN_SIGNALS_PER_SESSION:
                    sessions.append(current_session)
                current_session = [sig]
            else:
                current_session.append(sig)

        # Don't forget the last session
        if len(current_session) >= MIN_SIGNALS_PER_SESSION:
            sessions.append(current_session)

    return sessions


# ---------------------------------------------------------------------------
# URI → file path conversion
# ---------------------------------------------------------------------------

def uri_to_file_path(uri: str, workspace_root: str = None) -> str:
    """
    Convert a cog:// URI to a workspace-relative file path.

    Examples:
      cog://workspace/apps/cogos-v3/main.go -> apps/cogos-v3/main.go
      cog://workspace/.cog/mem/semantic/foo.md -> .cog/mem/semantic/foo.md
      cog://mem/semantic/foo.md -> .cog/mem/semantic/foo.md
      cog://search/pattern -> (returns empty — not a file)
      cog://glob/pattern -> (returns empty — not a file)
    """
    if uri.startswith("cog://workspace/"):
        path = uri[len("cog://workspace/"):]
        # Strip leading workspace name if duplicated (e.g., "cog-workspace/...")
        if workspace_root:
            ws_name = os.path.basename(workspace_root)
            if path.startswith(ws_name + "/"):
                path = path[len(ws_name) + 1:]
        return path
    elif uri.startswith("cog://mem/"):
        return ".cog/mem/" + uri[len("cog://mem/"):]
    else:
        return ""


def uri_to_search_query(uri: str) -> str:
    """
    Extract a search query from a cog:// URI.

    cog://search/pattern -> pattern
    cog://glob/pattern -> pattern
    cog://workspace/path -> filename as pseudo-query
    """
    if uri.startswith("cog://search/"):
        return uri[len("cog://search/"):]
    elif uri.startswith("cog://glob/"):
        return uri[len("cog://glob/"):]
    elif uri.startswith("cog://workspace/") or uri.startswith("cog://mem/"):
        # Use the filename as a pseudo-query
        path = uri.rsplit("/", 1)[-1]
        # Strip extension
        if "." in path:
            path = path.rsplit(".", 1)[0]
        return path
    return ""


# ---------------------------------------------------------------------------
# Session → Exchange conversion
# ---------------------------------------------------------------------------

def session_to_exchanges(
    signals: list[dict],
    workspace_root: str = None,
) -> list[dict]:
    """
    Convert a session (list of signals) into exchanges in the same format
    as mine_sessions.parse_session().

    Strategy:
    - Group consecutive signals into query-retrieval windows
    - 'search' signals become the query text
    - 'read' signals become retrievals
    - If no search signal precedes a read burst, synthesize a query from
      the read targets (the filename/path IS the implicit query)

    Returns list of exchanges, each containing:
      - user_message: str (reconstructed query)
      - reads: list[str] (file paths)
      - greps: list[str] (search patterns)
      - globs: list[str] (glob patterns)
      - edits: list[str] (edited files)
    """
    if workspace_root is None:
        workspace_root = os.path.expanduser("~/cog-workspace")

    exchanges = []
    current_query_parts = []
    current_reads = []
    current_greps = []
    current_globs = []
    current_edits = []

    def flush_exchange():
        """Save current accumulated signals as an exchange."""
        nonlocal current_query_parts, current_reads, current_greps, current_globs, current_edits

        if not current_reads:
            # No reads — not useful for training
            current_query_parts = []
            current_greps = []
            current_globs = []
            current_edits = []
            return

        # Build query from search patterns, or from read targets
        if current_query_parts:
            query = " ".join(current_query_parts)
        else:
            # Synthesize query from read targets
            parts = []
            for r in current_reads:
                name = os.path.basename(r)
                if "." in name:
                    name = name.rsplit(".", 1)[0]
                # Convert underscores/hyphens to spaces
                name = name.replace("-", " ").replace("_", " ")
                parts.append(name)
            query = " ".join(dict.fromkeys(parts))  # deduplicate, preserve order

        if len(query) < 10:
            # Too short to be useful
            current_query_parts = []
            current_reads = []
            current_greps = []
            current_globs = []
            current_edits = []
            return

        exchanges.append({
            "user_message": query,
            "reads": list(dict.fromkeys(current_reads)),  # deduplicate, preserve order
            "greps": list(dict.fromkeys(current_greps)),
            "globs": list(dict.fromkeys(current_globs)),
            "edits": list(dict.fromkeys(current_edits)),
        })

        current_query_parts = []
        current_reads = []
        current_greps = []
        current_globs = []
        current_edits = []

    for sig in signals:
        stype = sig.get("signal_type", "")
        uri = sig.get("target_uri", "")

        # Skip URIs we don't care about
        if not any(uri.startswith(p) for p in USEFUL_URI_PREFIXES):
            continue

        if stype in SEARCH_SIGNALS:
            # Search/traverse signal — this is a query indicator
            query_text = uri_to_search_query(uri)
            if query_text:
                current_query_parts.append(query_text)
            file_path = uri_to_file_path(uri, workspace_root)
            if file_path:
                current_greps.append(f"{query_text} in {file_path}")

        elif stype in RETRIEVAL_SIGNALS:
            # Read signal — this is a retrieval (positive training signal)
            file_path = uri_to_file_path(uri, workspace_root)
            if file_path:
                current_reads.append(file_path)

        elif stype in EDIT_SIGNALS:
            # Write/Edit signal — tracks edits
            file_path = uri_to_file_path(uri, workspace_root)
            if file_path:
                current_edits.append(file_path)
                # An edit after reads suggests the reads were relevant to a task.
                # Flush the current exchange so the reads get paired with
                # the pre-edit search context.
                flush_exchange()

        elif stype == "visit":
            # Visit signal — start of a new interaction burst.
            # Flush any accumulated exchange.
            flush_exchange()

    # Flush the last exchange
    flush_exchange()

    return exchanges


# ---------------------------------------------------------------------------
# Full pipeline: attention log → exchanges
# ---------------------------------------------------------------------------

def mine_attention_log(
    log_path: str = None,
    workspace_root: str = None,
    gap_seconds: int = SESSION_GAP_SECONDS,
) -> list[dict]:
    """
    Full pipeline: parse attention log, group into sessions,
    convert to exchanges.

    Returns list of exchanges in mine_sessions-compatible format:
      [{"user_message": str, "reads": [...], "greps": [...], ...}, ...]
    """
    if log_path is None:
        log_path = DEFAULT_LOG
    if workspace_root is None:
        workspace_root = os.path.expanduser("~/cog-workspace")

    # Parse
    signals = parse_attention_log(log_path)
    if not signals:
        return []

    # Group into sessions
    sessions = group_into_sessions(signals, gap_seconds)

    # Convert each session to exchanges
    all_exchanges = []
    for session_signals in sessions:
        exchanges = session_to_exchanges(session_signals, workspace_root)
        all_exchanges.extend(exchanges)

    return all_exchanges


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_for_prepare_sequences(exchanges: list[dict]) -> list[dict]:
    """
    Format exchanges for prepare_sequences.py consumption.

    prepare_sequences.py imports parse_session from mine_sessions and
    expects each exchange to have: user_message, reads, greps, globs, edits.
    This is already our native format, so this is mostly validation.
    """
    formatted = []
    for ex in exchanges:
        if not ex.get("reads"):
            continue
        formatted.append({
            "user_message": ex["user_message"],
            "reads": ex.get("reads", []),
            "greps": ex.get("greps", []),
            "globs": ex.get("globs", []),
            "edits": ex.get("edits", []),
        })
    return formatted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mine attention signals for TRM training data",
        epilog=(
            "Converts .cog/run/attention.jsonl into (query -> retrievals) "
            "exchanges compatible with mine_sessions.py and prepare_sequences.py. "
            "Closes the digestion loop: attention signals feed back into TRM training."
        ),
    )
    parser.add_argument(
        "--log", type=str, default=DEFAULT_LOG,
        help=f"Path to attention.jsonl (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--workspace", type=str,
        default=os.path.expanduser("~/cog-workspace"),
        help="Workspace root for path resolution",
    )
    parser.add_argument(
        "--gap", type=int, default=SESSION_GAP_SECONDS,
        help=f"Session gap in seconds (default: {SESSION_GAP_SECONDS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview: show exchanges without further processing",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (JSON). If not set, prints to stdout.",
    )
    parser.add_argument(
        "--format", choices=["exchanges", "sequences"], default="exchanges",
        help="Output format: 'exchanges' (mine_sessions format) or "
             "'sequences' (prepare_sequences format)",
    )
    parser.add_argument(
        "--min-reads", type=int, default=MIN_READS_PER_SESSION,
        help=f"Minimum reads per exchange (default: {MIN_READS_PER_SESSION})",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show statistics about the attention log",
    )
    args = parser.parse_args()

    # Parse attention log
    signals = parse_attention_log(args.log)
    print(f"Parsed {len(signals)} signals from {args.log}", file=sys.stderr)

    if args.stats:
        # Show statistics
        by_type = defaultdict(int)
        by_participant = defaultdict(int)
        for sig in signals:
            by_type[sig.get("signal_type", "unknown")] += 1
            by_participant[sig.get("participant_id", "unknown")] += 1

        print(f"\nSignal types:", file=sys.stderr)
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t:15s} {c:5d}", file=sys.stderr)

        print(f"\nParticipants:", file=sys.stderr)
        for p, c in sorted(by_participant.items(), key=lambda x: -x[1]):
            print(f"  {p:40s} {c:5d}", file=sys.stderr)

        sessions = group_into_sessions(signals, args.gap)
        print(f"\nSessions (gap={args.gap}s): {len(sessions)}", file=sys.stderr)
        for i, sess in enumerate(sessions):
            pid = sess[0].get("participant_id", "unknown")
            t0 = sess[0].get("occurred_at", "?")
            t1 = sess[-1].get("occurred_at", "?")
            print(f"  [{i}] {pid} | {len(sess)} signals | {t0} -> {t1}",
                  file=sys.stderr)
        return

    # Group and convert
    sessions = group_into_sessions(signals, args.gap)
    print(f"Grouped into {len(sessions)} sessions (gap={args.gap}s)", file=sys.stderr)

    all_exchanges = []
    for session_signals in sessions:
        exchanges = session_to_exchanges(session_signals, args.workspace)
        # Filter by min reads
        exchanges = [e for e in exchanges if len(e.get("reads", [])) >= args.min_reads]
        all_exchanges.extend(exchanges)

    print(f"Extracted {len(all_exchanges)} exchanges", file=sys.stderr)

    if not all_exchanges:
        print("No exchanges extracted.", file=sys.stderr)
        return

    # Format
    if args.format == "sequences":
        output = format_for_prepare_sequences(all_exchanges)
    else:
        output = all_exchanges

    # Display or save
    if args.dry_run:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ATTENTION LOG MINING RESULTS", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Signals:    {len(signals)}", file=sys.stderr)
        print(f"Sessions:   {len(sessions)}", file=sys.stderr)
        print(f"Exchanges:  {len(all_exchanges)}", file=sys.stderr)
        print(f"Avg reads:  {sum(len(e['reads']) for e in all_exchanges) / max(len(all_exchanges), 1):.1f}",
              file=sys.stderr)

        print(f"\nSample exchanges:", file=sys.stderr)
        for i, ex in enumerate(all_exchanges[:10]):
            print(f"\n  [{i}] Query: {ex['user_message'][:100]}", file=sys.stderr)
            print(f"      Reads ({len(ex['reads'])}):", file=sys.stderr)
            for r in ex["reads"][:5]:
                print(f"        - {r}", file=sys.stderr)
            if ex.get("greps"):
                print(f"      Greps: {ex['greps'][:3]}", file=sys.stderr)
            if ex.get("edits"):
                print(f"      Edits: {ex['edits'][:3]}", file=sys.stderr)

        # Most-accessed files
        read_counts = defaultdict(int)
        for ex in all_exchanges:
            for r in ex["reads"]:
                read_counts[r] += 1

        if read_counts:
            print(f"\nMost-read files:", file=sys.stderr)
            for path, count in sorted(read_counts.items(), key=lambda x: -x[1])[:15]:
                print(f"  {count:4d}  {path}", file=sys.stderr)

        return

    # Output
    output_json = json.dumps(output, indent=2)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Wrote {len(output)} exchanges to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
