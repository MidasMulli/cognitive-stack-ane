"""Retrieval shape #6 — Enumeration retrieval.

Detects "list all X" / "how many X" queries and returns the COMPLETE
set of records for a tag, bypassing top-k truncation.

Pure Python, no LLM calls. Loads tag_index.json built by
tools/build_tag_index.py.
"""
import json
import re
import time
from pathlib import Path

_TAG_INDEX_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "tag_index.json")

# ── Intent detection ───────────────────────────────────────────────────────

# m120_e_enumeration_dispatcher: widen intent regex so T88-style queries
# (e.g. "enumerate all published repos under my name") reach the shape
# dispatcher. Prior regex matched "enumerate" bare but the downstream
# tag-keyword gate still gated routing on a narrow tag list; the widening
# here also covers "list the {X}s", "what are the {X}s" plural, "show me
# all", "give me the full list of", bare "enumerate {X}", and
# "list all / every {X}". Conservative: every alternate still requires an
# explicit enumeration verb (list/show/enumerate/how-many/what-are-the/
# give-me/name/count/full list), so it will not match "what is the X".
_ENUM_INTENT = re.compile(
    r"(?i)\b(list\s+(?:all|every|the)|show\s+(?:all|every|me\s+all)|"
    r"how\s+many|what\s+are\s+(?:all\s+)?the|enumerate|"
    r"give\s+me\s+(?:all|every|the\s+full\s+list\s+of|a\s+full\s+list\s+of)|"
    r"(?:full|complete)\s+list\s+of|"
    r"name\s+(?:all|every)|"
    r"count\s+(?:all|the)|what\s+(?:dead|decisions?|measurements?|corrections?|plans?|preferences?))"
)

# Map from query keywords to tag names. Keys present in tag_index.json
# resolve to concrete records; keys NOT in the tag index (published_repo,
# standing_rule) still let the dispatcher recognize the shape, but the
# call ultimately returns None (handled by shape_dispatch_signal below).
_TAG_KEYWORDS = {
    "dead_path":      re.compile(r"(?i)\b(dead\s*path|killed|dead\s+end|dead\s+path|dead path|dead-path)s?\b"),
    "decision":       re.compile(r"(?i)\b(decision)s?\b"),
    "measurement":    re.compile(r"(?i)\b(measurement|metric|number|benchmark)s?\b"),
    "preference":     re.compile(r"(?i)\b(preference)s?\b"),
    "plan":           re.compile(r"(?i)\b(plan|roadmap|todo|next step)s?\b"),
    "correction":     re.compile(r"(?i)\b(correction|fix|bug\s*fix|retraction)s?\b"),
    "open_lead":      re.compile(r"(?i)\b(open\s+lead|revival|revived|open\s+question|future\s+work|TODO)s?\b"),
    # m120_e_enumeration_dispatcher: tags recognized by dispatcher but not
    # yet backed by tag_index.json. Shape classifies correctly; record
    # retrieval is a K9 architectural gap filed for M121+.
    "published_repo": re.compile(r"(?i)\b(published\s+repo|repo|repositor(?:y|ies)|published\s+project)s?\b"),
    "standing_rule":  re.compile(r"(?i)\b(standing\s+rule|rule)s?\b"),
}

# m120_e_enumeration_dispatcher: tags the dispatcher recognizes for shape
# classification but that are not yet present in the tag index. Callers
# using shape_dispatch_signal() can still log shape_fired='enumeration' for
# observability; enumerate_by_tag() still returns None so downstream
# behavior (fall-through to default_recall) is preserved until the
# architectural gap is closed.
#
# M121 Stream B: published_repo + standing_rule are now populated in
# tag_index.json by tools/build_tag_index.py (CLAUDE.md "Public repos
# current as of Main NN" + vault/knowledge/standing_rules.md). They are
# therefore REMOVED from _UNBACKED_TAGS; the frozenset is retained (empty)
# as the forward-compatible home for any future shape-recognized-but-
# unbacked tags.
_UNBACKED_TAGS: frozenset = frozenset()

# ── Tag index cache ────────────────────────────────────────────────────────

_cached_index: dict | None = None
_cached_mtime: float = 0.0


def _load_index() -> dict:
    """Load and cache the tag index, refreshing on file change."""
    global _cached_index, _cached_mtime
    p = Path(_TAG_INDEX_PATH)
    if not p.exists():
        return {}
    mtime = p.stat().st_mtime
    if _cached_index is not None and mtime == _cached_mtime:
        return _cached_index
    try:
        _cached_index = json.loads(p.read_text(encoding="utf-8"))
        _cached_mtime = mtime
    except Exception:
        _cached_index = {}
    return _cached_index


# ── Formatting ─────────────────────────────────────────────────────────────

def _format_dead_path(rec: dict) -> str:
    """Format a dead path record as a concise one-liner."""
    entity = rec.get("entity", "")
    kill_reason = rec.get("kill_reason", "")
    if kill_reason:
        return f"{entity}: {kill_reason}"
    # Corpus-sourced dead path — extract entity and reason from content
    content = rec.get("content", "")
    if len(content) > 120:
        content = content[:117] + "..."
    return content


def _format_generic(rec: dict) -> str:
    """Format a non-dead-path record concisely."""
    content = rec.get("content", "")
    session = rec.get("session_label", "")
    if len(content) > 150:
        content = content[:147] + "..."
    suffix = f" [{session}]" if session else ""
    return f"{content}{suffix}"


def _format_records(tag: str, records: list[dict], topic_filter: str | None = None) -> list[str]:
    """Format all records for a tag into a list of strings.

    For large sets (>30), uses one-line-per-item format.
    Optionally filters by a topic keyword.
    """
    if topic_filter:
        pat = re.compile(re.escape(topic_filter), re.IGNORECASE)
        records = [r for r in records if pat.search(r.get("content", ""))
                   or pat.search(r.get("entity", ""))]

    formatted = []
    for i, rec in enumerate(records, 1):
        if tag == "dead_path":
            line = _format_dead_path(rec)
        else:
            line = _format_generic(rec)
        formatted.append(f"{i}. {line}")

    return formatted


# ── Topic extraction ───────────────────────────────────────────────────────

def _extract_topic_filter(query: str) -> str | None:
    """Extract an optional topic filter from the query (e.g., 'about ANE').

    Only triggers on explicit 'about/regarding/related to' followed by a
    short noun phrase (1-4 words). Avoids grabbing trailing clauses like
    'with the kill reason for each'.
    """
    m = re.search(r"(?:about|regarding|related to|mentioning)\s+(?:the\s+)?([A-Za-z0-9][\w\s\-]{0,30}?)(?:\?|,|$|\.\s)", query, re.I)
    if m:
        topic = m.group(1).strip().rstrip("?., ")
        # Skip if the topic is just the tag name itself
        if topic.lower() in ("dead paths", "decisions", "measurements", "corrections",
                              "plans", "preferences", "open leads", ""):
            return None
        # Skip if too generic
        if len(topic) < 2:
            return None
        return topic
    return None


# ── Public API ─────────────────────────────────────────────────────────────

def shape_dispatch_signal(query: str) -> dict | None:
    """m120_e_enumeration_dispatcher: lightweight shape classifier.

    Returns a dict with shape classification info if the query matches the
    Enumeration shape, else None. Always returns something when intent AND
    a known tag keyword both match — even if the tag is in _UNBACKED_TAGS
    (K9 gap). This is the signal used to log shape_fired='enumeration' at
    the dispatch layer; actual record retrieval still goes through
    enumerate_by_tag() and remains gated by tag_index.json.
    """
    if not _ENUM_INTENT.search(query):
        return None
    matched_tag = None
    for tag, pattern in _TAG_KEYWORDS.items():
        if pattern.search(query):
            matched_tag = tag
            break
    if matched_tag is None:
        return None
    return {
        "shape": "enumeration",
        "tag": matched_tag,
        "backed": matched_tag not in _UNBACKED_TAGS,
    }


def enumerate_by_tag(query: str) -> dict | None:
    """Detect enumeration intent and return ALL matching records.

    Returns dict with records/tag/count/intent/latency_ms, or None if
    no enumeration intent detected.
    """
    t0 = time.monotonic()

    # Step 1: Check for enumeration intent
    if not _ENUM_INTENT.search(query):
        return None

    # Step 2: Find matching tag
    matched_tag = None
    for tag, pattern in _TAG_KEYWORDS.items():
        if pattern.search(query):
            matched_tag = tag
            break

    if matched_tag is None:
        return None

    # Step 3: Load index
    index = _load_index()
    if not index:
        return None

    records = index.get(matched_tag, [])
    if not records:
        return None

    # Step 4: Optional topic filter
    topic_filter = _extract_topic_filter(query)

    # Step 5: Sort — CLAUDE.md canonical entries first, then corpus
    if matched_tag == "dead_path":
        canonical = [r for r in records if r.get("session_label") == "CLAUDE.md"]
        corpus = [r for r in records if r.get("session_label") != "CLAUDE.md"]
        records = canonical + corpus

    # Step 6: Format ALL records (no top-k truncation)
    formatted = _format_records(matched_tag, records, topic_filter)

    latency_ms = (time.monotonic() - t0) * 1000

    return {
        "records": formatted,
        "tag": matched_tag,
        "count": len(formatted),
        "intent": "enumeration",
        "topic_filter": topic_filter,
        "latency_ms": round(latency_ms, 2),
    }


# ── CLI test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    queries = [
        "List every dead path we have documented, with the kill reason for each.",
        "What decisions have we made about the ANE?",
        "How many measurements do we have?",
        "Show all corrections.",
        "List every open lead.",
        "How are you?",
    ]
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        result = enumerate_by_tag(q)
        if result is None:
            print("  -> None (no enumeration intent)")
        else:
            print(f"  -> tag={result['tag']}, count={result['count']}, "
                  f"filter={result['topic_filter']}, latency={result['latency_ms']}ms")
            for line in result["records"][:10]:
                print(f"     {line}")
            if result["count"] > 10:
                print(f"     ... ({result['count'] - 10} more)")
