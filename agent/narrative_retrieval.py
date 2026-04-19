"""Phase 2 — Narrative synthesis retrieval.

Pre-retrieval pass: if the query matches a narrative thread (≥3 records
across ≥2 sessions), classify intent and return context:

  arc intent    → full narrative arc (chronological evolution)
  factual intent → most recent record + correction provenance
  no thread     → None (caller uses existing top-15 recall)

Pure Python, no LLM calls. Entity index lookup + arc trace + narrative
packaging. Target: <50ms total.
"""
import json
import re
import sys
import time
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_MIN_RECORDS = 3
_MIN_SESSIONS = 2
_MAX_RECORDS = 30

# M97 Fix 1 (narrative-shape over-firing gate). Calibrated against the M92
# 27-turn baseline: arc/factual-thread paths fired on 5 of 27 turns today
# (T05/T06/T07/T12/T13 after existing topical gate). Target ≤3.
# Gate at (top-record score >= 2.3 AND query-match-ratio >= 0.60) cuts T06
# and T12 to land at T05/T07/T13 — 3 of 27.
# Deadpath and registry paths are separate shapes with their own quality
# logic; this gate applies only to arc + factual-thread.
_M97_MIN_TOP_SCORE = 2.3
_M97_MIN_QUERY_MATCH_RATIO = 0.60
_CORPUS_DIR = str(Path(__file__).resolve().parent.parent.parent / "data" / "corpus_b_extractions")

# ── Intent classification (keyword v1) ──────────────────────────────────

_ARC_SIGNALS = re.compile(
    r"(?i)(where did we|what happened|how did|what.s the story|"
    r"what led to|trace the|walk me through|what changed|"
    r"what went wrong|why did we|how did .+ evolve|"
    r"what.s the history|summarize .+ across|"
    r"how did we characterize|how did we measure|"
    r"history of|evolution of)", re.IGNORECASE)

_FACTUAL_SIGNALS = re.compile(
    r"(?i)^(what is |what.s the |what are the |"
    r"what throughput |what speed |what latency |what bandwidth |"
    r"how (much|many|fast|big|long|far) |"
    r"list |define |show me the |tell me the )", re.IGNORECASE)


_DEADPATH_SIGNALS = re.compile(
    r"(?i)(dead path|dead.path|is .+ dead|is .+ a dead path|"
    r"didn.t we (kill|find .+ dead|abandon)|"
    r"should we (revisit|revive)|"
    r"is .+ still (viable|alive|worth)|is .+ killed)", re.IGNORECASE)


def _classify_intent(query: str) -> str:
    """Classify query as 'arc', 'factual', or 'deadpath'. Arc wins ties."""
    has_deadpath = bool(_DEADPATH_SIGNALS.search(query))
    if has_deadpath:
        return "deadpath"
    has_arc = bool(_ARC_SIGNALS.search(query))
    has_fact = bool(_FACTUAL_SIGNALS.search(query))
    if has_arc:
        return "arc"
    if has_fact:
        return "factual"
    # Default: if the query has past-tense verbs, lean arc
    past_tense = re.search(r"\b(did|was|were|happened|changed|evolved|went|led|found)\b", query, re.I)
    if past_tense:
        return "arc"
    return "arc"  # default to narrative when thread exists


# ── Dead-path lookup ────────────────────────────────────────────────────

_tag_index = None

def _load_tag_index():
    global _tag_index
    if _tag_index is not None:
        return _tag_index
    tag_path = Path(__file__).resolve().parent.parent.parent / "data" / "tag_index.json"
    if tag_path.exists():
        try:
            _tag_index = json.loads(tag_path.read_text())
        except Exception:
            _tag_index = {}
    else:
        _tag_index = {}
    return _tag_index


def _deadpath_lookup(query: str) -> str | None:
    """Check the tag index for dead-path entries matching query entities.

    Returns a formatted context block with authoritative kill status,
    or None if no dead-path entries match.
    """
    tags = _load_tag_index()
    dead_paths = tags.get("dead_path", [])
    if not dead_paths:
        return None

    ql = query.lower()
    # Extract entity-like terms from query
    query_terms = set(w.lower().strip("?.,!\"'") for w in query.split() if len(w) >= 2)
    # Common terms to skip
    skip = {"is", "it", "a", "the", "we", "did", "our", "path", "dead", "viable",
            "find", "didn", "still", "should", "try", "can", "use", "during",
            "research", "have", "been", "from", "with", "that", "this", "for"}
    entity_terms = query_terms - skip

    if not entity_terms:
        return None

    # Find matching dead-path entries
    matches = []
    for dp in dead_paths:
        content = dp.get("content", "").lower()
        entity = dp.get("entity", "").lower()
        # Check if any query entity term appears in the dead-path content or entity
        overlap = sum(1 for t in entity_terms if t in content or t in entity)
        if overlap > 0:
            matches.append((overlap, dp))

    if not matches:
        return None

    # Sort by overlap, take top 5
    matches.sort(key=lambda x: -x[0])
    top = [dp for _, dp in matches[:5]]

    lines = [
        "DEAD PATH — CONFIRMED KILLED. THIS IS NOT VIABLE.",
        "The following paths were tested, measured, and explicitly killed.",
        "Your answer MUST state that this is a dead path and explain why it was killed.",
        "Do NOT suggest it is viable, open, or worth revisiting.",
        ""
    ]
    for dp in top:
        sess = dp.get("session_label", "")
        content = dp.get("content", "")[:250]
        lines.append(f"  KILLED [{sess}]: {content}")

    return "\n".join(lines)


# ── Measurement registry ────────────────────────────────────────────────
#
# Main 38 P3: the registry is now auto-populated by a maintenance loop
# (vault/subconscious/maintenance.py auto_populate_measurement_registry).
# Loader is mtime-aware so running midas_ui picks up registry updates
# within a single session without a restart. Without this, a user-stated
# measurement in turn N would not surface until turn 1 of the next
# process, which misses the in-session feedback loop.

_measurement_registry = None
_measurement_registry_mtime = 0.0

def _load_registry():
    global _measurement_registry, _measurement_registry_mtime
    reg_path = Path(__file__).resolve().parent.parent.parent / "data" / "measurement_registry.json"
    if not reg_path.exists():
        _measurement_registry = {}
        _measurement_registry_mtime = 0.0
        return _measurement_registry
    try:
        current_mtime = reg_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    if (_measurement_registry is not None
            and current_mtime <= _measurement_registry_mtime):
        return _measurement_registry
    try:
        _measurement_registry = json.loads(reg_path.read_text())
        _measurement_registry_mtime = current_mtime
    except Exception:
        if _measurement_registry is None:
            _measurement_registry = {}
    return _measurement_registry


def _registry_lookup(query: str) -> str | None:
    """Check measurement registry for canonical values matching the query.

    Returns a formatted context block if matches found, None otherwise.
    """
    registry = _load_registry()
    if not registry:
        return None

    ql = query.lower()
    matches = []
    seen = set()

    for key, entry in registry.items():
        for alias in entry.get("aliases", []):
            if alias.lower() in ql:
                if key not in seen:
                    seen.add(key)
                    matches.append(entry)
                break

    if not matches:
        return None

    lines = ["CANONICAL MEASUREMENTS (verified, use these exact values):"]
    for m in matches:
        unit = f" {m['unit']}" if m.get('unit') else ""
        lines.append(f"  {m['entity']} {m['measurement_type']}: {m['value']}{unit} ({m['source']})")

    return "\n".join(lines)


# ── Factual context builder ─────────────────────────────────────────────

def _build_factual_context(thread: list[dict]) -> str:
    """For factual intent: extract the most recent state + correction provenance.

    Strategy:
      1. Find all correction records (they supersede earlier values)
      2. Take the most recent fact/decision record (current state)
      3. Combine: corrections first (provenance), then current value
      4. Keep it short — aim for 100-300 words
    """
    corrections = [r for r in thread if r.get("type") in ("correction", "decision")]
    facts = [r for r in thread if r.get("type") == "fact"]
    speculations = [r for r in thread if r.get("type") == "speculation"]

    # Thread is already chronologically sorted. Take from the end (most recent).
    recent_facts = facts[-3:] if facts else []
    recent_corrections = corrections[-3:] if corrections else []
    # Include one speculation if it's more recent than the latest fact
    recent_spec = []
    if speculations and facts:
        if thread.index(speculations[-1]) > thread.index(facts[-1]):
            recent_spec = [speculations[-1]]

    parts = []
    if recent_corrections:
        parts.append("Corrections/decisions (most recent):")
        for r in recent_corrections:
            sess = r.get("session_label", "")
            parts.append(f"  [{sess}] {r['content'][:300]}")

    if recent_facts:
        parts.append("\nCurrent state:")
        for r in recent_facts:
            sess = r.get("session_label", "")
            parts.append(f"  [{sess}] {r['content'][:300]}")

    if recent_spec:
        parts.append("\nOpen question:")
        for r in recent_spec:
            sess = r.get("session_label", "")
            parts.append(f"  [{sess}] {r['content'][:300]}")

    if not parts:
        # Fallback: just take the last 3 records
        for r in thread[-3:]:
            sess = r.get("session_label", "")
            parts.append(f"[{sess}, {r.get('type','')}] {r['content'][:300]}")

    return "\n".join(parts)


# ── Lazy imports ─────────────────────────────────────────────────────────

_thread_detector = None
_arc_tracer = None
_narrative_packager = None


def _ensure_imports():
    global _thread_detector, _arc_tracer, _narrative_packager
    if _thread_detector is None:
        from thread_detector import detect_thread
        from arc_tracer import trace_arc
        from narrative_packager import package_narrative
        _thread_detector = detect_thread
        _arc_tracer = trace_arc
        _narrative_packager = package_narrative


# ── Main entry point ─────────────────────────────────────────────────────

def try_narrative_context(query: str, corpus_dir: str = None,
                          max_records: int = _MAX_RECORDS,
                          min_records: int = _MIN_RECORDS,
                          min_sessions: int = _MIN_SESSIONS) -> dict | None:
    """Attempt narrative synthesis for a query.

    Returns dict with keys:
        narrative: str  — the context block for injection
        intent: str     — 'arc' or 'factual'
        n_records: int
        n_sessions: int
        arc_type: str   — only for arc intent
        latency_ms: float
    Or None if no qualifying thread found.
    """
    t0 = time.monotonic()
    try:
        intent = _classify_intent(query)
        elapsed_ms = lambda: round((time.monotonic() - t0) * 1000, 1)

        # Dead-path intent: check tag index FIRST (authoritative kill status)
        if intent == "deadpath":
            dp_block = _deadpath_lookup(query)
            if dp_block:
                return {
                    "narrative": dp_block,
                    "intent": "deadpath",
                    "n_records": 0,
                    "n_sessions": 0,
                    "arc_type": "deadpath",
                    "latency_ms": elapsed_ms(),
                }

        # Factual intent: check measurement registry FIRST (exact values)
        if intent == "factual":
            reg_block = _registry_lookup(query)
            if reg_block:
                # Registry match: check that the query isn't asking about
                # something the registry doesn't cover. If query has
                # significant terms not in any registry entry, skip.
                query_words = set(w.lower().strip("?.,!\"'") for w in query.split() if len(w) >= 4)
                stop_words = {"what", "does", "that", "this", "with", "from", "have", "been",
                              "about", "which", "where", "when", "many", "much", "measured"}
                query_specific = query_words - stop_words
                reg_lower = reg_block.lower()
                unmatched = [w for w in query_specific if w not in reg_lower]
                # If more than half the specific query terms aren't in the registry results,
                # the registry matched on entity but not on the actual question
                if len(unmatched) > len(query_specific) * 0.5 and len(query_specific) >= 2:
                    pass  # skip registry, fall through to thread detection
                else:
                    return {
                        "narrative": reg_block,
                        "intent": "factual_registry",
                        "n_records": 0,
                        "n_sessions": 0,
                        "arc_type": "factual",
                        "latency_ms": elapsed_ms(),
                    }

        _ensure_imports()
        corpus = corpus_dir or _CORPUS_DIR

        thread = _thread_detector(query, corpus, top_n=max_records)

        if len(thread) < min_records:
            return None

        sessions = set(r.get("session_label", "") for r in thread)
        if len(sessions) < min_sessions:
            return None

        # Topical relevance check: if the query contains significant terms
        # that appear in NONE of the thread records, the thread is entity-matched
        # but topically irrelevant. Return None to let the guard handle it.
        query_words = set(w.lower().strip("?.,!\"'") for w in query.split() if len(w) >= 4)
        stop_words = {"what", "does", "that", "this", "with", "from", "have", "been",
                      "about", "which", "where", "when", "many", "much", "happened"}
        query_specific = query_words - stop_words
        query_match_ratio = 1.0
        if query_specific:
            thread_text = " ".join(r.get("content", "") for r in thread).lower()
            unmatched = [w for w in query_specific if w not in thread_text]
            if len(unmatched) > len(query_specific) * 0.5 and len(query_specific) >= 2:
                return None  # topically irrelevant thread, let guard handle
            query_match_ratio = (len(query_specific) - len(unmatched)) / len(query_specific)

        # M97 Fix 1 — content-match gate. The top thread record's score
        # (entity-overlap + word-match boost) and the query-match ratio
        # together express how much of the query the narrative thread
        # actually addresses. Below threshold: thread is too weakly tied
        # to the query, fall through to default_recall. Calibrated on the
        # M92 baseline — see header comment.
        top_score = thread[0].get("score", 0) if thread else 0
        if top_score < _M97_MIN_TOP_SCORE or query_match_ratio < _M97_MIN_QUERY_MATCH_RATIO:
            return None

        if intent == "arc":
            arc = _arc_tracer(thread)
            narrative = _narrative_packager(arc)
            return {
                "narrative": narrative,
                "intent": "arc",
                "n_records": len(thread),
                "n_sessions": len(sessions),
                "arc_type": arc.get("arc_type", "unknown"),
                "latency_ms": elapsed_ms(),
            }
        else:
            # Factual: compressed context with most recent state + corrections
            context = _build_factual_context(thread)
            return {
                "narrative": context,
                "intent": "factual",
                "n_records": len(thread),
                "n_sessions": len(sessions),
                "arc_type": "factual",
                "latency_ms": elapsed_ms(),
            }
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"[narrative_retrieval] failed ({elapsed_ms:.0f}ms): {e}", flush=True)
        return None
