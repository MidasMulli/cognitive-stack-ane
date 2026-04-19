"""Phase 3 — Warm Session Open.

Loads profile + last-session bridge + pre-warms narrative cache on session start.
Everything is ready before the user types. No cold start.

Components:
  1. Profile loader: compressed profile block (~500 tokens)
  2. Last-session bridge: key records from most recent session (~300 tokens)
  3. Pre-loaded retrieval: top-N thread caches for likely first queries
  4. Profile weights: long-term topic weights for retrieval boosting
"""
import json
import os
import re
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_PROFILE_PATH = _ROOT / "data" / "user_profile_compiled.json"
_CORPUS_DIR = str(_ROOT / "data" / "corpus_b_extractions")
_THESIS = (
    "Subconscious is a continuous cognitive process, not a memory system. "
    "It runs perception, integration, consolidation, and recall preparation "
    "simultaneously with conscious generation, all the time, not in phases."
)

# ── 1. Profile Loader ───────────────────────────────────────────────────

def load_profile_block(path: str = None) -> str:
    """Load and compress user profile into a ~500-token context block."""
    p = Path(path) if path else _PROFILE_PATH
    if not p.exists():
        return ""
    try:
        profile = json.loads(p.read_text())
    except Exception:
        return ""

    parts = [f"COGNITIVE PROFILE (compiled from {profile.get('sessions_analyzed', '?')} sessions, "
             f"{profile.get('human_records', '?')} human-attributed records):"]

    # Thesis anchor
    parts.append(f"\nThesis: {_THESIS}")

    # Active research threads (by topic spread across sessions)
    spread = profile.get("topic_session_spread", {})
    if spread:
        top_threads = sorted(spread.items(), key=lambda kv: -kv[1])[:6]
        thread_strs = [f"{t} ({n}sess)" for t, n in top_threads]
        parts.append(f"\nActive threads: {', '.join(thread_strs)}")

    # Standing decisions (most recent, not superseded)
    decisions = profile.get("decisions", [])
    if decisions:
        # Take last 5 decisions (chronologically most recent in the list)
        recent_decs = decisions[-5:]
        parts.append("\nStanding decisions:")
        for d in recent_decs:
            if isinstance(d, dict):
                content = d.get("content", "")[:120]
                sess = d.get("session", "")
                parts.append(f"  [{sess}] {content}")
            else:
                parts.append(f"  {str(d)[:120]}")

    # Preferences
    prefs = profile.get("preferences", [])
    if prefs:
        parts.append("\nPreferences:")
        for p in prefs[-3:]:
            if isinstance(p, dict):
                parts.append(f"  {p.get('content', '')[:120]}")
            else:
                parts.append(f"  {str(p)[:120]}")

    # Methodology (from type distribution)
    type_dist = profile.get("type_distribution_human", {})
    if type_dist:
        total = sum(type_dist.values())
        if total:
            spec_pct = round(100 * type_dist.get("speculation", 0) / total)
            dec_pct = round(100 * type_dist.get("decision", 0) / total)
            corr_pct = round(100 * type_dist.get("correction", 0) / total)
            parts.append(f"\nMethodology: {spec_pct}% speculations, {dec_pct}% decisions, "
                         f"{corr_pct}% corrections in human turns. "
                         f"Direction by question, strategic-director not implementer.")

    block = "\n".join(parts)
    # Hard cap at ~500 tokens (~2000 chars)
    if len(block) > 2000:
        block = block[:1997] + "..."
    return block


# ── 2. Last-Session Bridge ──────────────────────────────────────────────

def _parse_session_date(label: str, session_name: str) -> str | None:
    """Extract YYYY-MM-DD date string from label or session_name.

    Handles patterns like:
      cc_cowork_2026_03_29_...  -> 2026-03-29
      cc_cowork_2026-03-29_...  -> 2026-03-29
      cc_home_2026-03-29_...    -> 2026-03-29
    Returns None if no date found.
    """
    for text in (label, session_name):
        if not text:
            continue
        # Try YYYY-MM-DD
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # Try YYYY_MM_DD
        m = re.search(r'(\d{4})_(\d{2})_(\d{2})', text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _is_interactive_session(session_name: str, label: str) -> bool:
    """Check if a session looks interactive (main/Main sessions)."""
    for text in (session_name, label):
        if text and re.search(r'\b[Mm]ain\b', text):
            return True
    return False


_SUMMARY_DIR = str(_ROOT / "data" / "session_summaries")


def _load_session_summary_bridge() -> str | None:
    """Check data/session_summaries/ for the most recent session summary.

    Returns a formatted bridge block if a recent summary exists, else None.
    Falls through to extraction-based bridge when no summary is available.
    """
    if not os.path.isdir(_SUMMARY_DIR):
        return None

    summaries = []
    for f in os.listdir(_SUMMARY_DIR):
        if not f.startswith("session_") or not f.endswith(".json"):
            continue
        fp = os.path.join(_SUMMARY_DIR, f)
        try:
            data = json.loads(open(fp).read())
            summaries.append((os.path.getmtime(fp), fp, data))
        except Exception:
            continue

    if not summaries:
        return None

    # Pick the most recent by file mtime
    summaries.sort(key=lambda x: x[0], reverse=True)
    _, path, data = summaries[0]

    ts = data.get("timestamp", "unknown")
    n_turns = data.get("duration_turns", 0)
    n_queries = data.get("n_total_queries", 0)
    last_topic = data.get("last_topic")
    queries = data.get("user_queries", [])

    if n_queries < 1:
        return None

    parts = [f"LAST SESSION ({ts}, {n_turns} turns):"]

    if last_topic:
        parts.append(f"  Discussed: {last_topic}")

    if queries:
        parts.append("  Key queries:")
        for q in queries[-5:]:
            # Truncate long queries for the bridge block
            q_short = q[:120].replace("\n", " ").strip()
            if q_short:
                parts.append(f'    - "{q_short}"')

    topic_weights = data.get("topic_weights")
    if topic_weights and isinstance(topic_weights, dict):
        # Show top 3 topics by weight
        sorted_topics = sorted(topic_weights.items(), key=lambda kv: -kv[1])[:3]
        topic_strs = [f"{t} ({w:.0%})" for t, w in sorted_topics if w > 0.05]
        if topic_strs:
            parts.append(f"  Active topics: {', '.join(topic_strs)}")

    if last_topic:
        parts.append(f"  Last active topic: {last_topic}")

    block = "\n".join(parts)
    # Hard cap at ~300 tokens (~1200 chars)
    if len(block) > 1200:
        block = block[:1197] + "..."

    print(f"[warm_open] bridge from session summary: {path} ({n_queries} queries)", flush=True)
    return block


def load_last_session_bridge(corpus_dir: str = None) -> str:
    """Find the most recent session and extract key records for warm handoff.

    Checks session summaries first (from midas_ui exit capture), then falls
    back to extraction-based bridge from corpus files.

    Sorts by session timestamp (from JSON metadata), not file mtime.
    Prefers interactive sessions (main/Main) when timestamps are within 24h.
    """
    # Try session summary bridge first (more recent, captures actual queries)
    summary_bridge = _load_session_summary_bridge()

    cdir = corpus_dir or _CORPUS_DIR

    # If no extraction corpus, return summary bridge or empty
    if not os.path.isdir(cdir):
        return summary_bridge or ""

    # Load metadata from each extraction file to find the truly most recent session
    files = []
    for f in os.listdir(cdir):
        if not f.endswith(".json") or "summary" in f or "config" in f:
            continue
        fp = os.path.join(cdir, f)
        try:
            data = json.loads(open(fp).read())
        except Exception:
            continue
        if not data.get("extractions") and not data.get("raw_chunks"):
            continue

        label = data.get("label", f.replace(".json", ""))
        session_name = data.get("session_name", "")
        created_at = data.get("created_at")  # explicit timestamp if present

        # Determine session date: prefer created_at, then parse from name, then file mtime
        sort_date = None
        date_source = "mtime"
        if created_at:
            # Handle ISO format or similar
            try:
                sort_date = created_at[:10]  # YYYY-MM-DD prefix
                date_source = "created_at"
            except Exception:
                pass
        if not sort_date:
            sort_date = _parse_session_date(label, session_name)
            if sort_date:
                date_source = "name_parsed"
        if not sort_date:
            # Last resort: file mtime
            mtime = os.path.getmtime(fp)
            sort_date = time.strftime("%Y-%m-%d", time.localtime(mtime))
            date_source = "mtime"

        interactive = _is_interactive_session(session_name, label)
        files.append((sort_date, interactive, fp, f, date_source, session_name))

    if not files:
        return summary_bridge or ""

    # Sort: primary by date descending, secondary by interactive flag (True > False)
    # This means among sessions on the same date, interactive ones win.
    # For sessions within 24h (same or adjacent dates), prefer interactive.
    files.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # If top result is not interactive but there's an interactive session within 24h, prefer it
    top = files[0]
    if not top[1]:  # not interactive
        for candidate in files[1:]:
            if not candidate[1]:  # also not interactive
                continue
            # Check if within ~24h (same date or one day apart)
            try:
                from datetime import date as dt_date
                top_d = dt_date.fromisoformat(top[0])
                cand_d = dt_date.fromisoformat(candidate[0])
                delta = abs((top_d - cand_d).days)
                if delta <= 1:
                    top = candidate
                    break
            except Exception:
                break

    latest_path = top[2]
    latest_name = top[3].replace(".json", "")
    _date_source = top[4]
    _session_name = top[5]
    print(f"[warm_open] bridge selected: {latest_name} "
          f"(date={top[0]}, source={_date_source}, "
          f"interactive={top[1]}, session={_session_name})", flush=True)

    try:
        data = json.loads(open(latest_path).read())
    except Exception:
        return ""

    exts = data.get("extractions", [])
    if not exts:
        return ""

    # Extract key records by type
    decisions = [r for r in exts if r.get("type") == "decision"]
    plans = [r for r in exts if r.get("type") == "plan"]
    corrections = [r for r in exts if r.get("type") == "correction"]
    questions = [r for r in exts if r.get("type") == "question"]
    speculations = [r for r in exts if r.get("type") == "speculation"]

    parts = [f"SINCE LAST SESSION ({latest_name}, {data.get('session_name', '')}):"]

    if decisions:
        parts.append("Decisions made:")
        for d in decisions[-3:]:
            parts.append(f"  - {d['content'][:150]}")

    if corrections:
        parts.append("Corrections applied:")
        for c in corrections[-2:]:
            parts.append(f"  - {c['content'][:150]}")

    if plans:
        parts.append("Plans stated:")
        for p in plans[-2:]:
            parts.append(f"  - {p['content'][:150]}")

    if questions:
        parts.append("Questions left open:")
        for q in questions[-2:]:
            parts.append(f"  - {q['content'][:150]}")

    if speculations and not decisions and not corrections:
        parts.append("Open speculations:")
        for s in speculations[-2:]:
            parts.append(f"  - {s['content'][:150]}")

    if len(parts) == 1:
        # Only header, no key records found
        parts.append(f"  {len(exts)} records extracted, mostly factual.")

    extraction_block = "\n".join(parts)
    # Hard cap at ~300 tokens (~1200 chars)
    if len(extraction_block) > 1200:
        extraction_block = extraction_block[:1197] + "..."

    # Prefer session summary if it exists and is more recent than extraction
    if summary_bridge:
        # Compare: session summary mtime vs extraction file date
        try:
            summary_files = sorted(
                [os.path.join(_SUMMARY_DIR, f) for f in os.listdir(_SUMMARY_DIR)
                 if f.startswith("session_") and f.endswith(".json")],
                key=os.path.getmtime, reverse=True
            )
            if summary_files:
                summary_mtime = os.path.getmtime(summary_files[0])
                extraction_mtime = os.path.getmtime(latest_path)
                if summary_mtime >= extraction_mtime:
                    return summary_bridge
        except Exception:
            pass

    return extraction_block


# ── 3. Pre-loaded Retrieval Cache ───────────────────────────────────────

_narrative_cache: dict = {}  # query_key -> narrative_result


def prewarm_narrative_cache(profile_block: str = "", n_topics: int = 5):
    """Pre-warm narrative pipeline for the top-N likely first queries.

    Uses profile's active threads to guess what the user will ask about.
    Cache is a dict mapping query strings to narrative results.
    """
    global _narrative_cache
    _narrative_cache.clear()

    try:
        from narrative_retrieval import try_narrative_context
    except ImportError:
        return

    # Extract topic keywords from profile block
    topics = []
    try:
        profile = json.loads(_PROFILE_PATH.read_text())
        spread = profile.get("topic_session_spread", {})
        topics = sorted(spread.keys(), key=lambda k: -spread[k])[:n_topics]
    except Exception:
        pass

    if not topics:
        return

    t0 = time.monotonic()
    for topic in topics:
        # Construct a likely arc query for each topic
        query = f"what happened with {topic}"
        result = try_narrative_context(query)
        if result:
            _narrative_cache[topic.lower()] = result

    elapsed = (time.monotonic() - t0) * 1000
    print(f"[warm_open] pre-warmed {len(_narrative_cache)} narrative caches "
          f"from {len(topics)} topics in {elapsed:.0f}ms", flush=True)


def get_cached_narrative(query: str) -> dict | None:
    """Check if a narrative is already cached for this query's topic."""
    if not _narrative_cache:
        return None
    ql = query.lower()
    for topic, result in _narrative_cache.items():
        if topic in ql:
            return result
    return None


# ── 4. Profile Weights for Retrieval ────────────────────────────────────

_profile_weights: dict = {}  # topic -> weight (0.0-0.30)


def load_profile_weights() -> dict:
    """Load long-term topic weights from profile for retrieval boosting.

    Returns dict mapping topic keywords to boost weights (0.0-0.30).
    Profile frequency determines weight: most frequent topic = 0.20,
    others proportional. Session context tracker moves on top (max wins).
    """
    global _profile_weights
    _profile_weights.clear()

    try:
        profile = json.loads(_PROFILE_PATH.read_text())
    except Exception:
        return _profile_weights

    spread = profile.get("topic_session_spread", {})
    if not spread:
        return _profile_weights

    max_count = max(spread.values())
    for topic, count in spread.items():
        # Scale: most frequent = 0.20, others proportional
        weight = round(0.20 * count / max(max_count, 1), 3)
        if weight >= 0.02:  # skip noise
            _profile_weights[topic.lower()] = weight

    return _profile_weights


def get_profile_boost(query: str) -> float:
    """Get profile-based boost for a query. Returns 0.0-0.20."""
    if not _profile_weights:
        return 0.0
    ql = query.lower()
    best = 0.0
    for topic, weight in _profile_weights.items():
        if topic in ql:
            best = max(best, weight)
    return best


# ── 5. Session Open Orchestrator ────────────────────────────────────────

_warm_state = {
    "profile_block": "",
    "bridge_block": "",
    "loaded_at": None,
    "profile_weights": {},
}


def session_open() -> dict:
    """Run the full warm-open sequence. Call once on midas_ui start or session reset.

    Returns the warm state dict with profile_block, bridge_block, profile_weights.
    """
    t0 = time.monotonic()

    _warm_state["profile_block"] = load_profile_block()
    _warm_state["bridge_block"] = load_last_session_bridge()
    _warm_state["profile_weights"] = load_profile_weights()
    _warm_state["loaded_at"] = time.time()

    # Pre-warm narrative cache (runs thread detection for top topics)
    prewarm_narrative_cache(_warm_state["profile_block"])

    elapsed = (time.monotonic() - t0) * 1000
    print(f"[warm_open] session open complete in {elapsed:.0f}ms", flush=True)
    print(f"[warm_open] profile: {len(_warm_state['profile_block'])} chars", flush=True)
    print(f"[warm_open] bridge: {len(_warm_state['bridge_block'])} chars", flush=True)
    print(f"[warm_open] weights: {len(_warm_state['profile_weights'])} topics", flush=True)
    print(f"[warm_open] cache: {len(_narrative_cache)} pre-warmed threads", flush=True)

    return _warm_state


def get_warm_context() -> str:
    """Get the combined profile + bridge block for system prompt injection.

    Returns a single string, or empty string if no warm state loaded.
    """
    parts = []
    if _warm_state["profile_block"]:
        parts.append(_warm_state["profile_block"])
    if _warm_state["bridge_block"]:
        parts.append(_warm_state["bridge_block"])
    return "\n\n".join(parts)


def get_session_context_extension() -> dict:
    """Extension data for /api/session/context endpoint."""
    return {
        "profile_loaded": bool(_warm_state["profile_block"]),
        "profile_chars": len(_warm_state["profile_block"]),
        "bridge_loaded": bool(_warm_state["bridge_block"]),
        "bridge_chars": len(_warm_state["bridge_block"]),
        "profile_weights_n": len(_warm_state.get("profile_weights", {})),
        "narrative_cache_n": len(_narrative_cache),
        "loaded_at": _warm_state.get("loaded_at"),
    }
