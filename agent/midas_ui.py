#!/usr/bin/env python3
"""
Midas Agent — Web UI
Single-file Flask app with embedded HTML/CSS/JS.
Operations console aesthetic. Dark, monospace, terminal-native.
"""

import atexit
import io
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.request
import urllib.error
import warnings
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response

# ── Suppress noisy loggers ──────────────────────────────────────────────────

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
for noisy in ("httpx", "httpcore", "openai", "sentence_transformers",
              "chromadb", "huggingface_hub", "werkzeug"):
    lg = logging.getLogger(noisy)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False

# ── Boot memory silently ────────────────────────────────────────────────────

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))

from memory_bridge import MemoryBridge
from router import route, layer1_route
from tool_executor import execute, set_memory, set_browser
from synthesizer import synthesize
from idle_queue import IdleQueue
import research_tools
from query_type_dispatch import dispatch as _query_type_dispatch
# m125_a2_shape_precedence — canonical-lookup classifier for narrative
# suppression. See shape_precedence.py module docstring for the authoritative
# spec. K6-safe: negative patterns gate FIRST, then M122 A4 reuse, then
# A2-local positives. Under-fit by design.
from shape_precedence import (
    is_canonical_lookup as _m125_a2_is_canonical_lookup,
    DISPATCH_DECISION_NARRATIVE_PRIMARY as _M125_A2_DD_NARRATIVE,
    DISPATCH_DECISION_RECALL_PRIMARY_CANONICAL_LOOKUP as _M125_A2_DD_RECALL_PRIMARY,
    DISPATCH_DECISION_NARRATIVE_SUPPRESSED_CLASSIFIER as _M125_A2_DD_NARRATIVE_SUPPRESSED,
    DISPATCH_DECISION_OTHER as _M125_A2_DD_OTHER,
)

# m118_e_sub_shape_2 — anaphor resolution at retrieval layer. Stream D
# shipped anaphor_resolver.resolve_antecedent; wire it into the two recall
# sites so demonstrative follow-ups ("what was the purpose of this
# classifier?", "is that all?") carry the antecedent phrase into the
# embedding query. Without this the bare demonstrative embeds to ~0 cosine
# against every stored memory and absence gate fires (M117 T9/T18).
try:
    from anaphor_resolver import resolve_antecedent as _anaphor_resolve_antecedent
except Exception:
    _anaphor_resolve_antecedent = None


def _m118_e_expand_query_with_antecedent(message: str, history: list) -> str:
    """Return a query string with the resolved antecedent prepended when
    a demonstrative / 3p pronoun is present and the prior assistant turn
    yields a confident referent. Falls back to the original message when
    no resolver, no prior turn, or no match. Never shortens the query —
    only additive.

    Called by both recall call-sites (/api/chat and streaming). Keeps the
    extension inside the retrieval layer so downstream scrub/grounding
    logic sees the resolved text and not just "this".
    """
    if _anaphor_resolve_antecedent is None or not message:
        return message
    prior_assistant = ""
    for _h in reversed(history or []):
        if _h.get("role") == "assistant":
            prior_assistant = _h.get("content", "") or ""
            break
    if not prior_assistant:
        return message
    try:
        res = _anaphor_resolve_antecedent(message, prior_assistant)
    except Exception:
        return message
    ant = res.get("antecedent_entity")
    conf = res.get("confidence")
    if not ant or conf == "none":
        return message
    # Accept both "high" and "medium" — the retrieval embedding is robust
    # to an extra phrase; the cost of a false antecedent is a slightly
    # softer cosine, the cost of missing a true antecedent is 0.0 recall
    # and an unnecessary absence-gate fire.
    if ant.lower() in message.lower():
        return message
    return f"{message} {ant}"

# ── Config ──────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "/Users/midas/models/gemma-4-31b-it-4bit")
MAX_HISTORY = 24
PORT = 8450

# ── Singletons ──────────────────────────────────────────────────────────────

memory = MemoryBridge()
_last_stats = {}
_session = {
    "messages_sent": 0,
    "memories_recalled": 0,
    "facts_extracted": 0,
    "tools_used": 0,
    "start_time": datetime.now().isoformat(),
}
_feed = []  # last 20 memory events
_last_subconscious = []  # memories injected on last turn
_history = []  # conversation history
_prev_memory_store_total = None  # M94: track store total for per-turn delta
_all_user_queries = []  # M94: untrimmed list of all user queries for session summary
# M125 A5.1: prior-turn tool stash for anaphoric fold-in across tools.
# Updated after each chat turn with the tool_name that was actually
# dispatched. Used by router.route()/layer1_route() on the next turn
# to re-fire time-sensitive tools (browse_x_feed, browse_search) on
# subject-continuity follow-ups and to fold prior subject into
# memory_recall/vault_read/vault_research on anaphoric follow-ups.
_last_tool = ""

# Main 25 Build 0: stable per-session briefing for prompt cache hit rate.
# Built once at session start (or every N=5 turns), reused across turns so
# the system message stays byte-stable and the verifier's prefix KV cache
# hits on every subsequent turn. Per-query memories ride in the user message
# (tail) so they don't invalidate the prefix.
_session_briefing = None
_session_briefing_built_at_turn = -1
_SESSION_BRIEFING_REFRESH_EVERY = 5
_lock = threading.Lock()
_idle_queue = None  # Initialized after memory.start()

# Main 35 +5 Tier 1: live ContextTracker singleton.
# The tracker built in S+1 (multi-topic + brief-query guard, S+4) was
# only used by smoke tests until now. Wire it into the live chat path
# so its state reflects real conversations and can be exposed via
# /api/session/context for the viz to render the active-topic HUD.
_context_tracker = None
def _get_context_tracker():
    global _context_tracker
    if _context_tracker is None:
        try:
            import sys as _sys
            _sys.path.insert(0, "/Users/midas/Desktop/cowork/orion-ane/agent")
            from context_tracker import ContextTracker
            _context_tracker = ContextTracker()
            _context_tracker.on_session_start()
        except Exception as e:
            print(f"context tracker init failed: {e}")
            _context_tracker = None
    return _context_tracker

# ── Session Bridge: summary capture on exit ───────────────────────────────

_SESSION_OPEN_TS = time.time()

# Regex patterns for extracting decisions, corrections, and standing rules
# from the user's own messages. These fire only on user text (never assistant
# output) so the summary captures what the user actually said, not what the
# model produced.
_DECISION_RE = re.compile(
    r"(?:\blet'?s\s+(?:go|do|ship|use|try|build|start|move|focus)\b"
    r"|\bwe(?:'?ll| will| should| are going to| gonna)\s+\w+"
    r"|\bdecided?\s*:?\s*\w+"
    r"|\bgoing with\b"
    r"|\bship(?:ping)?\s+(?:it|this|that)\b"
    r"|\bmerge\s+(?:it|this)\b)",
    re.IGNORECASE,
)
_CORRECTION_RE = re.compile(
    r"(?:\bno[,\s]+(?:don'?t|not|that'?s|it'?s|wrong)"
    r"|\bactually\s*,?\s+\w+"
    r"|\bwrong\b"
    r"|\bcorrect(?:ion)?\s*:"
    r"|\bthat'?s\s+not\s+(?:right|correct)"
    r"|\bstop\s+(?:doing|using|saying)\b"
    r"|\brevert\b)",
    re.IGNORECASE,
)
_RULE_RE = re.compile(
    # Main 38 Session 3: fast-path only. This regex catches obvious,
    # unambiguous rule statements. Anything ambiguous (preference vs
    # conversational remark) is a semantic classification problem that
    # regex cannot solve reliably. Two prior Main 38 regex attempts
    # either missed real preferences ("remember to limit verbosity")
    # or produced false positives ("i don't mind"). The architectural
    # fix is a 72B classification call for ambiguous cases; this
    # regex handles the obvious ones at zero cost. See state synthesis
    # Section 17 for the deferred architectural work.
    #
    # Key tightening in this revision:
    #   - added `remember to <verb>` so "remember to limit verbosity" matches
    #   - generalized keep-noun-adjective so any noun between keep and the
    #     adjective works ("keep answers concise", "keep responses short")
    #   - restricted `don'?t` to a closed set of imperative verbs so
    #     "i don't mind" / "i don't know" / "i don't think" no longer
    #     trigger. Trade-off: new verbs ("don't paginate") are missed,
    #     but that's what the deferred 72B classifier is for.
    r"(?:\bstanding\s+rule\b"
    r"|\bnew\s+rule\b"
    r"|\bfrom now on\b"
    r"|\balways\s+\w+"
    r"|\bnever\s+\w+"
    r"|\brule\s*:\s*\w+"
    r"|\bremember\s+(?:that\s+)?(?:this\s+is\s+)?important"
    r"|\bremember\s+to\s+\w+"
    r"|\bi\s+(?:like|prefer|want)\s+\w+"
    r"|\bkeep\s+\w+\s+(?:short|shorter|concise|brief|briefer|tight|tighter|simple|simpler|terse|minimal)"
    r"|\bmake\s+sure\s+(?:to|you)\s+\w+"
    r"|\bplease\s+always\s+\w+"
    r"|\bplease\s+never\s+\w+"
    r"|\bdon'?t\s+(?:ever\s+)?(?:say|do|use|cite|suggest|revive|claim|mention|pad|make|propose|assume|override|revert|restart|kill|delete|forget|add|waste|repeat|enumerate|list|bullet|number|apologize|defer|promise|fabricate)\b)",
    re.IGNORECASE,
)


# Main 39 P1: ambiguous rule candidate detection. The fast-path _RULE_RE
# above catches obvious unambiguous statements at zero cost. Anything that
# trips one of these candidate words but does NOT match the fast path is
# ambiguous and routes to the 72B classifier at session close. Two prior
# Main 38 sessions burned regex iterations on this surface — see state
# synthesis 17.1 for why semantic classification is the architectural fix.
_RULE_CANDIDATE_RE = re.compile(
    r"\b(?:remember|keep|prefer|like|don'?t|do not|make\s+sure|important|"
    r"please|stop|avoid|use|don'?t\s+ever)\b",
    re.IGNORECASE,
)

# Cap on how many ambiguous messages we'll classify per session close.
# 72B classification at ~500ms each batched into one call is still
# bounded so we don't blow out the close path on a 100-turn session.
_RULE_CLASSIFY_MAX = 30


def _classify_ambiguous_rules_with_72b(ambiguous_msgs):
    """Batch-classify ambiguous user messages as RULE or REMARK.

    Single 72B call with a numbered list. Returns the subset of input
    messages classified as RULE, in original order. Empty list on any
    failure (network, parse, model offline).
    """
    if not ambiguous_msgs:
        return []
    msgs = ambiguous_msgs[:_RULE_CLASSIFY_MAX]
    numbered = "\n".join(f"{i+1}. {m[:240]}" for i, m in enumerate(msgs))
    sys = (
        "You classify user messages from a chat session as either RULE or "
        "REMARK.\n\n"
        "RULE = a durable behavioral preference, instruction, or standing "
        "rule the user wants applied across this session and future sessions. "
        'Examples: "keep responses short", "always cite sources", "remember '
        'to use Q8 on ANE", "stop offering menus".\n\n'
        "REMARK = a one-off conversational comment that does not establish "
        "any rule. Examples: \"I don't mind if it's not related\", "
        '"remember when we tried that last week", "I like how that turned '
        'out", "important meeting tomorrow".\n\n'
        "Output exactly one line per input, in the form `<number>: RULE` or "
        "`<number>: REMARK`. No explanation, no extra text, no markdown."
    )
    user = f"Classify each message:\n\n{numbered}"
    classified_rules = []
    try:
        out = llm_route_fn(
            [{"role": "system", "content": sys},
             {"role": "user", "content": user}],
            max_tokens=8 * len(msgs) + 16,
            temperature=0.0,
        )
        # Parse "N: RULE" or "N: REMARK" lines
        for line in (out or "").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[:.\)]\s*(RULE|REMARK)\b", line, re.I)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            label = m.group(2).upper()
            if 0 <= idx < len(msgs) and label == "RULE":
                classified_rules.append(msgs[idx])
    except Exception as e:
        print(f"[rule-classify] 72B classification failed: {e}", flush=True)
        return []
    return classified_rules


def _extract_user_signals(user_msgs):
    """Return (decisions, corrections, rules) from user message texts."""
    decisions, corrections, rules = [], [], []
    for msg in user_msgs:
        text = (msg or "").strip()
        if not text:
            continue
        snippet = text[:240]
        if _DECISION_RE.search(text):
            decisions.append(snippet)
        if _CORRECTION_RE.search(text):
            corrections.append(snippet)
        if _RULE_RE.search(text):
            rules.append(snippet)
    # Dedup while preserving order
    def _uniq(seq):
        seen, out = set(), []
        for x in seq:
            k = x[:120]
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out
    return _uniq(decisions), _uniq(corrections), _uniq(rules)


def _write_session_summary():
    """Capture session summary on exit for bridge continuity.

    Writes data/session_summaries/session_TIMESTAMP.json with:
      - start_ts / end_ts / duration_minutes / duration_turns
      - user_queries (last 10)
      - last_topic + active_topics (top 5) from ContextTracker
      - decisions / corrections / standing_rules extracted from user text
      - last_subconscious (debug tail: what the bridge last surfaced)
    """
    if not _history or len(_history) < 2:
        return  # no conversation to summarize

    summary_dir = "/Users/midas/Desktop/cowork/data/session_summaries"
    os.makedirs(summary_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # M94: use _all_user_queries (untrimmed) instead of extracting from _history
    user_texts = list(_all_user_queries)
    # Truncated for persistence
    user_queries = [t[:200] for t in user_texts]

    decisions, corrections, rules = _extract_user_signals(user_texts)
    fast_path_rule_count = len(rules)

    # Main 39 P1: 72B semantic classification of ambiguous candidates.
    # Anything that contains a rule-candidate signal word but did NOT
    # match the fast-path regex is sent to the 72B for one-shot
    # classification. RULE results are appended to the standing rules
    # list and flow through the same persistence path as regex matches.
    classified_rule_count = 0
    try:
        regex_match_set = {(t or "").strip() for t in user_texts
                           if _RULE_RE.search(t or "")}
        ambiguous = []
        for t in user_texts:
            text = (t or "").strip()
            if not text or text in regex_match_set:
                continue
            if _RULE_CANDIDATE_RE.search(text):
                ambiguous.append(text)
        if ambiguous:
            classified = _classify_ambiguous_rules_with_72b(ambiguous)
            if classified:
                seen = {r[:120] for r in rules}
                for c in classified:
                    if c[:120] not in seen:
                        rules.append(c[:240])
                        seen.add(c[:120])
                        classified_rule_count += 1
    except Exception as e:
        print(f"[rule-classify] session-close pass failed: {e}", flush=True)

    # Context tracker state: CORRECT keys are "active_topic" and
    # "topic_weights". Earlier this block used "dominant_topic" and
    # "topics" which silently returned None. (Fixed Main 37.)
    tracker_state = None
    try:
        tracker = _get_context_tracker()
        if tracker:
            tracker_state = tracker.state()
    except Exception:
        pass

    last_topic = tracker_state.get("active_topic") if tracker_state else None
    topic_weights = tracker_state.get("topic_weights") if tracker_state else None
    active_topics = tracker_state.get("active_topics") if tracker_state else None

    end_ts = time.time()
    duration_s = max(0.0, end_ts - _SESSION_OPEN_TS)

    summary = {
        "start_ts": _SESSION_OPEN_TS,
        "end_ts": end_ts,
        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                   time.localtime(_SESSION_OPEN_TS)),
        "end_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                 time.localtime(end_ts)),
        "duration_minutes": round(duration_s / 60.0, 1),
        "duration_turns": len(_all_user_queries),
        "user_queries": user_queries[-10:],
        "n_total_queries": len(user_queries),
        "last_topic": last_topic,
        "active_topics": active_topics,
        "topic_weights": topic_weights,
        "decisions": decisions[:10],
        "corrections": corrections[:10],
        "standing_rules": rules[:10],
        "last_subconscious": [
            m.get("text", "")[:100] for m in _last_subconscious[:5]
        ] if _last_subconscious else [],
    }

    path = os.path.join(summary_dir, f"session_{ts}.json")
    try:
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[session] summary saved: {path} "
            f"({len(user_queries)} queries, {len(decisions)} decisions, "
            f"{len(corrections)} corrections, {len(rules)} rules "
            f"[regex={fast_path_rule_count} 72b={classified_rule_count}])",
            flush=True,
        )
    except Exception as e:
        print(f"[session] summary write failed: {e}", flush=True)

atexit.register(_write_session_summary)


# Main 37: atexit hooks only fire on normal Python shutdown. SIGTERM
# (kill, launchd stop) bypasses them, which means the session summary
# never gets written in most real-world exits. Install signal handlers
# that call sys.exit(0) so atexit runs in those cases too.
def _sigterm_handler(signum, frame):
    print(f"[session] received signal {signum}, writing summary and exiting",
          flush=True)
    sys.exit(0)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_sig, _sigterm_handler)
    except (ValueError, OSError):
        pass  # some signals unavailable in non-main threads


# ── Session Bridge: load most recent summary for continuity ──────────────
#
# On session open we load the most recent session_summaries/*.json and
# format it into a compact "PRIOR SESSION" block that rides at the top of
# the system briefing on message #1. This is what lets the user ask "what
# did we talk about last" and get an answer grounded in the actual prior
# conversation, not just extraction-derived memories.

_SESSION_SUMMARIES_DIR = "/Users/midas/Desktop/cowork/data/session_summaries"


def load_last_session_summary():
    """Return the most recent session summary dict, or None.

    Public (no leading underscore) so router.recency routing can reuse it.
    """
    return load_nth_session_summary(1)


# M125.1 Stream B — recency-ordinal N-back handler.
# Pilot T22/T23/T24 (sess_20260422_203807_1385) showed the recency
# short-circuit returning the latest session summary for every query,
# ignoring ordinal N-back references: "three sessions ago", "two
# sessions before last", "10 sessions ago". Sorted session summaries
# already exist on disk (~/Desktop/cowork/data/session_summaries/),
# indexed by mtime-desc — "N sessions ago" == candidates[N-1]. Fix
# shape is detection + index, not new storage. Under-fit scope:
# "sessions ago"-only; "turns ago" (within-session ordinal) is a
# different data store (conversation history) and is deferred.
_ORDINAL_WORD_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# Match: "N sessions ago", "N sessions back", "N sessions before last".
# N may be digits (1-99) OR a spelled word from _ORDINAL_WORD_MAP.
_RECENCY_ORDINAL_NBACK_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+sessions?\s+(?:ago|back|before\s+last)\b",
    re.IGNORECASE,
)

# M125.2 Stream D — turn-scope ordinal N-back.
# Within-session turn references: "N turns ago", "N turns back",
# "N messages ago", "N messages back", "N exchanges ago",
# "N exchanges back". Under-fit discipline: require an explicit
# ordinal marker (digit or spelled 1-10) AND one of the turn-level
# scope nouns. "earlier in session" / "before that" / "previous
# response" are deliberately NOT matched here — M125.1 B K15
# ambiguous-form deferral to M125.3. A5.1 anaphoric resolver
# continues to own subject-continuity follow-ups that may
# incidentally mention "turns" without an ordinal marker.
_TURN_ORDINAL_NBACK_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+(?:turns?|messages?|exchanges?)\s+(?:ago|back|before\s+last)\b",
    re.IGNORECASE,
)


def _parse_ordinal_nback(text: str):
    """Classify ordinal N-back references into (N, scope).

    Returns:
      - (N, 'turn') for "N turns ago" / "N messages back" /
        "N exchanges ago" (within-session, M125.2 Stream D).
      - (N, 'session') for "N sessions ago" / "N sessions back" /
        "N sessions before last" (cross-session, M125.1 Stream B).
      - None when no ordinal marker is present.

    Turn-scope is checked FIRST so a query carrying both "turns" and
    "sessions" (which should be rare; under-fit discipline) routes to
    the more specific within-session accessor. Scope is an explicit
    return value so dispatch sites route to the correct loader.
    """
    if not text:
        return None
    # M125.2 Stream D: turn-scope first. Under-fit: explicit ordinal
    # marker required so anaphoric queries that mention "turns" without
    # a count (e.g. "across several turns") stay with A5.1.
    m_turn = _TURN_ORDINAL_NBACK_RE.search(text)
    if m_turn:
        raw = m_turn.group(1).lower()
        if raw.isdigit():
            n = int(raw)
        else:
            n = _ORDINAL_WORD_MAP.get(raw, 0)
        if n >= 1:
            return (n, "turn")
    m_sess = _RECENCY_ORDINAL_NBACK_RE.search(text)
    if m_sess:
        raw = m_sess.group(1).lower()
        if raw.isdigit():
            n = int(raw)
        else:
            n = _ORDINAL_WORD_MAP.get(raw, 0)
        if n >= 1:
            return (n, "session")
    return None


def load_nth_session_summary(n: int = 1):
    """Return the Nth-most-recent session summary (1 = latest, 2 =
    previous, etc.), or None if N exceeds available history. Shared
    impl for load_last_session_summary (N=1) and M125.1 ordinal
    N-back handler.
    """
    try:
        if not os.path.isdir(_SESSION_SUMMARIES_DIR):
            return None
        candidates = [
            os.path.join(_SESSION_SUMMARIES_DIR, f)
            for f in os.listdir(_SESSION_SUMMARIES_DIR)
            if f.startswith("session_") and f.endswith(".json")
        ]
        if not candidates or n < 1 or n > len(candidates):
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        with open(candidates[n - 1]) as f:
            summary = json.load(f)
        summary["_path"] = candidates[n - 1]
        summary["_nback"] = n
        summary["_total_available"] = len(candidates)
        return summary
    except Exception as e:
        print(f"[session] load_nth_session_summary({n}) failed: {e}",
              flush=True)
        return None


def load_nth_turn(n: int, session_log_dir: str = None):
    """Return the Nth-most-recent COMPLETED turn record from the current
    session's on-disk turn log, or None if N exceeds available history.

    Canonical store:
      data/session_logs/{session_id}/turn_NNNN.json

    Indexing convention (M125.2 Stream D):
      N=1 → the most recent completed turn (written to disk before the
            current in-flight turn began).
      N=2 → two turns ago.
      Etc.

    The in-flight turn itself is excluded: _current_turn has not been
    written to disk yet at short-circuit time, and "0 turns ago" is not
    a useful query boundary.

    Returns a dict shaped for _build_recency_response to render. Keys
    (all optional; callers handle None gracefully):
      _nback           : int — the N that was requested
      _total_available : int — count of on-disk turn JSONs
      _path            : str — absolute path of the turn JSON loaded
      turn_number      : int — 1-based turn index within session
      user_query       : str — the user message at that turn
      assistant_response : str — the response the system returned
      tools_called     : list — L0/L1/L2 tools dispatched for that turn
      route_layer      : str — L0/L1/L2
      shape_fired      : str — retrieval shape fired, if any
      recall_filtered  : list — recall_filtered snapshot, if any
    """
    try:
        if session_log_dir is None:
            session_log_dir = _SESSION_LOG_DIR
        if not session_log_dir or not os.path.isdir(session_log_dir):
            return None
        candidates = [
            os.path.join(session_log_dir, f)
            for f in os.listdir(session_log_dir)
            if f.startswith("turn_") and f.endswith(".json")
        ]
        if not candidates or n < 1 or n > len(candidates):
            return None
        # Sort by the 4-digit turn number in the filename so we get
        # a deterministic order independent of mtime races. N=1 is
        # the HIGHEST turn number on disk (most recent completed).
        def _turn_num(path):
            try:
                return int(os.path.basename(path)[5:9])
            except Exception:
                return -1
        candidates.sort(key=_turn_num, reverse=True)
        chosen = candidates[n - 1]
        with open(chosen) as f:
            raw = json.load(f)
        record = {
            "_nback": n,
            "_total_available": len(candidates),
            "_path": chosen,
            "turn_number": raw.get("turn_number"),
            "user_query": (raw.get("input") or {}).get("query", ""),
            "assistant_response": (raw.get("generation") or {}).get(
                "response_text", "") or "",
            "tools_called": (raw.get("routing") or {}).get(
                "tools_called", []) or [],
            "route_layer": (raw.get("routing") or {}).get(
                "route_layer", ""),
            "shape_fired": (raw.get("retrieval") or {}).get(
                "shape_fired", ""),
            "recall_filtered": (raw.get("retrieval") or {}).get(
                "recall_filtered", []) or [],
        }
        return record
    except Exception as e:
        print(f"[session] load_nth_turn({n}) failed: {e}", flush=True)
        return None


def format_last_session_block(summary=None, max_chars=1200):
    """Format a prior-session summary as a compact text block.

    Pattern: "Last session (DATE, DURATION): discussed [topics]. Key
    exchanges: [2-3 queries]. Decisions: [...]. Corrections: [...].
    Last active topic: [X]."
    """
    if summary is None:
        summary = load_last_session_summary()
    if not summary:
        return None

    date_str = summary.get("end_iso") or summary.get("start_iso") or "?"
    if isinstance(date_str, str) and "T" in date_str:
        date_str = date_str.replace("T", " ")[:16]

    duration_m = summary.get("duration_minutes")
    turns = summary.get("duration_turns") or summary.get("n_total_queries") or 0
    if duration_m and duration_m > 0:
        duration_str = f"{duration_m:.0f} min, {turns} turns"
    else:
        duration_str = f"{turns} turns"

    topics_list = summary.get("active_topics") or []
    if topics_list:
        topic_names = []
        for t in topics_list[:5]:
            if isinstance(t, dict):
                topic_names.append(t.get("topic") or "")
            elif isinstance(t, (list, tuple)) and t:
                topic_names.append(str(t[0]))
            else:
                topic_names.append(str(t))
        topic_names = [t for t in topic_names if t]
    else:
        last = summary.get("last_topic")
        topic_names = [last] if last else []

    topics_str = ", ".join(topic_names) if topic_names else "unclassified"

    queries = summary.get("user_queries") or []
    # Prefer the most substantive queries (longest, up to 3) since the
    # final few are often "thanks" / "exit" / short acknowledgements.
    ranked = sorted(queries, key=len, reverse=True)[:3]
    if not ranked:
        key_exchanges = "none"
    else:
        key_exchanges = " | ".join(f'"{q.strip()[:140]}"' for q in ranked)

    decisions = summary.get("decisions") or []
    corrections = summary.get("corrections") or []
    # Main 40 P2: standing rules are NOT enumerated in the briefing
    # block. Section 17.7 finding: rules in briefing context made the
    # 72B treat them as elaboration material rather than constraints,
    # producing 3× longer responses than baseline. The system-message
    # directive layer (P1.5) is the authoritative location for rules.
    # Recency-query tool responses (`_build_recency_response`) still
    # enumerate them because that path only fires when the user
    # explicitly asks about the prior session.

    def _join(lst, n=3, maxlen=120):
        if not lst:
            return "none"
        return "; ".join(s.strip()[:maxlen] for s in lst[:n])

    last_topic = summary.get("last_topic") or "none"

    block = (
        f"Last session ({date_str}, {duration_str}): discussed {topics_str}. "
        f"Key exchanges: {key_exchanges}. "
        f"Decisions: {_join(decisions)}. "
        f"Corrections: {_join(corrections)}. "
        f"Last active topic: {last_topic}."
    )
    if len(block) > max_chars:
        block = block[:max_chars - 3] + "..."
    return block


# Recency-query detection. Queries matching these patterns route DIRECTLY
# to the session summary file and never touch narrative retrieval. The user
# asking "what did we talk about last" should not wait on embedding search,
# scoring, or LLM synthesis. They should get the prior session block verbatim.
_RECENCY_QUERY_RE = re.compile(
    # Main 37 Fix 1: added chat(?:ted|ting)? so "what did we chat about last"
    # triggers the short-circuit. Previously only talk/discuss matched and
    # the bridge fell through to the full pipeline which lost the signal.
    # M121 Stream C: added bare-phatic coverage — "where were we", "what
    # were we working on", "what were we doing" — that previously fell
    # through the regex entirely. M120 B conjunction check still decides
    # whether a matched phatic short-circuits vs falls through to L2.
    # M125.2 Stream C: added "what happened" + explicit-ordinal-session
    # marker alternation. Closes M125.1 B Q4 replay gap: "what happened
    # two sessions before last" reaches _parse_ordinal_nback downstream.
    # Under-fit discipline (K8): "what happened" alone, or "what happened
    # at Main 42" (no ordinal marker), MUST NOT match here — those are
    # factual-recall shapes handled by L2. Required ordinal markers mirror
    # _RECENCY_ORDINAL_NBACK_RE: (digits|spelled-1-10) + sessions? +
    # (ago|back|before last). K9 ("what did we cover / what was the focus"
    # variants) deferred to M125.3 per M125.2 directive §3.3 scope.
    r"(?:what\s+(?:did\s+we|were\s+we|have\s+we)\s+(?:talk(?:ed|ing)?|discuss(?:ed|ing)?|chat(?:ted|ting)?)"
    r"|what\s+was\s+(?:our\s+)?last\s+(?:session|conversation|chat)"
    r"|last\s+session"
    r"|previous\s+(?:session|conversation|chat|discussion)"
    r"|what\s+did\s+we\s+do\s+last"
    r"|where\s+did\s+we\s+leave\s+off"
    r"|where\s+(?:were|was)\s+we"  # m121_c_bare_phatic: "where were we?" / "where were we on X"
    r"|what\s+(?:were|was)\s+we\s+(?:working\s+on|doing)"  # m121_c_bare_phatic: un-last-suffixed working/doing
    r"|what.{0,8}(?:was|were)\s+we\s+(?:working\s+on|doing|discussing|chatting\s+about)\s+last"
    r"|what\s+happened\s+(?:.{0,30}?\s+)?"  # m125_2_c: "what happened" with optional short bridge...
    r"(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"  # ...N (digit 1-99 or spelled 1-10)...
    r"\s+sessions?\s+(?:ago|back|before\s+last))",  # ...+ sessions? + ago|back|before last
    re.IGNORECASE,
)


def _is_recency_query(text: str) -> bool:
    if not text:
        return False
    return bool(_RECENCY_QUERY_RE.search(text))


# m120_b_phatic_anchor_check
# M120 Stream B — T84 (M119 routing_phatic_anchor_miss).
#
# The L0 phatic recency short-circuit was firing on queries like
# "so where did we leave off on the direct IOKit loading thread?" —
# the phatic head "where did we leave off" captured the query, ran
# session_summary_recall, and never let L2 see the specific topical
# anchor "the direct IOKit loading thread". Operator got a templated
# prior-session block unrelated to what they asked about.
#
# Fix: if a phatic-recency query also carries a trailing topical
# anchor (on/about/for/regarding/re: + multi-word specific phrase),
# suppress the short-circuit and fall through to L2. Bare phatic
# ("where did we leave off?") still short-circuits, so the fast path
# is preserved. Default: fall-through on ambiguous matches. +1 L2
# routing pass is an acceptable latency cost; faster-but-wrong is
# worse than slower-but-correct.
#
# Conservative edges:
#   - pronoun-only anchor ("...with that?") → still short-circuits
#   - single short/common noun ("...on stuff") → still short-circuits
#   - multi-word specific phrase ("...on the direct IOKit loading")
#     → fall-through to L2

# Pronouns and non-topical fillers that should NOT defeat the
# short-circuit even when they follow a topic marker.
_M120_B_PHATIC_PRONOUNS = {
    "that", "this", "it", "them", "those", "these",
    "us", "you", "me", "stuff", "things", "thing",
    "everything", "anything", "something", "nothing",
    "here", "there", "where", "whatever",
}

# Topic marker followed by at least one word of length >= 3. The
# conjunction check inspects the captured tail for topical specificity.
_M120_B_PHATIC_TOPIC_MARKER_RE = re.compile(
    r"\b(?:on|about|for|regarding|re:)\s+(.+)$",
    re.IGNORECASE,
)


def _m120_b_has_topical_anchor(text: str) -> bool:
    """Return True when a phatic-recency query carries a specific
    trailing topical anchor that should route through L2 semantic
    routing instead of the L0 short-circuit.

    Policy (from M120 Stream B directive):
      - Explicit marker (on/about/for/regarding/re:) followed by a
        multi-word phrase where at least one word has length >= 3
        and is not a pronoun/generic filler → topical anchor present.
      - Pronoun anchor only ("...with that?") → no topical anchor.
      - Single short/common noun ("...on stuff") → no topical anchor.
      - Ambiguous → default False (preserve short-circuit) to avoid
        breaking the fast path; the multi-word requirement covers
        T84-class queries while leaving bare phatic intact.
    """
    if not text:
        return False
    m = _M120_B_PHATIC_TOPIC_MARKER_RE.search(text)
    if not m:
        return False
    tail = m.group(1).strip().rstrip("?.!,;:")
    if not tail:
        return False
    # Strip leading determiners so "the direct IOKit loading" →
    # "direct IOKit loading" for word-count and content checks.
    _DETERMINERS = {"the", "a", "an", "our", "my", "your", "their"}
    tokens = [t for t in re.split(r"\s+", tail) if t]
    while tokens and tokens[0].lower().strip(".,;:?!") in _DETERMINERS:
        tokens = tokens[1:]
    if not tokens:
        return False
    # Content tokens = length >= 3 AND not a pronoun/generic filler.
    content_tokens = []
    for tok in tokens:
        norm = tok.lower().strip(".,;:?!\"'")
        if len(norm) < 3:
            continue
        if norm in _M120_B_PHATIC_PRONOUNS:
            continue
        content_tokens.append(norm)
    # Conservative: require at least two content tokens OR one
    # content token that is clearly specific (mixed-case / has a
    # digit / length >= 6). Single common noun like "stuff" or
    # "things" falls out via the pronoun set above; single specific
    # noun like "IOKit" or "Bridge" survives via the specificity
    # check below.
    if len(content_tokens) >= 2:
        return True
    if len(content_tokens) == 1:
        only = content_tokens[0]
        raw = tokens[-1] if tokens else only
        # Specific single-word anchor: contains a digit, has internal
        # capitalization in the raw token (e.g. "IOKit", "OpenAI"),
        # or length >= 6 (discriminates "IOKit" vs "stuff").
        if any(c.isdigit() for c in only):
            return True
        if any(c.isupper() for c in raw[1:]):
            return True
        if len(only) >= 6:
            return True
    return False


def _m120_b_should_short_circuit(text: str) -> bool:
    """Combined predicate: phatic-recency match AND no topical anchor.

    This is the gate the stream + non-stream handlers consult before
    firing the L0 recency short-circuit. See _m120_b_has_topical_anchor
    for the conjunction check.
    """
    if not _is_recency_query(text):
        return False
    if _m120_b_has_topical_anchor(text):
        return False
    return True


def _is_turn_ordinal_query(text: str) -> bool:
    """M125.2 Stream D — gate for turn-scope ordinal N-back short-circuit.

    Turn-scope queries ("what did I ask 2 turns ago", "what was your
    response 3 messages back", "what did we cover 5 exchanges ago")
    do NOT match _RECENCY_QUERY_RE, which is phatic-session-shaped.
    This predicate provides an independent entry point for the turn-
    scope loader without widening the session-scope phatic regex.

    Under-fit: requires an explicit ordinal marker (digit or spelled
    1-10) + turn-level scope noun (turns/messages/exchanges). Anaphoric
    queries that mention "turns" without a count are left to A5.1.
    """
    if not text:
        return False
    return bool(_TURN_ORDINAL_NBACK_RE.search(text))


def _build_turn_recency_response(record):
    """Compose a turn-level response for "N turns ago" queries.

    M125.2 Stream D. Distinct from session-level _build_recency_response:
    this renders ONE prior turn's user query + assistant response,
    not a whole-session rollup. Header format:
        "N turns ago, you asked: «query». I responded: «excerpt»."
    """
    if not record:
        return ("I don't have a prior turn on file yet for that index. "
                "Either this is early in the session or the turn "
                "number you referenced exceeds the current history.")
    nback = record.get("_nback") or 1
    total = record.get("_total_available") or 0
    user_q = (record.get("user_query") or "").strip()
    resp = (record.get("assistant_response") or "").strip()
    # Cap the response excerpt so a long prior answer doesn't dominate
    # the output. 600 chars matches the recency_bridge rough budget.
    EXCERPT = 600
    excerpt = resp if len(resp) <= EXCERPT else resp[:EXCERPT].rstrip() + "…"
    turn_ref = f"{nback} turn{'s' if nback != 1 else ''} ago"
    lines = []
    if user_q:
        lines.append(
            f"{turn_ref.capitalize()}, you asked: «{user_q}»."
        )
    else:
        lines.append(
            f"{turn_ref.capitalize()}, you didn't log a query text."
        )
    if excerpt:
        lines.append(f"I responded: «{excerpt}»")
    else:
        lines.append("I didn't log a response for that turn.")
    tools = record.get("tools_called") or []
    if tools:
        lines.append(
            f"(Tools dispatched that turn: {', '.join(tools)}. "
            f"Turn #{record.get('turn_number')} of {total} on disk.)"
        )
    else:
        lines.append(
            f"(Turn #{record.get('turn_number')} of {total} on disk.)"
        )
    return "\n".join(lines)


def _build_recency_response(summary):
    """Compose a natural prior-session response from the cached summary."""
    if not summary:
        return ("I don't have a prior session summary on file yet. "
                "This appears to be the first recorded session.")

    end_iso = summary.get("end_iso") or summary.get("start_iso") or "?"
    if isinstance(end_iso, str) and "T" in end_iso:
        end_iso = end_iso.replace("T", " ")[:16]

    duration_m = summary.get("duration_minutes") or 0
    turns = summary.get("duration_turns") or 0

    topics_list = summary.get("active_topics") or []
    topic_names = []
    for t in topics_list[:5]:
        if isinstance(t, dict):
            topic_names.append(t.get("topic") or "")
        elif isinstance(t, (list, tuple)) and t:
            topic_names.append(str(t[0]))
        else:
            topic_names.append(str(t))
    topic_names = [t for t in topic_names if t]
    last_topic = summary.get("last_topic") or "none"

    queries = summary.get("user_queries") or []
    ranked = sorted(queries, key=len, reverse=True)[:3]

    decisions = summary.get("decisions") or []
    corrections = summary.get("corrections") or []
    rules = summary.get("standing_rules") or []

    # M125.1 Stream B: if ordinal N-back was requested, swap the header
    # to identify which session we're returning ("3 sessions ago" rather
    # than an unqualified "last session").
    lines = []
    nback = summary.get("_nback") or 1
    if nback == 1:
        header = f"Our last session ended {end_iso}"
    else:
        header = f"The session {nback} sessions ago ended {end_iso}"
    lines.append(
        f"{header} and ran {duration_m:.0f} minutes across {turns} turns."
    )
    if topic_names:
        lines.append(
            f"We focused on: {', '.join(topic_names)}. "
            f"Last active topic: {last_topic}."
        )
    else:
        lines.append(f"Last active topic: {last_topic}.")

    if ranked:
        lines.append("Key exchanges you raised:")
        for q in ranked:
            lines.append(f'  - "{q.strip()[:200]}"')

    if decisions:
        lines.append("Decisions we made:")
        for d in decisions[:5]:
            lines.append(f"  - {d.strip()[:200]}")

    if corrections:
        lines.append("Corrections you applied:")
        for c in corrections[:5]:
            lines.append(f"  - {c.strip()[:200]}")

    if rules:
        lines.append("Standing rules set or reinforced:")
        for r in rules[:5]:
            lines.append(f"  - {r.strip()[:200]}")

    return "\n".join(lines)


# Cache at module load so the first message doesn't pay the disk read.
_LAST_SESSION_SUMMARY = load_last_session_summary()
_LAST_SESSION_BLOCK = format_last_session_block(_LAST_SESSION_SUMMARY)


# ── Current-era session anchor: grounds briefing against retrieved stale memos ──
#
# M100 Agent A2: M87 T01/T19 + M99 T01/T02/T09/T11 all showed the model
# narrating "We're in the aftermath of Main 42" / "transition from Main 53
# to Main 54" months after those sessions closed. Audit found the stale
# identifiers did NOT come from briefing_text — they came from per-query
# recalled `claude_automemory` memos (e.g. `session_20260412_main42.md`)
# labeled as "RELEVANT MEASUREMENTS" and from a stale `Roadmap.md` tool
# result ("Updated: 2026-04-11 (Main 38)"). Without a current-era anchor
# in the briefing, the model anchors on the highest-scored historical memo
# and treats it as the current state.
#
# Fix (scope-narrow per directive §2.2, avoids A1's retrieval layer):
# prepend a CURRENT SESSION CONTEXT block to the briefing that surfaces
# the 2-3 most recent session-log dates with their latest bullet. This
# gives the model an explicit "today is M100" anchor so retrieved M42/M53
# memos read as history, not current state.
#
# Source: vault/CLAUDE_session_log.md — the append-only ground truth.
# Parsed at module load + refreshed every _SESSION_BRIEFING_REFRESH_EVERY
# turns so new close-log entries land in-session.

_CLAUDE_SESSION_LOG_PATH = (
    "/Users/midas/Desktop/cowork/vault/CLAUDE_session_log.md"
)


def _parse_current_session_anchor(
    log_path: str = _CLAUDE_SESSION_LOG_PATH,
    max_dates: int = 3,
    max_chars: int = 900,
) -> str:
    """Extract the 2-3 most recent session-log day blocks and render a
    compact anchor for the briefing. Surfaces session identifiers
    (M95, M96, M99, M100, ...) so the model grounds "current era"
    against them before synthesizing from retrieved memos.

    Returns "" on any failure — the briefing still works without this
    block, it just loses the staleness anchor.
    """
    try:
        with open(log_path) as fh:
            text = fh.read()
    except Exception:
        return ""
    if not text:
        return ""

    # Find date-section headers: lines starting with "### YYYY-MM-DD"
    # The log is newest-first by convention; we take the first N and,
    # under each, the last 2 bullets (most-recent intra-day entries).
    import re as _re
    lines = text.splitlines()
    header_idxs = [
        i for i, ln in enumerate(lines)
        if _re.match(r"^###\s+\d{4}-\d{2}-\d{2}", ln)
    ]
    if not header_idxs:
        return ""

    # Collect up to max_dates day blocks (newest first in the file).
    blocks = []
    for i, start in enumerate(header_idxs[:max_dates]):
        end = header_idxs[i + 1] if i + 1 < len(header_idxs) else len(lines)
        header = lines[start].strip()
        # Extract date part
        m = _re.match(r"^###\s+(\d{4}-\d{2}-\d{2})", header)
        date_str = m.group(1) if m else header
        # Collect bullets in this section
        bullets = []
        for ln in lines[start + 1:end]:
            s = ln.strip()
            if s.startswith("- "):
                bullets.append(s[2:].strip())
        # Take last 2 bullets (most recent intra-day) for the narrative.
        tail = bullets[-2:] if bullets else []
        # Extract session identifiers from ALL bullets in the date block so
        # earlier-in-day work (M96/M97 opens) still surfaces even when the
        # last 2 bullets happen to be closes/corrections. Sort descending so
        # the anchor shows the highest (most recent) session identifiers.
        ids = []
        for b in bullets:
            for tok in _re.findall(
                r"(?<![\w-])(?:[Mm](?:ain)?\s*)(\d{1,3})\b", b
            ):
                tok_n = int(tok)
                if 1 <= tok_n <= 999 and tok_n not in ids:
                    ids.append(tok_n)
        ids.sort(reverse=True)
        blocks.append((date_str, tail, ids))

    if not blocks:
        return ""

    # Render compact block. Show date + session IDs inline + 1-line summary.
    parts = ["CURRENT SESSION CONTEXT (anchors current era — treat any "
             "retrieved 'Main <NN>' memory from an earlier era as history, "
             "not current state):"]
    for date_str, tail, ids in blocks:
        id_str = (
            " sessions=" + ",".join(f"M{n}" for n in ids[:4])
            if ids else ""
        )
        parts.append(f"  [{date_str}]{id_str}")
        for b in tail:
            # Strip bold markdown ** and timestamp prefix for compactness
            clean = _re.sub(r"\*\*\[[^\]]+\]\*\*\s*", "", b)
            clean = _re.sub(r"\*\*", "", clean)
            # Trim to ~180 chars per bullet
            if len(clean) > 180:
                clean = clean[:177] + "..."
            parts.append(f"    - {clean}")

    block = "\n".join(parts)
    if len(block) > max_chars:
        block = block[:max_chars - 3] + "..."
    return block


# Cached at module load; refreshed periodically by _build_presentation_briefing.
_CURRENT_SESSION_ANCHOR = _parse_current_session_anchor()
_CURRENT_SESSION_ANCHOR_LOADED_AT = time.time()


def _load_accumulated_standing_rules(max_summaries: int = 20) -> list:
    """Walk the last N session summaries and union their standing_rules.

    Main 39 P1.5 / Section 17.6: standing rules accumulate across
    sessions. Loading only the most recent summary fails when that
    session had no rule-shaped queries — the rules from the prior
    session get orphaned. Rules should persist until explicitly
    retracted (retraction mechanism is M40+ scope). For now, union
    across the last 20 sessions, dedup by lowered prefix.

    Walks newest → oldest, dedup keeps the most recent phrasing.
    """
    out = []
    seen = set()
    try:
        if not os.path.isdir(_SESSION_SUMMARIES_DIR):
            return out
        files = [
            os.path.join(_SESSION_SUMMARIES_DIR, f)
            for f in os.listdir(_SESSION_SUMMARIES_DIR)
            if f.startswith("session_") and f.endswith(".json")
        ]
        files.sort(key=os.path.getmtime, reverse=True)
        for path in files[:max_summaries]:
            try:
                with open(path) as f:
                    summary = json.load(f)
            except Exception:
                continue
            for r in (summary.get("standing_rules") or []):
                r = (r or "").strip()
                if not r:
                    continue
                k = r[:120].lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(r[:240])
    except Exception as e:
        print(f"[session] _load_accumulated_standing_rules failed: {e}",
              flush=True)
    return out


# Main 39 P1.5: hold all accumulated standing rules across the last 20
# session summaries in a module-level cache. Injected into the
# synthesizer system message on every turn (see synthesizer.py
# format_standing_rules_block). Persistence path (rules → summary →
# bridge → briefing) was working at end of P1; what didn't work was
# (a) enforcement (rules in briefing context, not system directive)
# and (b) cross-session orphaning when an intermediate session had no
# rules. This load fixes both — every accumulated rule ends up in the
# system message of every turn until M40 ships a retraction mechanism.
#
# Pre-P1 historical summaries contain regex false positives ("i don't
# mind ...", "keep ... concise" embedded in questions, etc.). Run each
# accumulated rule back through the 72B classifier at boot to filter
# them out. The classifier already exists from P1's session-close pass.
# Boot cost: ~3-5s per batch of 30 rules. If the classifier fails
# (72B offline at boot), fall back to keeping all rules — better to
# carry a few stale rules than lose all real preferences.
_RAW_ACCUMULATED_RULES: list = _load_accumulated_standing_rules()


def _revalidate_accumulated_rules(raw_rules: list) -> list:
    """Pass each accumulated rule through the 72B classifier and return
    the subset still classified as RULE. Uses a sentinel to distinguish
    classifier failure from legitimate "all REMARK" outcomes — if the
    sentinel (a known unambiguous rule) doesn't survive classification,
    the classifier is broken and we fall back to the raw list rather
    than nuke standing state on a transient model failure.
    """
    if not raw_rules:
        return []
    sentinel = "always cite source files when referencing code"
    try:
        out = _classify_ambiguous_rules_with_72b([sentinel] + raw_rules)
        if sentinel not in out:
            print(
                f"[session] rule revalidation health check failed "
                f"(sentinel rejected) — keeping raw list of "
                f"{len(raw_rules)} rules as fallback",
                flush=True,
            )
            return raw_rules
        validated = [r for r in out if r != sentinel]
        dropped = len(raw_rules) - len(validated)
        if dropped:
            print(
                f"[session] rule revalidation dropped {dropped} "
                f"historical false positives",
                flush=True,
            )
        return validated
    except Exception as e:
        print(
            f"[session] rule revalidation crashed: {e} — keeping raw list",
            flush=True,
        )
        return raw_rules


# Initial value: raw accumulated list. Revalidation against the 72B
# happens inside boot() after llm_route_fn is defined further down in
# the module. The raw list is the safe fallback if boot() is never
# called (e.g. test imports).
_ACTIVE_STANDING_RULES: list = list(_RAW_ACCUMULATED_RULES)

if _LAST_SESSION_BLOCK:
    print(
        f"[session] prior-session bridge loaded: "
        f"{len((_LAST_SESSION_SUMMARY or {}).get('user_queries', []))} queries, "
        f"{(_LAST_SESSION_SUMMARY or {}).get('duration_minutes','?')} min",
        flush=True,
    )
if _RAW_ACCUMULATED_RULES:
    print(
        f"[session] {len(_RAW_ACCUMULATED_RULES)} raw standing rules "
        f"accumulated from session summaries — revalidation pending in boot()",
        flush=True,
    )


# ── Main 37: per-turn session logger ───────────────────────────────────
#
# Every turn writes a full internal-state JSON record to disk at
# data/session_logs/<session_id>/turn_<NNNN>.json. No real-time observer
# required. The file accumulates, the live session becomes the dataset,
# and analysis tools under tools/ read the session logs to produce
# quality reports, anomaly flags, and cross-session comparisons.
#
# Sections written per turn:
#   input      - timestamp, query, turn_number, time_since_last
#   routing    - l1_match, l2_decision, tools_called, tools_requested_not_called
#   retrieval  - shape_fired, recall scores, narrative/enumeration/absence state
#   context    - briefing_text, mem_ctx_text, token estimates
#   generation - ttft_ms, total_ms, tps, accept_rate, response_text
#   extraction - fired, count, types, latency_ms
#   quality    - fabrication flags, stale entities, score stats
#   post_turn  - memories stored, store total, daemon events by type
#
# All logger operations are defensive-wrapped so a bug in the logger
# cannot break a user turn.

_SESSION_ID = "sess_" + time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
_SESSION_LOG_DIR = f"/Users/midas/Desktop/cowork/data/session_logs/{_SESSION_ID}"
try:
    os.makedirs(_SESSION_LOG_DIR, exist_ok=True)
    # Write a session header once at boot so readers can identify the run
    with open(os.path.join(_SESSION_LOG_DIR, "session.json"), "w") as _sf:
        json.dump({
            "session_id": _SESSION_ID,
            "pid": os.getpid(),
            "start_ts": time.time(),
            "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prior_session_loaded": bool(_LAST_SESSION_BLOCK),
            "prior_session_path": (_LAST_SESSION_SUMMARY or {}).get("_path"),
        }, _sf, indent=2, default=str)
    print(f"[turnlog] session={_SESSION_ID} dir={_SESSION_LOG_DIR}", flush=True)
except Exception as _e:
    print(f"[turnlog] session-dir create failed: {_e}", flush=True)
    _SESSION_LOG_DIR = None

# Known-stale entities that should no longer appear in retrieved context.
# Updates here are effective on next midas restart. Keep this list tight:
# only patterns that are definitely wrong as of current canonical state.
_STALE_ENTITY_PATTERNS = [
    "ChromaDB",
    "chroma_db",
    "4,677 memories",
    "4677 memories",
    "Living Model Revival",
    "Living Model — revive",
    "Living Model needs to be revived",
    "reviving the Living Model",
    "SRAM pipelining",
    "34→138 GB/s",
    "34 to 138 GB/s",
    "apple-slc-cache-hints push",
    "d3Force",
    "d3Force library",
    "broke the entire animation loop",
    "viz_render_loop_diagnosis",
    "8B /tmp incident and",
    "2C feature flag",
    "kANEFEnableLateLatchKey",
    "kANEFKeepModelMemoryWiredKey",
    "kANEFEnableFWToFWSignal",
]

# Phrases in the 72B response that claim a tool ran. If the phrase fires
# but the matching tool is NOT in tools_called, it's a fabricated tool
# claim (the model hallucinated that a tool was invoked).
_TOOL_CLAIM_PATTERNS = {
    "browse_search": [
        r"\bweb search (?:didn'?t|did not|returned|found|shows?|indicates?|yielded|provided|gave)",
        r"\bsearch(?:ed)? (?:the web|online|the internet)",
        r"\bbased on (?:the |my )?(?:web )?search(?: result)?s?",
        r"\bi (?:just |have )?searched",
        r"\bthe search (?:returned|did not|didn'?t|found|shows?)",
    ],
    "vault_read": [
        r"\bvault_read (?:returned|shows?|indicates?)",
        r"\bthe roadmap(?:\.md)? (?:shows?|says?|contains?|doesn'?t)",
    ],
}

_TURN_LOG: dict = {}
_LAST_TURN_END_TS = None


# ── m109_zeta instrumentation helpers ──────────────────────────────────
# Additive-only logging helpers for directive M109 ζ. Each function is pure
# (no side-effects on module state) and returns a dict/scalar that is spliced
# into the turn log at the write points in the two handlers. Upstream sources
# were verified before wiring:
#   - role_weight formula: local_store.py:434-435 + daemon.py:527-528
#   - canonical boost multiplier (1.30) same two files
#   - grounding gate state: answer_scrub.py:41 (TIER1_ENABLED)
#   - stop_reason: no upstream signal preserved; 72B server always emits
#     finish_reason="stop" (qwen_spec_decode_server.py:1628/1678). Inferred
#     from tokens_decoded vs max_tokens and loop-detector flag per directive.


# m109_zeta_field_1 + field_8: role_weight + canonical_boost multiplier.
# Formula verified in local_store.py:434-435 and daemon.py:527-528.
# Additional downweights (assistant 0.50, research 0.05 possessive,
# user 0.30 possessive) are conditional on query-level flags not
# exposed at this log point; we log the base provenance multiplier
# that would apply at scoring time unconditionally.
def _m109_role_weight(source_role: str) -> float:
    """Return the multiplicative weight the recall scorer applies to a
    memory based on its source_role tag. Derived from the formula at
    local_store.py:434-435 and daemon.py:527-528. assistant=0.50,
    canonical=1.30, everything else (user, vault, research, unknown)=1.00
    at the base scoring level. Possessive-intent downweights (research
    0.05, user 0.30) are query-conditional and not applied here."""
    if source_role == "canonical":
        return 1.30
    if source_role == "assistant":
        return 0.50
    return 1.00


def _m109_canonical_boost_multiplier(source_role: str) -> float:
    """Per-record counterfactual: what multiplier WOULD canonical-boost
    apply to this record if the global canonical_boost toggle were active.
    Non-canonical records get 1.00 (no change). Canonical records get
    1.30. Logged regardless of current global state per directive M107."""
    if source_role == "canonical":
        return 1.30
    return 1.00


def _m109_grounding_state() -> str:
    """Report the current state of the Tier 1 grounding gate. Returns
    one of: 'enabled', 'disabled', 'unknown'. Tier 1 is disabled by
    default for Gemma 4 per answer_scrub.py:41 (TIER1_ENABLED)."""
    try:
        from answer_scrub import TIER1_ENABLED as _t1
        return "enabled" if _t1 else "disabled"
    except Exception:
        return "unknown"


def _m109_content_hash(text: str) -> str:
    """Short content hash: SHA256 first 16 hex chars of the text. Used
    for the post_turn.memories_stored_this_turn manifest so turn-to-turn
    duplication can be detected offline without pulling the full text."""
    try:
        import hashlib
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _m109_infer_stop_reason(tokens_decoded, max_tokens, loop_detected,
                             gen_error=None, user_cancel=False):
    """Midas-side inference of decode stop reason. The 72B server always
    emits finish_reason='stop' regardless of the true cause, so we infer:
      - user_cancel: caller sets flag (not currently plumbed; reserved)
      - verifier_error: exception raised during streaming
      - max_tokens: tokens_decoded reached (or exceeded) max_tokens cap
      - loop_detected: Main 42 in-flight n-gram loop detector tripped
      - eos: tokens_decoded < max_tokens and no error/loop => natural EOS
      - unknown_v2: insufficient signal
    Per directive M109 §3.1 field 4, this is best-effort inference —
    the ideal fix is upstream preservation in qwen_spec_decode_server.py
    (OUT OF SCOPE for this directive)."""
    if user_cancel:
        return "user_cancel"
    if gen_error:
        return "verifier_error"
    if loop_detected:
        return "loop_detected"
    try:
        if tokens_decoded is not None and max_tokens is not None:
            if int(tokens_decoded) >= int(max_tokens):
                return "max_tokens"
            if int(tokens_decoded) > 0:
                return "eos"
    except Exception:
        pass
    return "unknown_v2"


def _m123_a2_completion_reason(stop_reason):
    """M123 Stream A2: derive generation.completion_reason enum from the
    existing M109 stop_reason signal. Diagnose-first-then-fix found T36/T39
    were Midas-side in-flight loop detector trips, NOT model-level truncation.
    This field discriminates user-visible completion shapes:

      - stop_token_end     : natural EOS from model
      - max_tokens_reached : hit configured max_tokens cap
      - stream_terminated  : Midas-side loop detector or scrub intervention
      - completion_signal  : upstream finish_reason='stop' (generic)
      - timeout            : wall-clock deadline exceeded (reserved)
      - unknown            : insufficient signal

    Conditional emission per ζ v2.2 extension. Additive; does not alter
    M109 stop_reason semantics. Rationale: Lets M123+ separate legitimate
    natural completions from truncation-class interventions without
    disturbing existing M109 consumers."""
    if stop_reason in ("eos", "stop_token_end"):
        return "stop_token_end"
    if stop_reason in ("max_tokens", "max_tokens_reached"):
        return "max_tokens_reached"
    if stop_reason in ("loop_detected", "user_cancel", "verifier_error",
                        "stream_terminated"):
        return "stream_terminated"
    if stop_reason == "timeout":
        return "timeout"
    if stop_reason in ("unknown_v2", "unknown", None, ""):
        return "unknown"
    # Default pass-through for any upstream server-emitted finish_reason we
    # don't explicitly recognize (preserves forward compatibility).
    return "completion_signal"


def _m109_memory_manifest(delta_count):
    """Best-effort manifest of memories stored in the just-completed turn.
    Reads daemon._session_facts (in-memory list of facts stored this
    session) and returns the last `delta_count` entries as
    {id, content_hash, source_role, extractor}. extractor is derived
    from fact['extraction_source'] ('ane_8b' → '8b_ane', absent → 'cpu_heuristic')."""
    try:
        if not delta_count or delta_count <= 0:
            return []
        d = getattr(memory, "daemon", None)
        if d is None:
            return []
        facts = getattr(d, "_session_facts", None)
        if not facts:
            return []
        recent = facts[-int(delta_count):]
        manifest = []
        for f in recent:
            if not isinstance(f, dict):
                continue
            ext_src = f.get("extraction_source") or ""
            if ext_src == "ane_8b":
                extractor = "8b_ane"
            elif ext_src:
                extractor = ext_src
            else:
                extractor = "cpu_heuristic"
            manifest.append({
                "id": f.get("id") or "",  # store.store() fact_id is returned but
                                          # not written back into the dict; may be empty
                "content_hash": _m109_content_hash(f.get("text", "")),
                "source_role": f.get("source_role", "") or "",
                "extractor": extractor,
                "type": f.get("type", "") or "",
            })
        return manifest
    except Exception:
        return []


def _m109_annotate_recall(filtered_list):
    """Given the post-filter recall list, return a parallel list of
    per-record telemetry dicts with role_weight (field 1), provenance
    (field 7), and canonical_boost counterfactual (field 8). Designed
    to be spliced into retrieval.recall_filtered alongside existing
    score/source_role/type/text fields — the writer merges this by
    index position."""
    out = []
    for r in filtered_list or []:
        meta = r.get("metadata", {}) or {}
        src = (r.get("source_role") or meta.get("source_role", "")) or ""
        # Origin derivation — per directive M108 field 7.
        # Heuristic: source_role is the primary signal. extraction_source
        # is set on daemon-stored facts; user_utterance vs extraction is
        # inferred from role tag + presence of extractor metadata.
        origin = "unknown"
        if src == "user":
            origin = "user_utterance"
        elif src == "assistant":
            origin = "extraction"  # assistant-sourced facts come from extractor
        elif src == "canonical":
            origin = "canonical_inject"
        elif src == "vault":
            origin = "vault_sync"
        elif src == "research":
            origin = "research_import"
        ext = (r.get("extraction_source")
               or meta.get("extraction_source") or "")
        if ext == "ane_8b":
            extractor = "8b_ane"
        elif ext:
            extractor = ext
        else:
            extractor = None
        out.append({
            "role_weight": _m109_role_weight(src),
            "canonical_boost_multiplier_if_active":
                _m109_canonical_boost_multiplier(src),
            "provenance": {
                "origin": origin,
                "original_turn": (r.get("original_turn")
                                   or meta.get("original_turn")
                                   or "not_consumed_this_turn"),
                "extractor_if_any": extractor,
            },
        })
    return out


def _m109_tag_history_messages(msgs, briefing_present):
    """For each message in the assembled-prompt msgs list, tag the
    source_turn. system message → 'system'; briefing message (if the
    system slot contains the briefing) → 'briefing'; history messages
    → 'T<N>' via reverse-indexing from the current turn number.
    Per directive §3.1 field 6 — makes conversation-history provenance
    explicit instead of only inferrable by ordering."""
    tagged = []
    if not msgs:
        return tagged
    cur_turn = 0
    try:
        if _TURN_LOG:
            cur_turn = int(_TURN_LOG.get("turn_number", 0) or 0)
    except Exception:
        cur_turn = 0
    # Count history messages (non-system, non-tail-user). Last message
    # is the current user turn. Anything before it that is not the
    # system slot is history.
    n = len(msgs)
    # Identify the last user message index (current turn).
    last_user_idx = -1
    for i in range(n - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    # Walk history pairs backward assigning turn numbers. Each prior
    # (user, assistant) pair = one prior turn.
    hist_indices = [i for i in range(n)
                    if i != 0 and i != last_user_idx
                    and msgs[i].get("role") in ("user", "assistant")]
    # Pair them: every consecutive (user, assistant) pair represents
    # one prior turn. Count backwards from (cur_turn - 1).
    # Simple approach: assign turn number by descending index — most
    # recent non-tail message = cur_turn - 1, next earlier = cur_turn - 2.
    hist_turn_map = {}
    pair_turn = cur_turn - 1 if cur_turn > 0 else 0
    # Group consecutive user+assistant into one turn.
    i = len(hist_indices) - 1
    while i >= 0:
        # assistant typically follows user; walk backward pairs
        idx = hist_indices[i]
        hist_turn_map[idx] = pair_turn
        # previous index (user of same pair), assign same turn
        if i - 1 >= 0 and msgs[hist_indices[i-1]].get("role") == "user":
            hist_turn_map[hist_indices[i-1]] = pair_turn
            i -= 2
        else:
            i -= 1
        pair_turn = max(0, pair_turn - 1)

    for i, m in enumerate(msgs):
        role = m.get("role", "")
        if i == 0 and role == "system":
            # System slot: briefing if briefing_present else pure system prompt
            src_turn = "briefing" if briefing_present else "system"
        elif i == last_user_idx:
            src_turn = f"T{cur_turn}"  # current turn
        elif i in hist_turn_map:
            src_turn = f"T{hist_turn_map[i]}"
        else:
            src_turn = "not_consumed_this_turn"
        tagged.append({
            "role": role,
            "content": m.get("content", ""),
            "source_turn": src_turn,
        })
    return tagged


def _rough_tokens(text: str) -> int:
    """Char-based token estimate (chars / 4). Good enough for logging
    without pulling in tiktoken."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _detect_stale_entities(text: str) -> list:
    """Return list of stale-entity patterns that appear in `text`."""
    if not text:
        return []
    hits = []
    low = text.lower()
    for pat in _STALE_ENTITY_PATTERNS:
        if pat.lower() in low:
            hits.append(pat)
    return hits


# Rolling window of recently dispatched tools across turns. The FAB
# detector consults this so that a legitimate prior-turn reference
# ("I did perform the search, it returned no results" in response to a
# user question about a prior tool call) is not flagged as fabrication.
# Session 2 turn 6 false positive drove this fix. maxlen=3 keeps the
# last three turns of dispatch history, which covers single-hop
# reference follow-ups without diluting true fabrications that emerge
# many turns later.
from collections import deque as _deque
_RECENT_TOOLS_CALLED: _deque = _deque(maxlen=3)


# Main 39 P2: assistant-ingest gating for conversation-route turns.
# Section 17.2 of the state synthesis: when l2_decision == 'conversation'
# AND tools_called is empty AND the retrieval shape is not in this
# allow-list, the 72B is paraphrasing its own training data and
# ingesting that as "memory" pollutes the store with content that is
# not novel and competes with real user-sourced facts on retrieval.
# Across the last 15 sessions: 35% of turns hit this gate, accounting
# for 47% of all assistant ingestions at a mean 63.4 facts per turn.
# The gate target is below 20 facts/turn on conversation-route turns;
# Option A (skip entirely) trivially achieves it.
# Main 42: removed "narrative" from allowlist. Conversation-route narrative
# turns produce 12-21 model-paraphrased facts/turn, including extracting
# from honest abstentions ("I don't have information..."). Same provenance
# contamination pattern as the M38 entity leak. Only recency_bridge and
# enumeration survive: these are structural queries where the user's
# question itself carries novel information worth storing.
_INGEST_SHAPE_ALLOWLIST = {"recency_bridge", "enumeration"}


def _should_skip_assistant_ingest(turn_log: dict) -> tuple:
    """Return (skip: bool, reason: str). Reads from the live turn log.

    Main 39 P2: gate conversation-route turns where the model paraphrases
    training data with no tool grounding.
    Main 41 extension: also gate tool-route turns. The assistant response
    on a tool-route turn is a paraphrase of the tool result -- the tool
    result is already in the vault/store, the user query is already
    ingested, and the assistant paraphrase adds no novel information.
    Data: 91% of assistant-sourced ingestion came from tool-route turns
    (632 facts / 10 sessions), flooding the store with model paraphrase.
    """
    rt = turn_log.get("routing", {}) or {}
    ret = turn_log.get("retrieval", {}) or {}
    # Gate ALL assistant ingestion. The user message is still ingested
    # separately. Only allow through recency/enumeration shapes on
    # conversation routes (structural queries with novel user info).
    if rt.get("tools_called"):
        return (True, "tool_route_paraphrase")
    if rt.get("l2_decision") == "conversation":
        shape = ret.get("shape_fired")
        if shape in _INGEST_SHAPE_ALLOWLIST:
            return (False, "")
        return (True, f"conversation_no_tools shape={shape!r}")
    return (False, "")


def _record_dispatched_tools(tools: list) -> None:
    """Call after a turn's tool execution to update the rolling window."""
    try:
        if tools:
            _RECENT_TOOLS_CALLED.append(set(tools))
        else:
            # Record an empty set so the window shifts naturally. This
            # prevents stale entries from persisting indefinitely when
            # a conversation-only turn is sandwiched between tool turns.
            _RECENT_TOOLS_CALLED.append(set())
    except Exception:
        pass


def _recently_dispatched(tool_name: str) -> bool:
    """True if `tool_name` appears in any of the recent turns' dispatch
    sets. Used by the FAB detector to avoid flagging contextual
    references to prior-turn tool results.
    """
    try:
        for s in _RECENT_TOOLS_CALLED:
            if tool_name in s:
                return True
    except Exception:
        pass
    return False


def _detect_fabricated_tool_claims(response: str, tools_called: list) -> list:
    """Return list of (tool_name, matched_phrase) pairs where the
    response claims a tool ran but the tool was never invoked.

    Main 38 Session 2 fix: also consults `_RECENT_TOOLS_CALLED` so that
    a turn following a tool-dispatching turn can legitimately reference
    the prior result without being flagged. Only claims where the tool
    has NOT run in the current turn AND has NOT run in any of the last
    3 turns are counted as fabrications.
    """
    if not response:
        return []
    called = set(tools_called or [])
    flagged = []
    for tool_name, patterns in _TOOL_CLAIM_PATTERNS.items():
        if tool_name in called:
            continue  # actually ran this turn
        if _recently_dispatched(tool_name):
            continue  # ran in a recent prior turn, reference is legitimate
        for pat in patterns:
            try:
                m = re.search(pat, response, re.IGNORECASE)
                if m:
                    flagged.append({
                        "tool": tool_name,
                        "phrase": m.group(0)[:120],
                    })
                    break  # one hit per tool is enough
            except Exception:
                continue
    return flagged


def _retrieval_score_stats(filtered: list) -> dict:
    """Score distribution summary for the retrieval layer."""
    if not filtered:
        return {"count": 0, "max": 0, "min": 0, "mean": 0,
                "above_07": 0, "below_04": 0}
    scores = [float(r.get("score", 0)) for r in filtered]
    n = len(scores)
    return {
        "count": n,
        "max": round(max(scores), 3),
        "min": round(min(scores), 3),
        "mean": round(sum(scores) / n, 3),
        "above_07": sum(1 for s in scores if s > 0.7),
        "below_04": sum(1 for s in scores if s < 0.4),
    }


def _detect_tools_requested_not_called(query: str, tools_called: list) -> list:
    """Heuristic: if the query contains an intent keyword for a tool but
    the tool was never dispatched, flag it as a missed dispatch."""
    if not query:
        return []
    low = query.lower()
    called = set(tools_called or [])
    missed = []
    intent_map = {
        "browse_search": [
            "web search", "search the web", "search online",
            "from the internet", "google it", "look online",
            "look up online", "search for",
        ],
    }
    for tool, phrases in intent_map.items():
        if tool in called:
            continue
        for p in phrases:
            if p in low:
                missed.append({"tool": tool, "phrase": p})
                break
    return missed


def _turn_start(message: str) -> None:
    """Initialize the current turn log dict. Called at the top of every
    request handler, before any pipeline work."""
    global _TURN_LOG, _LAST_TURN_END_TS
    try:
        now = time.time()
        since_last = None
        if _LAST_TURN_END_TS is not None:
            since_last = round(now - _LAST_TURN_END_TS, 2)
        _TURN_LOG = {
            "schema_version": "2.3",  # m125_a2_zeta_v23 — adds retrieval.dispatch_decision
            "session_id": _SESSION_ID,
            "turn_number": _session.get("messages_sent", 0),
            "input": {
                "timestamp": now,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(now)),
                "query": message,
                "query_chars": len(message or ""),
                "query_tokens_est": _rough_tokens(message or ""),
                "time_since_last_turn_s": since_last,
            },
            "routing": {"l1_match": None, "l2_decision": None,
                        "tools_called": [], "tools_requested_not_called": []},
            "retrieval": {"shape_fired": None,
                          "recall_filtered": [],
                          "narrative": None,
                          "enumeration": None,
                          "absence_guard": None,
                          "contradictions": None,
                          "context_tracker": None,
                          "score_stats": None},
            "context": {"briefing_chars": 0,
                        "briefing_has_prior_session": False,
                        "briefing_preview": "",
                        "mem_ctx_chars": 0,
                        "per_query_chars": 0,
                        "system_tokens_est": 0,
                        "user_tokens_est": 0},
            "generation": {"ttft_ms": None,
                           "total_ms": None,
                           "tps": None,
                           "accept_rate": None,
                           "response_text": None,
                           "response_chars": 0,
                           "response_tokens_est": 0},
            "extraction": {"fired": False, "count": 0, "types": [],
                           "latency_ms": None},
            "quality": {"fabricated_tool_claims": [],
                        "stale_entities_in_response": [],
                        "stale_entities_in_context": [],
                        "tools_requested_not_called": [],
                        "grounded": None},
            "post_turn": {"memories_stored_this_turn": 0,
                          "memory_store_total": None,
                          "daemon_events": {}},
        }
    except Exception as _e:
        print(f"[turnlog] _turn_start failed: {_e}", flush=True)
        _TURN_LOG = {}


def _turn_record(section: str, **kwargs) -> None:
    """Merge kwargs into the given section of the current turn log.
    Silently ignores section misses so logger bugs can't break turns."""
    try:
        if not _TURN_LOG:
            return
        _TURN_LOG.setdefault(section, {}).update(kwargs)
    except Exception:
        pass


def _turn_append(section: str, key: str, value) -> None:
    """Append `value` to a list at section[key]. Used for tools_called,
    daemon_events, etc."""
    try:
        if not _TURN_LOG:
            return
        sec = _TURN_LOG.setdefault(section, {})
        if not isinstance(sec.get(key), list):
            sec[key] = []
        sec[key].append(value)
    except Exception:
        pass


def _turn_count_event(ev_type: str) -> None:
    """Increment a daemon-event counter in the current turn."""
    try:
        if not _TURN_LOG:
            return
        pt = _TURN_LOG.setdefault("post_turn", {})
        events = pt.setdefault("daemon_events", {})
        events[ev_type] = events.get(ev_type, 0) + 1
    except Exception:
        pass


def _turn_write() -> None:
    """Write the current turn log to disk and update the last-turn-end
    timestamp. Called at the very end of the request handler."""
    global _LAST_TURN_END_TS
    try:
        if not _TURN_LOG or not _SESSION_LOG_DIR:
            return
        turn_num = _TURN_LOG.get("turn_number", 0)
        path = os.path.join(_SESSION_LOG_DIR, f"turn_{turn_num:04d}.json")
        with open(path, "w") as fh:
            json.dump(_TURN_LOG, fh, indent=2, default=str)
        _LAST_TURN_END_TS = time.time()
    except Exception as _e:
        print(f"[turnlog] _turn_write failed: {_e}", flush=True)


# ── Main 37: shared per-turn context pipeline ───────────────────────────
#
# Prior to Main 37 these helpers were inlined inside /api/chat. /api/chat/stream
# had NO equivalent pipeline: no context tracker update, no stable seed-query
# briefing, no warm context, no message-#1 session-open block, no PRIOR SESSION
# bridge, and build_messages was called without the `briefing=` kwarg. That
# meant the terminal client (which only hits the stream endpoint) ran on a
# Main-25-era raw-mem-dump prompt while the paper and evaluation harness were
# measured through /api/chat.
#
# These three helpers (_update_context_tracker, _build_presentation_briefing,
# _build_per_query_block) are now called from BOTH endpoints so they share
# the same full pipeline.


def _update_context_tracker(message: str) -> None:
    """Feed a user message into the live ContextTracker and emit the
    topic-state event for the viz.

    Shared between /api/chat and /api/chat/stream so terminal traffic
    drives the tracker the same way web UI traffic does.
    """
    try:
        tracker = _get_context_tracker()
        if tracker is None:
            return
        tracker.on_message(message, role="human")
        _emit_subconscious_event(
            "context_topic_update",
            active_topics=[
                {"topic": t, "weight": round(w, 3)}
                for t, w in tracker.active_topics
            ],
            message_count=tracker.message_count,
        )
    except Exception:
        pass


def _build_presentation_briefing(is_first_message: bool):
    """Assemble the full per-turn system-slot briefing used by both
    /api/chat and /api/chat/stream.

    Pipeline:
      1. Rebuild the stable seed-query session briefing on cadence
         (_SESSION_BRIEFING_REFRESH_EVERY turns). This gives the
         verifier's prefix KV cache a byte-stable string to hit against.
      2. Prepend the warm-open profile/bridge/pre-warmed block if present.
      3. On message #1 only, prepend the session-open status line and
         the PRIOR SESSION bridge (Main 37) so the first-turn response
         has explicit continuity from the prior session summary.

    Mutates the module-level `_session_briefing` and
    `_session_briefing_built_at_turn` globals. Returns a string (the
    composed briefing) or None on total failure, in which case the
    synthesizer falls back to the raw memory-bullet path.
    """
    global _session_briefing, _session_briefing_built_at_turn

    presentation_briefing = None

    # Step 1: stable seed-query briefing, refreshed on cadence.
    try:
        turns_since_build = (
            _session["messages_sent"] - _session_briefing_built_at_turn
        )
        if (_session_briefing is None
                or turns_since_build >= _SESSION_BRIEFING_REFRESH_EVERY):
            import sys as _sys
            _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
            from multi_path_retrieve import present as _present
            SEED_QUERY = ("Current state and active work on Apple "
                          "Silicon ML, ANE, spec decode, "
                          "Subconscious, paper status.")
            seed_recall = memory.recall(SEED_QUERY, n_results=12)
            seed_filtered = []
            for r in (seed_recall.get("results") or []):
                if r.get("score", 0) < 0.30:
                    continue
                t = r.get("text", "")
                if t.startswith("[") and ".md]" in t[:50]:
                    continue
                seed_filtered.append(r)
                if len(seed_filtered) >= 8:
                    break
            _session_briefing = _present(
                seed_filtered, SEED_QUERY, max_chars=1500)
            _session_briefing_built_at_turn = _session["messages_sent"]
    except Exception:
        # Leave _session_briefing at whatever it was (None or stale-but-valid)
        pass

    # Step 2: wrap with warm context (profile + bridge + pre-warmed cache).
    try:
        _wc = get_warm_context()
    except Exception:
        _wc = ""
    if _wc and _session_briefing:
        presentation_briefing = _wc + "\n\n" + _session_briefing
    elif _wc:
        presentation_briefing = _wc
    else:
        presentation_briefing = _session_briefing

    # Step 2b (M100 A2): prepend the current-session-era anchor so retrieved
    # stale "Main 42 Session" / "Main 53 cross-turn" memos don't anchor the
    # model's "current state" narration. Refresh on the same cadence as the
    # seed-query briefing (cheap: one file read + regex).
    try:
        global _CURRENT_SESSION_ANCHOR, _CURRENT_SESSION_ANCHOR_LOADED_AT
        # Refresh when the seed briefing refreshes (or first time through)
        _anchor_age_s = time.time() - (_CURRENT_SESSION_ANCHOR_LOADED_AT or 0)
        if (not _CURRENT_SESSION_ANCHOR) or _anchor_age_s > 600:
            _CURRENT_SESSION_ANCHOR = _parse_current_session_anchor()
            _CURRENT_SESSION_ANCHOR_LOADED_AT = time.time()
        if _CURRENT_SESSION_ANCHOR:
            if presentation_briefing:
                presentation_briefing = (
                    _CURRENT_SESSION_ANCHOR + "\n\n" + presentation_briefing
                )
            else:
                presentation_briefing = _CURRENT_SESSION_ANCHOR
    except Exception as _e:
        # Anchor is defense in depth; briefing still works without it.
        print(f"[briefing] current-session anchor build failed: {_e}",
              flush=True)

    # Step 3: on message #1, prepend session-open status + PRIOR SESSION bridge.
    if is_first_message:
        try:
            import urllib.request as _ur
            with _ur.urlopen(
                "http://127.0.0.1:8450/api/session/context",
                timeout=2.5,
            ) as r:
                ctx = json.loads(r.read())
            lines = []
            if _LAST_SESSION_BLOCK:
                lines.append("PRIOR SESSION: what we discussed last time:")
                lines.append(f"  {_LAST_SESSION_BLOCK}")
                lines.append("")
            lines.append("SESSION OPEN: system context for this turn:")
            d = ctx.get("daemon", {}) or {}
            if d.get("uptime_h") is not None:
                lines.append(
                    f"  daemon uptime {d['uptime_h']}h, "
                    f"{d.get('events_emitted','?')} events emitted, "
                    f"{d.get('components_running','?')}/5 components running"
                )
            sl = ctx.get("since_last_event", {}) or {}
            if sl.get("hours_ago") is not None:
                lines.append(
                    f"  last activity {sl['hours_ago']}h ago, "
                    f"{sl.get('n_events_24h',0)} events / "
                    f"{sl.get('loops_fired_24h',0)} maint loops / "
                    f"{sl.get('memories_recalled_24h',0)} recalls / "
                    f"{sl.get('enrichments_24h',0)} enrichments in 24h"
                )
            ac = ctx.get("active_context", {}) or {}
            if ac.get("primary_topic_guess"):
                lines.append(
                    f"  current focus appears to be {ac['primary_topic_guess']} "
                    f"(topic counts: {ac.get('topic_counts_24h',{})})"
                )
            rq = ctx.get("recent_queues", []) or []
            if rq:
                queues_str = ", ".join(
                    f"{q['file']} ({q['modified_h_ago']}h ago)"
                    for q in rq[:3]
                )
                lines.append(f"  recent overnight queues: {queues_str}")
            mem = ctx.get("memory", {}) or {}
            if mem.get("total"):
                lines.append(
                    f"  memory store: {mem['total']} memories")
            session_context_block = "\n".join(lines)
            if presentation_briefing:
                presentation_briefing = (
                    session_context_block
                    + "\n\n"
                    + presentation_briefing
                )
            else:
                presentation_briefing = session_context_block
        except Exception:
            # Session-open context is bonus, not required.
            pass

    # Main 43 Phase 4: abstention directive for cross-experiment reasoning.
    # Ablation tested Main 44: model passes P03/P06 WITHOUT this directive.
    # Phase 1-3 architectural fixes are sufficient. Directive kept as defense
    # in depth — does no harm, may help on edge cases.
    if presentation_briefing:
        presentation_briefing += (
            "\n\nWhen connecting findings from different sources or sessions, "
            "state the connection explicitly. If you cannot cite a single source "
            "for a causal claim, say \"I believe X but this connects findings "
            "from separate measurements.\""
        )

    return presentation_briefing


# ── Main 41: measurement registry injection into main recall path ──────────
# The registry was previously only reachable through narrative_retrieval's
# factual-intent classifier, which misses most query phrasings. Now it runs
# on every recall path: match query words against registry aliases, prepend
# matching canonical measurements to mem_ctx at highest priority.
_MEASUREMENT_REGISTRY_CACHE = None
_MEASUREMENT_REGISTRY_MTIME = 0

def _measurement_registry_lookup(query: str, max_results: int = 5) -> list[str]:
    """Match query against measurement registry aliases. Returns formatted strings."""
    global _MEASUREMENT_REGISTRY_CACHE, _MEASUREMENT_REGISTRY_MTIME
    reg_path = os.path.join(BASE, "data/measurement_registry.json")
    try:
        mtime = os.path.getmtime(reg_path)
        if _MEASUREMENT_REGISTRY_CACHE is None or mtime > _MEASUREMENT_REGISTRY_MTIME:
            with open(reg_path) as f:
                _MEASUREMENT_REGISTRY_CACHE = json.load(f)
            _MEASUREMENT_REGISTRY_MTIME = mtime
    except Exception:
        return []
    if not _MEASUREMENT_REGISTRY_CACHE:
        return []

    q_lower = query.lower()
    q_words = set(w.strip("?.,!\"'()") for w in q_lower.split() if len(w) >= 2)
    # Remove common stopwords
    q_words -= {'the', 'what', 'is', 'are', 'our', 'how', 'does', 'do', 'we',
                'have', 'any', 'about', 'for', 'and', 'can', 'you', 'tell',
                'me', 'much', 'many', 'find', 'show', 'give', 'list'}

    matches = []
    # M104: mirror of M102 narrative_retrieval fix. Some registry entries are
    # bare scalars (float/bool/str) written by pipeline tools (e.g.
    # tools/m96/m96_analyze.py) that bypass the dict schema. Skip them here
    # so a single malformed row can't abort the main recall path's
    # measurement-registry lookup with `'float' object has no attribute 'get'`.
    # Root cause + full catalog in
    # vault/agent_reports/m102_narrative_retrieval_fix.md.
    for key, entry in _MEASUREMENT_REGISTRY_CACHE.items():
        if not isinstance(entry, dict):
            continue
        aliases = [a.lower() for a in entry.get("aliases", [])]
        entity = entry.get("entity", "").lower()
        mtype = entry.get("measurement_type", "").lower().replace("_", " ")
        # Match: any alias appears in query OR entity word appears in query words
        alias_hit = any(a in q_lower for a in aliases)
        entity_hit = any(w in q_words for w in entity.split("_") if len(w) >= 2)
        type_hit = any(w in q_words for w in mtype.split() if len(w) >= 3)
        if alias_hit or (entity_hit and type_hit):
            val = entry.get("value", "?")
            unit = entry.get("unit", "")
            source = entry.get("source", "")
            label = key.replace(".", " ").replace("_", " ")
            matches.append(f"[MEASUREMENT] {label}: {val} {unit} ({source})")
    return matches[:max_results]


def _build_prior_turn_context(query: str, session_dir: str,
                                current_turn: int) -> str:
    """M53 P1: Inject prior-turn context when user references prior response.

    Detects referential queries ("you mentioned X", "earlier you said", "that",
    "the one you described") and reads the previous turn's log to extract the
    response + tool_result_preview. Returns a context block to inject so the
    model can answer based on what it actually said, not based on its current
    recall state (which may have shifted between turns).

    Fixes the T14-T15 false-self-accusation class: model forgets its own
    correct prior response when the recall surface changes between turns.
    """
    q_low = query.lower().strip()
    # Referential markers that signal "about my prior response"
    referential = [
        "you mentioned", "you said", "you described", "you wrote",
        "earlier you", "before you", "in your last", "in your response",
        "in your previous", "that response", "your answer",
        "what you just", "you just said", "you just mentioned",
        # Pronoun-only references to prior topic
        "but you ", "you did ",
    ]
    if not any(m in q_low for m in referential):
        return ""

    # Read the immediately prior turn's log
    prior_turn = current_turn - 1
    if prior_turn < 1:
        return ""

    try:
        import json as _json
        prior_path = os.path.join(session_dir,
                                  f"turn_{prior_turn:04d}.json")
        if not os.path.exists(prior_path):
            return ""
        with open(prior_path) as f:
            prior = _json.load(f)

        gen = prior.get("generation") or {}
        prior_resp = (gen.get("response_text") or "").strip()
        if not prior_resp:
            return ""

        routing = prior.get("routing") or {}
        tool_preview = (routing.get("tool_result_preview") or "")[:800]
        tool_called = routing.get("tools_called") or []

        # Truncate prior response to avoid context bloat
        resp_excerpt = prior_resp[:1500]
        if len(prior_resp) > 1500:
            resp_excerpt += "..."

        lines = [
            f"[PRIOR TURN {prior_turn} CONTEXT — user is referencing this]",
            f"Your previous response was:",
            f"  {resp_excerpt}",
        ]
        if tool_preview:
            lines.append(
                f"Based on tool result from {tool_called or 'recalled memory'}:")
            lines.append(f"  {tool_preview[:600]}")
        lines.append(
            "[Use this context to answer the follow-up. Do NOT claim you "
            "hallucinated prior content that was grounded in tool results "
            "or recalled memory.]")
        return "\n".join(lines)
    except Exception:
        return ""


def _build_per_query_block(filtered, query: str):
    """Format the per-query filtered memories as a text block that rides
    in the user-message tail (not the system slot) so the verifier's
    prefix KV cache stays byte-stable across turns.

    Returns None if filtered is empty or formatting fails.

    M122 A1: max_chars raised 1500 -> 2400 (K2 fires). Canonical-reserve
    in multi_path_retrieve.present() alone closes T51 (specific canonical
    at pool pos 4-5 now renders) but not T68 (query-relevant canonical at
    pool pos 5 is outscored by an off-topic canonical at pool pos 3 so
    reserve picks the wrong one). Raising to 2400 widens the budget so
    both the off-topic and query-relevant canonicals render. Prefix-cache
    impact measured in M122 A1 stream report; revert path on K1 is to
    narrow the canonical-reserve slot instead.
    """
    if not filtered:
        return None
    try:
        import sys as _sys
        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
        # m122_a3_zeta_v22: import the truncation-manifest accessor so the
        # per-query block can emit retrieval.per_query_truncated_items into
        # the turn log. Additive + optional per Pattern-1 guardrail.
        from multi_path_retrieve import present as _present
        try:
            from multi_path_retrieve import (
                get_last_truncated_items as _get_truncated,
            )
        except ImportError:
            _get_truncated = None  # older module without the accessor
        _block = _present(filtered, query, max_chars=2400)
        # m122_a3_zeta_v22: capture per-query truncation manifest from the
        # side channel. A1 will populate fuller heuristics; A3 ships the
        # field. Emit only when non-empty to keep turn JSONs lean on turns
        # where nothing was truncated (Pattern-1 conditional emit).
        if _get_truncated is not None:
            try:
                _truncated = _get_truncated()
                if _truncated:
                    _turn_record("retrieval",
                                 per_query_truncated_items=_truncated)
            except Exception:
                pass
        return _block
    except Exception:
        return None


# Phase 3: Warm Session Open — profile + bridge + pre-loaded cache
_warm_context = ""
try:
    from warm_open import session_open as _warm_session_open, get_warm_context, \
        get_cached_narrative, get_profile_boost, get_session_context_extension
    _warm_state = _warm_session_open()
    _warm_context = get_warm_context()
except Exception as _e:
    print(f"[warm_open] init failed: {_e}", flush=True)
    get_warm_context = lambda: ""
    get_cached_narrative = lambda q: None
    get_profile_boost = lambda q: 0.0
    get_session_context_extension = lambda: {}

# ── LLM ─────────────────────────────────────────────────────────────────────

def _derive_message_roles(messages):
    """Main 55 L2b: map each chat message to a source-role tag for the
    L2b attention-bias port (ngram-engine/l2b_attention_bias.py).

    Rules:
      - system message          → "canonical" (briefing + standing rules)
      - assistant history turn  → "assistant"
      - user history/final turn → "user"

    Matches the Qwen 0.5B prototype's role→weight distribution so the
    production port can be validated against those numbers directly.

    Main 55 close: DEFAULT DISABLED. Production A/B showed L2b with current
    weights (canonical +0.262) causes grounded confabulation on Gemma 4 31B
    — zero-fabrication rate collapses 5/5 → 1/5 (see
    vault/agent_reports/main55_p3_scoring.md). Re-enable only for targeted
    A/B runs with L2B_DISABLE_ROLES=0 after weights are re-tuned.
    """
    if os.environ.get("L2B_DISABLE_ROLES", "1") != "0":
        return None
    roles = []
    for m in messages:
        r = m.get("role", "user")
        if r == "system":
            roles.append("canonical")
        elif r == "assistant":
            roles.append("assistant")
        else:
            roles.append("user")
    return roles


def _llm_call(messages, max_tokens, temperature, stop=None,
              repetition_penalty=1.0, min_p=0.0, top_p=0.0):
    body = {
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "min_p": min_p, "top_p": top_p,
    }
    if stop:
        body["stop"] = stop
    # Main 55 L2b: attach per-message source-role tags. Server expands to
    # per-token via its chat template. When the server is unpatched or
    # rejects the field, it is ignored (OpenAI-compat extension).
    mr = _derive_message_roles(messages)
    if mr:
        body["message_roles"] = mr
    payload = json.dumps(body).encode()
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                MLX_BASE_URL.rstrip("/") + "/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "Connection": "close"})
            # Main 35 +2: bumped 120→900 per real-transcript prefill diagnosis
            # (m35-72b-real-transcript-wedge.md). Researcher tool calls
            # under 72B contention need long timeouts.
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read())
        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            retryable = any(k in err_name for k in
                          ['URL', 'Connection', 'Remote', 'Broken', 'Reset', 'EOF', 'Timeout'])
            if not retryable:
                raise
            time.sleep(1 + attempt)
    raise last_err


def llm_fn(messages, max_tokens=1500, temperature=0.7, stop=None, **_kw):
    global _last_stats
    data = _llm_call(messages, max_tokens, temperature, stop=stop)
    _last_stats = data.get("x_spec_decode", {})
    return data["choices"][0]["message"]["content"] or ""


def llm_stream(messages, max_tokens=600, temperature=0.7,
               repetition_penalty=1.0, min_p=0.0, top_p=0.0):
    """Streaming LLM call — yields text chunks as they arrive."""
    global _last_stats
    _body = {
        "model": MLX_MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "min_p": min_p, "top_p": top_p, "stream": True,
    }
    # Main 55 L2b: attach per-message source-role tags (see _llm_call).
    _mr = _derive_message_roles(messages)
    if _mr:
        _body["message_roles"] = _mr
    payload = json.dumps(_body).encode()
    req = urllib.request.Request(
        MLX_BASE_URL.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    # Main 35 +2: streaming reads — also bumped to 900s per the same diagnosis.
    resp = urllib.request.urlopen(req, timeout=900)

    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            stats = chunk.get("x_spec_decode")
            if stats:
                _last_stats = stats
            if content:
                yield content
        except json.JSONDecodeError:
            continue


def llm_route_fn(messages, max_tokens=120, temperature=0.0, **_kw):
    data = _llm_call(messages, max_tokens, temperature)
    return data["choices"][0]["message"]["content"] or ""


def _clean_response(text):
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    for special in ['<|endoftext|>', '<|im_end|>', '<|im_start|>',
                     '<|eot_id|>', '<|start_header_id|>', '<|end_header_id|>',
                     '<|begin_of_text|>', 'assistant']:
        text = text.replace(special, '')
    text = re.sub(r'\n(user|assistant|system)\s*$', '', text)
    # Main 22 follow-up: strip training-template token leakage. The 72B
    # sometimes generates "Human: ... Assistant: ..." patterns at the tail
    # of long completions, leaking its training format. Cut everything from
    # the first such marker onward.
    text = re.split(r'\n*(?:Human|User|### ?Human|### ?User|<\|eos\|>|</s>)\s*:?\s*', text, maxsplit=1)[0]
    text = text.rstrip()
    # Paragraph-level dedup
    lines = text.split('\n')
    seen = set()
    deduped = []
    for line in lines:
        key = line.strip().lower()
        if len(key) > 30 and key in seen:
            continue
        if len(key) > 30:
            seen.add(key)
        deduped.append(line)
    text = '\n'.join(deduped)
    # Phrase-level repetition — scan from long phrases down to 2-word
    # Main 42: lowered floor from 5 to 2 to catch n-gram drafter loops
    # like "the 8B the 8B the 8B..." which are 2-word repeats.
    words = text.split()
    for phrase_len in range(12, 1, -1):
        for i in range(len(words) - phrase_len):
            phrase = ' '.join(words[i:i + phrase_len])
            rest = ' '.join(words[i + phrase_len:])
            # For short phrases (<=3 words), require more repetitions to
            # avoid false positives on common bigrams like "the model"
            min_repeats = 3 if phrase_len <= 3 else 1
            if rest.count(phrase) >= min_repeats:
                first_end = text.find(phrase) + len(phrase)
                second_start = text.find(phrase, first_end)
                if second_start > 0:
                    text = text[:second_start].rstrip(' \n,-')
                    break
        else:
            continue
        break
    # Strip self-referential / meta-commentary sentences
    meta_patterns = [
        r"I rewrote.*", r"I rephrased.*", r"I made a mistake.*",
        r"I should have.*", r"Let me (?:give|rephrase|correct).*",
        r"I'll rephrase.*", r"\d+ priorities are set.*",
    ]
    for pat in meta_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)

    # Sentence-level dedup: if a sentence repeats (even with minor variations), keep first
    import difflib
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) > 2:
        kept = [sentences[0]]
        for s in sentences[1:]:
            is_dup = False
            for k in kept:
                ratio = difflib.SequenceMatcher(None, s.lower(), k.lower()).ratio()
                if ratio > 0.6:  # Tighter threshold
                    is_dup = True
                    break
            if not is_dup and len(s) > 5:
                kept.append(s)
        text = ' '.join(kept)

    # Final cleanup: remove double spaces and trailing fragments
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _regenerate_if_garbage(llm_fn, messages, response, max_retries=1):
    """Regenerate if response is garbage (too short, just a number, etc)."""
    if len(response.strip()) < 10 and not any(c.isalpha() for c in response[:5]):
        for _ in range(max_retries):
            retry = llm_fn(messages, max_tokens=300, temperature=0.5)
            if len(retry.strip()) >= 10:
                return retry
    return response


def _add_feed(event_type, text):
    _feed.insert(0, {
        "type": event_type,
        "text": text[:120],
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    if len(_feed) > 20:
        _feed.pop()


def _trim_history(history, max_count):
    if len(history) <= max_count:
        return history
    cut = len(history) - max_count
    dropped = history[:cut]
    parts = []
    for msg in dropped:
        c = msg.get("content", "")
        if not c:
            continue
        if msg["role"] == "user":
            parts.append(f"User: {c[:150]}")
        elif msg["role"] == "assistant":
            parts.append(f"Midas: {c[:150]}")
    summary = "Earlier in this conversation:\n" + "\n".join(parts[-6:]) if parts else ""
    kept = history[cut:]
    if summary:
        # Main 42: use "user" role, not "system". build_messages() puts the
        # real system message at index 0; a second system-role entry in
        # history causes Qwen's chat template to reject the request with
        # "System message must be at the beginning." This was the root cause
        # of T9/T10 empty responses in the IOReport session.
        kept.insert(0, {"role": "user", "content": f"[context] {summary}"})
    return kept


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    global _history, _last_subconscious, _session_briefing, _session_briefing_built_at_turn, _all_user_queries
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    inject_context = (data or {}).get("inject_context", True)
    if not message:
        return jsonify({"error": "empty message"}), 400

    with _lock:
        try:
            # Cancel any idle work — user message takes priority
            if _idle_queue:
                _idle_queue.cancel()

            _session["messages_sent"] += 1

            # Main 37 refactor: shared helper (also called from stream path).
            _update_context_tracker(message)

            # Main 37: recency short-circuit. "What did we talk about last"
            # goes straight to the most recent session_summaries/*.json and
            # returns its contents directly. No memory recall, no narrative
            # retrieval, no LLM synthesis. This is the route the directive
            # specifies: "route to the session summary directly, not through
            # narrative retrieval."
            # m120_b_phatic_anchor_check: conjunction check — phatic-recency
            # with a trailing topical anchor falls through to L2 instead of
            # firing the L0 short-circuit. Closes M119 T84.
            # M125.2 Stream D: turn-scope short-circuit. Independent
            # gate from phatic session-recency — fires on explicit
            # "N turns/messages/exchanges ago" queries and routes to
            # the within-session turn log rather than the cross-session
            # summary corpus. Checked BEFORE the session-scope gate so
            # turn-scope doesn't get swallowed by a session-phatic
            # false positive.
            _parsed_scoped = _parse_ordinal_nback(message)
            if (_parsed_scoped and _parsed_scoped[1] == "turn"
                    and _is_turn_ordinal_query(message)):
                _turn_n = _parsed_scoped[0]
                turn_record = load_nth_turn(_turn_n)
                turn_response = _build_turn_recency_response(turn_record)
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant",
                                 "content": turn_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _add_feed("recency",
                          f"turn-ago bridge: {message[:60]}")
                return jsonify({
                    "response": turn_response,
                    "tool": "turn_log_recall",
                    "tool_args": {
                        "turn_path": (turn_record or {}).get("_path"),
                        "nback": _turn_n,
                        "scope": "turn",
                        "total_available":
                            (turn_record or {}).get("_total_available", 0),
                    },
                    "route_layer": "L0",
                    "memories_recalled": [],
                    "stats": {},
                })

            if _m120_b_should_short_circuit(message):
                # M125.1 Stream B: if query carries ordinal N-back
                # ("three sessions ago", "2 sessions back"), load the
                # Nth-from-latest summary instead of [0]. If N exceeds
                # stored history, fall back to latest with caveat text.
                # M125.2 Stream D: classifier now returns (n, scope);
                # only session-scope reaches this branch (turn-scope
                # is short-circuited above).
                _parsed_sess = _parse_ordinal_nback(message)
                if (_parsed_sess and _parsed_sess[1] == "session"
                        and _parsed_sess[0] > 1):
                    recency_summary = load_nth_session_summary(
                        _parsed_sess[0])
                    if recency_summary is None:
                        recency_summary = load_last_session_summary()
                else:
                    recency_summary = load_last_session_summary()
                recency_response = _build_recency_response(recency_summary)
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant", "content": recency_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _add_feed("recency", f"prior session bridge: {message[:60]}")
                return jsonify({
                    "response": recency_response,
                    "tool": "session_summary_recall",
                    "tool_args": {
                        "summary_path": (recency_summary or {}).get("_path"),
                        "nback": (recency_summary or {}).get("_nback", 1),
                    },
                    "route_layer": "L0",
                    "memories_recalled": [],
                    "stats": {},
                })

            # 1. Ingest user message
            ingest_result = memory.ingest("user", message)
            if ingest_result.get("extracted", 0) > 0:
                _session["facts_extracted"] += ingest_result.get("extracted", 0)
                _add_feed("extraction", f"[user] {message[:80]}")

            # 2. Route
            # Main 62 Bug 1: pass the prior user turn so anaphoric
            # web-search follow-ups ("can you do a web search?") can
            # fold in the prior subject.
            # M125 A5.1: also pass `_last_tool` so anaphoric follow-ups
            # can re-fire the prior turn's tool with subject-fold-in.
            global _last_tool
            _prior_user = ""
            for _h in reversed(_history):
                if _h.get("role") == "user":
                    _prior_user = _h.get("content", "")
                    break
            l1_result = layer1_route(message,
                                      prior_user_message=_prior_user,
                                      prior_tool=_last_tool or "")
            tool_name, tool_args = route(
                message, llm_fn=llm_route_fn,
                prior_user_message=_prior_user,
                prior_tool=_last_tool or "")
            route_layer = "L1" if l1_result else "L2"

            # 3. Recall memories (subconscious) — skip for casual greetings
            mem_ctx = None
            presentation_briefing = None
            _last_subconscious = []
            _casual_greetings = {"hey","yo","hi","hello","sup","morning","evening",
                                 "k","ok","thanks","bye","later","night","yo!","hey!"}
            skip_memory = (not inject_context) or (message.lower().rstrip("!?., ") in _casual_greetings)
            try:
                if not skip_memory:
                    # Main 43 Phase 3: pass topic boost from context tracker
                    _ct = _get_context_tracker()
                    _cb = _ct.get_retrieval_boost() if _ct else None
                    # m118_e_sub_shape_2: fold antecedent into recall query
                    # when user uses a demonstrative/pronoun.
                    _recall_q = _m118_e_expand_query_with_antecedent(
                        message, _history)
                    recall_result = memory.recall(_recall_q, n_results=15,
                                                  context_boost=_cb)
                else:
                    recall_result = None
                if recall_result and recall_result.get("results"):
                    filtered = []
                    for r in recall_result["results"]:
                        if r.get("score", 0) < 0.40:
                            continue
                        text = r.get("text", "")
                        if text.startswith("[") and ".md]" in text[:50]:
                            continue
                        if text.strip().endswith("?"):
                            continue
                        filtered.append(r)
                        if len(filtered) >= 12:
                            break
                    # Dedup: cluster similar memories, keep most specific.
                    # Main 24 Build 1: compare *body* not prefix. Meta memories
                    # share a fixed "Session work on YYYY-MM-DD ... — shipped,
                    # changed, decided: " prefix that made every meta entry
                    # dedup against the first one. Strip a leading prefix
                    # (everything before the first ": ") before comparing.
                    import difflib as _dl
                    def _dedup_body(t: str) -> str:
                        s = t.lower().strip()
                        if ": " in s[:140]:
                            s = s.split(": ", 1)[1]
                        return s[:160]
                    deduped = []
                    for r in filtered:
                        body = _dedup_body(r["text"])
                        is_dup = False
                        for kept in deduped:
                            ratio = _dl.SequenceMatcher(None, body,
                                _dedup_body(kept["text"])).ratio()
                            if ratio > 0.78:
                                is_dup = True
                                break
                        if not is_dup:
                            deduped.append(r)
                    filtered = deduped[:8]
                    mem_ctx = [r["text"] for r in filtered]
                    _emit_subconscious_event(
                        "memory_recalled",
                        query=message[:100],
                        top_k=len(filtered),
                        top_score=round(filtered[0].get("score", 0), 3) if filtered else 0,
                        path="api_chat")
                    # Main 35 +5 Tier 1: emit one per-memory event so the
                    # viz can animate the retrieval flow as N sequential
                    # node activations rather than a single top-1 flash.
                    #
                    # Rich payload (post Tier 1 fallback fix): include topic,
                    # source_role, type, text so the viz can do 3-tier
                    # fallback matching when `file` is empty (legacy memories
                    # have no source file). Viz tries file → text fuzzy →
                    # topic cluster pulse in that order.
                    for _i, _r in enumerate(filtered):
                        # MemoryBridge.recall flattens the result — fields
                        # live at the top level, not under 'metadata'.
                        _meta = _r.get("metadata", {}) or {}  # legacy fallback
                        _src = (_r.get("source") or _r.get("file")
                                or _meta.get("source") or _meta.get("file") or "")
                        _basename = _src.rsplit("/", 1)[-1] if _src else ""
                        # Derive a "topic" from entities/type since the
                        # bridge doesn't pass through topic directly. Use
                        # query_category as a hint if present.
                        _entities = _r.get("entities") or []
                        _topic_hint = _r.get("query_category", "") or ""
                        _emit_subconscious_event(
                            "memory_recalled_item",
                            index=_i,
                            of=len(filtered),
                            score=round(_r.get("score", 0), 3),
                            file=_basename,
                            text=_r.get("text", "")[:200],
                            topic=_topic_hint,
                            source_role=(_r.get("source_role")
                                          or _meta.get("source_role", "")) or "",
                            mem_type=(_r.get("type")
                                       or _meta.get("type", "")) or "",
                            entities=_entities[:5] if isinstance(_entities, list) else [],
                            query=message[:80],
                        )
                    _last_subconscious = [
                        {"text": r["text"][:200], "score": r["score"]}
                        for r in filtered
                    ]
                    if mem_ctx:
                        _session["memories_recalled"] += len(mem_ctx)
                        for m in mem_ctx[:3]:
                            _add_feed("recall", m[:80])
                # Phase 2c: measurement registry injection (Main 41)
                try:
                    _reg_matches = _measurement_registry_lookup(message)
                    if _reg_matches:
                        if mem_ctx:
                            mem_ctx = _reg_matches + mem_ctx
                        else:
                            mem_ctx = _reg_matches
                except Exception:
                    pass

                # Main 37 refactor: shared helpers (also called from stream path).
                presentation_briefing = _build_presentation_briefing(
                    is_first_message=(_session.get("messages_sent") == 1)
                )
                per_query_block = None
                if not skip_memory and recall_result and recall_result.get("results"):
                    per_query_block = _build_per_query_block(filtered, message)
            except Exception:
                pass

            # 4. Execute tool or direct conversation
            tool_result = None
            # M125 A5.2: meta-instruction acknowledge-only short-circuit.
            # tool_name == "meta_instruction_ack" means L1 detected a
            # standing instruction / behavior-modification cue. We skip
            # generation entirely and return the prebuilt acknowledgement
            # so the model does NOT treat the instruction as a content
            # query. Ship level per K15: ACK ONLY — commit-to-rules is
            # separate work.
            if tool_name == "meta_instruction_ack":
                _meta_response = (tool_args or {}).get(
                    "response",
                    "Understood — I've noted that instruction.")
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant",
                                 "content": _meta_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _last_tool = "meta_instruction_ack"
                return jsonify({
                    "response": _meta_response,
                    "tool": tool_name,
                    "tool_args": tool_args,
                    "route_layer": route_layer,
                    "memories_recalled": [],
                    "stats": {},
                })
            if tool_name != "conversation":
                _session["tools_used"] += 1
                tool_result = execute(tool_name, tool_args)

            # 5. Synthesize response
            # Storage operations skip the LLM entirely. The tool already did the
            # work; the LLM has nothing to add and would bleed prior context
            # into the acknowledgement. Hard-code the ack with the extracted count.
            if tool_name == "memory_ingest":
                extracted = 0
                if isinstance(tool_result, dict):
                    extracted = tool_result.get("extracted", 0)
                if extracted > 0:
                    response = (f"Got it. Stored {extracted} fact"
                                f"{'s' if extracted != 1 else ''}.")
                else:
                    response = "Got it. Noted."
            elif tool_name == "conversation":
                # Main 22 follow-up: casual greetings bypass the LLM entirely.
                # With no memories injected (skip_memory=True) and a very
                # short user message, the 72B free-associates into unrelated
                # content (Chinese COVID text, Hungarian translations, etc.)
                # — even with a tight max_tokens clamp. Same structural fix
                # as the memory_ingest LLM bypass: hardcode the response.
                # The agent has nothing useful to add to a "hey" anyway.
                if skip_memory:
                    import random as _rnd
                    _greetings = [
                        "Hey Nick. What's up?",
                        "Hey. What can I help with?",
                        "Hey Nick. Need an update or have a specific question?",
                        "Hey. Ready when you are.",
                        "Hey Nick. What do you want to dig into?",
                    ]
                    response = _rnd.choice(_greetings)
                else:
                    # Main 25 Build 0: per-query memories go into the user
                    # message (tail) so the system message stays byte-stable
                    # for the verifier's prefix cache. Memory_context kwarg
                    # is dropped to avoid the synthesizer's fallback path
                    # also injecting them into system.
                    # Main 41: measurement registry block
                    _reg_block = None
                    try:
                        _reg_m = _measurement_registry_lookup(message)
                        if _reg_m:
                            _reg_block = "\n".join(_reg_m)
                    except Exception:
                        pass
                    augmented = message
                    if _reg_block and per_query_block:
                        augmented = f"{_reg_block}\n\n{per_query_block}\n\n---\n\n{message}"
                    elif _reg_block:
                        augmented = f"{_reg_block}\n\n---\n\n{message}"
                    elif per_query_block:
                        augmented = f"{per_query_block}\n\n---\n\n{message}"
                    # m113_alpha_history_filter: see /api/chat/stream wiring
                    # above for rationale. Same detector, same shape (a).
                    try:
                        from history_reinforcement_filter import (
                            filter_history_for_reinforcement as _m113_filter,
                            count_marked as _m113_count_marked,
                        )
                        _history_for_prompt = _m113_filter(
                            _history, current_query=message)
                        _turn_record(
                            "history_filter",
                            marked=_m113_count_marked(_history_for_prompt),
                            total_history_turns=len(_history),
                            filter_shape="mark_low_confidence",
                            variant="m113_alpha",
                        )
                    except Exception as _m113_err:
                        print(f"[m113_alpha_history_filter] bypass: {_m113_err}",
                              flush=True)
                        _history_for_prompt = _history
                    response = synthesize(llm_fn, _history_for_prompt, augmented,
                                          temperature=0.3,
                                          briefing=presentation_briefing,
                                          standing_rules=_ACTIVE_STANDING_RULES)
            else:
                augmented = message
                if per_query_block:
                    augmented = (f"{per_query_block}\n\n---\n\n{message}")
                # m113_alpha_history_filter: non-conversation (tool) path.
                try:
                    from history_reinforcement_filter import (
                        filter_history_for_reinforcement as _m113_filter,
                        count_marked as _m113_count_marked,
                    )
                    _history_for_prompt = _m113_filter(
                        _history, current_query=message)
                    _turn_record(
                        "history_filter",
                        marked=_m113_count_marked(_history_for_prompt),
                        total_history_turns=len(_history),
                        filter_shape="mark_low_confidence",
                        variant="m113_alpha",
                    )
                except Exception as _m113_err:
                    print(f"[m113_alpha_history_filter] bypass: {_m113_err}",
                          flush=True)
                    _history_for_prompt = _history
                response = synthesize(
                    llm_fn, _history_for_prompt, augmented,
                    tool_name=tool_name, tool_args=tool_args,
                    tool_result=tool_result, temperature=0.3,
                    max_tokens=1500,
                    briefing=presentation_briefing,
                    standing_rules=_ACTIVE_STANDING_RULES,
                )

            response = _clean_response(response)
            # Regenerate if garbage (too short, just a number) — but skip
            # for memory_ingest, which intentionally returns a short ack.
            if len(response.strip()) < 10 and tool_name != "memory_ingest":
                # m113_alpha_history_filter: regeneration path inherits the
                # same filtered history as the first attempt.
                response = synthesize(
                    llm_fn, _history_for_prompt, augmented,
                    tool_name=tool_name if tool_name != "conversation" else None,
                    tool_args=tool_args if tool_name != "conversation" else None,
                    tool_result=tool_result if tool_name != "conversation" else None,
                    temperature=0.5,
                    briefing=presentation_briefing,
                    standing_rules=_ACTIVE_STANDING_RULES)
                response = _clean_response(response)
            if not response and tool_result:
                response = str(tool_result)[:2000]
            if not response or len(response.strip()) < 2:
                response = "I don't have information about that right now."

            # 6. Update history
            _all_user_queries.append(message)
            _history.append({"role": "user", "content": message})
            _history.append({"role": "assistant", "content": response})
            _history = _trim_history(_history, MAX_HISTORY)

            # 7. Ingest assistant response — Main 39 P2 gates
            # conversation-route paraphrase turns. Same gate as the
            # streaming endpoint.
            # Main 45: pass recall_filtered texts as grounding context.
            # Assistant-sourced facts whose entities don't appear in
            # recall context are discarded (training-distribution hallucination).
            ai_result = {}
            ai_skip, ai_skip_reason = _should_skip_assistant_ingest(_TURN_LOG)
            if response and not ai_skip:
                _recall_texts = [r.get("text", "") for r in
                                 (_TURN_LOG.get("retrieval", {})
                                  .get("recall_filtered") or [])]
                ai_result = memory.ingest("assistant", response,
                                          recall_context=_recall_texts or None)
                _extracted_count = ai_result.get("extracted", 0)
                _turn_record("extraction",
                             count=int(_extracted_count),
                             types=ai_result.get("types", []))
                if _extracted_count > 0:
                    _session["facts_extracted"] += _extracted_count
                    _add_feed("extraction", f"[midas] {response[:80]}")
                # Main 45: log discarded ungrounded facts
                if ai_result.get("discarded_ungrounded"):
                    _turn_record("extraction",
                                 discarded_ungrounded=ai_result["discarded_ungrounded"])

            # 8. Stats from last LLM call
            stats = {}
            s = _last_stats
            if s:
                stats = {
                    "tps": s.get("tps", 0),
                    "tokens": s.get("tokens", 0),
                    "accept_rate": s.get("accept_rate", 0),
                    "ngram_drafted": s.get("ngram_drafted", 0),
                    "ngram_accepted": s.get("ngram_accepted", 0),
                    "cpu_drafted": s.get("cpu_drafted", 0),
                    "cpu_accepted": s.get("cpu_accepted", 0),
                    "elapsed": s.get("elapsed", 0),
                }

            # Schedule idle tasks (contradiction scan + retrieval scoring)
            if _idle_queue and response:
                _idle_queue.schedule(
                    injected_memories=_last_subconscious,
                    response_text=response)

            # M125 A5.1: stash the tool dispatched this turn so the next
            # anaphoric / subject-continuity follow-up can re-fire it
            # with subject composed in.
            _last_tool = tool_name or ""

            return jsonify({
                "response": response,
                "tool": tool_name,
                "tool_args": tool_args,
                "route_layer": route_layer,
                "memories_recalled": _last_subconscious,
                "stats": stats,
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500


# ── /api/research — Midas Researcher v1 (Main 27 close) ────────────────────
#
# Agentic read-only research loop. The 70B verifier on :8899 plans tool calls,
# we dispatch them, append results to the rolling context, re-prompt, and stop
# when FINAL_REPORT: is emitted or the iteration cap is hit. Tools are read-
# only (grep, read_file, list_dir, recall_memory) and sandboxed to ~/cowork.
#
# Designed to be delegated to from external agents (e.g. CC subagent loop):
# POST a research goal, get back {report, trace, iterations, elapsed_s}. The
# trace lets the caller audit what Midas actually verified vs hallucinated.
#
# Calibration history lives in vault/CLAUDE_session_log.md and the auto-memory
# `finding_midas_calibration.md`. The first calibration question is the 8B
# fusion target enumeration that motivated this build.

RESEARCHER_SYSTEM_PROMPT = """You are Midas Researcher, an agentic research assistant with read-only access to a software-engineering codebase under /Users/midas/Desktop/cowork. Your job is to investigate a research question by iteratively calling tools, gathering evidence, and producing a final report grounded in real file:line citations.

""" + research_tools.TOOL_CATALOG + """

Rules:
1. Call ONE tool per turn. Wait for the result before calling the next.
2. Plan first. Before your first tool call, write a 1-3 sentence plan.
3. Cite file:line for every code claim in the final report. No fabricated citations.
4. Memory recall is for priors; current code state MUST be verified with grep or read_file.
5. If a sub-claim cannot be verified after 3 tool calls, mark it UNVERIFIED in the report and move on. Do not loop.
6. **STOP CRITERION: as soon as you have ONE concrete grep or read_file result that supports or refutes a claim, that claim is RESOLVED. Do not seek confirmation from a second source. Emit FINAL_REPORT: as soon as every claim is resolved or marked UNVERIFIED.** Padding is failure. Looping past resolution is failure.
7. The iteration cap is 18 turns and the wall clock cap is 480 seconds. Budget accordingly.
8. The FINAL_REPORT marker MUST be at the start of a line. Do not put the literal string "FINAL_REPORT:" anywhere else in your output until you are actually finalizing.
"""

RESEARCHER_MAX_ITERS = 18
RESEARCHER_MAX_WALL_S = 480
RESEARCHER_TOOL_RESULT_CHARS = 6000  # truncate big results to keep context manageable


def _strip_chat_template_tokens(text):
    """Remove Qwen/Llama special tokens and any post-EOS hallucinated turns.

    Calibration test 1 showed the 70B emits `<|im_end|>` after a tool call
    JSON and then hallucinates a fake `Human: ... Assistant: ...` turn from
    its training distribution. We need to truncate at the first EOS marker
    before parsing TOOL_CALL / FINAL_REPORT, otherwise the parser sees noise
    and the FINAL_REPORT regex matches echoed nudge-prompt text.
    """
    if not text:
        return ""
    for marker in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    # Also cut at any post-completion fake human turn
    text = re.split(r'\n*(?:Human|User|### ?Human|### ?User)\s*:\s*', text, maxsplit=1)[0]
    return text.strip()


def _parse_tool_call(text):
    """Extract a TOOL_CALL: {...} JSON object or detect FINAL_REPORT:.

    Returns ('tool', tool_name, args) | ('final', report_text) | ('none', raw_text).

    Calibration test 1 fixes:
    - Strip chat-template special tokens BEFORE parsing (model emits
      `<|im_end|>` immediately after the JSON brace, no newline).
    - FINAL_REPORT match requires the marker at the START of a line (^), so
      echoed nudge-prompt text containing the substring doesn't false-match.
    - TOOL_CALL terminator accepts EOL OR EOS — JSON object boundary is
      detected by balanced-brace counting, not by the trailing newline.
    """
    if not text:
        return ("none", "")
    text = _strip_chat_template_tokens(text)
    if not text:
        return ("none", "")

    # Final report? Marker MUST be at start of a line (avoids echoed prompt match).
    final_match = re.search(r'(?:^|\n)\s*FINAL_REPORT:\s*(.*)', text, re.DOTALL)
    if final_match:
        report = final_match.group(1).strip()
        if report:
            return ("final", report)

    # Tool call: find "TOOL_CALL:" then parse the next balanced JSON object.
    tc_idx = text.find("TOOL_CALL:")
    if tc_idx >= 0:
        # Skip past "TOOL_CALL:" and any whitespace
        cursor = tc_idx + len("TOOL_CALL:")
        while cursor < len(text) and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            # Walk balanced braces to find the JSON object end
            depth = 0
            in_string = False
            escape = False
            end = cursor
            for i in range(cursor, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_blob = text[cursor:end]
            try:
                obj = json.loads(json_blob)
                tool = obj.get("tool")
                args = obj.get("args", {})
                if tool:
                    return ("tool", tool, args)
            except json.JSONDecodeError:
                pass
    return ("none", text)


@app.route("/api/research", methods=["POST"])
def api_research():
    """Run a directed research loop with read-only tools.

    Request: {"goal": "...", "max_iters": 15 (optional), "max_wall_s": 240 (opt)}
    Response: {"report": "...", "trace": [...], "iterations": N, "elapsed_s": F,
               "stop_reason": "final|cap|wall|error", "tool_calls": N}
    """
    data = request.get_json() or {}
    goal = (data.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "empty goal"}), 400

    max_iters = min(int(data.get("max_iters", RESEARCHER_MAX_ITERS)), 30)
    max_wall = min(float(data.get("max_wall_s", RESEARCHER_MAX_WALL_S)), 600)

    started = time.time()
    trace = []
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research goal:\n\n{goal}\n\nBegin with your plan, then your first tool call."},
    ]

    stop_reason = "cap"
    final_report = None
    tool_call_count = 0

    for iteration in range(1, max_iters + 1):
        if time.time() - started > max_wall:
            stop_reason = "wall"
            break

        try:
            # Calibration test 3 finding: qwen_spec_decode_server.py ignores
            # the OpenAI-compat `stop` parameter, so the 70B keeps generating
            # fabricated training-data Q&A after every <|im_end|> token (5-10x
            # token waste per turn). max_tokens=300 caps the bleeding — the
            # actual TOOL_CALL JSON or FINAL_REPORT line fits in <250 tokens.
            # The trailing garbage gets stripped client-side by _parse_tool_call.
            # On the FINAL turn we want the full report so detect "evidence
            # gathered" by allowing more tokens once we've made >=2 tool calls.
            this_turn_max = 300 if tool_call_count < 2 else 800
            assistant_text = llm_fn(messages, max_tokens=this_turn_max, temperature=0.2,
                                    stop=["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:"])
        except Exception as e:
            trace.append({"iter": iteration, "kind": "error", "detail": f"LLM call failed: {type(e).__name__}: {e}"})
            stop_reason = "error"
            break

        parsed = _parse_tool_call(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        trace.append({"iter": iteration, "kind": "assistant", "text": assistant_text})

        if parsed[0] == "final":
            final_report = parsed[1]
            stop_reason = "final"
            break
        elif parsed[0] == "tool":
            _, tool_name, args = parsed
            tool_call_count += 1
            try:
                result = research_tools.dispatch(tool_name, args, memory_bridge=memory)
            except Exception as e:
                result = f"ERROR: tool dispatch raised {type(e).__name__}: {e}"
            if len(result) > RESEARCHER_TOOL_RESULT_CHARS:
                result = result[:RESEARCHER_TOOL_RESULT_CHARS] + f"\n... (truncated; total was {len(result)} chars)"
            trace.append({"iter": iteration, "kind": "tool", "tool": tool_name, "args": args, "result": result})
            messages.append({
                "role": "user",
                "content": f"TOOL_RESULT ({tool_name}):\n{result}\n\nContinue. Call the next tool, or emit FINAL_REPORT: when done."
            })
        else:
            # Model emitted neither TOOL_CALL nor FINAL_REPORT — nudge it.
            # Calibration test 1: avoid putting the literal terminator strings
            # in the nudge prompt because the model may echo them back and
            # cause false-match exits. Refer to them obliquely instead.
            trace.append({"iter": iteration, "kind": "nudge"})
            messages.append({
                "role": "user",
                "content": "I could not parse a tool invocation or terminator from your last message. Re-emit either a tool call line (the TOOL underscore CALL prefix followed by a single JSON object) or a final-report line (the FINAL underscore REPORT prefix at the start of a line, followed by your report body). Use those exact prefixes with the underscores removed when you actually emit them. Continue."
            })

    elapsed = time.time() - started
    return jsonify({
        "report": final_report or "(no final report — loop ended via " + stop_reason + ")",
        "trace": trace,
        "iterations": iteration,
        "tool_calls": tool_call_count,
        "elapsed_s": round(elapsed, 2),
        "stop_reason": stop_reason,
    })


# ── /api/research/queue — Main 30 B2 ───────────────────────────────────────
#
# Queue multiple research tasks for sequential overnight processing. The 70B
# server can only serve one /api/research at a time (single-threaded MLX
# generation), so the queue runs tasks one after another in a background
# thread and writes each result to vault/agent_reports/queue/<queue_id>_*.md.
#
# Result files use wiki-link convention so they're picked up by the Main 30
# B1 realtime enricher and become recallable from Subconscious immediately
# after each task completes.
#
# v1: in-memory queue, lost on server restart. Future: persist to disk.

import threading
import uuid as _uuid
# Main 34 S2A: disk-backed queue lifecycle log so pending tasks survive restart.
import queue_persistence as _qp


def _emit_subconscious_event(ev_type: str, **details) -> None:
    """Main 34 S3 KT4: fire-and-forget POST to subconscious_daemon event bus.
    Daemon down = no event, never blocks the chat path.

    Main 37: also counts events into the current turn log so the per-turn
    session record captures daemon-event cardinality without requiring a
    separate WebSocket observer.
    """
    # Turn-log side effect: count events by type in the current turn.
    _turn_count_event(ev_type)
    try:
        import urllib.request as _ur
        body = json.dumps({
            "type": ev_type, "component": "midas_ui", "details": details
        }).encode()
        req = _ur.Request(
            "http://127.0.0.1:8454/api/subconscious/emit",
            data=body, headers={"Content-Type": "application/json"})
        _ur.urlopen(req, timeout=0.5).read()
    except Exception:
        pass

_queue_state: dict = {}  # queue_id → {tasks, completed, status, result_paths}
_queue_lock = threading.Lock()
_queue_thread: threading.Thread | None = None
QUEUE_DIR = "/Users/midas/Desktop/cowork/vault/agent_reports/queue"
PER_TASK_TIMEOUT_S = 3600  # Main 35 +4: 600→3600. The Researcher needs
                            # ~30-50 min on real-content paginated reads of
                            # 100KB+ JSON files. Same class as the _llm_call
                            # 120→900 fix from S+2 — original 600s was a
                            # synthetic-input estimate that doesn't survive
                            # real prefill latency.


def _process_queue(queue_id: str):
    """Worker thread: process all tasks in the queue sequentially.

    Wraps the body in try/except so any crash updates state to 'failed'
    rather than leaving it stuck in 'running'.

    Main 34 S4 P1: honors `idx_offset` from state (set by recovery so the
    resumed worker writes lifecycle records and result filenames using
    the GLOBAL task index from the original enqueue, not its local index
    over the remaining-tasks slice. Without this, a second crash-resume
    cycle would scramble the lifecycle log.
    """
    try:
        _process_queue_inner(queue_id)
    except Exception as e:
        with _queue_lock:
            state = _queue_state.get(queue_id)
            if state is not None:
                state["status"] = "failed"
                state["error"] = f"{type(e).__name__}: {e}"
                state["finished_at"] = datetime.now().isoformat()
        try: _qp.log_fail(queue_id, f"{type(e).__name__}: {e}")
        except Exception: pass
        traceback.print_exc()


def _process_queue_inner(queue_id: str):
    """Worker body — sequential task processing with per-task timeout."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    state = _queue_state[queue_id]
    # P1: when resumed, log/filename indices use the global numbering.
    idx_offset = state.get("idx_offset", 0)
    summary_lines = [
        f"# Research queue {queue_id}",
        f"",
        f"> Started: {state['started_at']}",
        f"> Tasks: {len(state['tasks'])}",
        f"",
    ]

    for i, task in enumerate(state["tasks"]):
        query = task.get("query", "")
        if not query:
            continue
        global_idx = i + idx_offset
        with _queue_lock:
            state["current"] = i
            state["status"] = "running"
        try: _qp.log_start(queue_id, global_idx)
        except Exception: pass

        task_started = time.time()
        # Build a synthetic /api/research call inline. Reuse the same loop driver.
        messages = [
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research goal:\n\n{query}\n\nBegin with your plan, then your first tool call."},
        ]
        trace = []
        final_report = None
        stop_reason = "cap"
        tool_call_count = 0
        max_iters = min(int(task.get("max_iters", 18)), 30)
        for iteration in range(1, max_iters + 1):
            if time.time() - task_started > PER_TASK_TIMEOUT_S:
                stop_reason = "timeout"
                break
            try:
                this_turn_max = 300 if tool_call_count < 2 else 800
                assistant_text = llm_fn(messages, max_tokens=this_turn_max, temperature=0.2,
                                        stop=["<|im_end|>", "<|endoftext|>", "\nHuman:", "\nUser:"])
            except Exception as e:
                trace.append({"iter": iteration, "kind": "error", "detail": f"{type(e).__name__}: {e}"})
                stop_reason = "error"
                break
            parsed = _parse_tool_call(assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
            trace.append({"iter": iteration, "kind": "assistant", "text": assistant_text})
            if parsed[0] == "final":
                final_report = parsed[1]
                stop_reason = "final"
                break
            elif parsed[0] == "tool":
                _, tool_name, args = parsed
                tool_call_count += 1
                try:
                    result = research_tools.dispatch(tool_name, args, memory_bridge=memory)
                except Exception as e:
                    result = f"ERROR: {type(e).__name__}: {e}"
                if len(result) > RESEARCHER_TOOL_RESULT_CHARS:
                    result = result[:RESEARCHER_TOOL_RESULT_CHARS] + "\n... (truncated)"
                trace.append({"iter": iteration, "kind": "tool", "tool": tool_name, "args": args, "result": result})
                messages.append({"role": "user", "content": f"TOOL_RESULT ({tool_name}):\n{result}\n\nContinue. Call the next tool, or emit FINAL_REPORT: when done."})
            else:
                trace.append({"iter": iteration, "kind": "nudge"})
                messages.append({"role": "user", "content": "I could not parse a tool invocation or terminator. Re-emit either a tool call line or a final-report line."})

        elapsed = time.time() - task_started
        # Write result file (P1: global index for resume safety)
        result_path = f"{QUEUE_DIR}/{queue_id}_task_{global_idx+1:02d}.md"
        report_text = final_report or f"(no final report — stop_reason={stop_reason})"
        body = [
            f"# Queue task {i+1}/{len(state['tasks'])}",
            f"",
            f"> queue: [[{queue_id}_summary|{queue_id}]]",
            f"> stop_reason: {stop_reason} | iterations: {iteration} | tool_calls: {tool_call_count} | elapsed: {elapsed:.1f}s",
            f"",
            f"## Query",
            f"",
            f"```",
            query,
            f"```",
            f"",
            f"## Final Report",
            f"",
            report_text,
            f"",
            f"## Tool Trace",
            f"",
        ]
        for t in trace:
            if t.get("kind") == "tool":
                body.append(f"- **i{t['iter']} {t['tool']}** `{json.dumps(t.get('args', {}))[:140]}`")
                body.append(f"  - {(t.get('result') or '')[:240].replace(chr(10), ' | ')}")
            elif t.get("kind") == "nudge":
                body.append(f"- i{t['iter']} NUDGE")
            elif t.get("kind") == "error":
                body.append(f"- i{t['iter']} ERROR: {t.get('detail', '')[:200]}")
        try:
            with open(result_path, "w") as fh:
                fh.write("\n".join(body) + "\n")
        except OSError as e:
            print(f"queue: failed to write {result_path}: {e}")
        with _queue_lock:
            state["completed"] = i + 1
            state["result_paths"].append(result_path)
        try: _qp.log_task_complete(queue_id, global_idx, result_path)
        except Exception: pass
        summary_lines.append(f"- Task {i+1}: stop={stop_reason} elapsed={elapsed:.1f}s [[{Path(result_path).stem}]]")

    # Write summary file
    summary_lines.append("")
    summary_lines.append(f"> Finished: {datetime.now().isoformat()}")
    summary_path = f"{QUEUE_DIR}/{queue_id}_summary.md"
    try:
        with open(summary_path, "w") as fh:
            fh.write("\n".join(summary_lines) + "\n")
    except OSError as e:
        print(f"queue: failed to write summary {summary_path}: {e}")
    with _queue_lock:
        state["status"] = "complete"
        state["summary_path"] = summary_path
        state["finished_at"] = datetime.now().isoformat()
    try: _qp.log_complete(queue_id)
    except Exception: pass


@app.route("/api/research/queue", methods=["POST"])
def api_research_queue():
    """Queue multiple research tasks for sequential processing.

    Body: {"tasks": [{"query": "..."}, {"query": "..."}, ...]}
    Response: {"queue_id": "...", "task_count": N, "status": "queued"}
    """
    global _queue_thread
    data = request.get_json() or {}
    tasks = data.get("tasks") or []
    if not tasks or not isinstance(tasks, list):
        return jsonify({"error": "tasks must be a non-empty list"}), 400

    queue_id = "q" + datetime.now().strftime("%Y%m%d%H%M%S")
    state = {
        "queue_id": queue_id,
        "tasks": tasks,
        "completed": 0,
        "current": 0,
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "result_paths": [],
    }
    with _queue_lock:
        _queue_state[queue_id] = state
    try: _qp.log_enqueue(queue_id, tasks)
    except Exception: pass

    t = threading.Thread(target=_process_queue, args=(queue_id,), daemon=True)
    t.start()

    return jsonify({"queue_id": queue_id, "task_count": len(tasks), "status": "queued"})


@app.route("/api/research/queue/<queue_id>", methods=["GET"])
def api_research_queue_status(queue_id):
    with _queue_lock:
        state = _queue_state.get(queue_id)
    if not state:
        return jsonify({"error": f"unknown queue {queue_id}"}), 404
    return jsonify({
        "queue_id": state["queue_id"],
        "completed": state["completed"],
        "total": len(state["tasks"]),
        "status": state["status"],
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "summary_path": state.get("summary_path"),
        "result_paths": state.get("result_paths", []),
    })


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    """Streaming chat — SSE response with tokens as they generate."""
    global _history, _last_subconscious, _all_user_queries
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    inject_context = (data or {}).get("inject_context", True)
    if not message:
        return jsonify({"error": "empty message"}), 400

    def generate_sse():
        global _history, _last_subconscious
        with _lock:
            if _idle_queue:
                _idle_queue.cancel()
            _session["messages_sent"] += 1

            # Main 37: initialize the per-turn session log before any
            # pipeline work so every stage can record into it.
            _turn_start(message)

            # Main 46: Pipeline extraction. Fire 8B on the PREVIOUS
            # turn's assistant response at the START of this turn,
            # so ANE extracts while GPU generates. The 8B and GPU
            # don't contend (-4.7% measured, noise floor).
            _pipeline_extraction_thread = None
            _pipeline_start_ts = None
            _pending = getattr(api_chat_stream, '_pending_response', None)
            if os.environ.get("DISABLE_PIPELINE"):
                _pending = None  # P2 harness: disable pipeline for A/B
            if _pending and _pending.get("text"):
                import threading as _thr
                _pipeline_start_ts = time.time()
                def _pipeline_extract(resp_text, recall_ctx):
                    try:
                        _ai_result = memory.ingest("assistant", resp_text,
                                      recall_context=recall_ctx)
                        # M104 F1: mirror /api/chat midas-extraction feed
                        # write. Pipeline extraction is async so the feed
                        # entry lands on the NEXT turn's start, tagged
                        # with the stashed response text.
                        try:
                            if isinstance(_ai_result, dict) and _ai_result.get("extracted", 0) > 0:
                                _session["facts_extracted"] = _session.get("facts_extracted", 0) + int(_ai_result["extracted"])
                                _add_feed("extraction", f"[midas] {resp_text[:80]}")
                        except Exception:
                            pass
                    except Exception as _pe:
                        print(f"[pipeline] extraction error: {_pe}",
                              flush=True)
                _pipeline_extraction_thread = _thr.Thread(
                    target=_pipeline_extract,
                    args=(_pending["text"],
                          _pending.get("recall_context")),
                    daemon=True)
                _pipeline_extraction_thread.start()
                api_chat_stream._pending_response = None
                _turn_record("extraction",
                             fired=True,
                             pipeline_fired=True,
                             pipeline_target=f"turn_{_pending.get('turn', '?')}_assistant",
                             pipeline_start_ts=_pipeline_start_ts)

            # Main 37 refactor: same context-tracker update as /api/chat.
            # Prior to this, terminal-driven traffic never fed the tracker,
            # so active_topic stayed None and the viz HUD was blind.
            _update_context_tracker(message)
            try:
                _tracker = _get_context_tracker()
                if _tracker is not None:
                    _turn_record("retrieval", context_tracker={
                        "active_topics": [
                            {"topic": t, "weight": round(w, 3)}
                            for t, w in _tracker.active_topics
                        ],
                        "message_count": _tracker.message_count,
                    })
            except Exception:
                pass

            # Ingest
            _ingest_start = time.time()
            ingest_result = memory.ingest("user", message)
            _ingest_end = time.time()
            _turn_record("extraction",
                         user_ingest_count=int(ingest_result.get("extracted", 0)
                                               if isinstance(ingest_result, dict) else 0),
                         user_ingest_ms=int((_ingest_end - _ingest_start) * 1000),
                         user_ingest_started_ts=_ingest_start,
                         user_ingest_ended_ts=_ingest_end)
            # M104 F1: mirror /api/chat feed-write on user extraction so
            # the feed populates for stream-endpoint traffic (UI). The
            # recall-path and midas-ext mirrors live further below.
            try:
                if isinstance(ingest_result, dict) and ingest_result.get("extracted", 0) > 0:
                    _session["facts_extracted"] = _session.get("facts_extracted", 0) + int(ingest_result["extracted"])
                    _add_feed("extraction", f"[user] {message[:80]}")
            except Exception:
                pass

            # Main 37: recency short-circuit (stream variant). Matches the
            # /api/chat behavior — route "what did we talk about last"
            # directly to the prior session summary, not through retrieval.
            # m120_b_phatic_anchor_check: conjunction check — phatic-recency
            # with a trailing topical anchor falls through to L2 instead of
            # firing the L0 short-circuit. Closes M119 T84.
            # M125.2 Stream D: turn-scope short-circuit (stream variant).
            # Independent from phatic session-recency gate — fires on
            # explicit "N turns/messages/exchanges ago" queries and
            # routes to the within-session turn log. See /api/chat
            # handler for the non-stream twin.
            _parsed_scoped = _parse_ordinal_nback(message)
            if (_parsed_scoped and _parsed_scoped[1] == "turn"
                    and _is_turn_ordinal_query(message)):
                _turn_n = _parsed_scoped[0]
                turn_record = load_nth_turn(_turn_n)
                turn_response = _build_turn_recency_response(turn_record)
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant",
                                 "content": turn_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _add_feed("recency",
                          f"turn-ago bridge (stream): {message[:60]}")
                _turn_record("routing",
                             route_layer="L0",
                             l1_match="turn_ordinal_short_circuit",
                             l2_decision=None,
                             tools_called=["turn_log_recall"])
                _turn_record("retrieval",
                             shape_fired="turn_recency_bridge",
                             turn_path=(turn_record or {}).get("_path"),
                             ordinal_nback=_turn_n,
                             ordinal_scope="turn",
                             turn_total_available=(turn_record or {}).get(
                                 "_total_available", 0))
                _turn_record("generation",
                             response_text=turn_response,
                             response_chars=len(turn_response),
                             response_tokens_est=_rough_tokens(turn_response),
                             ttft_ms=0, total_ms=0)
                _turn_write()
                yield f"data: {json.dumps({'type':'token','content':turn_response})}\n\n"
                yield f"data: {json.dumps({'type':'done','stats':{},'memories_recalled':0})}\n\n"
                return

            if _m120_b_should_short_circuit(message):
                # M125.1 Stream B: ordinal N-back routing (stream variant).
                # See /api/chat handler above for the same logic.
                # M125.2 Stream D: classifier returns (n, scope);
                # only session-scope reaches this branch.
                _parsed_sess = _parse_ordinal_nback(message)
                if (_parsed_sess and _parsed_sess[1] == "session"
                        and _parsed_sess[0] > 1):
                    recency_summary = load_nth_session_summary(
                        _parsed_sess[0])
                    if recency_summary is None:
                        recency_summary = load_last_session_summary()
                else:
                    recency_summary = load_last_session_summary()
                recency_response = _build_recency_response(recency_summary)
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant", "content": recency_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _add_feed("recency", f"prior session bridge (stream): {message[:60]}")
                _turn_record("routing",
                             route_layer="L0",
                             l1_match="recency_short_circuit",
                             l2_decision=None,
                             tools_called=["session_summary_recall"])
                _turn_record("retrieval", shape_fired="recency_bridge",
                             summary_path=(recency_summary or {}).get("_path"),
                             ordinal_nback=(recency_summary or {}).get("_nback", 1))
                _turn_record("generation",
                             response_text=recency_response,
                             response_chars=len(recency_response),
                             response_tokens_est=_rough_tokens(recency_response),
                             ttft_ms=0, total_ms=0)
                _turn_write()
                yield f"data: {json.dumps({'type':'token','content':recency_response})}\n\n"
                yield f"data: {json.dumps({'type':'done','stats':{},'memories_recalled':0})}\n\n"
                return

            # Route
            # Main 62 Bug 1: pass prior user turn for anaphoric fold-in.
            # M125 A5.1: pass _last_tool so anaphoric follow-ups re-fire
            # the prior tool with subject composed in.
            global _last_tool
            _prior_user = ""
            for _h in reversed(_history):
                if _h.get("role") == "user":
                    _prior_user = _h.get("content", "")
                    break
            l1_result = layer1_route(message,
                                      prior_user_message=_prior_user,
                                      prior_tool=_last_tool or "")
            tool_name, tool_args = route(
                message, llm_fn=llm_route_fn,
                prior_user_message=_prior_user,
                prior_tool=_last_tool or "")
            _turn_record("routing",
                         l1_match=(l1_result[0] if l1_result else None),
                         l2_decision=tool_name,
                         route_layer=("L1" if l1_result else "L2"))

            # M125 A5.2: meta-instruction short-circuit in stream path.
            # Same acknowledge-only behavior as non-stream path. Emits the
            # prebuilt ACK response and returns immediately — no recall,
            # no generation, no LLM contact.
            if tool_name == "meta_instruction_ack":
                _meta_response = (tool_args or {}).get(
                    "response",
                    "Understood — I've noted that instruction.")
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant",
                                 "content": _meta_response})
                _history = _trim_history(_history, MAX_HISTORY)
                _last_tool = "meta_instruction_ack"
                _turn_record("generation",
                             response_text=_meta_response,
                             response_chars=len(_meta_response),
                             response_tokens_est=_rough_tokens(_meta_response),
                             ttft_ms=0, total_ms=0,
                             skipped=True,
                             skip_reason="meta_instruction_ack")
                _turn_write()
                yield f"data: {json.dumps({'type':'token','content':_meta_response})}\n\n"
                yield f"data: {json.dumps({'type':'done','stats':{},'memories_recalled':0})}\n\n"
                return

            # tools_called list is populated only on actual execute() below,
            # not on routing intent. l2_decision captures the intent.
            # Heuristic: did the user ask for a tool by name or intent
            # that the router didn't route to?
            _missed = _detect_tools_requested_not_called(
                message,
                [tool_name] if tool_name and tool_name != "conversation" else [])
            if _missed:
                _turn_record("routing", tools_requested_not_called=_missed)

            # M73 Track B P1: static query-type dispatch.
            # M74 Agent 2: retrieval_modifiers consumption controlled by
            # env M74_A2_ACTIVE. =0 (default) → threshold=0.5 for all
            # queries (current prod behavior). =1 → per-query-type dispatch.
            _m73_dispatch = _query_type_dispatch(message)
            _m74_retrieval_modifiers = _m73_dispatch["retrieval_modifiers"]
            _m74_a2_active = (os.environ.get("M74_A2_ACTIVE", "0") == "1")
            if _m74_a2_active:
                _absence_sensitivity = _m74_retrieval_modifiers.get(
                    "absence_gate_sensitivity", "normal")
            else:
                _absence_sensitivity = "normal"
            _absence_threshold = {
                "strict": 0.7, "normal": 0.5, "loose": 0.3
            }.get(_absence_sensitivity, 0.5)
            _turn_record(
                "dispatch",
                query_type=_m73_dispatch["query_type"],
                sample_profile=_m73_dispatch["sample_profile"],
                retrieval_modifiers=_m73_dispatch["retrieval_modifiers"],
                retrieval_modifiers_applied={
                    "m74_a2_active": _m74_a2_active,
                    "absence_gate_sensitivity_effective": _absence_sensitivity,
                    "absence_gate_threshold": _absence_threshold,
                    "canonical_boost_active": False,  # M75 stub
                    "scope_rerank_weight_active": False,  # M75 stub
                },
                source=_m73_dispatch["source"])

            # Phase 2: try enumeration first, then narrative, fall back to top-15 recall
            mem_ctx = None
            presentation_briefing = None
            _last_subconscious = []
            _narrative_used = False
            nr = None  # track narrative result for absence guard
            # m125_3_e_enumeration_plumbing: dedicated enumeration payload
            # block that reaches the prompt via the user-message tail
            # (mirrors per_query_block/_reg_block). Prior bug: enumeration
            # records were put in mem_ctx only, which is gated by
            # synthesizer.build_messages' `elif briefing:` branch — briefing
            # is always present, so records never reached the prompt. The
            # absence-guard word-overlap check (Sub-Gate 3) also overwrote
            # mem_ctx to a KNOWLEDGE GAP warning on enumeration turns
            # (Q10 in M125.2 E pilot readiness: 4 published_repo records
            # replaced by CRITICAL GAP warning, then silently dropped at
            # synthesizer.build_messages). Track the enumeration payload
            # as a separate block and flag the source so both drop sites
            # are bypassed. See vault/agent_reports/m125_3_e_enumeration_plumbing.md.
            _enumeration_block = None
            _enumeration_active = False

            # Phase 2a: enumeration check (shape #6) — complete tag-based retrieval
            try:
                if inject_context:
                    from enumeration_retrieval import enumerate_by_tag, shape_dispatch_signal
                    _enum = enumerate_by_tag(message)
                    if _enum:
                        mem_ctx = _enum["records"]
                        _narrative_used = True
                        # m125_3_e_enumeration_plumbing: build the
                        # enumeration block that will ride in the user
                        # message tail so the model sees it. Format
                        # matches the existing per_query_block /
                        # _reg_block conventions (header + bullet list).
                        try:
                            _enum_header = (
                                f"[ENUMERATION: complete set of "
                                f"{_enum.get('tag')} records "
                                f"(count={_enum.get('count')})]"
                            )
                            _enum_lines = list(_enum.get("records") or [])
                            if _enum_lines:
                                _enumeration_block = (
                                    _enum_header + "\n" + "\n".join(_enum_lines)
                                )
                                _enumeration_active = True
                        except Exception:
                            _enumeration_block = None
                            _enumeration_active = False
                        _emit_subconscious_event(
                            "enumeration_fired",
                            query=message[:100],
                            tag=_enum["tag"],
                            count=_enum["count"],
                            topic_filter=_enum.get("topic_filter"),
                            latency_ms=_enum["latency_ms"],
                            path="api_chat_stream")
                        _turn_record("retrieval",
                                     shape_fired="enumeration",
                                     enumeration={
                                         "tag": _enum.get("tag"),
                                         "count": _enum.get("count"),
                                         "topic_filter": _enum.get("topic_filter"),
                                         "latency_ms": _enum.get("latency_ms"),
                                     })
                    else:
                        # m120_e_enumeration_dispatcher: even when tag-query
                        # execution returns None (K9 gap: unbacked tag like
                        # published_repo), classify the shape so the turn
                        # log reflects that Enumeration fired. Records fall
                        # through to default_recall below; the missing
                        # backing index is surfaced via backed=False so
                        # observability catches the architectural gap.
                        _sig = shape_dispatch_signal(message)
                        if _sig:
                            _turn_record("retrieval",
                                         shape_fired="enumeration",
                                         enumeration={
                                             "tag": _sig.get("tag"),
                                             "backed": _sig.get("backed"),
                                             "count": 0,
                                             "unbacked_gap": not _sig.get("backed"),
                                         })
                            _emit_subconscious_event(
                                "enumeration_unbacked",
                                query=message[:100],
                                tag=_sig.get("tag"),
                                backed=_sig.get("backed"),
                                path="api_chat_stream")
            except Exception:
                pass

            # m125_a2_shape_precedence: classify query BEFORE narrative
            # synthesis. If is_canonical_lookup fires, suppress narrative and
            # jump directly to default_recall so canonical pool records load.
            # See shape_precedence.py + M124 Stream A §Cause 4 (K5).
            _m125_a2_is_cl = False
            _m125_a2_diag = {"decision_reason": "not_classified"}
            try:
                _m125_a2_is_cl, _m125_a2_diag = _m125_a2_is_canonical_lookup(
                    message or "")
            except Exception as _e:
                _m125_a2_is_cl = False
                _m125_a2_diag = {"decision_reason": f"classifier_exception:{_e}"}
            _m125_a2_dispatch_decision = _M125_A2_DD_OTHER

            # Phase 2b: narrative synthesis
            try:
                if (inject_context and not _narrative_used
                        and not _m125_a2_is_cl):
                    # Phase 3: check pre-warmed cache first
                    nr = get_cached_narrative(message)
                    if not nr:
                        from narrative_retrieval import try_narrative_context
                        nr = try_narrative_context(message)
                    if nr:
                        mem_ctx = [nr["narrative"]]
                        _narrative_used = True
                        _m125_a2_dispatch_decision = _M125_A2_DD_NARRATIVE
                        _emit_subconscious_event(
                            "narrative_synthesized",
                            query=message[:100],
                            n_records=nr["n_records"],
                            n_sessions=nr["n_sessions"],
                            arc_type=nr["arc_type"],
                            latency_ms=nr["latency_ms"],
                            path="api_chat_stream")
                        _turn_record("retrieval",
                                     shape_fired="narrative",
                                     narrative={
                                         "n_records": nr.get("n_records"),
                                         "n_sessions": nr.get("n_sessions"),
                                         "arc_type": nr.get("arc_type"),
                                         "latency_ms": nr.get("latency_ms"),
                                     })
                elif (inject_context and not _narrative_used
                        and _m125_a2_is_cl):
                    # Classifier suppressed narrative; default_recall runs next.
                    _m125_a2_dispatch_decision = _M125_A2_DD_NARRATIVE_SUPPRESSED
                    _emit_subconscious_event(
                        "m125_a2_narrative_suppressed",
                        query=message[:100],
                        positive_match=_m125_a2_diag.get("positive_match"),
                        decision_reason=_m125_a2_diag.get("decision_reason"),
                        path="api_chat_stream")
            except Exception:
                pass
            try:
                if not _narrative_used:
                    # Main 43 Phase 3: pass topic boost from context tracker
                    _ct_s = _get_context_tracker()
                    _cb_s = _ct_s.get_retrieval_boost() if _ct_s else None
                    # m118_e_sub_shape_2: fold antecedent when demonstrative
                    _recall_q_s = _m118_e_expand_query_with_antecedent(
                        message, _history)
                    recall_result = memory.recall(_recall_q_s, n_results=15,
                                                  context_boost=_cb_s) if inject_context else None
                    if recall_result and recall_result.get("results"):
                        filtered = []
                        for r in recall_result["results"]:
                            if r.get("score", 0) < 0.40:
                                continue
                            text = r.get("text", "")
                            if text.startswith("[") and ".md]" in text[:50]:
                                continue
                            if text.strip().endswith("?"):
                                continue
                            filtered.append(r)
                            if len(filtered) >= 8:
                                break
                        # Main 46: prepend source labels to prevent
                        # measurement conflation. The model needs to know
                        # which memory is canonical vs assistant-sourced.
                        def _label_memory(r):
                            src = r.get("source_role", "")
                            tag = f"[{src}] " if src else ""
                            return tag + r["text"]
                        mem_ctx = [_label_memory(r) for r in filtered]
                        _last_subconscious = [{"text": r["text"][:200], "score": r["score"]} for r in filtered]
                        # Turn log: capture the filtered recall with scores,
                        # source_role, and type so per-turn analysis can
                        # distinguish canonical from derived memories.
                        # m109_zeta_field_1+7+8: per-record role_weight,
                        # provenance, canonical_boost counterfactual.
                        _m109_annot = _m109_annotate_recall(filtered)
                        _recall_filtered_log = []
                        for _idx, _r in enumerate(filtered):
                            _rec = {
                                "score": round(_r.get("score", 0), 3),
                                "source_role": (_r.get("source_role")
                                    or (_r.get("metadata", {}) or {}).get("source_role", "")),
                                "type": (_r.get("type")
                                    or (_r.get("metadata", {}) or {}).get("type", "")),
                                "text": (_r.get("text") or "")[:400],
                            }
                            if _idx < len(_m109_annot):
                                _rec.update(_m109_annot[_idx])
                            _recall_filtered_log.append(_rec)
                        # m125_a2: on successful default_recall when
                        # classifier had suppressed narrative, record
                        # recall_primary_canonical_lookup. Otherwise leave
                        # OTHER (e.g. narrative was skipped for a non-A2
                        # reason, or classifier didn't fire but narrative
                        # also produced no result).
                        if _m125_a2_is_cl:
                            _m125_a2_dispatch_decision = (
                                _M125_A2_DD_RECALL_PRIMARY)
                        _turn_record("retrieval",
                                     shape_fired="default_recall",
                                     recall_filtered=_recall_filtered_log,
                                     score_stats=_retrieval_score_stats(filtered))
                        _emit_subconscious_event(
                            "memory_recalled",
                            query=message[:100],
                            top_k=len(filtered),
                            top_score=round(filtered[0].get("score", 0), 3) if filtered else 0,
                            path="api_chat_stream")
                        for _i, _r in enumerate(filtered):
                            _meta = _r.get("metadata", {}) or {}
                            _src = (_r.get("source") or _r.get("file")
                                    or _meta.get("source") or _meta.get("file") or "")
                            _basename = _src.rsplit("/", 1)[-1] if _src else ""
                            _entities = _r.get("entities") or []
                            _topic_hint = _r.get("query_category", "") or ""
                            _emit_subconscious_event(
                                "memory_recalled_item",
                                index=_i,
                                of=len(filtered),
                                score=round(_r.get("score", 0), 3),
                                file=_basename,
                                text=_r.get("text", "")[:200],
                                topic=_topic_hint,
                                source_role=(_r.get("source_role")
                                              or _meta.get("source_role", "")) or "",
                                mem_type=(_r.get("type")
                                           or _meta.get("type", "")) or "",
                                entities=_entities[:5] if isinstance(_entities, list) else [],
                                query=message[:80],
                            )
                        # M104 F1: mirror /api/chat recall feed-writes so
                        # the UI feed shows recall events on stream traffic.
                        if mem_ctx:
                            _session["memories_recalled"] = (
                                _session.get("memories_recalled", 0) + len(mem_ctx))
                            for _m in mem_ctx[:3]:
                                _add_feed("recall", str(_m)[:80])
            except Exception:
                pass

            # m125_a2: emit dispatch_decision to ζ v2.3 turn log.
            # Enum: narrative_primary | recall_primary_canonical_lookup |
            #       narrative_suppressed_classifier | other.
            # Written once per turn, after dispatch branches settle.
            try:
                _turn_record(
                    "retrieval",
                    dispatch_decision=_m125_a2_dispatch_decision,
                    dispatch_decision_diagnostic={
                        "is_canonical_lookup": bool(_m125_a2_is_cl),
                        "positive_match": _m125_a2_diag.get("positive_match"),
                        "negative_match": _m125_a2_diag.get("negative_match"),
                        "a4_specific_matched": _m125_a2_diag.get(
                            "a4_specific_matched", False),
                        "decision_reason": _m125_a2_diag.get("decision_reason"),
                    })
            except Exception:
                pass

            # Phase 2c: measurement registry injection (Main 41)
            # Runs on every recall path, not just narrative. Prepends
            # canonical measurements to mem_ctx so point-value queries
            # always surface the registry entry.
            try:
                _reg_matches = _measurement_registry_lookup(message)
                if _reg_matches:
                    if mem_ctx:
                        mem_ctx = _reg_matches + mem_ctx
                    else:
                        mem_ctx = _reg_matches
            except Exception:
                pass

            # Phase 3: proactive contradiction surfacing
            try:
                from contradiction_check import check_contradictions
                _contradictions = check_contradictions(message)
                if _contradictions and mem_ctx:
                    mem_ctx.extend(_contradictions)
                elif _contradictions:
                    mem_ctx = _contradictions
            except Exception:
                pass

            # Phase 4: absence guard (after all retrieval)
            # Fires when: no narrative/registry/enumeration matched AND
            # recall only returned low-confidence tangential matches.
            # The guard checks scores, not just emptiness.
            # Phase 4: absence guard with topical relevance check
            # M115 β three-sub-gate split — see
            # vault/agent_reports/m115_a1_beta_split_gate.md
            #   Sub-Gate 1: pool==0 + narr=False → fire unconditionally
            #               (drop domain-relevance silencing; A3 §9).
            #   Sub-Gate 2: pool>0, max<0.5, word-overlap mismatch → fire.
            #   Sub-Gate 3: pool>0, max>=0.5, word-overlap mismatch → DO
            #               NOT fire; pass filtered (not []) so check_absence
            #               has score context.
            # m116_d_drop_narrative_used: eligibility predicate no longer
            # consults _narrative_used. Silent hole at T1/T7/T8/T10 in M114
            # pilot (pool==0 AND narrative_used==True → guard silent → model
            # confabulates without safety net). β sub-gates (M115) evaluate
            # narrative adequacy naturally via pool_size + max_score — when
            # narrative correctly answers, pool is typically non-empty and
            # max_score correlates, so Sub-Gate 3 suppresses. narrative_used
            # retained in feature_vector below for M109 ζ F5 observability.
            _guard_eligible = True  # m116_d_drop_narrative_used
            _guard_fired = False
            try:
                if _guard_eligible and not mem_ctx:
                    # m115_beta_sub_gate_1: empty pool + narrative unused
                    # → fire unconditionally. absence_guard.check_absence
                    # now drops the _is_domain_relevant check on this
                    # branch (A3 §9). Passing [] is correct here because
                    # there truly is no recall pool.
                    # m116_d_drop_narrative_used: pass None as narrative_result
                    # so β sub-gates can run regardless of narrative state
                    # (callee's narrative short-circuit would otherwise
                    # re-introduce the silent hole downstream).
                    from absence_guard import check_absence
                    guard = check_absence(message, [], None)
                    if guard:
                        mem_ctx = [guard]
                        _guard_fired = True
                        _emit_subconscious_event("absence_guard_fired",
                            query=message[:100])
                elif _guard_eligible and mem_ctx and not _enumeration_active:
                    # m125_3_e_enumeration_plumbing: skip Sub-Gate 3 on
                    # enumeration turns. Enumeration records are an
                    # exhaustive deterministic payload from the tag
                    # index — the query shape ("list all X") will
                    # rarely share content-words with a record list
                    # like "orion-ane: public GitHub repo", so the
                    # word-overlap heuristic false-fires and overwrites
                    # mem_ctx with a KNOWLEDGE GAP warning (Q10 failure
                    # mode in M125.2 E pilot readiness). The enumeration
                    # block itself still rides the user-message tail via
                    # _enumeration_block regardless of this branch.
                    # Topical relevance: do the recalled memories actually
                    # address the specific question, or just match the entity?
                    _q_words = set(w.lower().strip("?.,!\"'") for w in message.split() if len(w) >= 4)
                    _stop = {"what", "does", "that", "this", "with", "from", "have",
                             "been", "about", "which", "where", "when", "many", "much",
                             "strategy", "should", "would", "could", "explain", "tell"}
                    _q_specific = _q_words - _stop
                    if _q_specific and len(_q_specific) >= 2:
                        _mem_text = " ".join(str(m) for m in mem_ctx).lower()
                        _unmatched = [w for w in _q_specific if w not in _mem_text]
                        if len(_unmatched) > len(_q_specific) * 0.5:
                            # m115_beta_sub_gate_3: word-overlap mismatch.
                            # Pass the actual `filtered` recall pool (not
                            # []) so check_absence can see the score
                            # context. High-score rows (>=0.5) suppress
                            # the guard (Sub-Gate 3 FP fix). Low-score or
                            # empty rows fall through to Sub-Gate 2
                            # low-score firing path.
                            # m116_d_drop_narrative_used: pass None so β
                            # sub-gates evaluate independently of narrative.
                            from absence_guard import check_absence
                            _sg3_pool = (filtered
                                         if 'filtered' in locals()
                                         and filtered else [])
                            guard = check_absence(message, _sg3_pool, None)
                            if guard:
                                mem_ctx = [guard]
                                _guard_fired = True
                                _emit_subconscious_event("absence_guard_fired",
                                    query=message[:100])
            except Exception:
                pass
            _recall_score_max_for_log = 0.0
            if 'filtered' in locals() and filtered:
                _recall_score_max_for_log = max(
                    float(r.get("score", 0)) for r in filtered)
            # m109_zeta_field_5: full absence-guard feature vector.
            # Per M108 Finding 3, the operative predicate is the conjunction
            # at line ~3378: (_guard_fired AND recall_score_max < threshold
            # AND tool_name == 'conversation'|None). Log everything that
            # conjunction consumes so M108 validation can reconstruct the
            # decision offline. word-overlap path consumes _q_specific and
            # _unmatched — log those too when that branch fired.
            _abs_pool_size = (len(filtered) if 'filtered' in locals()
                              and filtered else 0)
            _abs_features = {
                "max_score": round(_recall_score_max_for_log, 3),
                "pool_size": _abs_pool_size,
                "threshold_applied": _absence_threshold,
                "threshold_sensitivity": _absence_sensitivity,
                "query_type_dispatch": _m73_dispatch.get("query_type"),
                "narrative_used": _narrative_used,
                "mem_ctx_empty_branch": (_guard_eligible and
                                          not ('filtered' in locals()
                                               and filtered)),
                "word_overlap_branch_eligible":
                    _guard_eligible and bool(mem_ctx)
                    and ('filtered' in locals() and bool(filtered)),
                "m74_a2_active": _m74_a2_active,
            }
            # word-overlap branch features (only populated if that branch ran)
            try:
                if '_q_specific' in locals() and _q_specific:
                    _abs_features["q_specific_count"] = len(_q_specific)
                    _abs_features["q_unmatched_count"] = (
                        len(_unmatched) if '_unmatched' in locals() else 0)
                    _abs_features["unmatched_ratio"] = round(
                        (len(_unmatched) / max(1, len(_q_specific)))
                        if '_unmatched' in locals() else 0.0, 3)
                else:
                    _abs_features["q_specific_count"] = "not_consumed_this_turn"
                    _abs_features["q_unmatched_count"] = "not_consumed_this_turn"
                    _abs_features["unmatched_ratio"] = "not_consumed_this_turn"
            except Exception:
                pass
            # m115_beta_zeta_v2_1: expose _is_domain_relevant() result +
            # matched terms so post-M115 analyses can reconstruct the
            # Gate-B decision without the local-keyword approximation
            # used in M114. Per A3 §9 ζ v2.1 spec.
            try:
                from discovery_retrieval import _is_domain_relevant as _dr
                _dr_relevant, _dr_terms = _dr(message)
                _abs_features["is_domain_relevant_result"] = bool(_dr_relevant)
                _abs_features["is_domain_relevant_matched_terms"] = (
                    list(_dr_terms) if _dr_terms else [])
            except Exception:
                _abs_features["is_domain_relevant_result"] = "not_instrumented_v2_1"
                _abs_features["is_domain_relevant_matched_terms"] = "not_instrumented_v2_1"
            _turn_record("retrieval", absence_guard={
                "eligible": _guard_eligible,
                "fired": _guard_fired,
                "recall_score_max": round(_recall_score_max_for_log, 3),
                "feature_vector": _abs_features,  # m109_zeta_field_5
            })

            # Main 46: Hard absence gate. If the guard fired on a
            # conversation-route turn, skip 27B generation entirely.
            # The 27B ignores abstention directives 3/3 times (Main 46
            # audit). Don't ask a model to abstain — don't call it.
            # Main 47 P1: Don't hard-gate when recall has high-scoring
            # memories. The word-overlap heuristic over-fires when
            # recalled content uses different vocabulary than the query
            # (T03, T13 regression: 8 memories, scores up to 2.25,
            # but gate blocked because terms didn't word-match).
            # M74 Agent 2: threshold now content-type-aware via dispatch
            # retrieval_modifiers.absence_gate_sensitivity.
            #   strict  → 0.7 (canonical_lookup, historical, decision)
            #   normal  → 0.5 (current production default; code_gen, debugging, chit_chat)
            #   loose   → 0.3 (synthesis, conceptual, brainstorm — tolerate weak recall)
            # M123 A3: Definitional-query suppression. Before the hard
            # gate evaluates, classify the query. On a True fire, let
            # the model generate from parametric knowledge (Tier 1 FAB
            # scrub + Tier 2 binding scrub still run downstream — only
            # the hard absence gate is suppressed). Under-fit; see
            # absence_guard.is_definitional_query for pattern list.
            _m123_a3_definitional_suppression = False
            _m123_a3_definitional_diag = None
            try:
                from absence_guard import is_definitional_query
                _def_fired, _def_diag = is_definitional_query(message)
                _m123_a3_definitional_diag = _def_diag
                if (_def_fired
                        and _guard_fired
                        and _recall_score_max_for_log < _absence_threshold
                        and (tool_name == "conversation" or not tool_name)):
                    _m123_a3_definitional_suppression = True
            except Exception:
                _m123_a3_definitional_suppression = False
            # ζ v2.2 emission — conditional field per M123 directive §3.3
            try:
                if _m123_a3_definitional_suppression:
                    _turn_record(
                        "guard",
                        absence_gate={
                            "definitional_suppression": True,
                            "definitional_diagnostic":
                                _m123_a3_definitional_diag,
                        },
                    )
            except Exception:
                pass
            if (_guard_fired
                    and _recall_score_max_for_log < _absence_threshold
                    and (tool_name == "conversation" or not tool_name)
                    and not _m123_a3_definitional_suppression):
                _absence_response = (
                    "I don't have information about that in our research. "
                    "This hasn't come up in any session I can find."
                )
                # If there's any recalled content, mention the closest topic
                if mem_ctx and mem_ctx[0] and not mem_ctx[0].startswith("=== CRITICAL"):
                    _top_mem = str(mem_ctx[0])[:150].strip()
                    _absence_response = (
                        "I don't have specific information about that. "
                        f"The closest related topic in memory is: {_top_mem}... "
                        "Would you like to explore that instead?"
                    )
                _turn_record("generation",
                             skipped=True,
                             skip_reason="absence_gate",
                             response_text=_absence_response,
                             response_chars=len(_absence_response))
                _all_user_queries.append(message)
                _history.append({"role": "user", "content": message})
                _history.append({"role": "assistant", "content": _absence_response})
                _history = _trim_history(_history, MAX_HISTORY)
                # M123 A5 fix: removed double-increment of messages_sent.
                # Line 3252 (stream handler entry) already incremented
                # messages_sent for this turn; re-incrementing here skipped
                # a turn-number in the sequence and produced the "5 capture
                # gaps" finding in M122 C. Pilot had 6 absence-gate fires
                # (turns 11, 28, 30, 37, 47, 50) each followed by a
                # phantom missing turn-number (12, 29, 31, 38, 48, 51).
                # Absence-gate turns ARE captured correctly — only the
                # counter was wrong. See vault/agent_reports/m123_a5_turn_write_reliability.md.
                _turn_write()
                yield f"data: {json.dumps({'type':'token','content':_absence_response})}\n\n"
                yield f"data: {json.dumps({'type':'done','stats':{},'memories_recalled':len(_last_subconscious)})}\n\n"
                return

            # Main 37 refactor: full briefing pipeline on the stream path.
            # Before this fix the stream endpoint built no briefing at all
            # and fell through to the synthesizer's raw mem_ctx bullet
            # dump. Now it runs the same stable seed-query briefing +
            # warm context + message-#1 session-open + PRIOR SESSION
            # bridge that /api/chat uses, which also re-enables the 72B
            # verifier's prefix KV cache (byte-stable system slot) on
            # terminal traffic.
            presentation_briefing = _build_presentation_briefing(
                is_first_message=(_session.get("messages_sent") == 1)
            )

            # Per-query memories ride in the user message tail so the
            # system slot stays byte-stable for prefix-cache hits.
            per_query_block = None
            if inject_context and 'filtered' in locals() and filtered:
                per_query_block = _build_per_query_block(filtered, message)

            # Main 41: measurement registry block prepended to per-query
            _reg_block = None
            try:
                _reg_matches = _measurement_registry_lookup(message)
                if _reg_matches:
                    _reg_block = "\n".join(_reg_matches)
            except Exception:
                pass

            # M53 P1: cross-turn context for referential queries.
            # When the user references the model's prior response
            # ("you mentioned X", "earlier you said"), inject the previous
            # turn's response + tool result to prevent false self-
            # accusation when recall shifts between turns.
            _prior_turn_ctx = None
            try:
                if _SESSION_LOG_DIR:
                    _current_turn = _TURN_LOG.get("turn_number", 0) if _TURN_LOG else 0
                    _prior_turn_ctx = _build_prior_turn_context(
                        message, _SESSION_LOG_DIR, _current_turn)
                    if not _prior_turn_ctx:
                        _prior_turn_ctx = None
            except Exception:
                _prior_turn_ctx = None

            # M54 Phase 2.3: possessive-intent directive.
            # When the user asks about "our X" and a vault tool was/will
            # be dispatched, inject a directive telling the model how to
            # interpret recalled content. Vault contains both internal
            # capability docs AND external research notes; the model
            # otherwise attributes external research as ours (Orion LoRA
            # bug). This is a model-level guardrail since recall filtering
            # alone wasn't sufficient.
            _possessive_directive = None
            _q_low = message.lower()
            # M54 Phase 2.4: distinguish capability vs knowledge intent.
            # "Do we KNOW about X" is asking what info we have, not what
            # we built — should surface research. "Do we HAVE/USE X"
            # is asking about capability — should warn about external
            # vs internal distinction.
            _cap_markers = [
                "our ", "we have", "we've", "do we have", "do we use",
                "did we build", "did we ship", "did we deploy",
                "are we using", "are we running", "have we built",
                "have we deployed",
            ]
            _know_markers = [
                "do we know", "what do we know", "have we researched",
                "have we explored", "have we investigated", "have we read",
                "what have we found", "have we documented",
                "have we studied",
            ]
            _has_cap = any(m in _q_low for m in _cap_markers)
            _has_know = any(m in _q_low for m in _know_markers)
            _is_possessive = _has_cap and not _has_know
            if _is_possessive:
                _possessive_directive = (
                    "[POSSESSIVE-INTENT NOTE: User asked about OUR system. "
                    "Recalled vault content and tool results may include "
                    "external research notes (Orion, third-party papers, "
                    "prior art). Do NOT attribute external research as our "
                    "internal capability. If the recalled content describes "
                    "an external project rather than our system's actual "
                    "capability, ABSTAIN: 'We do not have X. The vault has "
                    "research notes on the external Y project but we have "
                    "not built/deployed this internally.']"
                )

            augmented = message
            blocks = []
            if _reg_block:
                blocks.append(_reg_block)
            if _prior_turn_ctx:
                blocks.append(_prior_turn_ctx)
            if _possessive_directive:
                blocks.append(_possessive_directive)
            # m125_3_e_enumeration_plumbing: enumeration payload reaches the
            # prompt via the user-message tail (mirrors per_query_block).
            # Fires only when the enumeration classifier dispatched and
            # records were available — non-enumeration turns leave this
            # block None and the user message is unchanged.
            if _enumeration_block:
                blocks.append(_enumeration_block)
            if per_query_block:
                blocks.append(per_query_block)
            if blocks:
                augmented = "\n\n".join(blocks) + f"\n\n---\n\n{message}"

            # Main 37 turn log: record the full context that will be
            # passed to the 72B. Briefing preview + per_query preview so
            # analysis tools can inspect what the model actually saw.
            _turn_record("context",
                         briefing_chars=len(presentation_briefing or ""),
                         briefing_has_prior_session=bool(
                             presentation_briefing and
                             "PRIOR SESSION" in presentation_briefing),
                         briefing_preview=(presentation_briefing or "")[:600],
                         briefing_text=(presentation_briefing or ""),
                         mem_ctx_chars=sum(len(str(m)) for m in (mem_ctx or [])),
                         mem_ctx_text=(mem_ctx or []),
                         per_query_chars=len(per_query_block or ""),
                         per_query_preview=(per_query_block or "")[:600],
                         enumeration_chars=len(_enumeration_block or ""),
                         enumeration_active=bool(_enumeration_active),
                         system_tokens_est=_rough_tokens(presentation_briefing or ""),
                         user_tokens_est=_rough_tokens(augmented))

            # Build messages for LLM
            from synthesizer import build_messages
            # Main 46: when absence guard fired, inject the knowledge
            # gap warning as a standing rule so it reaches the system
            # prompt directive layer (highest authority), not just the
            # per-turn memory context where the 27B ignores it.
            _effective_rules = list(_ACTIVE_STANDING_RULES or [])
            if _guard_fired:
                _effective_rules.append(
                    "KNOWLEDGE GAP ACTIVE: You have NO recalled memories "
                    "for this query. Do NOT fabricate. Say you don't have "
                    "information about this topic.")

            _tool_result_str = None  # M49 P2: available for grounding corpus
            # m113_alpha_history_filter: mark prior-assistant abstain+project-
            # state-claim turns low-confidence before history joins the prompt.
            # Prevents T11→T20 cascade where "Milestone 1C pending" leaked
            # across 5 turns via conversation history with zero supporting
            # recall entries. Non-destructive: _history global is unchanged.
            try:
                from history_reinforcement_filter import (
                    filter_history_for_reinforcement as _m113_filter,
                    count_marked as _m113_count_marked,
                )
                _history_for_prompt = _m113_filter(_history, current_query=message)
                _turn_record(
                    "history_filter",
                    marked=_m113_count_marked(_history_for_prompt),
                    total_history_turns=len(_history),
                    filter_shape="mark_low_confidence",
                    variant="m113_alpha",
                )
            except Exception as _m113_err:
                print(f"[m113_alpha_history_filter] bypass: {_m113_err}",
                      flush=True)
                _history_for_prompt = _history
            if tool_name == "conversation":
                msgs = build_messages(
                    _history_for_prompt, augmented,
                    memory_context=mem_ctx,
                    briefing=presentation_briefing,
                    standing_rules=_effective_rules,
                )
            else:
                _tool_start = time.time()
                tool_result = execute(tool_name, tool_args)
                _turn_append("routing", "tools_called", tool_name)
                _tool_result_str = str(tool_result)
                try:
                    _tool_args_log = dict(tool_args) if isinstance(tool_args, dict) else tool_args
                except Exception:
                    _tool_args_log = str(tool_args)
                _existing_args = _TURN_LOG.get("routing", {}).get("tool_args") or {}
                if not isinstance(_existing_args, dict):
                    _existing_args = {}
                _existing_args[tool_name] = _tool_args_log
                _turn_record("routing",
                             tool_args=_existing_args,
                             tool_result_preview=_tool_result_str[:4000],
                             tool_result_chars_total=len(_tool_result_str),
                             tool_latency_ms=int((time.time() - _tool_start) * 1000))

                # m116_c_instruction_misclass: memory_ingest LLM bypass on
                # stream path. /api/chat has this bypass at line ~2323; the
                # stream path was missing it, which caused T37
                # ("commit this to memory - ...") to produce the
                # "You have not asked a question" refusal even though the
                # tool had already stored 38 facts. The LLM has nothing to
                # add to a storage ack and its generation conflicts with
                # the briefing/history. Hardcode the confirmation using
                # the tool's reported count — evidence-based, not
                # fabricated (satisfies Stream A confabulation guard).
                if tool_name == "memory_ingest":
                    _extracted = 0
                    if isinstance(tool_result, dict):
                        _extracted = tool_result.get("extracted", 0)
                    if _extracted > 0:
                        _ack = (f"Got it. Stored {_extracted} fact"
                                f"{'s' if _extracted != 1 else ''}.")
                    else:
                        _ack = "Got it. Noted."
                    # Yield ack as a single SSE token then finish.
                    yield f"data: {json.dumps({'type':'token','content':_ack})}\n\n"
                    # Update history, record turn, emit done event.
                    _all_user_queries.append(message)
                    _history.append({"role": "user", "content": message})
                    _history.append({"role": "assistant", "content": _ack})
                    _history = _trim_history(_history, MAX_HISTORY)
                    _turn_record("generation",
                                 response_text=_ack,
                                 response_chars=len(_ack),
                                 response_tokens_est=_rough_tokens(_ack),
                                 ttft_ms=0, total_ms=0,
                                 bypass="memory_ingest_m116_c")
                    _turn_write()
                    yield f"data: {json.dumps({'type':'done','stats':{},'memories_recalled':0})}\n\n"
                    return
                # M113 gamma: two-regime recall-quality gate.
                # M108 Finding 2: when a tool_result is present the
                # assembler defaults to the contracted regime
                # (history[-4:] = last 2 pairs). T30/T31 failed because
                # the follow-up query needed T25 context but T25 was
                # structurally evicted AND memory_recall returned only
                # low-score off-topic hits (0.268/0.152, both <0.4).
                # Threshold 0.4 sits below absence-gate strict (0.7)
                # and normal (0.5), above loose (0.3), and cleanly
                # separates the T30/T31 miss scores (0.268/0.152) from
                # the borderline 0.462 data point. When triggered, flip
                # to expanded-regime history walk so prior turns are
                # restored; the tool_result tail is still appended, so
                # tool grounding is preserved — this only widens
                # conversation history.
                _m113_gamma_low_recall = bool(
                    _recall_score_max_for_log < 0.4)
                _turn_record("assembly",
                             regime_gate_m113_gamma={
                                 "recall_max_score": round(
                                     _recall_score_max_for_log, 3),
                                 "threshold": 0.4,
                                 "flipped_to_expanded": _m113_gamma_low_recall,
                             })
                msgs = build_messages(
                    _history_for_prompt, augmented,
                    tool_name=tool_name,
                    tool_args=tool_args, tool_result=tool_result,
                    memory_context=mem_ctx,
                    briefing=presentation_briefing,
                    standing_rules=_effective_rules,
                    expand_history_on_low_recall=_m113_gamma_low_recall,
                )

            # Prefill token estimate from the actual msgs list that will be
            # sent to the 72B. Char-based rough estimate (chars / 4).
            _prefill_tokens_est = 0
            try:
                for _m in msgs:
                    _prefill_tokens_est += _rough_tokens(_m.get("content", "") or "")
            except Exception:
                pass
            # m109_zeta_field_6: per-message source_turn provenance tags.
            # System slot → 'system' or 'briefing', current user → 'T<N>',
            # prior history → 'T<N-k>' paired by (user, assistant) pairs.
            _m109_tagged_msgs = _m109_tag_history_messages(
                msgs, briefing_present=bool(presentation_briefing))
            _turn_record("assembled_prompt",
                         messages=_m109_tagged_msgs,
                         total_chars=sum(len(_m.get("content", "") or "") for _m in msgs),
                         total_tokens_est=_prefill_tokens_est,
                         role_structure=[_m.get("role", "") for _m in msgs],
                         n_messages=len(msgs),
                         mem_ctx_count=len(mem_ctx) if mem_ctx else 0,
                         tool_result_chars=len(_tool_result_str) if _tool_result_str else 0)

            # Stream response — Main 40 P1: apply standing-rule
            # token cap if any active rule sets a quantitative
            # constraint. Tightens the default 600-token budget
            # without ever loosening it.
            from synthesizer import parse_max_tokens_from_rules
            _stream_max_tokens = parse_max_tokens_from_rules(
                _ACTIVE_STANDING_RULES, default_max=600)
            # M50 P1: CPU maintenance during GPU decode window.
            # Fire on a background thread so it runs while the GPU
            # streams tokens. CPU is idle during decode otherwise.
            # M50 P1: CPU maintenance during GPU decode.
            # Fire on background thread, runs while GPU streams tokens.
            _cpu_maint_thread = None
            _cpu_maint_result = [False]
            import threading as _thr
            def _cpu_maint():
                # M56 P6: sleep 100ms so maintenance runs during GPU
                # decode, not during prefill/tokenization. Daemon-side
                # idle gate (M55) handles reentrancy; this approximates
                # "is GPU decoding?" by deferring past prefill start.
                time.sleep(0.1)
                _cpu_maint_result[0] = memory.run_maintenance_if_idle()
            _cpu_maint_thread = _thr.Thread(target=_cpu_maint, daemon=True)
            _cpu_maint_thread.start()

            _gen_start = time.time()
            _first_token_ts = None
            full_response = []
            _loop_window = ""  # Main 42: streaming loop detector
            # m109_zeta_field_4: track stop cause for midas-side inference.
            # 72B server always emits finish_reason='stop' so we must
            # reconstruct cause from tokens_decoded + max_tokens + flags.
            _m109_loop_broke = False
            _m109_gen_error = None
            # M73 P1.5 Agent 3: plumb full sample_profile to llm_stream.
            # Server now honors temperature + repetition_penalty + min_p + top_p.
            _m73_sp = _m73_dispatch["sample_profile"]
            # m113_delta_denial_streak: when two or more prior-assistant turns
            # in the recent window match α's abstain-reinforcement shape,
            # reduce temperature and repetition_penalty for THIS turn so the
            # sampler stops amplifying the denial streak. Import-guarded; any
            # failure falls open to the base profile.
            _m113_delta_telemetry = None
            try:
                from denial_streak_sampler import maybe_override_sample_profile
                _m73_sp, _m113_delta_telemetry = maybe_override_sample_profile(
                    _m73_sp, _history)
            except Exception as _dserr:
                print(f"[m113_delta] override import/run failed, using base profile: {_dserr}")
            if _m113_delta_telemetry is not None:
                _turn_record("sampler_streak_override", **_m113_delta_telemetry)
            _m73_temperature = float(_m73_sp["temperature"])
            if mem_ctx and _m73_temperature > 0.3:
                _m73_temperature = 0.3
            _m73_rep_penalty = float(_m73_sp.get("repetition_penalty", 1.0))
            _m73_min_p = float(_m73_sp.get("min_p", 0.0))
            try:
                for chunk in llm_stream(msgs, max_tokens=_stream_max_tokens,
                                        temperature=_m73_temperature,
                                        repetition_penalty=_m73_rep_penalty,
                                        min_p=_m73_min_p):
                    if _first_token_ts is None:
                        _first_token_ts = time.time()
                    full_response.append(chunk)
                    sse_data = json.dumps({"type": "token", "content": chunk})
                    yield f"data: {sse_data}\n\n"
                    # Main 42: detect n-gram drafter loops in-flight.
                    # Check last 200 chars for a repeating pattern.
                    _loop_window += chunk
                    if len(_loop_window) > 200:
                        _tail = _loop_window[-200:]
                        # Find any 6-30 char substring that repeats 4+ times
                        _looping = False
                        for _plen in range(6, 31):
                            _pat = _tail[-_plen:]
                            if _tail.count(_pat) >= 4:
                                _looping = True
                                break
                        if _looping:
                            _m109_loop_broke = True  # m109_zeta_field_4
                            break  # stop generation, let _clean_response trim
            except Exception as _gerr:
                _m109_gen_error = str(_gerr)[:200]  # m109_zeta_field_4
                # Do not re-raise — preserve existing behavior where the
                # request still completes with whatever tokens we captured.
            _gen_end = time.time()

            # M50 P1: check if CPU maintenance completed during decode.
            # The thread had 8-17s (full decode window) to run.
            # Give it 2s grace to finish if decode was very fast.
            _cpu_maint_ran = False
            if _cpu_maint_thread:
                _cpu_maint_thread.join(timeout=2.0)
                _cpu_maint_ran = _cpu_maint_result[0]
            _turn_record("extraction",
                         cpu_maintenance_during_decode=_cpu_maint_ran)

            response = _clean_response("".join(full_response))

            # Main 46: Answer scrub (Phase 4 — Reflection).
            # M49 P2: scrub ALL routes, include tool results in grounding.
            _scrub_result = None
            if response:
                try:
                    from answer_scrub import scrub_response as _scrub_fn
                    from grounding_corpus import build_grounding_corpus_from_parts
                    _gc = build_grounding_corpus_from_parts(
                        mem_ctx_text=mem_ctx,
                        briefing_text=presentation_briefing,
                        tool_result_preview=_tool_result_str[:4000] if _tool_result_str else None)
                    _tools_called = _TURN_LOG.get("routing", {}).get(
                        "tools_called", [])
                    if isinstance(_tools_called, str):
                        _tools_called = [_tools_called]
                    # M125.3 Stream B P2: pass canonical_atom rows from
                    # retrieved context to scrub so the relevance gate
                    # can protect grounded content. Pull from the
                    # recall_filtered snapshot already emitted into
                    # _TURN_LOG.retrieval (persisted pre-scrub).
                    _m125_3_b_canonical_atoms = []
                    try:
                        _rf = (_TURN_LOG.get("retrieval", {}) or {}).get(
                            "recall_filtered", []) or []
                        for _r in _rf:
                            _rtype = (_r.get("type") or "").lower()
                            if _rtype == "canonical_atom":
                                _m125_3_b_canonical_atoms.append({
                                    "id": _r.get("id", ""),
                                    "text": _r.get("text", ""),
                                    "type": _rtype,
                                    "source_role": _r.get(
                                        "source_role", ""),
                                })
                    except Exception:
                        _m125_3_b_canonical_atoms = []
                    _scrub_result = _scrub_fn(
                        response, _gc,
                        user_query=message,
                        tools_called=_tools_called,
                        canonical_atoms=_m125_3_b_canonical_atoms)
                    if _scrub_result and _scrub_result["total_flags"] > 0:
                        cleaned = _scrub_result["cleaned_response"]
                        if cleaned is None:
                            # Everything stripped — fall back to honest response
                            cleaned = (
                                "I found related information but couldn't "
                                "verify the specific details. Could you "
                                "rephrase or ask about a more specific topic?")
                        if cleaned != response:
                            response = cleaned
                            # Send a replacement event so the UI can update
                            yield f"data: {json.dumps({'type':'scrub','content':cleaned})}\n\n"
                except Exception as _scrub_err:
                    # Scrub must never break the response path
                    print(f"[scrub] error: {_scrub_err}", flush=True)

            # m116_a_confab_guard: post-scrub grounding-signal join.
            # Consumes (content, retrieval_hits, tool_calls_made,
            # grounded_memory). Shape (a) REPLACE per directive §3.1 —
            # chosen for M117 pilot attribution clarity (shape (c)
            # telemetry-only would muddy signal). See
            # confabulation_shape_detector.py for the three signals.
            _confab_diag = None
            if response:  # m116_a_confab_guard
                try:
                    from confabulation_shape_detector import (  # m116_a_confab_guard
                        apply_confabulation_guard as _confab_guard_fn,
                        ABSTAIN_MESSAGE as _CONFAB_ABSTAIN,
                    )
                    if response == _CONFAB_ABSTAIN:
                        # Already-abstained upstream; skip.
                        raise RuntimeError("already abstained")
                    _confab_tools = _TURN_LOG.get("routing", {}).get(
                        "tools_called", [])  # m116_a_confab_guard
                    if isinstance(_confab_tools, str):
                        _confab_tools = [_confab_tools]
                    _confab_retrieval = _TURN_LOG.get("retrieval", {}).get(
                        "recall_filtered", []) or []  # m116_a_confab_guard
                    _confab_grounded = (
                        mem_ctx if isinstance(mem_ctx, list)
                        else ([mem_ctx] if mem_ctx else []))  # m116_a_confab_guard
                    # m118_d_referent_binding: pull the prior assistant
                    # turn text so Signal 4 can resolve demonstratives
                    # ("this classifier") to an antecedent entity from
                    # the last turn. _history has NOT yet been appended
                    # with the current user/assistant pair at this point
                    # (see appends at lines ~4079/4080), so _history[-1]
                    # is the prior assistant turn when available.
                    _prior_turn_content = ""  # m118_d_referent_binding
                    try:
                        for _h in reversed(_history):
                            if _h.get("role") == "assistant":
                                _prior_turn_content = _h.get("content", "") or ""
                                break
                    except Exception:
                        _prior_turn_content = ""
                    _guarded, _confab_diag = _confab_guard_fn(
                        response,
                        retrieval_hits=_confab_retrieval,
                        tool_calls_made=_confab_tools,
                        grounded_memory=_confab_grounded,
                        prior_turn_content=_prior_turn_content,  # m118_d_referent_binding
                        user_query=message,                      # m118_d_referent_binding
                    )  # m116_a_confab_guard
                    if _confab_diag and _confab_diag.get("verdict"):
                        print(
                            f"[m116_a_confab_guard] FIRED signals="
                            f"S1={_confab_diag['signal_1_fired']} "
                            f"S2={_confab_diag['signal_2_fired']} "
                            f"S3={_confab_diag['signal_3_fired']} "
                            f"S4={_confab_diag.get('signal_4_fired', False)} "
                            f"phrases={_confab_diag['matched_phrases'][:5]}",
                            flush=True,
                        )
                        if _guarded != response:
                            response = _guarded
                            yield (
                                f"data: {json.dumps({'type':'confab_guard','content':_guarded})}\n\n"
                            )
                except Exception as _confab_err:  # m116_a_confab_guard
                    # Guard must never break the response path
                    if str(_confab_err) != "already abstained":
                        print(
                            f"[m116_a_confab_guard] error: {_confab_err}",
                            flush=True,
                        )

            # --- inference telemetry derivations ---
            _stats = _last_stats or {}
            _ttft_ms = (int((_first_token_ts - _gen_start) * 1000)
                        if _first_token_ts else None)
            _total_ms = int((_gen_end - _gen_start) * 1000)
            _decode_ms = (_total_ms - (_ttft_ms or 0)) if _ttft_ms else None
            _tokens_decoded = _stats.get("tokens") or _rough_tokens(response)

            _decode_tok_s = None
            if _decode_ms and _decode_ms > 0 and _tokens_decoded:
                _decode_tok_s = round(
                    _tokens_decoded / (_decode_ms / 1000), 2)

            _prefill_tok_s = None
            if _ttft_ms and _ttft_ms > 0 and _prefill_tokens_est:
                _prefill_tok_s = round(
                    _prefill_tokens_est / (_ttft_ms / 1000), 1)

            # Bandwidth estimate: Q4 72B weights are ~40 GB on disk.
            # Each decoded token scans the full weight set once.
            # bandwidth = weight_bytes / ms_per_token * 1000
            _model_gb = 40.0
            _bandwidth_gb_s = None
            _dram_floor = 307.0
            _bandwidth_gap = None
            if _decode_ms and _tokens_decoded:
                _ms_per_tok = _decode_ms / _tokens_decoded
                if _ms_per_tok > 0:
                    _bandwidth_gb_s = round(
                        _model_gb / (_ms_per_tok / 1000), 1)
                    _bandwidth_gap = round(_dram_floor - _bandwidth_gb_s, 1)

            _ngram_drafted = _stats.get("ngram_drafted", 0)
            _ngram_accepted = _stats.get("ngram_accepted", 0)
            _drafts_per_accept = None
            if _ngram_accepted > 0:
                _drafts_per_accept = round(
                    _ngram_drafted / _ngram_accepted, 2)

            # m109_zeta_field_4: infer stop_reason from captured signals.
            # 72B server always emits finish_reason='stop' so we reconstruct
            # cause offline: loop_detected > verifier_error > max_tokens > eos.
            _m109_stop_reason = _m109_infer_stop_reason(
                tokens_decoded=_tokens_decoded,
                max_tokens=_stream_max_tokens,
                loop_detected=_m109_loop_broke,
                gen_error=_m109_gen_error,
                user_cancel=False)  # user_cancel not plumbed in stream path
            _turn_record(
                "generation",
                generation_started_ts=_gen_start,
                generation_ended_ts=_gen_end,
                first_token_ts=_first_token_ts,
                ttft_ms=_ttft_ms,
                total_ms=_total_ms,
                decode_ms=_decode_ms,
                prefill_tokens_est=_prefill_tokens_est,
                prefill_tok_s=_prefill_tok_s,
                decode_tok_s=_decode_tok_s,
                tokens_decoded=_tokens_decoded,
                tps=_stats.get("tps"),
                accept_rate=_stats.get("accept_rate"),
                ngram_drafted=_ngram_drafted,
                ngram_accepted=_ngram_accepted,
                cpu_drafted=_stats.get("cpu_drafted", 0),
                cpu_accepted=_stats.get("cpu_accepted", 0),
                temperature_used=_m73_temperature,
                repetition_penalty_used=_m73_rep_penalty,
                min_p_used=_m73_min_p,
                sampler_mode=("argmax" if _m73_temperature == 0.0 and _m73_rep_penalty == 1.0
                              else "temperature_only" if _m73_rep_penalty == 1.0
                              else "temp+rep_penalty"),
                query_type=_m73_dispatch["query_type"],
                drafts_per_accept_ngram=_drafts_per_accept,
                model_size_gb_assumed=_model_gb,
                dram_floor_gb_s=_dram_floor,
                bandwidth_gb_s_est=_bandwidth_gb_s,
                bandwidth_gap_vs_dram=_bandwidth_gap,
                response_text=response,
                response_chars=len(response),
                response_tokens_est=_rough_tokens(response),
                stop_reason=_m109_stop_reason,  # m109_zeta_field_4
                stop_reason_max_tokens_cap=_stream_max_tokens,  # m109_zeta_field_4
                stop_reason_loop_broke=_m109_loop_broke,  # m109_zeta_field_4
                stop_reason_gen_error=_m109_gen_error,  # m109_zeta_field_4
                # m123_a2_completion_reason: ζ v2.2 extension per M123
                # Stream A2. Additive enum derived from M109 stop_reason;
                # discriminates stream-terminated (truncation-class)
                # completions from natural EOS. Backward-compatible — old
                # consumers keep using stop_reason; new consumers get the
                # coarser, observably-stable enum.
                completion_reason=_m123_a2_completion_reason(
                    _m109_stop_reason),
            )

            # Update history
            _all_user_queries.append(message)
            _history.append({"role": "user", "content": message})
            _history.append({"role": "assistant", "content": response})
            _history = _trim_history(_history, MAX_HISTORY)

            # Main 46 v2: Pipeline extraction is a PARALLEL path to P2.
            # P2 stays on the old inline ingest path (unchanged, still
            # blocks conversation routes). The pipeline stashes the
            # SCRUBBED response for next turn's concurrent 8B extraction
            # with its own five-layer quality chain: scrub → grounding
            # gate → provenance → reactive dedup → reactive contradiction.
            # Stash on ALL non-empty, non-gated turns regardless of P2.
            if response and len(response) > 50:
                _recall_texts = [r.get("text", "") for r in
                                 (_TURN_LOG.get("retrieval", {})
                                  .get("recall_filtered") or [])]
                api_chat_stream._pending_response = {
                    "text": response,  # already scrubbed at this point
                    "recall_context": _recall_texts or None,
                    "turn": _TURN_LOG.get("turn_number", 0),
                }
            else:
                api_chat_stream._pending_response = None

            # Measure pipeline overlap. The bridge thread finishes in
            # ~300ms (queues to daemon + sleeps). The REAL extraction
            # runs in the daemon's _ane_extract_worker on ANE (~80s).
            # Check the 8B server status to detect actual ANE activity.
            _overlap_ms = 0
            _ane_active = False
            if _pipeline_start_ts:
                try:
                    import urllib.request as _ur
                    _h = json.loads(_ur.urlopen(
                        "http://127.0.0.1:8891/health", timeout=2).read())
                    _ane_active = _h.get("status") == "generating"
                except Exception:
                    pass
                if _ane_active:
                    # 8B is actively extracting on ANE while we just
                    # finished generating on GPU = true concurrency
                    _overlap_ms = int((time.time() - _pipeline_start_ts) * 1000)
                _turn_record("extraction",
                             ane_active_at_gen_end=_ane_active)
            _turn_record("extraction",
                         overlap_72b_8b_ms=_overlap_ms,
                         pipeline_stashed=bool(api_chat_stream._pending_response))

            # Quality signals: fabrication detection + stale-entity flags.
            # Main 38 Session 2 fix: record this turn's dispatched tools
            # into the rolling window BEFORE computing FAB claims so
            # that contextual references to prior turns don't false-
            # positive flag. The detector itself also checks prior
            # windows, so either way works, but recording first is safer.
            _tools_called_list = _TURN_LOG.get("routing", {}).get("tools_called", []) or []
            _record_dispatched_tools(_tools_called_list)
            _fab_claims = _detect_fabricated_tool_claims(response, _tools_called_list)
            # Main 40 P3: use the shared grounding-corpus module so the
            # live writer and the offline analyzers (flag_anomalies)
            # build identical context blobs. Previously the live writer
            # cherry-picked briefing + mem_ctx and OMITTED the tool
            # result preview, which meant `_detect_stale_entities` was
            # blind to stale entities surfacing through tool results
            # (the same gap that motivated the M38 close NOGROUND fix).
            from grounding_corpus import build_grounding_corpus_from_parts
            _ctx_blob = build_grounding_corpus_from_parts(
                mem_ctx_text=mem_ctx,
                briefing_text=presentation_briefing,
                tool_result_preview=_TURN_LOG.get("routing", {}).get("tool_result_preview"),
            )
            _turn_record("quality",
                         fabricated_tool_claims=_fab_claims,
                         stale_entities_in_response=_detect_stale_entities(response),
                         stale_entities_in_context=_detect_stale_entities(_ctx_blob),
                         tools_requested_not_called=_TURN_LOG.get(
                             "routing", {}).get("tools_requested_not_called", []),
                         # m109_zeta_field_2: populate grounded with gate state
                         # rather than leaving null. Tier 1 is disabled for
                         # Gemma 4 per answer_scrub.py:41.
                         grounded=_m109_grounding_state())

            # Post-turn: memory store total, turn complete.
            # M89 fix: _ai_result doesn't exist in streaming path (NameError was
            # silently swallowed). Use daemon stats for total; per-turn count is
            # inherently lagged in streaming path (extraction is async ~80s).
            try:
                global _prev_memory_store_total
                _mstats = memory.stats() or {}
                _cur_total = _mstats.get("total_memories") or 0
                _stored_delta = max(0, _cur_total - (_prev_memory_store_total or _cur_total))
                _prev_memory_store_total = _cur_total
                # m109_zeta_field_3: per-memory manifest. Preserve backward
                # compat by keeping `memories_stored_this_turn` as the count
                # (int) AND adding a parallel `memories_stored_this_turn_manifest`
                # list. Existing parsers that read `memories_stored_this_turn`
                # as an int continue to work; new parsers can consume the
                # manifest field. Approach chosen: sibling field (not nested).
                _m109_manifest = _m109_memory_manifest(_stored_delta)
                _turn_record("post_turn",
                             memory_store_total=_cur_total,
                             memories_stored_this_turn=_stored_delta,
                             memories_stored_this_turn_manifest=_m109_manifest)
            except Exception as _e:
                _turn_record("post_turn", memory_store_total=None,
                             memories_stored_this_turn=0,
                             memories_stored_this_turn_manifest=[],
                             post_turn_error=str(_e)[:100])

            # Main 46: log scrub results
            # M120 C: add tier2_conflict_flagged (registry-value conflicts).
            # m122_a3_zeta_v22: tier2_binding.skip_reason stub (A2 will populate
            # the semantics). Emit only when there is signal to report —
            # additive + optional per Pattern-1 guardrail. A2 will wire the
            # bare-scalar-guard + abstention-pattern outputs through scrub_response
            # and populate the per-evaluation skip_reason list; until then this
            # carries an empty list so consumers never KeyError.
            if _scrub_result:
                _tier2_skip_reasons = _scrub_result.get(
                    "tier2_binding_skip_reasons", [])  # m122_a3_zeta_v22
                _scrub_fields = dict(
                    tier1_flags=_scrub_result.get("tier1_flags", []),
                    tier2_flags=_scrub_result.get("tier2_flags", []),
                    tier2_conflict_flagged=_scrub_result.get("tier2_conflict_flagged", []),
                    total_flags=_scrub_result.get("total_flags", 0),
                    sentences_stripped=_scrub_result.get("sentences_stripped", 0),
                    latency_ms=_scrub_result.get("latency_ms", 0),
                    original_response_chars=_scrub_result.get("original_response_chars", 0),
                    cleaned_response_chars=_scrub_result.get("cleaned_response_chars", 0),
                    # M125.3 Stream B P1: scrub-tier attribution.
                    # Closes the Phase 0 gap where total_flags>0 but
                    # tier1/tier2/tier2_conflict arrays were all empty.
                    tier0_flags=_scrub_result.get("tier0_flags", []),
                    tier2_narrative_flags=_scrub_result.get(
                        "tier2_narrative_flags", []),
                    tier2_narrative_verdict=_scrub_result.get(
                        "tier2_narrative_verdict", "PASS"),
                    tier3_flags=_scrub_result.get("tier3_flags", []),
                    narrative_overstrip_triggered=_scrub_result.get(
                        "narrative_overstrip_triggered", False),
                    scrub_mechanism_fired=_scrub_result.get(
                        "scrub_mechanism_fired", []),
                    # M125.3 Stream B P1: pre-gate tier list — which
                    # tier would have fired absent the P2 relevance gate.
                    scrub_mechanism_pre_gate=_scrub_result.get(
                        "scrub_mechanism_pre_gate", []),
                    # M125.3 Stream B P1 pre-gate attribution counts.
                    tier1_flags_pre_gate_count=_scrub_result.get(
                        "tier1_flags_pre_gate_count", 0),
                    tier2_flags_pre_gate_count=_scrub_result.get(
                        "tier2_flags_pre_gate_count", 0),
                    tier2_narrative_flags_pre_gate_count=_scrub_result.get(
                        "tier2_narrative_flags_pre_gate_count", 0),
                    tier3_flags_pre_gate_count=_scrub_result.get(
                        "tier3_flags_pre_gate_count", 0),
                    # M125.3 Stream B P2: scrub-relevance gate state.
                    canonical_atom_skip_count=_scrub_result.get(
                        "canonical_atom_skip_count", 0),
                    canonical_atoms_considered=_scrub_result.get(
                        "canonical_atoms_considered", 0),
                )
                # Emit canonical_atom_skips detail only when non-empty
                # (keeps turn JSON lean on the common path).
                _m125_3_skips = _scrub_result.get(
                    "canonical_atom_skips", []) or []
                if _m125_3_skips:
                    _scrub_fields["canonical_atom_skips"] = _m125_3_skips
                if _tier2_skip_reasons:  # m122_a3_zeta_v22 — emit only when non-empty
                    _scrub_fields["tier2_binding"] = {
                        "skip_reason": _tier2_skip_reasons,
                    }
                _turn_record("scrub", **_scrub_fields)

            # m122_a3_zeta_v22: confab_guard_diagnostic — per-signal firing
            # state from confabulation_shape_detector. Distinguishes
            # "guard ran + passed" from "guard didn't run" from
            # "guard ran + fired". Emit only when guard was invoked
            # (i.e. _confab_diag is not None).
            if _confab_diag is not None:  # m122_a3_zeta_v22
                _turn_record("scrub",
                             confab_guard_diagnostic={
                                 "signal_1_fired": bool(_confab_diag.get("signal_1_fired", False)),
                                 "signal_2_fired": bool(_confab_diag.get("signal_2_fired", False)),
                                 "signal_3_fired": bool(_confab_diag.get("signal_3_fired", False)),
                                 "signal_4_fired": bool(_confab_diag.get("signal_4_fired", False)),
                                 "match_text_per_signal": _confab_diag.get("matched_phrases", [])[:20],
                                 "verdict": bool(_confab_diag.get("verdict", False)),
                             })

            _turn_write()

            # M125 A5.1: stash tool dispatched this turn for next-turn
            # anaphoric / subject-continuity fold-in.
            _last_tool = tool_name or ""

            # Schedule idle
            if _idle_queue and response:
                _idle_queue.schedule(injected_memories=_last_subconscious, response_text=response)

            # Send final stats
            stats = _last_stats or {}
            sse_data = json.dumps({"type": "done", "stats": {
                "tps": stats.get("tps", 0), "tokens": stats.get("tokens", 0),
                "accept_rate": stats.get("accept_rate", 0),
                "elapsed": stats.get("elapsed", 0),
            }, "memories_recalled": len(_last_subconscious)})
            yield f"data: {sse_data}\n\n"

    return app.response_class(generate_sse(), mimetype="text/event-stream")


@app.route("/health/system")
def api_health_system():
    """M83 Priority 0 — exposes latest system_health probe reading.

    Reads data/system_health/latest.json written by tools/system_health_probe.py
    (launchd com.midas.system_health.probe, 5-min cadence + session-open
    synchronous + pre-multi-agent + post-measurement). Returns 503 if probe
    heartbeat is stale (>15 min) — indicates probe itself has failed.

    Compressor-aware DRAM threshold per m83_rule1_amendment.
    """
    base = Path("/Users/midas/Desktop/cowork/data/system_health")
    latest = base / "latest.json"
    heartbeat = base / "probe_heartbeat"
    if not latest.exists():
        return jsonify({"error": "probe has not run yet",
                        "expected_path": str(latest)}), 503
    try:
        snapshot = json.loads(latest.read_text())
    except Exception as e:
        return jsonify({"error": f"latest.json unreadable: {e}"}), 503
    # Heartbeat staleness check
    stale = False
    if heartbeat.exists():
        try:
            hb_age = time.time() - heartbeat.stat().st_mtime
            snapshot["_probe_heartbeat_age_sec"] = round(hb_age, 1)
            stale = hb_age > 900  # 15 min
        except Exception:
            pass
    if stale:
        snapshot["_warning"] = "probe heartbeat >15 min stale; probe itself may have failed"
        return jsonify(snapshot), 503
    return jsonify(snapshot)


@app.route("/health/trajectory")
def api_health_trajectory():
    """M98 Stage 2 Agent 2e — operator-facing trajectory surface.

    Returns the last 60 trajectory records (60 min at 60s cadence) from
    data/system_health/trajectory.jsonl, the latest delta record (all windows)
    from deltas.jsonl, and the current v2 probe snapshot from latest.json.

    Graceful degradation: if trajectory.jsonl or deltas.jsonl do not yet exist
    (2c / 2d still running or not yet shipped), return empty arrays with
    status=awaiting_data rather than 5xx. latest.json is expected to exist
    (probe has been live since M83); if absent we still return 200 with
    snapshot=None and status=awaiting_probe.

    Format: JSON-RPC style (not HTML). This endpoint is operator tooling +
    dashboard source, not a server-rendered page.
    """
    base = Path("/Users/midas/Desktop/cowork/data/system_health")
    trajectory_path = base / "trajectory.jsonl"
    deltas_path = base / "deltas.jsonl"
    latest_path = base / "latest.json"

    out = {
        "status": "ok",
        "served_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "trajectory": [],
        "trajectory_count": 0,
        "latest_delta": None,
        "snapshot": None,
        "sources": {
            "trajectory": str(trajectory_path),
            "deltas": str(deltas_path),
            "snapshot": str(latest_path),
        },
        "awaiting": [],
    }

    # ── Trajectory: last 60 records ──
    if trajectory_path.exists():
        try:
            # Efficient tail-read: slurp lines, keep last 60
            with trajectory_path.open("r") as fh:
                lines = fh.readlines()
            tail = lines[-60:] if len(lines) > 60 else lines
            records = []
            for line in tail:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    # Skip malformed lines; don't fail the whole endpoint
                    continue
            out["trajectory"] = records
            out["trajectory_count"] = len(records)
        except Exception as e:
            out["trajectory_error"] = str(e)[:200]
            out["awaiting"].append("trajectory")
    else:
        out["awaiting"].append("trajectory")

    # ── Deltas: latest record (all windows) ──
    if deltas_path.exists():
        try:
            with deltas_path.open("r") as fh:
                lines = fh.readlines()
            # Find last non-empty JSON line
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    out["latest_delta"] = json.loads(line)
                    break
                except Exception:
                    continue
            if out["latest_delta"] is None:
                out["awaiting"].append("deltas")
        except Exception as e:
            out["deltas_error"] = str(e)[:200]
            out["awaiting"].append("deltas")
    else:
        out["awaiting"].append("deltas")

    # ── Snapshot: current probe reading ──
    if latest_path.exists():
        try:
            out["snapshot"] = json.loads(latest_path.read_text())
        except Exception as e:
            out["snapshot_error"] = str(e)[:200]
            out["awaiting"].append("snapshot")
    else:
        out["awaiting"].append("snapshot")

    if out["awaiting"]:
        out["status"] = "awaiting_data"

    return jsonify(out)


@app.route("/api/subconscious/health")
def api_subconscious_health():
    """Main 34 S1C: aggregated subconscious health.

    Foundation for the Phase 3 unified daemon. Surfaces:
      - memory store size
      - ane queue depth
      - unresolved contradiction count (from semantic_supersede records)
      - upstream server health (72B :8899, 8B ANE :8891)
    """
    out = {"ts": time.time()}
    try:
        out["memory_stats"] = memory.stats()
    except Exception as e:
        out["memory_stats"] = {"error": str(e)}
    try:
        import sys as _sys
        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
        from semantic_supersede import count_unresolved_contradictions
        out["contradictions_unresolved"] = count_unresolved_contradictions()
    except Exception as e:
        out["contradictions_unresolved"] = -1
        out["contradictions_error"] = str(e)
    upstream = {}
    for name, url in (("qwen72b", "http://127.0.0.1:8899/health"),
                      ("ane8b", "http://127.0.0.1:8891/health")):
        try:
            import urllib.request as _u
            with _u.urlopen(url, timeout=1.0) as r:
                upstream[name] = {"ok": True, "code": r.status}
        except Exception as e:
            upstream[name] = {"ok": False, "error": str(e)[:120]}
    out["upstream"] = upstream
    out["session"] = {
        "messages_sent": _session.get("messages_sent", 0),
        "memories_recalled": _session.get("memories_recalled", 0),
        "facts_extracted": _session.get("facts_extracted", 0),
    }
    return jsonify(out)


@app.route("/api/session/context")
def api_session_context():
    """Main 35 +3 T3 — what does the user see when they sit down at Midas?

    Aggregates: time-since-last-session, recent vault activity, daemon
    event log summary, and a heuristic active-topic guess. Returns a
    compact JSON the chat surface can render as a brief status line.
    """
    import os as _os, glob as _glob, json as _json
    out = {"ts": time.time()}

    # ── 1. recent event log activity (since 24h) ──
    log_dir = "/Users/midas/Desktop/cowork/vault/subconscious"
    cutoff = time.time() - 24 * 3600
    event_counts = {}
    n_events = 0
    n_loops = 0
    n_recalls = 0
    n_enrichments = 0
    last_event_ts = None
    for fp in sorted(_glob.glob(f"{log_dir}/event_log_*.jsonl")):
        try:
            with open(fp) as fh:
                for line in fh:
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    ts_str = rec.get("ts") or ""
                    # ts is ISO8601 with Z; quick parse
                    try:
                        from datetime import datetime as _dt
                        ts_epoch = _dt.fromisoformat(ts_str.rstrip("Z")).timestamp()
                    except Exception:
                        continue
                    if ts_epoch < cutoff:
                        continue
                    n_events += 1
                    last_event_ts = max(last_event_ts or 0, ts_epoch)
                    t = rec.get("type", "?")
                    event_counts[t] = event_counts.get(t, 0) + 1
                    if t == "loop_fired":
                        n_loops += 1
                    elif t == "memory_recalled":
                        n_recalls += 1
                    elif t == "enrichment_complete":
                        n_enrichments += 1
        except OSError:
            continue
    hours_since_last = (time.time() - last_event_ts) / 3600.0 if last_event_ts else None
    out["since_last_event"] = {
        "hours_ago": round(hours_since_last, 2) if hours_since_last else None,
        "n_events_24h": n_events,
        "loops_fired_24h": n_loops,
        "memories_recalled_24h": n_recalls,
        "enrichments_24h": n_enrichments,
        "by_type": event_counts,
    }

    # ── 2. daemon health ──
    try:
        import urllib.request as _u
        with _u.urlopen("http://127.0.0.1:8452/api/subconscious/status",
                        timeout=2.0) as r:
            ds = _json.loads(r.read())
        out["daemon"] = {
            "uptime_h": round(ds.get("uptime_s", 0) / 3600, 1),
            "events_emitted": ds["components"]["event_bus"]["events_emitted"],
            "components_running": sum(
                1 for c in ds["components"].values()
                if c.get("status") == "running"),
            "upstream": ds["components"]["health_monitor"].get("upstream", {}),
        }
    except Exception as e:
        out["daemon"] = {"error": str(e)[:120]}

    # ── 3. memory store snapshot ──
    try:
        ms = memory.stats()
        out["memory"] = {
            "total": ms.get("total_memories", 0),
            "session": ms.get("session"),
        }
    except Exception as e:
        out["memory"] = {"error": str(e)[:120]}

    # ── 4. recent queue activity ──
    try:
        from pathlib import Path
        queue_dir = Path("/Users/midas/Desktop/cowork/vault/agent_reports/queue")
        recent = sorted(queue_dir.glob("*_summary.md"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        out["recent_queues"] = [
            {"file": p.name, "modified_h_ago":
             round((time.time() - p.stat().st_mtime) / 3600, 1)}
            for p in recent
        ]
    except Exception as e:
        out["recent_queues"] = []

    # ── 5. heuristic active topic (from recent event log topic tags) ──
    # If the daemon has a context tracker hooked up later, swap to that.
    # For now: count topic mentions in last-24h memory_recalled events.
    topic_counts = {}
    try:
        for fp in sorted(_glob.glob(f"{log_dir}/event_log_*.jsonl"))[-2:]:
            with open(fp) as fh:
                for line in fh:
                    try:
                        rec = _json.loads(line)
                    except Exception:
                        continue
                    if rec.get("type") not in ("memory_recalled", "memory_ingested"):
                        continue
                    details = rec.get("details") or {}
                    q = (details.get("query") or "").lower()
                    for kw, tag in (
                        ("slc", "hardware"), ("ane", "hardware"),
                        ("amcc", "hardware"), ("dcs", "hardware"),
                        ("midas", "midas"), ("/api", "midas"),
                        ("subconscious", "subconscious"),
                        ("memory", "subconscious"),
                        ("paper", "paper"), ("locomo", "paper"),
                    ):
                        if kw in q:
                            topic_counts[tag] = topic_counts.get(tag, 0) + 1
                            break
    except Exception:
        pass
    primary = max(topic_counts, key=topic_counts.get) if topic_counts else None
    out["active_context"] = {
        "primary_topic_guess": primary,
        "topic_counts_24h": topic_counts,
    }

    # Main 35 +5 Tier 1: live context tracker state. This is the SOURCE
    # OF TRUTH for the viz's active-topic HUD — it reflects the actual
    # multi-topic weight vector after on_message has been called for every
    # chat message in this midas process. Falls back to the heuristic
    # primary_topic_guess above if the tracker hasn't seen any messages.
    try:
        tracker = _get_context_tracker()
        if tracker is not None:
            out["context_tracker"] = tracker.state()
    except Exception as e:
        out["context_tracker"] = {"error": str(e)[:120]}

    # Phase 3: warm open state
    try:
        out["warm_open"] = get_session_context_extension()
    except Exception:
        pass

    return jsonify(out)


@app.route("/api/queue_depth")
def api_queue_depth():
    try:
        s = memory.stats()
        return jsonify({"ane_queue_depth": s.get("ane_queue_depth", 0)})
    except Exception:
        return jsonify({"ane_queue_depth": -1})


@app.route("/api/stats")
def api_stats():
    mem_stats = {}
    try:
        mem_stats = memory.stats()
    except Exception:
        pass

    # Check hardware
    gpu_online = False
    try:
        r = urllib.request.urlopen(MLX_BASE_URL.rstrip("/") + "/models", timeout=2)
        gpu_online = r.status == 200
    except Exception:
        pass

    ane_online = False
    try:
        # 8B ANE extractor on :8891 (Main 18+). Old port 8423 was the
        # Llama-1B ANE server retired when the verifier swapped to Qwen.
        ane_online = urllib.request.urlopen("http://localhost:8891/health", timeout=2).status == 200
    except Exception:
        pass

    return jsonify({
        "hardware": {
            "gpu": {"status": "active" if gpu_online else "down", "label": "Gemma 4 31B Q4 | 17.5 tok/s"},
            "ane": {"status": "active" if ane_online else "idle", "label": "8B Q8 | 7.9 tok/s | 72d"},
            "cpu": {"status": "active", "label": "AMX + scrub"},
            "memory": {"status": "active", "label": f"{mem_stats.get('total_memories', 0):,} facts | LocalMemoryStore"},
        },
        "session": _session,
        "subconscious": _last_subconscious,
    })


@app.route("/api/feed")
def api_feed():
    return jsonify({"events": _feed})


# ── HTML/CSS/JS ─────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MIDAS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #0a0e14;
    --bg-elevated: #0d1117;
    --bg-panel: #0c1018;
    --border: #1a1f2e;
    --text: #c5cdd8;
    --text-dim: #4a5568;
    --text-muted: #2d3748;
    --cyan: #00d4ff;
    --green: #00ff88;
    --red: #ff4444;
    --cyan-glow: rgba(0, 212, 255, 0.15);
    --green-glow: rgba(0, 255, 136, 0.15);
}

html, body {
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1a1f2e; }
::-webkit-scrollbar-thumb:hover { background: #2d3748; }

/* Layout */
.container {
    display: flex;
    height: 100vh;
    width: 100vw;
}

/* Left panel: Chat */
.chat-panel {
    flex: 0 0 65%;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    min-width: 0;
}

.chat-header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
}

.chat-header .logo {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 4px;
    color: var(--cyan);
}

.chat-header .status {
    font-size: 11px;
    color: var(--text-dim);
    letter-spacing: 1px;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.msg {
    max-width: 85%;
    padding: 8px 12px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

.msg-user {
    align-self: flex-end;
    border: 1px solid var(--border);
    color: var(--text);
}

.msg-midas {
    align-self: flex-start;
    border-left: 2px solid var(--cyan);
    padding-left: 14px;
    color: var(--text);
}

.msg-meta {
    font-size: 11px;
    color: var(--text-dim);
    padding: 2px 14px 8px 14px;
    align-self: flex-start;
}

.msg-meta span { margin-right: 12px; }
.msg-meta .route { color: var(--cyan); opacity: 0.6; }
.msg-meta .tps { color: var(--green); opacity: 0.6; }
.msg-meta .tool-tag { color: var(--text-dim); }

.typing-indicator {
    align-self: flex-start;
    padding: 8px 14px;
    color: var(--cyan);
    font-size: 12px;
    opacity: 0.7;
    display: none;
}

.typing-indicator.visible { display: block; }

.chat-input-bar {
    padding: 12px 20px;
    border-top: 1px solid var(--border);
    flex-shrink: 0;
}

.chat-input-bar input {
    width: 100%;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text);
    font-family: inherit;
    font-size: 13px;
    padding: 10px 12px;
    outline: none;
    transition: border-color 0.2s;
}

.chat-input-bar input:focus {
    border-color: var(--cyan);
}

.chat-input-bar input::placeholder {
    color: var(--text-muted);
}

/* Right panel: Dashboard */
.dash-panel {
    flex: 0 0 35%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
}

.dash-section {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
}

.dash-section.feed-section {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.dash-section.sub-section {
    flex: 0 0 auto;
    max-height: 180px;
    overflow-y: auto;
}

.section-title {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Hardware rows */
.hw-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 11px;
}

.hw-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}

.hw-dot.active {
    background: var(--green);
    box-shadow: 0 0 6px var(--green-glow);
}

.hw-dot.idle {
    background: var(--text-muted);
}

.hw-dot.down {
    background: var(--red);
}

.hw-dot.pulse {
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.hw-label {
    width: 52px;
    color: var(--text-dim);
    flex-shrink: 0;
    font-size: 11px;
}

.hw-metric {
    color: var(--text);
    font-size: 11px;
}

/* Session counters */
.counter-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 16px;
    font-size: 11px;
}

.counter-item {
    display: flex;
    justify-content: space-between;
}

.counter-label { color: var(--text-dim); }
.counter-value { color: var(--text); font-weight: 500; }

/* Feed */
.feed-item {
    padding: 4px 0;
    font-size: 11px;
    line-height: 1.4;
    display: flex;
    gap: 8px;
}

.feed-time {
    color: var(--text-muted);
    flex-shrink: 0;
    font-size: 10px;
}

.feed-text {
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.feed-item.extraction .feed-text { color: var(--green); opacity: 0.7; }
.feed-item.recall .feed-text { color: var(--cyan); opacity: 0.7; }

/* Subconscious */
.sub-item {
    padding: 3px 0;
    font-size: 11px;
    line-height: 1.4;
    display: flex;
    gap: 8px;
}

.sub-score {
    color: var(--cyan);
    opacity: 0.6;
    flex-shrink: 0;
    font-size: 10px;
    min-width: 32px;
    text-align: right;
}

.sub-text {
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.empty-state {
    font-size: 11px;
    color: var(--text-muted);
    font-style: italic;
}

/* Code blocks in responses */
.msg code {
    background: var(--bg-elevated);
    padding: 1px 4px;
    font-size: 12px;
}

.msg pre {
    background: var(--bg-elevated);
    padding: 8px 10px;
    margin: 4px 0;
    overflow-x: auto;
    font-size: 12px;
}
</style>
</head>
<body>
<div class="container">

    <!-- Left: Chat -->
    <div class="chat-panel">
        <div class="chat-header">
            <span class="logo">MIDAS</span>
            <span class="status">LOCAL / GEMMA 4 31B Q4 / BARE METAL</span>
        </div>
        <div class="chat-messages" id="messages"></div>
        <div class="typing-indicator" id="typing">processing...</div>
        <div class="chat-input-bar">
            <input type="text" id="input" placeholder="..." autocomplete="off" spellcheck="false">
        </div>
    </div>

    <!-- Right: Dashboard -->
    <div class="dash-panel">
        <div class="dash-section">
            <div class="section-title">HARDWARE</div>
            <div id="hw-gpu" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">GPU</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-ane" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">ANE</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-cpu" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">CPU</span>
                <span class="hw-metric">--</span>
            </div>
            <div id="hw-mem" class="hw-row">
                <div class="hw-dot idle"></div>
                <span class="hw-label">MEM</span>
                <span class="hw-metric">--</span>
            </div>
        </div>

        <div class="dash-section">
            <div class="section-title">THIS SESSION</div>
            <div class="counter-grid">
                <div class="counter-item">
                    <span class="counter-label">messages</span>
                    <span class="counter-value" id="ct-msg">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">recalled</span>
                    <span class="counter-value" id="ct-recall">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">extracted</span>
                    <span class="counter-value" id="ct-extract">0</span>
                </div>
                <div class="counter-item">
                    <span class="counter-label">tools</span>
                    <span class="counter-value" id="ct-tools">0</span>
                </div>
            </div>
        </div>

        <div class="dash-section feed-section">
            <div class="section-title">MEMORY FEED</div>
            <div id="feed"><div class="empty-state">no activity yet</div></div>
        </div>

        <div class="dash-section sub-section">
            <div class="section-title">SUBCONSCIOUS</div>
            <div id="subconscious"><div class="empty-state">no memories injected</div></div>
        </div>
    </div>

</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const typingEl = document.getElementById('typing');
let sending = false;

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function addMessage(role, text, meta) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + role;
    div.textContent = text;
    messagesEl.appendChild(div);

    if (meta && role === 'midas') {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        let parts = [];
        if (meta.route_layer) parts.push('<span class="route">' + meta.route_layer + '</span>');
        if (meta.tool && meta.tool !== 'conversation') parts.push('<span class="tool-tag">' + escapeHtml(meta.tool) + '</span>');
        if (meta.stats && meta.stats.tps) parts.push('<span class="tps">' + meta.stats.tps + ' tok/s</span>');
        if (meta.stats && meta.stats.accept_rate) parts.push('<span class="route">' + meta.stats.accept_rate + '% acc</span>');
        if (meta.stats && meta.stats.tokens) parts.push('<span class="route">' + meta.stats.tokens + ' tok</span>');
        if (parts.length) {
            metaDiv.innerHTML = parts.join('');
            messagesEl.appendChild(metaDiv);
        }
    }

    messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || sending) return;
    sending = true;
    inputEl.value = '';
    addMessage('user', text);
    typingEl.classList.add('visible');

    // Pulse GPU dot while generating
    const gpuDot = document.querySelector('#hw-gpu .hw-dot');
    gpuDot && gpuDot.classList.add('pulse');

    try {
        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: text}),
        });

        // Create message div for streaming
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg msg-midas';
        msgDiv.textContent = '';
        messagesEl.appendChild(msgDiv);

        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        messagesEl.appendChild(metaDiv);

        let fullText = '';
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const dataStr = line.slice(6).trim();
                if (!dataStr || dataStr === '[DONE]') continue;

                try {
                    const chunk = JSON.parse(dataStr);
                    if (chunk.type === 'token') {
                        fullText += chunk.content;
                        msgDiv.textContent = fullText;
                        messagesEl.scrollTop = messagesEl.scrollHeight;
                    } else if (chunk.type === 'done' && chunk.stats) {
                        const s = chunk.stats;
                        let parts = [];
                        if (s.tps) parts.push('<span class="tps">' + s.tps.toFixed(1) + ' tok/s</span>');
                        if (s.accept_rate) parts.push('<span class="route">' + s.accept_rate.toFixed(0) + '% acc</span>');
                        if (s.tokens) parts.push('<span class="route">' + s.tokens + ' tok</span>');
                        metaDiv.innerHTML = parts.join(' ');
                    }
                } catch (e) {}
            }
        }
    } catch (e) {
        addMessage('midas', 'CONNECTION ERROR: ' + e.message);
    }

    gpuDot && gpuDot.classList.remove('pulse');
    typingEl.classList.remove('visible');
    sending = false;
    refreshFeed();
    refreshStats();
}

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Dashboard polling
function updateHw(id, data) {
    const row = document.getElementById(id);
    if (!row || !data) return;
    const dot = row.querySelector('.hw-dot');
    const metric = row.querySelector('.hw-metric');
    dot.className = 'hw-dot ' + data.status;
    if (data.status === 'active') {
        dot.style.background = 'var(--green)';
        dot.style.boxShadow = '0 0 6px var(--green-glow)';
    } else if (data.status === 'down') {
        dot.style.background = 'var(--red)';
        dot.style.boxShadow = 'none';
    } else {
        dot.style.background = 'var(--text-muted)';
        dot.style.boxShadow = 'none';
    }
    metric.textContent = data.label;
}

function updateSubconscious(items) {
    const el = document.getElementById('subconscious');
    if (!items || !items.length) {
        el.innerHTML = '<div class="empty-state">no memories injected</div>';
        return;
    }
    el.innerHTML = items.map(m =>
        '<div class="sub-item">' +
        '<span class="sub-score">' + (m.score || 0).toFixed(2) + '</span>' +
        '<span class="sub-text">' + escapeHtml(m.text) + '</span>' +
        '</div>'
    ).join('');
}

async function refreshStats() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();
        if (data.hardware) {
            updateHw('hw-gpu', data.hardware.gpu);
            updateHw('hw-ane', data.hardware.ane);
            updateHw('hw-cpu', data.hardware.cpu);
            updateHw('hw-mem', data.hardware.memory);
        }
        if (data.session) {
            document.getElementById('ct-msg').textContent = data.session.messages_sent || 0;
            document.getElementById('ct-recall').textContent = data.session.memories_recalled || 0;
            document.getElementById('ct-extract').textContent = data.session.facts_extracted || 0;
            document.getElementById('ct-tools').textContent = data.session.tools_used || 0;
        }
        if (data.subconscious) {
            updateSubconscious(data.subconscious);
        }
    } catch (e) {}
}

async function refreshFeed() {
    try {
        const resp = await fetch('/api/feed');
        const data = await resp.json();
        const el = document.getElementById('feed');
        if (!data.events || !data.events.length) {
            el.innerHTML = '<div class="empty-state">no activity yet</div>';
            return;
        }
        el.innerHTML = data.events.map(ev =>
            '<div class="feed-item ' + ev.type + '">' +
            '<span class="feed-time">' + ev.time + '</span>' +
            '<span class="feed-text">' + escapeHtml(ev.text) + '</span>' +
            '</div>'
        ).join('');
    } catch (e) {}
}

// Poll every 2s
setInterval(refreshStats, 2000);
setInterval(refreshFeed, 5000);

// Initial load
refreshStats();
refreshFeed();
inputEl.focus();
</script>
</body>
</html>
"""


# ── Boot & Run ──────────────────────────────────────────────────────────────

def boot():
    """Initialize memory and wire singletons."""
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        memory.start()
    except Exception as e:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        print(f"Boot failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    sys.stdout, sys.stderr = old_stdout, old_stderr

    set_memory(memory)

    global _idle_queue
    _idle_queue = IdleQueue(memory)

    # Browser (optional)
    try:
        from browser import BrowserBridge
        browser = BrowserBridge()
        if browser.is_available():
            browser.connect()
            set_browser(browser)
    except Exception:
        pass

    # Test LLM
    try:
        llm_fn([{"role": "user", "content": "hi"}], max_tokens=5, temperature=0.0)
    except Exception:
        print("LLM offline -- is MLX server running on :8899?")
        sys.exit(1)

    # Main 39 P1.5: revalidate the accumulated standing rules now that
    # the LLM is confirmed reachable. Filters historical regex false
    # positives. See _revalidate_accumulated_rules for the sentinel
    # health check that protects against transient classifier failures.
    global _ACTIVE_STANDING_RULES
    if _RAW_ACCUMULATED_RULES:
        _ACTIVE_STANDING_RULES = _revalidate_accumulated_rules(_RAW_ACCUMULATED_RULES)
        if _ACTIVE_STANDING_RULES:
            print(
                f"[session] {len(_ACTIVE_STANDING_RULES)} standing rules active "
                f"(post-revalidation) — injected into system message every turn:",
                flush=True,
            )
            for _r in _ACTIVE_STANDING_RULES:
                print(f"  - {_r}", flush=True)
        else:
            print("[session] no standing rules survived revalidation", flush=True)

    stats = memory.stats()
    print(f"  MIDAS UI booting...")
    print(f"  {stats.get('total_memories', 0)} memories loaded")

    # Main 38 P3: run the auto-populating measurement registry loop at
    # boot so user-stated measurements from the prior session surface in
    # data/measurement_registry.json before the first turn of this new
    # session. Hourly launchd maintenance also runs it, but boot-time
    # invocation guarantees "next session open shows the updated value"
    # per the directive, independent of launchd timing.
    try:
        import sys as _sys
        _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
        from maintenance import auto_populate_measurement_registry
        reg_stats = auto_populate_measurement_registry()
        print(f"  registry auto-populate: +{reg_stats.get('added',0)} "
              f"updated {reg_stats.get('updated',0)} "
              f"total {reg_stats.get('total',0)}")
    except Exception as _e:
        print(f"  registry auto-populate failed: {_e}")

    # M50 P2: Boot-time self-consistency check.
    # Compare registry canonical values against active memories.
    # Would have caught T47 (memory says overlap=0, registry says 22402).
    try:
        _consistency_issues = _boot_consistency_check()
        if _consistency_issues:
            print(f"  CONSISTENCY: {len(_consistency_issues)} issue(s) found:")
            for _ci in _consistency_issues:
                print(f"    {_ci}")
        else:
            print(f"  consistency check: CLEAN")
    except Exception as _ce:
        print(f"  consistency check failed: {_ce}")

    print(f"  http://127.0.0.1:{PORT}")


def _boot_consistency_check() -> list:
    """M50 P2: Check registry canonical values against active memories.

    For each registry entry with status=canonical, search the memory
    store for active memories mentioning the same entity. If a memory
    contains a contradictory value (e.g., "overlap = 0" when registry
    says 22402), log it. Returns list of issue strings.
    """
    issues = []
    _base = "/Users/midas/Desktop/cowork"
    try:
        reg_path = os.path.join(_base, "data/measurement_registry.json")
        with open(reg_path) as f:
            reg = json.load(f)

        import sqlite3
        db_path = os.path.join(_base,
                               "orion-ane/memory/chromadb_live/memory_local.db")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        # Check key canonical entries for contradictions.
        # M103: mirror of M102 narrative_retrieval fix. Some registry entries
        # are bare scalars (float/bool/str) written by pipeline tools (e.g.
        # tools/m96/m96_analyze.py) that bypass the dict schema. Skip them
        # here so a single malformed row can't trigger the boot CHECK_ERROR
        # `'float' object has no attribute 'get'`. Root cause + full catalog
        # in vault/agent_reports/m102_narrative_retrieval_fix.md.
        check_entries = [
            k for k, v in reg.items()
            if isinstance(v, dict)
            and v.get("status") == "canonical"
            and v.get("value")
            and str(v["value"]).replace("+", "").replace("-", "").replace("<", "").replace("%", "").strip().replace(".", "").isdigit()
        ]

        for key in check_entries:
            entry = reg[key]
            entity = entry.get("entity", "")
            value = str(entry.get("value", ""))
            # Search for canonical memories about this entity that might
            # contain contradictory values
            rows = conn.execute('''
                SELECT id, text FROM memories
                WHERE source_role = 'canonical'
                AND superseded_by IS NULL
                AND lower(text) LIKE ?
            ''', (f"%{entity.lower()}%",)).fetchall()

            for mem_id, mem_text in rows:
                # Check for explicit contradictions:
                # e.g., memory says "overlap = 0" but registry says "22402"
                # or memory says "fully serial" but registry entity is "concurrency"
                mem_lower = mem_text.lower()
                if entity == "concurrency" and (
                    "serial" in mem_lower or "overlap = 0" in mem_lower
                    or "do not overlap" in mem_lower
                ):
                    if value != "0":
                        issues.append(
                            f"CONTRADICTION: {key}={value} but memory "
                            f"{mem_id} says serial/overlap=0")
                # Generic: if the memory explicitly states a different
                # numeric value for this measurement
                if entry.get("unit") and value.isdigit():
                    # Look for "entity ... 0" patterns when value isn't 0
                    import re
                    zero_pattern = re.compile(
                        rf'{re.escape(entity)}.*?\b0\s*(?:{re.escape(str(entry.get("unit", "")))}|ms|%)',
                        re.IGNORECASE)
                    if zero_pattern.search(mem_text) and value != "0":
                        issues.append(
                            f"CONTRADICTION: {key}={value}{entry.get('unit','')} "
                            f"but memory {mem_id} says 0")

        conn.close()
    except Exception as e:
        issues.append(f"CHECK_ERROR: {e}")

    return issues


def _resume_incomplete_queues():
    """Main 34 S2A: on boot, scan disk-backed queue log and re-fire any
    queues that started but never reached complete. Each recovered queue
    gets a fresh worker thread for its remaining tasks under the SAME
    queue_id so its lifecycle log is continuous.
    """
    try:
        pending = _qp.recover_incomplete_queues()
    except Exception as e:
        print(f"queue recovery failed: {e}")
        return
    if not pending:
        return
    print(f"queue recovery: re-firing {len(pending)} incomplete queue(s)")
    for p in pending:
        qid = p["queue_id"]
        remaining = p["remaining_tasks"]
        completed_so_far = p["completed"]
        state = {
            "queue_id": qid,
            "tasks": remaining,
            "completed": 0,
            "current": 0,
            "status": "queued",
            "started_at": datetime.now().isoformat(),
            "result_paths": [],
            "recovered_from_disk": True,
            "completed_before_recovery": completed_so_far,
            "idx_offset": completed_so_far,  # P1: write global indices in lifecycle log
        }
        with _queue_lock:
            _queue_state[qid] = state
        threading.Thread(target=_process_queue, args=(qid,), daemon=True).start()
        print(f"  resumed {qid}: {len(remaining)} tasks remaining")


if __name__ == "__main__":
    boot()
    _resume_incomplete_queues()
    # Suppress Flask request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
