"""Absence guard — prevents hallucination when all retrieval stores are empty.

Called as a LAST RESORT after every retrieval path has been tried (narrative,
registry, enumeration, recall, discovery). If all returned empty AND the query
is domain-relevant, injects a guard message that instructs the 72B not to
fabricate answers.

If the query is NOT domain-relevant (weather, general knowledge, etc.), returns
None and lets the model answer freely.

Retrieval shape #7 of 7 — the safety net.

M123 A3 — Definitional-query suppression
========================================

Per M123 directive §3.3, the absence gate was over-applied on definitional
queries in M122 C pilot (T47 "what is information theory?", T62 per verdict).
The model treated definitional queries as research-scope absence cases and
abstained when the answer was parametric knowledge.

``is_definitional_query()`` classifies a query as a generic / factual-concept
definitional request. When it returns True, the caller MAY suppress the
HARD absence gate (skip the absence response; let the model generate from
parametric knowledge). Tier 1 FAB scrub and Tier 2 binding scrub still run.

The classifier is deliberately UNDER-FIT (K7-safe):
  - Positive patterns anchored at query start only.
  - Negative patterns disqualify any possessive / project-state framing.
  - Project-specific lexicon (ANE, SharedEvents, Tier 2, etc.) disqualifies
    even when a positive pattern matches — project-specific definitional
    still needs grounded retrieval.

Under-fit is the rule: better to miss 2-3 legitimate definitional queries
than to falsely suppress the absence gate on project-state queries.

M115 β — three-sub-gate split
==============================

Per vault/agent_reports/m114_a3_beta_data_harvest.md §9 authoritative spec:

- **Sub-Gate 1 — Empty-pool unconditional fire**
  `pool_size == 0 AND narrative_used == False` → FIRE unconditionally.
  Drops the `_is_domain_relevant(query)` check entirely from this branch.
  Rationale: 6 reproducible FN confabulations in pilot
  (T6/T14/T15/T24/T26/T27) — Gate B silencing on non-domain-keyword queries
  was the load-bearing FN root cause.

- **Sub-Gate 2 — Low-score word-overlap fire (PRESERVED)**
  `pool_size > 0 AND max_score < 0.5 AND word_overlap_branch_eligible AND
  unmatched_ratio > 0.5` → FIRE. Unchanged. T16/T18 in pilot.

- **Sub-Gate 3 — High-score Path 2 FP fix**
  `pool_size > 0 AND max_score >= 0.5 AND word_overlap_branch_eligible AND
  unmatched_ratio > 0.5` → DO NOT FIRE. Fix: caller passes
  `recall_results=filtered` (the actual recall pool) NOT `[]`, which restores
  the score context that was being stripped. 3 reproducible FPs fixed
  (T9/T11/T37 at scores 1.178/2.079/1.359 with 8/8/5-row pools).
"""

import re
import sys
from pathlib import Path

# Ensure discovery_retrieval is importable (same directory)
_AGENT_DIR = str(Path(__file__).resolve().parent)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from discovery_retrieval import _is_domain_relevant, try_discovery  # noqa: F401

_GUARD_MESSAGE = (
    "=== CRITICAL: KNOWLEDGE GAP ===\n"
    "The memory store has NO information about this topic. "
    "Zero relevant memories were found. "
    "You have NO factual basis to answer this question.\n\n"
    "MANDATORY: You MUST respond with a short statement like: "
    "\"I don't have any information about that in my memory store.\" "
    "Do NOT guess. Do NOT use your training data. Do NOT invent "
    "file names, numbers, measurements, or code paths. "
    "Any specific claim you make WILL be fabricated because "
    "you have no source material.\n"
    "=== END KNOWLEDGE GAP ==="
)

# Score threshold for Sub-Gate 3: a recall_results entry at/above this is
# treated as sufficient evidence that the pool substantively addresses the
# query, suppressing a Path-2 high-score + word-overlap-mismatch FP.
# Per A3 §9, 0.5 is the authoritative cutoff (same number used elsewhere
# as _absence_threshold default for "normal" sensitivity).
_SUB_GATE_3_SCORE_FLOOR = 0.5  # m115_beta_sub_gate_3


def check_absence(
    query: str,
    recall_results: list,
    narrative_result: dict | None,
) -> str | None:
    """Check whether the absence guard should fire.

    Args:
        query: The user's message.
        recall_results: List of recall results (may be empty).
            NOTE: Callers that previously passed `[]` to strip score context
            should pass the real `filtered` pool here so Sub-Gate 3 can
            evaluate score floors. Passing `[]` now means "there was no
            recall pool" — Sub-Gate 1 territory.
        narrative_result: Result from narrative retrieval (may be None).

    Returns:
        Guard message string if the guard fires, None otherwise.
    """
    # Narrative preempt — if narrative retrieval found something, no guard.
    # (Note: pilot turn T1 showed narrative can preempt even when the
    # narrative result doesn't fully answer. Flagged M114 §8 as separate
    # "narrative preemption silent hole" — NOT in β scope.)
    if narrative_result is not None:
        return None

    pool_size = len(recall_results) if recall_results else 0

    # m115_beta_sub_gate_1: Empty-pool unconditional fire.
    # pool_size == 0 AND narrative_used == False → FIRE unconditionally.
    # Domain-relevance check is intentionally DROPPED here (A3 §9).
    # 6 FN evidence: T6/T14/T15/T24/T26/T27 all matched this predicate in
    # pilot and Gate B silencing let the model confabulate.
    if pool_size == 0:
        return _GUARD_MESSAGE

    # m115_beta_sub_gate_3: High-score fix.
    # If the caller passed a non-empty recall_results pool and any row
    # scores at or above the score floor, the pool substantively addresses
    # the query — DO NOT fire. This is the fix for T9/T11/T37 where
    # midas_ui previously stripped score context by passing `[]` to
    # check_absence from the word-overlap branch.
    high_quality = [
        r for r in recall_results
        if (r.get("score", 0) if isinstance(r, dict) else 0)
        >= _SUB_GATE_3_SCORE_FLOOR
    ]
    if high_quality:
        return None

    # m115_beta_sub_gate_2 (preserved): pool has rows, but max_score < 0.5.
    # Fall through to legacy domain-relevance check, which is the correct
    # firing path for the low-score + word-overlap-mismatch case
    # (T16/T18 in pilot). Domain-relevance is preserved here because
    # low-score content IS tangential; filtering by domain keywords keeps
    # the guard from over-firing on genuinely out-of-scope queries
    # (weather, general knowledge without domain terms).
    relevant, _matched_terms = _is_domain_relevant(query)
    if not relevant:
        return None

    return _GUARD_MESSAGE


# ──────────────────────────────────────────────────────────────────────
# M123 A3 — Definitional-query classifier
# ──────────────────────────────────────────────────────────────────────

# Positive patterns — anchored at query start. Generic definitional
# request shapes that call for parametric / encyclopedic knowledge.
# Under-fit intentionally: "what is" + a bare noun, "define X", etc.
# Patterns are compiled case-insensitive, re.IGNORECASE at use time.
#
# M125 A5.3 extension: M123 A3 shipped with ZERO production fires in
# M123 C pilot. The positive list was too narrow — organic
# definitional-framed turns used shapes like "why does X work", "how
# does X compare to Y", "what are the tradeoffs of X",
# lowercased/informal "whats ane architecture", etc. Extended here with
# K16 guard: project-state queries (possessive markers, M-session
# markers, project lexicon) MUST NOT false-positive. Re-run the M123 A3
# 21/21 battery to verify.
_M123_A3_POSITIVE_PATTERNS: tuple[str, ...] = (
    r"^\s*what\s+is\s+an?\b",           # "what is a", "what is an"
    r"^\s*what's\s+an?\b",               # "what's a", "what's an"
    # "whats a/an" — informal (no apostrophe). Standalone informal shapes
    # in M125 A5.3 extension.
    r"^\s*whats\s+an?\b",
    r"^\s*what\s+is\s+(?!our\b|my\b|the\s+status\b|the\s+current\b)",  # "what is <topic>"
    r"^\s*what's\s+(?!our\b|my\b|the\s+status\b|the\s+current\b)",      # "what's <topic>"
    r"^\s*whats\s+(?!our\b|my\b|the\s+status\b|the\s+current\b)",        # informal
    r"^\s*define\b",                     # "define information theory"
    r"^\s*explain\s+the\s+concept\s+of\b",
    r"^\s*describe\s+(?:the\s+)?(?:concept|theory|idea|principle)\s+of\b",
    r"^\s*who\s+was\b",                  # "who was Shannon"
    r"^\s*who\s+is\b(?!\s+(?:our|my)\b)",
    # M125 A5.3: definitional-explanatory shape.
    # "why does X work" / "why is X used" — encyclopedic causal question
    # about a generic concept. Disqualified by negative patterns +
    # project-lexicon if X is project-specific.
    r"^\s*why\s+(?:does|do|is|are|would)\s+"
    r"(?!(?:we|our|you|i|my|the\s+(?:status|current))\b)",
    # M125 A5.3: comparative-definitional.
    # "how does X compare to Y", "how is X different from Y",
    # "what is the difference between X and Y".
    r"^\s*how\s+(?:does|do|is|are|were)\s+(?!our\b|we\b|my\b|you\b|the\b)"
    r"[\w\s\-]{1,60}\s+"
    r"(?:compare|differ|relate|work|function|fit)\s+",
    r"^\s*what\s+is\s+the\s+difference\s+between\b"
    r"(?!\s+our\b|\s+my\b)",
    r"^\s*what'?s\s+the\s+difference\s+between\b"
    r"(?!\s+our\b|\s+my\b)",
    # M125 A5.3: definitional-analytical.
    # "what are the tradeoffs of X", "what are the benefits of X",
    # "what are the advantages of X", "what are the pros and cons of X".
    r"^\s*what\s+(?:are|is)\s+the\s+"
    r"(?:tradeoffs?|trade-?offs?|benefits?|advantages?|disadvantages?|"
    r"pros\s+and\s+cons|limitations?|drawbacks?|applications?|uses?)\s+"
    r"of\b(?!\s+our\b|\s+my\b)",
    # M125 A5.3: "how do you" + generic action verb (pedagogical framing).
    # Disqualified when possessive / project-state markers intercept.
    # e.g. "how do you implement backpropagation", "how do you train a
    # transformer". Guard: project lexicon still vetoes.
    r"^\s*how\s+(?:do\s+you|can\s+you|would\s+you|should\s+you)\s+"
    r"(?:use|implement|train|build|compute|calculate|derive|understand|"
    r"explain|describe|define|model)\b",
)

# Negative patterns — disqualify the classifier even if a positive matched.
# These catch possessive / project-state framings that LOOK definitional but
# actually require grounded retrieval.
_M123_A3_NEGATIVE_PATTERNS: tuple[str, ...] = (
    r"^\s*what\s+is\s+our\b",            # "what is our strict pass rate"
    r"^\s*what's\s+our\b",
    r"^\s*what\s+is\s+my\b",
    r"^\s*what's\s+my\b",
    r"^\s*what\s+is\s+the\s+status\b",   # "what is the status of X"
    r"^\s*what's\s+the\s+status\b",
    r"^\s*what\s+is\s+the\s+current\b",  # "what is the current X"
    r"^\s*what's\s+the\s+current\b",
    r"^\s*what\s+did\s+(?:we|you|i)\b",  # "what did we measure"
    r"^\s*what\s+have\s+(?:we|you|i)\b",
    r"^\s*when\s+did\s+(?:we|you|i)\b",
    r"^\s*where\s+did\s+(?:we|you|i)\b",
    r"^\s*which\s+",                     # "which Apple Silicon generation" — T62 shape
    r"^\s*how\s+(?:does|did|is|was|are|were)\s+(?:our|we|my|the|you)\b",
)

# Project-specific lexicon — when present anywhere in the query, the
# classifier does NOT fire even if a positive pattern matched. K7
# discipline: "Define Tier 2 scrub" matches `^define` positively but
# contains "tier 2 scrub" which is project-specific, so it still routes
# to grounded retrieval. Encyclopedic definitional queries
# ("define information theory") contain no project-specific tokens.
_M123_A3_PROJECT_LEXICON: tuple[str, ...] = (
    # Hardware / silicon
    "ane", "amx", "sme", "slc", "agx", "nax", "mcc", "dcs", "amcc",
    "coreml", "core ml", "iosurface", "iokit", "kext",
    "apple silicon", "m5 pro", "m1", "m2", "m3", "m4", "m5",
    "tb5", "pcore", "p-core", "ecore", "e-core",
    # Our projects / artifacts
    "subconscious", "midas", "orion-ane", "ane-compiler", "ane-dispatch",
    "ngram-engine", "four-path", "locomo", "neuron 80m",
    "sharedevents", "shared events", "shared-event",
    "gemma", "qwen", "llama", "eagle-3", "pard",
    # Session / methodology vocabulary (internal)
    "tier 1", "tier 2", "tier 3", "tier1", "tier2", "tier3",
    "tier-1", "tier-2", "tier-3",
    "answer scrub", "answer_scrub", "absence gate", "absence_gate",
    "briefing", "recall pool", "canonical boost",
    "spec decode", "speculative decode", "drafter",
    "fab scrub", "tier 2 scrub", "tier 2 binding",
    "confab guard", "confabulation guard",
    # Stream / directive / session identifiers
    "directive", "stream a1", "stream a2", "stream a3", "stream a4",
    "stream a5", "stream b", "stream c",
)

# M-session regex — matches M1..M999 anywhere in query (e.g. "at M122",
# "M108-M117"). Treated as project-specific marker.
_M123_A3_M_SESSION_REGEX = re.compile(
    r"\bM[0-9]{1,3}(?:[a-z])?\b", re.IGNORECASE
)


def is_definitional_query(query: str) -> tuple[bool, dict]:
    """Classify whether `query` is a generic definitional request.

    Returns (fired, diagnostic) where:
      fired:      True iff the classifier decides this is a definitional
                  query for which the absence gate should be suppressed.
      diagnostic: dict of matched-pattern / negative-match / lexicon-hit
                  metadata for ζ v2.2 logging.

    UNDER-FIT by design (M123 A3, K7-safe):
      - Positive must match AT QUERY START.
      - Negative patterns disqualify (possessive/project-state framings).
      - Project lexicon hit anywhere disqualifies (project-specific
        definitional still needs grounded retrieval).
    """
    diagnostic = {
        "positive_match": None,
        "negative_match": None,
        "project_lexicon_hit": None,
        "m_session_hit": False,
        "decision_reason": None,
    }

    if not query or not isinstance(query, str):
        diagnostic["decision_reason"] = "empty_or_non_string"
        return (False, diagnostic)

    q = query.strip()

    # Negative gate first — cheapest rejection.
    for neg in _M123_A3_NEGATIVE_PATTERNS:
        if re.search(neg, q, re.IGNORECASE):
            diagnostic["negative_match"] = neg
            diagnostic["decision_reason"] = "negative_pattern_match"
            return (False, diagnostic)

    # Positive gate.
    matched_pos = None
    for pos in _M123_A3_POSITIVE_PATTERNS:
        if re.search(pos, q, re.IGNORECASE):
            matched_pos = pos
            break
    if matched_pos is None:
        diagnostic["decision_reason"] = "no_positive_pattern"
        return (False, diagnostic)
    diagnostic["positive_match"] = matched_pos

    # Project-lexicon gate — project-specific definitional routes to
    # grounded retrieval (K7 discipline).
    q_lower = q.lower()
    for term in _M123_A3_PROJECT_LEXICON:
        # Use word-boundary-ish check: term substring with simple
        # surrounding-char test to avoid "sharedeventspath" false-match.
        if term in q_lower:
            # Reject only if the term is token-like (not merely a
            # substring of a longer word). Cheap check: surrounded by
            # non-alnum on both sides, or at string boundary.
            idx = q_lower.find(term)
            left_ok = (idx == 0) or (not q_lower[idx - 1].isalnum())
            right_end = idx + len(term)
            right_ok = (right_end >= len(q_lower)) or (
                not q_lower[right_end].isalnum())
            if left_ok and right_ok:
                diagnostic["project_lexicon_hit"] = term
                diagnostic["decision_reason"] = "project_lexicon_hit"
                return (False, diagnostic)

    # M-session marker check.
    if _M123_A3_M_SESSION_REGEX.search(q):
        diagnostic["m_session_hit"] = True
        diagnostic["decision_reason"] = "m_session_marker"
        return (False, diagnostic)

    # All gates cleared: classifier fires.
    diagnostic["decision_reason"] = "definitional_fired"
    return (True, diagnostic)
