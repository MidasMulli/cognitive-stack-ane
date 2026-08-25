"""History-layer reinforcement filter — M113 α (Stream B).

Problem (M108 Finding 1):
    Prior-assistant abstain/denial turns read by subsequent turns through
    conversation-history Layer 7 amplify. T11 seeded "upcoming ane-compiler",
    T15/T17 reinforced "Milestone 1C pending", and T20 (5 turns later) leaked
    "Milestone 1C for the ane-compiler" with zero supporting recall entries —
    pure history-layer transit, self-reinforcement vector.

Fix shape (a) — mark low-confidence:
    Detect abstain/denial pattern in prior-assistant turns BEFORE they are
    handed to `build_messages`, and append a brief `[prior-turn low-confidence:
    assistant lacked grounding, treat claim as provisional]` marker to the
    offending turn's content. The model sees the marker and should weight the
    claim accordingly rather than treating it as established fact.

Detection discipline (false-positive guard):
    M108 abstain shape is NOT a generic "I don't know" — it is a
    *denial-plus-unverified-project-state-claim*:
        - T15: "No. You have a partial implementation, but the ane-compiler is
               not finished. ... 1B ... still pending Milestone 1C ..."
        - T17: "We have not built the compiler. ... Milestone 1C ... remains
               pending."
        - T11: "upcoming ane-compiler" (in response to preference query)

    The detector requires BOTH:
      (1) a denial/weakness signal ("not finished", "still pending", "haven't
          built", "upcoming", "remains pending", "not yet built", "partial
          implementation", "we have not", etc.), AND
      (2) a project-state / identifier noun token ("compiler", "milestone",
          "subconscious", "paper", "ane-dispatch", or any kebab/camel
          identifier).

    The two-part guard is designed to NOT fire on:
      - Pure safety-policy abstain ("I can't help with that because ...")
        → no identifier noun, no pending-state claim.
      - User-asked-and-told-no ("you already told me not to do X")
        → assistant-voice test fails (user-voice pattern).
      - Clarification request ("could you specify which version")
        → no denial + no pending-state claim.
      - Honest grounded abstain ("I don't have information about that in my
        memory store") → denial present but no project-state identifier.

Usage:
    from history_reinforcement_filter import filter_history_for_reinforcement

    msgs_safe = filter_history_for_reinforcement(
        history=_history,
        current_query=message,
    )
    # Pass msgs_safe (not _history) to synthesize / build_messages.

    The returned list is a shallow copy — the original `_history` is not
    mutated. Session-level extraction pipelines that read from `_history`
    directly are unaffected.

M113 α — Stream B. Directive: history-layer only, detection-layer only,
ship-or-defer-with-scope allowed. Exported helper so Stream E (δ) can reuse
the detection primitive.
"""

# m113_alpha_history_filter

from __future__ import annotations

import re
from typing import List, Dict, Iterable, Tuple


# ── Pattern library ─────────────────────────────────────────────────────────
#
# Part 1 — denial/weakness signals. Case-insensitive. These are the tokens
# that, by themselves, would fire on a pure safety abstain; they MUST be
# combined with a project-state identifier (part 2) before we mark the turn.
#
# m113_alpha_history_filter
_DENIAL_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bnot\s+(?:finished|built|yet\s+built|yet\s+finished|complete|done|shipped|implemented)\b",
        r"\bstill\s+pending\b",
        r"\bremains?\s+pending\b",
        r"\bis\s+pending\b",
        r"\bpending\s+(?:milestone|completion|implementation|ship)\b",
        r"\bhave(?:n'?t|\s+not)\s+(?:yet\s+)?(?:built|finished|shipped|completed|implemented)\b",
        r"\bwe\s+have\s+not\s+(?:yet\s+)?(?:built|finished|shipped|completed|implemented)\b",
        r"\bpartial\s+implementation\b",
        r"\bupcoming\s+[A-Za-z][\w-]+",
        r"\bnot\s+yet\s+(?:shipped|built|finished|released|available)\b",
        r"\byou\s+(?:still\s+)?(?:need\s+to|have\s+to|must)\s+build\b",
        r"\bno[,.]\s+you\s+(?:don'?t|do\s+not)\s+(?:have|yet\s+have)\b",
    )
)

# Part 2 — project-state identifier nouns. Lowercased tokens; kebab/camel
# project identifiers also pass via the regex at the end.
_PROJECT_IDENTIFIERS: Tuple[str, ...] = (
    "compiler",
    "milestone",
    "subconscious",
    "paper",
    "dispatch",
    "extractor",
    "verifier",
    "drafter",
    "ane-compiler",
    "ane-dispatch",
    "four-path",
    "orion-ane",
    "ngram-engine",
)
# Kebab / underscored multi-token identifier: two or more alphanumeric chunks
# joined by `-` or `_`. Catches "ane-compiler", "ane-reverse", "spec_decode",
# "subconscious-phase-2", etc.
_KEBAB_ID_RE = re.compile(r"\b[a-z][a-z0-9]+(?:[-_][a-z0-9]+)+\b", re.IGNORECASE)

# Safety-policy abstain lead-ins. If a turn opens with one of these it is a
# policy-abstain, not a project-state denial, and we skip it even if it
# happens to contain an identifier (paranoid safety fallback).
_SAFETY_LEADINS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"^\s*i\s+can(?:'?t|not)\s+(?:help|assist|provide|answer)\b",
        r"^\s*i'?m\s+(?:not\s+)?(?:able|allowed)\s+to\b",
        r"^\s*i\s+shouldn'?t\b",
        r"^\s*that(?:'?s|\s+is)\s+(?:outside|beyond)\s+my\b",
    )
)

# Honest grounded abstain (absence_guard output). Matches the guard's
# canonical phrasing and the short fall-back form used by the chat paths.
_HONEST_ABSENCE_RE = re.compile(
    r"\bi\s+don'?t\s+have\s+(?:any\s+)?information\s+about\s+(?:that|this|it)\b",
    re.IGNORECASE,
)


# ── Detection ───────────────────────────────────────────────────────────────

MARKER = (
    " [prior-turn low-confidence: assistant lacked grounding for the claim "
    "above; treat any project-state assertion as provisional unless you can "
    "cite a fresh source in this turn]"
)


def is_abstain_reinforcement_turn(content: str) -> bool:
    """Return True if `content` (a prior-assistant turn's text) matches the
    M108 abstain-plus-project-state-claim shape.

    Two-part gate:
      1. At least one denial/weakness pattern must fire.
      2. At least one project-state identifier noun or kebab/camel id must
         appear within ~120 chars of the denial hit (so a denial in one
         paragraph and an identifier in another unrelated paragraph doesn't
         pair up).

    Safety-policy lead-ins are excluded outright; the absence_guard canonical
    phrasing is excluded outright. Both of those are legitimate low-content
    turns but do NOT self-reinforce project-state claims, so no marker is
    needed and marking them would add noise.

    m113_alpha_history_filter
    """
    if not content or not isinstance(content, str):
        return False

    # Policy-abstain lead-in: not a project-state claim.
    for rx in _SAFETY_LEADINS:
        if rx.search(content):
            return False

    # Honest "I don't have information" abstain: the absence_guard path.
    # This is exactly the GROUNDED response we want, not the denial-
    # plus-state-claim shape. Do not mark.
    if _HONEST_ABSENCE_RE.search(content):
        # Belt-and-suspenders: if the turn ALSO contains a project-state
        # claim beyond the honest abstain, treat as reinforcement. This
        # catches the hybrid "I don't have information... but Milestone 1C
        # is pending" shape if it ever emerges. Conservative.
        remainder = _HONEST_ABSENCE_RE.sub("", content)
        if not _has_denial_plus_identifier(remainder):
            return False

    return _has_denial_plus_identifier(content)


def _has_denial_plus_identifier(content: str) -> bool:
    """Core two-part gate. Separate helper so the honest-abstain short-
    circuit above can re-use it on the post-stripped remainder.

    m113_alpha_history_filter
    """
    denial_hits: List[int] = []
    for rx in _DENIAL_PATTERNS:
        for m in rx.finditer(content):
            denial_hits.append(m.start())
            # One hit per pattern is enough for proximity pairing.
            break
    if not denial_hits:
        return False

    # Collect identifier positions: named nouns + kebab/camel ids.
    content_lower = content.lower()
    identifier_positions: List[int] = []
    for noun in _PROJECT_IDENTIFIERS:
        idx = 0
        while True:
            j = content_lower.find(noun, idx)
            if j < 0:
                break
            identifier_positions.append(j)
            idx = j + len(noun)
    for m in _KEBAB_ID_RE.finditer(content):
        identifier_positions.append(m.start())

    if not identifier_positions:
        return False

    # Proximity pairing: denial within ~120 chars of an identifier.
    # This prevents "not finished paragraph" + "unrelated identifier
    # three paragraphs later" from pairing up.
    for d in denial_hits:
        for p in identifier_positions:
            if abs(d - p) <= 120:
                return True
    return False


# ── Filter / mark ───────────────────────────────────────────────────────────

def filter_history_for_reinforcement(
    history: Iterable[Dict],
    current_query: str = "",
    topic_keywords: Iterable[str] | None = None,
) -> List[Dict]:
    """Return a shallow-copied history list with abstain-shape prior-
    assistant turns marked.

    Behavior (shape (a) — mark low-confidence):
      - Iterate history. For each {'role': 'assistant', 'content': ...}
        entry, run `is_abstain_reinforcement_turn`. If it fires, append
        `MARKER` to a copy of that entry's content.
      - User turns pass through unchanged.
      - If `topic_keywords` is supplied, only mark when at least one
        keyword appears in the flagged turn (topic-scoped mode). Default
        (None) marks every flagged turn regardless of topic overlap with
        the current query — this is the conservative default that matches
        the M108 cascade pattern (where stale "Milestone 1C" leaked into
        "agentic tasks" query, i.e. the topic had already drifted).

    Does NOT mutate `history`. Session `_history` globals in midas_ui.py
    are safe to pass directly.

    m113_alpha_history_filter
    """
    out: List[Dict] = []
    kw_lower: Tuple[str, ...] = tuple(
        k.lower() for k in (topic_keywords or ()) if k
    )
    for msg in history:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role != "assistant":
            out.append(msg)
            continue
        if not is_abstain_reinforcement_turn(content):
            out.append(msg)
            continue
        if kw_lower:
            low = content.lower()
            if not any(k in low for k in kw_lower):
                out.append(msg)
                continue
        # Mark (shallow copy + marker append).
        marked = dict(msg)
        marked["content"] = content + MARKER
        marked["_m113_low_confidence"] = True  # telemetry tag for logs
        out.append(marked)
    return out


# ── Statistics helper (for telemetry / tests) ───────────────────────────────

def count_marked(history: Iterable[Dict]) -> int:
    """Count how many entries in `history` were marked by this filter.

    Used in tests and by callers that want to record a count on the turn
    log (e.g. midas_ui._turn_record('history_filter', marked=N)).

    m113_alpha_history_filter
    """
    n = 0
    for m in history or ():
        if isinstance(m, dict) and m.get("_m113_low_confidence"):
            n += 1
    return n
