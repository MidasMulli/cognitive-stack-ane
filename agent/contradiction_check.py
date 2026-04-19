"""Proactive Contradiction Surfacing.

Scans user messages for entities, checks those entities against correction
records in the entity index, and returns matching corrections for injection
into the system prompt alongside normal retrieval results.

Pure Python, no LLM calls. Target <20ms per check.
"""
import json
import os
import re
import sys
from pathlib import Path

# ── Import entity extraction from tools/build_entity_index.py ──────────────

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent.parent / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from build_entity_index import extract_entities

# ── Paths ──────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_ENTITY_INDEX_PATH = _ROOT / "data" / "entity_index.json"

# ── Cached index (loaded once) ─────────────────────────────────────────────

_entity_index: dict | None = None
_corrections_by_entity: dict | None = None  # entity -> list of correction records


def _load_index():
    """Load entity index and pre-filter to correction records only."""
    global _entity_index, _corrections_by_entity
    if _corrections_by_entity is not None:
        return

    if not _ENTITY_INDEX_PATH.exists():
        _entity_index = {}
        _corrections_by_entity = {}
        return

    try:
        _entity_index = json.loads(_ENTITY_INDEX_PATH.read_text())
    except Exception:
        _entity_index = {}
        _corrections_by_entity = {}
        return

    _corrections_by_entity = {}
    for entity, records in _entity_index.items():
        corrections = [r for r in records if r.get("type") == "correction"]
        if corrections:
            _corrections_by_entity[entity] = corrections


def _content_suggests_precorrection(message_lower: str, snippet: str) -> bool:
    """Heuristic: does the user's message suggest the pre-correction state?

    Checks if the message contains language that the correction addresses.
    Simple approach: look for value/approach mentions in the correction that
    also appear (or are implied) in the user message.
    """
    snippet_lower = snippet.lower()

    # Extract key terms from the correction snippet (skip very common words)
    # Look for things like "not", "wrong", "actually", "dead", numeric values
    # that indicate what was corrected
    correction_indicators = [
        # If correction says "X is dead/wrong/not Y" and user mentions X positively
        (r"\bdead\b", "positive_mention"),
        (r"\bwrong\b", "positive_mention"),
        (r"\bdoesn'?t work\b", "positive_mention"),
        (r"\bnot\s+\w+", "negation"),
        (r"\bpenalty\b", "positive_mention"),
        (r"\bslower\b", "positive_mention"),
        (r"\bworse\b", "positive_mention"),
        (r"\bkilled?\b", "positive_mention"),
        (r"\brefuted?\b", "positive_mention"),
    ]

    # If the correction snippet contains negative language about something
    # the user is mentioning positively, that's a contradiction signal.
    has_negative_in_correction = any(
        re.search(pat, snippet_lower) for pat, _ in correction_indicators
    )

    # For corrections, the entity match itself is usually sufficient signal
    # since corrections are inherently about something being wrong.
    # But we boost confidence if the user's language is positive/suggestive.
    positive_user_patterns = [
        r"\bworks?\s+well\b",
        r"\bshould\s+(try|use)\b",
        r"\blet'?s\s+(try|use)\b",
        r"\bgood\s+for\b",
        r"\bcan\s+we\b",
        r"\bwe\s+should\b",
        r"\bwant\s+to\b",
    ]
    user_is_positive = any(
        re.search(pat, message_lower) for pat in positive_user_patterns
    )

    # Return true if: correction has negative language AND user is being
    # positive/suggestive, OR just if the entity match is strong enough
    # (correction records are inherently about what's wrong)
    return has_negative_in_correction or user_is_positive


_MAX_CONTRADICTIONS = 3


def check_contradictions(message: str) -> list[str]:
    """Check a user message for potential contradictions against correction records.

    Returns the top 1-3 most relevant corrections, ranked by entity overlap
    density with the user's message. One sharp contradiction is more useful
    than fourteen vague ones.

    Target: <20ms per check.
    """
    if not message or not message.strip():
        return []

    _load_index()
    if not _corrections_by_entity:
        return []

    message_entities = extract_entities(message)
    if not message_entities:
        return []

    named_entities = {e for e in message_entities if not e.startswith("measurement:")}
    if not named_entities:
        return []

    message_lower = message.lower()
    message_words = set(w.lower().strip("\"'?.,!") for w in message.split() if len(w) >= 2)

    # Collect all candidate corrections with relevance scores
    candidates = []  # (score, alert_str)
    seen_snippets = set()

    for entity in named_entities:
        corrections = _corrections_by_entity.get(entity)
        if not corrections:
            continue

        for rec in corrections:
            snippet = rec.get("content_snippet", "")
            if not snippet or snippet in seen_snippets:
                continue

            if not _content_suggests_precorrection(message_lower, snippet):
                continue

            seen_snippets.add(snippet)
            session = rec.get("session_label", "unknown")

            # Score by relevance to the user's actual message:
            # 1. Entity overlap: how many of the user's entities appear in this correction
            rec_entities = set()
            for e2 in _corrections_by_entity:
                for r2 in _corrections_by_entity[e2]:
                    if r2.get("content_snippet") == snippet:
                        rec_entities.add(e2)
            entity_overlap = len(named_entities & rec_entities)

            # 2. Word overlap: how many of the user's words appear in the snippet
            snippet_words = set(w.lower().strip("\"'?.,!") for w in snippet.split() if len(w) >= 2)
            word_overlap = len(message_words & snippet_words)

            # 3. Specificity: shorter, punchier corrections rank higher
            specificity = 1.0 / max(len(snippet), 10)

            score = entity_overlap * 3.0 + word_overlap * 1.0 + specificity * 100

            alert = (
                f"[CORRECTION from {session}] "
                f"Re: {entity} -- {snippet}"
            )
            candidates.append((score, alert))

    if not candidates:
        return []

    # Rank by score, return top N
    candidates.sort(key=lambda x: -x[0])
    return [alert for _, alert in candidates[:_MAX_CONTRADICTIONS]]


# ── Quick self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    tests = [
        ("the 0.8B drafter works well for spec decode",
         "should surface 0.8B drafter dead path"),
        ("we should try Q4 on the ANE",
         "should surface Q4 dequant penalty"),
        ("hello how are you",
         "should return empty"),
    ]

    for msg, expectation in tests:
        t0 = time.monotonic()
        results = check_contradictions(msg)
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"\n--- {expectation} ---")
        print(f"  Query: {msg!r}")
        print(f"  Time: {elapsed_ms:.1f}ms")
        if results:
            for r in results:
                print(f"  -> {r}")
        else:
            print("  -> (no contradictions)")
