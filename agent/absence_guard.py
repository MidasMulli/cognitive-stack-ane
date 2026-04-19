"""Absence guard — prevents hallucination when all retrieval stores are empty.

Called as a LAST RESORT after every retrieval path has been tried (narrative,
registry, enumeration, recall, discovery). If all returned empty AND the query
is domain-relevant, injects a guard message that instructs the 72B not to
fabricate answers.

If the query is NOT domain-relevant (weather, general knowledge, etc.), returns
None and lets the model answer freely.

Retrieval shape #7 of 7 — the safety net.
"""

import sys
from pathlib import Path

# Ensure discovery_retrieval is importable (same directory)
_AGENT_DIR = str(Path(__file__).resolve().parent)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from discovery_retrieval import _is_domain_relevant, try_discovery

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


def check_absence(
    query: str,
    recall_results: list,
    narrative_result: dict | None,
) -> str | None:
    """Check whether the absence guard should fire.

    Args:
        query: The user's message.
        recall_results: List of recall results (may be empty).
        narrative_result: Result from narrative retrieval (may be None).

    Returns:
        Guard message string if the guard fires, None otherwise.
    """
    # If narrative retrieval found something, no guard needed
    if narrative_result is not None:
        return None

    # If recall returned high-confidence results, no guard needed.
    # After provenance weighting, assistant-generated records score 0.5x.
    # Genuine human/vault hits score 0.7+. Tangential hits score 0.4-0.6.
    # Threshold: need at least 1 result above 0.70 to be confident.
    if recall_results:
        high_quality = [
            r for r in recall_results
            if (r.get("score", 0) if isinstance(r, dict) else 0) > 0.70
        ]
        if len(high_quality) >= 1:
            return None

    # All stores empty — check domain relevance
    relevant, matched_terms = _is_domain_relevant(query)

    if not relevant:
        # General question — let the model answer freely
        return None

    # Domain-relevant AND all stores empty → fire the guard
    return _GUARD_MESSAGE
