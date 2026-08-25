"""M125 Stream A2 — shape-precedence arbitration / canonical-lookup classifier.

Authoritative spec:
    vault/directives/in_progress/2026-04-23T01-03-39_m125_m125-day1-architectural-commit-pool-gap.md §3.2
    vault/agent_reports/m124_a_pool_gap_diagnosis.md §Cause 4 (K5-refined)

Mechanism anchor: M124 Stream A K5-refined C4. Narrative synthesis
    (``midas_ui.py:3482-3508``) runs BEFORE default_recall. On narrative
    success, ``_narrative_used=True`` short-circuits the default_recall
    branch — canonical pool records never load even when they exist at
    healthy cosine. 5 pool-gap turns attributable (T22, T24, T59, T60, T61).

Fix shipped: pattern 3 (classifier + suppression).
    - If ``is_canonical_lookup(query)`` returns True, dispatch skips
      narrative synthesis and runs default_recall directly.
    - Otherwise the current narrative-primary dispatch is preserved.

UNDER-FIT by design (K4/K6 discipline):
    - Positive patterns are narrow: specific-value numeric, path-shape,
      quoted strings, explicit specificity modifiers, measurement-probe
      shape, "faster/slower than", "overhead/cost of".
    - Negative patterns (summary-eligible / continuation-eligible)
      disqualify FIRST — a query that looks narrative-eligible is never
      classified as canonical_lookup even if it also matches a positive
      pattern. This is the K6-safe ordering.
    - Project-state / possessive framings are handled by M123 A3
      is_definitional_query; A2 does not overlap.
    - M122 A4 is_a4_specific_shape_query is REUSED (not duplicated) for
      the numeric/path/quoted axis; A2 extends only where A4 does not
      cover (measurement-probe shape, overhead-of shape, faster-than
      shape).

Returns: (bool, dict) — (is_canonical_lookup, diagnostic).

    diagnostic fields:
      "positive_match":       matched pattern id (str) or None
      "negative_match":       matched summary/continuation pattern (str) or None
      "a4_specific_matched":  bool (M122 A4 reuse)
      "decision_reason":      short explanation string

m125_a2_shape_precedence
"""

from __future__ import annotations

import os
import re
import sys
from typing import Tuple, Dict

# Reuse M122 A4 specific-shape classifier — do not duplicate its regex.
_VAULT_SUBCONSCIOUS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "vault", "subconscious"))
if _VAULT_SUBCONSCIOUS not in sys.path:
    sys.path.insert(0, _VAULT_SUBCONSCIOUS)

try:
    from multi_path_retrieve import is_a4_specific_shape_query  # type: ignore
except Exception:  # pragma: no cover — import-time failure fallback
    def is_a4_specific_shape_query(_q: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# Negative patterns — summary-eligible / continuation-eligible / narrative-
# eligible queries MUST NOT be classified as canonical_lookup. These check
# FIRST so a query like "summarize what was the exact tok/s" stays narrative.
# Anchored to query-start (or whole-message) to avoid false-positive on
# embedded matches in long queries.
# ---------------------------------------------------------------------------
_A2_NEGATIVE_PATTERNS: Tuple[str, ...] = (
    # Summary-eligible
    r"^\s*summari[sz]e\b",
    r"^\s*(?:give|provide|write|produce)\s+(?:me\s+)?(?:a\s+)?(?:summary|overview|digest|recap|rundown)\b",
    r"\bconsolidate\b",
    r"\btie together\b",
    r"^\s*(?:tell|tell\s+me)\s+about\b",
    r"\bwhat\s+happened\b",
    r"^\s*walk\s+me\s+through\b",
    r"^\s*step\s+(?:me\s+)?through\b",
    r"^\s*take\s+me\s+through\b",
    # Continuation-eligible
    r"^\s*continue\s*$",
    r"^\s*keep\s+going\b",
    r"^\s*go\s+on\b",
    r"^\s*(?:and\s+)?(?:then\s+)?what(?:'s|\s+is)?\s+next\b",
    r"\bnext\s+steps?\b",
    # Anaphoric "what about X we discussed/mentioned/said"
    r"\bwhat\s+about\s+.*\b(?:we\s+discussed|we\s+mentioned|we\s+talked|we\s+said|earlier|yesterday|last\s+session)\b",
    # Narrative-arc framings
    r"\bcatch\s+me\s+up\b",
    r"\bstory\s+of\b",
    r"\boverall\s+picture\b",
)
_A2_NEGATIVE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _A2_NEGATIVE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Positive patterns — canonical-lookup specific-value shapes.
# Extends M122 A4 coverage where needed. Each pattern is narrow and
# anchored; prefer explicit markers over loose phrasing.
# ---------------------------------------------------------------------------
_A2_POSITIVE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # Measurement-probe shape: "did we measure X" / "did X measure Y"
    ("measurement_probe",
     r"\bdid\s+(?:we|you|i|they|the\s+test|the\s+benchmark)\s+(?:measure|test|benchmark|find|show|verify|confirm|observe|record)\b"),
    # "what did the X show" / "what did the test measure"
    ("test_show",
     r"\bwhat\s+did\s+the\s+(?:test|measurement|benchmark|probe|experiment|trial|run)\s+(?:show|say|find|produce|reveal|measure|record|report)\b"),
    # Introduced/show/reveal framings tied to a measurable concept
    ("introduced_overhead",
     r"\bintroduce[ds]?\s+(?:any\s+)?(?:overhead|penalty|delay|latency|cost|regression|slowdown|bottleneck)\b"),
    # Overhead-of / cost-of / bandwidth-of specific noun
    ("overhead_of",
     r"\b(?:what\s+(?:was|is|were)\s+)?the\s+(?:memory\s+)?(?:overhead|cost|penalty|delay|latency|cost|bandwidth)\s+of\b"),
    # Faster-than / slower-than / X vs Y comparisons with measurement shape
    ("comparison_measured",
     r"\b(?:faster|slower|cheaper|costlier|higher|lower|bigger|smaller|quicker)\s+than\b"),
    # Quantitative-rate shape ("rate of X", "throughput of X")
    ("rate_of",
     r"\b(?:rate|throughput|bandwidth|latency|percentage|ratio)\s+of\s+\w+\b"),
    # M125.3 Stream D — named-entity interrogative lookup shapes.
    # Narrative classifier (`narrative_retrieval._classify_intent`) defaults
    # any unmatched "what..." query to arc intent, misrouting canonical-
    # lookup asks like "what compression does ANE use" to narrative synthesis.
    # Q3 defect (M125.2 E §5). Negative-gate runs first so narrative-eligible
    # shapes ("tell me about", "what happened", "what about X we discussed")
    # are protected.
    # K18 discipline: narrow to interrogative + action verb + entity-object
    # surface. No bare "what is X" (that's A3 absence-gate's territory).
    ("named_entity_uses",
     r"\bwhat\s+\w+\s+(?:does|do|did)\s+(?:the\s+)?\S+\s+"
     r"(?:use|uses|used|employ|employs|employed|run|runs|ran|"
     r"adopt|adopts|adopted|require|requires|required|"
     r"support|supports|supported|have|has|had)\b"),
    # Possessive-attribute shape: "what is ANE's compression",
    # "what is Llama-8B's context length". Requires explicit possessive
    # marker ('s or s') — rules out generic "what is X".
    ("named_entity_possessive",
     r"\bwhat(?:\s+\w+)?\s+is\s+(?:the\s+)?\S+(?:'s|s')\s+\w+\b"),
)

_A2_POSITIVE_COMPILED = tuple(
    (pid, re.compile(p, re.IGNORECASE))
    for pid, p in _A2_POSITIVE_PATTERNS
)


def is_canonical_lookup(query: str) -> Tuple[bool, Dict]:
    """Classify whether ``query`` is a canonical-lookup specific-value ask.

    Returns (fired, diagnostic).

    Ordering (K6-safe):
      1. Empty / non-string → False.
      2. Negative gate (summary / continuation / narrative) → False.
         Even a positive match cannot overcome a negative match.
      3. M122 A4 specific-shape reuse → True if fires.
      4. A2-local positive patterns → True if fires.
      5. Fall through → False (prefer under-fit).
    """
    diagnostic: Dict = {
        "positive_match": None,
        "negative_match": None,
        "a4_specific_matched": False,
        "decision_reason": None,
    }

    if not query or not isinstance(query, str):
        diagnostic["decision_reason"] = "empty_or_non_string"
        return (False, diagnostic)

    q = query.strip()
    if not q:
        diagnostic["decision_reason"] = "empty_after_strip"
        return (False, diagnostic)

    # --- Negative gate first (K6 discipline)
    neg_match = _A2_NEGATIVE_RE.search(q)
    if neg_match:
        diagnostic["negative_match"] = neg_match.group(0)
        diagnostic["decision_reason"] = "negative_pattern_match"
        return (False, diagnostic)

    # --- Reuse M122 A4 specific-shape classifier (numeric / path / quoted /
    # explicit-specific-value markers). Cheapest positive signal.
    if is_a4_specific_shape_query(q):
        diagnostic["a4_specific_matched"] = True
        diagnostic["positive_match"] = "a4_specific_shape"
        diagnostic["decision_reason"] = "a4_specific_shape_query"
        return (True, diagnostic)

    # --- A2-local positive patterns (measurement-probe / overhead-of /
    # comparison-measured shapes not covered by A4).
    for pid, regex in _A2_POSITIVE_COMPILED:
        m = regex.search(q)
        if m:
            diagnostic["positive_match"] = pid
            diagnostic["decision_reason"] = f"a2_positive_{pid}"
            return (True, diagnostic)

    diagnostic["decision_reason"] = "no_positive_pattern"
    return (False, diagnostic)


# Enum values for ζ v2.3 retrieval.dispatch_decision field.
DISPATCH_DECISION_NARRATIVE_PRIMARY = "narrative_primary"
DISPATCH_DECISION_RECALL_PRIMARY_CANONICAL_LOOKUP = "recall_primary_canonical_lookup"
DISPATCH_DECISION_NARRATIVE_SUPPRESSED_CLASSIFIER = "narrative_suppressed_classifier"
DISPATCH_DECISION_OTHER = "other"


if __name__ == "__main__":
    samples = [
        # C4 targets (must fire)
        "What's the exact tok/s on Llama-8B Q8 through the ANE pipeline?",
        "Did MIE introduce any DRAM-bandwidth overhead? What did the test show?",
        "what was the exact acceptance rate of EAGLE-3 on Q3 70B?",
        "what was the memory overhead of the MIE on DRAM bandwidth?",
        "did we measure the ANE to be faster than the GPU on small matmul?",
        # Narrative-eligible (must NOT fire)
        "summarize the last five sessions",
        "tell me about the subconscious architecture",
        "walk me through M115-M118 fix surface",
        "what happened in Main 46?",
        "continue",
        "keep going",
        "what about the NAX probe we discussed?",
    ]
    for s in samples:
        fired, diag = is_canonical_lookup(s)
        print(f"fired={fired}  reason={diag['decision_reason']:<40}  q={s[:70]}")
