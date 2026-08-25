"""Tests for M125.3 Stream D — named-entity interrogative misroute fix.

Authoritative directive:
    vault/directives/in_progress/
      2026-04-23T21-28-40_m125_3open_m125-3-synthesis-grounding-phase-0-audit.md §3.4

Defect anchor:
    vault/agent_reports/m125_2_e_pilot_readiness.md §5 — Q3 smoke
    "what compression does the ANE use" routed through narrative-shape
    classifier (narrative_retrieval._classify_intent defaults to 'arc' on
    any unmatched 'what...' query) instead of canonical-lookup/recall.
    Atomic row present in pool, never consulted.

K17 audit result:
    M125 A2 canonical_lookup classifier did NOT cover "what X does Y use"
    shape. Verified by direct call to is_canonical_lookup("what compression
    does ANE use") before Stream D edit → returned (False, no_positive_pattern).
    Stream D extends A2 with two narrow positive patterns
    (named_entity_uses, named_entity_possessive). K17 did NOT fire.

Fix surface:
    orion-ane/agent/shape_precedence.py — added two positive patterns after
    existing M122 A4 + A2-local patterns, in K6-safe order (negative gate
    runs first, A4 reuse, then A2-local positives).

Q3 replay + 8 variant battery + 20-query narrative regression.

Run:
    ~/.mlx-env/bin/python orion-ane/tests/test_m125_3_d_named_entity.py

Registry values produced:
    m125_3.d.verdict                   : shipped / deferred
    m125_3.d.classifier_refined        : 1 / 0
    m125_3.d.q3_replay_pass            : 1 / 0
    m125_3.d.variant_pass_count        : int
    m125_3.d.m125_a2_predicate_covers  : 1 / 0  (pre-Stream-D audit)

m125_3_d_named_entity
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
_VAULT_SUBC = os.path.join(_REPO_ROOT, "vault", "subconscious")
for _p in (_AGENT_DIR, _VAULT_SUBC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shape_precedence import is_canonical_lookup  # noqa: E402


# ------------------------------------------------------------------
# Q3 DEFECT — the exact smoke-turn query from M125.2 E pilot readiness.
# ------------------------------------------------------------------
Q3_QUERY = "what compression does the ANE use"


# ------------------------------------------------------------------
# 8 interrogative + named-entity variants. Each MUST fire after
# Stream D extension. K17 audit shows all returned False pre-Stream-D.
# Shapes covered:
#   - "what X does Y use"       (named_entity_uses)
#   - "what X does Y employ"    (named_entity_uses, employ verb)
#   - "what X does Y support"   (named_entity_uses, support verb)
#   - "what is Y's X"           (named_entity_possessive)
# ------------------------------------------------------------------
Q3_VARIANTS = [
    # --- named_entity_uses pattern family (6)
    "what compression does ANE use",                    # Q3 bare
    "what quantization does Llama-8B use",
    "what method does the extractor employ",
    "what framework does Midas use",
    "what precision does the ANE use",
    "what encoding does the model use",
    # --- named_entity_possessive pattern family (2)
    "what is ANE's compression",
    "what is Llama-8B's context length",
]


# ------------------------------------------------------------------
# NARRATIVE REGRESSION — 20 narrative-eligible queries. All MUST
# stay narrative (classifier returns False). Shared battery with
# M125 A2 test; Stream D must not regress any of these.
# ------------------------------------------------------------------
NARRATIVE_REGRESSION = [
    "summarize the last five sessions",
    "summarize M124 findings",
    "tell me about the subconscious architecture",
    "tell me about the ANE reverse engineering work",
    "walk me through M115-M118 fix surface",
    "step me through the spec-decode pipeline",
    "take me through what we shipped this week",
    "what happened in Main 46?",
    "what happened yesterday?",
    "continue",
    "keep going",
    "go on",
    "what's next?",
    "what about the NAX probe we discussed?",
    "what about the dead path we talked about earlier?",
    "catch me up on last week",
    "give me a summary of M120",
    "give me an overview of paper 3",
    "provide a recap of the last session",
    "tell me the story of the exclave wall",
]


# ------------------------------------------------------------------
# M125.2 E OTHER SMOKE ARCHETYPES — representative queries from
# pilot readiness report §3 that should retain their current routing
# behavior after Stream D ships.
#   Q1  : "How does Subconscious memory work?" — narrative/arc target
#   Q4  : "When did Rule 1 get filed?" — recall, classifier-neutral
#   Q5  : "Walk me through the fix surface from M115 through M118"
#         → narrative (MUST stay narrative)
#   Q6  : "What happened two sessions before last?" — recency shape
#   Q7  : "What did I ask 2 turns ago?" — turn-recency shape
#   Q8  : "Was the M42 1/20 battery caused by stop tokens or stream ordering?"
#         → A4 specific-shape → canonical-lookup fires (pre-existing)
#   Q9  : "Is EAGLE-3 dead for quantized 70B?" — deadpath, classifier-neutral
# For Stream D, the load-bearing assertion is that Q3-adjacent archetypes
# (open-ended "what does X" narrative, recency/turn, deadpath) do NOT
# falsely trigger the new named_entity_uses pattern.
# ------------------------------------------------------------------
OTHER_SMOKE_ARCHETYPES = [
    # narrative/arc (must NOT fire)
    ("Q1_narrative_arc", "How does Subconscious memory work?", False),
    # recency (must NOT fire — handled by recency_bridge, not A2)
    ("Q6_recency", "what happened two sessions before last", False),
    # turn-recency (must NOT fire)
    ("Q7_turn_recency", "what did I ask 2 turns ago", False),
    # narrative walk-me-through (must NOT fire)
    ("Q5_narrative_walk",
     "walk me through the fix surface from M115 through M118", False),
    # A4 specific-shape preserved (must FIRE — via A4 reuse)
    ("Q8_a4_specific",
     "was the M42 1/20 battery caused by stop tokens or stream ordering?",
     True),
    # deadpath — classifier-agnostic (must NOT fire)
    ("Q9_deadpath", "is EAGLE-3 dead for quantized 70B", False),
]


# ------------------------------------------------------------------
# K18 GUARD — predicate should NOT false-positive on open-ended,
# project-state, or opinion queries that superficially look like
# interrogative + entity but are not canonical-lookup shape.
# ------------------------------------------------------------------
K18_FALSE_POSITIVE_GUARD = [
    # project-state (M123 A3 negative-gate territory — should not fire A2)
    "what is our current strict pass rate",
    "what is my current directive",
    "what is the status of M125",
    "what is the current session",
    # opinion / phatic
    "what do you think about this",
    "what do you mean",
    # ambiguous without explicit verb marker
    "what is going on",
    "what is happening",
    # A3 encyclopedic definitional (absence-gate territory, not A2)
    "what is information theory",
    "define entropy",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def audit_m125_a2_predicate_coverage() -> dict:
    """K17 audit: did M125 A2 already cover Q3 pre-Stream-D?

    Runs the Q3 query through the *current* (post-Stream-D) classifier
    and records the match reason. The Stream D ship extends A2's
    positive patterns, so post-ship the Q3 query DOES fire — but it
    fires via the Stream-D-introduced patterns (named_entity_uses or
    named_entity_possessive). We classify "A2 covered pre-D" as True
    only if the fire reason is an A4 reuse OR a pre-Stream-D positive
    pattern (measurement_probe / test_show / introduced_overhead /
    overhead_of / comparison_measured / rate_of).
    """
    fired, diag = is_canonical_lookup(Q3_QUERY)
    reason = diag.get("decision_reason", "")
    pre_stream_d_positives = {
        "a4_specific_shape_query",
        "a2_positive_measurement_probe",
        "a2_positive_test_show",
        "a2_positive_introduced_overhead",
        "a2_positive_overhead_of",
        "a2_positive_comparison_measured",
        "a2_positive_rate_of",
    }
    covers_pre_d = fired and reason in pre_stream_d_positives
    return {
        "q3_fires_now": fired,
        "q3_reason": reason,
        "m125_a2_predicate_covered_pre_stream_d": covers_pre_d,
    }


def run_q3_replay() -> dict:
    fired, diag = is_canonical_lookup(Q3_QUERY)
    return {
        "query": Q3_QUERY,
        "fired": fired,
        "reason": diag.get("decision_reason"),
        "positive_match": diag.get("positive_match"),
        "pass": fired is True,
    }


def run_variants() -> dict:
    results = []
    passes = 0
    for q in Q3_VARIANTS:
        fired, diag = is_canonical_lookup(q)
        ok = fired is True
        if ok:
            passes += 1
        results.append({
            "query": q,
            "fired": fired,
            "reason": diag.get("decision_reason"),
            "positive_match": diag.get("positive_match"),
            "pass": ok,
        })
    return {"count_total": len(Q3_VARIANTS),
            "count_pass": passes,
            "results": results}


def run_narrative_regression() -> dict:
    results = []
    passes = 0
    for q in NARRATIVE_REGRESSION:
        fired, diag = is_canonical_lookup(q)
        ok = fired is False  # MUST NOT fire
        if ok:
            passes += 1
        results.append({
            "query": q,
            "fired": fired,
            "reason": diag.get("decision_reason"),
            "pass": ok,
        })
    return {"count_total": len(NARRATIVE_REGRESSION),
            "count_pass": passes,
            "results": results}


def run_other_smoke_archetypes() -> dict:
    results = []
    passes = 0
    for tag, q, expected in OTHER_SMOKE_ARCHETYPES:
        fired, diag = is_canonical_lookup(q)
        ok = fired is expected
        if ok:
            passes += 1
        results.append({
            "tag": tag,
            "query": q,
            "expected": expected,
            "fired": fired,
            "reason": diag.get("decision_reason"),
            "pass": ok,
        })
    return {"count_total": len(OTHER_SMOKE_ARCHETYPES),
            "count_pass": passes,
            "results": results}


def run_k18_false_positive_guard() -> dict:
    results = []
    passes = 0
    for q in K18_FALSE_POSITIVE_GUARD:
        fired, diag = is_canonical_lookup(q)
        ok = fired is False  # MUST NOT fire
        if ok:
            passes += 1
        results.append({
            "query": q,
            "fired": fired,
            "reason": diag.get("decision_reason"),
            "pass": ok,
        })
    return {"count_total": len(K18_FALSE_POSITIVE_GUARD),
            "count_pass": passes,
            "results": results}


# ------------------------------------------------------------------
# Registry write
# ------------------------------------------------------------------
def write_registry(
    verdict: str,
    classifier_refined: bool,
    q3_replay_pass: bool,
    variant_pass_count: int,
    m125_a2_predicate_covers: bool,
) -> None:
    reg_path = Path(_REPO_ROOT) / "data" / "measurement_registry.json"
    if not reg_path.exists():
        print(f"[registry] {reg_path} missing, skipping registry write")
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = reg_path.with_name(
        f"measurement_registry.json.bak_m125_3_d_{ts}")
    backup_path.write_bytes(reg_path.read_bytes())

    reg = json.loads(reg_path.read_text())
    entries = reg.setdefault("entries", {})
    # Dict-envelope per CC coordination convention.
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")
    author = "claude_ai:m125_3_d_named_entity"
    updates = {
        "m125_3.d.verdict": verdict,
        "m125_3.d.classifier_refined": bool(classifier_refined),
        "m125_3.d.q3_replay_pass": bool(q3_replay_pass),
        "m125_3.d.variant_pass_count": int(variant_pass_count),
        "m125_3.d.m125_a2_predicate_covers": bool(m125_a2_predicate_covers),
    }
    for k, v in updates.items():
        entries[k] = {
            "value": v,
            "timestamp": now_iso,
            "author": author,
            "source": "orion-ane/tests/test_m125_3_d_named_entity.py",
        }
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True))
    print(f"[registry] wrote {len(updates)} keys "
          f"(backup: {backup_path.name})")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("M125.3 Stream D — named-entity misroute classifier refinement")
    print("=" * 70)

    # [1/5] M125 A2 predicate coverage audit (K17)
    print("\n[1/5] M125 A2 predicate coverage audit (K17)")
    audit = audit_m125_a2_predicate_coverage()
    covered_pre_d = audit["m125_a2_predicate_covered_pre_stream_d"]
    print(f"  Q3 fires now      : {audit['q3_fires_now']}")
    print(f"  Q3 reason         : {audit['q3_reason']}")
    print(f"  A2 covered pre-D  : {covered_pre_d}")
    print(f"  K17 verdict       : "
          f"{'A2 covered — scope: integration only' if covered_pre_d else 'A2 gap — Stream D extended A2 positives'}")

    # [2/5] Q3 replay
    print("\n[2/5] Q3 replay — 'what compression does the ANE use'")
    q3 = run_q3_replay()
    print(f"  fired={q3['fired']}  reason={q3['reason']}  pass={q3['pass']}")

    # [3/5] 8 variants
    print("\n[3/5] Q3-class variants (8) — interrogative + named-entity")
    variants = run_variants()
    for r in variants["results"]:
        status = "OK" if r["pass"] else "FAIL"
        print(f"  [{status}] fired={r['fired']:<5} reason={r['reason']:<40} q={r['query']}")
    print(f"  → {variants['count_pass']}/{variants['count_total']} variants pass")

    # [4/5] narrative regression
    print("\n[4/5] Narrative regression (20 narrative-eligible, MUST NOT fire)")
    narr = run_narrative_regression()
    for r in narr["results"]:
        if not r["pass"]:
            print(f"  FAIL fired={r['fired']:<5} reason={r['reason']:<40} q={r['query']}")
    print(f"  → {narr['count_pass']}/{narr['count_total']} narrative preserved")

    # [5/5] other smoke archetypes
    print("\n[5/5] M125.2 E other-smoke archetype preservation (6 queries)")
    other = run_other_smoke_archetypes()
    for r in other["results"]:
        status = "OK" if r["pass"] else "FAIL"
        print(f"  [{status}] {r['tag']:<22} expected={r['expected']!s:<5} "
              f"fired={r['fired']!s:<5} reason={r['reason']}")
    print(f"  → {other['count_pass']}/{other['count_total']} archetypes preserved")

    # K18 guard (bonus)
    print("\n[K18 guard] False-positive guard on project-state / opinion / A3")
    k18 = run_k18_false_positive_guard()
    for r in k18["results"]:
        if not r["pass"]:
            print(f"  FAIL fired={r['fired']:<5} reason={r['reason']:<40} q={r['query']}")
    print(f"  → {k18['count_pass']}/{k18['count_total']} K18 guards held")

    # Verdict
    verdict_ok = (
        q3["pass"]
        and variants["count_pass"] == variants["count_total"]
        and narr["count_pass"] == narr["count_total"]
        and other["count_pass"] == other["count_total"]
        and k18["count_pass"] == k18["count_total"]
    )
    verdict = "shipped" if verdict_ok else "deferred"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  k17_m125_a2_covered_pre_stream_d : {covered_pre_d}")
    print(f"  q3_replay                        : {q3['pass']}")
    print(f"  variant_pass                     : {variants['count_pass']}/{variants['count_total']}")
    print(f"  narrative_regression             : {narr['count_pass']}/{narr['count_total']}")
    print(f"  other_smoke_archetypes           : {other['count_pass']}/{other['count_total']}")
    print(f"  k18_guard                        : {k18['count_pass']}/{k18['count_total']}")
    print("=" * 70)

    # Registry
    try:
        write_registry(
            verdict=verdict,
            classifier_refined=True,
            q3_replay_pass=q3["pass"],
            variant_pass_count=variants["count_pass"],
            m125_a2_predicate_covers=covered_pre_d,
        )
    except Exception as e:
        print(f"[registry] write failed (non-fatal): {e}")

    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
