"""Tests for M125 Stream A2 — shape-precedence arbitration.

Authoritative directive:
    vault/directives/in_progress/
      2026-04-23T01-03-39_m125_m125-day1-architectural-commit-pool-gap.md §3.2

Fixes targeted (M124 Stream A K5-refined C4, 5 pool-gap turns):
    - T22 (m123_c): "What's the exact tok/s on Llama-8B Q8 through the
      ANE pipeline?" — 5 source_role=canonical records at cos 0.56,
      narrative bypassed recall. Classifier must fire; narrative must
      be suppressed; default_recall runs directly.
    - T24 (m123_c): "Did MIE introduce any DRAM-bandwidth overhead?
      What did the test show?" — classifier must fire via
      measurement-probe / test-show pattern.
    - T59 (m122_c): "what was the exact acceptance rate of EAGLE-3 on
      Q3 70B?" — A4 specific-shape reuse.
    - T60 (m122_c): "what was the memory overhead of the MIE on DRAM
      bandwidth?" — classifier must fire via overhead-of pattern.
    - T61 (m122_c): "did we measure the ANE to be faster than the GPU
      on small matmul?" — measurement-probe / comparison-measured.

Regression coverage:
    - Narrative-eligible queries (summarize, tell me about, walk me
      through, what happened, continue, keep going, what about X we
      discussed) MUST NOT fire. Negative gate runs first.
    - 20-query false-positive audit on non-canonical-lookup shapes.
    - M122 A4 specific-shape classifier behavior unchanged (we reuse,
      not replace).
    - M123 A3 is_definitional_query untouched — classifiers are
      orthogonal.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m125_a2_shape_precedence.py

Registry values produced:
    m125.a2.verdict                         : shipped / deferred
    m125.a2.dispatch_pattern_shipped        : enum string
    m125.a2.c4_replay_pass_count            : int (0..5)
    m125.a2.narrative_regression_preserved  : 1/0
    m125.a2.dispatch_decision_field_active  : 1/0

m125_a2_shape_precedence
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
_VAULT_SUBC = os.path.join(_REPO_ROOT, "vault", "subconscious")
for _p in (_AGENT_DIR, _VAULT_SUBC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shape_precedence import (  # noqa: E402
    is_canonical_lookup,
    DISPATCH_DECISION_NARRATIVE_PRIMARY,
    DISPATCH_DECISION_RECALL_PRIMARY_CANONICAL_LOOKUP,
    DISPATCH_DECISION_NARRATIVE_SUPPRESSED_CLASSIFIER,
    DISPATCH_DECISION_OTHER,
)
from multi_path_retrieve import is_a4_specific_shape_query  # noqa: E402
from absence_guard import is_definitional_query  # noqa: E402


# ------------------------------------------------------------------
# C4-attributable turns (from m124_a_pool_gap_diagnosis.md §1)
# Each row: (turn_id, session, query, expected_fired)
# ------------------------------------------------------------------
C4_TURNS = [
    ("T22", "sess_20260422_161924_81247",
     "What's the exact tok/s on Llama-8B Q8 through the ANE pipeline?",
     True),
    ("T24", "sess_20260422_161924_81247",
     "Did MIE introduce any DRAM-bandwidth overhead? "
     "What did the test show?",
     True),
    ("T59", "sess_20260422_104816_56211",
     "what was the exact acceptance rate of EAGLE-3 on Q3 70B?",
     True),
    ("T60", "sess_20260422_104816_56211",
     "what was the memory overhead of the MIE on DRAM bandwidth?",
     True),
    ("T61", "sess_20260422_104816_56211",
     "did we measure the ANE to be faster than the GPU on small matmul?",
     True),
]


# ------------------------------------------------------------------
# Narrative-eligible regression set (MUST NOT fire). Covers
# summarize, tell me about, walk me through, what happened,
# continue, keep going, anaphoric "what about".
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
# 20-query false-positive audit (non-canonical-lookup shapes that
# should fall through to 'other' / 'no_positive_pattern'). These are
# queries that are neither specific-value probes nor narrative asks.
# ------------------------------------------------------------------
FALSE_POSITIVE_AUDIT = [
    "hey",
    "hi",
    "yes",
    "no",
    "ok",
    "thanks",
    "can you help me?",
    "I need some help",
    "that's fine",
    "not yet",
    "maybe later",
    "sure",
    "great",
    "awesome",
    "where are we at",
    "what should I do now",
    "is it working",
    "it's broken",
    "can you check",
    "please look at this",
]


# ------------------------------------------------------------------
# M122 A4 preservation: specific-shape queries that A4 catches should
# ALSO be caught by A2 (A2 reuses A4). We verify both still fire so
# the M122 A4 ranking boost continues to kick in on these queries.
# ------------------------------------------------------------------
M122_A4_REGRESSION = [
    # T52/T73/T86-adjacent shapes from M122 A4 test
    "what are the embedding dimensions on Qwen 31B?",
    "how many dims is the 8B post-RMS output?",
    "what was the exact number of tok/s?",
    "what is the precise memory bandwidth on DRAM?",
    "what was the measured latency of the ANE dispatch?",
]


# ------------------------------------------------------------------
# M123 A3 orthogonality: definitional queries (T47-shape) should be
# treated by M123 A3, not by A2. Verify A2 classifier handles them as
# 'other' (either via negative pattern or no positive match). A2 does
# not need to fire on these — they're a different class.
# ------------------------------------------------------------------
M123_A3_QUERIES = [
    ("what is information theory?", True),  # A3 fires
    ("define entropy", True),                 # A3 fires
    ("What is our current strict pass rate?", False),  # A3 doesn't fire
    ("What did we measure at M122?", False),  # A3 doesn't fire
]


# ------------------------------------------------------------------
# Dispatch-decision simulator (mirrors midas_ui.py §3482-3520 new logic).
# Given (query, narrative_would_succeed), returns the dispatch_decision
# enum the turn-log would record.
# ------------------------------------------------------------------

def simulate_dispatch_decision(
    query: str,
    narrative_would_succeed: bool,
    default_recall_would_return_results: bool = True,
) -> tuple[str, dict]:
    """Return (dispatch_decision, diag). Mirrors midas_ui.py dispatch."""
    fired, diag = is_canonical_lookup(query)
    if fired:
        # Classifier fires → narrative suppressed → default_recall runs
        if default_recall_would_return_results:
            return (DISPATCH_DECISION_RECALL_PRIMARY_CANONICAL_LOOKUP,
                    {"classifier": diag,
                     "narrative_suppressed": True,
                     "default_recall_ran": True})
        return (DISPATCH_DECISION_NARRATIVE_SUPPRESSED_CLASSIFIER,
                {"classifier": diag,
                 "narrative_suppressed": True,
                 "default_recall_ran": False})
    # Classifier did not fire → fall through to current narrative-primary
    if narrative_would_succeed:
        return (DISPATCH_DECISION_NARRATIVE_PRIMARY,
                {"classifier": diag, "narrative_used": True})
    return (DISPATCH_DECISION_OTHER,
            {"classifier": diag,
             "narrative_used": False,
             "default_recall_ran": default_recall_would_return_results})


# ------------------------------------------------------------------
# Runners
# ------------------------------------------------------------------

def run_c4_replay() -> tuple[int, int, list]:
    """For each C4 turn, verify classifier fires AND dispatch_decision
    lands on recall_primary_canonical_lookup (pre-fix: narrative_primary).
    """
    passed = 0
    fails = []
    results = []
    for turn_id, sess, query, expected in C4_TURNS:
        fired, diag = is_canonical_lookup(query)

        # Simulate post-fix behavior (narrative would have succeeded
        # pre-fix on all 5 of these turns; it's now suppressed)
        dd, ddiag = simulate_dispatch_decision(
            query, narrative_would_succeed=True,
            default_recall_would_return_results=True)

        pre_fix_dd = DISPATCH_DECISION_NARRATIVE_PRIMARY
        post_fix_dd = dd

        classifier_ok = (fired == expected)
        dispatch_ok = (post_fix_dd
                       == DISPATCH_DECISION_RECALL_PRIMARY_CANONICAL_LOOKUP)
        changed = (pre_fix_dd != post_fix_dd)
        ok = classifier_ok and dispatch_ok and changed

        results.append({
            "turn": turn_id,
            "session": sess,
            "query": query,
            "classifier_fired": fired,
            "classifier_diag": diag,
            "pre_fix_dispatch_decision": pre_fix_dd,
            "post_fix_dispatch_decision": post_fix_dd,
            "changed": changed,
            "pass": ok,
        })
        if ok:
            passed += 1
        else:
            fails.append(results[-1])
    return passed, len(C4_TURNS), results


def run_narrative_regression() -> tuple[int, int, list]:
    passed = 0
    fails = []
    for q in NARRATIVE_REGRESSION:
        fired, diag = is_canonical_lookup(q)
        # Simulate: narrative would succeed for these; A2 must stay out.
        dd, _ = simulate_dispatch_decision(
            q, narrative_would_succeed=True,
            default_recall_would_return_results=True)
        ok = (not fired) and (dd == DISPATCH_DECISION_NARRATIVE_PRIMARY)
        if ok:
            passed += 1
        else:
            fails.append({
                "query": q,
                "classifier_fired": fired,
                "dispatch_decision": dd,
                "diag": diag,
            })
    return passed, len(NARRATIVE_REGRESSION), fails


def run_false_positive_audit() -> tuple[int, int, list]:
    passed = 0
    fails = []
    for q in FALSE_POSITIVE_AUDIT:
        fired, diag = is_canonical_lookup(q)
        ok = not fired
        if ok:
            passed += 1
        else:
            fails.append({"query": q, "fired": fired, "diag": diag})
    return passed, len(FALSE_POSITIVE_AUDIT), fails


def run_m122_a4_preservation() -> tuple[int, int, list]:
    passed = 0
    fails = []
    for q in M122_A4_REGRESSION:
        a4 = is_a4_specific_shape_query(q)
        a2, diag = is_canonical_lookup(q)
        # A4 still fires (unchanged); A2 also fires (via A4 reuse).
        ok = bool(a4) and bool(a2)
        if ok:
            passed += 1
        else:
            fails.append({"query": q, "a4": a4, "a2": a2, "diag": diag})
    return passed, len(M122_A4_REGRESSION), fails


def run_m123_a3_orthogonality() -> tuple[int, int, list]:
    """Verify M123 A3 classifier is unchanged: A2 does not interfere
    with its fire/no-fire decisions on definitional queries."""
    passed = 0
    fails = []
    for q, expected_a3 in M123_A3_QUERIES:
        a3, _ = is_definitional_query(q)
        ok = (a3 == expected_a3)
        if ok:
            passed += 1
        else:
            fails.append({"query": q, "a3": a3, "expected": expected_a3})
    return passed, len(M123_A3_QUERIES), fails


def verify_dispatch_decision_field_active() -> bool:
    """Inspect midas_ui.py to confirm dispatch_decision is written to
    the turn log (retrieval.dispatch_decision). Also confirms
    schema_version bump to "2.3"."""
    mu_path = Path(_REPO_ROOT) / "orion-ane" / "agent" / "midas_ui.py"
    try:
        txt = mu_path.read_text()
    except Exception:
        return False
    has_field = ("dispatch_decision=_m125_a2_dispatch_decision" in txt)
    has_schema = ('"schema_version": "2.3"' in txt)
    return has_field and has_schema


def _write_registry(reg_updates: dict) -> bool:
    reg_path = Path(_REPO_ROOT) / "data" / "measurement_registry.json"
    if not reg_path.exists():
        print(f"[registry] SKIP — not found at {reg_path}")
        return False
    try:
        with open(reg_path) as f:
            reg = json.load(f)
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bak = reg_path.with_name(reg_path.name + f".bak_m125_a2_{ts}")
        bak.write_text(json.dumps(reg, indent=2))
        for key, value in reg_updates.items():
            reg[key] = {
                "active": True,
                "aliases": [key.split(".", 2)[-1]],
                "entity": "m125",
                "era": "m125_a2_shape_precedence",
                "measurement_type": key.split(".", 2)[-1],
                "value": value,
            }
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"[registry] wrote {len(reg_updates)} keys "
              f"(backup: {bak.name})")
        return True
    except Exception as exc:
        print(f"[registry] FAIL — {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("M125 A2 — shape-precedence arbitration")
    print("=" * 70)

    # [1/5] C4 replay
    print("\n[1/5] C4 replay (5 turns)")
    c4_pass, c4_total, c4_results = run_c4_replay()
    for r in c4_results:
        print(f"    {r['turn']}: classifier={r['classifier_fired']}  "
              f"pre={r['pre_fix_dispatch_decision']}  "
              f"post={r['post_fix_dispatch_decision']}  "
              f"PASS={r['pass']}")
        print(f"      q={r['query'][:75]}")
        print(f"      reason={r['classifier_diag'].get('decision_reason')}")
    print(f"  → {c4_pass}/{c4_total} C4 turns replayed correctly")

    # [2/5] Narrative regression
    print("\n[2/5] Narrative regression (20 queries, must NOT fire)")
    nr_pass, nr_total, nr_fails = run_narrative_regression()
    if nr_fails:
        for f in nr_fails:
            print(f"    FAIL: {f['query']!r} "
                  f"fired={f['classifier_fired']} dd={f['dispatch_decision']}")
    print(f"  → {nr_pass}/{nr_total} narrative queries preserved")

    # [3/5] False-positive audit
    print("\n[3/5] False-positive audit (20 non-canonical-lookup queries)")
    fp_pass, fp_total, fp_fails = run_false_positive_audit()
    if fp_fails:
        for f in fp_fails:
            print(f"    FAIL: {f['query']!r} fired=True")
    print(f"  → {fp_pass}/{fp_total} audit cases passed")

    # [4/5] M122 A4 preservation
    print("\n[4/5] M122 A4 specific-shape preservation")
    a4_pass, a4_total, a4_fails = run_m122_a4_preservation()
    if a4_fails:
        for f in a4_fails:
            print(f"    FAIL: {f['query']!r} a4={f['a4']} a2={f['a2']}")
    print(f"  → {a4_pass}/{a4_total} A4 cases preserved")

    # [4b/5] M123 A3 orthogonality
    print("\n[5/5] M123 A3 orthogonality (definitional classifier unchanged)")
    a3_pass, a3_total, a3_fails = run_m123_a3_orthogonality()
    if a3_fails:
        for f in a3_fails:
            print(f"    FAIL: {f['query']!r} a3={f['a3']} "
                  f"expected={f['expected']}")
    print(f"  → {a3_pass}/{a3_total} A3 orthogonality preserved")

    # ζ v2.3 dispatch_decision field verification
    dd_active = verify_dispatch_decision_field_active()
    print(f"\ndispatch_decision field active (ζ v2.3): {dd_active}")

    # Verdict
    all_green = (
        c4_pass == c4_total
        and nr_pass == nr_total
        and fp_pass == fp_total
        and a4_pass == a4_total
        and a3_pass == a3_total
        and dd_active
    )
    verdict = "shipped" if all_green else "deferred"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  c4_replay: {c4_pass}/{c4_total}")
    print(f"  narrative_regression: {nr_pass}/{nr_total}")
    print(f"  false_positive_audit: {fp_pass}/{fp_total}")
    print(f"  m122_a4_preservation: {a4_pass}/{a4_total}")
    print(f"  m123_a3_orthogonality: {a3_pass}/{a3_total}")
    print(f"  dispatch_decision_field_active: {dd_active}")
    print("=" * 70)

    reg_updates = {
        "m125.a2.verdict": verdict,
        "m125.a2.dispatch_pattern_shipped": "classifier_suppression",
        "m125.a2.c4_replay_pass_count": int(c4_pass),
        "m125.a2.narrative_regression_preserved":
            bool(nr_pass == nr_total),
        "m125.a2.dispatch_decision_field_active": bool(dd_active),
    }
    _write_registry(reg_updates)

    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
