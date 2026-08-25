"""Tests for the M115 β three-sub-gate absence-guard split.

Authoritative spec: vault/agent_reports/m114_a3_beta_data_harvest.md §9
Fix targets:
    Sub-Gate 1 — pool=0 + narrative=False → fire unconditionally
                 (drops _is_domain_relevant silencing; 6 FN fix)
    Sub-Gate 2 — pool>0, max<0.5, word-overlap mismatch → fire (preserve)
    Sub-Gate 3 — pool>0, max>=0.5, word-overlap mismatch → DO NOT fire
                 (score-context-strip FP fix: pass filtered not [])

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m115_beta_split_gate.py

Registry values produced:
    m115.beta.sub_gate_1_active                 : 1/0
    m115.beta.sub_gate_2_preserved              : 1/0
    m115.beta.sub_gate_3_active                 : 1/0
    m115.beta.regression_t16_t18_still_fires    : 1/0
    m115.beta.regression_t9_t11_t37_no_longer_fires : 1/0
    m115.beta.regression_t6_plus_five_now_fires : 1/0
    m115.beta.synthetic_battery_pass_rate       : <fraction>
    m115.beta.verdict                            : SHIP/DEFER

m115_beta_split_gate
"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from absence_guard import check_absence  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Simulator for the midas_ui Phase-4 absence-guard block.
# Mirrors the production code path *exactly* as rewritten in M115 so the
# test validates the full caller-side predicate stack, not only the
# check_absence callee.
# ──────────────────────────────────────────────────────────────────────

def simulate_absence_guard(
    query: str,
    filtered: list,
    narrative_used: bool,
) -> tuple[bool, str]:
    """Return (fired, branch_name) — mirrors midas_ui Phase 4.

    branch_name ∈ {"narrative_preempt", "sub_gate_1", "sub_gate_3_or_2",
                   "word_overlap_not_triggered", "mem_ctx_not_evaluated"}
    """
    # narrative_result: non-None if narrative produced content
    nr = {"intent": "narrative"} if narrative_used else None

    # Build mem_ctx the way midas_ui does: if `filtered` has content,
    # mem_ctx is populated (we only care about truthiness here for the
    # branch decision).
    mem_ctx = list(filtered) if filtered else []

    _guard_eligible = not narrative_used
    if not _guard_eligible:
        return (False, "narrative_preempt")

    # Branch 1 — empty mem_ctx (Sub-Gate 1)
    if not mem_ctx:
        guard = check_absence(query, [], nr)
        return (guard is not None, "sub_gate_1")

    # Branch 2 — word-overlap mismatch path (Sub-Gates 2 and 3 live here)
    _q_words = set(
        w.lower().strip("?.,!\"'") for w in query.split() if len(w) >= 4)
    _stop = {"what", "does", "that", "this", "with", "from", "have",
             "been", "about", "which", "where", "when", "many", "much",
             "strategy", "should", "would", "could", "explain", "tell"}
    _q_specific = _q_words - _stop

    if not _q_specific or len(_q_specific) < 2:
        return (False, "word_overlap_not_triggered")

    _mem_text = " ".join(
        (r.get("text", "") if isinstance(r, dict) else str(r))
        for r in mem_ctx).lower()
    _unmatched = [w for w in _q_specific if w not in _mem_text]

    if len(_unmatched) <= len(_q_specific) * 0.5:
        return (False, "word_overlap_not_triggered")

    # m115_beta_sub_gate_3 call-site: pass filtered, not []
    guard = check_absence(query, filtered, nr)
    return (guard is not None, "sub_gate_3_or_2")


# ──────────────────────────────────────────────────────────────────────
# Test Class 1 — Pilot replay (36-turn corpus at data/session_logs/…)
# ──────────────────────────────────────────────────────────────────────

def _load_pilot_feature_vectors() -> dict[int, dict]:
    """Load the 36 pilot feature vectors keyed by turn number."""
    path = os.path.join(_REPO_ROOT, "data", "m114",
                        "beta_feature_vectors.jsonl")
    rows: dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["turn_number"]] = obj
    return rows


def _synthesize_pool_from_fv(fv_row: dict) -> list:
    """Reconstruct a synthetic filtered pool from feature vector stats.

    The feature vector captures pool_size + max_score + word-overlap
    matched vs unmatched counts. That's sufficient for predicate
    evaluation: we don't need the original texts — only (a) score values
    on recall_results dicts so Sub-Gate 3 high_quality filter can trip,
    and (b) a synthetic combined text so the word-overlap predicate
    re-evaluates to the same unmatched_ratio as the original turn.
    """
    fv = fv_row["feature_vector"]
    pool_size = fv.get("pool_size", 0)
    max_score = fv.get("max_score", 0.0)
    if pool_size == 0:
        return []

    # Reconstruct word-overlap conditions:
    # the word-overlap branch enters iff unmatched_ratio > 0.5 in
    # midas_ui, so we rebuild _mem_text so that exactly the right number
    # of query-specific words appear vs don't appear.
    q = fv_row["query"]
    _q_words = set(
        w.lower().strip("?.,!\"'") for w in q.split() if len(w) >= 4)
    _stop = {"what", "does", "that", "this", "with", "from", "have",
             "been", "about", "which", "where", "when", "many", "much",
             "strategy", "should", "would", "could", "explain", "tell"}
    _q_specific = sorted(_q_words - _stop)

    q_unmatched_count = fv.get("q_unmatched_count", 0)
    if isinstance(q_unmatched_count, str):  # "not_consumed_this_turn"
        q_unmatched_count = 0

    # Split _q_specific into matched (words we DO include in mem_text)
    # and unmatched (words we DO NOT).
    if q_unmatched_count >= len(_q_specific):
        matched_words = []
    else:
        matched_words = _q_specific[q_unmatched_count:]

    # Synthetic mem_text contains matched_words only.
    synth_text = " ".join(matched_words) if matched_words else "placeholder"

    # Build pool entries — one carries max_score, rest score 0.0.
    pool = [{"text": synth_text, "score": float(max_score)}]
    for _ in range(pool_size - 1):
        pool.append({"text": synth_text, "score": 0.0})
    return pool


def test_pilot_replay(pilot: dict) -> tuple[int, int, list[str]]:
    """Run the 11 pilot-marked turns and check M115 predictions.

    Expected per M115 A3 §9 spec:
        Sub-Gate 2 fires (regression-preserve):  T16, T18
        Sub-Gate 3 no-fires (FP fix):            T9, T11, T37
        Sub-Gate 1 fires (FN fix):               T6, T14, T15, T24, T26, T27
    """
    cases = [
        # (turn, expected_fired, gate_label)
        (16, True, "Sub-Gate 2 (preserve)"),
        (18, True, "Sub-Gate 2 (preserve)"),
        (9, False, "Sub-Gate 3 (FP fix)"),
        (11, False, "Sub-Gate 3 (FP fix)"),
        (37, False, "Sub-Gate 3 (FP fix)"),
        (6, True, "Sub-Gate 1 (FN fix)"),
        (14, True, "Sub-Gate 1 (FN fix)"),
        (15, True, "Sub-Gate 1 (FN fix)"),
        (24, True, "Sub-Gate 1 (FN fix)"),
        (26, True, "Sub-Gate 1 (FN fix)"),
        (27, True, "Sub-Gate 1 (FN fix)"),
    ]

    passed = 0
    failed = 0
    lines = []
    for turn_n, expected_fired, label in cases:
        row = pilot[turn_n]
        q = row["query"]
        narr_used = row["feature_vector"].get("narrative_used", False)
        pool = _synthesize_pool_from_fv(row)
        fired, branch = simulate_absence_guard(q, pool, narr_used)
        ok = (fired == expected_fired)
        mark = "PASS" if ok else "FAIL"
        lines.append(
            f"  [{mark}] T{turn_n:02d} {label:30s} expected fire={expected_fired}"
            f" got fire={fired} branch={branch} q={q[:60]!r}"
        )
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Test Class 2 — Synthetic battery (empty-pool variants + edges)
# ──────────────────────────────────────────────────────────────────────

def test_synthetic_battery() -> tuple[int, int, list[str]]:
    """10 empty-pool queries + 2 edges.

    All 10 empty-pool queries must fire (Sub-Gate 1 unconditional).
    Edge 1 (pool=0 + narrative=True) must NOT fire.
    Edge 2 (pool=0 + chit-chat general knowledge) must fire
    (this is the general-knowledge confabulation class — Gate B silencing
    is the bug Sub-Gate 1 fixes).
    """
    empty_pool_cases = [
        # (query, note)
        ("how does entropy apply to our work?", "domain: work"),
        ("what is CLIP?", "general-knowledge, no domain terms"),
        ("what are my thoughts on characterization?", "meta, no domain"),
        ("can you check online for Opus 4.7", "no domain keyword"),
        ("opinion on Opus 4.7", "no domain keyword"),
        ("alternative form of communication", "no domain keyword"),
        ("what is the ane-compiler build status", "domain: ane-compiler"),
        ("tell me about dispatch queueing", "domain: dispatch"),
        ("what happened yesterday with spec decode", "domain: spec decode"),
        ("give me the Boltzmann constant", "pure general-knowledge"),
    ]

    lines = []
    passed = 0
    failed = 0

    for q, note in empty_pool_cases:
        fired, branch = simulate_absence_guard(q, [], False)
        ok = fired and branch == "sub_gate_1"
        mark = "PASS" if ok else "FAIL"
        lines.append(
            f"  [{mark}] empty-pool '{q[:50]}' ({note}) fired={fired} "
            f"branch={branch}"
        )
        if ok:
            passed += 1
        else:
            failed += 1

    # Edge 1: pool=0, narrative_used=True → NOT fire (narrative preempt)
    fired, branch = simulate_absence_guard(
        "what is the weather", [], narrative_used=True)
    ok1 = (not fired) and branch == "narrative_preempt"
    lines.append(
        f"  [{'PASS' if ok1 else 'FAIL'}] edge1 pool=0+narr=True "
        f"fired={fired} branch={branch} (expected NOT fire)"
    )
    if ok1:
        passed += 1
    else:
        failed += 1

    # Edge 2: pool=0, chit-chat general-knowledge → MUST fire (Sub-Gate 1).
    # Pre-fix this was silenced by Gate B (_is_domain_relevant=False on
    # "what is a quasar"). Post-fix Sub-Gate 1 fires unconditionally.
    fired, branch = simulate_absence_guard(
        "what is a quasar", [], narrative_used=False)
    ok2 = fired and branch == "sub_gate_1"
    lines.append(
        f"  [{'PASS' if ok2 else 'FAIL'}] edge2 general-knowledge+pool=0 "
        f"fired={fired} branch={branch} (expected fire)"
    )
    if ok2:
        passed += 1
    else:
        failed += 1

    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Test Class 3 — Unit tests on check_absence directly
# ──────────────────────────────────────────────────────────────────────

def test_check_absence_unit() -> tuple[int, int, list[str]]:
    """Direct invariants on the check_absence callee."""
    lines = []
    passed = 0
    failed = 0

    def _expect(desc: str, got, exp):
        nonlocal passed, failed
        ok = (got is None) == (exp is None) and (
            exp is None or got is not None)
        mark = "PASS" if ok else "FAIL"
        lines.append(
            f"  [{mark}] unit: {desc} "
            f"got={'FIRE' if got else 'SILENT'} "
            f"expected={'FIRE' if exp else 'SILENT'}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Sub-Gate 1: pool=[] + narrative=None always fires, even w/ non-domain q
    _expect("pool=[], narr=None, non-domain query",
            check_absence("wqwqwq xyzzy", [], None),
            "FIRE")

    # narrative_result != None suppresses
    _expect("narrative_result present",
            check_absence("ANE compilation", [], {"intent": "narrative"}),
            None)

    # Sub-Gate 3: pool with high-score row suppresses even on domain query
    _expect("high-score pool suppresses",
            check_absence("ANE compilation",
                          [{"score": 1.5, "text": "foo"}], None),
            None)

    # Sub-Gate 2 fires: low-score pool + domain query
    _expect("low-score pool + domain query fires",
            check_absence("ANE compilation",
                          [{"score": 0.3, "text": "foo"}], None),
            "FIRE")

    # Low-score pool + non-domain query still silences
    _expect("low-score pool + non-domain query silences",
            check_absence("wqwqwq xyzzy",
                          [{"score": 0.3, "text": "foo"}], None),
            None)

    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("M115 β three-sub-gate split — regression matrix")
    print("=" * 72)

    pilot = _load_pilot_feature_vectors()
    print(f"Loaded {len(pilot)} pilot turn feature vectors")
    print()

    print("[Class 1] Pilot replay (11 targeted turns)")
    p1, f1, L1 = test_pilot_replay(pilot)
    for ln in L1:
        print(ln)
    print(f"  Class 1: {p1}/{p1 + f1} passed")
    print()

    print("[Class 2] Synthetic battery (10 empty-pool + 2 edges)")
    p2, f2, L2 = test_synthetic_battery()
    for ln in L2:
        print(ln)
    print(f"  Class 2: {p2}/{p2 + f2} passed")
    print()

    print("[Class 3] check_absence unit tests")
    p3, f3, L3 = test_check_absence_unit()
    for ln in L3:
        print(ln)
    print(f"  Class 3: {p3}/{p3 + f3} passed")
    print()

    total_pass = p1 + p2 + p3
    total_fail = f1 + f2 + f3
    total = total_pass + total_fail

    # Check specific registry sub-metrics
    t16_18 = sum(1 for ln in L1 if "T16" in ln or "T18" in ln
                 and "PASS" in ln)
    t9_11_37 = sum(1 for ln in L1 if (
        "T09" in ln or "T11" in ln or "T37" in ln) and "PASS" in ln)
    t6_plus = sum(1 for ln in L1 if (
        "T06" in ln or "T14" in ln or "T15" in ln or "T24" in ln
        or "T26" in ln or "T27" in ln) and "PASS" in ln)

    print("=" * 72)
    print("Registry sub-metrics:")
    print(f"  regression_t16_t18_still_fires            : "
          f"{'1' if t16_18 == 2 else '0'} (passed {t16_18}/2)")
    print(f"  regression_t9_t11_t37_no_longer_fires     : "
          f"{'1' if t9_11_37 == 3 else '0'} (passed {t9_11_37}/3)")
    print(f"  regression_t6_plus_five_now_fires         : "
          f"{'1' if t6_plus == 6 else '0'} (passed {t6_plus}/6)")
    print(f"  synthetic_battery_pass_rate               : "
          f"{p2}/{p2 + f2} = {p2 / max(1, p2 + f2):.2%}")
    print()
    print(f"TOTAL: {total_pass}/{total} passed  ({total_fail} failed)")

    verdict = "SHIP" if total_fail == 0 else "DEFER"
    print(f"Verdict: {verdict}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
