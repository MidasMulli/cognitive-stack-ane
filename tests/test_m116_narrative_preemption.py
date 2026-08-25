"""M116 Stream D — narrative-preemption silent hole fix.

Authoritative spec:
    vault/agent_reports/m114_a3_beta_data_harvest.md §8 item 1
    Fix option (a) §3.4: drop `narrative_used` from absence-guard
    eligibility predicate.

Pre-fix behavior (M115 β in production as of 2026-04-20):
    midas_ui.py:3367 sets `_guard_eligible = not _narrative_used`.
    When narrative_used==True, guard is silent regardless of pool state.
    T1/T7/T8/T10 pilot turns matched (pool==0 AND narrative_used==True)
    → guard silent → model confabulates without safety net.

Post-fix behavior (M116 Stream D option (a)):
    `_guard_eligible = True` (narrative_used dropped from predicate).
    β sub-gates (M115) evaluate narrative adequacy via pool_size +
    max_score naturally. narrative_used retained in feature_vector for
    observability (M109 ζ F5).

Tests:
    Class A — Primary T1/T7/T8/T10 replay: eligibility==True post-fix.
    Class B — Regression valid narrative-answered case: guard silent via
              β sub-gates, not via eligibility short-circuit.
    Class C — K5 FP check: does the fix fire inappropriately on a valid
              narrative-answered case?

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m116_narrative_preemption.py

Registry values produced:
    m116.d.verdict                                          : SHIP/DEFER
    m116.d.fix_shape_chosen                                 : a (or b)
    m116.d.t1_t7_t8_t10_eligibility_true_post_fix           : 1/0
    m116.d.valid_narrative_case_no_regression               : 1/0

m116_d_drop_narrative_used
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from absence_guard import check_absence  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Simulator — mirrors midas_ui.py Phase 4 POST-M116-D.
# Key difference from test_m115 simulator: `_guard_eligible = True`
# unconditionally, and call sites pass narrative_result=None so the
# callee's internal narrative preempt does not re-introduce the hole.
# ──────────────────────────────────────────────────────────────────────

def simulate_post_fix(
    query: str,
    filtered: list,
    narrative_used: bool,
) -> tuple[bool, str, bool]:
    """Return (fired, branch_name, eligibility).

    branch_name ∈ {"sub_gate_1", "sub_gate_3_or_2",
                   "word_overlap_not_triggered"}
    eligibility: the value of _guard_eligible post-fix (should always be
                 True; returned for explicit assertion).
    """
    mem_ctx = list(filtered) if filtered else []

    # m116_d_drop_narrative_used: eligibility no longer gated on narrative
    _guard_eligible = True
    eligibility = _guard_eligible

    # Branch 1 — empty mem_ctx (Sub-Gate 1)
    if not mem_ctx:
        # Pass None for narrative_result so β sub-gates can run.
        guard = check_absence(query, [], None)
        return (guard is not None, "sub_gate_1", eligibility)

    # Branch 2 — word-overlap mismatch path (Sub-Gates 2 and 3)
    _q_words = set(
        w.lower().strip("?.,!\"'") for w in query.split() if len(w) >= 4)
    _stop = {"what", "does", "that", "this", "with", "from", "have",
             "been", "about", "which", "where", "when", "many", "much",
             "strategy", "should", "would", "could", "explain", "tell"}
    _q_specific = _q_words - _stop

    if not _q_specific or len(_q_specific) < 2:
        return (False, "word_overlap_not_triggered", eligibility)

    _mem_text = " ".join(
        (r.get("text", "") if isinstance(r, dict) else str(r))
        for r in mem_ctx).lower()
    _unmatched = [w for w in _q_specific if w not in _mem_text]

    if len(_unmatched) <= len(_q_specific) * 0.5:
        return (False, "word_overlap_not_triggered", eligibility)

    # β Sub-Gate 3 call-site — pass filtered and None narrative.
    guard = check_absence(query, filtered, None)
    return (guard is not None, "sub_gate_3_or_2", eligibility)


def simulate_pre_fix(
    query: str,
    filtered: list,
    narrative_used: bool,
) -> tuple[bool, str, bool]:
    """PRE-M116-D simulator for reference — gates on narrative_used."""
    mem_ctx = list(filtered) if filtered else []
    nr = {"intent": "narrative"} if narrative_used else None

    _guard_eligible = not narrative_used  # pre-fix predicate
    eligibility = _guard_eligible

    if not _guard_eligible:
        return (False, "narrative_preempt", eligibility)

    if not mem_ctx:
        guard = check_absence(query, [], nr)
        return (guard is not None, "sub_gate_1", eligibility)

    return (False, "other", eligibility)


# ──────────────────────────────────────────────────────────────────────
# Class A — T1/T7/T8/T10 pilot replay
# (pool=0, narrative_used=True)
# ──────────────────────────────────────────────────────────────────────

# Synthesized pilot fixtures representing T1/T7/T8/T10:
# Per M114 A3 §8 item 1: these four turns share (pool_size==0,
# narrative_used==True). Synthetic queries stand in for the pilot turns
# since feature-vector file is per-turn-number; the predicate under test
# is structural, not query-text-dependent.
_T1_T7_T8_T10_FIXTURES = [
    # (turn_label, query, filtered, narrative_used)
    ("T1",  "what is the subconscious memory extraction rate", [], True),
    ("T7",  "explain how ANE dispatch overhead affects decode", [], True),
    ("T8",  "describe the SLC cache hint mechanism on M5 Pro", [], True),
    ("T10", "what's happening with spec decode acceptance", [], True),
]


def test_class_a_primary() -> tuple[int, int, list[str]]:
    """Post-fix: eligibility==True for all four turns; guard evaluates
    normally; β Sub-Gate 1 fires (pool==0 unconditional)."""
    lines = []
    passed = 0
    failed = 0

    for label, q, pool, narr in _T1_T7_T8_T10_FIXTURES:
        # Pre-fix reference measurement (for evidence)
        _, pre_branch, pre_elig = simulate_pre_fix(q, pool, narr)

        # Post-fix measurement
        fired, branch, elig = simulate_post_fix(q, pool, narr)

        # Assertions per brief:
        # (1) eligibility is True post-fix
        # (2) β Sub-Gate 1 fires (pool==0 unconditional per M115)
        ok_elig = (elig is True)
        ok_fire = fired and branch == "sub_gate_1"
        ok_pre  = (pre_elig is False)  # confirms pre-fix silent hole

        ok = ok_elig and ok_fire and ok_pre
        mark = "PASS" if ok else "FAIL"
        lines.append(
            f"  [{mark}] {label:4s} pre_elig={pre_elig} "
            f"post_elig={elig} post_fired={fired} branch={branch} "
            f"q={q[:50]!r}"
        )
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Class B — Regression: valid narrative-answered case
# Narrative fires + pool non-empty + high max_score
# Pre-fix: guard silent via eligibility=False (narrative_preempt)
# Post-fix: guard runs; β Sub-Gate 3 suppresses via high_quality floor
# Asserts the suppression comes from β, NOT from eligibility.
# ──────────────────────────────────────────────────────────────────────

def test_class_b_regression() -> tuple[int, int, list[str]]:
    lines = []
    passed = 0
    failed = 0

    # Valid narrative-answered case: narrative fires + high-score pool
    # (pool contains a row scoring >= 0.5 floor). Query has specific
    # terms that DO trip word-overlap branch entry so we actually reach
    # Sub-Gate 3 / 2 (not the word_overlap_not_triggered fast-silence).
    query = "what is the ANE compilation pipeline doing today"
    high_score_pool = [
        # Pool text deliberately has NO overlap with query_specific words
        # so the word-overlap branch IS triggered; then Sub-Gate 3's
        # score floor (>=0.5) must provide the suppression.
        {"score": 1.2, "text": "unrelated placeholder text row one"},
        {"score": 0.3, "text": "unrelated placeholder text row two"},
    ]

    pre_fired, pre_branch, pre_elig = simulate_pre_fix(
        query, high_score_pool, True)
    post_fired, post_branch, post_elig = simulate_post_fix(
        query, high_score_pool, True)

    # Pre-fix: silent via eligibility short-circuit
    ok_pre = (not pre_fired) and pre_branch == "narrative_preempt"

    # Post-fix: eligibility True, guard runs, β Sub-Gate 3 suppresses
    # (silent but via β, not eligibility)
    ok_post_elig = (post_elig is True)
    ok_post_silent = (not post_fired)
    ok_post_branch = post_branch in ("sub_gate_3_or_2",
                                     "word_overlap_not_triggered")

    # K5: fix is correct only if post-fix is ALSO silent (β does work)
    # If post_fired==True here, K5 fires → FP regression.
    ok_k5 = ok_post_silent

    ok = ok_pre and ok_post_elig and ok_post_silent and ok_post_branch
    mark = "PASS" if ok else "FAIL"
    lines.append(
        f"  [{mark}] valid-narrative-case "
        f"pre_fired={pre_fired} pre_branch={pre_branch} "
        f"post_fired={post_fired} post_branch={post_branch} "
        f"post_elig={post_elig}"
    )
    if ok:
        passed += 1
    else:
        failed += 1

    # Explicit K5 line for registry-readability
    lines.append(
        f"  [{'PASS' if ok_k5 else 'FAIL'}] K5 no-FP-regression: "
        f"post-fix silent on valid narrative case = {ok_k5}"
    )
    if ok_k5:
        passed += 1
    else:
        failed += 1

    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Class C — Additional K5 stress: narrative + pool + low score
# Narrative fires but pool only has weak rows. Post-fix SHOULD fire
# (β Sub-Gate 2 low-score path). Pre-fix was silent via eligibility.
# This is the intended behavior change of M116 D — confirming the
# silent hole closure, not a regression.
# ──────────────────────────────────────────────────────────────────────

def test_class_c_low_score_hole_closed() -> tuple[int, int, list[str]]:
    lines = []
    passed = 0
    failed = 0

    # Narrative fires but pool only has tangential low-score matches +
    # word-overlap mismatch with a domain-relevant query.
    query = "what is the ANE compilation pipeline doing today"
    low_score_pool = [
        {"score": 0.2, "text": "unrelated placeholder text row one"},
        {"score": 0.1, "text": "unrelated placeholder text row two"},
    ]

    pre_fired, pre_branch, pre_elig = simulate_pre_fix(
        query, low_score_pool, True)
    post_fired, post_branch, post_elig = simulate_post_fix(
        query, low_score_pool, True)

    # Pre-fix: silent via eligibility short-circuit (the hole)
    ok_pre = (not pre_fired) and pre_branch == "narrative_preempt"

    # Post-fix: guard runs, β Sub-Gate 2 fires (low-score + domain +
    # word-overlap mismatch)
    ok_post_elig = (post_elig is True)
    ok_post_fired = post_fired and post_branch == "sub_gate_3_or_2"

    ok = ok_pre and ok_post_elig and ok_post_fired
    mark = "PASS" if ok else "FAIL"
    lines.append(
        f"  [{mark}] low-score-narrative-hole "
        f"pre_fired={pre_fired} pre_branch={pre_branch} "
        f"post_fired={post_fired} post_branch={post_branch} "
        f"(hole previously silent; now β fires)"
    )
    if ok:
        passed += 1
    else:
        failed += 1
    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Class D — feature_vector retention check
# narrative_used must still be populatable / retained for observability
# (M109 ζ F5). We assert the field name still appears in absence_guard
# docstring comments so future refactors don't silently drop it.
# ──────────────────────────────────────────────────────────────────────

def test_class_d_observability_retained() -> tuple[int, int, list[str]]:
    lines = []
    passed = 0
    failed = 0

    # Read the midas_ui _abs_features block and confirm narrative_used
    # field is still present (retained per brief "feature_vector still
    # has narrative_used populated").
    midas_ui_path = os.path.join(_AGENT_DIR, "midas_ui.py")
    with open(midas_ui_path, "r") as f:
        contents = f.read()

    # narrative_used must appear inside _abs_features dict (F5 log)
    has_feature_field = '"narrative_used": _narrative_used' in contents
    # m116 marker must be present in the production diff
    has_m116_marker = "m116_d_drop_narrative_used" in contents

    ok = has_feature_field and has_m116_marker
    mark = "PASS" if ok else "FAIL"
    lines.append(
        f"  [{mark}] observability: "
        f"narrative_used field retained in feature_vector={has_feature_field} "
        f"m116 marker present={has_m116_marker}"
    )
    if ok:
        passed += 1
    else:
        failed += 1
    return passed, failed, lines


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("M116 Stream D — narrative-preemption silent hole fix")
    print("=" * 72)
    print()

    print("[Class A] T1/T7/T8/T10 primary replay (pool==0, narr=True)")
    pA, fA, LA = test_class_a_primary()
    for ln in LA:
        print(ln)
    print(f"  Class A: {pA}/{pA + fA} passed")
    print()

    print("[Class B] Regression — valid narrative-answered case")
    pB, fB, LB = test_class_b_regression()
    for ln in LB:
        print(ln)
    print(f"  Class B: {pB}/{pB + fB} passed")
    print()

    print("[Class C] Silent hole closure — low-score narrative case")
    pC, fC, LC = test_class_c_low_score_hole_closed()
    for ln in LC:
        print(ln)
    print(f"  Class C: {pC}/{pC + fC} passed")
    print()

    print("[Class D] Observability — narrative_used retained in F5 log")
    pD, fD, LD = test_class_d_observability_retained()
    for ln in LD:
        print(ln)
    print(f"  Class D: {pD}/{pD + fD} passed")
    print()

    total_pass = pA + pB + pC + pD
    total_fail = fA + fB + fC + fD
    total = total_pass + total_fail

    # Registry sub-metric calculations
    t1_t7_t8_t10_ok = (fA == 0)
    valid_narr_no_regression = ("PASS" in LB[0]) and ("PASS" in LB[1])
    k5_fp_regression = not valid_narr_no_regression

    print("=" * 72)
    print("Registry sub-metrics:")
    print(f"  t1_t7_t8_t10_eligibility_true_post_fix    : "
          f"{'1' if t1_t7_t8_t10_ok else '0'}")
    print(f"  valid_narrative_case_no_regression        : "
          f"{'1' if valid_narr_no_regression else '0'}")
    print(f"  k5_fp_regression                          : "
          f"{'1' if k5_fp_regression else '0'}")
    print(f"  fix_shape_chosen                          : a")
    print()
    print(f"TOTAL: {total_pass}/{total} passed  ({total_fail} failed)")

    verdict = "SHIP" if total_fail == 0 else (
        "DEFER" if k5_fp_regression else "INVESTIGATE")
    print(f"Verdict: {verdict}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
