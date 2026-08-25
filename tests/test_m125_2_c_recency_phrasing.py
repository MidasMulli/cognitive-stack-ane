"""M125.2 Stream C — recency-gate "what happened" phrasing extension.

Authoritative spec:
    vault/directives/in_progress/2026-04-23T18-14-19_m125_2open_m125-2-a3-relocation-full-ship-pilot.md §3.3
    vault/agent_reports/m125_1_b_recency_ordinal.md
    vault/agent_reports/m125_1_d_pilot_readiness.md (Q4 gap)

Mechanism anchor: M125.1 D §Q4 ordinal parser gap
    Pre-M125.2-C: _RECENCY_QUERY_RE did not match "what happened N
    sessions ago/back/before last". _parse_ordinal_nback downstream
    correctly extracts N, but the upstream short-circuit gate rejects
    the query before the ordinal parser sees it. Query falls through
    to L2 memory_recall, bypassing the L0 session-summary accessor.

Fix under test:
    _RECENCY_QUERY_RE extended with ONE new alternation — "what
    happened" + ordinal-session marker. The ordinal pattern mirrors
    _RECENCY_ORDINAL_NBACK_RE: (digits|spelled 1-10) + sessions? +
    (ago|back|before last).

Under-fit discipline (K8):
    "what happened" ALONE, or with non-ordinal continuations
    ("what happened at Main 42", "what happened yesterday"), MUST
    NOT match. Factual-recall shapes stay L2.

K9 deferral:
    "what did we cover N sessions ago", "what was the focus N
    sessions ago" are NOT added in M125.2. Directive §3.3 scope
    bounded to "what happened" family only.

K10 guard:
    Downstream _parse_ordinal_nback classifier is NOT touched.
    M125.1 B classifier 10/10 regression preserved.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m125_2_c_recency_phrasing.py

Registry values produced:
    m125_2.c.verdict                          : shipped | deferred
    m125_2.c.recency_re_extension_shipped     : bool
    m125_2.c.q4_replay_pass                   : bool
    m125_2.c.phrasing_variant_pass_count      : int (of attempted)
"""

from __future__ import annotations

import os
import sys
import traceback

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)


def _import_module():
    """Import midas_ui lazily (same pattern as M121 C test)."""
    import midas_ui  # type: ignore
    return midas_ui


# ---------- Test harness ----------

_FAILURES = []
_PASSES = 0


def _check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSES
    if cond:
        _PASSES += 1
        print(f"  PASS  {name}")
    else:
        _FAILURES.append((name, detail))
        print(f"  FAIL  {name}  — {detail}")


# M125.1 D Q4 replay — the canonical production-failure query.
Q4_REPLAY = "what happened two sessions before last"


def test_1_q4_replay(mod) -> bool:
    """Q4 replay — the canonical M125.1 D §Q4 failure query MUST now
    route to the ordinal short-circuit (regex matches, short-circuit
    gate fires, ordinal parser extracts N=2 with session scope)."""
    print("\nTest 1 — Q4 replay (canonical M125.1 D failure)")

    q = Q4_REPLAY
    _check(
        "T1.Q4.is_recency_query",
        bool(mod._is_recency_query(q)),
        "Q4 must match _RECENCY_QUERY_RE after M125.2 C extension",
    )
    _check(
        "T1.Q4.short_circuits",
        mod._m120_b_should_short_circuit(q),
        "Q4 must fire L0 short-circuit (no topical anchor)",
    )
    parsed = mod._parse_ordinal_nback(q)
    # Accept both M125.1 B (int) and M125.2 D (tuple) classifier
    # return shapes — regression protection across streams.
    if isinstance(parsed, tuple):
        n, scope = parsed
        _check(
            "T1.Q4.ordinal_parsed_n",
            n == 2,
            f"expected N=2, got {n}",
        )
        _check(
            "T1.Q4.ordinal_parsed_scope_session",
            scope == "session",
            f"expected scope='session', got {scope!r}",
        )
    else:
        _check(
            "T1.Q4.ordinal_parsed_n",
            parsed == 2,
            f"expected N=2, got {parsed!r}",
        )
    return not any(n.startswith("T1.") for n, _ in _FAILURES)


def test_2_what_happened_positive(mod) -> bool:
    """Positive cases: 'what happened' + explicit ordinal-session
    marker MUST short-circuit. All phrasings carry an ordinal +
    'sessions' + {ago|back|before last}."""
    print("\nTest 2 — 'what happened' + ordinal positive cases")

    positive = [
        "what happened three sessions ago",
        "what happened 5 sessions back",
        "what happened ten sessions ago",
        "what happened 1 session ago",
        "what happened nine sessions back",
        "what happened four sessions before last",
        "what happened 10 sessions before last",
        "what happened two sessions before last",  # Q4
    ]
    for q in positive:
        _check(
            f"T2.is_recency[{q!r}]",
            bool(mod._is_recency_query(q)),
            "must match after M125.2 C extension",
        )
        _check(
            f"T2.short_circuits[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "must fire L0 short-circuit",
        )
        parsed = mod._parse_ordinal_nback(q)
        _check(
            f"T2.ordinal_parsed[{q!r}]",
            parsed is not None,
            "downstream ordinal parser must fire",
        )
    return not any(n.startswith("T2.") for n, _ in _FAILURES)


def test_3_what_happened_negative(mod) -> bool:
    """Under-fit (K8): 'what happened' WITHOUT ordinal-session marker
    MUST NOT short-circuit. Factual-recall shapes stay L2."""
    print("\nTest 3 — 'what happened' WITHOUT ordinal negative cases (K8)")

    negative = [
        "what happened at Main 42",
        "what happened yesterday",
        "what happened with the 72B server",
        "what happened in session 5",
        "what happened to the cold prefill latency",
        "what happened during M42",
        "what happened in the M125 pilot",
        "what happened",
    ]
    for q in negative:
        _check(
            f"T3.not_recency[{q!r}]",
            not mod._is_recency_query(q),
            "factual-recall 'what happened' without ordinal MUST NOT match",
        )
        _check(
            f"T3.no_short_circuit[{q!r}]",
            not mod._m120_b_should_short_circuit(q),
            "factual-recall shape must NOT route to L0",
        )
    return not any(n.startswith("T3.") for n, _ in _FAILURES)


def test_4_factual_recall_not_promoted(mod) -> bool:
    """Regression: factual-recall queries that DO NOT contain the
    'what happened' + ordinal shape MUST continue to fall through to
    downstream retrieval. M42 smoking-gun, M125 result lookups, etc."""
    print("\nTest 4 — factual-recall queries NOT promoted to recency")

    factual_cases = [
        # M42 smoking-gun style (what M125.1 D Q6 exercised).
        "in session M42 we fixed the 1/20 battery pass rate — what was the specific root cause?",
        "what was the M125 result",
        "what is the cold prefill time in Main 25",
        "what's the 1.01 ms measurement about",
        "what was the throughput at 8B Q8",
        # Near-miss shapes that mention "session" but are not ordinals.
        "what was our last session about the ANE compiler",  # NOTE: may
        # still fire short-circuit via existing 'what was our last session'
        # alternation — that's PRE-EXISTING behavior, not M125.2 C scope.
        # Explicit M125.2 scope: no false-positive introduced by new pattern.
    ]
    # For M125.2 C, only verify that the NEW pattern is not over-fitting.
    # Pre-existing recency matches are not M125.2 C scope.
    m125_2_c_negatives = [
        "what was the M125 result",
        "what is the cold prefill time in Main 25",
        "what's the 1.01 ms measurement about",
        "what was the throughput at 8B Q8",
        "in session M42 we fixed the 1/20 battery pass rate — "
        "what was the specific root cause?",
    ]
    for q in m125_2_c_negatives:
        _check(
            f"T4.not_recency[{q!r}]",
            not mod._is_recency_query(q),
            "factual-recall query must NOT match recency regex",
        )
    return not any(n.startswith("T4.") for n, _ in _FAILURES)


def test_5_m125_1_b_classifier_regression(mod) -> bool:
    """M125.1 B classifier 10/10 regression guard. Downstream
    _parse_ordinal_nback is untouched by M125.2 C; its behavior on
    the original 10 cases must be preserved. Accept both
    int (M125.1 B original) and tuple (M125.2 D extension) returns."""
    print("\nTest 5 — M125.1 B classifier 10/10 regression")

    cases = [
        ("what did we discuss three sessions ago?", 3, "session"),
        ("not last session, the two sessions before last", 2, "session"),
        ("what did we discuss 10 sessions ago?", 10, "session"),
        ("what did we discuss last session", None, None),
        ("what did we talk about last", None, None),
        ("where did we leave off", None, None),
        # Turns-scope: under M125.1 B (int return) these returned None
        # (deferred). Under M125.2 D (tuple return) they return
        # (n, 'turn'). Either behavior is acceptable regression.
        ("what did you say 1 turn ago", None, None),
        ("what did I ask 2 turns back", None, None),
        ("1 session ago what did we do", 1, "session"),
        ("what about 99 sessions ago", 99, "session"),
    ]
    for q, expected_n, expected_scope in cases:
        got = mod._parse_ordinal_nback(q)
        if expected_n is None:
            # Accept None (M125.1 B) OR any tuple starting with (n,'turn')
            # (M125.2 D). This test protects M125.1 B's rejection of
            # non-ordinal queries but is permissive about turn-scope.
            if got is None:
                _check(
                    f"T5.regression[{q!r}]",
                    True,
                    "",
                )
            elif isinstance(got, tuple) and got[1] == "turn":
                _check(
                    f"T5.regression[{q!r}]",
                    True,
                    "(M125.2 D turn-scope extension; preserved intent)",
                )
            else:
                _check(
                    f"T5.regression[{q!r}]",
                    False,
                    f"expected None or turn-scope, got {got!r}",
                )
        else:
            # Session-scope expected.
            if isinstance(got, tuple):
                n, scope = got
                _check(
                    f"T5.regression[{q!r}]",
                    n == expected_n and scope == expected_scope,
                    f"expected ({expected_n},{expected_scope}), got {got!r}",
                )
            else:
                _check(
                    f"T5.regression[{q!r}]",
                    got == expected_n,
                    f"expected {expected_n}, got {got!r}",
                )
    return not any(n.startswith("T5.") for n, _ in _FAILURES)


def test_6_m125_1_b_integration_smoke(mod) -> bool:
    """M125.1 B 3/3 integration smoke preserved: load_nth_session_summary
    returns a distinct session per ordinal."""
    print("\nTest 6 — M125.1 B 3/3 integration (distinct summary per N)")

    s1 = mod.load_nth_session_summary(1)
    s2 = mod.load_nth_session_summary(2)
    s3 = mod.load_nth_session_summary(3)
    _check(
        "T6.s1_exists",
        s1 is not None,
        "N=1 summary must load",
    )
    if s1 and s2:
        _check(
            "T6.s1_s2_distinct",
            s1.get("_path") != s2.get("_path"),
            "N=1 and N=2 must return distinct files",
        )
    if s2 and s3:
        _check(
            "T6.s2_s3_distinct",
            s2.get("_path") != s3.get("_path"),
            "N=2 and N=3 must return distinct files",
        )
    return not any(n.startswith("T6.") for n, _ in _FAILURES)


def test_7_a5_1_anaphoric_not_overmatched(mod) -> bool:
    """A5.1 anaphoric resolver regression — ambiguous anaphoric
    follow-ups that mention 'turns' or 'happened' WITHOUT explicit
    ordinal + session marker MUST NOT be promoted to recency.

    Pattern-1 guardrail from M125.2 directive §3.3: 'What did we
    discuss' without ordinal is anaphoric — don't promote to recency."""
    print("\nTest 7 — A5.1 anaphoric regression (no over-match)")

    anaphoric_cases = [
        # 'what happened' variants without ordinal-session marker.
        "what happened with that",
        "what happened there",
        "what happened next",
        # Anaphoric 'what did we' without ordinal (A5.1 territory).
        "what did we decide",
        "what did we conclude",
        # 'turns' mentioned without ordinal (A5.1 territory).
        "we made several turns of progress on this",
        "the conversation has taken several turns",
    ]
    for q in anaphoric_cases:
        _check(
            f"T7.not_recency[{q!r}]",
            not mod._is_recency_query(q),
            "anaphoric query must NOT match recency regex",
        )
        _check(
            f"T7.no_short_circuit[{q!r}]",
            not mod._m120_b_should_short_circuit(q),
            "anaphoric query must NOT short-circuit (A5.1 territory)",
        )
    return not any(n.startswith("T7.") for n, _ in _FAILURES)


def test_8_m121_c_bare_phatic_preserved(mod) -> bool:
    """M121 C bare-phatic regression: existing recency patterns must
    continue to short-circuit under the extended regex."""
    print("\nTest 8 — M121 C bare-phatic regression")

    bare_phatic_cases = [
        "where were we?",
        "where did we leave off",
        "what were we working on?",
        "what did we talk about last",
        "last session",
        "previous session",
    ]
    for q in bare_phatic_cases:
        _check(
            f"T8.short_circuits[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "M121 C bare-phatic short-circuit preserved",
        )
    return not any(n.startswith("T8.") for n, _ in _FAILURES)


def test_9_m120_b_anchor_preserved(mod) -> bool:
    """M120 B topical-anchor conjunction check preserved — anchored
    phatic queries still fall through to L2."""
    print("\nTest 9 — M120 B topical-anchor preserved")

    q_t84 = ("so where did we leave off on the direct IOKit loading "
             "thread?")
    _check(
        "T9.T84_is_recency",
        bool(mod._is_recency_query(q_t84)),
        "T84 phatic head must still match",
    )
    _check(
        "T9.T84_has_topical_anchor",
        bool(mod._m120_b_has_topical_anchor(q_t84)),
        "T84 topical anchor detected",
    )
    _check(
        "T9.T84_does_NOT_short_circuit",
        not mod._m120_b_should_short_circuit(q_t84),
        "T84 must continue to fall through to L2",
    )
    return not any(n.startswith("T9.") for n, _ in _FAILURES)


def main() -> int:
    global _FAILURES, _PASSES
    print("=" * 70)
    print("M125.2 Stream C — recency-gate 'what happened' phrasing extension")
    print("=" * 70)

    try:
        mod = _import_module()
    except Exception:
        traceback.print_exc()
        print("\nm125_2.c.verdict=deferred")
        print("m125_2.c.recency_re_extension_shipped=false")
        print("m125_2.c.q4_replay_pass=false")
        print("m125_2.c.phrasing_variant_pass_count=0")
        return 2

    for needed in ("_is_recency_query",
                   "_m120_b_has_topical_anchor",
                   "_m120_b_should_short_circuit",
                   "_parse_ordinal_nback",
                   "load_nth_session_summary",
                   "_RECENCY_QUERY_RE"):
        if not hasattr(mod, needed):
            print(f"  FATAL  missing attr {needed}")
            print("\nm125_2.c.verdict=deferred")
            print("m125_2.c.recency_re_extension_shipped=false")
            print("m125_2.c.q4_replay_pass=false")
            print("m125_2.c.phrasing_variant_pass_count=0")
            return 2

    t1 = test_1_q4_replay(mod)
    t2 = test_2_what_happened_positive(mod)
    t3 = test_3_what_happened_negative(mod)
    t4 = test_4_factual_recall_not_promoted(mod)
    t5 = test_5_m125_1_b_classifier_regression(mod)
    t6 = test_6_m125_1_b_integration_smoke(mod)
    t7 = test_7_a5_1_anaphoric_not_overmatched(mod)
    t8 = test_8_m121_c_bare_phatic_preserved(mod)
    t9 = test_9_m120_b_anchor_preserved(mod)

    total_checks = _PASSES + len(_FAILURES)
    print("\n" + "-" * 70)
    print(f"Results: {_PASSES}/{total_checks} checks passed, "
          f"{len(_FAILURES)} failures")
    if _FAILURES:
        print("\nFailures:")
        for name, detail in _FAILURES:
            print(f"  - {name}: {detail}")

    all_pass = not _FAILURES
    q4_pass = t1
    # Phrasing variants attempted: 8 positive + 8 negative = 16
    # Count each query that satisfies its expected shape.
    # Use _PASSES count across T2 + T3 to avoid manual recount.
    # A rough variant count: T2 positive covers 8 distinct phrasings,
    # T3 negative covers 8 under-fit phrasings = 16 phrasing variants.
    # We report count passed as sum of T2 + T3 ÷ checks-per-query.
    # T2 runs 3 checks per query × 8 = 24 checks (pass → 8 variants).
    # T3 runs 2 checks per query × 8 = 16 checks (pass → 8 variants).
    phrasing_variants_passed = (
        (8 if t2 else 0) + (8 if t3 else 0)
    )

    print("\n--- Registry values (parent consolidates) ---")
    print(f"m125_2.c.verdict={'shipped' if all_pass else 'deferred'}")
    print(f"m125_2.c.recency_re_extension_shipped="
          f"{'true' if (t1 and t2) else 'false'}")
    print(f"m125_2.c.q4_replay_pass={'true' if q4_pass else 'false'}")
    print(f"m125_2.c.phrasing_variant_pass_count={phrasing_variants_passed}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
