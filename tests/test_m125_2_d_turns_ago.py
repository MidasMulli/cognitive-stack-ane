"""Tests for M125.2 Stream D — turns-ago within-session accessor.

Authoritative directive:
    vault/directives/in_progress/
      2026-04-23T18-14-19_m125_2open_m125-2-a3-relocation-full-ship-pilot.md §3.4

Extends M125.1 Stream B (session-scope ordinal N-back) to also handle
turn-scope references: "N turns ago", "N turns back", "N messages ago",
"N messages back", "N exchanges ago", "N exchanges back".

Data store (canonical within-session):
    data/session_logs/{session_id}/turn_NNNN.json

Classifier return shape:
    (n, scope) | None  where scope ∈ {'turn', 'session'}

Under-fit discipline:
    - Explicit ordinal marker required (digit or spelled 1-10).
    - Anaphoric queries that mention "turns" without a count left to
      A5.1 (regression test below).
    - "earlier in session" / "before that" deliberately NOT routed
      — M125.1 B K15 deferral to M125.3.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m125_2_d_turns_ago.py

Registry values produced:
    m125_2.d.verdict                              : shipped / deferred
    m125_2.d.turn_scope_classifier_shipped        : bool
    m125_2.d.within_session_accessor_shipped      : bool
    m125_2.d.response_assembly_shipped            : bool
    m125_2.d.t22_t23_t24_replay_pass_count        : int
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)


def _import_midas_pieces():
    """Pull the individual functions we test without booting the full
    midas_ui app (importing the module is expensive, but doing it once
    is cheap — the module doesn't listen on a port at import time).
    """
    import midas_ui as m
    return m


# ─── Fixtures ──────────────────────────────────────────────────────────

def _fake_turn_json(turn_num, user_q, resp, tools=None, route_layer="L2",
                   shape_fired="", recall_filtered=None):
    return {
        "schema_version": "2.3",
        "session_id": "sess_test_d",
        "turn_number": turn_num,
        "input": {
            "timestamp": 1700000000.0 + turn_num,
            "iso": f"2026-04-23T00:00:{turn_num:02d}",
            "query": user_q,
            "query_chars": len(user_q),
            "query_tokens_est": max(1, len(user_q) // 4),
            "time_since_last_turn_s": 1.0,
        },
        "routing": {
            "l1_match": None,
            "l2_decision": None,
            "tools_called": tools or [],
            "tools_requested_not_called": [],
            "route_layer": route_layer,
        },
        "retrieval": {
            "shape_fired": shape_fired,
            "recall_filtered": recall_filtered or [],
        },
        "generation": {
            "response_text": resp,
            "response_chars": len(resp),
        },
    }


def _make_fake_session_log(tmpdir, n_turns=5):
    """Write N fake turn_NNNN.json files and return the directory."""
    for i in range(1, n_turns + 1):
        with open(os.path.join(tmpdir, f"turn_{i:04d}.json"), "w") as fh:
            json.dump(
                _fake_turn_json(
                    turn_num=i,
                    user_q=f"query at turn {i}",
                    resp=f"response at turn {i}",
                    tools=[f"tool_{i}"],
                ),
                fh,
            )
    return tmpdir


# ─── 1. Classifier: turn-scope patterns ────────────────────────────────

def test_classifier_turn_scope_patterns():
    m = _import_midas_pieces()
    cases = [
        ("what did I ask 2 turns ago", (2, "turn")),
        ("what did you say 3 turns back", (3, "turn")),
        ("what was your response 3 messages back", (3, "turn")),
        ("5 messages ago what did we cover", (5, "turn")),
        ("what did we cover 5 exchanges ago", (5, "turn")),
        ("two exchanges back please", (2, "turn")),
        ("what did I say one turn ago", (1, "turn")),
        ("10 messages ago what was the topic", (10, "turn")),
    ]
    failures = []
    for q, expected in cases:
        got = m._parse_ordinal_nback(q)
        if got != expected:
            failures.append((q, expected, got))
    assert not failures, f"turn-scope classifier misses: {failures}"
    return len(cases)


# ─── 2. Classifier: session-scope preserved (M125.1 B regression) ──────

def test_classifier_session_scope_preserved():
    """M125.1 B shipped 10/10 on session-scope; the classifier return
    shape changed (int → tuple) but the N must stay correct, and
    scope must be 'session'. Backward-compat discipline.
    """
    m = _import_midas_pieces()
    cases = [
        ("what did we discuss three sessions ago?", (3, "session")),
        ("not last session, the two sessions before last", (2, "session")),
        ("what did we discuss 10 sessions ago?", (10, "session")),
        ("1 session ago what did we do", (1, "session")),
        ("what about 99 sessions ago", (99, "session")),
    ]
    failures = []
    for q, expected in cases:
        got = m._parse_ordinal_nback(q)
        if got != expected:
            failures.append((q, expected, got))
    assert not failures, (
        f"M125.1 B session-scope regression: {failures}"
    )


def test_classifier_no_ordinal_marker_returns_none():
    """M125.1 B 10/10 negative cases: queries without an explicit
    ordinal marker return None (fall through to existing routing)."""
    m = _import_midas_pieces()
    negative = [
        "what did we discuss last session",
        "what did we talk about last",
        "where did we leave off",
    ]
    for q in negative:
        assert m._parse_ordinal_nback(q) is None, (
            f"false positive on {q!r}: got {m._parse_ordinal_nback(q)!r}"
        )


# ─── 3. Classifier: ambiguous / K15 deferral ──────────────────────────

def test_classifier_ambiguous_earlier_in_session_not_routed():
    """K15: 'earlier in session' / 'before that' / 'previous
    response' are ambiguous between anaphoric (A5.1) and ordinal. Do
    NOT auto-route in M125.2; classifier returns None so the query
    falls through to existing layers.
    """
    m = _import_midas_pieces()
    ambiguous = [
        "earlier in session we covered X",
        "before that what did you say",
        "previous response",
        "can you repeat what you said earlier",
    ]
    for q in ambiguous:
        assert m._parse_ordinal_nback(q) is None, (
            f"K15 regression — {q!r} routed: {m._parse_ordinal_nback(q)!r}"
        )


# ─── 4. Classifier: A5.1 anaphoric resolver not regressed ─────────────

def test_classifier_a5_1_anaphoric_queries_not_promoted():
    """A5.1 owns subject-continuity queries. Anaphoric follow-ups that
    may incidentally mention 'turns' without an explicit ordinal count
    must stay None so A5.1's prior_tool fold-in still runs.

    Under-fit: require digit or spelled-1-10 + noun.
    """
    m = _import_midas_pieces()
    anaphoric = [
        "can you do that again",
        "and for the 8B",
        "what about that",
        "across several turns we saw drift",  # "turns" without count
        "many messages have gone by",          # "messages" without count
        "some exchanges earlier we decided",  # "exchanges" without count
    ]
    for q in anaphoric:
        assert m._parse_ordinal_nback(q) is None, (
            f"A5.1 regression — anaphoric {q!r} classified as: "
            f"{m._parse_ordinal_nback(q)!r}"
        )


# ─── 5. Accessor: within-session turn N returns correct record ────────

def test_accessor_returns_correct_turn():
    m = _import_midas_pieces()
    tmp = tempfile.mkdtemp(prefix="m125_2_d_accessor_")
    try:
        _make_fake_session_log(tmp, n_turns=5)
        # N=1 should be the MOST RECENT completed turn (highest num).
        rec1 = m.load_nth_turn(1, session_log_dir=tmp)
        assert rec1 is not None, "N=1 should find a turn"
        assert rec1["turn_number"] == 5, (
            f"N=1 expected turn 5 (most recent), got "
            f"{rec1['turn_number']}"
        )
        assert rec1["user_query"] == "query at turn 5"
        assert rec1["assistant_response"] == "response at turn 5"
        assert rec1["_nback"] == 1
        assert rec1["_total_available"] == 5

        # N=3 should be turn 3 (three turns ago from current).
        rec3 = m.load_nth_turn(3, session_log_dir=tmp)
        assert rec3 is not None, "N=3 should find a turn"
        assert rec3["turn_number"] == 3, (
            f"N=3 expected turn 3, got {rec3['turn_number']}"
        )
        assert rec3["user_query"] == "query at turn 3"

        # N=5 should be the oldest.
        rec5 = m.load_nth_turn(5, session_log_dir=tmp)
        assert rec5 is not None
        assert rec5["turn_number"] == 1

        # N=6 exceeds history → None.
        rec6 = m.load_nth_turn(6, session_log_dir=tmp)
        assert rec6 is None, "N>history should return None"

        # N=0 / negative should be None.
        assert m.load_nth_turn(0, session_log_dir=tmp) is None
        assert m.load_nth_turn(-1, session_log_dir=tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_accessor_missing_dir_returns_none():
    m = _import_midas_pieces()
    assert m.load_nth_turn(1, session_log_dir="/nonexistent/path") is None


# ─── 6. Response assembly: turn-level header distinct from session ────

def test_response_assembly_turn_level_header_distinct():
    m = _import_midas_pieces()
    tmp = tempfile.mkdtemp(prefix="m125_2_d_resp_")
    try:
        _make_fake_session_log(tmp, n_turns=4)
        rec = m.load_nth_turn(2, session_log_dir=tmp)
        resp = m._build_turn_recency_response(rec)
        assert "2 turns ago" in resp.lower(), (
            f"expected '2 turns ago' header, got: {resp[:200]!r}"
        )
        assert "you asked" in resp.lower(), (
            f"expected 'you asked' phrasing, got: {resp[:200]!r}"
        )
        assert "i responded" in resp.lower(), (
            f"expected 'I responded' phrasing, got: {resp[:200]!r}"
        )
        # Turn-level must NOT say "our last session ended" (the
        # session-level header).
        assert "last session ended" not in resp.lower(), (
            f"turn header leaked session template: {resp[:200]!r}"
        )
        assert "sessions ago" not in resp.lower(), (
            f"turn header used session-scope phrasing: {resp[:200]!r}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_response_assembly_missing_record_graceful():
    m = _import_midas_pieces()
    resp = m._build_turn_recency_response(None)
    assert resp, "missing-record response should not be empty"
    assert "don't have" in resp.lower() or "no prior" in resp.lower() \
        or "exceeds" in resp.lower(), (
            f"expected graceful missing-record message, got: {resp!r}"
        )


# ─── 7. M125.1 B 3/3 integration smoke preserved ──────────────────────

def test_m125_1_b_integration_smoke_preserved():
    """The 3/3 live summary smoke from M125.1 B must still work:
    three distinct ordinal N values must load distinct summaries.
    Verifies load_nth_session_summary untouched.
    """
    m = _import_midas_pieces()
    summaries_dir = "/Users/midas/Desktop/cowork/data/session_summaries"
    if not os.path.isdir(summaries_dir):
        # If the corpus is missing, skip rather than fail — the
        # classifier + accessor tests are the load-bearing checks.
        return
    # Count available summaries; cap test at that count.
    n_avail = len([
        f for f in os.listdir(summaries_dir)
        if f.startswith("session_") and f.endswith(".json")
    ])
    if n_avail < 3:
        return
    s1 = m.load_nth_session_summary(1)
    s2 = m.load_nth_session_summary(2)
    s3 = m.load_nth_session_summary(3)
    assert s1 and s2 and s3, "expected three summaries to load"
    paths = {s1["_path"], s2["_path"], s3["_path"]}
    assert len(paths) == 3, (
        f"M125.1 B regression: three N values should load three "
        f"distinct summaries, got {paths}"
    )


# ─── 8. T22/T23/T24-class replay on turn-scope queries ────────────────

def test_t22_t23_t24_class_replay():
    """Representative turn-ago queries per directive §3.4 Phase 6.
    Each should classify as (N, 'turn') AND, when paired with a
    fake session log, return the correct turn record.
    """
    m = _import_midas_pieces()
    tmp = tempfile.mkdtemp(prefix="m125_2_d_t22_")
    try:
        _make_fake_session_log(tmp, n_turns=10)
        cases = [
            ("what did I ask 2 turns ago", 2),
            ("what was your response 3 messages back", 3),
            ("what did we cover 5 exchanges ago", 5),
        ]
        pass_count = 0
        for q, expected_n in cases:
            parsed = m._parse_ordinal_nback(q)
            assert parsed is not None, (
                f"T22/T23/T24 replay failed — classifier returned "
                f"None for {q!r}"
            )
            got_n, scope = parsed
            assert got_n == expected_n, (
                f"T22/T23/T24 N mismatch for {q!r}: "
                f"expected {expected_n}, got {got_n}"
            )
            assert scope == "turn", (
                f"T22/T23/T24 scope mismatch for {q!r}: "
                f"expected 'turn', got {scope!r}"
            )
            rec = m.load_nth_turn(got_n, session_log_dir=tmp)
            assert rec is not None, (
                f"T22/T23/T24 accessor miss for {q!r}"
            )
            expected_turn = 10 - (got_n - 1)
            assert rec["turn_number"] == expected_turn, (
                f"T22/T23/T24 record mismatch for {q!r}: "
                f"expected turn_number {expected_turn}, got "
                f"{rec['turn_number']}"
            )
            pass_count += 1
        assert pass_count == 3, (
            f"T22/T23/T24 replay only {pass_count}/3"
        )
        return pass_count
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 9. Turn-scope gate predicate ─────────────────────────────────────

def test_turn_scope_gate_predicate():
    m = _import_midas_pieces()
    positive = [
        "what did I ask 2 turns ago",
        "3 messages back",
        "5 exchanges ago",
    ]
    for q in positive:
        assert m._is_turn_ordinal_query(q), (
            f"turn-scope gate should fire on {q!r}"
        )
    negative = [
        "what did we discuss three sessions ago",
        "where did we leave off",
        "what about that",
        "earlier in session",
    ]
    for q in negative:
        assert not m._is_turn_ordinal_query(q), (
            f"turn-scope gate should NOT fire on {q!r}"
        )


# ─── Runner ────────────────────────────────────────────────────────────

def _run():
    results = []
    tests = [
        ("classifier_turn_scope_patterns",
         test_classifier_turn_scope_patterns),
        ("classifier_session_scope_preserved",
         test_classifier_session_scope_preserved),
        ("classifier_no_ordinal_marker_returns_none",
         test_classifier_no_ordinal_marker_returns_none),
        ("classifier_ambiguous_earlier_in_session_not_routed",
         test_classifier_ambiguous_earlier_in_session_not_routed),
        ("classifier_a5_1_anaphoric_queries_not_promoted",
         test_classifier_a5_1_anaphoric_queries_not_promoted),
        ("accessor_returns_correct_turn",
         test_accessor_returns_correct_turn),
        ("accessor_missing_dir_returns_none",
         test_accessor_missing_dir_returns_none),
        ("response_assembly_turn_level_header_distinct",
         test_response_assembly_turn_level_header_distinct),
        ("response_assembly_missing_record_graceful",
         test_response_assembly_missing_record_graceful),
        ("m125_1_b_integration_smoke_preserved",
         test_m125_1_b_integration_smoke_preserved),
        ("t22_t23_t24_class_replay",
         test_t22_t23_t24_class_replay),
        ("turn_scope_gate_predicate",
         test_turn_scope_gate_predicate),
    ]
    for name, fn in tests:
        try:
            ret = fn()
            results.append((name, "PASS", ret))
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            results.append((name, "ERROR", f"{type(e).__name__}: {e}"))
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_total = len(results)
    print(f"\nM125.2 Stream D — turns-ago accessor tests\n")
    for name, status, detail in results:
        tag = "[PASS]" if status == "PASS" else f"[{status}]"
        print(f"  {tag} {name}")
        if status != "PASS":
            print(f"         {detail}")
    print(f"\n  {n_pass}/{n_total} passed\n")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(_run())
