"""Tests for m116 Stream C: memory-commit imperative misclassification.

Authoritative spec: M116 Stream C directive, M114 A2 K4 pilot finding.
Fix targets:
    L1 router — recognize "commit this to memory", "i want you to remember",
                "i have made a statement" and route to memory_ingest.
    Stream-path bypass — midas_ui.py /api/chat/stream now has the same
                hardcoded ack for memory_ingest that /api/chat has. Keeps
                LLM out of the storage-ack path.

Pilot primary sources:
    data/session_logs/sess_20260421_143211_70559/turn_0037.json
    data/session_logs/sess_20260421_143211_70559/turn_0038.json

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m116_instruction_misclassification.py

Registry values produced:
    m116.c.verdict                          : SHIP/DEFER
    m116.c.fix_shape_chosen                 : a
    m116.c.t37_t38_replay_pass              : 1/0
    m116.c.regression_non_question_non_route: 1/0
    m116.c.regression_question_not_ingest   : 1/0
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from router import layer1_route  # noqa: E402


# ── Pilot replay cases ────────────────────────────────────────────────────

T37_QUERY = 'commit this to memory - "we do not characterize without purpose"'
T38_QUERY = 'i have made a statement that i want you to remember'


def test_t37_memory_commit_routes_to_ingest():
    """T37: 'commit this to memory - X' must route to memory_ingest at L1."""
    result = layer1_route(T37_QUERY)
    assert result is not None, (
        f"T37 query fell through L1: {T37_QUERY!r} — should match "
        f"'commit this to memory' keyword")
    tool, args = result
    assert tool == "memory_ingest", (
        f"T37 routed to {tool!r}, expected memory_ingest")
    assert args.get("role") == "user"
    assert args.get("text") == T37_QUERY
    print(f"  PASS T37 → {tool} (args={args!r})")


def test_t38_statement_to_remember_routes_to_ingest():
    """T38: 'i have made a statement that i want you to remember' → ingest."""
    result = layer1_route(T38_QUERY)
    assert result is not None, (
        f"T38 query fell through L1: {T38_QUERY!r} — should match "
        f"'i have made a statement' or 'i want you to remember' keyword")
    tool, args = result
    assert tool == "memory_ingest", (
        f"T38 routed to {tool!r}, expected memory_ingest")
    assert args.get("role") == "user"
    assert args.get("text") == T38_QUERY
    print(f"  PASS T38 → {tool} (args={args!r})")


# ── Regression: non-questions should NOT route to memory_ingest ──────────

def test_regression_short_acks_do_not_route():
    """'yes', 'ok', '' should not match memory_ingest (L1 returns None)."""
    for q in ["yes", "ok", "thanks", "cool", ""]:
        result = layer1_route(q)
        if result is not None:
            tool, _ = result
            assert tool != "memory_ingest", (
                f"Short ack {q!r} misrouted to memory_ingest")
        print(f"  PASS short-ack {q!r} → {'None' if result is None else result[0]}")


def test_regression_questions_do_not_ingest():
    """Actual questions should NOT route to memory_ingest."""
    cases = [
        "what is our current ANE throughput?",
        "how many memories are in the store?",
        "did we build this?",
        "is Gemma 4 in production?",
    ]
    for q in cases:
        result = layer1_route(q)
        if result is not None:
            tool, _ = result
            assert tool != "memory_ingest", (
                f"Question {q!r} misrouted to memory_ingest → {tool}")
        print(f"  PASS question {q!r} → {'None' if result is None else result[0]}")


# ── Regression: previously-supported memory-commit shapes still work ─────

def test_regression_legacy_memory_commit_still_routes():
    """Prior-art keywords 'remember this', 'save this', etc. must still route."""
    cases = [
        "remember this: X happened",
        "remember that the ANE has 111 GB/s",
        "save this for later",
        "note that Gemma swapped in M52",
        "store this in memory",
    ]
    for q in cases:
        result = layer1_route(q)
        assert result is not None, f"Legacy keyword failed: {q!r}"
        tool, _ = result
        assert tool == "memory_ingest", (
            f"Legacy keyword {q!r} routed to {tool}, expected memory_ingest")
        print(f"  PASS legacy {q!r} → {tool}")


# ── Additional imperative shapes from the fix ────────────────────────────

def test_new_imperative_shapes():
    """New m116 keywords should route correctly."""
    cases = [
        ("commit to memory: the ANE ran at 111 GB/s", "commit to memory"),
        ("add this to memory please", "add this to memory"),
        ("i want you to remember the bridge dims are 4096 and 5376",
         "i want you to remember"),
        ("log this: test session started", "log this"),
        ("record that we shipped M113 gamma", "record that"),
    ]
    for q, kw in cases:
        result = layer1_route(q)
        assert result is not None, f"New keyword fell through: {q!r} (kw={kw})"
        tool, _ = result
        assert tool == "memory_ingest", (
            f"New keyword {q!r} routed to {tool}, expected memory_ingest")
        print(f"  PASS new-shape {kw!r} → {tool}")


# ── Runner ───────────────────────────────────────────────────────────────

def _run():
    tests = [
        ("t37_pilot_replay", test_t37_memory_commit_routes_to_ingest),
        ("t38_pilot_replay", test_t38_statement_to_remember_routes_to_ingest),
        ("regression_short_acks", test_regression_short_acks_do_not_route),
        ("regression_questions", test_regression_questions_do_not_ingest),
        ("regression_legacy", test_regression_legacy_memory_commit_still_routes),
        ("new_imperative_shapes", test_new_imperative_shapes),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append((name, f"{type(e).__name__}: {e}"))
    print()
    print(f"Result: {passed}/{len(tests)} passed")
    if failed:
        print("Failures:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    # Registry values
    t37_t38_pass = 1 if all(
        n in [p for p, _ in [(tn, None) for tn in ("t37_pilot_replay",
                                                    "t38_pilot_replay")]]
        and n not in [f[0] for f in failed]
        for n in ("t37_pilot_replay", "t38_pilot_replay")) else 0
    # Simpler: t37+t38 pass iff neither is in failed list
    _failed_names = {f[0] for f in failed}
    t37_t38_pass = 0 if ("t37_pilot_replay" in _failed_names
                          or "t38_pilot_replay" in _failed_names) else 1
    reg_nonq = 0 if "regression_short_acks" in _failed_names else 1
    reg_q = 0 if "regression_questions" in _failed_names else 1
    verdict = "SHIP" if not failed else "DEFER"
    print()
    print("Registry values:")
    print(f"  m116.c.verdict                          : {verdict}")
    print(f"  m116.c.fix_shape_chosen                 : a")
    print(f"  m116.c.t37_t38_replay_pass              : {t37_t38_pass}")
    print(f"  m116.c.regression_non_question_non_route: {reg_nonq}")
    print(f"  m116.c.regression_question_not_ingest   : {reg_q}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run())
