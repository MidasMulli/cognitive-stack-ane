"""M120 Stream B — L1 phatic-anchor conjunction check.

Authoritative spec:
    vault/directives/in_progress/2026-04-22T12-30-20_m120_m120-catalog-refinement-routing-retrieva.md
    vault/agent_reports/m119_parent_synthesis.md
    vault/agent_reports/m119_b_round2_verdicts.md (T84)

Mechanism anchor: M119 T84.
    Query: "so where did we leave off on the direct IOKit loading thread?"
    Observed: L1 phatic matcher fired `recency_short_circuit` →
    `session_summary_recall`, bypassing L2. Templated session-summary
    response unrelated to the specific topical anchor
    "direct IOKit loading thread".

Fix under test:
    `_m120_b_should_short_circuit` wraps `_is_recency_query` with a
    topical-anchor conjunction check. Phatic-recency queries that
    carry a trailing topic-marker + multi-word specific phrase
    (on/about/for/regarding/re: + >=2 content tokens, or a single
    specific token with digit / internal caps / length >= 6) fall
    through to L2 routing. Bare phatic queries and pronoun/common-noun
    tails still short-circuit (fast path preserved).

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m120_b_routing_phatic_anchor.py

Registry values produced by this suite (written to stdout; parent
consolidates into measurement_registry.json):
    m120.b.verdict                              : shipped | deferred
    m120.b.phatic_fallthrough_active            : bool
    m120.b.t84_replay_pass                      : bool
    m120.b.regression_plain_phatic_preserved    : bool

m120_b_phatic_anchor_check
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)


def _import_module():
    """Import midas_ui lazily to isolate import-time failures (agent
    module loads many heavy deps). We only need the two predicates
    and the phatic regex, but importing the whole module is cheap
    and closest to production behavior.

    If the heavyweight import fails (missing optional deps in a
    sandbox environment), fall back to parsing the source file and
    exec'ing just the phatic section — but this stays a last resort.
    """
    try:
        import midas_ui  # type: ignore
        return midas_ui
    except Exception as e:  # pragma: no cover - sandbox fallback
        print(f"[m120_b] full midas_ui import failed: {e}", flush=True)
        print("[m120_b] falling back to source-extract harness",
              flush=True)
        return _fallback_extract_predicates()


def _fallback_extract_predicates():  # pragma: no cover
    """Parse midas_ui.py and exec just the _RECENCY_QUERY_RE and the
    _m120_b_* helpers into a namespace. Used only when the full
    module import fails in constrained environments."""
    import re as _re
    import types
    src_path = os.path.join(_AGENT_DIR, "midas_ui.py")
    with open(src_path, "r") as f:
        src = f.read()
    # Extract the regex + helpers by finding the two anchor markers.
    start = src.find("_RECENCY_QUERY_RE = re.compile(")
    end_marker = "def _m120_b_should_short_circuit"
    end_idx = src.find(end_marker, start)
    # Capture full _m120_b_should_short_circuit body by scanning for
    # the next top-level def/class after it.
    tail_scan = src[end_idx:]
    next_top = _re.search(r"\n(def |class |\w+ = )", tail_scan[40:])
    if next_top is None:
        body = tail_scan
    else:
        body = tail_scan[: 40 + next_top.start()]
    snippet = "import re\n" + src[start:end_idx] + body
    ns = types.ModuleType("midas_ui_m120_b_fallback")
    exec(snippet, ns.__dict__)
    return ns


# ---------- Test harness ----------

_FAILURES = []
_PASSES = 0
_FIXTURE_PATH = os.path.join(
    _REPO_ROOT,
    "data", "session_logs",
    "sess_20260421_202025_96397", "turn_0084.json",
)


def _check(name: str, cond: bool, detail: str = "") -> None:
    global _PASSES
    if cond:
        _PASSES += 1
        print(f"  PASS  {name}")
    else:
        _FAILURES.append((name, detail))
        print(f"  FAIL  {name}  — {detail}")


def test_1_t84_replay(mod) -> bool:
    """T84 replay — the exact M119 operator-pilot query must fall
    through to L2 (not short-circuit to session_summary_recall)."""
    print("\nTest 1 — T84 replay (operator pilot n=34)")
    q = "so where did we leave off on the direct IOKit loading thread?"

    # If the fixture exists, sanity-check we're replaying the same query.
    fixture_query = None
    if os.path.exists(_FIXTURE_PATH):
        try:
            with open(_FIXTURE_PATH, "r") as f:
                fx = json.load(f)
            fixture_query = (fx.get("input") or {}).get("query")
        except Exception:
            pass
    if fixture_query:
        _check(
            "T1.fixture_matches",
            fixture_query.strip() == q,
            f"fixture query was {fixture_query!r}",
        )
    else:
        print(f"  note  fixture {_FIXTURE_PATH} not present; synthetic only")

    _check(
        "T1.is_recency_query",
        bool(mod._is_recency_query(q)),
        "phatic head must still match; conjunction check suppresses",
    )
    _check(
        "T1.has_topical_anchor",
        bool(mod._m120_b_has_topical_anchor(q)),
        "topical anchor 'the direct IOKit loading thread' must be detected",
    )
    _check(
        "T1.does_NOT_short_circuit",
        not mod._m120_b_should_short_circuit(q),
        "T84 must fall through to L2 (this is the fix)",
    )
    return not any(n.startswith("T1.") for n, _ in _FAILURES)


def test_2_regression_plain_phatic(mod) -> bool:
    """Regression — bare phatic with no topical anchor still
    short-circuits (fast path preserved)."""
    print("\nTest 2 — regression: plain phatic preserves fast path")

    # Cases covered by the existing _RECENCY_QUERY_RE (unchanged by
    # M120 Stream B). The bare-"where were we?" case is a pre-existing
    # regex gap — it does NOT match _RECENCY_QUERY_RE today, so it
    # never short-circuited to begin with. Logged as an M121+ scope
    # recommendation (see Stream B report); out of scope here because
    # the M120 directive forbids router refactors.
    bare_cases = [
        "where did we leave off?",
        "where did we leave off",
        "what did we talk about last?",
        "what was our last session?",
        "last session",
        "what did we chat about last?",
    ]
    for q in bare_cases:
        _check(
            f"T2.short_circuits[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "bare phatic must still short-circuit",
        )

    # M121 Stream C closed the "where were we" bare-phatic gap by
    # extending _RECENCY_QUERY_RE. Post-M121-C, bare "where were we?"
    # matches the regex AND (no topical anchor) short-circuits. The
    # canary assertion now verifies the post-fix behavior; the M120 B
    # conjunction check continues to suppress the short-circuit when a
    # topical anchor is present (verified in Test 3).
    post_m121c_bare = "where were we?"
    _check(
        "T2.m121c_bare_where_were_we_short_circuits",
        mod._m120_b_should_short_circuit(post_m121c_bare),
        ("post-M121-C: bare 'where were we?' must short-circuit to "
         "session_summary_recall (regex extended, no topical anchor)."),
    )
    return not any(n.startswith("T2.") for n, _ in _FAILURES)


def test_3_topical_anchor_variations(mod) -> bool:
    """Topical anchor variations — each should fall through."""
    print("\nTest 3 — topical anchor variations fall through to L2")

    fall_through_cases = [
        "where were we on the Bridge M64 paper?",
        "where did we leave off about the ANE dispatch stack?",
        ("so where did we leave off on the direct IOKit loading "
         "thread?"),
        "where did we leave off regarding the subconscious maintenance loops?",
        "where were we on the Llama 8B Q8 benchmarks?",
    ]
    for q in fall_through_cases:
        _check(
            f"T3.falls_through[{q!r}]",
            not mod._m120_b_should_short_circuit(q),
            "topical anchor present; must route via L2",
        )

    # Unrelated non-phatic query — routes normally (no short-circuit,
    # not because we suppressed one, but because phatic never fired).
    unrelated = "what's the status on ANE compilation?"
    _check(
        "T3.unrelated_not_phatic",
        not mod._is_recency_query(unrelated),
        "non-phatic query must not match recency regex at all",
    )
    _check(
        "T3.unrelated_no_short_circuit",
        not mod._m120_b_should_short_circuit(unrelated),
        "unrelated query must not short-circuit",
    )
    return not any(n.startswith("T3.") for n, _ in _FAILURES)


def test_4_ambiguous_anchor(mod) -> bool:
    """Ambiguous anchor — pronouns and generic fillers preserve the
    short-circuit (conservative default)."""
    print("\nTest 4 — ambiguous / pronoun anchors preserve short-circuit")

    # Only cases where _RECENCY_QUERY_RE actually matches today
    # (bare "where were we" is a pre-existing regex gap — see Test 2).
    preserve_cases = [
        "where did we leave off with that?",
        "where did we leave off on it?",
        "where did we leave off on stuff?",
        "where did we leave off on things?",
        "where did we leave off on that?",
    ]
    for q in preserve_cases:
        _check(
            f"T4.preserves_short_circuit[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "pronoun/filler anchor must preserve short-circuit",
        )
    return not any(n.startswith("T4.") for n, _ in _FAILURES)


def main() -> int:
    global _FAILURES, _PASSES
    print("=" * 70)
    print("M120 Stream B — L1 phatic-anchor conjunction check")
    print("=" * 70)

    try:
        mod = _import_module()
    except Exception:
        traceback.print_exc()
        print("\nm120.b.verdict=deferred")
        print("m120.b.phatic_fallthrough_active=false")
        print("m120.b.t84_replay_pass=false")
        print("m120.b.regression_plain_phatic_preserved=false")
        return 2

    # Sanity: both predicates exist.
    for needed in ("_is_recency_query",
                   "_m120_b_has_topical_anchor",
                   "_m120_b_should_short_circuit"):
        if not hasattr(mod, needed):
            print(f"  FATAL  missing attr {needed}")
            print("\nm120.b.verdict=deferred")
            print("m120.b.phatic_fallthrough_active=false")
            print("m120.b.t84_replay_pass=false")
            print("m120.b.regression_plain_phatic_preserved=false")
            return 2

    t1 = test_1_t84_replay(mod)
    t2 = test_2_regression_plain_phatic(mod)
    t3 = test_3_topical_anchor_variations(mod)
    t4 = test_4_ambiguous_anchor(mod)

    total_checks = _PASSES + len(_FAILURES)
    print("\n" + "-" * 70)
    print(f"Results: {_PASSES}/{total_checks} checks passed, "
          f"{len(_FAILURES)} failures")
    if _FAILURES:
        print("\nFailures:")
        for name, detail in _FAILURES:
            print(f"  - {name}: {detail}")

    # Registry values (stream B parent consolidates into the registry).
    all_pass = not _FAILURES
    print("\n--- Registry values (parent consolidates) ---")
    print(f"m120.b.verdict={'shipped' if all_pass else 'deferred'}")
    print(f"m120.b.phatic_fallthrough_active="
          f"{'true' if all_pass else 'false'}")
    print(f"m120.b.t84_replay_pass={'true' if t1 else 'false'}")
    print(f"m120.b.regression_plain_phatic_preserved="
          f"{'true' if t2 else 'false'}")
    print(f"m120.b.topical_variations_pass={'true' if t3 else 'false'}")
    print(f"m120.b.ambiguous_anchor_preserved={'true' if t4 else 'false'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
