"""M121 Stream C — bare phatic regex gap closure.

Authoritative spec:
    vault/directives/in_progress/2026-04-22T12-56-46_m121_m121-synthesis-residual-diagnosis-and-ed.md
    vault/agent_reports/m120_b_phatic_anchor_fix.md

Mechanism anchor: M120 B §6 (upstream gap).
    Pre-M121-C: _RECENCY_QUERY_RE did not match bare "where were we?"
    or un-last-suffixed "what were we working on" / "what were we
    doing". Bare-phatic queries fell through to L2 semantic routing
    instead of the L0 session_summary_recall short-circuit.

Fix under test:
    _RECENCY_QUERY_RE extended with three bare-phatic alternations:
      (a) where\s+(?:were|was)\s+we
      (b) what\s+(?:were|was)\s+we\s+(working\s+on|doing)
    plus the pre-existing discussing/chatting/talking coverage via
    the "what were we talk/discuss/chat" alternation (alt #1).

    M120 B conjunction check (_m120_b_should_short_circuit) continues
    to decide whether a matched phatic short-circuits (no topical
    anchor) vs falls through to L2 (topical anchor present).

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m121_c_phatic_regex.py

Registry values produced:
    m121.c.verdict                                : shipped | deferred
    m121.c.bare_phatic_regex_active               : bool
    m121.c.m120_b_conjunction_check_preserved     : bool

m121_c_bare_phatic
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
    """Import midas_ui lazily (same pattern as M120 B test)."""
    try:
        import midas_ui  # type: ignore
        return midas_ui
    except Exception as e:  # pragma: no cover - sandbox fallback
        print(f"[m121_c] full midas_ui import failed: {e}", flush=True)
        print("[m121_c] falling back to source-extract harness",
              flush=True)
        return _fallback_extract_predicates()


def _fallback_extract_predicates():  # pragma: no cover
    """Parse midas_ui.py and exec just the phatic/M120-B helpers into
    a namespace. Used only when the full module import fails in
    constrained environments."""
    import re as _re
    import types
    src_path = os.path.join(_AGENT_DIR, "midas_ui.py")
    with open(src_path, "r") as f:
        src = f.read()
    start = src.find("_RECENCY_QUERY_RE = re.compile(")
    end_marker = "def _m120_b_should_short_circuit"
    end_idx = src.find(end_marker, start)
    tail_scan = src[end_idx:]
    next_top = _re.search(r"\n(def |class |\w+ = )", tail_scan[40:])
    if next_top is None:
        body = tail_scan
    else:
        body = tail_scan[: 40 + next_top.start()]
    snippet = "import re\n" + src[start:end_idx] + body
    ns = types.ModuleType("midas_ui_m121_c_fallback")
    exec(snippet, ns.__dict__)
    return ns


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


def test_1_bare_phatic_short_circuits(mod) -> bool:
    """Bare phatic with no topical anchor MUST short-circuit to
    session_summary_recall (recency-based handler). These are the
    patterns M120 B noted as pre-existing gaps."""
    print("\nTest 1 — bare phatic short-circuits (M121 C regex gap closure)")

    # Core directive-specified bare-phatic patterns.
    bare_cases = [
        "where were we?",
        "where were we",
        "where were we last?",
        "where did we leave off",
        "where did we leave off?",
        "what were we working on?",
        "what were we working on",
        "what were we discussing?",
        "what were we discussing",
        "what were we doing?",
        "what were we talking about?",
    ]
    for q in bare_cases:
        _check(
            f"T1.is_recency_query[{q!r}]",
            bool(mod._is_recency_query(q)),
            "bare-phatic regex must match (M121 C extension)",
        )
        _check(
            f"T1.short_circuits[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "bare phatic (no topical anchor) must short-circuit",
        )
    return not any(n.startswith("T1.") for n, _ in _FAILURES)


def test_2_anchored_falls_through(mod) -> bool:
    """Anchored phatic — M120 B conjunction check must still suppress
    the short-circuit and route via L2. Regression guard on the M120
    B fix: adding bare-phatic regex coverage must NOT defeat the
    anchor check."""
    print("\nTest 2 — anchored phatic falls through (M120 B preserved)")

    anchored_cases = [
        # Directive-specified regression targets.
        "where were we on ANE compilation?",
        "where did we leave off on the compiler work?",
        # Additional anchored variants that now go through the regex
        # (they all carry multi-word specific trailing topic).
        "where were we on the Bridge M64 paper?",
        "where did we leave off about the ANE dispatch stack?",
        ("so where did we leave off on the direct IOKit loading "
         "thread?"),
        "where did we leave off regarding the subconscious maintenance loops?",
        "where were we on the Llama 8B Q8 benchmarks?",
        "what were we working on regarding the subconscious pipeline?",
        "what were we discussing about the IOKit selectors?",
    ]
    for q in anchored_cases:
        _check(
            f"T2.is_recency_query[{q!r}]",
            bool(mod._is_recency_query(q)),
            ("phatic head must still match; conjunction check "
             "decides destination"),
        )
        _check(
            f"T2.has_topical_anchor[{q!r}]",
            bool(mod._m120_b_has_topical_anchor(q)),
            "topical anchor must be detected by M120 B check",
        )
        _check(
            f"T2.falls_through[{q!r}]",
            not mod._m120_b_should_short_circuit(q),
            "anchored phatic must fall through to L2",
        )
    return not any(n.startswith("T2.") for n, _ in _FAILURES)


def test_3_pronoun_ambiguous_anchor(mod) -> bool:
    """Pronoun-anchored phatic: "where were we on it" style. M120 B
    conjunction check treats pronouns/generic fillers as NOT
    topical — conservative rule: preserve the short-circuit. Document
    the rule; test it across the new regex surface."""
    print("\nTest 3 — pronoun/ambiguous anchors preserve short-circuit")

    # Rule (inherited from M120 B): pronoun/generic-filler tail is
    # not a topical anchor; short-circuit is preserved. This is the
    # conservative default — pronoun-only tails carry no specific
    # topic for L2 to route on.
    preserve_cases = [
        # New M121 C regex surface + pronoun/filler tail.
        "where were we on it?",
        "where were we on that?",
        "where were we with that?",
        "where were we on stuff?",
        "what were we working on that?",  # noun-tail = pronoun "that"
        "what were we doing there?",
        # Pre-existing M120 B coverage (regression guard).
        "where did we leave off with that?",
        "where did we leave off on it?",
        "where did we leave off on stuff?",
        "where did we leave off on things?",
    ]
    for q in preserve_cases:
        _check(
            f"T3.preserves_short_circuit[{q!r}]",
            mod._m120_b_should_short_circuit(q),
            "pronoun/filler anchor must preserve short-circuit",
        )
    return not any(n.startswith("T3.") for n, _ in _FAILURES)


def test_4_non_phatic_not_matched(mod) -> bool:
    """Under-fit check — regex must NOT match non-phatic query shapes.
    Under-fit beats over-fit: false positives (short-circuiting a
    non-phatic query) ship wrong templated responses to the user."""
    print("\nTest 4 — under-fit: non-phatic queries do NOT match")

    non_phatic_cases = [
        # Directive-specified under-fit targets.
        "where is the file",
        "where is the file?",
        # Genuinely non-recency "where" queries.
        "where does the model live?",
        "where is the config",
        "where does this function get called?",
        # "what" queries that are not recency-shaped.
        "what is the throughput?",
        "what is the file path?",
        "what does the ANE do?",
        # Status / current-state queries — NOT recency.
        "what's the status on ANE compilation?",
        "what is the current tok/s?",
    ]
    for q in non_phatic_cases:
        _check(
            f"T4.not_recency[{q!r}]",
            not mod._is_recency_query(q),
            "non-phatic query must NOT match recency regex",
        )
        _check(
            f"T4.no_short_circuit[{q!r}]",
            not mod._m120_b_should_short_circuit(q),
            "non-phatic query must NOT short-circuit",
        )
    return not any(n.startswith("T4.") for n, _ in _FAILURES)


def test_5_m120_b_regression_guard(mod) -> bool:
    """M120 B regression guard — T84 replay canonical. The exact
    operator-pilot query must still fall through to L2."""
    print("\nTest 5 — M120 B T84 regression guard")

    q = "so where did we leave off on the direct IOKit loading thread?"
    _check(
        "T5.T84_is_recency",
        bool(mod._is_recency_query(q)),
        "T84 phatic head must still match",
    )
    _check(
        "T5.T84_has_topical_anchor",
        bool(mod._m120_b_has_topical_anchor(q)),
        "T84 topical anchor 'the direct IOKit loading thread' detected",
    )
    _check(
        "T5.T84_does_NOT_short_circuit",
        not mod._m120_b_should_short_circuit(q),
        "T84 must continue to fall through to L2 (M120 B invariant)",
    )
    return not any(n.startswith("T5.") for n, _ in _FAILURES)


def main() -> int:
    global _FAILURES, _PASSES
    print("=" * 70)
    print("M121 Stream C — bare phatic regex gap closure")
    print("=" * 70)

    try:
        mod = _import_module()
    except Exception:
        traceback.print_exc()
        print("\nm121.c.verdict=deferred")
        print("m121.c.bare_phatic_regex_active=false")
        print("m121.c.m120_b_conjunction_check_preserved=false")
        return 2

    # Sanity: all three predicates exist.
    for needed in ("_is_recency_query",
                   "_m120_b_has_topical_anchor",
                   "_m120_b_should_short_circuit"):
        if not hasattr(mod, needed):
            print(f"  FATAL  missing attr {needed}")
            print("\nm121.c.verdict=deferred")
            print("m121.c.bare_phatic_regex_active=false")
            print("m121.c.m120_b_conjunction_check_preserved=false")
            return 2

    t1 = test_1_bare_phatic_short_circuits(mod)
    t2 = test_2_anchored_falls_through(mod)
    t3 = test_3_pronoun_ambiguous_anchor(mod)
    t4 = test_4_non_phatic_not_matched(mod)
    t5 = test_5_m120_b_regression_guard(mod)

    total_checks = _PASSES + len(_FAILURES)
    print("\n" + "-" * 70)
    print(f"Results: {_PASSES}/{total_checks} checks passed, "
          f"{len(_FAILURES)} failures")
    if _FAILURES:
        print("\nFailures:")
        for name, detail in _FAILURES:
            print(f"  - {name}: {detail}")

    all_pass = not _FAILURES
    bare_active = t1 and t4
    m120_b_preserved = t2 and t3 and t5

    print("\n--- Registry values (parent consolidates) ---")
    print(f"m121.c.verdict={'shipped' if all_pass else 'deferred'}")
    print(f"m121.c.bare_phatic_regex_active="
          f"{'true' if bare_active else 'false'}")
    print(f"m121.c.m120_b_conjunction_check_preserved="
          f"{'true' if m120_b_preserved else 'false'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
