"""M120 Stream E — Enumeration shape dispatcher tests.

Scope: behavioral shell tests for the shape classifier only. Tag-query
execution for unbacked tags (published_repo, standing_rule) is K9
architectural gap — see vault/agent_reports/m120_e_enumeration_dispatcher.md.

Run:
  ~/.mlx-env/bin/python3 orion-ane/tests/test_m120_e_enumeration_dispatch.py

Exit code 0 = all pass, non-zero = failure (including per-test trace).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT = os.path.normpath(os.path.join(_HERE, "..", "agent"))
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from enumeration_retrieval import (  # noqa: E402
    shape_dispatch_signal,
    enumerate_by_tag,
    _ENUM_INTENT,
    _TAG_KEYWORDS,
    _UNBACKED_TAGS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _shape_of(q: str) -> str:
    """Emulate midas_ui.py Phase 2a dispatch: if shape_dispatch_signal
    matches, shape_fired='enumeration'; otherwise the caller falls through
    to narrative/default_recall paths."""
    sig = shape_dispatch_signal(q)
    if sig:
        return "enumeration"
    return "default_recall"


# ── Test cases ─────────────────────────────────────────────────────────────

def test_t88_replay():
    """T88 replay: 'enumerate all published repos under my name' fires
    Enumeration shape. Post-M121-B: published_repo is now backed by
    tag_index.json (CLAUDE.md Main 26 canonical line), so backed=True and
    records are expected to return."""
    q = "enumerate all published repos under my name"
    sig = shape_dispatch_signal(q)
    assert sig is not None, f"no shape matched for T88 query: {q!r}"
    assert sig["shape"] == "enumeration", f"wrong shape: {sig}"
    assert sig["tag"] == "published_repo", f"wrong tag: {sig}"
    assert sig["backed"] is True, f"published_repo unexpectedly unbacked: {sig}"
    assert _shape_of(q) == "enumeration"


def test_plural_noun_standing_rules():
    """Plural-noun variant: 'what are the standing rules?' fires
    Enumeration. Post-M121-B: standing_rule is now backed by tag_index.json
    (vault/knowledge/standing_rules.md A+B sections)."""
    q = "what are the standing rules?"
    sig = shape_dispatch_signal(q)
    assert sig is not None, f"no shape matched: {q!r}"
    assert sig["shape"] == "enumeration"
    assert sig["tag"] == "standing_rule"
    assert sig["backed"] is True
    assert _shape_of(q) == "enumeration"


def test_regression_singular_standing_rule():
    """Regression: singular 'what is the standing rule about X' must NOT
    fire Enumeration. This is point-retrieval, not enumeration."""
    q = "what is the standing rule about X"
    sig = shape_dispatch_signal(q)
    assert sig is None, f"false Enumeration trigger: {sig}"
    assert _shape_of(q) == "default_recall"


def test_list_imperative_dead_paths():
    """'list all dead paths' fires Enumeration AND resolves to backed tag
    (dead_path is in tag_index.json). Full round-trip via enumerate_by_tag."""
    q = "list all dead paths"
    sig = shape_dispatch_signal(q)
    assert sig is not None
    assert sig["shape"] == "enumeration"
    assert sig["tag"] == "dead_path"
    assert sig["backed"] is True, "dead_path should be backed"
    result = enumerate_by_tag(q)
    assert result is not None, "backed tag should return records"
    assert result["tag"] == "dead_path"
    assert result["count"] > 0


# ── Additional regression / widening coverage ──────────────────────────────

def test_widening_list_the_plural():
    """'list the repositories' widening — 'list the {X}s' form."""
    q = "list the repositories"
    sig = shape_dispatch_signal(q)
    assert sig is not None, "widening miss: 'list the Xs'"
    assert sig["shape"] == "enumeration"
    assert sig["tag"] == "published_repo"


def test_widening_full_list_of():
    """'give me the full list of measurements' widening."""
    q = "give me the full list of measurements"
    sig = shape_dispatch_signal(q)
    assert sig is not None, "widening miss: 'give me the full list of'"
    assert sig["shape"] == "enumeration"
    assert sig["tag"] == "measurement"


def test_regression_what_is():
    """Regression: 'what is X' is point-retrieval, never Enumeration."""
    for q in ["what is the answer", "what is a decision",
              "what is the measurement we took"]:
        sig = shape_dispatch_signal(q)
        assert sig is None, f"false Enumeration trigger on point-query: {q!r} -> {sig}"


def test_regression_greeting():
    """Regression: casual conversation never fires Enumeration."""
    for q in ["how are you?", "hi there", "what's up"]:
        sig = shape_dispatch_signal(q)
        assert sig is None, f"false Enumeration trigger on phatic: {q!r} -> {sig}"


# ── Runner ─────────────────────────────────────────────────────────────────

_TESTS = [
    test_t88_replay,
    test_plural_noun_standing_rules,
    test_regression_singular_standing_rule,
    test_list_imperative_dead_paths,
    test_widening_list_the_plural,
    test_widening_full_list_of,
    test_regression_what_is,
    test_regression_greeting,
]


def _run() -> int:
    failed = 0
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
