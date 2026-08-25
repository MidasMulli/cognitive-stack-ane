"""M123 Stream A2 — response_truncation diagnose-first-then-fix.

This stream diagnosed M122 C novel-shape `response_truncation` on turns
T36/T39/T66 from `sess_20260422_104816_56211`. Verdict: DEFER the fix,
SHIP the `generation.completion_reason` ζ v2.2 field. See
`vault/agent_reports/m123_a2_response_truncation.md` for full evidence.

Test discipline (per M109 testing pattern):
  - Helper-level unit tests on `_m123_a2_completion_reason` enum mapping.
  - Replay fixtures from T36/T39/T66 via the captured `stop_reason`
    already in the turn JSONs (verifies the enum maps correctly).
  - Regression: T66's shape (stop_reason=eos, short response) must map
    to `stop_token_end`, NOT `stream_terminated` — i.e. T66 is NOT a
    truncation case. This is the load-bearing regression: M122 C
    mis-attributed T66 to the truncation bucket when it is actually a
    retrieval pool-gap abstention.
  - Backward-compat: adding `completion_reason` does NOT alter existing
    `stop_reason` / `stop_reason_loop_broke` / `stop_reason_gen_error`
    semantics.

Run:
    python3 -m pytest orion-ane/tests/test_m123_a2_response_truncation.py -v
Or:
    python3 orion-ane/tests/test_m123_a2_response_truncation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import midas_ui  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Enum mapping — the full ζ v2.2 completion_reason enum surface.
# ─────────────────────────────────────────────────────────────────────────────

def test_completion_reason_eos_maps_to_stop_token_end():
    """Natural EOS (model produced end-of-sequence) → stop_token_end."""
    assert midas_ui._m123_a2_completion_reason("eos") == "stop_token_end"


def test_completion_reason_stop_token_end_pass_through():
    """Direct stop_token_end value survives pass-through."""
    assert midas_ui._m123_a2_completion_reason("stop_token_end") == "stop_token_end"


def test_completion_reason_max_tokens_maps_to_max_tokens_reached():
    """Decode budget exhausted → max_tokens_reached."""
    assert midas_ui._m123_a2_completion_reason("max_tokens") == "max_tokens_reached"


def test_completion_reason_loop_detected_maps_to_stream_terminated():
    """Main 42 loop detector trip → stream_terminated.

    Load-bearing: T36/T39 both captured stop_reason=loop_detected. The
    completion_reason MUST surface these as stream_terminated so
    downstream pilot scoring can separate truncation-class completions
    from natural EOS without consulting an internal Main 42 field name.
    """
    assert midas_ui._m123_a2_completion_reason("loop_detected") == "stream_terminated"


def test_completion_reason_verifier_error_maps_to_stream_terminated():
    """Streaming exception → stream_terminated (caller retains partial tokens)."""
    assert midas_ui._m123_a2_completion_reason("verifier_error") == "stream_terminated"


def test_completion_reason_user_cancel_maps_to_stream_terminated():
    """Reserved user-cancel plumbing → stream_terminated."""
    assert midas_ui._m123_a2_completion_reason("user_cancel") == "stream_terminated"


def test_completion_reason_timeout_maps_to_timeout():
    """Reserved timeout signal → timeout."""
    assert midas_ui._m123_a2_completion_reason("timeout") == "timeout"


def test_completion_reason_unknown_v2_maps_to_unknown():
    """M109's unknown_v2 sentinel normalizes to the ζ v2.2 enum `unknown`."""
    assert midas_ui._m123_a2_completion_reason("unknown_v2") == "unknown"


def test_completion_reason_none_input_safe():
    """Defensive: None input (missing stop_reason) → unknown, not crash."""
    assert midas_ui._m123_a2_completion_reason(None) == "unknown"


def test_completion_reason_empty_string_input_safe():
    """Defensive: empty string → unknown."""
    assert midas_ui._m123_a2_completion_reason("") == "unknown"


def test_completion_reason_unrecognized_upstream_value():
    """Forward compat: unexpected stop_reason from future upstream server
    maps to completion_signal (not crash, not unknown). Preserves ability
    to surface novel upstream reasons into the ζ stream for later review."""
    assert (midas_ui._m123_a2_completion_reason("future_server_reason")
            == "completion_signal")


# ─────────────────────────────────────────────────────────────────────────────
# Replay: target turn fixtures from M122 C pilot session
# sess_20260422_104816_56211
#
# T36/T39: captured stop_reason=loop_detected → stream_terminated
# T66   : captured stop_reason=eos            → stop_token_end
#
# The existence of these three turn JSONs is the diagnose-first evidence
# base. If any are missing (disk rotated / test runs in a stripped repo)
# the test SKIPS rather than hard-fails — stripping the fixtures does not
# imply a regression, just a capture-state change.
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_DIR = Path(
    "/Users/midas/Desktop/cowork/data/session_logs/"
    "sess_20260422_104816_56211"
)


def _load_turn(n):
    p = _SESSION_DIR / f"turn_{n:04d}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def test_replay_t36_loop_detected_maps_to_stream_terminated():
    """T36 `detailed step by step walkthrough` query. Captured stop_reason
    is loop_detected (Main 42 in-flight detector on bullet-list enum).
    completion_reason MUST surface as stream_terminated."""
    t = _load_turn(36)
    if t is None:
        import pytest
        pytest.skip("T36 fixture unavailable")
    stop_reason = t.get("generation", {}).get("stop_reason")
    assert stop_reason == "loop_detected", (
        f"T36 regression: expected stop_reason=loop_detected, got {stop_reason!r}")
    assert (midas_ui._m123_a2_completion_reason(stop_reason)
            == "stream_terminated")


def test_replay_t39_loop_detected_maps_to_stream_terminated():
    """T39 `continue with this question` (follow-up to T36/T37 abstention).
    Same mechanism as T36 — formatting enumeration tripped the detector."""
    t = _load_turn(39)
    if t is None:
        import pytest
        pytest.skip("T39 fixture unavailable")
    stop_reason = t.get("generation", {}).get("stop_reason")
    assert stop_reason == "loop_detected", (
        f"T39 regression: expected stop_reason=loop_detected, got {stop_reason!r}")
    assert (midas_ui._m123_a2_completion_reason(stop_reason)
            == "stream_terminated")


def test_replay_t66_eos_maps_to_stop_token_end_NOT_truncation():
    """T66 `walk me through M108-M121 fixes` operator query. The M122 C
    pilot flagged this as response_truncation, but the turn JSON shows
    stop_reason=eos and response_tokens_est=36 (a short abstention, not
    a truncation). The completion_reason enum correctly surfaces this as
    stop_token_end, re-classifying T66 OUT of the truncation bucket.

    This is the load-bearing M123 A2 regression test: the new field lets
    downstream pilot scoring (and M124 scoping) distinguish pool-gap
    abstentions from mid-stream truncations."""
    t = _load_turn(66)
    if t is None:
        import pytest
        pytest.skip("T66 fixture unavailable")
    stop_reason = t.get("generation", {}).get("stop_reason")
    tokens = t.get("generation", {}).get("response_tokens_est", -1)
    assert stop_reason == "eos", (
        f"T66 regression: expected stop_reason=eos, got {stop_reason!r}")
    assert tokens < 100, (
        f"T66 short-abstention regression: expected <100 tokens, got {tokens}")
    assert (midas_ui._m123_a2_completion_reason(stop_reason)
            == "stop_token_end")


# ─────────────────────────────────────────────────────────────────────────────
# Regression: helper additions must not perturb existing M109 fields.
# ─────────────────────────────────────────────────────────────────────────────

def test_existing_m109_infer_stop_reason_unchanged_eos():
    """_m109_infer_stop_reason still returns 'eos' on the natural case."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=50, max_tokens=600,
        loop_detected=False, gen_error=None) == "eos"


def test_existing_m109_infer_stop_reason_unchanged_loop():
    """_m109_infer_stop_reason still returns 'loop_detected' on the flag."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=192, max_tokens=600,
        loop_detected=True, gen_error=None) == "loop_detected"


def test_existing_m109_infer_stop_reason_unchanged_max_tokens():
    """_m109_infer_stop_reason still returns 'max_tokens' at cap."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=600, max_tokens=600,
        loop_detected=False, gen_error=None) == "max_tokens"


def test_existing_m109_infer_stop_reason_unchanged_verifier_error():
    """_m109_infer_stop_reason still returns 'verifier_error' on exception."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=10, max_tokens=600,
        loop_detected=False, gen_error="RuntimeError: boom") == "verifier_error"


def test_completion_reason_composes_with_m109_infer():
    """End-to-end: _m109_infer_stop_reason → _m123_a2_completion_reason
    composes cleanly for all M109 return values."""
    cases = [
        (dict(tokens_decoded=50, max_tokens=600, loop_detected=False,
              gen_error=None), "stop_token_end"),
        (dict(tokens_decoded=600, max_tokens=600, loop_detected=False,
              gen_error=None), "max_tokens_reached"),
        (dict(tokens_decoded=192, max_tokens=600, loop_detected=True,
              gen_error=None), "stream_terminated"),
        (dict(tokens_decoded=10, max_tokens=600, loop_detected=False,
              gen_error="boom"), "stream_terminated"),
        # M109 returns "unknown_v2" when tokens_decoded is None (insufficient
        # signal). None-input propagates to the "unknown" enum bucket.
        (dict(tokens_decoded=None, max_tokens=None, loop_detected=False,
              gen_error=None), "unknown"),
    ]
    for kwargs, expected in cases:
        stop = midas_ui._m109_infer_stop_reason(**kwargs)
        assert midas_ui._m123_a2_completion_reason(stop) == expected, (
            f"compose failed for {kwargs} → stop={stop} → "
            f"completion_reason != {expected}")


if __name__ == "__main__":
    import traceback
    tests = [name for name in globals() if name.startswith("test_")]
    failed = 0
    passed = 0
    skipped = 0
    for name in tests:
        fn = globals()[name]
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:  # pytest.skip raises Skipped
            etype = type(e).__name__
            if etype in ("Skipped", "_pytest.outcomes.Skipped"):
                print(f"SKIP {name} — {e}")
                skipped += 1
            else:
                print(f"FAIL {name} — {etype}: {e}")
                traceback.print_exc()
                failed += 1
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
