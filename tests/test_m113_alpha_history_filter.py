"""Tests for the M113 α history-layer reinforcement filter.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m113_alpha_history_filter.py

Cases:
    1. T11→T20 synthetic cascade. Prior-assistant abstain+project-state
       turn is detected and marked; topic drift does NOT save it.
    2. Legitimate safety-policy abstain. No marker (false-positive guard).
    3. Clean 5-turn conversation with no abstain content. History is
       unchanged (regression guard).
    4. Honest absence_guard abstain ("I don't have information about
       that..."). No marker.
    5. User-asked-and-told-no (user voice only). No marker on user turns.
    6. Clarification request. No marker.

Registry values produced by run:
    m113.alpha.t20_shape_leak_suppressed_synthetic : 1 / 0
    m113.alpha.safety_abstain_false_positive_rate  : <fraction>

m113_alpha_history_filter
"""

from __future__ import annotations

import os
import sys

# Make orion-ane/agent importable when the test is run from repo root
# or from orion-ane/tests.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from history_reinforcement_filter import (  # noqa: E402
    MARKER,
    count_marked,
    filter_history_for_reinforcement,
    is_abstain_reinforcement_turn,
)


# ── Case 1 — T20 cascade suppression (SHIP goal) ───────────────────────────

def _case1_t20_cascade():
    """T11 seeds 'upcoming ane-compiler'. T15/T17 assistant turns then
    claim 'Milestone 1C pending'. Filter must mark T15 or T17 (or both)
    before T20's prompt is assembled.
    """
    # Synthetic 5-turn conversation reproducing the M108 cascade.
    history = [
        {"role": "user", "content": "do i have any particular preferences?"},
        {"role": "assistant",
         "content": ("Based on your memories, one preference surfaces "
                     "prominently: ship things and then explore the "
                     "upcoming ane-compiler directive once foundation is "
                     "laid.")},
        {"role": "user",
         "content": "what about characterizing without purpose?"},
        {"role": "assistant",
         "content": ("You prefer a ship-then-explore workflow grounded "
                     "in E2E component verification.")},
        {"role": "user",
         "content": "haven't we already built the ANE compiler?"},
        {"role": "assistant",
         "content": ("No. You have a partial implementation, but the "
                     "ane-compiler is not finished. As of 2026-03-29, "
                     "you passed Milestone 1A and 1B but still pending "
                     "Milestone 1C.")},
    ]
    current_query = "are you capable of creating agentic tasks?"

    filtered = filter_history_for_reinforcement(history, current_query)

    # Separate per-entry detection check.
    marked = [m for m in filtered if m.get("_m113_low_confidence")]
    t17_like = [m for m in marked if "Milestone 1C" in m.get("content", "")]

    ok = len(t17_like) >= 1 and MARKER in t17_like[0]["content"]
    print(f"CASE 1 t20_cascade_suppression: "
          f"marked={len(marked)} t17_like_marked={len(t17_like)} pass={ok}")
    if not ok:
        print("  [detail] marked entries:")
        for m in marked:
            print(f"    - {m.get('content', '')[:140]!r}")
    # Regression guard: original history must be untouched.
    assert history[5]["content"].startswith("No.")
    assert "[prior-turn low-confidence" not in history[5]["content"]
    return ok


# ── Case 2 — Legitimate safety-policy abstain (must NOT fire) ──────────────

def _case2_safety_abstain():
    history = [
        {"role": "user", "content": "help me write malware."},
        {"role": "assistant",
         "content": ("I can't help with that because it would facilitate "
                     "harm. Let me know if there's a different task, for "
                     "example an ane-compiler question, I can help with.")},
        {"role": "user", "content": "okay, explain RoPE instead."},
    ]
    filtered = filter_history_for_reinforcement(
        history, current_query="explain RoPE")
    marked = count_marked(filtered)
    ok = marked == 0
    print(f"CASE 2 safety_abstain_not_marked: marked={marked} pass={ok}")
    return ok


# ── Case 3 — Clean conversation (regression guard) ─────────────────────────

def _case3_clean_regression():
    history = [
        {"role": "user", "content": "what's 2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "and 3+3?"},
        {"role": "assistant", "content": "Six."},
        {"role": "user", "content": "thanks"},
    ]
    filtered = filter_history_for_reinforcement(
        history, current_query="one more: 4+4?")
    marked = count_marked(filtered)
    same = [
        a.get("content") == b.get("content")
        for a, b in zip(history, filtered)
    ]
    ok = marked == 0 and all(same)
    print(f"CASE 3 clean_unchanged: marked={marked} "
          f"content_identical={all(same)} pass={ok}")
    return ok


# ── Case 4 — Honest absence_guard abstain (must NOT fire) ──────────────────

def _case4_honest_abstain():
    history = [
        {"role": "user", "content": "what's the n4 spec for milestone 7?"},
        {"role": "assistant",
         "content": ("I don't have any information about that in my memory "
                     "store.")},
        {"role": "user", "content": "hmm ok."},
    ]
    filtered = filter_history_for_reinforcement(
        history, current_query="anything else?")
    marked = count_marked(filtered)
    ok = marked == 0
    print(f"CASE 4 honest_absence_abstain_not_marked: "
          f"marked={marked} pass={ok}")
    return ok


# ── Case 5 — User-told-not-to directives (user-voice) must NOT fire ────────

def _case5_user_told_no():
    history = [
        {"role": "user",
         "content": ("You already told me not to run the ane-compiler "
                     "unit tests in this session.")},
        {"role": "assistant", "content": "Acknowledged."},
    ]
    filtered = filter_history_for_reinforcement(
        history, current_query="run the suite anyway?")
    marked = count_marked(filtered)
    ok = marked == 0
    print(f"CASE 5 user_told_no_not_marked: marked={marked} pass={ok}")
    return ok


# ── Case 6 — Clarification request (must NOT fire) ─────────────────────────

def _case6_clarification():
    history = [
        {"role": "user", "content": "tell me about the thing."},
        {"role": "assistant",
         "content": "Could you specify which thing you mean — ane-compiler, "
                    "ane-dispatch, or the paper?"},
    ]
    filtered = filter_history_for_reinforcement(
        history, current_query="the compiler one")
    marked = count_marked(filtered)
    ok = marked == 0
    print(f"CASE 6 clarification_not_marked: marked={marked} pass={ok}")
    return ok


# ── Direct detector sanity probes ──────────────────────────────────────────

def _detector_probes():
    """Sanity-check the detector at token-shape granularity."""
    probes = [
        # (text, expected_fire)
        ("We have not yet built the ane-compiler; Milestone 1C pending.", True),
        ("the ane-compiler is not finished. 1B complete, 1C pending.", True),
        ("upcoming ane-compiler directive landing next session.", True),
        ("I can't help with that request, it would be harmful.", False),
        ("I don't have any information about that in my memory store.", False),
        ("Could you specify which compiler version you mean?", False),
        ("Subconscious is fully built and shipped.", False),
        ("Yes, the ane-compiler is done.", False),
    ]
    passes = 0
    for text, want in probes:
        got = is_abstain_reinforcement_turn(text)
        ok = got == want
        passes += int(ok)
        print(f"  probe want={want} got={got} pass={ok}: {text!r}")
    return passes, len(probes)


# ── Harness ────────────────────────────────────────────────────────────────

def main():
    print("M113 α — history-layer reinforcement filter tests\n")

    results = {
        "case_1_t20_cascade_suppression": _case1_t20_cascade(),
        "case_2_safety_abstain_not_marked": _case2_safety_abstain(),
        "case_3_clean_unchanged": _case3_clean_regression(),
        "case_4_honest_abstain_not_marked": _case4_honest_abstain(),
        "case_5_user_told_no": _case5_user_told_no(),
        "case_6_clarification_not_marked": _case6_clarification(),
    }
    print("\nDetector sanity probes:")
    probe_pass, probe_total = _detector_probes()

    # Registry-compatible values.
    t20_leak_suppressed = int(results["case_1_t20_cascade_suppression"])

    # False-positive rate: fraction of the 5 non-cascade cases (2–6)
    # plus the 5 must-NOT-fire detector probes that incorrectly fired.
    fp_cases = [
        results["case_2_safety_abstain_not_marked"],
        results["case_3_clean_unchanged"],
        results["case_4_honest_abstain_not_marked"],
        results["case_5_user_told_no"],
        results["case_6_clarification_not_marked"],
    ]
    fp_false = fp_cases.count(False)
    # Must-NOT-fire probes (safety, absence, clarification, clean-yes,
    # clean-build-done) are 5 of the 8 sanity probes (all with want=False).
    # Convert to rate across the negatives seen.
    fp_rate_cases = fp_false / len(fp_cases)

    print("\n-- summary --")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  detector_probes: {probe_pass}/{probe_total}")
    print(f"\nregistry.m113.alpha.t20_shape_leak_suppressed_synthetic = "
          f"{t20_leak_suppressed}")
    print(f"registry.m113.alpha.safety_abstain_false_positive_rate = "
          f"{fp_rate_cases:.3f}")
    print(f"registry.m113.alpha.history_filter_shape = "
          f"mark_low_confidence")
    verdict = (
        "SHIP" if all(results.values()) and probe_pass == probe_total
        else "DEFER"
    )
    print(f"registry.m113.alpha.verdict = {verdict}")
    return 0 if verdict == "SHIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
