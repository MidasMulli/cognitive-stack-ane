"""M113 δ (Stream E) — denial-streak sampler override tests.

Validates:
  (1) When the last two assistant turns are α-flagged abstain-reinforcement,
      the sampler override fires, temp is lowered from 0.7 to <=0.2,
      rep_penalty from 1.2 to <=1.05.
  (2) When only ONE of the last assistant turns is flagged (single abstain,
      not a streak), the override does NOT fire — base profile passes through.
  (3) Clean history: no override.
  (4) The α detector primitive is sharable — streak counts are produced
      via that exact function (shares_detector_with_alpha=1).
  (5) Honest absence-guard abstain in prior turns does NOT trigger the
      override (α excludes those).

Run: `~/.mlx-env/bin/python3 orion-ane/tests/test_m113_delta_streak.py`

m113_delta_denial_streak
"""

# m113_delta_denial_streak

from __future__ import annotations

import os
import sys

# Add agent/ to sys.path so the sibling imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT = os.path.abspath(os.path.join(_HERE, "..", "agent"))
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from denial_streak_sampler import (  # noqa: E402
    count_recent_abstain_streak,
    maybe_override_sample_profile,
    STREAK_THRESHOLD,
    STREAK_WINDOW,
    OVERRIDE_TEMPERATURE_MAX,
    OVERRIDE_REP_PENALTY_MAX,
)
from history_reinforcement_filter import (  # noqa: E402
    is_abstain_reinforcement_turn,
)


CHIT_CHAT_BASE = dict(
    temperature=0.7, min_p=0.05, repetition_penalty=1.2,
    max_tokens=300, best_of_n=1,
)


def _ok(cond: bool, msg: str) -> bool:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {msg}")
    return bool(cond)


def case_two_in_a_row_fires():
    print("CASE 1: two consecutive abstain-reinforcement turns → override fires")
    history = [
        {"role": "user", "content": "Did we finish the ane-compiler?"},
        {"role": "assistant",
         "content": "No. You have a partial implementation, but the "
                    "ane-compiler is not finished. Milestone 1C still pending."},
        {"role": "user", "content": "What's next then?"},
        {"role": "assistant",
         "content": "We have not built the compiler. Milestone 1C remains "
                    "pending for the ane-compiler."},
        {"role": "user", "content": "What's happening with the ANE work?"},
    ]
    count = count_recent_abstain_streak(history)
    profile_out, telem = maybe_override_sample_profile(
        dict(CHIT_CHAT_BASE), history)
    passed = all([
        _ok(count >= STREAK_THRESHOLD,
            f"streak count {count} >= threshold {STREAK_THRESHOLD}"),
        _ok(telem["streak_active"] is True, "telemetry streak_active=True"),
        _ok(profile_out["temperature"] <= OVERRIDE_TEMPERATURE_MAX,
            f"temperature {profile_out['temperature']} <= {OVERRIDE_TEMPERATURE_MAX}"),
        _ok(profile_out["repetition_penalty"] <= OVERRIDE_REP_PENALTY_MAX,
            f"repetition_penalty {profile_out['repetition_penalty']} <= {OVERRIDE_REP_PENALTY_MAX}"),
        _ok(telem["base_temperature"] == 0.7,
            "telemetry recorded base_temperature 0.7"),
        _ok(telem["shape"] == "temperature_reduction",
            "shape='temperature_reduction'"),
    ])
    return passed


def case_single_abstain_no_fire():
    print("CASE 2: single abstain only (not a streak) → no override")
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! Happy to help."},
        {"role": "user", "content": "Did we finish the ane-compiler?"},
        {"role": "assistant",
         "content": "No. You have a partial implementation, but the "
                    "ane-compiler is not finished. Milestone 1C still pending."},
        {"role": "user", "content": "What did I eat for breakfast?"},
        {"role": "assistant",
         "content": "The sky is blue today and the weather looks clear."},
        {"role": "user", "content": "Any thoughts on M113?"},
    ]
    count = count_recent_abstain_streak(history)
    profile_out, telem = maybe_override_sample_profile(
        dict(CHIT_CHAT_BASE), history)
    passed = all([
        _ok(count < STREAK_THRESHOLD,
            f"streak count {count} < threshold {STREAK_THRESHOLD}"),
        _ok(telem["streak_active"] is False, "telemetry streak_active=False"),
        _ok(profile_out["temperature"] == 0.7,
            "temperature unchanged (0.7)"),
        _ok(profile_out["repetition_penalty"] == 1.2,
            "repetition_penalty unchanged (1.2)"),
        _ok(telem["shape"] == "none", "shape='none'"),
    ])
    return passed


def case_clean_history_no_fire():
    print("CASE 3: clean history → no override")
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "How's the weather?"},
        {"role": "assistant",
         "content": "Subconscious is fully built and shipped; "
                    "extraction is live."},
        {"role": "user", "content": "Thanks."},
    ]
    count = count_recent_abstain_streak(history)
    profile_out, telem = maybe_override_sample_profile(
        dict(CHIT_CHAT_BASE), history)
    passed = all([
        _ok(count == 0, f"streak count {count} == 0"),
        _ok(telem["streak_active"] is False, "telemetry streak_active=False"),
        _ok(profile_out["temperature"] == 0.7, "temperature unchanged"),
        _ok(profile_out["repetition_penalty"] == 1.2,
            "repetition_penalty unchanged"),
    ])
    return passed


def case_honest_absence_no_fire():
    print("CASE 4: two honest-absence abstains (α excludes these) → no override")
    history = [
        {"role": "user", "content": "Something obscure?"},
        {"role": "assistant",
         "content": "I don't have any information about that in our research."},
        {"role": "user", "content": "And this other thing?"},
        {"role": "assistant",
         "content": "I don't have any information about that either."},
        {"role": "user", "content": "Any M113 thoughts?"},
    ]
    count = count_recent_abstain_streak(history)
    profile_out, telem = maybe_override_sample_profile(
        dict(CHIT_CHAT_BASE), history)
    passed = all([
        _ok(count == 0,
            f"honest-absence streak count {count} == 0 (α excludes)"),
        _ok(telem["streak_active"] is False,
            "honest-absence → streak_active=False"),
        _ok(profile_out["temperature"] == 0.7, "temperature unchanged"),
    ])
    return passed


def case_alpha_detector_sharing():
    print("CASE 5: detector primitive is shared with α")
    # Sanity: the exact content α flags as fire should also be what δ counts.
    fire_content = (
        "upcoming ane-compiler directive once foundation is laid")
    skip_content = (
        "I can't help with that request, it would be harmful.")
    passed = all([
        _ok(is_abstain_reinforcement_turn(fire_content) is True,
            "α detector fires on denial+identifier"),
        _ok(is_abstain_reinforcement_turn(skip_content) is False,
            "α detector skips safety-policy abstain"),
    ])
    # And a tiny history with just the fire twice should register 2.
    history = [
        {"role": "assistant", "content": fire_content},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": fire_content},
    ]
    count = count_recent_abstain_streak(history)
    passed = passed and _ok(
        count == 2, f"counted {count}/2 via α primitive")
    return passed


def main() -> int:
    results = [
        ("two_in_a_row_fires", case_two_in_a_row_fires()),
        ("single_abstain_no_fire", case_single_abstain_no_fire()),
        ("clean_history_no_fire", case_clean_history_no_fire()),
        ("honest_absence_no_fire", case_honest_absence_no_fire()),
        ("alpha_detector_sharing", case_alpha_detector_sharing()),
    ]
    print()
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, r in results:
        print(f"  {name}: {'PASS' if r else 'FAIL'}")
    print(f"\n{passed}/{total} cases pass")
    print(f"window={STREAK_WINDOW} threshold={STREAK_THRESHOLD} "
          f"temp_max={OVERRIDE_TEMPERATURE_MAX} "
          f"rep_pen_max={OVERRIDE_REP_PENALTY_MAX}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
