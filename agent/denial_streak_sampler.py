"""Denial-streak sampler override — M113 δ (Stream E).

Problem (M108 Finding 2 part 2, T32 re-expanded):
    Two or more consecutive prior-assistant abstain/denial turns (the same
    abstain-plus-project-state-claim shape α detects) create a self-
    reinforcing pattern that the `chit_chat` sampler (temp=0.7,
    repetition_penalty=1.2) amplifies: the model literally samples the
    next denial because it's the highest-probability continuation of two
    previous denials stitched into the prompt. T32 denied even with the
    relevant memory turns (T23a/T23b/T25) back in the attention window.

Fix shape (a) — temperature reduction on streak-active turns:
    When at least K=2 of the last W=4 prior-assistant turns match α's
    `is_abstain_reinforcement_turn`, override the sample profile for THIS
    turn only: temperature 0.7 → 0.2, repetition_penalty 1.2 → 1.05.
    Lower temp + lower rep-penalty together tilt the next-token distribution
    away from the high-probability denial continuation that the streak is
    reinforcing. The base `_SAMPLE_PROFILES` dict is untouched; the override
    is a per-turn local adjustment before `llm_stream` is called.

Coordination with α (Stream B):
    This module imports `is_abstain_reinforcement_turn` from
    `history_reinforcement_filter` (α's primitive, exported for δ reuse per
    the M113 directive). Shares detector definition — no divergence per §K6.

Discipline:
    - Detection-layer reuse, sampler-layer override only.
    - No retrieval, router, scrub, or history-filter mutation.
    - Shallow override; dispatch's `_SAMPLE_PROFILES` never changes.
    - Import-guarded at call site so a bad import fails open
      (base sampler preserved).

m113_delta_denial_streak
"""

# m113_delta_denial_streak

from __future__ import annotations

from typing import Iterable, Dict, Optional, Tuple

try:
    from history_reinforcement_filter import is_abstain_reinforcement_turn
    _ALPHA_DETECTOR_OK = True
except Exception:  # pragma: no cover — fail-open if α import breaks.
    _ALPHA_DETECTOR_OK = False

    def is_abstain_reinforcement_turn(content: str) -> bool:  # type: ignore
        return False


# Streak parameters. Tight defaults so the override is specific to
# the M108 Finding-2-part-2 failure mode (two prior denials in a row).
# Widening W allows a normal-turn between denials to still count as a
# streak (the T23/T25 recovery case would not have protected T32 if the
# streak is strict-consecutive with W=K=2).
STREAK_WINDOW: int = 4   # look at last W assistant turns
STREAK_THRESHOLD: int = 2  # need K flagged turns in that window

# Override shape (a). Target: chit_chat base is {temp=0.7, rep_pen=1.2}.
# Both are dialled down together — lower temp narrows the distribution,
# lower rep-penalty stops the denial pattern from being penalised *less*
# than other tokens (which paradoxically amplifies it under repeated
# prefix). Shape chosen per directive §3 recommendation (a).
OVERRIDE_TEMPERATURE_MAX: float = 0.2
OVERRIDE_REP_PENALTY_MAX: float = 1.05


def count_recent_abstain_streak(
    history: Iterable[Dict],
    window: int = STREAK_WINDOW,
) -> int:
    """Return count of α-flagged assistant turns in the last `window`
    assistant messages of `history`.

    Scans from the end of `history` backward, collecting the last `window`
    entries with `role == 'assistant'` and returning how many of those
    match `is_abstain_reinforcement_turn`. User turns between them do not
    break the streak — the amplification reads through the *assistant*
    voice across turns, not through user interjections.

    m113_delta_denial_streak
    """
    if not history:
        return 0
    assistant_entries = []
    for msg in reversed(list(history)):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        assistant_entries.append(msg.get("content", "") or "")
        if len(assistant_entries) >= window:
            break
    return sum(
        1 for content in assistant_entries
        if is_abstain_reinforcement_turn(content)
    )


def streak_active(
    history: Iterable[Dict],
    window: int = STREAK_WINDOW,
    threshold: int = STREAK_THRESHOLD,
) -> bool:
    """True if enough recent assistant turns match α's reinforcement pattern
    to trigger the sampler override.

    m113_delta_denial_streak
    """
    return count_recent_abstain_streak(history, window=window) >= threshold


def maybe_override_sample_profile(
    sample_profile: Dict,
    history: Iterable[Dict],
    *,
    window: int = STREAK_WINDOW,
    threshold: int = STREAK_THRESHOLD,
) -> Tuple[Dict, Dict]:
    """Return a (possibly-overridden) sample_profile and a telemetry dict.

    If a denial streak is active on `history`, produce a shallow copy of
    `sample_profile` with temperature and repetition_penalty lowered (only
    if they were higher than the override ceilings — never *raise* them).
    Otherwise return the input profile unchanged.

    Returns:
        (profile_out, telemetry) where telemetry has keys:
          - streak_active: bool
          - streak_count: int
          - window: int
          - threshold: int
          - alpha_detector_ok: bool (α import success)
          - base_temperature: float
          - base_repetition_penalty: float
          - override_temperature: float | None
          - override_repetition_penalty: float | None
          - shape: "temperature_reduction" | "none"

    m113_delta_denial_streak
    """
    base_temp = float(sample_profile.get("temperature", 0.0))
    base_rep = float(sample_profile.get("repetition_penalty", 1.0))
    count = count_recent_abstain_streak(history, window=window)
    active = count >= threshold
    telemetry = {
        "streak_active": bool(active),
        "streak_count": int(count),
        "window": int(window),
        "threshold": int(threshold),
        "alpha_detector_ok": bool(_ALPHA_DETECTOR_OK),
        "base_temperature": base_temp,
        "base_repetition_penalty": base_rep,
        "override_temperature": None,
        "override_repetition_penalty": None,
        "shape": "none",
    }
    if not active:
        return sample_profile, telemetry

    out = dict(sample_profile)
    new_temp = min(base_temp, OVERRIDE_TEMPERATURE_MAX)
    new_rep = min(base_rep, OVERRIDE_REP_PENALTY_MAX)
    out["temperature"] = new_temp
    out["repetition_penalty"] = new_rep
    telemetry["override_temperature"] = new_temp
    telemetry["override_repetition_penalty"] = new_rep
    telemetry["shape"] = "temperature_reduction"
    return out, telemetry


__all__ = [
    "count_recent_abstain_streak",
    "streak_active",
    "maybe_override_sample_profile",
    "STREAK_WINDOW",
    "STREAK_THRESHOLD",
    "OVERRIDE_TEMPERATURE_MAX",
    "OVERRIDE_REP_PENALTY_MAX",
]
