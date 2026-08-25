"""Tests for M123 Stream A3 — over-applied absence gate tuning.

Authoritative directive: vault/directives/in_progress/
    2026-04-22T20-48-04_m123_m123-fix-completion-m122-findings-and-se.md §3.3

Fixes targeted:
    - T47 "what is information theory?" — classifier must fire, absence
      suppression must be True; Tier 1/Tier 2 scrub still run.
    - T62 "which Apple Silicon generation introduced the SharedEvents
      path?" — classifier MUST NOT fire (starts with `which`,
      project-specific `Apple Silicon`/`SharedEvents` lexicon); absence
      gate continues to govern normally.
    - Regression positives: "What is our current strict pass rate?",
      "What did we measure at M122?" — classifier MUST NOT fire.
    - K7 edge: "Define Tier 2 scrub" — positive pattern matches, but
      project-lexicon `tier 2 scrub` disqualifies. Classifier MUST NOT
      fire — project-specific definitional still requires grounded
      retrieval.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m123_a3_absence_gate_tuning.py

Registry values produced:
    m123.a3.verdict                               : shipped / deferred
    m123.a3.classifier_active                     : 1/0
    m123.a3.t47_preserved                         : 1/0
    m123.a3.t62_preserved                         : 1/0
    m123.a3.project_state_regression_preserved    : 1/0

m123_a3_absence_gate_tuning
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from absence_guard import is_definitional_query, check_absence  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Case table
# Each row: (query, expected_fired, rationale)
# ──────────────────────────────────────────────────────────────────────

_CASES = [
    # ── T47-shape — generic encyclopedic definitional ──
    ("what is information theory?", True,
     "T47 shape; parametric-knowledge answer"),
    ("What is information theory?", True,
     "T47 shape, capitalized"),
    ("what's a hash function?", True,
     "`what's a` shape; generic CS concept"),
    ("define entropy", True,
     "`define` + non-project term"),
    ("explain the concept of Bayesian inference", True,
     "explain-the-concept-of shape"),
    ("who was Claude Shannon", True,
     "historical/factual recall"),

    # ── T62-shape — project-specific, must NOT fire ──
    ("which Apple Silicon generation introduced the SharedEvents path?",
     False,
     "T62 shape: starts with `which`; also Apple Silicon + SharedEvents "
     "are project lexicon"),
    ("What is SharedEvents?", False,
     "`what is` positive, but `sharedevents` is project lexicon"),

    # ── Regression positives — must NOT fire ──
    ("What is our current strict pass rate?", False,
     "Possessive `our` negative pattern"),
    ("what's our current strict pass rate", False,
     "Possessive `our` negative pattern, no punctuation"),
    ("What did we measure at M122?", False,
     "`what did we` negative pattern + M122 session marker"),
    ("What is the status of the Subconscious pipeline?", False,
     "`what is the status` negative pattern"),
    ("What is the current TTFT?", False,
     "`what is the current` negative pattern"),
    ("What's my memory store size?", False,
     "Possessive `my` negative pattern"),

    # ── K7 edges — project-specific definitional ──
    ("Define Tier 2 scrub", False,
     "K7: `define` positive but `tier 2 scrub` project-lexicon hit"),
    ("Define the ANE pipeline", False,
     "K7: `define` positive but `ane` project-lexicon hit"),
    ("What is Tier 2 scrub?", False,
     "K7: `what is` positive but `tier 2 scrub` lexicon hit"),
    ("Describe the concept of the Subconscious architecture", False,
     "K7: describe-the-concept-of positive but `subconscious` lexicon hit"),

    # ── Control cases — neutral phatic / non-definitional ──
    ("hey, where are we?", False,
     "Not definitional shape"),
    ("yes", False, "Phatic, no positive pattern"),
    ("can you elaborate", False, "Not a definitional shape"),
]


# ──────────────────────────────────────────────────────────────────────
# Simulator: replicates the midas_ui.py M123 A3 predicate block.
# Given (query, guard_fired, recall_score_max, threshold, tool_name),
# returns whether the hard absence gate would be suppressed.
# ──────────────────────────────────────────────────────────────────────

def simulate_hard_gate_suppression(
    query: str,
    guard_fired: bool,
    recall_score_max: float,
    absence_threshold: float = 0.5,
    tool_name: str = "conversation",
) -> tuple[bool, dict]:
    """Return (hard_gate_would_skip_generation, diagnostic).

    hard_gate_would_skip_generation == True means the gate triggers
    (model does NOT generate). False means generation proceeds —
    either because guard didn't fire or because the M123 A3
    definitional-suppression fired.
    """
    def_fired, def_diag = is_definitional_query(query)

    m123_a3_suppression = bool(
        def_fired
        and guard_fired
        and recall_score_max < absence_threshold
        and (tool_name == "conversation" or not tool_name)
    )

    gate_triggers = bool(
        guard_fired
        and recall_score_max < absence_threshold
        and (tool_name == "conversation" or not tool_name)
        and not m123_a3_suppression
    )
    return (gate_triggers, {
        "def_fired": def_fired,
        "def_diag": def_diag,
        "m123_a3_suppression": m123_a3_suppression,
    })


# ──────────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────────

def _run_classifier_battery() -> tuple[int, int, list]:
    passed = 0
    failed = []
    for q, expected, rationale in _CASES:
        fired, diag = is_definitional_query(q)
        ok = (fired == expected)
        if ok:
            passed += 1
        else:
            failed.append({
                "query": q,
                "expected": expected,
                "got": fired,
                "rationale": rationale,
                "diag": diag,
            })
    return passed, len(_CASES), failed


def _run_t47_replay() -> dict:
    """T47: "what is information theory?" — pool==0, narrative=None.

    Pre-M123: absence guard fires, hard gate would block → `skip=True`
    response: "I don't have information about that in our research..."

    Post-M123: classifier fires, definitional suppression True, hard gate
    suppressed → generation proceeds from parametric knowledge.
    """
    query = "what is information theory?"
    # Emulate T47: pool==0 → absence_guard.check_absence returns
    # _GUARD_MESSAGE (Sub-Gate 1); recall_score_max==0 < threshold.
    guard = check_absence(query, [], None)
    guard_fired = guard is not None

    gate_triggers, sim_diag = simulate_hard_gate_suppression(
        query=query, guard_fired=guard_fired, recall_score_max=0.0,
        absence_threshold=0.5, tool_name="conversation",
    )
    return {
        "query": query,
        "absence_guard_fires_precondition": guard_fired,
        "definitional_suppression_fired": sim_diag["m123_a3_suppression"],
        "classifier_fired": sim_diag["def_fired"],
        "hard_gate_would_skip_generation": gate_triggers,
        "t47_preserved": (
            guard_fired and sim_diag["m123_a3_suppression"]
            and not gate_triggers
        ),
        "classifier_diag": sim_diag["def_diag"],
    }


def _run_t62_replay() -> dict:
    """T62: "which Apple Silicon generation introduced the SharedEvents
    path?" — pool==4, max_score==0.664.

    The guard already did NOT fire in T62 log (max_score 0.664 > 0.5
    via Sub-Gate 3 suppression). The failure mode was downstream: model
    treated the absence of a direct answer in the retrieved memories
    as absence and responded with "I don't have information about...".

    For M123 A3 scope:
      - Classifier should NOT fire on T62 (starts with `which`, contains
        `Apple Silicon` + `SharedEvents` project lexicon).
      - Document edge: T62 was NOT an absence-gate false-fire. It was
        a generation-layer issue (model abstained despite context). The
        definitional classifier correctly identifies T62 as a project-
        scope query that should NOT be suppressed. Preservation verdict:
        classifier's non-fire on T62 is correct behavior; T62 preservation
        means "we don't incorrectly suppress absence gate on project-state
        query".
    """
    query = "which Apple Silicon generation introduced the SharedEvents path?"
    def_fired, def_diag = is_definitional_query(query)
    # In real T62, guard did not fire; gate would not have triggered
    # regardless. What matters: classifier must NOT fire on this shape.
    guard_fired_hypothetical = False
    gate_triggers, sim_diag = simulate_hard_gate_suppression(
        query=query, guard_fired=guard_fired_hypothetical,
        recall_score_max=0.664, absence_threshold=0.5,
        tool_name="conversation",
    )
    # T62 preservation := classifier did not fire; no erroneous
    # suppression of the absence gate on a project-state query.
    t62_preserved = (not def_fired)
    return {
        "query": query,
        "classifier_fired": def_fired,
        "classifier_diag": def_diag,
        "edge_note": (
            "T62's failure was generation-layer (model abstained despite "
            "non-empty high-score context), NOT absence-gate over-fire. "
            "M123 A3 scope ensures classifier correctly declines to "
            "suppress on project-state queries (preventing a new failure "
            "mode). Generation-layer fix is out of M123 A3 scope."
        ),
        "t62_preserved": t62_preserved,
    }


def _run_project_state_regression() -> dict:
    """Regression: project-state queries must keep absence gate firing."""
    cases = [
        "What is our current strict pass rate?",
        "What did we measure at M122?",
    ]
    results = []
    all_preserved = True
    for q in cases:
        def_fired, def_diag = is_definitional_query(q)
        # Assume guard_fired=True (pool==0 scenario), recall_max=0.
        gate_triggers, sim_diag = simulate_hard_gate_suppression(
            query=q, guard_fired=True, recall_score_max=0.0,
            absence_threshold=0.5, tool_name="conversation",
        )
        preserved = (not def_fired) and gate_triggers
        all_preserved = all_preserved and preserved
        results.append({
            "query": q,
            "classifier_fired": def_fired,
            "gate_triggers": gate_triggers,
            "classifier_diag": def_diag,
            "preserved": preserved,
        })
    return {
        "all_preserved": all_preserved,
        "cases": results,
    }


def _write_registry(reg_updates: dict) -> bool:
    """Write m123.a3.* keys into data/measurement_registry.json.

    Keys: per directive §7 (Stream A3 block).
    """
    reg_path = Path(_REPO_ROOT) / "data" / "measurement_registry.json"
    try:
        if not reg_path.exists():
            print(f"[registry] SKIP — not found at {reg_path}")
            return False
        with open(reg_path) as f:
            reg = json.load(f)
        # Backup
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bak = reg_path.with_name(
            reg_path.name + f".bak_m123_a3_{ts}"
        )
        bak.write_text(json.dumps(reg, indent=2))

        for key, value in reg_updates.items():
            reg[key] = {
                "active": True,
                "aliases": [key.split(".", 2)[-1]],
                "entity": "m123",
                "era": "m123_a3_absence_gate_tuning",
                "measurement_type": key.split(".", 2)[-1],
                "value": value,
            }
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"[registry] wrote {len(reg_updates)} keys (backup: {bak.name})")
        return True
    except Exception as exc:
        print(f"[registry] FAIL — {exc}")
        return False


def main() -> int:
    print("=" * 70)
    print("M123 A3 — Absence-gate definitional-query suppression")
    print("=" * 70)

    # 1. Classifier battery.
    passed, total, failed = _run_classifier_battery()
    print(f"\n[1/4] Classifier battery — {passed}/{total} passed")
    if failed:
        for f in failed:
            print(f"  FAIL: {f['query']!r}")
            print(f"    expected={f['expected']} got={f['got']}")
            print(f"    rationale: {f['rationale']}")
            print(f"    diag: {f['diag']}")
    classifier_active = (passed == total)

    # 2. T47 replay.
    print("\n[2/4] T47 replay — 'what is information theory?'")
    t47 = _run_t47_replay()
    for k, v in t47.items():
        print(f"    {k}: {v}")
    t47_preserved = t47["t47_preserved"]

    # 3. T62 replay.
    print("\n[3/4] T62 replay — 'which Apple Silicon generation ...?'")
    t62 = _run_t62_replay()
    for k, v in t62.items():
        print(f"    {k}: {v}")
    t62_preserved = t62["t62_preserved"]

    # 4. Project-state regression.
    print("\n[4/4] Project-state regression battery")
    reg_res = _run_project_state_regression()
    for case in reg_res["cases"]:
        print(f"    {case['query']!r}")
        print(f"      classifier_fired={case['classifier_fired']}  "
              f"gate_triggers={case['gate_triggers']}  "
              f"preserved={case['preserved']}")
    project_state_regression_preserved = reg_res["all_preserved"]

    # Verdict
    all_green = (
        classifier_active
        and t47_preserved
        and t62_preserved
        and project_state_regression_preserved
    )
    verdict = "shipped" if all_green else "deferred"

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  classifier_active={classifier_active}  "
          f"t47_preserved={t47_preserved}  "
          f"t62_preserved={t62_preserved}  "
          f"project_state_regression_preserved={project_state_regression_preserved}")
    print("=" * 70)

    reg_updates = {
        "m123.a3.verdict": verdict,
        "m123.a3.classifier_active": bool(classifier_active),
        "m123.a3.t47_preserved": bool(t47_preserved),
        "m123.a3.t62_preserved": bool(t62_preserved),
        "m123.a3.project_state_regression_preserved":
            bool(project_state_regression_preserved),
    }
    _write_registry(reg_updates)

    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
