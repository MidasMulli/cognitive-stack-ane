"""M125.3 Stream E — Enumeration payload-to-prompt plumbing tests.

Scope: verify the plumbing fix that routes enumeration records from
Phase 2a (enumeration_retrieval.enumerate_by_tag) through to the final
user-message tail (the `augmented` string passed to synthesizer.build_messages).

Drop site identified (pre-fix):
  1. orion-ane/agent/synthesizer.py:471-476 — `elif memory_context:` branch
     is dead whenever briefing is present (always). Enumeration records
     put in mem_ctx never reached the prompt.
  2. orion-ane/agent/midas_ui.py:~4201 absence-guard Sub-Gate 3 — word-
     overlap heuristic overwrote mem_ctx with a KNOWLEDGE GAP warning on
     enumeration turns (Q10 in M125.2 E pilot readiness: 4
     published_repo records replaced by hedge template).

Fix (pre-generation prompt-assembly surface, stream path only):
  A. Build a dedicated `_enumeration_block` string alongside mem_ctx in
     the enumeration branch.
  B. Gate absence-guard Sub-Gate 3 on `not _enumeration_active` so the
     enumeration payload survives into mem_ctx (grounding corpus).
  C. Append `_enumeration_block` to the `blocks` list that composes
     `augmented` — reaches the prompt via the user-message tail, which
     is how per_query_block / _reg_block / _possessive_directive already
     reach the prompt.

Regression guardrail:
  - Non-enumeration queries must NOT acquire an enumeration block.
  - M120 E 8-archetype battery: shape classifier unchanged by this fix.
  - Assembly-side flag is scoped to Phase 2a; narrative / default_recall
    do not produce an _enumeration_block.

Run:
  ~/.mlx-env/bin/python3 orion-ane/tests/test_m125_3_e_enumeration_plumbing.py

Exit code 0 = all pass.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT = os.path.normpath(os.path.join(_HERE, "..", "agent"))
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from enumeration_retrieval import enumerate_by_tag, shape_dispatch_signal  # noqa: E402


# ── Plumbing simulator ─────────────────────────────────────────────────────
#
# Mirrors the stream-path Phase 2a / assembly logic added in
# midas_ui.py (m125_3_e_enumeration_plumbing). Returns the final
# `augmented` user-message string the way the production code does.
# Keeps the test self-contained so regressions in the simulator surface
# with the same symptoms they would in midas_ui.

_GUARD_TEMPLATE = (
    "=== CRITICAL: KNOWLEDGE GAP ===\n"
    "The memory store has NO information about this topic. "
    "Zero relevant memories were found. You have NO factual basis to "
    "answer this question.\n"
)


def _absence_guard_sub_gate_3(message, mem_ctx, enumeration_active):
    """Replica of midas_ui.py:~4201 Sub-Gate 3 word-overlap check.

    Post-fix, this path is gated on `not enumeration_active` so the
    enumeration payload survives. Returns (mem_ctx, fired).
    """
    if not (mem_ctx and not enumeration_active):
        return mem_ctx, False
    _q_words = set(
        w.lower().strip("?.,!\"'")
        for w in message.split()
        if len(w) >= 4
    )
    _stop = {"what", "does", "that", "this", "with", "from", "have",
             "been", "about", "which", "where", "when", "many", "much",
             "strategy", "should", "would", "could", "explain", "tell"}
    _q_specific = _q_words - _stop
    if _q_specific and len(_q_specific) >= 2:
        _mem_text = " ".join(str(m) for m in mem_ctx).lower()
        _unmatched = [w for w in _q_specific if w not in _mem_text]
        if len(_unmatched) > len(_q_specific) * 0.5:
            return [_GUARD_TEMPLATE], True
    return mem_ctx, False


def assemble_augmented_user_message(message, inject_context=True,
                                     briefing="BRIEFING-PLACEHOLDER",
                                     pre_fix=False):
    """Run the fix-relevant subset of midas_ui.py stream-path Phase 2a.

    `pre_fix=True` simulates the pre-M125.3-E behavior (no dedicated
    enumeration block; absence-guard Sub-Gate 3 not enumeration-aware).
    Returns a dict with the observable outputs: mem_ctx, blocks,
    augmented, enumeration_active.
    """
    mem_ctx = None
    _enumeration_block = None
    _enumeration_active = False

    # Phase 2a
    if inject_context:
        _enum = enumerate_by_tag(message)
        if _enum:
            mem_ctx = list(_enum["records"])
            if not pre_fix:
                header = (
                    f"[ENUMERATION: complete set of "
                    f"{_enum.get('tag')} records "
                    f"(count={_enum.get('count')})]"
                )
                lines = list(_enum.get("records") or [])
                if lines:
                    _enumeration_block = header + "\n" + "\n".join(lines)
                    _enumeration_active = True

    # Absence-guard Sub-Gate 3 — word-overlap check.
    # Pre-fix: runs unconditionally when mem_ctx present.
    # Post-fix: gated on `not _enumeration_active`.
    mem_ctx, _guard_fired = _absence_guard_sub_gate_3(
        message, mem_ctx,
        enumeration_active=(_enumeration_active and not pre_fix))

    # Assembly: blocks list → user-message tail.
    blocks = []
    if not pre_fix and _enumeration_block:
        blocks.append(_enumeration_block)
    # (per_query_block / _reg_block omitted here; this test targets the
    # enumeration injection site only.)
    if blocks:
        augmented = "\n\n".join(blocks) + f"\n\n---\n\n{message}"
    else:
        augmented = message

    return {
        "mem_ctx": mem_ctx,
        "enumeration_block": _enumeration_block,
        "enumeration_active": _enumeration_active,
        "guard_fired": _guard_fired,
        "augmented": augmented,
        "blocks": blocks,
    }


# ── M120 E 8-archetype battery (plumbing replay) ──────────────────────────
#
# The original M120 E battery (test_m120_e_enumeration_dispatch.py) checks
# the classifier. This battery replays those archetypes through the full
# assembly simulator and asserts enumeration records reach the augmented
# user message when the shape is backed.

_M120_E_BATTERY = [
    {
        "name": "t88_replay_published_repo",
        "query": "enumerate all published repos under my name",
        "expect_enum": True,
        "expect_tag": "published_repo",
    },
    {
        "name": "plural_noun_standing_rules",
        "query": "what are the standing rules?",
        "expect_enum": True,
        "expect_tag": "standing_rule",
    },
    {
        "name": "regression_singular_standing_rule",
        "query": "what is the standing rule about X",
        "expect_enum": False,
    },
    {
        "name": "list_imperative_dead_paths",
        "query": "list all dead paths",
        "expect_enum": True,
        "expect_tag": "dead_path",
    },
    {
        "name": "widening_list_the_plural",
        "query": "list the repositories",
        "expect_enum": True,
        "expect_tag": "published_repo",
    },
    {
        "name": "widening_full_list_of",
        "query": "give me the full list of measurements",
        "expect_enum": True,
        "expect_tag": "measurement",
    },
    {
        "name": "regression_what_is",
        "query": "what is the measurement we took",
        "expect_enum": False,
    },
    {
        "name": "regression_greeting",
        "query": "how are you?",
        "expect_enum": False,
    },
]


def test_m120_e_battery_plumbing():
    """Replay M120 E 8-archetype battery through the plumbing. For every
    archetype where the classifier fires an enumeration shape against a
    backed tag, the augmented user message must contain the enumeration
    records AND the ENUMERATION header.

    Counts pass = archetype produced the expected augmented-message
    content (backed -> records reach prompt; unbacked/non-enum -> no
    enumeration block).
    """
    passes = 0
    total = len(_M120_E_BATTERY)
    failures = []
    for case in _M120_E_BATTERY:
        out = assemble_augmented_user_message(case["query"])
        name = case["name"]
        if case["expect_enum"]:
            _enum = enumerate_by_tag(case["query"])
            if _enum is None:
                # unbacked tag — no records to surface; assert no
                # enumeration_block so we don't fabricate.
                ok = not out["enumeration_active"]
            else:
                ok = (
                    out["enumeration_active"]
                    and "[ENUMERATION:" in out["augmented"]
                    and any(
                        rec in out["augmented"]
                        for rec in _enum["records"]
                    )
                )
        else:
            ok = (
                not out["enumeration_active"]
                and out["enumeration_block"] is None
                and out["augmented"] == case["query"]
            )
        if ok:
            passes += 1
        else:
            failures.append(
                f"{name}: enum_active={out['enumeration_active']} "
                f"block_len={len(out['enumeration_block'] or '')} "
                f"augmented_len={len(out['augmented'])}"
            )
    assert passes == total, (
        f"M120 E battery plumbing: {passes}/{total} pass. Failures: "
        + "; ".join(failures)
    )
    return passes, total


# ── Q10 replay (M125.2 E pilot-readiness regression) ──────────────────────


def test_q10_replay_records_reach_prompt():
    """Q10: 'List all GitHub repositories published under my name.'

    Pre-fix (M125.2 E): enumeration payload assembled (4 records) but
    neither the records nor an ENUMERATION section appeared in the
    assembled prompt. Verified against turn_0010.json from
    sess_20260423_134018_69617.

    Post-fix: records must reach `augmented` and survive absence-guard.
    """
    q = "List all GitHub repositories published under my name."
    # Pre-fix simulation: records in mem_ctx but absence guard overwrites
    # and synthesizer.build_messages never appends memory_context.
    pre = assemble_augmented_user_message(q, pre_fix=True)
    assert pre["guard_fired"] is True, (
        "Pre-fix simulation should reproduce the Q10 Sub-Gate 3 false "
        "fire (enumeration records lack 5/6 query-specific words)."
    )
    assert "orion-ane" not in pre["augmented"], (
        "Pre-fix: Q10 augmented message should NOT contain orion-ane "
        "(verified against turn_0010.json)."
    )

    # Post-fix behavior
    post = assemble_augmented_user_message(q)
    assert post["enumeration_active"] is True, (
        f"post-fix: enumeration should be active, got "
        f"{post['enumeration_active']}"
    )
    assert "[ENUMERATION:" in post["augmented"], (
        f"post-fix: ENUMERATION header missing from augmented: "
        f"{post['augmented'][:200]!r}"
    )
    for repo in ("orion-ane", "ane-compiler", "ngram-engine",
                 "subconscious"):
        assert repo in post["augmented"], (
            f"post-fix: {repo!r} missing from augmented user message"
        )
    assert post["guard_fired"] is False, (
        f"post-fix: absence-guard must not fire on enumeration turn; "
        f"got guard_fired={post['guard_fired']}"
    )
    # mem_ctx preserved for grounding corpus (scrub path)
    assert post["mem_ctx"] is not None and any(
        "orion-ane" in str(m) for m in post["mem_ctx"]
    ), "post-fix: mem_ctx must preserve enumeration records for scrub"


# ── Non-enumeration regression (spot-check) ───────────────────────────────


_NON_ENUMERATION_QUERIES = [
    "what compression does ANE use",             # M125.2 E Q3 (narrative)
    "what was the cold prefill on main 25",      # M125.2 F T21-style
    "how is the prefix cache working today",     # recall_primary
    "walk me through fix surface m115 m118",     # narrative synthesis
    "2 turns ago",                                # turn_recency_bridge
    "what happened two sessions before last",    # session_recency_bridge
    "hey",                                        # phatic
    "tell me about the 1B drafter",              # canonical_lookup
    "what is information theory",                # definitional
    "what is the session id",                     # probe
]


def test_non_enumeration_regression():
    """Non-enumeration queries must not acquire an enumeration block.

    If the fix over-applies, any of these queries would gain an
    enumeration section and the prompt would shift. Asserts the
    assembly is byte-identical to the raw message for each archetype.
    """
    failures = []
    for q in _NON_ENUMERATION_QUERIES:
        out = assemble_augmented_user_message(q)
        if out["enumeration_active"]:
            failures.append(f"{q!r}: enumeration_active=True (over-apply)")
        if out["enumeration_block"] is not None:
            failures.append(f"{q!r}: enumeration_block populated")
        if out["augmented"] != q:
            failures.append(
                f"{q!r}: augmented != message "
                f"({out['augmented'][:80]!r})"
            )
    assert not failures, (
        "Non-enumeration regression: "
        + "; ".join(failures)
    )


# ── Production-code surface verification ──────────────────────────────────


def test_production_surface_has_plumbing():
    """Verify the fix landed in midas_ui.py (not just in the
    simulator). Greps for the m125_3_e_enumeration_plumbing marker and
    both load-bearing code paths.
    """
    ui_path = os.path.normpath(
        os.path.join(_AGENT, "midas_ui.py"))
    with open(ui_path, encoding="utf-8") as _f:
        src = _f.read()
    assert "m125_3_e_enumeration_plumbing" in src, (
        "fix marker missing from midas_ui.py"
    )
    assert "_enumeration_block" in src, (
        "_enumeration_block local missing from midas_ui.py"
    )
    assert "_enumeration_active" in src, (
        "_enumeration_active local missing from midas_ui.py"
    )
    # Absence-guard Sub-Gate 3 must be gated on _enumeration_active
    assert "not _enumeration_active" in src, (
        "absence-guard Sub-Gate 3 is not enumeration-aware"
    )
    # Block must be added to the blocks list (indented body, not
    # comment); use a tight multiline match.
    assert "if _enumeration_block:\n                blocks.append(_enumeration_block)" in src, (
        "_enumeration_block not appended to blocks in user-message assembly"
    )


# ── Runner ────────────────────────────────────────────────────────────────

_TESTS = [
    test_m120_e_battery_plumbing,
    test_q10_replay_records_reach_prompt,
    test_non_enumeration_regression,
    test_production_surface_has_plumbing,
]


def _run() -> int:
    failed = 0
    m120_passes = m120_total = 0
    for fn in _TESTS:
        name = fn.__name__
        try:
            rv = fn()
            if name == "test_m120_e_battery_plumbing" and rv:
                m120_passes, m120_total = rv
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} test functions passed")
    if m120_total:
        print(f"M120 E battery (plumbing replay): "
              f"{m120_passes}/{m120_total} archetypes pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
