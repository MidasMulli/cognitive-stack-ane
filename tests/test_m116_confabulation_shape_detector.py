"""M116 Stream A tests — confabulation_shape_detector + guard.

Pilot-turn replay fixtures + synthetic coverage. 9 primitive cases +
2 guard behavior cases. All must pass.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m116_confabulation_shape_detector.py
    (or pytest orion-ane/tests/test_m116_confabulation_shape_detector.py)

Fixture source:
    data/session_logs/sess_20260421_143211_70559/turn_00{06,09,11,14,15}.json
"""

import os
import sys
import json
import unittest

# Make the agent module importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT = os.path.normpath(os.path.join(_HERE, "..", "agent"))
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from confabulation_shape_detector import (  # noqa: E402
    is_confabulation_shape,
    apply_confabulation_guard,
    ABSTAIN_MESSAGE,
)

_PILOT_DIR = os.path.normpath(
    os.path.join(
        _HERE, "..", "..",
        "data", "session_logs", "sess_20260421_143211_70559",
    )
)


def _load_turn(n: int) -> dict:
    path = os.path.join(_PILOT_DIR, f"turn_{n:04d}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_fixture(turn: dict):
    """Pull the four inputs the detector consumes from a turn log."""
    content = (turn.get("generation", {}) or {}).get("response_text", "") or ""
    retrieval = turn.get("retrieval", {}) or {}
    retrieval_hits = retrieval.get("recall_filtered", []) or []
    tool_calls_made = (turn.get("routing", {}) or {}).get("tools_called", []) or []
    # grounded_memory surrogate: mem_ctx_text from assembly; fallback to
    # per-query memories if present. Pilot logs expose mem_ctx_text.
    ctx = turn.get("context", {}) or {}
    grounded_memory = ctx.get("mem_ctx_text", []) or []
    return content, retrieval_hits, tool_calls_made, grounded_memory


class TestConfabulationShapeDetector(unittest.TestCase):
    # ── Pilot replay cases ──────────────────────────────────────────

    def test_01_T14_replay_tool_substitution(self):
        """T14: model said "checked online" (paraphrased by its own
        response 'user reports on Reddit describe...') after a tool
        DID run (browse_search). Signal 1 must NOT false-fire because
        browse_search is in tool_calls_made. BUT: T14's scrub stripped
        a fabricated '15% lift' sentence, and the actual response
        contains specifics without memory support — Signal 2 fires."""
        turn = _load_turn(14)
        content, rh, tcm, gm = _extract_fixture(turn)

        # Primary check: Signal 2 fires — specific claims, empty recall,
        # empty grounded memory.
        verdict, diag = is_confabulation_shape(content, rh, tcm, gm)
        self.assertTrue(verdict, f"T14 should flag; diag={diag}")
        self.assertTrue(diag["signal_2_fired"],
                        f"T14 Signal 2 expected; diag={diag}")

        # Signal 1 should NOT fire because browse_search ran.
        self.assertFalse(diag["signal_1_fired"],
                         "Signal 1 must not false-fire when "
                         "browse_search is in tool_calls_made")

    def test_02_T14_synthetic_tool_substitution(self):
        """Synthetic T14 variant: the "checked online" phrasing with
        NO tool call. This is the K4 new mechanism T14 first surfaced
        in M114 before browse_search was wired. Signal 1 must fire."""
        content = ("I checked online and found that Opus 4.7 has a 1M "
                   "context window and a 10% lift over Opus 4.6.")
        verdict, diag = is_confabulation_shape(
            content,
            retrieval_hits=[],
            tool_calls_made=[],
            grounded_memory=[],
        )
        self.assertTrue(verdict)
        self.assertTrue(diag["signal_1_fired"])
        # Signal 2 should also fire (specific claims, empty grounding).
        self.assertTrue(diag["signal_2_fired"])

    def test_03_T15_replay_opinion_from_void(self):
        """T15: "what is your opinion on Opus 4.7?" — zero retrieval,
        zero tools, content chains authoritative assertion + specific
        claims. Signal 2 and/or Signal 3 must fire."""
        turn = _load_turn(15)
        content, rh, tcm, gm = _extract_fixture(turn)
        verdict, diag = is_confabulation_shape(content, rh, tcm, gm)
        self.assertTrue(verdict, f"T15 should flag; diag={diag}")
        self.assertTrue(
            diag["signal_2_fired"] or diag["signal_3_fired"],
            f"T15 Signal 2 or 3 expected; diag={diag}",
        )

    def test_04_T6_replay_entropy_metaphor(self):
        """T6: "how does entropy apply to our work" — authoritative
        assertion language ("we use it", "our work") with zero recall
        and zero grounded memory. Signal 3 fires (Signal 2 may also
        fire on quoted material like "decay"/"centroid drift")."""
        turn = _load_turn(6)
        content, rh, tcm, gm = _extract_fixture(turn)
        verdict, diag = is_confabulation_shape(content, rh, tcm, gm)
        self.assertTrue(verdict, f"T6 should flag; diag={diag}")
        # Either Signal 2 (specific claims) or Signal 3 (authoritative)
        # is acceptable — the key point is empty grounding triggers it.
        self.assertTrue(
            diag["signal_2_fired"] or diag["signal_3_fired"],
            f"T6 Signal 2 or 3 expected; diag={diag}",
        )

    def test_05_T9_replay_grounded_passes(self):
        """T9: ANE content-correct per A3 — vault_read ran, recall
        yielded 8 hits. Content is technically correct and supported.
        Verdict must be False."""
        turn = _load_turn(9)
        content, rh, tcm, gm = _extract_fixture(turn)
        # Pilot log shows recall_filtered=8 and vault_read ran.
        self.assertTrue(len(rh) > 0, "T9 fixture should have hits")
        verdict, diag = is_confabulation_shape(content, rh, tcm, gm)
        self.assertFalse(
            verdict,
            f"T9 grounded should pass (retrieval_hits present); "
            f"diag={diag}",
        )

    def test_06_T11_replay_grounded_passes(self):
        """T11: ANE Enclave — vault_read ran, 8 recall hits, content
        is content-correct. Must pass."""
        turn = _load_turn(11)
        content, rh, tcm, gm = _extract_fixture(turn)
        self.assertTrue(len(rh) > 0, "T11 fixture should have hits")
        verdict, diag = is_confabulation_shape(content, rh, tcm, gm)
        self.assertFalse(
            verdict,
            f"T11 grounded should pass; diag={diag}",
        )

    def test_07_synthetic_grounded_positive(self):
        """Specific claims + retrieval hits supporting. No signal
        fires because grounding is present."""
        content = ("ane-compiler v1.0 shipped on 2026-03-30. "
                   "The 8B Q8 pipeline runs at 7.9 tok/s.")
        rh = [
            {"text": "ane-compiler v1.0 shipped Main 26 on 2026-03-30",
             "score": 0.93},
            {"text": "8B Q8 ANE 7.9 tok/s 72 dispatches",
             "score": 0.88},
        ]
        verdict, diag = is_confabulation_shape(
            content,
            retrieval_hits=rh,
            tool_calls_made=[],
            grounded_memory=[],
        )
        self.assertFalse(verdict, f"Grounded positive should pass; "
                                  f"diag={diag}")

    def test_08_synthetic_ungrounded_positive(self):
        """Content with specific claims + authoritative framing + NO
        support on any channel. Must flag."""
        content = ("Our system shipped a 45 TFLOPS throughput record "
                   "last Tuesday. We measured 99.2% acceptance on "
                   "the benchmark.")
        verdict, diag = is_confabulation_shape(
            content,
            retrieval_hits=[],
            tool_calls_made=[],
            grounded_memory=[],
        )
        self.assertTrue(verdict, f"Ungrounded positive should flag; "
                                 f"diag={diag}")
        self.assertTrue(diag["signal_2_fired"] or
                        diag["signal_3_fired"])

    def test_09_empty_content_passes(self):
        """Edge case: empty content must not flag."""
        for empty in ("", "   ", None):
            verdict, diag = is_confabulation_shape(
                empty or "",
                retrieval_hits=[],
                tool_calls_made=[],
                grounded_memory=[],
            )
            self.assertFalse(verdict,
                             f"Empty content must not flag: {empty!r}")

    # ── Guard behavior cases ────────────────────────────────────────

    def test_10_guard_fires_on_T14_synthetic(self):
        """Guard with shape (a) REPLACE must swap the response for the
        abstain message when the detector flags."""
        content = "I checked online and Opus 4.7 lifts task success 15%."
        out, diag = apply_confabulation_guard(
            content,
            retrieval_hits=[],
            tool_calls_made=[],
            grounded_memory=[],
        )
        self.assertTrue(diag["verdict"])
        self.assertEqual(out, ABSTAIN_MESSAGE)

    def test_11_guard_passthrough_on_T9(self):
        """Guard with shape (a) must NOT swap the response when the
        detector does not flag (T9 grounded)."""
        turn = _load_turn(9)
        content, rh, tcm, gm = _extract_fixture(turn)
        out, diag = apply_confabulation_guard(content, rh, tcm, gm)
        self.assertFalse(diag["verdict"])
        self.assertEqual(out, content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
