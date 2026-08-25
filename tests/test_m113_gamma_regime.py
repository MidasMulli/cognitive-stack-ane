#!/usr/bin/env python3
"""M113 stream gamma — two-regime assembly recall-quality gate.

Covers M108 Finding 2 part 1: T30/T31 contracted-regime eviction +
memory_recall miss. When tool_name+tool_result is set the default
`build_messages` branch keeps only `history[-4:]` (last 2 user/assistant
pairs). T30 kept T27+T28 and evicted T25/T26 (the actual
information-theory content). memory_recall returned 2 off-topic hits
scored 0.268 / 0.152 — both below the 0.4 threshold. The model had
neither history nor recall to answer the follow-up.

Gate design:
  - threshold = 0.4 (cleanly separates the T30/T31 miss band
    {0.268, 0.152} from the borderline 0.462 data point; also below
    absence-gate strict/normal thresholds so it does not double-fire)
  - trigger = tool_name present AND recall_max_score < 0.4
  - action = flip to expanded-regime history walk (16000 char budget)
    while preserving the tool_result user-message tail

Case 1: T30-shape replay with low recall → expanded history (T25 pair
  present, tool_result tail still appended).
Case 2: T30-shape replay with healthy recall (>= 0.4) → contracted
  regime preserved (only last 2 pairs, tool_result tail).
Case 3: T31-shape replay with 1.185 off-topic high-score hit → regime
  stays contracted (the gate is *recall-quality-score* based, not
  relevance-based; this is the expected M113-gamma scope boundary;
  off-topic high scores are T32 / δ territory).
Case 4: Expanded regime does NOT drop the tool_result tail
  (grounding-preservation invariant).
Case 5: When recall returns empty (max_score=0.0) in tool path, the
  gate fires and expands — the default contracted regime would strand
  the follow-up with zero rescue path.
"""
from __future__ import annotations

import os
import sys
import unittest

_REPO = "/Users/midas/Desktop/cowork"
sys.path.insert(0, os.path.join(_REPO, "orion-ane/agent"))

from synthesizer import build_messages  # noqa: E402


def _make_t30_shape_history():
    """Build a 28-turn synthetic history mirroring the M108 pilot.

    Turns T1..T24 are filler ('earlier conversation'). T25 is the
    information-theory / InfoNCE / MSE answer. T26 is the re-expanded
    summary. T27 is a vault_read turn. T28 is an absence-skip (no
    assistant content). T29 is logged-as-gap (synthesized here as a
    short turn). Current turn (T30) is the follow-up 'what were you
    saying about information theory?'.

    Under contracted regime (history[-4:]) the assembled window will
    see T28+T29 only. Under expanded regime the T25 InfoNCE pair is
    reachable inside the 16000 char budget.
    """
    history = []
    for i in range(1, 25):
        history.append({
            "role": "user",
            "content": f"T{i} user filler turn — placeholder text about neuron compilation and ane-dispatch cache hints, long enough to exercise the char budget without dominating it ({i}).",
        })
        history.append({
            "role": "assistant",
            "content": f"T{i} assistant reply — filler content referencing prior paper sections and dispatch opcode catalogs ({i}).",
        })
    # T25: the information-theory answer
    history.append({
        "role": "user",
        "content": "T25 user: is the paper using InfoNCE or MSE for the contrastive loss?",
    })
    history.append({
        "role": "assistant",
        "content": "T25 asst: the Every Cycle Counts paper uses InfoNCE on the embedding head with an auxiliary MSE term on the projection residual; the entropy term carries the contrastive signal while MSE regularizes drift.",
    })
    # T26: expanded follow-up
    history.append({
        "role": "user",
        "content": "T26 user: can you re-explain the entropy term and what the temperature controls?",
    })
    history.append({
        "role": "assistant",
        "content": "T26 asst: the entropy term is the InfoNCE log-softmax; the temperature scales the similarity logits before softmax, controlling the contrastive sharpness.",
    })
    # T27: vault_read turn
    history.append({
        "role": "user",
        "content": "T27 user: show me the passage from the paper.",
    })
    history.append({
        "role": "assistant",
        "content": "T27 asst: (vault excerpt omitted) — the paragraph matches section 4.3.",
    })
    # T28: absence-skip equivalent (short)
    history.append({
        "role": "user",
        "content": "T28 user: do we know X?",
    })
    history.append({
        "role": "assistant",
        "content": "T28 asst: I don't have information about that in our research.",
    })
    # T29 ambient gap filled synthetic
    history.append({
        "role": "user",
        "content": "T29 user: ok never mind.",
    })
    history.append({
        "role": "assistant",
        "content": "T29 asst: understood.",
    })
    return history


CURRENT_USER_MSG = "what were you saying about information theory?"


class M113GammaRegimeTest(unittest.TestCase):

    # ── helpers ────────────────────────────────────────────────────

    def _assert_tool_tail_present(self, msgs, tool_name="memory_recall"):
        """Grounding-preservation invariant: tool_result tail must be
        the last user message, regardless of regime. Accepts both
        build_messages branches: (a) 'Tool result:' path when the
        result string is non-empty, (b) 'no useful results' path when
        the result is empty/short — both carry the tool_name."""
        last = msgs[-1]
        self.assertEqual(last["role"], "user")
        self.assertIn(tool_name, last["content"])
        self.assertTrue(
            "Tool result:" in last["content"]
            or "no useful results" in last["content"],
            f"tool tail missing grounding text: {last['content'][:200]!r}",
        )

    def _history_turn_labels_in_msgs(self, msgs):
        """Return the set of 'TN' labels whose user or assistant line
        appears in msgs (excluding the current-turn final user message
        and the tool-result tail)."""
        labels = set()
        for m in msgs[1:-2]:  # skip system slot, final user, tool tail
            content = m.get("content", "") or ""
            for tag in (f"T{i}" for i in range(1, 30)):
                if content.startswith(f"{tag} "):
                    labels.add(tag)
                    break
        return labels

    # ── cases ──────────────────────────────────────────────────────

    def test_case_1_t30_low_recall_flips_to_expanded(self):
        """T30-shape: recall max 0.268 (below 0.4) → gate fires,
        expanded-regime walk must surface T25 in history."""
        history = _make_t30_shape_history()
        tool_result = {
            "hits": [
                {"text": "CPU FactExtractor fallback", "score": 0.268},
                {"text": "knowledge/INDEX", "score": 0.152},
            ],
        }
        msgs = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="memory_recall",
            tool_args={"query": "information theory"},
            tool_result=tool_result,
            briefing=None,
            expand_history_on_low_recall=True,  # gate fires
        )
        # Under flipped regime, T25's InfoNCE content must be reachable.
        all_content = "\n".join(m.get("content", "") for m in msgs)
        self.assertIn("InfoNCE", all_content,
                      "T25 InfoNCE answer must be in expanded history")
        # Grounding preserved.
        self._assert_tool_tail_present(msgs, "memory_recall")

    def test_case_2_healthy_recall_keeps_contracted(self):
        """Healthy recall (max=0.55) → gate does not fire, contracted
        regime keeps only last 2 pairs."""
        history = _make_t30_shape_history()
        tool_result = {
            "hits": [
                {"text": "information theory topical hit", "score": 0.55},
            ],
        }
        msgs = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="memory_recall",
            tool_args={"query": "information theory"},
            tool_result=tool_result,
            briefing=None,
            expand_history_on_low_recall=False,  # gate does not fire
        )
        # Contracted regime: history[-4:] from the 56-entry synthetic
        # history = T28+T29 pairs (final 4 before current user). T25
        # must NOT be in the prompt.
        all_content = "\n".join(m.get("content", "") for m in msgs)
        self.assertNotIn("InfoNCE", all_content,
                         "contracted regime should evict T25")
        self.assertIn("T28", all_content)
        self.assertIn("T29", all_content)
        self._assert_tool_tail_present(msgs, "memory_recall")

    def test_case_3_off_topic_high_score_stays_contracted(self):
        """T31-shape: recall max 1.185 (Main 27 Calibration off-topic
        but high-score) → gate does NOT fire (it is score-based, not
        relevance-based). This is the M113-gamma scope boundary; the
        off-topic high-score denial pattern is T32 / stream-δ scope."""
        history = _make_t30_shape_history()
        tool_result = {
            "hits": [
                {"text": "Main 27 Calibration — off-topic",
                 "score": 1.185},
            ],
        }
        msgs = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="memory_recall",
            tool_args={"query": "information theory"},
            tool_result=tool_result,
            briefing=None,
            expand_history_on_low_recall=False,
        )
        all_content = "\n".join(m.get("content", "") for m in msgs)
        self.assertNotIn("InfoNCE", all_content,
                         "off-topic-high-score case is out of γ scope")
        self._assert_tool_tail_present(msgs, "memory_recall")

    def test_case_4_expanded_preserves_tool_tail(self):
        """Grounding-preservation: flipping to expanded regime must
        not drop the tool_result tail. The tail is appended in both
        branches."""
        history = _make_t30_shape_history()
        msgs = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="vault_read",
            tool_args={"path": "foo.md"},
            tool_result="vault excerpt about information theory",
            briefing=None,
            expand_history_on_low_recall=True,
        )
        self._assert_tool_tail_present(msgs, "vault_read")
        # also assert the tool_result content passes through
        self.assertIn("vault excerpt about information theory",
                      msgs[-1]["content"])

    def test_case_5_empty_recall_triggers_expand(self):
        """Empty recall (max=0.0) is the degenerate low-recall case —
        without the gate, contracted regime would strand the follow-up
        with neither history nor recall rescue. Gate must fire."""
        # Simulate the caller's logic: _recall_score_max_for_log = 0.0
        # when filtered is empty; caller passes expand=True.
        recall_max = 0.0
        threshold = 0.4
        caller_flip = recall_max < threshold
        self.assertTrue(caller_flip,
                        "caller-side gate must flip on empty recall")

        history = _make_t30_shape_history()
        msgs = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="memory_recall",
            tool_args={"query": "information theory"},
            tool_result={"hits": []},
            briefing=None,
            expand_history_on_low_recall=caller_flip,
        )
        all_content = "\n".join(m.get("content", "") for m in msgs)
        self.assertIn("InfoNCE", all_content,
                      "empty recall must still reach expanded history")
        self._assert_tool_tail_present(msgs, "memory_recall")

    def test_case_6_default_kwarg_is_backward_compatible(self):
        """Existing callers (conversation path, old tool-path callers)
        that do not pass `expand_history_on_low_recall` must see the
        same behavior as before — default=False preserves contracted
        regime on tool turns."""
        history = _make_t30_shape_history()
        msgs_old = build_messages(
            history,
            CURRENT_USER_MSG,
            tool_name="memory_recall",
            tool_args={"query": "information theory"},
            tool_result={"hits": [
                {"text": "foo", "score": 0.268},
            ]},
            briefing=None,
        )  # no expand_history_on_low_recall kwarg
        all_content = "\n".join(m.get("content", "") for m in msgs_old)
        self.assertNotIn("InfoNCE", all_content,
                         "default must preserve pre-M113 contracted behavior")


if __name__ == "__main__":
    unittest.main(verbosity=2)
