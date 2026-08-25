"""M122 A1 — Context truncation guard tests.

Validates that canonical-reserve + max_chars=2400 close the T51/T68/T82
synthesis-residual turns diagnosed in M121 A without regressing non-
sub-shape-B renders.

Session under study: sess_20260421_202025_96397
  T51 query: "remind me what tps Llama-1B hits on the combined stack"
  T68 query: "what's the extraction recall at 61% — which model tier is that?"
  T82 query: "in session M42 we fixed the 1/20 battery pass rate — what was the specific root cause?"

Mechanism: `present()` reserves up to CANONICAL_RESERVE_CHARS for the
highest-scoring canonical that would NOT render under a plain budget
walk. `_build_per_query_block()` calls with max_chars=2400 (K2 fired)
so query-relevant canonicals at pool pos 4-6 surface even when summary-
level records dominate top-K.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

REPO = "/Users/midas/Desktop/cowork"
sys.path.insert(0, os.path.join(REPO, "vault/subconscious"))

from multi_path_retrieve import present  # noqa: E402

SESSION_DIR = os.path.join(REPO, "data/session_logs/sess_20260421_202025_96397")


def load_turn(tid: str):
    with open(os.path.join(SESSION_DIR, f"turn_{tid}.json")) as f:
        return json.load(f)


class TestT51Replay(unittest.TestCase):
    """T51: canonical at pool pos 4 score 0.798 must render despite summary
    memory at pool pos 0 score 2.0 consuming most of the budget."""

    def test_t51_canonical_renders(self):
        d = load_turn("0051")
        q = d["input"]["query"]
        rf = d["retrieval"]["recall_filtered"]
        out = present(rf, q, max_chars=2400)
        # Canonical content markers
        self.assertIn("50.2 tok/s", out, "T51 canonical tok/s marker absent")
        self.assertIn("Llama-1B", out, "T51 canonical model marker absent")
        self.assertIn("[canonical]", out, "T51 canonical tag prefix absent")

    def test_t51_canonical_reserve_closes_per_query_chars_22_case(self):
        """The pre-fix 22-char case was bare header. Post-fix must have
        rendered memory bodies."""
        d = load_turn("0051")
        q = d["input"]["query"]
        rf = d["retrieval"]["recall_filtered"]
        out = present(rf, q, max_chars=2400)
        self.assertGreater(len(out), 500, "T51 per_query block still near-empty")


class TestT68Replay(unittest.TestCase):
    """T68: Tier 1 canonical at pool pos 5 score 0.714 must render despite
    Main 45 summary at pool pos 0 score 1.568 consuming most of the budget."""

    def test_t68_tier_canonical_renders(self):
        d = load_turn("0068")
        q = d["input"]["query"]
        rf = d["retrieval"]["recall_filtered"]
        out = present(rf, q, max_chars=2400)
        # At least one Tier 1 marker must be present
        markers = ["3B solo", "Tier 1 confirmed", "61%"]
        hit = any(m in out for m in markers)
        self.assertTrue(hit, f"T68: no Tier 1 canonical marker in output "
                              f"(checked {markers})")


class TestT82Replay(unittest.TestCase):
    """T82: pool pos 2 full-answer memory ("Stop token IDs" +
    "stream-before-stop") must render within budget."""

    def test_t82_full_answer_renders(self):
        d = load_turn("0082")
        q = d["input"]["query"]
        rf = d["retrieval"]["recall_filtered"]
        out = present(rf, q, max_chars=2400)
        self.assertIn("Stop token IDs", out, "T82 stop-token marker absent")
        self.assertIn("stream-before-stop", out,
                      "T82 stream-ordering marker absent")


class TestRegression(unittest.TestCase):
    """Verify non-sub-shape-B rendering unchanged when no canonical in pool
    or top-1 is already canonical."""

    def test_no_canonical_in_pool(self):
        pool = [
            {"text": "Alpha finding.", "score": 1.0,
             "source_role": "claude_automemory", "role_weight": 1.0},
            {"text": "Beta finding.", "score": 0.9,
             "source_role": "claude_automemory", "role_weight": 1.0},
        ]
        out = present(pool, "what is alpha", max_chars=1500)
        self.assertIn("Alpha finding", out)
        self.assertIn("Beta finding", out)
        # No canonical tag injected anywhere
        self.assertNotIn("[canonical]", out)

    def test_canonical_is_top_one(self):
        """When top-1 is already canonical, reserve should be no-op: the
        canonical renders naturally at position 1."""
        pool = [
            {"text": "Canonical fact.", "score": 2.0,
             "source_role": "canonical", "role_weight": 1.3},
            {"text": "Secondary summary.", "score": 1.5,
             "source_role": "claude_automemory", "role_weight": 1.0},
        ]
        out = present(pool, "test query", max_chars=1500)
        # Canonical must appear once, not twice
        self.assertEqual(out.count("Canonical fact"), 1,
                         "Top-1 canonical rendered twice — reserve mis-fired")
        self.assertIn("Secondary summary", out)

    def test_empty_pool(self):
        self.assertEqual(present([], "q", max_chars=1500), "")

    def test_all_fit_naturally(self):
        """Short pool where everything fits within max_chars — reserve
        should not double-render the canonical."""
        pool = [
            {"text": "One.", "score": 1.0,
             "source_role": "claude_automemory", "role_weight": 1.0},
            {"text": "Two.", "score": 0.5,
             "source_role": "canonical", "role_weight": 1.3},
            {"text": "Three.", "score": 0.4,
             "source_role": "claude_automemory", "role_weight": 1.0},
        ]
        out = present(pool, "q", max_chars=1500)
        self.assertEqual(out.count("[canonical] Two"), 1,
                         "Canonical double-rendered")


class TestPrefixCacheStability(unittest.TestCase):
    """Measure assembled-prompt per_query_block length delta between
    max_chars=1500 (pre-M122-A1) and max_chars=2400 (post).

    The prefix cache in qwen_spec_decode_server.py is keyed on the
    SYSTEM-prefix hash (Main 25 design). The per_query_block rides in the
    user-message tail (midas_ui.py:3840-3850). So even a 2x per_query
    growth does not invalidate system-prefix cache — ΔHit-rate expected
    ~0%. This test records the size delta for the readiness report; the
    actual cache hit-rate is measured in Stream B.
    """

    def test_prefix_cache_delta_measurement(self):
        battery = ["0051", "0068", "0082", "0001", "0014", "0030",
                   "0050", "0070", "0080", "0090"]
        before_total = 0
        after_total = 0
        n = 0
        for tid in battery:
            path = os.path.join(SESSION_DIR, f"turn_{tid}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                d = json.load(f)
            rf = d["retrieval"]["recall_filtered"]
            q = d["input"]["query"]
            if not rf:
                continue
            b = present(rf, q, max_chars=1500)
            a = present(rf, q, max_chars=2400)
            before_total += len(b)
            after_total += len(a)
            n += 1
        self.assertGreater(n, 0, "No turns sampled")
        avg_growth_chars = (after_total - before_total) / n
        # Sanity: growth is positive and bounded (not 10x)
        self.assertGreaterEqual(avg_growth_chars, 0)
        self.assertLess(avg_growth_chars, 2500)


if __name__ == "__main__":
    unittest.main()
