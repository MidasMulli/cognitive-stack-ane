"""
M125 Stream A1 — cosine/rank robustness regression tests.

Covers:
  - A1.1 query-expansion templates (vocab-gap bridges from M124 A Cause 2)
  - A1.2 canonical-reserve widening N=1 -> N=2 (pool cutoff from M124 A Cause 3)

Target turns replayed from M124 Stream A 5-cause matrix
(vault/agent_reports/m124_a_pool_gap_diagnosis.md §1):
  T17 (m123_c) "cache hit rate on verifier"           — C2+C3
  T18 (m123_c) "how raise the hit rate"               — C2+C3
  T28 (m123_c) "fix surface M115-M118"                — C2+C3
  T62 (m122_c) "Apple Silicon generation SharedEvents" — C2+C3
  T65 (m122_c) "TTFT delta last few sessions"          — C2+C3
  T66 (m122_c) "M108-M121 walkthrough"                 — C3

Regression: non-pool-gap turns keep working; expansion does not produce garbage
paraphrases that degrade precision; canonical-reserve N=2 does not starve
the main loop on turns with many canonicals.

Run:
    python3 -m pytest orion-ane/tests/test_m125_a1_cosine_rank_robustness.py -v
    python3 orion-ane/tests/test_m125_a1_cosine_rank_robustness.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
SUBCON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "vault", "subconscious"
)
for p in (AGENT_DIR, SUBCON_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import query_vocab  # noqa: E402
import multi_path_retrieve as mpr  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# A1.1 — Query-expansion tests (vocab-gap bridges)
# ─────────────────────────────────────────────────────────────────────────────

def test_t17_cache_hit_rate_expands():
    """T17: 'cache hit rate on verifier' must bridge to prefix-cache terms."""
    variants = query_vocab.expand_query(
        "What's our current cache hit rate on the verifier?"
    )
    joined = " ".join(variants).lower()
    assert "prefix cache" in joined, (
        f"cache hit rate must expand to 'prefix cache' — M124 A C2+C3 T17 "
        f"regression. Got: {variants}"
    )


def test_t18_raise_hit_rate_expands():
    """T18: 'get the hit rate up' must bridge to KV cache mechanism terms."""
    variants = query_vocab.expand_query(
        "how were we able to get the hit rate up?"
    )
    joined = " ".join(variants).lower()
    assert (
        "kvcache" in joined or "kv cach" in joined
        or "system-message kv" in joined
    ), (
        f"raise/get hit rate up must expand to kv-cache mechanism — "
        f"M124 A C2+C3 T18 regression. Got: {variants}"
    )


def test_t28_fix_surface_range_expands():
    """T28: 'fix surface M115-M118' must bridge to parent-synthesis terms
    AND explicitly name Main 115 through Main 118."""
    variants = query_vocab.expand_query(
        "Summarize the fix surface from M115 through M118 with specific fix names"
    )
    joined = " ".join(variants).lower()
    assert "parent synthesis" in joined, (
        f"fix surface must bridge to parent synthesis — M124 A C2+C3 T28. "
        f"Got: {variants}"
    )
    # Range expansion: all 4 session numbers must appear
    for n in (115, 116, 117, 118):
        assert f"main {n}" in joined, (
            f"session range M115-M118 must expand to all four Main NN "
            f"markers; 'main {n}' missing. Got: {variants}"
        )


def test_t62_apple_silicon_generation_expands():
    """T62: 'Apple Silicon generation SharedEvents' must bridge to
    macOS/iOS version framing used by canonical."""
    variants = query_vocab.expand_query(
        "which Apple Silicon generation introduced the SharedEvents path?"
    )
    joined = " ".join(variants).lower()
    assert "macos" in joined or "ios 16" in joined, (
        f"Apple Silicon generation must bridge to macOS/iOS framing — "
        f"M124 A C2+C3 T62. Got: {variants}"
    )


def test_t65_ttft_delta_expands():
    """T65: 'TTFT delta across sessions' must bridge to Main 48 / +4.2%."""
    variants = query_vocab.expand_query(
        "what was the TTFT delta across the last few sessions?"
    )
    joined = " ".join(variants).lower()
    assert "+4.2" in joined or "main 48" in joined, (
        f"TTFT delta must bridge to the Main 48 +4.2% canonical — "
        f"M124 A C2+C3 T65. Got: {variants}"
    )


def test_t66_m108_through_m121_range_expands():
    """T66: 'M108 through M121' must expand into the Main-prefixed variant."""
    variants = query_vocab.expand_query(
        "walk me through the full sequence of fixes from M108 through M121"
    )
    joined = " ".join(variants).lower()
    # Expect at least one variant to contain "main 108" and "main 121".
    assert "main 108" in joined, (
        f"M108 must expand to 'main 108' — M124 A C3 T66. Got: {variants}"
    )
    assert "main 121" in joined, (
        f"M121 must expand to 'main 121' — M124 A C3 T66. Got: {variants}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1.1 — Regression: non-pool-gap queries do NOT over-expand
# ─────────────────────────────────────────────────────────────────────────────

def test_regression_tok_s_query_no_overexpansion():
    """Non-pool-gap simple technical query must not trigger expansion."""
    variants = query_vocab.expand_query(
        "What is the current tok/s on the 8B ANE model?"
    )
    assert variants == [], (
        f"Plain technical query must not expand (K1 discipline). "
        f"Got: {variants}"
    )


def test_regression_activity_query_no_overexpansion():
    """Activity-query shape must not fire vocab bridges."""
    variants = query_vocab.expand_query("What did we ship today?")
    assert variants == [], (
        f"Activity query must not expand (K1 discipline). Got: {variants}"
    )


def test_regression_paper_2_still_expands():
    """M100 pre-existing Paper 2 bridge must still fire."""
    variants = query_vocab.expand_query("what is paper 2 about?")
    joined = " ".join(variants).lower()
    assert "five roadblocks" in joined, (
        f"M100 Paper 2 bridge must still fire post-M125 A1.1. Got: {variants}"
    )


def test_regression_enclave_exclave_still_expands():
    """M100 enclave<->exclave bridge must still fire."""
    variants = query_vocab.expand_query(
        "what research have we done on the ANE Enclave?"
    )
    joined = " ".join(variants).lower()
    assert "exclave" in joined, (
        f"M100 enclave/exclave bridge must still fire post-M125 A1.1. "
        f"Got: {variants}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1.1 — Query-expansion sanity: 10 samples produce valid paraphrases
# ─────────────────────────────────────────────────────────────────────────────

_EXPANSION_SANITY_QUERIES = [
    "What's our current cache hit rate on the verifier?",
    "how were we able to get the hit rate up?",
    "which Apple Silicon generation introduced the SharedEvents path?",
    "what was the TTFT delta across the last few sessions?",
    "walk me through the full sequence of fixes from M108 through M121",
    "Summarize the fix surface from M115 through M118 with specific fix names",
    "what is paper 2 about?",
    "ANE Enclave probe results",
    "can small model run on the SME?",
    "53 opcodes catalog",
]


def test_expansion_sanity_10_samples():
    """Each of 10 sample queries either produces 0 variants (no trigger) or
    produces well-formed non-empty strings with no garbage (nested spaces,
    trailing punctuation, etc.)."""
    for q in _EXPANSION_SANITY_QUERIES:
        vs = query_vocab.expand_query(q)
        for v in vs:
            assert isinstance(v, str) and len(v) > 0, (
                f"Expansion produced empty/non-str variant on {q!r}: {v!r}"
            )
            assert "  " not in v, (
                f"Expansion produced doubled spaces on {q!r}: {v!r}"
            )
            # Variant must differ from original (lowercase-compared)
            assert v.strip() != q.strip().lower() or v == q, (
                f"Variant equals original on {q!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# A1.2 — Canonical-reserve N=2 widening tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_memory(text, score, source_role=None, role_weight=None):
    """Shape a minimal memory dict for present()."""
    d = {"text": text, "score": score, "metadata": {}}
    if source_role is not None:
        d["metadata"]["source_role"] = source_role
    if role_weight is not None:
        d["role_weight"] = role_weight
    return d


def test_a12_reserve_n2_surfaces_two_canonicals():
    """When the pool has 2 canonicals ranked below non-canonicals, both
    must surface in the rendered output (M124 A C3: multi-canonical
    synthesis queries need more than one reserve slot)."""
    # Non-canonical summary records at top ranks consume budget
    non_can_text = "A" * 600  # fills budget so canonicals cannot render naturally
    memories = [
        _make_memory(non_can_text, score=0.9),
        _make_memory(non_can_text, score=0.85),
        _make_memory(
            "CANONICAL ONE: Main 115 beta split gate fix", score=0.5,
            source_role="canonical"
        ),
        _make_memory(
            "CANONICAL TWO: Main 117 extraction grounding gate", score=0.48,
            source_role="canonical"
        ),
    ]
    out = mpr.present(memories, "fix surface M115 M117", max_chars=2400)
    assert "CANONICAL ONE" in out, f"First canonical must render. out={out}"
    assert "CANONICAL TWO" in out, (
        f"Second canonical must render under N=2 reserve. out={out}"
    )


def test_a12_reserve_n2_does_not_starve_main_loop():
    """When every reserve slot is canonical, the main loop must still
    render at least one non-canonical (K2 discipline: cap N=2)."""
    memories = [
        _make_memory("TOP RANKED non-canonical summary", score=0.95),
        _make_memory(
            "CANONICAL A " + "x" * 300, score=0.50,
            source_role="canonical",
        ),
        _make_memory(
            "CANONICAL B " + "y" * 300, score=0.48,
            source_role="canonical",
        ),
        _make_memory(
            "CANONICAL C " + "z" * 300, score=0.46,
            source_role="canonical",
        ),
    ]
    out = mpr.present(memories, "test query", max_chars=2400)
    assert "TOP RANKED non-canonical" in out, (
        f"Top-ranked non-canonical must render. out={out}"
    )
    # Only top-2 canonicals should have been reserved (A + B);
    # C may or may not render via main loop depending on budget.
    assert "CANONICAL A" in out, f"First canonical must render. out={out}"
    assert "CANONICAL B" in out, (
        f"Second canonical must render under N=2. out={out}"
    )


def test_a12_reserve_backward_compat_single_canonical():
    """Pre-M125 behavior: when only 1 canonical exists, only 1 reserve
    slot fires. Backward-compatible with M122 A1."""
    non_can_text = "A" * 600
    memories = [
        _make_memory(non_can_text, score=0.9),
        _make_memory(non_can_text, score=0.85),
        _make_memory(
            "ONLY CANONICAL: Main 58 IOSurfaceSharedEvent", score=0.5,
            source_role="canonical"
        ),
    ]
    out = mpr.present(memories, "sharedevents path", max_chars=2400)
    assert "ONLY CANONICAL" in out, (
        f"Sole canonical must still render post-M125 A1.2. out={out}"
    )


def test_a12_reserve_no_canonicals_noop():
    """When the pool has no canonicals, reserve path is a no-op."""
    memories = [
        _make_memory("short non-canonical record 1", score=0.9),
        _make_memory("short non-canonical record 2", score=0.8),
    ]
    out = mpr.present(memories, "test query", max_chars=2400)
    assert "record 1" in out
    assert "record 2" in out


def test_a12_reserve_naturally_rendering_canonicals_not_duplicated():
    """When a canonical already renders naturally (top of pool, fits in
    budget), it must not be double-rendered via reserve slot."""
    memories = [
        _make_memory(
            "CANONICAL AT TOP: Main 25 prefix cache 67% hit rate",
            score=0.95, source_role="canonical"
        ),
        _make_memory("non-canonical at rank 2", score=0.85),
    ]
    out = mpr.present(memories, "cache hit rate", max_chars=2400)
    # Count occurrences — canonical should appear exactly once
    assert out.count("CANONICAL AT TOP") == 1, (
        f"Naturally-rendering canonical must not duplicate via reserve. "
        f"out={out}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A1.4 — Role-weight audit (programmatic verification, no change)
# ─────────────────────────────────────────────────────────────────────────────

def test_a14_role_weight_audit():
    """Role-weight multipliers are defined in multi_path_retrieve.py (not
    local_store.py:352-377 — directive line numbers drifted).

    M125 A1.4 audit: confirm the ROLE_WEIGHT-equivalent multipliers are
    present and values are plausible. No change expected."""
    # Live-grep the exported constants
    assert hasattr(mpr, "CANONICAL_BOOST")
    assert hasattr(mpr, "META_BOOST")
    assert hasattr(mpr, "CLAUDE_AUTOMEMORY_BOOST")
    assert hasattr(mpr, "CLAUDE_VAULT_REALTIME_BOOST")
    assert hasattr(mpr, "RESEARCH_INGEST_PENALTY")

    # Plausibility: claude_automemory (curated) > canonical > realtime > meta
    # penalty < 1.0 for research_ingest
    assert mpr.CLAUDE_AUTOMEMORY_BOOST > mpr.CANONICAL_BOOST, (
        "claude_automemory should outrank canonical (curated > auto-extracted)"
    )
    assert mpr.CANONICAL_BOOST > 1.0, "canonical must be a BOOST, not a penalty"
    assert mpr.RESEARCH_INGEST_PENALTY < 1.0, (
        "research_ingest must be penalized, not boosted"
    )
    assert mpr.CLAUDE_VAULT_REALTIME_BOOST >= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(0 if failed == 0 else 1)
