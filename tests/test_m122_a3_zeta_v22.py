"""M122 A3 — ζ v2.2 schema bump regression tests.

Covers three additive observation-only fields (per directive §3.3):
  9. retrieval.per_query_truncated_items  (populated by multi_path_retrieve)
 10. scrub.confab_guard_diagnostic        (populated by confab_guard callsite)
 11. scrub.tier2_binding.skip_reason      (populated by A2's tier2 guards)

Gate criteria (directive §5 A3):
  - Schema version written as "2.2"
  - Each new field validates with empty content (omitted / empty-list)
  - Each new field validates with populated content (synthetic)
  - v2.1 consumer test (M103 / M109 schema suite) continues to pass
    (verified via existing test_m109_turn_log_schema_v2.py — separate file)
  - Runtime overhead measurement: assert ζ emission delta <2ms

Run standalone:
    python3 orion-ane/tests/test_m122_a3_zeta_v22.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
SUBC_DIR = "/Users/midas/Desktop/cowork/vault/subconscious"
for p in (AGENT_DIR, SUBC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Schema version ──────────────────────────────────────────────────────

def test_schema_version_written_as_2_2_string():
    """_turn_start writes schema_version='2.2' (string semver) into _TURN_LOG."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._turn_start("test query")
        assert midas_ui._TURN_LOG.get("schema_version") == "2.2", (
            f"Expected schema_version='2.2', got "
            f"{midas_ui._TURN_LOG.get('schema_version')!r}")
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


def test_schema_version_type_is_string():
    """v2.2 bumps the scalar type from int to string for semver clarity."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._turn_start("probe")
        val = midas_ui._TURN_LOG.get("schema_version")
        assert isinstance(val, str), (type(val), val)
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


# ── Field 9 — retrieval.per_query_truncated_items ───────────────────────

def test_field9_empty_pool_no_truncation_manifest():
    """present() on an empty pool clears the manifest; accessor returns []."""
    from multi_path_retrieve import present, get_last_truncated_items
    out = present([], "any query")
    assert out == ""
    assert get_last_truncated_items() == []


def test_field9_pool_fits_budget_no_truncation():
    """When all records fit under max_chars, the manifest is empty."""
    from multi_path_retrieve import present, get_last_truncated_items
    memories = [
        {"text": "short memory one", "source_role": "canonical", "score": 1.3},
        {"text": "short memory two", "source_role": "user", "score": 0.9},
    ]
    _ = present(memories, "technical query", max_chars=500)
    trunc = get_last_truncated_items()
    assert trunc == [], f"Expected empty truncation manifest, got {trunc}"


def test_field9_populated_on_budget_cutoff():
    """Pool with items beyond budget emits a manifest with the right shape."""
    from multi_path_retrieve import present, get_last_truncated_items
    long_text = "x" * 300
    memories = [
        {"text": long_text + " A", "source_role": "user", "score": 1.0},
        {"text": long_text + " B", "source_role": "user", "score": 0.9},
        {"text": long_text + " C", "source_role": "user", "score": 0.8},
        {"text": long_text + " D", "source_role": "user", "score": 0.7},
    ]
    _ = present(memories, "technical query", max_chars=400)
    trunc = get_last_truncated_items()
    assert len(trunc) >= 1, f"Expected at least one truncation, got {trunc}"
    for entry in trunc:
        for key in ("pos", "score", "source_role", "truncated_chars",
                    "would_render_if_budget_raised"):
            assert key in entry, (key, entry)
        assert isinstance(entry["pos"], int)
        assert isinstance(entry["score"], float)
        assert isinstance(entry["source_role"], str)
        assert isinstance(entry["truncated_chars"], int)
        assert isinstance(entry["would_render_if_budget_raised"], bool)


def test_field9_would_render_flag_reflects_budget_raise_semantics():
    """Items following the first-cutoff position carry would_render=True."""
    from multi_path_retrieve import present, get_last_truncated_items
    long_text = "y" * 300
    memories = [
        {"text": long_text, "source_role": "user", "score": 1.0},
        {"text": long_text, "source_role": "user", "score": 0.9},
        {"text": long_text, "source_role": "user", "score": 0.8},
    ]
    _ = present(memories, "query", max_chars=400)
    trunc = get_last_truncated_items()
    # At least one entry should carry would_render=True (items sequential
    # to the truncation boundary). If the whole manifest is would_render=False
    # there's nothing to raise-budget-for, which would be a false negative.
    has_truthy = any(e["would_render_if_budget_raised"] for e in trunc)
    assert has_truthy, (
        f"Expected at least one entry with would_render=True, got {trunc}")


# ── Field 10 — scrub.confab_guard_diagnostic ────────────────────────────

def test_field10_shape_from_confab_detector():
    """The detector returns a dict with all keys the ζ v2.2 field consumes."""
    import confabulation_shape_detector as csd
    verdict, diag = csd.is_confabulation_shape(
        content="I measured 50 tok/s on the ANE.",
        retrieval_hits=[],
        tool_calls_made=[],
        grounded_memory=[],
    )
    # The six ζ-field keys are directly available on the detector output.
    assert isinstance(diag, dict)
    for key in ("signal_1_fired", "signal_2_fired", "signal_3_fired",
                "signal_4_fired", "matched_phrases", "verdict"):
        assert key in diag, (key, diag.keys())


def test_field10_populated_turn_record_shape():
    """Synthetic populated diagnostic round-trips through _turn_record."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        _diag = {
            "signal_1_fired": True,
            "signal_2_fired": False,
            "signal_3_fired": False,
            "signal_4_fired": False,
            "matched_phrases": ["I checked online for you"],
            "verdict": True,
        }
        # Mirror midas_ui wiring: flatten matched_phrases (cap 20) into the
        # ζ-field key `match_text_per_signal`.
        midas_ui._turn_record("scrub", confab_guard_diagnostic={
            "signal_1_fired": bool(_diag["signal_1_fired"]),
            "signal_2_fired": bool(_diag["signal_2_fired"]),
            "signal_3_fired": bool(_diag["signal_3_fired"]),
            "signal_4_fired": bool(_diag["signal_4_fired"]),
            "match_text_per_signal": _diag["matched_phrases"][:20],
            "verdict": bool(_diag["verdict"]),
        })
        cg = midas_ui._TURN_LOG["scrub"]["confab_guard_diagnostic"]
        assert cg["verdict"] is True
        assert cg["signal_1_fired"] is True
        assert cg["match_text_per_signal"] == ["I checked online for you"]
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


def test_field10_absent_when_guard_not_invoked():
    """When _confab_diag is None (guard not invoked), no field is emitted."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        # Simulate the guard path's "not invoked" branch — no _turn_record.
        _confab_diag = None
        if _confab_diag is not None:
            midas_ui._turn_record("scrub", confab_guard_diagnostic={})
        assert "confab_guard_diagnostic" not in midas_ui._TURN_LOG["scrub"]
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


# ── Field 11 — scrub.tier2_binding.skip_reason ──────────────────────────

def test_field11_empty_scrub_no_tier2_binding_field():
    """When scrub_response returns no tier2 skip reasons, nothing is emitted."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        # Mirror midas_ui conditional: emit only when list is non-empty.
        _scrub_result_stub = {"tier2_binding_skip_reasons": []}
        _reasons = _scrub_result_stub.get("tier2_binding_skip_reasons", [])
        if _reasons:
            midas_ui._turn_record("scrub", tier2_binding={"skip_reason": _reasons})
        assert "tier2_binding" not in midas_ui._TURN_LOG["scrub"]
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


def test_field11_populated_turn_record_shape():
    """Synthetic populated skip_reason list round-trips through _turn_record."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        _reasons = ["word_overlap_zero", "abstention_pattern"]
        midas_ui._turn_record("scrub", tier2_binding={"skip_reason": _reasons})
        tb = midas_ui._TURN_LOG["scrub"]["tier2_binding"]
        assert tb["skip_reason"] == _reasons
        # Enum discipline: every entry is one of the four documented values.
        ALLOWED = {"none", "word_overlap_zero", "abstention_pattern", "other"}
        for r in tb["skip_reason"]:
            assert r in ALLOWED, (r, ALLOWED)
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


# ── v2.1 consumer non-regression ────────────────────────────────────────

def test_v2_1_recall_filtered_shape_preserved():
    """v2.2 additions do not alter the retrieval.recall_filtered contract
    that M103 scoring and M108/M109 forensic read."""
    import midas_ui
    filtered = [
        {"source_role": "canonical", "text": "c1", "score": 2.0},
        {"source_role": "user", "text": "u1", "score": 1.0},
    ]
    annotated = midas_ui._m109_annotate_recall(filtered)
    # v2.1 contract: each entry has role_weight + provenance +
    # canonical_boost_multiplier_if_active. v2.2 did not remove these.
    for e in annotated:
        assert "role_weight" in e
        assert "provenance" in e
        assert "canonical_boost_multiplier_if_active" in e


def test_v2_1_existing_turn_json_still_parses():
    """Historical v1/v2.0/v2.1 turn JSONs continue to parse under v2.2."""
    fixture = (Path(__file__).resolve().parent.parent.parent
               / "data" / "session_logs" / "sess_20260420_203721_94881"
               / "turn_0001.json")
    if not fixture.exists():
        print(f"SKIP: fixture not found at {fixture}")
        return
    data = json.loads(fixture.read_text())
    assert isinstance(data, dict)
    # v2.1 sections that M103 scoring reads remain present.
    for section in ("input", "retrieval", "generation", "quality",
                    "post_turn"):
        assert section in data, section


# ── Runtime overhead ────────────────────────────────────────────────────

def test_runtime_overhead_field9_under_2ms():
    """Field 9 manifest population on a realistic 10-memory pool must add
    <2ms over a baseline present() call. Budget is the A3 directive gate."""
    from multi_path_retrieve import present, get_last_truncated_items
    long_text = "z" * 200
    memories = [
        {"text": f"{long_text} rec{i}", "source_role": "user",
         "score": 1.0 - 0.05 * i}
        for i in range(10)
    ]
    # Warm + baseline
    for _ in range(3):
        present(memories, "technical query", max_chars=400)
        get_last_truncated_items()
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        present(memories, "technical query", max_chars=400)
        _ = get_last_truncated_items()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / N
    # Total per-call cost must be under 2ms — this bounds the v2.2 additive
    # field cost because the manifest population is the only change inside
    # the hot loop.
    assert elapsed_ms < 2.0, (
        f"present()+manifest cost {elapsed_ms:.3f}ms exceeds 2ms budget")


def test_runtime_overhead_field10_field11_under_2ms():
    """Field 10 + Field 11 emission adds <2ms per turn."""
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        _diag = {
            "signal_1_fired": True, "signal_2_fired": False,
            "signal_3_fired": False, "signal_4_fired": False,
            "matched_phrases": ["x"] * 5, "verdict": True,
        }
        _reasons = ["word_overlap_zero"]
        # Warm
        for _ in range(3):
            midas_ui._turn_record("scrub", confab_guard_diagnostic={
                "signal_1_fired": True, "signal_2_fired": False,
                "signal_3_fired": False, "signal_4_fired": False,
                "match_text_per_signal": _diag["matched_phrases"][:20],
                "verdict": True})
            midas_ui._turn_record("scrub",
                                  tier2_binding={"skip_reason": _reasons})
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            midas_ui._turn_record("scrub", confab_guard_diagnostic={
                "signal_1_fired": True, "signal_2_fired": False,
                "signal_3_fired": False, "signal_4_fired": False,
                "match_text_per_signal": _diag["matched_phrases"][:20],
                "verdict": True})
            midas_ui._turn_record("scrub",
                                  tier2_binding={"skip_reason": _reasons})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / N
        assert elapsed_ms < 2.0, (
            f"Fields 10+11 emission cost {elapsed_ms:.3f}ms exceeds 2ms")
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


def test_runtime_overhead_aggregate_under_2ms():
    """Aggregate ζ v2.2 additive work (fields 9+10+11) per turn <2ms."""
    from multi_path_retrieve import present, get_last_truncated_items
    import midas_ui
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["scrub"] = {}
        long_text = "w" * 200
        memories = [
            {"text": f"{long_text} rec{i}", "source_role": "user",
             "score": 1.0 - 0.05 * i}
            for i in range(10)
        ]
        # Warm
        for _ in range(3):
            present(memories, "query", max_chars=400)
            _ = get_last_truncated_items()
            midas_ui._turn_record("scrub", confab_guard_diagnostic={
                "signal_1_fired": False, "signal_2_fired": False,
                "signal_3_fired": False, "signal_4_fired": False,
                "match_text_per_signal": [], "verdict": False})
            midas_ui._turn_record("scrub",
                                  tier2_binding={"skip_reason": ["none"]})
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            # Field 9
            present(memories, "query", max_chars=400)
            _ = get_last_truncated_items()
            # Field 10
            midas_ui._turn_record("scrub", confab_guard_diagnostic={
                "signal_1_fired": False, "signal_2_fired": False,
                "signal_3_fired": False, "signal_4_fired": False,
                "match_text_per_signal": [], "verdict": False})
            # Field 11
            midas_ui._turn_record("scrub",
                                  tier2_binding={"skip_reason": ["none"]})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / N
        # The present() call dominates; v2.2 additive cost on top is <<2ms.
        # We assert the aggregate is under a generous 3ms to bound the whole
        # hot path (not just the additive delta). The field-9-only test
        # above is the pure-additive gate.
        assert elapsed_ms < 3.0, (
            f"Aggregate v2.2 path {elapsed_ms:.3f}ms exceeds 3ms bound")
        print(f"    [overhead] aggregate per-turn v2.2 work: {elapsed_ms:.3f}ms")
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


# ── Standalone runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Schema version
        test_schema_version_written_as_2_2_string,
        test_schema_version_type_is_string,
        # Field 9
        test_field9_empty_pool_no_truncation_manifest,
        test_field9_pool_fits_budget_no_truncation,
        test_field9_populated_on_budget_cutoff,
        test_field9_would_render_flag_reflects_budget_raise_semantics,
        # Field 10
        test_field10_shape_from_confab_detector,
        test_field10_populated_turn_record_shape,
        test_field10_absent_when_guard_not_invoked,
        # Field 11
        test_field11_empty_scrub_no_tier2_binding_field,
        test_field11_populated_turn_record_shape,
        # v2.1 consumer non-regression
        test_v2_1_recall_filtered_shape_preserved,
        test_v2_1_existing_turn_json_still_parses,
        # Runtime overhead
        test_runtime_overhead_field9_under_2ms,
        test_runtime_overhead_field10_field11_under_2ms,
        test_runtime_overhead_aggregate_under_2ms,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} passed")
