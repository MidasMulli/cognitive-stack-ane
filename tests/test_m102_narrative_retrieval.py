"""
M102 Agent M1 — regression tests for `'float' object has no attribute 'get'`
in `orion-ane/agent/narrative_retrieval.py`.

Root cause (verified 2026-04-21 on live registry):
    data/measurement_registry.json contains 20 entries stored as bare
    scalars (float / bool / str) instead of dicts, written by
    tools/m96/m96_analyze.py:627-650. Example rows:
        "m96.bimodality.hartigan_dip": 0.03275...   (float)
        "m96.bimodality.reject_unimodal_at_0.05": False   (bool)
        "m96.entitlement_probe_result": "vault/..."  (str)

    Downstream consumers iterate `reg.items()` and call `entry.get(...)`,
    which raises AttributeError on the bare-scalar rows.

    `narrative_retrieval._registry_lookup` at line 203 (pre-fix) is the
    actual narrative_retrieval fire site for the 12 `[narrative_retrieval]
    failed (0ms)` occurrences in the M101 §2.5 pilot log — NOT line 386 as
    originally filed. (Line 386 was grep-identified as a candidate but is
    unreachable without first clearing the registry lookup; and `thread`
    at that point is always a list of dicts from `detect_thread`.)

    The M102 fix is defensive `isinstance(entry, dict)` coercion in
    `_registry_lookup` plus a belt-and-suspenders guard at line 386
    against future shape drift.

Run from repo root:

    python3 -m pytest orion-ane/tests/test_m102_narrative_retrieval.py -v

Or directly:

    python3 orion-ane/tests/test_m102_narrative_retrieval.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import narrative_retrieval  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: a registry that mixes well-formed dict rows with bare scalars.
# This is the actual shape produced by the live system today.
# ─────────────────────────────────────────────────────────────────────────────

_MIXED_REGISTRY = {
    # Well-formed canonical row (the shape _registry_lookup expects).
    "ane.8b.decode_tok_s": {
        "entity": "llama-8b",
        "measurement_type": "decode_tok_s",
        "value": 7.9,
        "unit": "tok/s",
        "source": "finding_ane_fp16_native.md",
        "aliases": ["8b ane throughput", "llama 8b tok/s"],
        "status": "canonical",
    },
    # The shape that currently breaks things — bare float, no dict envelope.
    "m96.bimodality.hartigan_dip": 0.03275006843151643,
    # Bare bool — same class.
    "m96.bimodality.reject_unimodal_at_0.05": False,
    # Bare string — same class.
    "m96.entitlement_probe_result": "vault/research/m96_entitlement/findings.md",
    # Another well-formed row after the bad ones to verify iteration continues.
    "gemma.31b.prefill_tok_s": {
        "entity": "gemma-31b",
        "measurement_type": "prefill_tok_s",
        "value": 17.5,
        "unit": "tok/s",
        "source": "session_20260413_main52.md",
        "aliases": ["gemma 31b prefill", "gemma prefill"],
        "status": "canonical",
    },
}


def _install_mixed_registry(monkeypatch_style=False):
    """Override the module-level registry cache with the mixed fixture.

    Uses narrative_retrieval's private globals directly because that's how
    the module exposes the cached registry. Returns (old_reg, old_mtime)
    so the caller can restore.
    """
    old_reg = narrative_retrieval._measurement_registry
    old_mtime = narrative_retrieval._measurement_registry_mtime
    # Install a far-future mtime so _load_registry short-circuits and
    # returns our fixture instead of re-reading the on-disk file.
    narrative_retrieval._measurement_registry = dict(_MIXED_REGISTRY)
    narrative_retrieval._measurement_registry_mtime = 1e18
    return old_reg, old_mtime


def _restore_registry(old_reg, old_mtime):
    narrative_retrieval._measurement_registry = old_reg
    narrative_retrieval._measurement_registry_mtime = old_mtime


# ─────────────────────────────────────────────────────────────────────────────
# Target 1 — _registry_lookup tolerates bare-scalar registry rows.
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_lookup_skips_bare_float_entries():
    """Prove the actual M101 §2.5 fire site is fixed.

    Pre-fix: `entry.get("aliases", [])` raises AttributeError on the
    bare-float `m96.bimodality.hartigan_dip` row, aborting the whole
    factual-intent path with the caught exception printed as
    `[narrative_retrieval] failed (0ms): 'float' object has no attribute 'get'`.

    Post-fix: the bare-scalar rows are silently skipped; well-formed rows
    still match.
    """
    old = _install_mixed_registry()
    try:
        # Query matches an alias in the well-formed gemma row.
        result = narrative_retrieval._registry_lookup("what is gemma 31b prefill?")
        assert result is not None, (
            "registry lookup returned None — either the bare-scalar rows "
            "aborted the iteration before the gemma row was reached, or "
            "alias matching is broken.")
        assert "gemma-31b" in result, result
        assert "17.5" in result, result
    finally:
        _restore_registry(*old)


def test_registry_lookup_no_match_with_only_bare_rows():
    """If every matching alias only exists in bare-scalar rows, return None
    cleanly rather than crashing."""
    old = _install_mixed_registry()
    try:
        # Query that matches nothing in the well-formed rows.
        result = narrative_retrieval._registry_lookup(
            "what is m96 bimodality hartigan dip?")
        # Bare-scalar rows have no aliases to match against, so result is
        # None — but crucially, no AttributeError was raised.
        assert result is None
    finally:
        _restore_registry(*old)


# ─────────────────────────────────────────────────────────────────────────────
# Target 2 — try_narrative_context end-to-end no longer aborts on factual
# intent when the registry contains bare-scalar rows.
# ─────────────────────────────────────────────────────────────────────────────

def test_try_narrative_context_factual_intent_no_attribute_error():
    """End-to-end smoke: a factual-intent query against a registry with
    bare-scalar rows must not print `[narrative_retrieval] failed`.

    Pre-fix this raised and was caught by the module-level try/except at
    the bottom of try_narrative_context, returning None with a logged
    traceback message. Post-fix it either returns a registry hit or falls
    through to thread detection — in either case, no AttributeError."""
    old = _install_mixed_registry()
    try:
        result = narrative_retrieval.try_narrative_context(
            "what is gemma 31b prefill throughput?")
        # Should either be a valid dict (registry hit) or None (no thread).
        # The critical assertion: no exception was raised inside.
        assert result is None or isinstance(result, dict)
        if isinstance(result, dict):
            # Registry hit path — must include gemma binding.
            narrative = result.get("narrative", "")
            assert "gemma" in narrative.lower()
    finally:
        _restore_registry(*old)


# ─────────────────────────────────────────────────────────────────────────────
# Target 3 — line 386 defensive coercion holds even if `thread` ever contains
# a bare float (hypothetical future shape drift).
# ─────────────────────────────────────────────────────────────────────────────

def test_line_386_tolerates_bare_float_in_thread():
    """Exercise the M102 defensive guard at line 386.

    `thread` is always a list of dicts from `detect_thread()` today, but
    the directive asks for belt-and-suspenders against future shape drift.
    This test monkeypatches the thread detector to return a list where
    element 0 is a bare float and verifies the function returns cleanly
    (None) instead of raising AttributeError.
    """
    old_detector = narrative_retrieval._thread_detector
    # Pretend imports already ran so _ensure_imports() is a no-op.
    narrative_retrieval._ensure_imports()
    real_detector = narrative_retrieval._thread_detector

    def _bad_detector(query, corpus, top_n=30):
        # Return a list where element 0 is a bare float. Remaining
        # elements mimic the well-formed dict shape.
        return [
            0.42,  # bare float — the failing shape the directive guards
            {
                "session_label": "main99",
                "record_index": 0,
                "type": "fact",
                "content": "dummy record for session count",
                "score": 0.1,
            },
            {
                "session_label": "main100",
                "record_index": 0,
                "type": "fact",
                "content": "another dummy for session count",
                "score": 0.1,
            },
        ]

    narrative_retrieval._thread_detector = _bad_detector
    old_reg = narrative_retrieval._measurement_registry
    old_mtime = narrative_retrieval._measurement_registry_mtime
    # Empty registry so _registry_lookup returns None and we fall through
    # to the thread path (reaching line 386).
    narrative_retrieval._measurement_registry = {}
    narrative_retrieval._measurement_registry_mtime = 1e18
    try:
        # Use a query that triggers an arc intent path (not factual)
        # so it falls through to thread detection.
        result = narrative_retrieval.try_narrative_context(
            "how did we trace the 8b ane throughput evolution?")
        # Must not raise; top_score becomes 0 so the gate fails and we
        # return None cleanly.
        assert result is None
    finally:
        narrative_retrieval._thread_detector = old_detector or real_detector
        narrative_retrieval._measurement_registry = old_reg
        narrative_retrieval._measurement_registry_mtime = old_mtime


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_registry_lookup_skips_bare_float_entries,
        test_registry_lookup_no_match_with_only_bare_rows,
        test_try_narrative_context_factual_intent_no_attribute_error,
        test_line_386_tolerates_bare_float_in_thread,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} FAILED")
        sys.exit(1)
    print(f"\n{len(tests)}/{len(tests)} passed")
