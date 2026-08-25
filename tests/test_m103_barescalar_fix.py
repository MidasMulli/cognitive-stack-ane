"""
M103 Agent B1 — regression tests for `'float' object has no attribute 'get'`
at two additional bare-scalar bug sites (mirrors M102 M1's narrative_retrieval
pattern).

Root cause (verified 2026-04-21 on live registry):
    data/measurement_registry.json contains 20 entries stored as bare
    scalars (float / bool / str) instead of dicts, written by
    tools/m96/m96_analyze.py:627-650. Example rows:
        "m96.bimodality.hartigan_dip": 0.03275...   (float)
        "m96.bimodality.reject_unimodal_at_0.05": False   (bool)
        "m96.entitlement_probe_result": "vault/..."  (str)

    Downstream consumers iterate `reg.items()` and call `entry.get(...)` /
    `v.get(...)`, which raises AttributeError on the bare-scalar rows.

    M102 M1 fixed `narrative_retrieval._registry_lookup` at line 203.
    M103 B1 mirrors that fix at two more sites that M1 flagged (§6):
      1. `answer_scrub._registry_lookup_by_value` at line 107
      2. `midas_ui._boot_consistency_check` check_entries list comp at 4722

    Both fixes apply the same `isinstance(entry/v, dict)` guard used by M1.

Run from repo root:

    python3 -m pytest orion-ane/tests/test_m103_barescalar_fix.py -v

Or directly:

    python3 orion-ane/tests/test_m103_barescalar_fix.py
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

import answer_scrub  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: a registry that mixes well-formed dict rows with bare scalars.
# This is the actual shape produced by the live system today (20 bare rows
# out of ~477 total, all m96.* prefix).
# ─────────────────────────────────────────────────────────────────────────────

_MIXED_REGISTRY = {
    # Well-formed canonical row with value+unit (the shape
    # _registry_lookup_by_value needs to match).
    "ane.8b.decode_tok_s": {
        "entity": "llama-8b",
        "measurement_type": "decode_tok_s",
        "value": "7.9",
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
    # Another well-formed canonical numeric row (for the boot consistency
    # check path, which requires status=canonical AND digit-shaped value).
    "gemma.31b.prefill_tok_s": {
        "entity": "gemma-31b",
        "measurement_type": "prefill_tok_s",
        "value": "17",
        "unit": "tok/s",
        "source": "session_20260413_main52.md",
        "aliases": ["gemma 31b prefill", "gemma prefill"],
        "status": "canonical",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Target 1 — answer_scrub._registry_lookup_by_value tolerates bare-scalar rows.
# ─────────────────────────────────────────────────────────────────────────────

def _install_scrub_registry(reg):
    """Override the answer_scrub module-level _REGISTRY cache."""
    old = answer_scrub._REGISTRY
    answer_scrub._REGISTRY = dict(reg)
    return old


def _restore_scrub_registry(old):
    answer_scrub._REGISTRY = old


def test_scrub_registry_lookup_skips_bare_float_entries():
    """Prove the M101 §2.5 `[scrub] error` fire site is fixed.

    Pre-fix: `entry.get("value", "")` raises AttributeError on the
    bare-float `m96.bimodality.hartigan_dip` row, aborting the lookup with
    the caught exception printed as `[scrub] error: 'float' object has no
    attribute 'get'`.

    Post-fix: the bare-scalar rows are silently skipped; well-formed rows
    still match by value+unit.
    """
    old = _install_scrub_registry(_MIXED_REGISTRY)
    try:
        # Query matches the well-formed llama-8b row by value+unit.
        result = answer_scrub._registry_lookup_by_value("7.9", "tok/s")
        assert result is not None, (
            "registry lookup returned None — either the bare-scalar rows "
            "aborted the iteration before the llama row was reached, or "
            "value+unit matching is broken.")
        assert result.get("entity") == "llama-8b", result
    finally:
        _restore_scrub_registry(old)


def test_scrub_registry_lookup_no_match_with_only_bare_rows():
    """If the value only exists in bare-scalar rows, return None cleanly
    rather than crashing.

    (The bare float 0.03275... in the fixture doesn't have a unit; lookup
    should skip it and return None without AttributeError.)
    """
    old = _install_scrub_registry(_MIXED_REGISTRY)
    try:
        # Value that only exists as a bare float in the registry.
        result = answer_scrub._registry_lookup_by_value("0.03275006843151643",
                                                         "")
        # Bare-scalar rows are skipped, so result is None — crucially, no
        # AttributeError was raised.
        assert result is None
    finally:
        _restore_scrub_registry(old)


# ─────────────────────────────────────────────────────────────────────────────
# Target 2 — midas_ui._boot_consistency_check tolerates bare-scalar registry
# rows. The check_entries list comp at line 4722 is the actual fire site:
# `v.get("status") == "canonical"` raises AttributeError on bare scalars.
#
# We can't cleanly import midas_ui (it has side effects at import time —
# subprocess launches, server binds, etc.), so we exercise the fix via an
# inlined copy of the list-comp + a dedicated sqlite stub. The test proves
# the isinstance() guard is load-bearing: pre-fix pattern raises, post-fix
# pattern filters cleanly.
# ─────────────────────────────────────────────────────────────────────────────

def test_boot_consistency_check_listcomp_pre_fix_raises():
    """Baseline: the pre-fix list comprehension raises AttributeError on
    bare-scalar rows. This test documents the failure mode M103 B1 fixes."""
    reg = _MIXED_REGISTRY
    try:
        # This is the exact pre-fix list comprehension from
        # midas_ui._boot_consistency_check line 4720-4725.
        _ = [
            k for k, v in reg.items()
            if v.get("status") == "canonical"
            and v.get("value")
            and str(v["value"]).replace("+", "").replace("-", "").replace(
                "<", "").replace("%", "").strip().replace(".", "").isdigit()
        ]
        raised = False
    except AttributeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "pre-fix list comprehension did NOT raise AttributeError — the "
        "fixture is probably missing bare-scalar rows.")
    assert "'float' object has no attribute 'get'" in msg, msg


def test_boot_consistency_check_listcomp_post_fix_clean():
    """The post-fix list comprehension (with isinstance guard) must iterate
    the full mixed registry without raising and return only the well-formed
    canonical digit-valued rows.

    This is the exact post-fix shape now shipping in midas_ui.py:4720-4727.
    """
    reg = _MIXED_REGISTRY
    # This is the exact post-fix list comprehension from
    # midas_ui._boot_consistency_check (M103 B1 patch).
    check_entries = [
        k for k, v in reg.items()
        if isinstance(v, dict)
        and v.get("status") == "canonical"
        and v.get("value")
        and str(v["value"]).replace("+", "").replace("-", "").replace(
            "<", "").replace("%", "").strip().replace(".", "").isdigit()
    ]
    # Only gemma.31b.prefill_tok_s has digit-shaped value "17"
    # (llama row value is "7.9" — has a dot, isdigit after dot-strip: "79"
    # IS digit, so it also qualifies). Expect BOTH well-formed rows.
    assert "gemma.31b.prefill_tok_s" in check_entries, check_entries
    assert "ane.8b.decode_tok_s" in check_entries, check_entries
    # The bare-scalar rows must NOT appear.
    for bad_key in (
        "m96.bimodality.hartigan_dip",
        "m96.bimodality.reject_unimodal_at_0.05",
        "m96.entitlement_probe_result",
    ):
        assert bad_key not in check_entries, (bad_key, check_entries)


# ─────────────────────────────────────────────────────────────────────────────
# Target 3 — integration: exercise both fixes against the LIVE on-disk
# registry. Pre-pilot scrub-error verification per directive §6 "Pre-pilot
# scrub-error count on test queries is zero."
# ─────────────────────────────────────────────────────────────────────────────

def test_scrub_lookup_against_live_registry_no_attribute_error():
    """Exercise _registry_lookup_by_value against the real on-disk registry
    (with its 20 bare-scalar rows). Must not raise AttributeError on any
    query."""
    # Force a fresh load from disk.
    answer_scrub._REGISTRY = None
    try:
        # Multiple lookups exercising the full iteration path.
        for (val, unit) in [("7.9", "tok/s"), ("17.5", "tok/s"),
                            ("0.03275", ""), ("false", ""),
                            ("arbitrary", "none")]:
            # Must return None or a dict. Crucially: must not raise.
            result = answer_scrub._registry_lookup_by_value(val, unit)
            assert result is None or isinstance(result, dict)
    finally:
        # Clear the cache so subsequent tests see their fixture.
        answer_scrub._REGISTRY = None


def test_boot_consistency_listcomp_against_live_registry_no_attribute_error():
    """Exercise the post-fix consistency-check list comp against the real
    on-disk registry. Must not raise AttributeError."""
    reg_path = (Path(__file__).resolve().parent.parent.parent
                / "data" / "measurement_registry.json")
    with open(reg_path) as f:
        reg = json.load(f)
    # Post-fix list comprehension (exact shape from midas_ui.py:4720-4727).
    check_entries = [
        k for k, v in reg.items()
        if isinstance(v, dict)
        and v.get("status") == "canonical"
        and v.get("value")
        and str(v["value"]).replace("+", "").replace("-", "").replace(
            "<", "").replace("%", "").strip().replace(".", "").isdigit()
    ]
    # No assertion on content — the live registry shape evolves. The
    # critical assertion is that iteration completed without AttributeError.
    assert isinstance(check_entries, list)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_scrub_registry_lookup_skips_bare_float_entries,
        test_scrub_registry_lookup_no_match_with_only_bare_rows,
        test_boot_consistency_check_listcomp_pre_fix_raises,
        test_boot_consistency_check_listcomp_post_fix_clean,
        test_scrub_lookup_against_live_registry_no_attribute_error,
        test_boot_consistency_listcomp_against_live_registry_no_attribute_error,
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
