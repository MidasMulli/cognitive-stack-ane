"""
M104 Agent F2 — regression tests for `'float' object has no attribute 'get'`
at the three remaining bare-scalar consumer sites (mirrors M102 M1 +
M103 B1 pattern).

Root cause (verified 2026-04-21 on live registry, same as M102/M103):
    data/measurement_registry.json contains 20 entries stored as bare
    scalars (float / bool / str) instead of dicts, written by
    tools/m96/m96_analyze.py:627-650.

    M102 M1 fixed: orion-ane/agent/narrative_retrieval.py (3 sites).
    M103 B1 fixed: orion-ane/agent/answer_scrub.py:107,
                   orion-ane/agent/midas_ui.py:4720-4727.
    M104 F2 fixes the three remaining flagged consumer sites:
      1. tools/mcp_server.py:500 — MCP `query_measurement_registry`
         handler (externally exposed via mcp.subconsciousmcp.com/mcp).
         HIGHEST PRIORITY: claude.ai agents can hit this before any
         internal path catches it.
      2. orion-ane/agent/grounding_corpus.py:63 — registry-to-corpus
         stringify.
      3. orion-ane/agent/midas_ui.py:1474 —
         _MEASUREMENT_REGISTRY_CACHE alias iterator inside
         `_measurement_registry_lookup`.

    All three fixes apply the same `isinstance(entry, dict): continue`
    guard used by M1 / B1.

Run from repo root:

    python3 -m pytest orion-ane/tests/test_m104_barescalar_remaining.py -v

Or directly:

    python3 orion-ane/tests/test_m104_barescalar_remaining.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: a registry that mixes well-formed dict rows with bare scalars —
# same shape as M103 B1's _MIXED_REGISTRY (live system has 20 bare rows).
# ─────────────────────────────────────────────────────────────────────────────

_MIXED_REGISTRY = {
    # Well-formed canonical row — should be kept by all three consumers.
    "ane.8b.decode_tok_s": {
        "entity": "llama-8b",
        "measurement_type": "decode_tok_s",
        "value": "7.9",
        "unit": "tok/s",
        "source": "finding_ane_fp16_native.md",
        "aliases": ["8b ane throughput", "llama 8b tok/s"],
        "status": "canonical",
    },
    # Bare-scalar rows (the shape that currently breaks consumers).
    "m96.bimodality.hartigan_dip": 0.03275006843151643,
    "m96.bimodality.reject_unimodal_at_0.05": False,
    "m96.entitlement_probe_result": "vault/research/m96_entitlement/findings.md",
    # Another well-formed row.
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


LIVE_REGISTRY_PATH = Path(REPO_ROOT) / "data" / "measurement_registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# Site 1 — tools/mcp_server.py:500 (MCP query_measurement_registry handler).
#
# We can't easily import mcp_server (it builds a FastMCP object and binds
# tool handlers at import time), so exercise the exact inlined filter
# shape. The test proves the isinstance() guard is load-bearing: pre-fix
# pattern raises, post-fix pattern filters cleanly. Mirrors M103 B1's
# approach for the midas_ui boot-consistency list comp.
# ─────────────────────────────────────────────────────────────────────────────

def _mcp_filter_pre_fix(registry, entity=None, measurement_type=None):
    """Pre-fix shape of tools/mcp_server.py:500 (for fail-before assertion)."""
    results = {}
    for key, entry in registry.items():
        if entity and entity.lower() != entry.get("entity", "").lower():
            aliases = [a.lower() for a in entry.get("aliases", [])]
            if entity.lower() not in aliases:
                continue
        if measurement_type and measurement_type.lower() not in entry.get(
                "measurement_type", "").lower():
            continue
        results[key] = entry
    return results


def _mcp_filter_post_fix(registry, entity=None, measurement_type=None):
    """Post-fix shape of tools/mcp_server.py:500 (M104 F2 patch)."""
    results = {}
    for key, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        if entity and entity.lower() != entry.get("entity", "").lower():
            aliases = [a.lower() for a in entry.get("aliases", [])]
            if entity.lower() not in aliases:
                continue
        if measurement_type and measurement_type.lower() not in entry.get(
                "measurement_type", "").lower():
            continue
        results[key] = entry
    return results


def test_mcp_query_registry_pre_fix_raises():
    """Baseline: pre-fix MCP filter raises AttributeError on bare-scalar
    rows. Documents the failure mode M104 F2 fixes.

    This is the MCP `query_measurement_registry` handler, exposed
    externally via mcp.subconsciousmcp.com/mcp. External consumers
    (claude.ai agents) hit this before any internal recovery can catch it.
    """
    raised = False
    msg = ""
    try:
        _mcp_filter_pre_fix(_MIXED_REGISTRY, entity="llama-8b")
    except AttributeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "pre-fix MCP filter did NOT raise AttributeError — the fixture is "
        "probably missing bare-scalar rows.")
    assert "'float' object has no attribute 'get'" in msg, msg


def test_mcp_query_registry_post_fix_clean():
    """Post-fix MCP filter iterates the full mixed registry without raising
    and returns only the well-formed row that matches the entity filter.

    Exact shape now shipping in tools/mcp_server.py:500.
    """
    results = _mcp_filter_post_fix(_MIXED_REGISTRY, entity="llama-8b")
    assert "ane.8b.decode_tok_s" in results, results
    # Bare-scalar rows must not appear.
    for bad_key in (
        "m96.bimodality.hartigan_dip",
        "m96.bimodality.reject_unimodal_at_0.05",
        "m96.entitlement_probe_result",
    ):
        assert bad_key not in results, (bad_key, results)


def test_mcp_query_registry_against_live_registry_no_attribute_error():
    """Integration: exercise the post-fix MCP filter against the real
    on-disk registry (with its 20 bare-scalar rows). Must not raise
    AttributeError."""
    with open(LIVE_REGISTRY_PATH) as f:
        reg = json.load(f)
    # Multiple filter combinations exercising the full iteration path.
    for (ent, mtype) in [("llama-8b", None), (None, "tok_s"),
                         ("72b", "recall"), (None, None),
                         ("gemma", "prefill_tok_s")]:
        results = _mcp_filter_post_fix(reg, entity=ent, measurement_type=mtype)
        # Critical assertion: iteration completed without AttributeError.
        assert isinstance(results, dict)


# ─────────────────────────────────────────────────────────────────────────────
# Site 2 — orion-ane/agent/grounding_corpus.py:63 (_load_registry_values).
# ─────────────────────────────────────────────────────────────────────────────

def test_grounding_corpus_registry_values_bare_scalars_no_attribute_error():
    """grounding_corpus._load_registry_values must iterate the live
    registry (with 20 bare-scalar rows) without raising AttributeError.

    Pre-fix: `entry.get("value", "")` raises on the first bare-scalar
    row (wrapped in a try/except that returns empty string — but the
    iteration aborts before producing any grounding values, silently
    degrading NOGROUND detector quality).
    Post-fix: bare-scalar rows are skipped; well-formed rows still
    contribute value+unit strings.
    """
    import grounding_corpus
    # Force a fresh load (module-level cache may already be populated).
    grounding_corpus._REGISTRY_VALUES_CACHE = None
    result = grounding_corpus._load_registry_values()
    # Critical: result is non-empty (proof that iteration produced values
    # from well-formed rows rather than aborting on the first bare-scalar
    # row). Pre-fix would return "" because the except branch caught the
    # AttributeError and set the cache to "".
    assert isinstance(result, str)
    assert len(result) > 0, (
        "grounding corpus registry values string is empty — iteration "
        "likely aborted on a bare-scalar row. Check the isinstance guard.")
    # Reset for other tests.
    grounding_corpus._REGISTRY_VALUES_CACHE = None


def test_grounding_corpus_registry_values_with_mixed_fixture():
    """Unit-level test: the post-fix loop shape filters a mixed fixture
    cleanly, producing value+unit parts only from the well-formed dict
    rows."""
    reg = _MIXED_REGISTRY
    parts = []
    # Exact post-fix shape from grounding_corpus.py:63.
    for key, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        val = entry.get("value", "")
        unit = entry.get("unit", "")
        parts.append(f"{val} {unit}")
        parts.append(f"{val}{unit}")
    joined = " ".join(parts)
    # Well-formed values appear.
    assert "7.9 tok/s" in joined, joined
    assert "17 tok/s" in joined, joined
    # Bare-scalar values must NOT appear as formatted parts (they were
    # skipped entirely). The raw float 0.03275... would have matched if
    # it had leaked through.
    assert "0.03275" not in joined, joined
    assert "False" not in joined, joined


# ─────────────────────────────────────────────────────────────────────────────
# Site 3 — orion-ane/agent/midas_ui.py:1474 (_measurement_registry_lookup).
#
# midas_ui imports have heavy side effects (server bind, subprocess launches,
# etc.), so exercise the exact inlined filter shape, matching M103 B1's
# approach for the boot-consistency list comp at line 4722.
# ─────────────────────────────────────────────────────────────────────────────

def _midas_measurement_lookup_pre_fix(cache, query):
    """Pre-fix shape of midas_ui.py:1474 (for fail-before assertion)."""
    q_lower = query.lower()
    q_words = set(w.strip("?.,!\"'()") for w in q_lower.split() if len(w) >= 2)
    matches = []
    for key, entry in cache.items():
        aliases = [a.lower() for a in entry.get("aliases", [])]
        entity = entry.get("entity", "").lower()
        mtype = entry.get("measurement_type", "").lower().replace("_", " ")
        alias_hit = any(a in q_lower for a in aliases)
        entity_hit = any(w in q_words for w in entity.split("_") if len(w) >= 2)
        type_hit = any(w in q_words for w in mtype.split() if len(w) >= 3)
        if alias_hit or (entity_hit and type_hit):
            val = entry.get("value", "?")
            unit = entry.get("unit", "")
            source = entry.get("source", "")
            label = key.replace(".", " ").replace("_", " ")
            matches.append(f"[MEASUREMENT] {label}: {val} {unit} ({source})")
    return matches


def _midas_measurement_lookup_post_fix(cache, query):
    """Post-fix shape of midas_ui.py:1474 (M104 F2 patch)."""
    q_lower = query.lower()
    q_words = set(w.strip("?.,!\"'()") for w in q_lower.split() if len(w) >= 2)
    matches = []
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        aliases = [a.lower() for a in entry.get("aliases", [])]
        entity = entry.get("entity", "").lower()
        mtype = entry.get("measurement_type", "").lower().replace("_", " ")
        alias_hit = any(a in q_lower for a in aliases)
        entity_hit = any(w in q_words for w in entity.split("_") if len(w) >= 2)
        type_hit = any(w in q_words for w in mtype.split() if len(w) >= 3)
        if alias_hit or (entity_hit and type_hit):
            val = entry.get("value", "?")
            unit = entry.get("unit", "")
            source = entry.get("source", "")
            label = key.replace(".", " ").replace("_", " ")
            matches.append(f"[MEASUREMENT] {label}: {val} {unit} ({source})")
    return matches


def test_midas_measurement_lookup_pre_fix_raises():
    """Baseline: the pre-fix midas_ui._measurement_registry_lookup loop
    raises AttributeError on bare-scalar rows. Documents the failure
    mode M104 F2 fixes."""
    raised = False
    msg = ""
    try:
        _midas_measurement_lookup_pre_fix(_MIXED_REGISTRY, "8b ane throughput")
    except AttributeError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "pre-fix midas measurement lookup did NOT raise AttributeError — "
        "the fixture is probably missing bare-scalar rows.")
    assert "'float' object has no attribute 'get'" in msg, msg


def test_midas_measurement_lookup_post_fix_clean():
    """Post-fix midas_ui measurement lookup iterates the mixed registry
    without raising and returns canonical matches only from well-formed
    rows.

    Exact shape now shipping in midas_ui.py:1474.
    """
    matches = _midas_measurement_lookup_post_fix(
        _MIXED_REGISTRY, "8b ane throughput")
    # Expect the well-formed llama-8b row to match via aliases.
    assert len(matches) >= 1, matches
    # All matches reference well-formed rows (source present).
    for m in matches:
        assert "[MEASUREMENT]" in m, m


def test_midas_measurement_lookup_against_live_registry_no_attribute_error():
    """Integration: exercise the post-fix lookup loop against the real
    on-disk registry (with its 20 bare-scalar rows). Must not raise
    AttributeError across representative queries."""
    with open(LIVE_REGISTRY_PATH) as f:
        reg = json.load(f)
    for query in ["8b ane throughput", "gemma prefill", "how fast is 70b",
                  "what's our recall", "tok/s on ane"]:
        matches = _midas_measurement_lookup_post_fix(reg, query)
        # Critical assertion: iteration completed without AttributeError.
        assert isinstance(matches, list)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Site 1 — mcp_server.py (highest priority, externally exposed).
        test_mcp_query_registry_pre_fix_raises,
        test_mcp_query_registry_post_fix_clean,
        test_mcp_query_registry_against_live_registry_no_attribute_error,
        # Site 2 — grounding_corpus.py.
        test_grounding_corpus_registry_values_bare_scalars_no_attribute_error,
        test_grounding_corpus_registry_values_with_mixed_fixture,
        # Site 3 — midas_ui.py:1474.
        test_midas_measurement_lookup_pre_fix_raises,
        test_midas_measurement_lookup_post_fix_clean,
        test_midas_measurement_lookup_against_live_registry_no_attribute_error,
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
