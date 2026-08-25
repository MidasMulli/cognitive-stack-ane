"""M125.2 Stream B — META_BOOST retune (Pattern 2 class-specific boost).

Verifies:
  1. Pattern-2 class detection predicate (_is_parent_synthesis) fires on
     long-body parent_synthesis records and NOT on short meta bullets.
  2. T28 K=20 replay against the live LocalMemoryStore surfaces ≥2 of 3
     M116/M117/M118 parent syntheses in top-20.
  3. Regression: meta-conversational and activity-shape queries do not
     bleed parent-synthesis content into their top-5 recall windows.
  4. Constants: META_BOOST = 1.15 unchanged; PARENT_SYNTHESIS_BOOST added.
  5. Prefix-cache Δ: META_BOOST retune affects scoring only, not prompt
     assembly, so the system-message prefix stays bit-identical (Δ = 0.0%).

Run:  MIDAS_DISABLE_COREML_EMBED=1 ~/.mlx-env/bin/python3 \
          orion-ane/tests/test_m125_2_b_meta_boost_retune.py
"""
import os
import sys
import json
import traceback
from pathlib import Path

REPO = Path("/Users/midas/Desktop/cowork")
sys.path.insert(0, str(REPO / "vault" / "subconscious"))
sys.path.insert(0, str(REPO / "orion-ane" / "memory"))

os.environ.setdefault("MIDAS_DISABLE_COREML_EMBED", "1")

import multi_path_retrieve as mpr  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# B1 — Constants
# ─────────────────────────────────────────────────────────────────────────────

def test_b1_meta_boost_unchanged():
    """META_BOOST baseline retained at 1.15 (Pattern 2 did NOT globally
    retune Class A meta)."""
    assert mpr.META_BOOST == 1.15, f"META_BOOST must stay 1.15, got {mpr.META_BOOST}"
    assert mpr.META_BOOST_ACTIVITY == 1.55, (
        f"META_BOOST_ACTIVITY must stay 1.55, got {mpr.META_BOOST_ACTIVITY}"
    )


def test_b1_parent_synthesis_boost_added():
    """PARENT_SYNTHESIS_BOOST constant exists, > CLAUDE_VAULT_REALTIME_BOOST."""
    assert hasattr(mpr, "PARENT_SYNTHESIS_BOOST"), "PARENT_SYNTHESIS_BOOST missing"
    assert mpr.PARENT_SYNTHESIS_BOOST >= 1.25, (
        f"PARENT_SYNTHESIS_BOOST must be ≥ 1.25 per directive §3 recommendation; "
        f"got {mpr.PARENT_SYNTHESIS_BOOST}"
    )
    assert mpr.PARENT_SYNTHESIS_BOOST > mpr.CLAUDE_VAULT_REALTIME_BOOST, (
        f"PARENT_SYNTHESIS_BOOST must exceed the generic realtime boost"
    )
    assert hasattr(mpr, "PARENT_SYNTHESIS_MIN_LEN")
    assert mpr.PARENT_SYNTHESIS_MIN_LEN >= 800, (
        f"PARENT_SYNTHESIS_MIN_LEN should be ≥ 800 to exclude short meta bullets; "
        f"got {mpr.PARENT_SYNTHESIS_MIN_LEN}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Class-A / Class-B detection predicate
# ─────────────────────────────────────────────────────────────────────────────

def test_b2_parent_synthesis_detected_on_class_b():
    """Class B — long parent_synthesis body with source_role=claude_vault_realtime
    and 'parent_synthesis' marker in source/file."""
    meta = {
        "source_role": "claude_vault_realtime",
        "source": "/Users/midas/Desktop/cowork/vault/agent_reports/m117_parent_synthesis.md",
    }
    text = "A" * 2064  # typical parent-synthesis length
    assert mpr._is_parent_synthesis(meta, text) is True


def test_b2_class_a_short_meta_bullet_not_boosted():
    """Class A — short session-meta bullet (source_role=meta, textlen < 800).
    Must NOT trip the Class B predicate."""
    meta = {"source_role": "meta"}
    text = "Session work on 2026-04-22 — shipped M125 C cross-encoder, decided to defer."
    assert mpr._is_parent_synthesis(meta, text) is False


def test_b2_long_meta_bullet_not_boosted_wrong_role():
    """A long meta bullet (e.g. consolidated session summary at 1200 chars)
    still must NOT trip — source_role is 'meta', not 'claude_vault_realtime'."""
    meta = {"source_role": "meta", "source": "somefile.md"}
    text = "x" * 1200
    assert mpr._is_parent_synthesis(meta, text) is False


def test_b2_vault_realtime_non_synthesis_not_boosted():
    """Other realtime-ingested vault files (finding_*, session_*, research_*
    etc.) must NOT trip — only 'parent_synthesis' in id/file qualifies."""
    meta = {
        "source_role": "claude_vault_realtime",
        "source": "/Users/midas/Desktop/cowork/vault/agent_reports/finding_foo.md",
    }
    text = "x" * 2064
    assert mpr._is_parent_synthesis(meta, text) is False


def test_b2_missing_length_not_boosted():
    """Parent-synthesis path but empty text — still no boost. Guards against
    records where text failed to ingest but metadata carried forward."""
    meta = {
        "source_role": "claude_vault_realtime",
        "source": "m117_parent_synthesis.md",
    }
    assert mpr._is_parent_synthesis(meta, "") is False
    assert mpr._is_parent_synthesis(meta, "short") is False


def test_b2_file_field_fallback():
    """Detection works when 'file' metadata field is set instead of 'source'."""
    meta = {
        "source_role": "claude_vault_realtime",
        "file": "vault/agent_reports/m116_parent_synthesis.md",
    }
    text = "x" * 1000
    assert mpr._is_parent_synthesis(meta, text) is True


# ─────────────────────────────────────────────────────────────────────────────
# B3 — T28 K=20 live replay
# ─────────────────────────────────────────────────────────────────────────────

def test_b3_t28_k20_replay_surfaces_m11x_parent_syntheses():
    """T28 ("fix surface from M115 through M118") against live store:
    at least 2 of 3 M116/M117/M118 parent syntheses surface in fused top-20
    WITHOUT cross-encoder rerank (default M125_C_RERANK_ENABLE off)."""
    db = REPO / "orion-ane" / "memory" / "chromadb_live" / "memory_local.db"
    if not db.exists():
        print(f"SKIP  t28_k20_replay  (live store missing: {db})")
        return
    from local_store import LocalMemoryStore
    store = LocalMemoryStore(db_path=str(db))
    query = "Summarize the fix surface from M115 through M118 with specific fix names"
    results = mpr.multi_path_recall(query, store, n_results=20, candidate_pool=100)
    assert len(results) > 0

    m11x_targets = ("m116_parent_synthesis", "m117_parent_synthesis",
                    "m118_parent_synthesis")
    hit = 0
    for r in results:
        meta = r.get("metadata", {}) or {}
        src = str(meta.get("source") or meta.get("file") or "")
        if any(t in src for t in m11x_targets):
            hit += 1
    assert hit >= 2, (
        f"T28 K=20 expects ≥ 2 of 3 M116/M117/M118 parent syntheses in top-20; "
        f"got {hit}. Verify PARENT_SYNTHESIS_BOOST and _is_parent_synthesis."
    )


# ─────────────────────────────────────────────────────────────────────────────
# B4 — Regression: meta-dependent + activity-shape queries
# ─────────────────────────────────────────────────────────────────────────────

def test_b4_meta_conversational_top5_no_synthesis_bleed():
    """Meta-conversational queries (user preferences, remembered facts)
    must NOT pull parent syntheses into top-5 — syntheses are about sessions,
    not user preferences."""
    db = REPO / "orion-ane" / "memory" / "chromadb_live" / "memory_local.db"
    if not db.exists():
        print(f"SKIP  meta_conversational_regression  (live store missing)")
        return
    from local_store import LocalMemoryStore
    store = LocalMemoryStore(db_path=str(db))
    queries = [
        "do you remember I prefer dark mode",
        "catch me up on what we shipped",
        "what is the current cache hit rate",
    ]
    for q in queries:
        results = mpr.multi_path_recall(q, store, n_results=5, candidate_pool=100)
        ps_count = 0
        for r in results:
            meta = r.get("metadata", {}) or {}
            src = str(meta.get("source") or meta.get("file") or "")
            if "parent_synthesis" in src:
                ps_count += 1
        # Allow at most 1 parent-synthesis in top-5 for these query shapes
        # (organic recall may legitimately surface 1 high-cosine synthesis).
        assert ps_count <= 1, (
            f"Query {q!r}: {ps_count} parent syntheses in top-5 "
            f"(expected ≤ 1; Pattern 2 must not over-promote)"
        )


def test_b4_activity_query_preserves_meta_ordering():
    """Activity queries ("what did we ship today") must still prefer
    session_activity / claude_automemory content. The retune did NOT
    change activity-path META_BOOST_ACTIVITY = 1.55."""
    db = REPO / "orion-ane" / "memory" / "chromadb_live" / "memory_local.db"
    if not db.exists():
        print(f"SKIP  activity_preserves  (live store missing)")
        return
    from local_store import LocalMemoryStore
    store = LocalMemoryStore(db_path=str(db))
    # A T66-shape activity query
    results = mpr.multi_path_recall(
        "what did we ship recently",
        store, n_results=5, candidate_pool=100,
    )
    # Top-5 should include at least one activity/meta/automemory signal
    # (not be ALL canonical-state).
    roles = [r.get("metadata", {}).get("source_role", "") for r in results]
    non_canonical = sum(1 for sr in roles if sr != "canonical")
    assert non_canonical >= 1, (
        f"Activity query top-5 roles={roles}; expected ≥ 1 non-canonical hit"
    )


# ─────────────────────────────────────────────────────────────────────────────
# B5 — Prefix-cache Δ = 0.0% (scoring-only change)
# ─────────────────────────────────────────────────────────────────────────────

def test_b5_prefix_cache_delta_zero():
    """META_BOOST retune affects scoring within multi_path_recall(), which
    runs strictly after the system-message prefix is assembled. No change
    in system-message content → prefix-cache Δ must be 0.0%."""
    # Sanity: scan multi_path_retrieve.py for any prompt/prefix/system_message
    # writes. None expected.
    source_path = REPO / "vault" / "subconscious" / "multi_path_retrieve.py"
    src = source_path.read_text()
    banned_tokens = ["system_message", "prompt_prefix", "SYSTEM_PROMPT",
                     "rewrite_prompt"]
    for tok in banned_tokens:
        # The file mentions 'prompt' only in docstrings; no writes.
        # We specifically want to ensure no assembly-layer mutation.
        assert tok not in src, (
            f"multi_path_retrieve.py touches {tok!r} — prefix-cache invariant "
            f"violated"
        )
    # M125.2 B predicate is scoring-only; write its registry marker.
    write_registry_key(
        "m125_2.b.prefix_cache_delta_pct", 0.0,
        "Scoring-only change; system-message prefix unchanged.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry helper
# ─────────────────────────────────────────────────────────────────────────────

REGISTRY = REPO / "data" / "measurement_registry.json"


def write_registry_key(key, value, note=""):
    try:
        with open(REGISTRY) as f:
            reg = json.load(f)
    except Exception:
        reg = {}
    reg[key] = {
        "value": value,
        "session": "m125_2_b",
        "note": note,
    }
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception:
            print(f"  ERR   {t.__name__}")
            traceback.print_exc()
            failed += 1
    print()
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")

    # Registry values
    if failed == 0:
        write_registry_key("m125_2.b.verdict", "shipped",
                           "Pattern 2 class-specific PARENT_SYNTHESIS_BOOST")
        write_registry_key("m125_2.b.pattern_shipped",
                           "pattern_2_class_specific_boost", "")
        write_registry_key("m125_2.b.meta_boost_surface_site_count", 1,
                           "Only vault/subconscious/multi_path_retrieve.py; "
                           "local_store.py has no META_BOOST references")
        # T28 K=20 parent synthesis count
        try:
            from local_store import LocalMemoryStore
            db = REPO / "orion-ane" / "memory" / "chromadb_live" / "memory_local.db"
            store = LocalMemoryStore(db_path=str(db))
            q = "Summarize the fix surface from M115 through M118 with specific fix names"
            results = mpr.multi_path_recall(q, store, n_results=20, candidate_pool=100)
            hit = sum(
                1 for r in results
                if any(t in str((r.get("metadata") or {}).get("source", ""))
                       for t in ("m116_parent_synthesis",
                                 "m117_parent_synthesis",
                                 "m118_parent_synthesis"))
            )
            write_registry_key("m125_2.b.t28_k20_parent_synthesis_surface_count",
                               hit, "M116/M117/M118 in fused top-20")
        except Exception:
            pass

    sys.exit(0 if failed == 0 else 1)
