"""M125.2 Stream A — correct-location port tests.

Validates that the atomic-row cosine-dominant scorer shipped at
`orion-ane/memory/local_store.py:432-438` surfaces `canonical_atom` rows
on the production hot path (memory.recall → daemon.store.recall), while
preserving:

- M125.1 A `multi_path_retrieve.py:619` atomic path (not regressed).
- Non-atomic canonical rows scored under the generic 1.30× canonical path.
- M125 A1/A2/A5 and M122 A1 paths unaffected.

Run:
    /Users/midas/.mlx-env/bin/python3 -m pytest \
        orion-ane/tests/test_m125_2_a_correct_location.py -v
"""

from __future__ import annotations

import os
import sys

ROOT = "/Users/midas/Desktop/cowork"
sys.path.insert(0, os.path.join(ROOT, "orion-ane/agent"))
sys.path.insert(0, os.path.join(ROOT, "orion-ane/memory"))
sys.path.insert(0, os.path.join(ROOT, "vault/subconscious"))


A3_QUERIES = [
    "what was the cold prefill time in Main 25?",
    "1.01 ms GPU-to-ANE gate delay",
    "WKdm LZ4 compression",
    "M83 Rule 1 amendment date",
]


def _build_bridge():
    from memory_bridge import MemoryBridge
    mb = MemoryBridge()
    mb.start(enable_enricher=False)
    return mb


def test_production_hot_path_surfaces_atomic_rows():
    """memory.recall() (MemoryBridge → multi_path → store.recall) returns
    canonical_atom rows for all 4 A3 target queries."""
    mb = _build_bridge()
    try:
        for q in A3_QUERIES:
            res = mb.recall(q, n_results=15)
            assert res, f"no recall result for {q!r}"
            results = res.get("results", [])
            assert len(results) > 0, f"no results for {q!r}"
            atomic = [
                r for r in results
                if ((r.get("metadata") or {}).get("type") == "canonical_atom")
                or (r.get("type") == "canonical_atom")
            ]
            assert len(atomic) >= 1, (
                f"no canonical_atom rows surfaced for {q!r}; "
                f"top sources: "
                f"{[(r.get('source_role'), r.get('type')) for r in results[:5]]}"
            )
    finally:
        mb.stop()


def test_atomic_score_above_filter_threshold():
    """canonical_atom rows returned from memory.recall() score above the
    midas_ui 0.40 filter threshold."""
    mb = _build_bridge()
    try:
        q = "what was the cold prefill time in Main 25?"
        res = mb.recall(q, n_results=15)
        results = res.get("results", [])
        atomic = [
            r for r in results
            if ((r.get("metadata") or {}).get("type") == "canonical_atom")
            or (r.get("type") == "canonical_atom")
        ]
        assert atomic, "no atomic rows returned"
        for r in atomic[:3]:
            assert r.get("score", 0) > 0.40, (
                f"atomic row score {r.get('score')} below 0.40 filter: "
                f"{r.get('text','')[:80]!r}"
            )
    finally:
        mb.stop()


def test_store_recall_directly_scores_atomic_with_cosine_dominant():
    """LocalMemoryStore.recall() applies the cosine-dominant scorer to
    canonical_atom rows (type predicate == 'canonical_atom')."""
    mb = _build_bridge()
    try:
        store = mb.daemon.store
        q = "cold prefill time in Main 25 cached speedup"
        results = store.recall(q, n_results=30)
        atomic = [
            r for r in results
            if (r.get("metadata") or {}).get("type") == "canonical_atom"
        ]
        assert atomic, "store.recall surfaces no canonical_atom rows"
        # Verify at least one atomic row is above the naive 0.30 floor
        # (cosine-dominant 1.80× boost should lift it there).
        top = atomic[0]
        assert top["score"] > 0.30, (
            f"top atomic score {top['score']} too low; "
            f"cosine-dominant path did not apply"
        )
    finally:
        mb.stop()


def test_non_atomic_canonical_gets_generic_boost():
    """Non-atomic canonical rows (source_role='canonical', type != 'canonical_atom')
    are scored under the generic 1.30× multiplier, not the cosine-dominant path."""
    mb = _build_bridge()
    try:
        store = mb.daemon.store
        res = store.recall("production daemon ports services", n_results=30)
        # Find a non-atomic canonical row (e.g., type='state')
        non_atomic_canonical = [
            r for r in res
            if (r.get("metadata") or {}).get("source_role") == "canonical"
            and (r.get("metadata") or {}).get("type") != "canonical_atom"
        ]
        if not non_atomic_canonical:
            # Skip — no non-atomic canonicals in top 30 for this query
            import pytest
            pytest.skip("no non-atomic canonicals in top-30 for probe query")
        # Compute the expected range: similarity * 0.75 + rec * 0.25, * 1.30
        # which is strictly <= 1.30 (when similarity=1.0, rec=1.0). For
        # typical canonical-state matches, cosine ~0.40 → score ~0.45.
        for r in non_atomic_canonical[:3]:
            # Under generic 1.30×: score = (sim*0.75 + rec*0.25) * 1.30
            # Under cosine-dominant 1.80×: score = (sim*0.85 + rec*0.15) * 1.80
            # For a given similarity and rec, cosine-dominant is ~1.38× higher.
            # Bound check: generic should stay well below the cosine-dominant
            # peak; specifically, for similarity=0.4, rec=0.3: generic = 0.49,
            # cosine-dominant = 0.69. We expect the generic path here.
            assert r["score"] < 1.31, (
                f"non-atomic canonical score {r['score']} exceeds max generic "
                f"boost — unexpected cosine-dominant application"
            )
    finally:
        mb.stop()


def test_multi_path_atomic_fix_not_regressed():
    """M125.1 A fix at multi_path_retrieve.py:619 still returns atomic
    rows when called directly (do not remove it)."""
    from multi_path_retrieve import multi_path_recall
    mb = _build_bridge()
    try:
        store = mb.daemon.store
        for q in A3_QUERIES:
            mp = multi_path_recall(q, store, n_results=15, candidate_pool=100)
            assert mp, f"multi_path_recall empty for {q!r}"
            atomic = [
                r for r in mp
                if (r.get("metadata") or {}).get("type") == "canonical_atom"
            ]
            assert len(atomic) >= 1, (
                f"multi_path regression: no atomic for {q!r} "
                f"(M125.1 A path broken)"
            )
    finally:
        mb.stop()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
