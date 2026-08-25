#!/usr/bin/env python3
"""M125 Stream A3 — test harness for atomic canonical ingestion pipeline.

Tests:
  1. C5 4-turn replay: canonical facts now surface via substring match in
     candidates.json (dry-run) or memory_local.db (committed).
  2. Regression: pipeline does not duplicate existing source_role='canonical'
     rows (cosine-dedup >0.95 collapses near-duplicates).
  3. Idempotency: re-running produces identical candidate set modulo
     generated_at timestamp.
  4. Ingest + retrieve: any atomic row is findable via substring match on a
     distinctive numeric phrase from its text (surrogate for "fires on
     relevant specific query" in the dry-run regime).

Run:
    /Users/midas/.mlx-env/bin/python3 orion-ane/tests/test_m125_a3_atomic_ingestion.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/Users/midas/Desktop/cowork")
PIPELINE = ROOT / "tools" / "build_atomic_canonical_ingestion.py"
PYTHON = "/Users/midas/.mlx-env/bin/python3"


def _run(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [PYTHON, str(PIPELINE), *args, "--quiet"],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_c5_4_turn_replay() -> bool:
    """C5 target phrases from M124 Stream A must surface in candidates.json."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        rc, _, err = _run("--out", out)
        if rc != 0:
            print(f"[c5 replay] FAIL: pipeline rc={rc} err={err}")
            return False
        d = json.load(open(out))
        replay = d.get("c5_replay", {})
        required_groups = [
            "m123_c_T21_T37_main25_prefill",  # 4762 ms / 475 ms
            "m123_c_T27_gpu_ane_gate",        # 1.01 ms / 83.67 ms
            "m123_c_T41_compressor",          # WKdm / LZ4 / compressor
        ]
        missing = [g for g in required_groups if not replay.get(g, {}).get("any_hit")]
        if missing:
            print(f"[c5 replay] FAIL: no hits for {missing}")
            return False
        passed = sum(1 for g in required_groups if replay[g]["any_hit"])
        print(f"[c5 replay] PASS: {passed}/{len(required_groups)} groups hit")
        return True
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def test_no_canonical_duplicates() -> bool:
    """Pipeline output must not duplicate any existing canonical (cos>0.95)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        rc, _, err = _run("--out", out)
        if rc != 0:
            print(f"[dedup regression] FAIL: pipeline rc={rc}")
            return False
        d = json.load(open(out))
        conflicts = d.get("conflicts", [])
        # All conflicts must be resolved — either "kept_existing_canonical" or
        # "kept_longer_candidate". No unresolved duplicates.
        unresolved = [c for c in conflicts if c.get("resolution") not in
                      ("kept_existing_canonical", "kept_longer_candidate")]
        if unresolved:
            print(f"[dedup regression] FAIL: {len(unresolved)} unresolved conflicts")
            return False
        print(f"[dedup regression] PASS: {len(conflicts)} conflicts, all resolved")
        return True
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def test_idempotency() -> bool:
    """Two back-to-back runs produce identical candidate sets."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_a = tf.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_b = tf.name
    try:
        _run("--out", out_a)
        _run("--out", out_b)
        a = json.load(open(out_a))
        b = json.load(open(out_b))
        # Drop volatile fields.
        for d in (a, b):
            for k in ("generated_at", "extract_ms", "dedup_ms"):
                d.pop(k, None)
        same_count = a["kept_count"] == b["kept_count"]
        same_texts = [c["text"] for c in a["candidates"]] == \
                     [c["text"] for c in b["candidates"]]
        if not same_count:
            print(f"[idempotency] FAIL: counts differ a={a['kept_count']} b={b['kept_count']}")
            return False
        if not same_texts:
            print("[idempotency] FAIL: text order/content differs between runs")
            return False
        print(f"[idempotency] PASS: {a['kept_count']} candidates identical across 2 runs")
        return True
    finally:
        for p in (out_a, out_b):
            try:
                os.unlink(p)
            except Exception:
                pass


def test_atomic_row_retrievable() -> bool:
    """For each atomic candidate that contains a specific numeric phrase,
    confirm it lands in candidates.json with its distinguishing token intact.
    Surrogate for 'fires on relevant specific query' in dry-run mode."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        rc, _, err = _run("--out", out)
        if rc != 0:
            print(f"[retrievable] FAIL: pipeline rc={rc}")
            return False
        d = json.load(open(out))
        # Pick 5 specific phrases from the C5 set + a few adjacent numerics.
        probe_phrases = [
            "4762 ms",   # Main 25 cold prefill
            "475 ms",    # Main 25 cached prefill
            "1.01 ms",   # GPU→ANE gate delay
            "83.67 ms",  # ANE hold window
            "7.9 tok/s", # 8B Q8 ANE throughput (C4 turn, but also atomic)
        ]
        missing = []
        for p in probe_phrases:
            found = any(p.lower() in c["text"].lower() for c in d["candidates"])
            if not found:
                missing.append(p)
        if missing:
            print(f"[retrievable] FAIL: missing probe phrases {missing}")
            return False
        print(f"[retrievable] PASS: all {len(probe_phrases)} probe phrases present")
        return True
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass


def main() -> int:
    tests = [
        ("c5_4_turn_replay", test_c5_4_turn_replay),
        ("no_canonical_duplicates", test_no_canonical_duplicates),
        ("idempotency", test_idempotency),
        ("atomic_row_retrievable", test_atomic_row_retrievable),
    ]
    passed = 0
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"[{name}] CRASH: {type(e).__name__}: {e}")
            ok = False
        if ok:
            passed += 1
    print(f"\n=== m125_a3 tests: {passed}/{len(tests)} passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
