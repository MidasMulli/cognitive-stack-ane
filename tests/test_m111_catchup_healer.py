"""
M111 Phase 2 Agent A2 — regression tests for the startup catch-up healer
added to `tools/realtime_enricher.py` per M111 A2 directive §3.2.

Scope:
  1. Stale heartbeat + N fresh files not in store → exactly N ingested.
  2. Fresh heartbeat + all files in store → zero ingested, walk <1s.
  3. Mixed state (some in store, some not) → only delta ingested.
  4. Trigger condition (_healer_should_run) — unit contracts on the gate.

Run standalone (no pytest required, mirrors the M110 convention):
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m111_catchup_healer.py

Or via pytest:
    ~/.mlx-env/bin/python3 -m pytest orion-ane/tests/test_m111_catchup_healer.py -v
"""
from __future__ import annotations

import os
import sys
import time
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(REPO, "tools")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import realtime_enricher as re_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers — build a mini-vault + mini-DB on pytest tmp-style paths.
# ---------------------------------------------------------------------------

def _make_mini_db(path: Path):
    """Create a minimal memories table matching the production schema
    (id TEXT PRIMARY KEY) — the healer only needs `id`."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS memories "
                     "(id TEXT PRIMARY KEY, text TEXT, embedding BLOB)")
        conn.commit()
    finally:
        conn.close()


def _seed_id(path: Path, mem_id: str):
    """Insert a row with just the given id so the healer's IN-query finds it."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("INSERT OR IGNORE INTO memories(id, text, embedding) "
                     "VALUES (?, ?, ?)", (mem_id, "seed", b""))
        conn.commit()
    finally:
        conn.close()


def _write_md(root: Path, rel_path: str, content: str) -> Path:
    """Write `content` to root/rel_path, creating parents."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class _Fixture:
    """Isolated fixture that redirects VAULT_ROOT, STORE_DB, HEARTBEAT_FILE,
    and _HEALER_SCAN_DIRS to temp paths for the duration of the test.

    Also patches `ingest_file` to a no-op-but-count shim so we can assert
    on call shape without standing up CoreML / numpy / LocalMemoryStore.
    The healer's delta computation runs against the real sqlite DB; only
    the side-effecting write path is mocked (directive §4 "reuse, don't
    duplicate" — we exercise the reuse, we don't re-test ingest_file).
    """
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="m111_a2_"))
        self.vault = self.tmp / "vault"
        self.vault.mkdir()
        self.db = self.tmp / "memory_local.db"
        self.hb = self.tmp / ".realtime_enricher_heartbeat"
        _make_mini_db(self.db)

        # Save + swap module globals
        self._saved = {
            "VAULT_ROOT": re_mod.VAULT_ROOT,
            "STORE_DB": re_mod.STORE_DB,
            "HEARTBEAT_FILE": re_mod.HEARTBEAT_FILE,
            "_HEALER_SCAN_DIRS": list(re_mod._HEALER_SCAN_DIRS),
            "ingest_file": re_mod.ingest_file,
        }
        re_mod.VAULT_ROOT = self.vault
        re_mod.STORE_DB = str(self.db)
        re_mod.HEARTBEAT_FILE = self.hb
        # Mirror the production SCAN_DIRS shape against our temp vault —
        # nested dirs get their own entries so the recursive walk logic
        # is exercised even though the real production set is 15 entries.
        re_mod._HEALER_SCAN_DIRS = [
            self.vault,
            self.vault / "agent_reports",
            self.vault / "knowledge",
            self.vault / "subconscious",
        ]

        # Shim ingest_file so we count attempts without touching CoreML.
        # The shim also writes the deterministic id into the fixture DB so
        # subsequent same-process healer calls see it "in store" — this is
        # exactly the invariant the production ingest would establish.
        self.ingest_calls = []

        def _shim_ingest(path: Path) -> bool:
            self.ingest_calls.append(Path(path))
            try:
                abs_path = str(Path(path).resolve())
            except OSError:
                return False
            fid = re_mod.deterministic_id(abs_path)
            _seed_id(self.db, fid)
            return True

        re_mod.ingest_file = _shim_ingest

    def teardown(self):
        for k, v in self._saved.items():
            setattr(re_mod, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def det_id(self, p: Path) -> str:
        return re_mod.deterministic_id(str(Path(p).resolve()))


# ---------------------------------------------------------------------------
# Test 1 — stale heartbeat + N fresh files NOT in store → N ingested
# ---------------------------------------------------------------------------

def test_stale_heartbeat_N_fresh_files_ingests_N():
    """Simulates the exact M110 V-3 failure: heartbeat exists but the
    backlog files are not in the store. Healer must walk the vault and
    ingest all N missing files regardless of the heartbeat timestamp.
    """
    fx = _Fixture()
    try:
        # Seed a stale v2 heartbeat on disk so the trigger condition fires.
        stale_ts = time.time() - 86400  # 1 day old
        re_mod.write_heartbeat(
            stale_ts,
            watchdog_alive=True,
            last_ingest_ts=stale_ts,
            vault_fresh_mtime=stale_ts,
        )

        # Write N .md files into the mini-vault across a nested dir
        # (agent_reports/) and a top-level dir so both walk branches run.
        N = 4
        backlog = [
            _write_md(fx.vault, "agent_reports/m105_URGENT_SURFACE.md",
                      "# M105 URGENT\nsimulated backlog file 1"),
            _write_md(fx.vault, "agent_reports/m108_parent_synthesis.md",
                      "# M108 parent synth\nsimulated backlog file 2"),
            _write_md(fx.vault, "agent_reports/m109_zeta_parent_synthesis.md",
                      "# M109 zeta\nsimulated backlog file 3"),
            _write_md(fx.vault, "knowledge/turn_log_schema_v2.md",
                      "# turn_log_schema_v2\nsimulated backlog file 4"),
        ]
        assert len(backlog) == N

        # Healer trigger must fire on this heartbeat.
        hb_v2 = re_mod.read_heartbeat_v2()
        assert re_mod._healer_should_run(hb_v2), (
            f"trigger should fire on stale v2 heartbeat, got hb_v2={hb_v2}")

        metrics = re_mod.heal_catch_up()

        assert metrics["files_walked"] == N, metrics
        assert metrics["files_in_store"] == 0, metrics
        assert metrics["files_ingested_delta"] == N, metrics
        assert metrics["ingest_failures"] == 0, metrics
        assert len(fx.ingest_calls) == N, (
            f"expected {N} ingest_file calls, got {len(fx.ingest_calls)}")

        # Each backlog file was exactly the set ingest_file received.
        called_abs = {str(p.resolve()) for p in fx.ingest_calls}
        expected_abs = {str(p.resolve()) for p in backlog}
        assert called_abs == expected_abs, (called_abs, expected_abs)
    finally:
        fx.teardown()


# ---------------------------------------------------------------------------
# Test 2 — fresh heartbeat + all files in store → zero ingested, walk <1s
# ---------------------------------------------------------------------------

def test_fresh_state_idempotent_zero_delta_under_1s():
    """Idempotence gate (directive §9): second boot under healthy
    conditions walks + queries + finds zero delta + returns in <1s.
    """
    fx = _Fixture()
    try:
        # Pre-seed the store with IDs for every file we are about to write.
        files = [
            _write_md(fx.vault, f"knowledge/file_{i:02d}.md", f"content {i}")
            for i in range(10)
        ]
        for p in files:
            _seed_id(fx.db, fx.det_id(p))

        # Fresh heartbeat (healthy enricher, last ingest 30s ago).
        now = time.time()
        re_mod.write_heartbeat(
            now,
            watchdog_alive=True,
            last_ingest_ts=now - 30.0,
            vault_fresh_mtime=now - 30.0,
        )

        # With a fresh last_ingest_ts, the healer gate should NOT fire —
        # but the directive says "second boot should walk + query + find
        # zero delta + return <1s." So we test two paths:
        #   2a) gate evaluates to False (healer skipped entirely)
        #   2b) if invoked directly anyway, produces zero-delta <1s
        hb_v2 = re_mod.read_heartbeat_v2()
        assert re_mod._healer_should_run(hb_v2) is False, (
            f"fresh heartbeat should skip healer, got hb_v2={hb_v2}")

        # Directly invoke heal_catch_up to verify the idempotent-behavior
        # gate from the directive.
        t0 = time.time()
        metrics = re_mod.heal_catch_up()
        elapsed = time.time() - t0

        assert metrics["files_walked"] == len(files), metrics
        assert metrics["files_in_store"] == len(files), metrics
        assert metrics["files_ingested_delta"] == 0, metrics
        assert metrics["ingest_failures"] == 0, metrics
        assert len(fx.ingest_calls) == 0, (
            f"expected 0 ingest_file calls in idempotent path, "
            f"got {len(fx.ingest_calls)}")
        assert elapsed < 1.0, (
            f"idempotent walk must return <1s per directive gate, "
            f"got {elapsed:.3f}s (reported: {metrics['walk_wall_clock_s']})")
        # Metrics-reported wall clock should also satisfy <1s.
        assert metrics["walk_wall_clock_s"] < 1.0, metrics
    finally:
        fx.teardown()


# ---------------------------------------------------------------------------
# Test 3 — mixed state: some in store, some not → only delta ingested
# ---------------------------------------------------------------------------

def test_mixed_state_only_delta_ingested():
    """Three files in store, three not → healer ingests exactly the
    three missing ones."""
    fx = _Fixture()
    try:
        in_store = [
            _write_md(fx.vault, f"knowledge/have_{i}.md", f"have {i}")
            for i in range(3)
        ]
        not_in_store = [
            _write_md(fx.vault, f"agent_reports/missing_{i}.md", f"miss {i}")
            for i in range(3)
        ]
        for p in in_store:
            _seed_id(fx.db, fx.det_id(p))

        # Stale heartbeat so trigger fires.
        stale_ts = time.time() - 86400
        re_mod.write_heartbeat(
            stale_ts, watchdog_alive=True,
            last_ingest_ts=stale_ts, vault_fresh_mtime=stale_ts,
        )

        metrics = re_mod.heal_catch_up()
        assert metrics["files_walked"] == 6, metrics
        assert metrics["files_in_store"] == 3, metrics
        assert metrics["files_ingested_delta"] == 3, metrics
        assert metrics["ingest_failures"] == 0, metrics

        called_abs = {str(p.resolve()) for p in fx.ingest_calls}
        expected_abs = {str(p.resolve()) for p in not_in_store}
        assert called_abs == expected_abs, (
            f"healer ingested wrong set\n"
            f"  called:   {called_abs}\n"
            f"  expected: {expected_abs}")
        # And crucially: the in-store files were NOT re-ingested.
        in_store_abs = {str(p.resolve()) for p in in_store}
        assert not (called_abs & in_store_abs), (
            f"healer re-ingested already-in-store files: "
            f"{called_abs & in_store_abs}")
    finally:
        fx.teardown()


# ---------------------------------------------------------------------------
# Test 4 — trigger condition unit contracts (_healer_should_run)
# ---------------------------------------------------------------------------

def test_should_run_gate_contracts():
    """Gate fires exactly in the cases directive §3.2 specifies."""
    now = 1_000_000.0
    # v1 heartbeat → never fires (legacy mtime catch_up handles it)
    assert re_mod._healer_should_run(
        {"schema_version": 1, "last_ingest_ts": now - 10}, now=now) is False
    # v2 + last_ingest_ts absent → fires
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": None}, now=now) is True
    # v2 + last_ingest_ts = 0 → fires (treated as absent)
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": 0}, now=now) is True
    # v2 + fresh last_ingest_ts (within 300s) → does NOT fire
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": now - 30}, now=now) is False
    # v2 + stale last_ingest_ts (> 300s) → fires
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": now - 3600}, now=now) is True
    # v2 + just-over-300s boundary → fires
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": now - 301}, now=now) is True
    # v2 + just-under-300s boundary → does NOT fire
    assert re_mod._healer_should_run(
        {"schema_version": 2, "last_ingest_ts": now - 299}, now=now) is False


# ---------------------------------------------------------------------------
# Standalone runner (mirrors M110/M109 convention)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_stale_heartbeat_N_fresh_files_ingests_N,
        test_fresh_state_idempotent_zero_delta_under_1s,
        test_mixed_state_only_delta_ingested,
        test_should_run_gate_contracts,
    ]
    failures = []
    timings = []
    for t in tests:
        t0 = time.time()
        try:
            t()
            dt = time.time() - t0
            timings.append((t.__name__, dt, "PASS"))
            print(f"PASS  {t.__name__}  ({dt*1000:.1f} ms)")
        except Exception as e:
            dt = time.time() - t0
            timings.append((t.__name__, dt, f"FAIL: {e}"))
            failures.append((t.__name__, e))
            import traceback
            traceback.print_exc()
            print(f"FAIL  {t.__name__}  ({dt*1000:.1f} ms): {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} passed")
