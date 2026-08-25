"""
M110 Phase 1 Agent L1 — regression tests for the liveness guard added to
`tools/realtime_enricher.py` per M106 V1 spec.

Scope:
  1. Heartbeat format v2 round-trip (all four fields).
  2. Watchdog-dead path: observer.is_alive() returns False → warning.
  3. Ingestion-stale path: vault_fresh_mtime > last_ingest_ts + 5min → warning.
  4. Healthy path: both signals healthy → no warning.
  5. Backward compat: v1 bare-float heartbeat still reads cleanly.

Run standalone (no pytest required, mirrors the M103/M109 convention):
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m110_liveness_guard.py

Or via pytest:
    ~/.mlx-env/bin/python3 -m pytest orion-ane/tests/test_m110_liveness_guard.py -v
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(REPO, "tools")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# Import the enricher module. The top-level sys.path inserts inside the
# module are benign for this test — we never call run_watcher() or
# get_store(); only the pure helpers and constants.
import realtime_enricher as re_mod  # noqa: E402


# -----------------------------------------------------------------------------
# Helper: capture log records from the enricher's log handle
# -----------------------------------------------------------------------------

class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def warnings(self):
        return [r.getMessage() for r in self.records
                if r.levelno >= logging.WARNING]


def _with_capture(fn):
    """Run fn with a capture handler attached to re_mod.log, return
    (fn_result, warnings_list)."""
    cap = _CaptureHandler()
    re_mod.log.addHandler(cap)
    try:
        result = fn()
    finally:
        re_mod.log.removeHandler(cap)
    return result, cap.warnings()


# -----------------------------------------------------------------------------
# Helper: temp heartbeat file so tests do not touch the live vault heartbeat
# -----------------------------------------------------------------------------

class _temp_heartbeat:
    """Context manager: redirect re_mod.HEARTBEAT_FILE to a temp path for
    the duration of the test, then restore."""
    def __enter__(self):
        self._saved = re_mod.HEARTBEAT_FILE
        fd, path = tempfile.mkstemp(prefix="m110_hb_", suffix=".json")
        os.close(fd)
        os.unlink(path)  # write_heartbeat will recreate
        self.path = Path(path)
        re_mod.HEARTBEAT_FILE = self.path
        return self.path

    def __exit__(self, exc_type, exc, tb):
        re_mod.HEARTBEAT_FILE = self._saved
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Test 1 — heartbeat format v2 round trip
# -----------------------------------------------------------------------------

def test_heartbeat_format_v2_roundtrip():
    """write_heartbeat with all four fields populated produces JSON on
    disk, and read_heartbeat_v2 returns all four fields back."""
    with _temp_heartbeat() as hb_path:
        now = time.time()
        fresh = now - 10.0
        last = now - 120.0
        re_mod.write_heartbeat(
            now,
            watchdog_alive=True,
            last_ingest_ts=last,
            vault_fresh_mtime=fresh,
        )
        raw = hb_path.read_text().strip()
        assert raw.startswith("{"), (
            f"expected JSON payload on disk, got {raw!r}")
        payload = json.loads(raw)
        assert payload["schema_version"] == 2
        assert abs(payload["timestamp"] - now) < 0.01
        assert payload["watchdog_alive"] is True
        assert abs(payload["last_ingest_ts"] - last) < 0.01
        assert abs(payload["vault_fresh_mtime"] - fresh) < 0.01

        # read_heartbeat_v2 round trip
        v2 = re_mod.read_heartbeat_v2()
        assert v2["schema_version"] == 2
        assert abs(v2["timestamp"] - now) < 0.01
        assert v2["watchdog_alive"] is True
        assert abs(v2["last_ingest_ts"] - last) < 0.01
        assert abs(v2["vault_fresh_mtime"] - fresh) < 0.01

        # read_heartbeat (v1-style float getter) still returns timestamp
        assert abs(re_mod.read_heartbeat() - now) < 0.01


# -----------------------------------------------------------------------------
# Test 2 — watchdog-dead warning
# -----------------------------------------------------------------------------

def test_watchdog_dead_logs_warning():
    """check_liveness with a mocked observer whose is_alive() returns
    False must log a warning mentioning 'not alive'."""
    fake_observer = MagicMock()
    fake_observer.is_alive.return_value = False
    now = time.time()

    def run():
        return re_mod.check_liveness(
            fake_observer,
            last_ingest_ts=now - 30.0,
            vault_fresh_mtime=now - 30.0,  # no divergence; isolate dead-thread signal
        )

    status, warnings = _with_capture(run)
    assert status["watchdog_alive"] is False
    assert any("not alive" in w for w in warnings), (
        f"expected 'not alive' warning, got {warnings}")


# -----------------------------------------------------------------------------
# Test 3 — ingestion-stale warning (watchdog alive but stale)
# -----------------------------------------------------------------------------

def test_ingestion_stale_logs_warning():
    """Observer alive but vault has fresh writes older than last ingest
    by > 5 min → warning logged, divergence_min > 5."""
    fake_observer = MagicMock()
    fake_observer.is_alive.return_value = True
    now = time.time()
    last_ingest = now - 600.0  # 10 min ago
    vault_fresh = now           # fresh now

    def run():
        return re_mod.check_liveness(
            fake_observer,
            last_ingest_ts=last_ingest,
            vault_fresh_mtime=vault_fresh,
        )

    status, warnings = _with_capture(run)
    assert status["watchdog_alive"] is True
    assert status["divergence_min"] > 5.0
    assert any("ingestion stale" in w for w in warnings), (
        f"expected 'ingestion stale' warning, got {warnings}")


# -----------------------------------------------------------------------------
# Test 4 — healthy path (no warnings)
# -----------------------------------------------------------------------------

def test_healthy_path_emits_no_warning():
    """Observer alive + last_ingest within threshold → no warning."""
    fake_observer = MagicMock()
    fake_observer.is_alive.return_value = True
    now = time.time()
    # Ingest 30s ago, vault fresh mtime 30s ago → 0 divergence
    last_ingest = now - 30.0
    vault_fresh = now - 30.0

    def run():
        return re_mod.check_liveness(
            fake_observer,
            last_ingest_ts=last_ingest,
            vault_fresh_mtime=vault_fresh,
        )

    status, warnings = _with_capture(run)
    assert status["watchdog_alive"] is True
    assert status["divergence_min"] == 0.0
    assert warnings == [], (
        f"expected no warnings, got {warnings}")


# -----------------------------------------------------------------------------
# Test 5 — backward compat: v1 bare-float heartbeat still reads
# -----------------------------------------------------------------------------

def test_backward_compat_v1_bare_float_reads_cleanly():
    """A heartbeat file containing only a bare float (v1 format) must be
    read without crashing, and read_heartbeat must return the float.
    read_heartbeat_v2 returns a dict with timestamp populated and the
    other fields None/False."""
    with _temp_heartbeat() as hb_path:
        # Simulate a v1-era heartbeat on disk (what pre-M110 would have
        # written: a bare float with trailing newline).
        v1_ts = 1776786938.63
        hb_path.write_text(f"{v1_ts:.2f}\n")

        # v1 reader must still work
        got = re_mod.read_heartbeat()
        assert abs(got - v1_ts) < 0.01, (got, v1_ts)

        # v2 reader must also work (fallback branch) and not crash
        v2 = re_mod.read_heartbeat_v2()
        assert isinstance(v2, dict)
        assert abs(v2["timestamp"] - v1_ts) < 0.01
        # Other fields not present on v1 disk → defaults
        assert v2["watchdog_alive"] is None
        assert v2["last_ingest_ts"] is None
        assert v2["vault_fresh_mtime"] is None


# -----------------------------------------------------------------------------
# Bonus: exercise the `write_heartbeat(ts)` single-arg call (legacy shape)
# still produces a v2-readable file with no extra fields populated.
# -----------------------------------------------------------------------------

def test_write_heartbeat_legacy_single_arg_still_works():
    """write_heartbeat(ts) with no other args writes JSON with only
    `timestamp` + schema_version. Old single-arg callers continue to
    work without behavior change to downstream float readers."""
    with _temp_heartbeat() as hb_path:
        ts = time.time()
        re_mod.write_heartbeat(ts)
        raw = hb_path.read_text().strip()
        assert raw.startswith("{"), raw
        payload = json.loads(raw)
        assert payload["schema_version"] == 2
        assert abs(payload["timestamp"] - ts) < 0.01
        # Optional fields absent
        assert "watchdog_alive" not in payload
        assert "last_ingest_ts" not in payload
        assert "vault_fresh_mtime" not in payload
        # read_heartbeat must still return the float
        assert abs(re_mod.read_heartbeat() - ts) < 0.01


# -----------------------------------------------------------------------------
# Bonus: compute_vault_fresh_mtime returns 0.0 or a finite float without
# crashing. We do not assert on a specific value because it walks the live
# vault. We only assert the contract: nonneg float, no exception.
# -----------------------------------------------------------------------------

def test_compute_vault_fresh_mtime_contract():
    """Smoke test: compute_vault_fresh_mtime runs without exception and
    returns a non-negative float. The actual max mtime is environment-
    dependent."""
    v = re_mod.compute_vault_fresh_mtime()
    assert isinstance(v, float), type(v)
    assert v >= 0.0


# -----------------------------------------------------------------------------
# Standalone runner (mirrors test_m109_turn_log_schema_v2 convention)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_heartbeat_format_v2_roundtrip,
        test_watchdog_dead_logs_warning,
        test_ingestion_stale_logs_warning,
        test_healthy_path_emits_no_warning,
        test_backward_compat_v1_bare_float_reads_cleanly,
        test_write_heartbeat_legacy_single_arg_still_works,
        test_compute_vault_fresh_mtime_contract,
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
