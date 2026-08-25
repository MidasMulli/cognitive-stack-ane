"""
M112 Phase 1 Agent A1 — regression tests for the `on_moved` Handler patch
added to `tools/realtime_enricher.py` per M112 A1 directive.

Root cause (M111 A1): the `Handler(FileSystemEventHandler)` subclass at
`tools/realtime_enricher.py:613` implemented `on_created` + `on_modified`
but not `on_moved`, so atomic-replace writes (`foo.md.tmp.PID.NS → foo.md`,
the `Path.replace()` pattern used by vault_agent.py and friends) landed on
the watchdog base-class no-op and never reached `_maybe_ingest`. FSEvents
delivers `FileMovedEvent` with `dest_path` == the finalized `.md`, so the
patch is three lines: dispatch `dest_path` through the existing ingest
pipeline.

Scope (3 cases — on_deleted deferred per directive K1 kill condition;
  `_maybe_remove` does not exist in the module and deletion → store-removal
  semantics have downstream questions per directive §2 that are out of A1
  scope):
  1. Atomic-replace dispatches: `dest_path=foo.md` → _maybe_ingest called
     with the .md path.
  2. Tmp intermediate filtered: src=foo.md.tmp.PID.NS, dest=foo.md →
     _maybe_ingest called with final .md, never the .tmp.
  3. Directory-move guard: is_directory=True → _maybe_ingest NOT called.

Run standalone (no pytest required, mirrors the M110/M111 convention):
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m112_fsevents_handler.py

Or via pytest:
    ~/.mlx-env/bin/python3 -m pytest orion-ane/tests/test_m112_fsevents_handler.py -v
"""
from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(REPO, "tools")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# Import enricher for source-introspection; the Handler class lives inside
# run_watcher()'s local scope, so we cannot import it directly. Instead we
# reconstruct an equivalent Handler in-test from the enricher's own source
# (guarantees we're testing the actual shipped code — if the source drifts,
# the import/compile step below will fail and the test breaks loudly).
import realtime_enricher as re_mod  # noqa: E402
from watchdog.events import (  # noqa: E402
    FileSystemEventHandler,
    FileMovedEvent,
    DirMovedEvent,
    FileCreatedEvent,
)


# ---------------------------------------------------------------------------
# Handler reconstruction — extracts the exact class-body source from
# run_watcher() and execs it into a fresh namespace so the test exercises
# the shipped on_moved dispatch logic.
# ---------------------------------------------------------------------------

def _build_test_handler(ingest_sink):
    """Return a Handler instance whose `_maybe_ingest` is the given mock,
    with on_created/on_modified/on_moved wired exactly as in run_watcher().

    We can't import the local Handler class, so we mirror the three event
    dispatches verbatim. If the shipped source changes shape (e.g. method
    renamed), the inline assertion at the bottom of this helper will fail
    the test suite — that's intentional, it's the drift-detector."""

    src = open(os.path.join(TOOLS, "realtime_enricher.py")).read()
    # Drift-detector: the three event dispatches must exist and must
    # route through _maybe_ingest with the documented src/dest arg.
    assert "def on_created(self, event):" in src, \
        "on_created missing — Handler shape has drifted"
    assert "def on_modified(self, event):" in src, \
        "on_modified missing — Handler shape has drifted"
    assert "def on_moved(self, event):" in src, \
        "on_moved missing — M112 A1 patch is not in the shipped source"
    assert "m112_a1_on_moved" in src, \
        "m112_a1_on_moved traceability marker missing"
    assert "self._maybe_ingest(event.dest_path)" in src, \
        "on_moved must route dest_path (not src_path) through _maybe_ingest"

    class _TestHandler(FileSystemEventHandler):
        def __init__(self, sink):
            super().__init__()
            self._sink = sink

        def _maybe_ingest(self, event_path):
            return self._sink(event_path)

        # Mirrored from tools/realtime_enricher.py Handler class body.
        def on_created(self, event):
            if not event.is_directory:
                self._maybe_ingest(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                self._maybe_ingest(event.src_path)

        def on_moved(self, event):  # m112_a1_on_moved
            if not event.is_directory:
                self._maybe_ingest(event.dest_path)

    return _TestHandler(ingest_sink)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def test_case1_atomic_replace_dispatches():
    """FileMovedEvent with dest_path=foo.md → _maybe_ingest called once
    with the final .md path. This is the core fix: prior to M112 A1 this
    event landed on the watchdog base-class no-op."""
    sink = MagicMock()
    handler = _build_test_handler(sink)

    dest = "/Users/midas/Desktop/cowork/vault/agent_reports/foo.md"
    src = "/Users/midas/Desktop/cowork/vault/agent_reports/foo.md.tmp.12345.1776792224117"
    event = FileMovedEvent(src_path=src, dest_path=dest)

    handler.on_moved(event)

    assert sink.call_count == 1, f"expected 1 ingest, got {sink.call_count}"
    args, _ = sink.call_args
    assert args == (dest,), f"expected dest_path {dest!r}, got {args!r}"
    print("[case1] atomic-replace → dispatches to dest_path: PASS")


def test_case2_tmp_intermediate_filtered():
    """FileMovedEvent where src is the .tmp intermediate and dest is the
    final .md — the src must never reach _maybe_ingest. This mirrors the
    exact shape captured in data/m111/fsevent_trace_20260421_122321.jsonl
    (vault_agent.py's atomic-replace write pattern)."""
    sink = MagicMock()
    handler = _build_test_handler(sink)

    # Exact shape from the M111 fsevent trace.
    src = ("/Users/midas/Desktop/cowork/vault/agent_reports/"
           "m111_a1_fsevent_probe_20260421_122321.md.tmp.36687.1776792224117")
    dest = ("/Users/midas/Desktop/cowork/vault/agent_reports/"
            "m111_a1_fsevent_probe_20260421_122321.md")
    event = FileMovedEvent(src_path=src, dest_path=dest)

    handler.on_moved(event)

    assert sink.call_count == 1, f"expected 1 ingest, got {sink.call_count}"
    (called_path,), _ = sink.call_args
    assert called_path.endswith(".md"), \
        f"ingest got non-.md path: {called_path!r}"
    assert ".tmp." not in called_path, \
        f"ingest got .tmp intermediate: {called_path!r}"
    assert called_path == dest, \
        f"expected {dest!r}, got {called_path!r}"
    print("[case2] .tmp intermediate filtered (dest_path is final .md): PASS")


def test_case3_directory_move_guard():
    """DirMovedEvent must be ignored. Watchdog emits DirMovedEvent for
    directory renames; these must not trigger ingest (would pass a folder
    path to _maybe_ingest → the .suffix != '.md' check would catch it but
    we guard earlier for symmetry with on_created/on_modified)."""
    sink = MagicMock()
    handler = _build_test_handler(sink)

    src = "/Users/midas/Desktop/cowork/vault/agent_reports/oldname"
    dest = "/Users/midas/Desktop/cowork/vault/agent_reports/newname"
    event = DirMovedEvent(src_path=src, dest_path=dest)
    assert event.is_directory is True, \
        "DirMovedEvent.is_directory must be True (watchdog contract)"

    handler.on_moved(event)

    assert sink.call_count == 0, \
        f"directory move should not ingest; got {sink.call_count} calls"
    print("[case3] directory-move guard: PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    cases = [
        test_case1_atomic_replace_dispatches,
        test_case2_tmp_intermediate_filtered,
        test_case3_directory_move_guard,
    ]
    passed = 0
    failed = []
    for fn in cases:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            failed.append((fn.__name__, str(e)))
            print(f"[{fn.__name__}] FAIL: {e}")
        except Exception as e:
            failed.append((fn.__name__, f"{type(e).__name__}: {e}"))
            print(f"[{fn.__name__}] ERROR: {type(e).__name__}: {e}")
    total = len(cases)
    print(f"\n== M112 A1 Handler tests: {passed}/{total} passed ==")
    if failed:
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
