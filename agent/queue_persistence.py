"""Disk-backed queue state log — Main 34 S2A.

Append-only JSONL of queue lifecycle events. Both midas_ui (worker side) and
subconscious_daemon (observer side) read/write this file. On midas_ui boot,
`recover_incomplete_queues()` scans the log and returns any queue_ids that
were enqueued but never marked complete — caller then re-runs them.

Schema (one JSON object per line):
    {"ts": ISO8601, "op": "enqueue"|"start"|"task_complete"|"complete"|"fail",
     "queue_id": "qYYYYMMDDHHMMSS", "task_idx": int|null,
     "task_count": int|null, "tasks": [...]|null, "error": str|null}

The `enqueue` record contains the full task list so recovery can replay
without depending on any other store.
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path

QUEUE_LOG = Path("/Users/midas/Desktop/cowork/vault/subconscious/queue_state.jsonl")
_lock = threading.Lock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _append(record: dict) -> None:
    QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"))
    with _lock:
        with open(QUEUE_LOG, "a") as fh:
            fh.write(line + "\n")


def log_enqueue(queue_id: str, tasks: list) -> None:
    _append({"ts": _now(), "op": "enqueue", "queue_id": queue_id,
             "task_count": len(tasks), "tasks": tasks})


def log_start(queue_id: str, task_idx: int) -> None:
    _append({"ts": _now(), "op": "start", "queue_id": queue_id,
             "task_idx": task_idx})


def log_task_complete(queue_id: str, task_idx: int, result_path: str | None = None) -> None:
    _append({"ts": _now(), "op": "task_complete", "queue_id": queue_id,
             "task_idx": task_idx, "result_path": result_path})


def log_complete(queue_id: str) -> None:
    _append({"ts": _now(), "op": "complete", "queue_id": queue_id})


def log_fail(queue_id: str, error: str) -> None:
    _append({"ts": _now(), "op": "fail", "queue_id": queue_id, "error": error})


def replay() -> dict:
    """Return {queue_id: {tasks, last_completed_idx, status}} from full log."""
    state: dict = {}
    if not QUEUE_LOG.exists():
        return state
    with open(QUEUE_LOG) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qid = rec.get("queue_id")
            if not qid:
                continue
            entry = state.setdefault(qid, {
                "tasks": [], "last_completed_idx": -1,
                "status": "unknown", "first_seen": rec["ts"]})
            op = rec.get("op")
            if op == "enqueue":
                entry["tasks"] = rec.get("tasks") or []
                entry["status"] = "enqueued"
            elif op == "start":
                entry["status"] = "running"
            elif op == "task_complete":
                idx = rec.get("task_idx")
                if isinstance(idx, int) and idx > entry["last_completed_idx"]:
                    entry["last_completed_idx"] = idx
            elif op == "complete":
                entry["status"] = "complete"
            elif op == "fail":
                entry["status"] = "failed"
                entry["error"] = rec.get("error")
    return state


def recover_incomplete_queues() -> list[dict]:
    """Return [{queue_id, remaining_tasks}, ...] for queues that started but
    never reached `complete`. Suitable for re-firing on midas_ui boot.
    """
    state = replay()
    out = []
    for qid, entry in state.items():
        if entry["status"] in ("complete", "failed"):
            continue
        if not entry["tasks"]:
            continue
        remaining = entry["tasks"][entry["last_completed_idx"] + 1:]
        if remaining:
            out.append({"queue_id": qid, "remaining_tasks": remaining,
                        "original_count": len(entry["tasks"]),
                        "completed": entry["last_completed_idx"] + 1})
    return out


def tail_events(since_offset: int = 0):
    """Yield (offset, record) for each line at or after since_offset bytes.

    Used by subconscious_daemon to stream queue events into the event bus.
    Returns the new offset alongside.
    """
    if not QUEUE_LOG.exists():
        return
    with open(QUEUE_LOG) as fh:
        fh.seek(since_offset)
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield fh.tell(), rec
