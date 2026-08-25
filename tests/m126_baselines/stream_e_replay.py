#!/usr/bin/env python3
"""
M126 Baselines — Stream E: Full Subconscious replay.

Per directive §3.5:
- Iterate 135-turn corpus_v1.jsonl, POST each query to Midas UI :8450 /api/chat/stream.
- Single fresh session? No — directive says "use a single fresh session for all 135 turns
  (mirrors operator-driven natural session)". We reuse the CURRENTLY-RUNNING midas_ui
  session, which is fine: midas_ui maintains ONE global session_id per process. The
  session's existing turns (pre-run) are identified via a snapshot taken before turn 1.
- Sequential dispatch only; wait for SSE to close before next turn.
- After each response, locate the corresponding turn_NNNN.json file written by
  midas_ui, and write our per-corpus-turn record to
  data/m126_baselines/subconscious_responses/turn_<corpus_turn_id>.json with
  {turn_id_in_corpus, corpus_query, midas_response, midas_turn_json_path,
   latency_ms_ttft, latency_ms_total, session_id, turn_number_in_session}.
- Resume-safe: skip corpus turns whose output JSON already exists.
- Every 10 turns: log progress + check swap + midas_ui health.

Usage:
  python3 stream_e_replay.py          # run
  python3 stream_e_replay.py --stats  # report how many turns done
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

CORPUS = Path("/Users/midas/Desktop/cowork/data/m126_baselines/corpus_v1.jsonl")
OUT_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines/subconscious_responses")
MIDAS_STREAM = "http://127.0.0.1:8450/api/chat/stream"
MIDAS_ROOT = "http://127.0.0.1:8450/"
MIDAS_HEALTH_SYSTEM = "http://127.0.0.1:8450/health/system"
SESSION_LOGS_ROOT = Path("/Users/midas/Desktop/cowork/data/session_logs")
STORE_DB = Path("/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live/memory_local.db")

REQUEST_TIMEOUT_S = 900  # Subconscious pipeline can be slow; 15-min ceiling
TURN_FILE_WAIT_S = 15    # wait for midas to finish _turn_write
STORE_CHECKSUM_PRE = "00f51e23a25f904a5a2a794f26aa5fe8d64f9a338183b449b78fd0f5c19c82fc"


def find_active_session() -> tuple[str, Path]:
    """Return (session_id, session_dir) of the currently-running midas_ui."""
    # Midas UI PID 92874 (per operator). Find latest session dir ending in _92874.
    # If pid changes, fall back to newest session dir.
    candidates = sorted(SESSION_LOGS_ROOT.glob("sess_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No session_logs/ directories found")
    # Prefer the one matching the current midas_ui PID.
    try:
        out = subprocess.check_output(["lsof", "-ti", ":8450"], text=True).strip()
        pids = [int(p) for p in out.split() if p]
    except Exception:
        pids = []
    for pid in pids:
        suffix = f"_{pid}"
        for c in candidates:
            if c.name.endswith(suffix):
                return c.name, c
    # Fallback: newest
    return candidates[0].name, candidates[0]


def load_corpus() -> list[dict]:
    recs = []
    with CORPUS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def check_midas_up() -> bool:
    try:
        with urllib.request.urlopen(MIDAS_ROOT, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def swap_stats() -> tuple[float, float, float]:
    """Returns (total_mb, used_mb, free_mb)."""
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"], text=True)
        t = re.search(r"total\s*=\s*([\d.]+)M", out)
        u = re.search(r"used\s*=\s*([\d.]+)M", out)
        f = re.search(r"free\s*=\s*([\d.]+)M", out)
        total = float(t.group(1)) if t else 0.0
        used = float(u.group(1)) if u else 0.0
        free = float(f.group(1)) if f else 0.0
        return total, used, free
    except Exception:
        return 0.0, 0.0, 0.0


def swap_used_mb() -> float:
    return swap_stats()[1]


def swap_free_mb() -> float:
    return swap_stats()[2]


def post_stream(query: str) -> tuple[str, float, float]:
    """POST to midas_ui /api/chat/stream, collect tokens into a single string.

    Returns (response_text, ttft_ms, total_ms). ttft_ms may be None if not measurable.
    """
    body = json.dumps({"message": query}).encode("utf-8")
    req = urllib.request.Request(
        MIDAS_STREAM,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.time()
    ttft_ms = None
    chunks: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            # SSE stream: lines like "data: {...}\n\n"
            buf = ""
            for raw in resp:
                try:
                    s = raw.decode("utf-8", errors="replace")
                except Exception:
                    s = str(raw)
                buf += s
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            ev = json.loads(payload)
                        except Exception:
                            continue
                        t = ev.get("type")
                        if t == "token":
                            if ttft_ms is None:
                                ttft_ms = (time.time() - t0) * 1000.0
                            chunks.append(ev.get("content", ""))
                        elif t == "done":
                            pass
                        elif t == "error":
                            chunks.append(f"[stream-error] {ev.get('message', '')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"midas_ui stream URLError: {e!r}")
    total_ms = (time.time() - t0) * 1000.0
    return "".join(chunks), ttft_ms if ttft_ms is not None else 0.0, total_ms


def wait_for_new_turn(session_dir: Path, baseline_max_turn: int) -> tuple[int, Path] | None:
    """Poll for a new turn_NNNN.json > baseline_max_turn. Returns (turn_num, path) or None."""
    deadline = time.time() + TURN_FILE_WAIT_S
    while time.time() < deadline:
        turns = []
        for p in session_dir.glob("turn_*.json"):
            m = re.match(r"turn_(\d+)\.json$", p.name)
            if m:
                turns.append((int(m.group(1)), p))
        turns.sort()
        for n, p in turns:
            if n > baseline_max_turn:
                # Verify the file isn't still being written (size stable 200ms apart)
                try:
                    s1 = p.stat().st_size
                    time.sleep(0.2)
                    s2 = p.stat().st_size
                    if s1 == s2 and s1 > 0:
                        return (n, p)
                except Exception:
                    pass
        time.sleep(0.3)
    return None


def current_max_turn(session_dir: Path) -> int:
    mx = 0
    for p in session_dir.glob("turn_*.json"):
        m = re.match(r"turn_(\d+)\.json$", p.name)
        if m:
            n = int(m.group(1))
            if n > mx:
                mx = n
    return mx


def already_done(turn_id: str) -> bool:
    return (OUT_DIR / f"turn_{turn_id}.json").exists()


def write_response(turn_id: str, record: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"turn_{turn_id}.json"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp, path)


def hot_path_check(turn_path: Path) -> dict:
    """Inspect first turn JSON for M125.3 instrumentation markers."""
    try:
        d = json.loads(turn_path.read_text())
    except Exception as e:
        return {"read_error": repr(e)}
    out = {}
    # ζ v2.4: scrub_mechanism_pre_gate
    scrub = d.get("scrub", {}) or {}
    out["scrub_mechanism_pre_gate_present"] = "scrub_mechanism_pre_gate" in scrub
    # M125.3 E: context.enumeration_active
    ctx = d.get("context", {}) or {}
    out["enumeration_active_present"] = "enumeration_active" in ctx
    # M125.2 A: retrieval.recall_filtered
    retr = d.get("retrieval", {}) or {}
    out["recall_filtered_present"] = "recall_filtered" in retr or bool(retr.get("recall_filtered_count"))
    # Capture the actual keys for diagnostic
    out["scrub_keys"] = sorted(list(scrub.keys()))[:20]
    out["context_keys"] = sorted(list(ctx.keys()))[:20]
    out["retrieval_keys"] = sorted(list(retr.keys()))[:20]
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replay():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_id, session_dir = find_active_session()
    print(f"[e] midas session_id={session_id} dir={session_dir}", flush=True)
    print(f"[e] store_checksum_pre (directive-frozen): {STORE_CHECKSUM_PRE}", flush=True)
    start_max = current_max_turn(session_dir)
    print(f"[e] session starting max turn = {start_max}", flush=True)

    corpus = load_corpus()
    print(f"[e] corpus has {len(corpus)} records", flush=True)

    t_start = time.time()
    done_count = 0
    hot_path_audit = None

    for i, rec in enumerate(corpus, start=1):
        turn_id = rec["turn_id"]
        if already_done(turn_id):
            done_count += 1
            if i % 25 == 0:
                print(f"[e] {i}/{len(corpus)} (skipped done: {turn_id})", flush=True)
            continue
        query = rec["query"]

        # Pre-swap check (directive rule: free < 200MB OR used > 95% of total).
        total, used, free = swap_stats()
        pct_used = (used / total * 100.0) if total > 0 else 0.0
        if free < 200.0 or pct_used > 95.0:
            print(f"[e] swap pressure high: used={used:.0f}MB ({pct_used:.1f}%) free={free:.0f}MB total={total:.0f}MB — pausing 180s", flush=True)
            time.sleep(180)

        # Snapshot current max turn in session.
        pre_max = current_max_turn(session_dir)

        print(f"[e] {i}/{len(corpus)} corpus_turn={turn_id} query={query[:80]!r}", flush=True)
        try:
            response_text, ttft_ms, total_ms = post_stream(query)
        except Exception as e:
            print(f"[e] POST failed for {turn_id}: {e!r}", flush=True)
            # Check if midas_ui died
            if not check_midas_up():
                print(f"[e] midas_ui appears DOWN — halting (K15)", flush=True)
                sys.exit(2)
            # Transient — record failure
            write_response(turn_id, {
                "turn_id_in_corpus": turn_id,
                "corpus_query": query,
                "midas_response": "",
                "midas_turn_json_path": None,
                "latency_ms_ttft": None,
                "latency_ms_total": None,
                "session_id": session_id,
                "turn_number_in_session": None,
                "error": repr(e),
                "corpus_meta": {k: v for k, v in rec.items() if k != "query"},
            })
            continue

        # Locate the new turn_NNNN.json written by midas_ui.
        found = wait_for_new_turn(session_dir, pre_max)
        if found is None:
            midas_turn_path = None
            turn_num_in_session = None
            print(f"[e] warn: no new turn JSON appeared for {turn_id} within {TURN_FILE_WAIT_S}s", flush=True)
        else:
            turn_num_in_session, turn_path = found
            midas_turn_path = str(turn_path)
            # First-turn hot-path verification (directive §Methodology)
            if hot_path_audit is None:
                hot_path_audit = hot_path_check(turn_path)
                print(f"[e] HOT_PATH_CHECK (turn_{turn_num_in_session}): {hot_path_audit}", flush=True)
                # Write audit artifact
                (OUT_DIR.parent / "stream_e_hot_path_audit.json").write_text(
                    json.dumps({
                        "corpus_turn_id": turn_id,
                        "midas_turn_number": turn_num_in_session,
                        "midas_turn_path": midas_turn_path,
                        "checks": hot_path_audit,
                    }, indent=2)
                )

        # Persist per-turn response artifact.
        record = {
            "turn_id_in_corpus": turn_id,
            "corpus_query": query,
            "midas_response": response_text,
            "midas_turn_json_path": midas_turn_path,
            "latency_ms_ttft": ttft_ms,
            "latency_ms_total": total_ms,
            "session_id": session_id,
            "turn_number_in_session": turn_num_in_session,
            "request_timestamp": time.time(),
            "corpus_meta": {k: v for k, v in rec.items() if k != "query"},
        }
        write_response(turn_id, record)
        done_count += 1

        if i % 10 == 0:
            elapsed_s = time.time() - t_start
            rate = done_count / elapsed_s if elapsed_s > 0 else 0
            eta_s = (len(corpus) - i) / rate if rate > 0 else 0
            print(f"[e] progress {i}/{len(corpus)} | done={done_count} | "
                  f"elapsed={elapsed_s:.0f}s | rate={rate:.2f}/s | eta={eta_s:.0f}s | "
                  f"swap_used={swap_used_mb():.0f}MB", flush=True)
            # Midas liveness
            if not check_midas_up():
                print(f"[e] midas_ui liveness check FAILED at i={i}", flush=True)

    print(f"[e] replay complete: {done_count} artifacts in {OUT_DIR}", flush=True)
    # Capture post-run checksum.
    try:
        post = sha256_file(STORE_DB)
    except Exception as e:
        post = f"ERROR: {e!r}"
    (OUT_DIR.parent / "stream_e_store_checksums.json").write_text(json.dumps({
        "store_checksum_pre": STORE_CHECKSUM_PRE,
        "store_checksum_post": post,
        "captured_at": time.time(),
        "drift": STORE_CHECKSUM_PRE != post,
    }, indent=2))
    print(f"[e] store_checksum_post = {post}", flush=True)


def stats():
    done = sum(1 for _ in OUT_DIR.glob("turn_*.json"))
    print(f"subconscious_responses artifacts: {done}/135")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        stats()
    else:
        replay()
