#!/usr/bin/env python3
"""
M126 Baselines — Stream B (Baseline 1: raw Gemma 4 31B Q4).

Per directive §3.2:
- Constant minimal system prompt, zero context injection, zero memory, zero RAG.
- 135 turns from corpus_v1.jsonl, sequential dispatch to :8899 /v1/chat/completions.
- Per-turn JSON written immediately to disk for checkpointing.
- `system_prompt_hash` recorded per turn; MUST be identical across all 135.

Outputs:
- data/m126_baselines/baseline_1_responses/turn_<turn_id>.json  (135 files)

Run: python3 baseline_1_raw_model.py
Resume-safe: skips turns whose output JSON already exists.
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

CORPUS = Path("/Users/midas/Desktop/cowork/data/m126_baselines/corpus_v1.jsonl")
OUT_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_1_responses")
ENDPOINT = "http://127.0.0.1:8899/v1/chat/completions"
HEALTH = "http://127.0.0.1:8899/health"

# VERBATIM per directive §3.2 — DO NOT EDIT.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly. "
    'If you don\'t know the answer, say "I don\'t have information about that."'
)
SYSTEM_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

MAX_TOKENS = 512
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 300  # hard cap per request


def health_check() -> dict:
    with urllib.request.urlopen(HEALTH, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    model = data.get("model", "")
    expected = "/Users/midas/models/gemma-4-31b-it-4bit"
    if model != expected:
        raise RuntimeError(f"Unexpected model on :8899 -> {model!r} (expected {expected!r})")
    return data


def call_server(query: str) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "model": "gemma-4-31b-it-4bit",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        raw = resp.read().decode("utf-8")
    latency_ms = (time.perf_counter() - t0) * 1000.0
    obj = json.loads(raw)
    return obj, latency_ms


def iter_corpus():
    with CORPUS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    health = health_check()
    print(f"[M126 B] :8899 ready model={health['model']!r} status={health['status']}")
    print(f"[M126 B] system_prompt_hash={SYSTEM_PROMPT_HASH}")

    records = list(iter_corpus())
    total = len(records)
    print(f"[M126 B] corpus turns: {total}")

    done = 0
    skipped = 0
    errors = 0
    t_start = time.perf_counter()

    for i, rec in enumerate(records, start=1):
        turn_id = rec["turn_id"]
        out_path = OUT_DIR / f"turn_{turn_id}.json"
        if out_path.exists():
            skipped += 1
            done += 1
            if i % 10 == 0:
                print(f"[M126 B] {i}/{total} (skipped existing; turn_id={turn_id})")
            continue

        query = rec["query"]
        request_ts = time.time()
        try:
            obj, latency_ms = call_server(query)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            errors += 1
            print(f"[M126 B] ERROR turn_id={turn_id}: {e!r}")
            # Write an error stub so we can resume cleanly; manual fix required.
            stub = {
                "turn_id": turn_id,
                "query": query,
                "error": repr(e),
                "request_timestamp": request_ts,
                "system_prompt_hash": SYSTEM_PROMPT_HASH,
            }
            out_path.write_text(json.dumps(stub, indent=2))
            continue

        choices = obj.get("choices", [{}])
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        finish = choices[0].get("finish_reason", "unknown")
        usage = obj.get("usage", {})
        xsd = obj.get("x_spec_decode", {})

        payload_out = {
            "turn_id": turn_id,
            "query": query,
            "response": content,
            "latency_ms_total": round(latency_ms, 2),
            "latency_ms_ttft": None,  # non-streaming; TTFT not captured per §3.2
            "model_identifier": obj.get("model", ""),
            "system_prompt_hash": SYSTEM_PROMPT_HASH,
            "request_timestamp": request_ts,
            "completion_reason": finish,
            "usage": usage,
            "prefill_ms": xsd.get("prefill_ms"),
            "tps": xsd.get("tps"),
            # Corpus metadata preserved for downstream scoring convenience.
            "corpus_meta": {
                "ground_truth_answer": rec.get("ground_truth_answer"),
                "ground_truth_source_location": rec.get("ground_truth_source_location"),
                "ground_truth_ambiguous": rec.get("ground_truth_ambiguous"),
                "query_class": rec.get("query_class"),
                "topical_relevance_inferred": rec.get("topical_relevance_inferred"),
            },
        }
        out_path.write_text(json.dumps(payload_out, indent=2))
        done += 1

        if i % 10 == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            avg = elapsed / max(1, i - skipped if skipped < i else 1)
            print(
                f"[M126 B] {i}/{total} turn_id={turn_id} "
                f"latency={latency_ms/1000:.1f}s avg/turn={avg:.1f}s "
                f"elapsed={elapsed/60:.1f}min errors={errors}"
            )

    # Final hash-consistency verify: scan all response JSONs, confirm uniform hash.
    hashes = set()
    for p in sorted(OUT_DIR.glob("turn_*.json")):
        obj = json.loads(p.read_text())
        h = obj.get("system_prompt_hash")
        if h:
            hashes.add(h)
    if len(hashes) != 1 or SYSTEM_PROMPT_HASH not in hashes:
        print(f"[M126 B] FAIL: inconsistent system_prompt_hash across responses: {hashes}")
        sys.exit(2)
    print(f"[M126 B] system_prompt_hash uniform across {len(list(OUT_DIR.glob('turn_*.json')))} files -> OK")
    print(f"[M126 B] done={done} skipped={skipped} errors={errors} total={total}")


if __name__ == "__main__":
    main()
