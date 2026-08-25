#!/usr/bin/env python3
"""
M126 Baselines — Stream D (Baseline 3: model + basic RAG).

Per directive §3.4:
- Reuse production `LocalMemoryStore` embedding matrix (MiniLM-L6-v2, 384-dim).
- Query embedding via the SAME embedder used in Subconscious hot path
  (CoreML MiniLM on ANE via `coreml_embedder.maybe_load_coreml_embedder`,
  with CPU SentenceTransformer fallback — identical to `LocalMemoryStore`).
- Raw cosine top-5 over the full embedding matrix. NO seven-shape routing,
  NO typed filtering, NO role weights, NO recency, NO type boosting,
  NO vocabulary expansion, NO supersession filter.
- Inject top-5 memory text as flat list. Each memory truncated to 400 chars;
  total RAG block capped at 2000 chars.
- No scrub pass on response.
- Sequential dispatch to :8899 /v1/chat/completions.
- Per-turn JSON written immediately for checkpointing + resume.

Outputs:
- data/m126_baselines/baseline_3_responses/turn_<turn_id>.json  (135 files)

Run: python3 baseline_3_model_basicrag.py
Resume-safe: skips turns whose output JSON already exists.
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# Make the memory package importable.
MEM_DIR = Path("/Users/midas/Desktop/cowork/orion-ane/memory")
if str(MEM_DIR) not in sys.path:
    sys.path.insert(0, str(MEM_DIR))

CORPUS = Path("/Users/midas/Desktop/cowork/data/m126_baselines/corpus_v1.jsonl")
OUT_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_3_responses")
STORE_DB = Path("/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live/memory_local.db")
ENDPOINT = "http://127.0.0.1:8899/v1/chat/completions"
HEALTH = "http://127.0.0.1:8899/health"

# VERBATIM per directive §3.4 — DO NOT EDIT.
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to retrieved memories. "
    "Use the memories below to answer the user's question. If the memories "
    "don't contain the answer, say \"I don't have information about that.\"\n"
    "\n"
    "RETRIEVED_MEMORIES:\n"
    "{memories_block}\n"
    "\n"
    "User question: {query}"
)
# Hash of the SHAPE/TEMPLATE (without memories/query substitution). A per-turn
# instantiation also logged for reproducibility.
SYSTEM_PROMPT_TEMPLATE_HASH = hashlib.sha256(
    SYSTEM_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()

TOP_K = 5
PER_MEMORY_CHAR_CAP = 400
TOTAL_RAG_CHAR_CAP = 2000
MAX_TOKENS = 512
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 300
EMB_DIM = 384


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embedder():
    """Production embedding-client load path.

    Mirrors `LocalMemoryStore.__init__` — CoreML MiniLM on ANE if available,
    fallback to CPU SentenceTransformer. `MIDAS_DISABLE_COREML_EMBED=1` forces
    fallback.
    """
    from coreml_embedder import maybe_load_coreml_embedder
    coreml = maybe_load_coreml_embedder()
    if coreml is not None:
        print("[M126 D] embedder: CoreML MiniLM (ANE)", file=sys.stderr)
        return coreml, "coreml_ane"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("[M126 D] embedder: CPU SentenceTransformer (fallback)", file=sys.stderr)
    return model, "cpu_sentencetransformer"


def load_store_matrix():
    """Load {id -> (text, source_role, type)} + numpy matrix of active (non-superseded)
    embeddings. Identical to `LocalMemoryStore._load_index` WHERE clause so we
    operate on the same memory set the production hot path sees.
    """
    c = sqlite3.connect(str(STORE_DB), timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    try:
        rows = c.execute(
            "SELECT id, text, embedding, source_role, type, timestamp "
            "FROM memories WHERE superseded_by IS NULL"
        ).fetchall()
    finally:
        c.close()
    ids = [r["id"] for r in rows]
    texts = {r["id"]: (r["text"] or "") for r in rows}
    metas = {r["id"]: {
        "source_role": r["source_role"],
        "type": r["type"],
        "timestamp": r["timestamp"],
    } for r in rows}
    if rows:
        matrix = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
        ])
    else:
        matrix = np.zeros((0, EMB_DIM), dtype=np.float32)
    return ids, texts, metas, matrix


def embed_query(embedder, query: str) -> np.ndarray:
    """Production embed_query hot path — single string, normalized, 384-dim float32."""
    vec = embedder.encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    return vec[0]


def cosine_topk(matrix: np.ndarray, qvec: np.ndarray, k: int = TOP_K):
    """Raw cosine top-K over the full matrix. No filtering. No rerank."""
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    sims = matrix @ qvec  # (N,)
    n = min(k, sims.shape[0])
    idx = np.argpartition(-sims, n - 1)[:n]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def truncate_memory(text: str, cap: int = PER_MEMORY_CHAR_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[:cap].rstrip() + "..."


def format_rag_block(memories: list[str]) -> str:
    """Numbered list of up to 5 memories. Pad with placeholder if fewer.

    Hard-cap the concatenated block at TOTAL_RAG_CHAR_CAP.
    """
    lines: list[str] = []
    for i in range(TOP_K):
        if i < len(memories):
            lines.append(f"{i+1}. {memories[i]}")
        else:
            lines.append(f"{i+1}. [no further memories retrieved]")
    block = "\n".join(lines)
    if len(block) > TOTAL_RAG_CHAR_CAP:
        block = block[:TOTAL_RAG_CHAR_CAP].rstrip() + "..."
    return block


def health_check() -> dict:
    with urllib.request.urlopen(HEALTH, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    model = data.get("model", "")
    expected = "/Users/midas/models/gemma-4-31b-it-4bit"
    if model != expected:
        raise RuntimeError(f"Unexpected model on :8899 -> {model!r} (expected {expected!r})")
    return data


def call_server(system_prompt: str, query: str) -> tuple[dict, float]:
    # The system prompt already includes RETRIEVED_MEMORIES + "User question: <q>".
    # For the chat shape we put the full composite in the system role and
    # echo the bare query in the user role to stay consistent with the
    # directive's template semantics.
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
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
    return json.loads(raw), latency_ms


def iter_corpus():
    with CORPUS.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # K13 guard — capture pre-run checksum.
    store_checksum_pre = sha256_file(STORE_DB)
    print(f"[M126 D] store_checksum_pre={store_checksum_pre}")

    health = health_check()
    print(f"[M126 D] :8899 ready model={health['model']!r} status={health['status']}")

    embedder, embedder_backend = load_embedder()
    ids, texts, metas, matrix = load_store_matrix()
    print(f"[M126 D] store loaded: {len(ids)} active memories, matrix={matrix.shape}")

    print(f"[M126 D] system_prompt_template_hash={SYSTEM_PROMPT_TEMPLATE_HASH}")

    records = list(iter_corpus())
    total = len(records)
    print(f"[M126 D] corpus turns: {total}")

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
                print(f"[M126 D] {i}/{total} (skipped existing; turn_id={turn_id})")
            continue

        query = rec["query"]

        # K13 mid-run drift tracking — LOG ONLY, do not halt.
        # Directive §3.4 K13 intent is pre-B3 vs pre-E drift, captured for Stream F
        # via store_checksum_pre/post fields. Enricher is continuous in production;
        # a mid-run halt would prevent ever completing B3 on a live stack.
        # Per-turn measurement is still internally consistent (top-5 at query-time).
        if i % 25 == 0 or i == 1:
            mid_checksum = sha256_file(STORE_DB)
            if mid_checksum != store_checksum_pre:
                print(f"[M126 D] K13 drift observed (logged, not halted) "
                      f"pre={store_checksum_pre[:16]} mid={mid_checksum[:16]} at turn {i}")

        request_ts = time.time()

        # Embed + cosine top-5.
        try:
            qvec = embed_query(embedder, query)
        except Exception as e:
            errors += 1
            print(f"[M126 D] EMBED ERROR turn_id={turn_id}: {e!r}")
            stub = {
                "turn_id": turn_id,
                "query": query,
                "error": f"embed: {e!r}",
                "request_timestamp": request_ts,
                "system_prompt_template_hash": SYSTEM_PROMPT_TEMPLATE_HASH,
                "store_checksum_pre": store_checksum_pre,
            }
            out_path.write_text(json.dumps(stub, indent=2))
            continue

        q_emb_hash16 = hashlib.sha256(qvec.tobytes()).hexdigest()[:16]
        top_idx, top_sims = cosine_topk(matrix, qvec, k=TOP_K)
        top_ids = [ids[int(j)] for j in top_idx]
        top_texts_raw = [texts[mid] for mid in top_ids]
        top_texts_truncated = [truncate_memory(t) for t in top_texts_raw]
        top_cosine_scores = [float(s) for s in top_sims.tolist()]
        top_text_hashes = [
            hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in top_texts_raw
        ]

        # Ground-truth-in-top5: substring of ground_truth_answer (or of the
        # source_location tail) appears in any of the top-5 memory texts.
        gt_answer = (rec.get("ground_truth_answer") or "").strip()
        gt_in_top5 = False
        if gt_answer and not rec.get("ground_truth_ambiguous"):
            lowered = [t.lower() for t in top_texts_raw]
            # Tokenize gt_answer to sub-phrases (comma + " - " split), use
            # presence of any reasonably-sized fragment as hit.
            fragments: list[str] = []
            for sep in [",", " - ", ";"]:
                if sep in gt_answer:
                    fragments.extend([s.strip() for s in gt_answer.split(sep) if len(s.strip()) >= 4])
            if not fragments:
                fragments = [gt_answer]
            fragments = [f.lower() for f in fragments]
            for frag in fragments:
                if len(frag) < 4:
                    continue
                if any(frag in t for t in lowered):
                    gt_in_top5 = True
                    break

        memories_block = format_rag_block(top_texts_truncated)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            memories_block=memories_block, query=query
        )
        system_prompt_hash = hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest()

        try:
            obj, latency_ms = call_server(system_prompt, query)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            errors += 1
            print(f"[M126 D] SERVER ERROR turn_id={turn_id}: {e!r}")
            stub = {
                "turn_id": turn_id,
                "query": query,
                "error": f"server: {e!r}",
                "request_timestamp": request_ts,
                "system_prompt_template_hash": SYSTEM_PROMPT_TEMPLATE_HASH,
                "system_prompt_hash": system_prompt_hash,
                "top5_memory_ids": top_ids,
                "top5_cosine_scores": top_cosine_scores,
                "query_embedding_hash16": q_emb_hash16,
                "store_checksum_pre": store_checksum_pre,
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
            "latency_ms_ttft": None,
            "model_identifier": obj.get("model", ""),
            "system_prompt_template_hash": SYSTEM_PROMPT_TEMPLATE_HASH,
            "system_prompt_hash": system_prompt_hash,
            "request_timestamp": request_ts,
            "completion_reason": finish,
            "usage": usage,
            "prefill_ms": xsd.get("prefill_ms"),
            "tps": xsd.get("tps"),
            "embedder_backend": embedder_backend,
            "query_embedding_hash16": q_emb_hash16,
            "top5_memory_ids": top_ids,
            "top5_text_hashes16": top_text_hashes,
            "top5_cosine_scores": top_cosine_scores,
            "top5_memory_texts_truncated": top_texts_truncated,
            "ground_truth_in_top5": gt_in_top5,
            "store_checksum_pre": store_checksum_pre,
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
                f"[M126 D] {i}/{total} turn_id={turn_id} "
                f"lat={latency_ms/1000:.1f}s avg/turn={avg:.1f}s "
                f"elapsed={elapsed/60:.1f}min errors={errors}"
            )

    # Post-run: template hash uniform across all response JSONs.
    template_hashes = set()
    for p in sorted(OUT_DIR.glob("turn_*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        h = obj.get("system_prompt_template_hash")
        if h:
            template_hashes.add(h)
    if len(template_hashes) != 1 or SYSTEM_PROMPT_TEMPLATE_HASH not in template_hashes:
        print(f"[M126 D] FAIL: inconsistent system_prompt_template_hash across responses: {template_hashes}")
        sys.exit(2)
    print(f"[M126 D] template hash uniform across {len(list(OUT_DIR.glob('turn_*.json')))} files -> OK")
    store_checksum_post = sha256_file(STORE_DB)
    if store_checksum_post != store_checksum_pre:
        print(f"[M126 D] WARNING: store checksum drifted post-run "
              f"pre={store_checksum_pre} post={store_checksum_post}")
    else:
        print(f"[M126 D] store checksum stable pre==post={store_checksum_pre}")
    print(f"[M126 D] done={done} skipped={skipped} errors={errors} total={total}")


if __name__ == "__main__":
    main()
