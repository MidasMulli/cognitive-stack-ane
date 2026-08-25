#!/usr/bin/env python3
"""
M126 Baselines Stream C — Baseline 2: Gemma 4 31B Q4 + raw grep over vault/.

Per directive §3.3:
- Keyword extraction: tokenize on whitespace/punct, lowercase, drop stopwords, drop <4 char tokens
- Grep across vault/: grep -r -i -l <keyword>, aggregate files
- Extract snippets: grep -n -B 1 -A 3 -i <keyword> per file, cap 5 matches/file, 20 hits total
- Hard cap 4000 chars on GREP_RESULTS; truncation recorded
- System prompt verbatim from §3.3
- Sequential dispatch to :8899 (gen lock serializes)
- Per-turn JSON to data/m126_baselines/baseline_2_responses/turn_<turn_id>.json
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---- Paths (absolute, per environment) ----
CORPUS_PATH = Path("/Users/midas/Desktop/cowork/data/m126_baselines/corpus_v1.jsonl")
RESPONSES_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_2_responses")
VAULT_ROOT = Path("/Users/midas/Desktop/cowork/vault")
VERIFIER_URL = "http://127.0.0.1:8899/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8899/health"

# ---- Constants (per directive §3.3) ----
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "with", "by", "as", "and", "or", "but", "if",
    "we", "i", "you", "they", "he", "she", "it", "this", "that", "these",
    "those", "what", "when", "where", "who", "why", "how", "do", "does", "did",
    "have", "has", "had",
}
MIN_TOKEN_LEN = 4
MAX_MATCHES_PER_FILE = 5
MAX_TOTAL_HITS = 20
CONTEXT_CHAR_CAP = 4000
SNIPPET_CHAR_CAP = 200
TRACK2_PATTERNS = ("track2", "derivatives", "platform_x")

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to grep results over a knowledge "
    "vault. Use the grep results below to answer the user's question. If the "
    "grep results don't contain the answer, say \"I don't have information "
    "about that.\"\n\n"
    "GREP_RESULTS:\n"
    "{grep_block}\n\n"
    "User question: {query}"
)

# Hash the static shell of the system prompt (everything except the dynamic
# grep_block + query) so we have a constant, verifiable prompt identity.
SYSTEM_PROMPT_SHELL = (
    "You are a helpful assistant with access to grep results over a knowledge "
    "vault. Use the grep results below to answer the user's question. If the "
    "grep results don't contain the answer, say \"I don't have information "
    "about that.\"\n\n"
    "GREP_RESULTS:\n"
    "{grep_block}\n\n"
    "User question: {query}"
)
SYSTEM_PROMPT_HASH = hashlib.sha256(
    SYSTEM_PROMPT_SHELL.encode("utf-8")
).hexdigest()


def extract_keywords(query: str) -> list[str]:
    """Tokenize on whitespace+punct, lowercase, drop stopwords, drop <4 chars.

    Preserves order, dedupes while preserving first occurrence.
    """
    # Split on any non-alphanumeric (keeps underscores as part of tokens too?
    # directive says "whitespace + punct" — treat '_' as punct for safety).
    raw = re.findall(r"[A-Za-z0-9]+", query.lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _is_track2_path(p: str) -> bool:
    low = p.lower()
    return any(pat in low for pat in TRACK2_PATTERNS)


def grep_files_for_keyword(keyword: str) -> list[str]:
    """Run grep -r -i -l <keyword> under vault/. Return relative paths (to vault)."""
    try:
        proc = subprocess.run(
            ["grep", "-r", "-i", "-l", keyword, str(VAULT_ROOT)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        # 0 = hits, 1 = no hits, >1 = error
        return []
    files: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_track2_path(line):
            # K10 safety — should never happen since tree has no Track 2
            sys.stderr.write(f"K10 WARN: Track 2 path leaked: {line}\n")
            continue
        files.append(line)
    return files


def grep_snippets(path: str, keyword: str) -> list[tuple[int, str]]:
    """Run grep -n -B 1 -A 3 -i <keyword> <path>. Return (line_number, snippet)."""
    try:
        proc = subprocess.run(
            ["grep", "-n", "-B", "1", "-A", "3", "-i", keyword, path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        return []

    # grep -B/-A output separates match groups with '--' lines. Each group
    # contains lines of form "<line_no>:<text>" or "<line_no>-<text>" (for
    # context before/after). The match line uses ':' as separator, context
    # lines use '-'. We parse groups and take the matching line number +
    # the full group as the snippet.
    snippets: list[tuple[int, str]] = []
    groups: list[list[str]] = [[]]
    for line in proc.stdout.splitlines():
        if line == "--":
            groups.append([])
        else:
            groups[-1].append(line)

    for group in groups:
        if not group:
            continue
        # Find the first ':' separator line (the match) to pull line number
        match_lineno = None
        text_lines: list[str] = []
        for gl in group:
            # Try match separator ':' first
            m = re.match(r"^(\d+):(.*)$", gl)
            if m and match_lineno is None:
                match_lineno = int(m.group(1))
                text_lines.append(m.group(2))
                continue
            m2 = re.match(r"^(\d+)[-:](.*)$", gl)
            if m2:
                text_lines.append(m2.group(2))
            else:
                text_lines.append(gl)
        if match_lineno is None:
            continue
        snippet = " ".join(tl.strip() for tl in text_lines if tl.strip())
        if len(snippet) > SNIPPET_CHAR_CAP:
            snippet = snippet[:SNIPPET_CHAR_CAP]
        snippets.append((match_lineno, snippet))
        if len(snippets) >= MAX_MATCHES_PER_FILE:
            break
    return snippets


def build_grep_block(query: str) -> dict:
    """Run the full grep pipeline for a query. Returns dict with block + metadata."""
    keywords = extract_keywords(query)

    # Aggregate unique files hit by any keyword, with keyword assoc for snippeting
    file_to_keyword: dict[str, str] = {}
    for kw in keywords:
        hits = grep_files_for_keyword(kw)
        for f in hits:
            # Keep first keyword that hit the file (order preserved)
            if f not in file_to_keyword:
                file_to_keyword[f] = kw

    file_list = list(file_to_keyword.keys())

    # Extract snippets, cap at MAX_TOTAL_HITS across all files
    block_parts: list[str] = []
    total_hits = 0
    for f in file_list:
        if total_hits >= MAX_TOTAL_HITS:
            break
        kw = file_to_keyword[f]
        snippets = grep_snippets(f, kw)
        rel = os.path.relpath(f, str(VAULT_ROOT.parent))  # relative to cowork/
        for (lineno, snippet) in snippets:
            if total_hits >= MAX_TOTAL_HITS:
                break
            block_parts.append(f"[{rel}:{lineno}]\n{snippet}\n---")
            total_hits += 1

    full_block = "\n".join(block_parts) if block_parts else "(no grep hits)"
    truncated = False
    truncation_chars = 0
    if len(full_block) > CONTEXT_CHAR_CAP:
        truncation_chars = len(full_block) - CONTEXT_CHAR_CAP
        full_block = full_block[:CONTEXT_CHAR_CAP]
        truncated = True

    return {
        "keywords": keywords,
        "file_list": file_list,
        "hit_count": total_hits,
        "block": full_block,
        "context_chars": len(full_block),
        "truncated": truncated,
        "truncation_chars": truncation_chars,
    }


def call_verifier(system_prompt: str, user_query: str, max_tokens: int = 512) -> tuple[str, float]:
    """POST to :8899 /v1/chat/completions. Returns (response_text, latency_ms)."""
    payload = {
        "model": "gemma4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        VERIFIER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.time() - t0) * 1000.0
    text = data["choices"][0]["message"]["content"]
    return text, latency_ms


def process_turn(record: dict) -> dict:
    turn_id = record["turn_id"]
    query = record["query"]

    # Build grep context
    grep = build_grep_block(query)

    # Build system prompt (combined — we send the whole thing as "system" per §3.3)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        grep_block=grep["block"],
        query=query,
    )

    # User turn is the same query (it's already embedded in system, but the
    # OpenAI API shape requires a user message — we repeat the query verbatim)
    response_text, latency_ms = call_verifier(system_prompt, query)

    return {
        "turn_id": turn_id,
        "query": query,
        "grep_keywords": grep["keywords"],
        "grep_file_list": grep["file_list"],
        "grep_hit_count": grep["hit_count"],
        "context_chars": grep["context_chars"],
        "truncated": grep["truncated"],
        "truncation_chars": grep["truncation_chars"],
        "response": response_text,
        "latency_ms": latency_ms,
        "system_prompt_hash": SYSTEM_PROMPT_HASH,
        "ground_truth_source_location": record.get("ground_truth_source_location"),
        "query_class": record.get("query_class"),
        "ground_truth_ambiguous": record.get("ground_truth_ambiguous", False),
    }


def main() -> int:
    # Verifier health check
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        model = health.get("model", "")
        if "gemma" not in model.lower():
            sys.stderr.write(f"FATAL: verifier is not Gemma — model={model}\n")
            return 2
        sys.stdout.write(f"[health] model={model} engine_loaded={health.get('engine_loaded')}\n")
    except Exception as e:
        sys.stderr.write(f"FATAL: /health probe failed: {e}\n")
        return 2

    # Track 2 safety
    for pat in TRACK2_PATTERNS:
        if any(pat in str(p).lower() for p in VAULT_ROOT.rglob(f"*{pat}*")):
            sys.stderr.write(f"K10 HALT: Track 2 path detected under vault/: pattern={pat}\n")
            return 10

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    records: list[dict] = []
    with open(CORPUS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    total = len(records)
    sys.stdout.write(f"[corpus] loaded {total} records\n")

    # Allow resuming if partial run exists
    start_idx = 0
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    errors = 0
    for i, record in enumerate(records):
        if i < start_idx:
            continue
        turn_id = record["turn_id"]
        out_path = RESPONSES_DIR / f"turn_{turn_id}.json"
        if out_path.exists():
            sys.stdout.write(f"[{i+1}/{total}] {turn_id} SKIP (exists)\n")
            continue
        try:
            t0 = time.time()
            result = process_turn(record)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            dt = time.time() - t0
            sys.stdout.write(
                f"[{i+1}/{total}] {turn_id} "
                f"kw={len(result['grep_keywords'])} "
                f"files={len(result['grep_file_list'])} "
                f"hits={result['grep_hit_count']} "
                f"ctx={result['context_chars']} "
                f"trunc={result['truncated']} "
                f"lat={result['latency_ms']:.0f}ms "
                f"total={dt:.1f}s\n"
            )
            sys.stdout.flush()
        except Exception as e:
            errors += 1
            sys.stderr.write(f"[{i+1}/{total}] {turn_id} ERROR: {e}\n")
            sys.stderr.flush()
            # Continue — gen lock may have dropped; retry once
            try:
                time.sleep(2)
                result = process_turn(record)
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                sys.stdout.write(f"[{i+1}/{total}] {turn_id} RETRY OK\n")
                errors -= 1
            except Exception as e2:
                sys.stderr.write(f"[{i+1}/{total}] {turn_id} RETRY FAILED: {e2}\n")

    sys.stdout.write(f"[done] {total} turns processed, {errors} errors\n")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
