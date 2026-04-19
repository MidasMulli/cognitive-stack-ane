"""M74 Agent 1 — Tier 3 representation-level fabrication detector (MVP).

Runs after Tier 2, async to the response stream. For each claim boundary in
the generated response, asks the 8B ANE extractor (referee) to complete the
same context-up-to-boundary, compares the 8B's continuation to the 31B's
actual continuation via MiniLM cosine similarity. Low similarity flags the
claim as a fabrication candidate.

This is the text-level MVP. The representation-level version (CCA projection
of per-token hidden states) is described in the design memo and is a stretch
goal for later M74 / M75.

API:
    detect(prompt: str, response: str, query_type: str) -> list[ClaimFlag]

A ClaimFlag is {claim_text, boundary_char_index, referee_continuation,
                semantic_similarity, flagged, reason}.
"""

from __future__ import annotations
import json
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

EIGHTB_URL = "http://127.0.0.1:8891/analyze"
FLAG_SIMILARITY_THRESHOLD = 0.35   # tune via ROC on seeded probe
MAX_CLAIMS_PER_RESPONSE = 1         # 8B ANE serializes; cap at 1 call/turn for MVP
MIN_CLAIM_CHARS = 40                # skip trivial fragments
REFEREE_MAX_TOKENS = 50             # tighter for latency
REFEREE_TIMEOUT_S = 30              # 8B ANE is slow; allow more headroom

_RE_SENTENCE_END = re.compile(r'([.!?])(\s+|$)')


@dataclass
class ClaimFlag:
    claim_text: str
    boundary_char_index: int
    referee_continuation: str
    semantic_similarity: float
    flagged: bool
    reason: str
    latency_ms: int


def _find_claim_boundaries(response: str) -> list[int]:
    """Return character indices of sentence terminators in response.
    Bounded to MAX_CLAIMS_PER_RESPONSE. When capped at 1, picks the first
    sentence boundary (where fabrications most commonly first appear as
    attribution overreach)."""
    boundaries = []
    for m in _RE_SENTENCE_END.finditer(response):
        idx = m.end(1)
        if idx >= MIN_CLAIM_CHARS:
            boundaries.append(idx)
    if len(boundaries) > MAX_CLAIMS_PER_RESPONSE:
        # Pick the FIRST claim boundary — fabrications in M73 organic
        # probe emerged as mid-response attribution, which is after the
        # first sentence. But to bound latency at 1 call, first sentence
        # is the minimum viable signal. M75 upgrade: all boundaries.
        boundaries = boundaries[:MAX_CLAIMS_PER_RESPONSE]
    return boundaries


def _query_8b_referee(prompt: str, prefix: str) -> tuple[str, int]:
    """Send (prompt + prefix) to the 8B ANE referee's /analyze endpoint.
    Returns (continuation_text, latency_ms)."""
    body = {
        "prompt": f"{prompt}\n\nContinuing: {prefix}",
        "max_tokens": REFEREE_MAX_TOKENS,
    }
    payload = json.dumps(body).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            EIGHTB_URL, data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=REFEREE_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return f"<referee_error: {type(e).__name__}>", int((time.time() - t0) * 1000)
    text = data.get("result") or data.get("response") or data.get("text") or ""
    return str(text)[:500], int((time.time() - t0) * 1000)


_embed_fn = None


def _embed(text: str):
    """Embed a short text via MiniLM (shared with LocalMemoryStore).
    Lazy-loaded once per process."""
    global _embed_fn
    if _embed_fn is None:
        import sys
        from pathlib import Path
        # Agent dir is the same place this module lives
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from memory_bridge import MemoryBridge
        mb = MemoryBridge.get_shared() if hasattr(MemoryBridge, "get_shared") else None
        if mb and hasattr(mb, "_embedder") and mb._embedder is not None:
            _embed_fn = mb._embedder.embed
        else:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            _embed_fn = lambda t: _model.encode(t, show_progress_bar=False)
    import numpy as np
    v = _embed_fn(text)
    v = np.asarray(v).flatten()
    n = (v @ v) ** 0.5
    return v / (n + 1e-9)


def _cosine(a, b) -> float:
    return float(a @ b)


def detect(prompt: str, response: str, query_type: Optional[str] = None,
           threshold: float = FLAG_SIMILARITY_THRESHOLD) -> list[dict]:
    """Run Tier 3 detection. Returns list of ClaimFlag dicts (serializable)."""
    if not response or len(response) < MIN_CLAIM_CHARS:
        return []
    boundaries = _find_claim_boundaries(response)
    if not boundaries:
        return []

    flags = []
    for b_idx in boundaries:
        prefix = response[:b_idx].rstrip()
        # The "claim" is the text between the previous boundary and this one.
        # Find the start of this claim: the previous terminator or 0.
        prev = 0
        for p in boundaries:
            if p < b_idx:
                prev = p
        claim_text = response[prev:b_idx].strip()
        # Referee
        continuation, lat_ms = _query_8b_referee(prompt, prefix)
        if continuation.startswith("<referee_error"):
            flag = ClaimFlag(
                claim_text=claim_text, boundary_char_index=b_idx,
                referee_continuation=continuation,
                semantic_similarity=-1.0, flagged=False,
                reason="referee_unavailable", latency_ms=lat_ms)
            flags.append(asdict(flag))
            continue
        # Extract the 31B's actual continuation up to similar length
        actual_continuation = response[b_idx:b_idx + len(continuation) + 100].strip()
        if not actual_continuation or not continuation.strip():
            # No continuation to compare (end of response)
            continue
        try:
            emb_a = _embed(actual_continuation[:500])
            emb_b = _embed(continuation.strip()[:500])
            sim = _cosine(emb_a, emb_b)
        except Exception as e:
            flag = ClaimFlag(
                claim_text=claim_text, boundary_char_index=b_idx,
                referee_continuation=continuation,
                semantic_similarity=-1.0, flagged=False,
                reason=f"embed_error:{type(e).__name__}", latency_ms=lat_ms)
            flags.append(asdict(flag))
            continue
        flagged = sim < threshold
        flag = ClaimFlag(
            claim_text=claim_text, boundary_char_index=b_idx,
            referee_continuation=continuation,
            semantic_similarity=round(float(sim), 4),
            flagged=flagged,
            reason="low_semantic_similarity" if flagged else "ok",
            latency_ms=lat_ms)
        flags.append(asdict(flag))
    return flags


if __name__ == "__main__":
    # Self-test
    prompt = "What happened in Main 72?"
    clean_response = (
        "Main 72's bridge-LoRA training crashed due to memory pressure. "
        "Step 70 of epoch 0 was the last checkpoint. "
        "The training manifest and first 71 steps of log were preserved."
    )
    seeded_response = (
        "Main 72's bridge-LoRA training crashed due to memory pressure. "
        "The full root cause is documented in ane-reverse/main-355-py-spy-resolution-v2.md. "
        "The resolution was via sudo py-spy attach on the wedged worker."
    )
    print("=== clean response ===")
    for f in detect(prompt, clean_response):
        print(f)
    print("\n=== seeded response (fabricated file path) ===")
    for f in detect(prompt, seeded_response):
        print(f)
