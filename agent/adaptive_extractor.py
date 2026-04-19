"""Main 35 +1 Task 3 — AdaptiveExtractor.

Flips the extraction architecture from "two competing extractors merged"
to "8B perception primary + CPU regex supplement". The 8B is the meaning
layer (catches anything in any domain). The CPU is the precision layer
(annotates with structured values: numbers, units, file refs).

Old flow (Main 34 era):
    text -> CPU FactExtractor      -> facts[]
    text -> 8B (technical prompt)  -> facts[]
    merge(cpu, 8b) -> dedupe       -> store

Problem (per Main 34 M5 finding): the 8B's outputs were redundant with
CPU regex on technical content, and the 8B was actively harmful on
conversational content (LoCoMo -3.5pp from adding 8B to CPU). The
old prompt was domain-tuned to compete with regex.

New flow (Main 35 +1):
    text -> 8B perception (new prompt) -> perceptions[]   # primary
    text -> CPU fast-path              -> structured[]    # supplement
    for s in structured:
        if matching perception exists:
            enrich perception with s.value, s.unit, s.pattern_name
        else:
            convert s to a perception record and append
    return perceptions

Records carry confidence + speaker + topic metadata regardless of
which path produced them. The CPU layer's role becomes "annotate
quantitative anchor points the model glossed over" rather than
"compete with the model on extraction count".

Both pipelines remain importable so A/B testing is trivial.
"""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── Endpoints (override at construction time for testing) ──
ANE_8B_URL = "http://127.0.0.1:8891/analyze"
QWEN_72B_URL = "http://127.0.0.1:8899/v1/chat/completions"

# ── Make sibling modules importable ──
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "memory"))


@dataclass
class Perception:
    """Unified record produced by both 8B perception and CPU fast-path."""
    type: str                  # fact | speculation | correction | decision | preference | plan | relationship | context
    content: str
    confidence: str = "implied"  # certain | likely | speculative | implied
    speaker: str = "unknown"     # human | assistant | system
    topic: str = "general"
    structured_value: Optional[str] = None    # set by CPU fast-path enrichment
    structured_unit: Optional[str] = None
    structured_source: Optional[str] = None   # which CPU pattern matched
    source: str = "perception_8b"             # perception_8b | cpu_fastpath | enriched
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── 8B perception layer (primary) ──
def call_8b_perception(text: str, max_tokens: int = 600,
                       timeout: int = 360) -> list[Perception]:
    """Run the new perception prompt via the 8B ANE :8891 server."""
    from perception_prompt import PERCEPTION_PROMPT, parse_extractions
    prompt = PERCEPTION_PROMPT.format(text=text)
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        ANE_8B_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    raw_text = resp.get("result", "")
    records = parse_extractions(raw_text)
    out = []
    for r in records:
        out.append(Perception(
            type=r.get("type", "fact"),
            content=r.get("content", ""),
            confidence=r.get("confidence", "implied"),
            speaker=r.get("speaker", "unknown"),
            topic=r.get("topic", "general"),
            source="perception_8b",
            raw=r,
        ))
    return out


def call_72b_perception(text: str, max_tokens: int = 900,
                        timeout: int = 900) -> list[Perception]:
    """Run the new perception prompt via the 72B :8899 server.

    Slower than 8B per call, but the directive's 1C extraction uses
    72B for higher-quality records on representative sessions.
    """
    from perception_prompt import PERCEPTION_PROMPT, parse_extractions
    prompt = PERCEPTION_PROMPT.format(text=text)
    body = json.dumps({
        "model": "qwen2.5-72b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        QWEN_72B_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    raw_text = resp["choices"][0]["message"]["content"]
    records = parse_extractions(raw_text)
    out = []
    for r in records:
        out.append(Perception(
            type=r.get("type", "fact"),
            content=r.get("content", ""),
            confidence=r.get("confidence", "implied"),
            speaker=r.get("speaker", "unknown"),
            topic=r.get("topic", "general"),
            source="perception_72b",
            raw=r,
        ))
    return out


# ── CPU fast-path (supplement) ──
def call_cpu_fastpath(text: str, role: str = "user") -> list[dict]:
    """Wrap orion-ane/memory/daemon.py:FactExtractor in a list-of-dicts API."""
    from daemon import FactExtractor
    fe = FactExtractor()
    return fe.extract(text, role=role)


def cpu_fact_to_perception(fact: dict) -> Perception:
    """Convert a CPU FactExtractor dict into a Perception record."""
    text = fact.get("text", "")
    quantities = fact.get("quantities") or []
    entities = fact.get("entities") or []
    structured_value = ", ".join(quantities) if quantities else None
    return Perception(
        type=fact.get("type", "fact") or "fact",
        content=text,
        confidence="certain",  # regex patterns are explicit; we treat as certain
        speaker=fact.get("source_role", "unknown") or "unknown",
        topic=", ".join(entities[:3]) if entities else "general",
        structured_value=structured_value,
        structured_source="cpu_regex",
        source="cpu_fastpath",
        raw=fact,
    )


# ── Merge logic ──
def _normalize(s: str) -> set[str]:
    """Normalize text for fuzzy matching."""
    return set(w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 3)


def _find_matching_perception(structured: dict, perceptions: list[Perception],
                               min_overlap: int = 3) -> Optional[Perception]:
    """Find the perception that most closely covers a CPU-extracted fact."""
    structured_text = structured.get("text", "")
    s_words = _normalize(structured_text)
    if not s_words:
        return None
    best, best_score = None, 0
    for p in perceptions:
        p_words = _normalize(p.content)
        overlap = len(s_words & p_words)
        if overlap > best_score and overlap >= min_overlap:
            best, best_score = p, overlap
    return best


def merge_perceptions_with_structured(
    perceptions: list[Perception],
    structured: list[dict],
) -> list[Perception]:
    """The core merge: 8B is primary, CPU enriches or appends."""
    out = list(perceptions)
    for s in structured:
        match = _find_matching_perception(s, out)
        if match:
            quantities = s.get("quantities") or []
            entities = s.get("entities") or []
            if quantities and not match.structured_value:
                match.structured_value = ", ".join(quantities)
                match.structured_source = "cpu_regex_enriched"
            if entities and match.topic == "general":
                match.topic = ", ".join(entities[:3])
            match.source = "perception_8b+cpu_enriched" if match.source == "perception_8b" else match.source
        else:
            out.append(cpu_fact_to_perception(s))
    return out


# ── Top-level extractor class ──
class AdaptiveExtractor:
    """8B perception primary + CPU fast-path supplement.

    Use `model="8b"` (default, fast) or `model="72b"` (slower, higher quality)
    to pick the perception backend.
    """

    def __init__(self, model: str = "8b",
                 perception_max_tokens: int = 600,
                 perception_timeout: int = 360):
        self.model = model
        self.perception_max_tokens = perception_max_tokens
        self.perception_timeout = perception_timeout

    def extract(self, text: str, role: str = "user") -> list[Perception]:
        # 1. 8B (or 72B) perception
        if self.model == "72b":
            perceptions = call_72b_perception(
                text, max_tokens=self.perception_max_tokens,
                timeout=self.perception_timeout)
        else:
            perceptions = call_8b_perception(
                text, max_tokens=self.perception_max_tokens,
                timeout=self.perception_timeout)

        # 2. CPU fast-path
        try:
            structured = call_cpu_fastpath(text, role=role)
        except Exception as e:
            structured = []

        # 3. Merge (8B primary, CPU enriches or appends)
        return merge_perceptions_with_structured(perceptions, structured)

    def extract_dicts(self, text: str, role: str = "user") -> list[dict]:
        return [p.to_dict() for p in self.extract(text, role)]


# ── A/B helper for the kill test ──
def ab_compare_old_vs_new(text: str) -> dict:
    """Run the old extraction pipeline AND the new adaptive pipeline on
    the same text. Returns counts and the per-record output of each.

    Old pipeline = CPU FactExtractor only (the technical-extractor 8B
    has been retired in favor of the perception prompt; the OLD
    technical 8B prompt is preserved in tools/run_2b_ab.py for
    historical comparison).
    """
    from daemon import FactExtractor
    old_facts = FactExtractor().extract(text, role="user")
    new = AdaptiveExtractor(model="8b").extract(text)
    return {
        "old_n": len(old_facts),
        "new_n": len(new),
        "new_by_type": _count_by(new, "type"),
        "new_by_source": _count_by(new, "source"),
        "new_by_confidence": _count_by(new, "confidence"),
        "old_facts_sample": [f.get("text", "")[:120] for f in old_facts[:8]],
        "new_records_sample": [
            {"type": p.type, "content": p.content[:120], "source": p.source}
            for p in new[:8]
        ],
    }


def _count_by(perceptions: list[Perception], attr: str) -> dict:
    out: dict = {}
    for p in perceptions:
        k = getattr(p, attr) or "unknown"
        out[k] = out.get(k, 0) + 1
    return out


if __name__ == "__main__":
    # Smoke test
    sample = (
        "[USER] We need to test SLC partition writes on M5 Pro at 21 ways. "
        "I think the MACC0 might be dead but I'm not sure.\n\n"
        "[ASSISTANT] The SLC is 21-way confirmed. MACC0 was originally "
        "thought dead in Main 30 but Main 33 revived it via Phase 0 1C with "
        "fresh streaming alloc."
    )
    print("=== ab_compare_old_vs_new smoke test ===")
    out = ab_compare_old_vs_new(sample)
    print(json.dumps(out, indent=2))
