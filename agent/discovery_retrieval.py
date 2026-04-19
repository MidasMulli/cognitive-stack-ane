"""Discovery retrieval — retrieval shape #7 of 7.

Triggered when ALL stores return empty (narrative, registry, enumeration,
recall). Detects whether the query is domain-relevant and flags the knowledge
gap. v1 is detection-only: no web search, no LLM calls.

If domain-relevant: returns a structured gap descriptor suggesting a search.
If not domain-relevant: returns None (let the model answer freely).
"""

import json
import re
from pathlib import Path

# ── Domain keyword list ──────────────────────────────────────────────────
# Terms that might not be in the entity index yet but are clearly in-domain.

_DOMAIN_KEYWORDS = {
    "ane", "gpu", "cpu", "mlx", "apple silicon", "inference", "quantization",
    "memory", "neural engine", "coreml", "metal", "spec decode",
    "speculative decoding", "throughput", "bandwidth", "latency", "dispatch",
    "kext", "iokit", "slc", "dma", "sram", "fp16", "fp32", "q4", "q8",
    "transformer", "llm", "llama", "qwen", "drafter", "verifier",
    "subconscious", "midas", "extraction", "recall", "embedding",
    "tok/s", "tokens per second", "aned", "hwx", "fusion", "attention",
    "coreml", "amx", "amcc", "macc", "noc", "die", "numa",
    "orion", "phantom", "neuron", "classifier",
    "apple", "m5", "m4", "m3", "m2", "m1",
    "deployment", "server", "daemon", "maintenance",
    "ane-compiler", "ane-dispatch", "four-path",
}

# Compile a pattern for fast matching
_DOMAIN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in sorted(_DOMAIN_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ── Entity index loader ──────────────────────────────────────────────────

_entity_index_keys: set | None = None

def _load_entity_keys() -> set:
    """Load entity names from data/entity_index.json (cached)."""
    global _entity_index_keys
    if _entity_index_keys is not None:
        return _entity_index_keys

    entity_path = Path(__file__).resolve().parent.parent.parent / "data" / "entity_index.json"
    try:
        with open(entity_path, "r") as f:
            data = json.load(f)
        _entity_index_keys = set(data.keys())
    except Exception:
        _entity_index_keys = set()
    return _entity_index_keys


def _is_domain_relevant(query: str) -> tuple[bool, list[str]]:
    """Check if query contains domain-relevant entities or keywords.

    Returns (is_relevant, list_of_matched_terms).
    """
    q_lower = query.lower()
    matched = []

    # Check entity index keys
    for entity in _load_entity_keys():
        # Entity keys are typically lowercase slugs like "ane", "gpu", "llama-8b"
        # Match as whole words, allowing hyphens
        pattern = r"\b" + re.escape(entity) + r"\b"
        if re.search(pattern, q_lower):
            matched.append(entity)

    # Check domain keywords (may overlap with entity keys, but that's fine)
    for m in _DOMAIN_PATTERN.finditer(query):
        term = m.group(0).lower()
        if term not in matched:
            matched.append(term)

    return (len(matched) > 0, matched)


def try_discovery(query: str) -> dict | None:
    """Discovery retrieval — called when all stores return empty.

    Returns a gap descriptor if the query is domain-relevant, None otherwise.
    """
    relevant, matched_terms = _is_domain_relevant(query)

    if not relevant:
        return None

    # Build a suggested search query from the original query
    # Strip common question prefixes for a cleaner search
    suggested = re.sub(
        r"^(what is|what's|what are|how do|how does|how to|can you|tell me about|explain)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    ).strip(" ?.")

    if not suggested:
        suggested = query.strip(" ?.")

    return {
        "intent": "discovery",
        "gap_detected": True,
        "domain_relevant": True,
        "query": query,
        "matched_terms": matched_terms,
        "suggested_search": suggested,
        "message": (
            f"No information found in memory. This appears to be a knowledge gap "
            f"in a domain-relevant area (matched: {', '.join(matched_terms[:5])}). "
            f"Suggested search: {suggested}"
        ),
    }


# ── v2 Discovery Pipeline (future) ──────────────────────────────────────
#
# When a web search tool is wired into midas_ui, the full discovery
# pipeline will work as follows:
#
# 1. Web search via available MCP tools or direct API, using the
#    `suggested_search` from the gap descriptor as the query.
#
# 2. Results passed through 8B adaptive extractor on ANE (the same
#    pipeline used for vault ingestion), producing typed records:
#    - quantitative facts (measurements, benchmarks, specs)
#    - conceptual facts (explanations, relationships)
#    - decisions / preferences
#
# 3. Extracted records routed to format-matched stores:
#    - Facts with numeric values → measurement registry
#    - Narrative content → narrative store with entity tags
#    - Named entities → entity index
#
# 4. All records tagged with:
#    - source: "web_discovery"
#    - retrieval_date: ISO timestamp
#    - query: original user query
#    - url: source URL
#    This tag allows downstream systems to distinguish discovered
#    knowledge from session-derived knowledge.
#
# 5. The absence guard fires only if discovery ALSO returns empty
#    (no relevant web results found, or extraction yields zero records).
#
# 6. Rate limiting: max 1 web discovery per 60 seconds to avoid
#    excessive API calls during rapid-fire questioning.
