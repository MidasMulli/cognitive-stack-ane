"""Answer scrub layer — Phase 4 (Reflection) of the cognitive pipeline.

Main 46: Post-generation verification before response reaches the user.
Three tiers, all CPU heuristic, all <50ms:

Tier 0: Fabricated tool claim — response claims a tool ran ("the search
         returned", "browse_search found") but the tool was never
         dispatched. Catches T32/T46 class failures. (Main 49)

Tier 1: Grounding check — every number+unit in the response must appear
         in the grounding corpus (recalled memories + registry + briefing).
         Catches fabricated numbers.

Tier 2: Binding check — for numbers that ARE in the corpus, verify the
         entity context matches the measurement registry's entity for
         that value. Catches conflation (real number, wrong question).

If flags are found, strip the flagged sentences. If >50% stripped,
replace with a hedged response. If 100% stripped, return None (caller
falls back to absence-style response).

Copyright 2026 Nick Lo. MIT License.
"""

import re
import json
import time
from pathlib import Path

# ── Feature flags ────────────────────────────────────────────────────
#
# Tier 1 (grounding check) is disabled by default as of Main 55.
# Gemma 4 (swapped in Main 52) produces materially better organic
# answer quality than Qwen 27B, and Tier 1's strict number+unit
# grounding has been producing false positives on legitimate answers
# where the model paraphrases a value that IS present in the corpus
# but in a different unit normalization. Tier 0 (fabricated tool
# claims) and Tier 2 (entity binding) remain active — they guard
# against the two high-impact failure modes and have not shown false
# positives under Gemma 4.
TIER1_ENABLED = False

# ── Number+unit extraction (same regex as flag_anomalies.py) ─────────

_NUMERIC_UNITS = (
    r"GB/s|tok/s|MB|GB|TB|ms|%|µs|microseconds|seconds|tokens?"
    r"|TFLOPS|FLOPS|fps|hz|HZ|kHz|MHz|GHz"
)
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:" + _NUMERIC_UNITS + r"))",
    re.IGNORECASE,
)

# Simple sentence splitter: split on period/exclamation/question
# followed by a space and uppercase letter, or newline.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\-\*])|(?:\n)+')


def _norm(s):
    """Normalize a number+unit pair for comparison."""
    return re.sub(r"\s+", "", s).lower()


def _split_sentences(text):
    """Split text into sentences, preserving list items."""
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _find_sentence_for_pair(text, pair):
    """Find the sentence containing a number+unit pair."""
    sentences = _split_sentences(text)
    pair_lower = pair.lower()
    for s in sentences:
        if pair_lower in s.lower() or _norm(pair) in _norm(s):
            return s
    return None


# ── Registry loader ──────────────────────────────────────────────────

_REGISTRY = None

def _load_registry():
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    try:
        reg_path = (Path(__file__).resolve().parent.parent.parent
                    / "data" / "measurement_registry.json")
        with open(reg_path) as f:
            _REGISTRY = json.load(f)
    except Exception:
        _REGISTRY = {}
    return _REGISTRY


def _registry_lookup_by_value(value_str, unit_str):
    """Find a registry entry matching a value+unit.

    Returns the entry dict or None.
    """
    reg = _load_registry()
    v_norm = value_str.strip()
    u_norm = unit_str.strip().lower()
    for key, entry in reg.items():
        if str(entry.get("value", "")).strip() == v_norm:
            entry_unit = str(entry.get("unit", "")).strip().lower()
            if entry_unit == u_norm or u_norm in entry_unit or entry_unit in u_norm:
                return entry
    return None


# ── Tier 0: Fabricated tool claims ──────────────────────────────────

# Phrases that imply a tool was used. Keyed by the tool name they imply.
_TOOL_CLAIM_PATTERNS = {
    "browse_search": [
        "the search returned",
        "search returned",
        "browse_search found",
        "browse_search returned",
        "web search returned",
        "the search found",
        "search results show",
        "according to the search",
    ],
    "browse_x_feed": [
        "the feed shows",
        "feed returned",
        "x feed returned",
        "twitter feed shows",
        "recent posts show",
    ],
    "vault_read": [
        "the vault shows",
        "vault_read returned",
        "the file contains",
    ],
}


def tier0_fab_check(response_text, tools_called=None):
    """Check if response claims tool usage that didn't happen.

    Returns list of flag dicts.
    """
    if not tools_called:
        tools_called = []
    tools_called_set = set(tools_called)

    flags = []
    response_lower = response_text.lower()

    for tool, patterns in _TOOL_CLAIM_PATTERNS.items():
        if tool in tools_called_set:
            continue  # Tool actually ran, claims are legitimate
        for pattern in patterns:
            if pattern in response_lower:
                sentence = _find_sentence_for_pair(response_text, pattern)
                if sentence:
                    flags.append({
                        "type": "FAB",
                        "claim": pattern,
                        "implied_tool": tool,
                        "sentence": sentence,
                    })
                break  # One flag per tool is enough

    return flags


# ── Tier 1: Grounding check ─────────────────────────────────────────

def tier1_grounding_check(response_text, grounding_corpus, user_query=""):
    """Check every number+unit in the response against the grounding corpus.

    Returns list of flag dicts.
    """
    response_pairs = _NUM_UNIT_RE.findall(response_text)
    if not response_pairs:
        return []

    corpus_pairs = set(_norm(p) for p in _NUM_UNIT_RE.findall(grounding_corpus))
    # Numbers the user introduced are grounded by definition
    if user_query:
        corpus_pairs |= set(_norm(p) for p in _NUM_UNIT_RE.findall(user_query))

    flags = []
    for pair in response_pairs:
        if _norm(pair) not in corpus_pairs:
            sentence = _find_sentence_for_pair(response_text, pair)
            flags.append({
                "type": "NOGROUND",
                "claim": pair,
                "sentence": sentence or "",
            })
    return flags


# ── Tier 2: Binding check ───────────────────────────────────────────

def tier2_binding_check(response_text, grounding_corpus, user_query=""):
    """For grounded numbers, verify entity binding against registry.

    A number is MISATTRIBUTED if it exists in the grounding corpus AND
    the registry knows its canonical entity, but the sentence context
    doesn't mention that entity or any of its aliases.

    Returns list of flag dicts.
    """
    response_pairs = _NUM_UNIT_RE.findall(response_text)
    if not response_pairs:
        return []

    corpus_pairs = set(_norm(p) for p in _NUM_UNIT_RE.findall(grounding_corpus))
    if user_query:
        corpus_pairs |= set(_norm(p) for p in _NUM_UNIT_RE.findall(user_query))

    flags = []
    seen = set()
    for pair in response_pairs:
        normed = _norm(pair)
        if normed in seen:
            continue
        seen.add(normed)

        if normed not in corpus_pairs:
            continue  # Tier 1 handles ungrounded pairs

        # Extract value and unit from the pair
        m = re.match(r"(\d+(?:\.\d+)?)\s*(.*)", pair)
        if not m:
            continue
        value_str, unit_str = m.group(1), m.group(2).strip()

        entry = _registry_lookup_by_value(value_str, unit_str)
        if not entry:
            continue  # Not in registry, can't check binding

        # Check if the sentence context mentions the registry entity
        sentence = _find_sentence_for_pair(response_text, pair)
        if not sentence:
            continue

        sentence_lower = sentence.lower()
        entity = entry.get("entity", "")
        aliases = entry.get("aliases", [])
        all_names = [entity] + aliases
        all_names = [n.lower() for n in all_names if n]

        if not any(name in sentence_lower for name in all_names):
            flags.append({
                "type": "MISATTRIBUTED",
                "claim": pair,
                "expected_entity": entity,
                "registry_key": next(
                    (k for k, v in _load_registry().items() if v is entry),
                    ""),
                "sentence": sentence,
            })

    return flags


# ── Strip flagged sentences ──────────────────────────────────────────

def strip_flagged_sentences(response_text, flags):
    """Remove sentences containing flagged claims.

    Returns cleaned response text, or None if everything was stripped.
    """
    if not flags:
        return response_text

    flagged_sentences = {f["sentence"] for f in flags if f.get("sentence")}
    if not flagged_sentences:
        return response_text

    sentences = _split_sentences(response_text)
    clean = []
    for s in sentences:
        if s not in flagged_sentences:
            clean.append(s)

    if not clean:
        return None
    if len(clean) < len(sentences) * 0.5:
        verified = " ".join(clean)
        return f"Here's what I can verify from our research: {verified}"
    return " ".join(clean)


# ── Main entry point ─────────────────────────────────────────────────

def tier3_phrase_repetition_check(response_text):
    """Detect repeated phrases/sentences in the response.

    Catches token-diverse phrase loops that escape the server-side
    repetition breaker (M91 Fix 3 gap). Operates on finished text,
    not token IDs.
    """
    flags = []
    sentences = _split_sentences(response_text)
    if len(sentences) < 3:
        return flags
    seen = {}
    for s in sentences:
        normed = s.strip().lower()
        if len(normed) < 15:
            continue
        if normed in seen:
            seen[normed] += 1
        else:
            seen[normed] = 1
    for s_normed, count in seen.items():
        if count >= 2:
            for s_orig in sentences:
                if s_orig.strip().lower() == s_normed:
                    flags.append({
                        "tier": 3,
                        "type": "PHRASE_REPEAT",
                        "sentence": s_orig,
                        "count": count,
                    })
                    break
    if not flags:
        words = response_text.split()
        for n in range(4, 10):
            if len(words) < n * 2:
                break
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = {}
            for ng in ngrams:
                ng_lower = ng.lower()
                ngram_counts[ng_lower] = ngram_counts.get(ng_lower, 0) + 1
            for ng_lower, count in ngram_counts.items():
                if count >= 3:
                    containing = _find_sentence_for_pair(response_text, ng_lower.split()[0])
                    flags.append({
                        "tier": 3,
                        "type": "NGRAM_REPEAT",
                        "ngram": ng_lower,
                        "n": n,
                        "count": count,
                        "sentence": containing,
                    })
            if flags:
                break
    return flags


def scrub_response(response_text, grounding_corpus,
                   user_query="", tools_called=None):
    """Run Tier 0 + Tier 1 + Tier 2 checks on a response.

    Returns dict:
        flags: list of all flags
        cleaned_response: response with flagged sentences removed (or None)
        tier0_flags: FAB flags (fabricated tool claims)
        tier1_flags: NOGROUND flags
        tier2_flags: MISATTRIBUTED flags
        latency_ms: total scrub time
        sentences_stripped: count
    """
    t0 = time.time()

    tier0 = tier0_fab_check(response_text, tools_called)
    if TIER1_ENABLED:
        tier1 = tier1_grounding_check(response_text, grounding_corpus, user_query)
    else:
        tier1 = []
    tier2 = tier2_binding_check(response_text, grounding_corpus, user_query)
    tier3 = tier3_phrase_repetition_check(response_text)
    all_flags = tier0 + tier1 + tier2 + tier3

    cleaned = strip_flagged_sentences(response_text, all_flags)
    sentences_stripped = 0
    if all_flags:
        flagged_sentences = {f["sentence"] for f in all_flags if f.get("sentence")}
        sentences_stripped = len(flagged_sentences)

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "flags": all_flags,
        "tier0_flags": tier0,
        "tier1_flags": tier1,
        "tier2_flags": tier2,
        "tier3_flags": tier3,
        "cleaned_response": cleaned,
        "total_flags": len(all_flags),
        "sentences_stripped": sentences_stripped,
        "original_response_chars": len(response_text),
        "cleaned_response_chars": len(cleaned) if cleaned else 0,
        "latency_ms": latency_ms,
    }
