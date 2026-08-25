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

Tier 2 narrative-drift (M118 C): embedding-similarity claim-vs-grounding
         check on specific-claim sentences (paths, proper nouns, quoted
         strings). Catches narrative-level fabrication that passes Tier 1
         (no numeric markers to ground) and Tier 2 binding (no registry
         hit). Pilot evidence: M117 K4 — T24/T30/T33/T35. See
         vault/agent_reports/m118_c_scrub_narrative_drift.md.

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
    # M103: mirror of M102 narrative_retrieval fix. Some registry entries are
    # bare scalars (float/bool/str) written by pipeline tools (e.g.
    # tools/m96/m96_analyze.py) that bypass the dict schema. Skip them here
    # so a single malformed row can't abort scrub registry lookup with
    # `'float' object has no attribute 'get'`. Root cause + full catalog in
    # vault/agent_reports/m102_narrative_retrieval_fix.md.
    for key, entry in reg.items():
        if not isinstance(entry, dict):
            continue
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

# ── M122 A2 Tier 2 refinement guards ────────────────────────────────
#
# Bare-scalar and abstention guards. Anchor: M121 A smoking-gun on T82
# (correct answer stripped via `5%` → m114.pilot.new_mechanism_count
# collision with zero word-overlap) + T68 (honest abstention stripped
# via `61%` → 3b_solo.extraction_recall collision inside an explicitly
# negated sentence). Guards fire BEFORE a MISATTRIBUTED flag is emitted:
# if either fires, the strip is skipped and a skip_reason is recorded in
# the parallel `tier2_binding_skips` list so forensics can distinguish
# principled skips from overfires. See vault/agent_reports/
# m121_a_synthesis_residual_diagnosis.md §3.1 + §6 Stream M122.A2.
#
# Pattern-1 discipline: prefer under-fit. Bare-scalar guard requires
# BOTH zero word-overlap AND non-empty sentence content words (else we
# strip as normal). Abstention guard requires the abstention phrase in
# the SAME clause as the matched scalar (not just anywhere in sentence).

# Minimal English stopword set (per M122 A2 directive guidance). Keep
# narrow — don't remove content words, just obvious function words.
_M122_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "and", "or", "but", "is", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "this", "that", "these",
    "those", "it", "its", "we", "our", "you", "your", "their", "they",
    "i", "my", "me",
})

# Abstention patterns (§3.2 fix 2). These match honest ignorance or
# hedged framings where a scalar appears in negated context rather than
# asserted. Order matters only for latency; any match gates the strip.
_M122_ABSTENTION_PATTERNS = (
    r"\bi don'?t have\b",
    r"\bi don'?t know\b",
    r"\bno measurement of\b",
    r"\bhaven'?t measured\b",
    r"\bcouldn'?t verify\b",
    r"\bnothing in memory\b",
    r"\bi'?m not sure\b",
    r"\bi lack\b",
    r"\bno information (?:about|on)\b",
)
_M122_ABSTENTION_RE = re.compile(
    "|".join(_M122_ABSTENTION_PATTERNS), re.IGNORECASE)


def _m122_tokenize(text):
    """Lowercase word-token extraction. Treats underscores/dots/hyphens
    as separators so registry keys like `m114.pilot.new_mechanism_count`
    split into {m114, pilot, new, mechanism, count}."""
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _m122_content_tokens(text):
    """Tokenize + strip stopwords + drop sub-2-char fragments."""
    toks = _m122_tokenize(text)
    return [t for t in toks if t not in _M122_STOPWORDS and len(t) >= 2]


def _m122_context_window(sentence, claim_pair, window_words=3):
    """Extract ±`window_words` words around the matched scalar span.
    Used by bare-scalar guard to focus overlap on local topic context
    rather than full sentence (reduces false skips on long sentences)."""
    if not sentence or not claim_pair:
        return sentence or ""
    lower_sent = sentence.lower()
    lower_claim = claim_pair.lower().strip()
    # Locate the claim span.
    idx = lower_sent.find(lower_claim)
    if idx < 0:
        # Fall back to whole-sentence content.
        return sentence
    # Split preceding and following text into word tokens, keep ±window.
    pre = sentence[:idx]
    post = sentence[idx + len(claim_pair):]
    pre_toks = re.findall(r"\S+", pre)[-window_words:]
    post_toks = re.findall(r"\S+", post)[:window_words]
    return " ".join(pre_toks + [claim_pair] + post_toks)


def _m122_clause_around_claim(sentence, claim_pair):
    """Extract the clause containing the claim. A clause is the span
    between the claim and the nearest clause boundary (comma/semicolon/
    period/dash) on each side, or sentence boundary if no punctuation.
    Used by abstention guard to enforce same-clause proximity."""
    if not sentence or not claim_pair:
        return sentence or ""
    lower_sent = sentence.lower()
    lower_claim = claim_pair.lower().strip()
    idx = lower_sent.find(lower_claim)
    if idx < 0:
        return sentence
    # Walk backward to find clause start.
    start = idx
    while start > 0 and sentence[start - 1] not in ",;:—–":
        start -= 1
    # Walk forward to find clause end.
    end = idx + len(claim_pair)
    while end < len(sentence) and sentence[end] not in ",;:—–":
        end += 1
    return sentence[start:end]


def _m123_evaluate_conflict_path_guards(sentence, claim_pair, entry):
    """M123 A1 — conflict-path guard variant.

    The conflict path (`tier2_registry_conflict_check`) pre-filters to
    sentences that mention the registry entity by alias. Running the
    binding-path bare-scalar guard here produces structural false
    positives — M120 C's canonical test ("The Neuron model runs at 7.9
    tok/s on the ANE") sits the entity name 4 words from the claim, so
    the ±3-word window misses it and the guard fires when it should not.

    For the conflict path we adapt the guard semantics:
      1. Abstention guard (unchanged): skip if an abstention phrase
         appears in the same clause as the claim.
      2. Topic-mismatch bare-scalar guard: skip if NEITHER the entity
         tokens NOR the measurement-type tokens appear in the ±5-word
         window around the claim. This is a topic-disambiguation check
         — it fires when the claim is locally about something else
         (T59: "0% acceptance rate" is topically about acceptance, not
         q4's dequant_penalty). Window widened from 3 → 5 so subjects
         4-5 words before the claim (e.g. "The Neuron model runs at
         7.9 tok/s") are still caught. Case-analysis evidence:
            T59 claim "0%" in "EAGLE-3 had a 0% acceptance rate on all
              quantized models, including Q3 and Q4" — window at ±5
              yields {eagle, acceptance, rate, all, quantized}; Q4
              registry tokens {q4, 4, bit, dequant, penalty} do not
              overlap → guard fires correctly.
            M120 C Neuron "7.9 tok/s" in "The Neuron model runs at 7.9
              tok/s on the ANE" — window at ±5 yields {neuron, model,
              runs, tok, ane}; entity tokens {neuron, classifier, 80m,
              throughput} overlap on `neuron` → guard does not fire.

    Registry metadata with empty/missing entity or measurement_type
    tokens falls through to strip-as-normal (no silent skip).

    K1 note (directive §3.1): helper duplicated rather than extracted
    because binding-path semantics require sentence-wide entity absence
    (pre-filter inverts this) and the conflict-path correct-answer set
    demands topic-disambiguation rather than pure zero-overlap. Candidate
    for consolidation in M125+ once both paths have stable test evidence.
    """
    if not sentence:
        return None

    entity = (entry.get("entity") or "").strip()
    aliases = entry.get("aliases") or []
    mtype = (entry.get("measurement_type") or "").strip()

    # Abstention guard first — preserves T68-shape honest abstentions
    # against false conflict flags on negated claims.
    clause = _m122_clause_around_claim(sentence, claim_pair)
    if _M122_ABSTENTION_RE.search(clause):
        return "abstention_pattern"

    # Topic-mismatch guard. Two disjoint token sets:
    #   entity_tokens   — from entity name + aliases
    #   mtype_tokens    — from measurement_type
    # Skip only when neither set overlaps the ±3-word window.
    entity_text = " ".join([entity] + list(aliases))
    entity_tokens = set(_m122_content_tokens(entity_text))
    mtype_tokens = set(_m122_content_tokens(mtype))
    if not entity_tokens and not mtype_tokens:
        return None  # missing metadata — conservative: don't skip

    window = _m122_context_window(sentence, claim_pair, window_words=5)
    window_tokens = set(_m122_content_tokens(window))
    if not window_tokens:
        return None  # empty context — conservative: don't skip

    entity_overlap = bool(entity_tokens & window_tokens) if entity_tokens else False
    mtype_overlap = bool(mtype_tokens & window_tokens) if mtype_tokens else False
    if not entity_overlap and not mtype_overlap:
        return "word_overlap_zero"

    return None


def _m122_evaluate_skip_guards(sentence, claim_pair, entry):
    """Return skip_reason string or None.

    skip_reason values:
        None — no guard fired; strip proceeds normally
        "word_overlap_zero" — bare-scalar guard fired
        "abstention_pattern" — abstention guard fired

    Guard order: bare-scalar first (primary), abstention second
    (secondary). First match wins for diagnostic.
    """
    if not sentence:
        return None

    # ── Bare-scalar guard (primary) ──────────────────────────────────
    entity = (entry.get("entity") or "").strip()
    aliases = entry.get("aliases") or []
    # Registry entity content tokens — include entity name, aliases, AND
    # tokenized registry key if it's dot/underscore-segmented (e.g.
    # `m114.pilot.new_mechanism_count` → {m114, pilot, new, mechanism,
    # count}). This gives the sentence a richer pool of potential
    # overlap than just the bare `m114` entity name.
    registry_text = " ".join([entity] + list(aliases))
    # Also include measurement_type for extra overlap surface.
    mtype = entry.get("measurement_type") or ""
    if mtype:
        registry_text += " " + mtype
    registry_tokens = set(_m122_content_tokens(registry_text))

    # Sentence context tokens — ±3 words around the matched scalar.
    window = _m122_context_window(sentence, claim_pair, window_words=3)
    sentence_tokens = set(_m122_content_tokens(window))

    # If registry has no content tokens to match against (all stopwords
    # or empty), we cannot reason about overlap — fall through to strip
    # as normal (don't let missing-metadata silently skip).
    if registry_tokens and sentence_tokens:
        if not (registry_tokens & sentence_tokens):
            return "word_overlap_zero"

    # ── Abstention-sentence guard (secondary) ────────────────────────
    # Require the abstention phrase to be in the SAME clause as the
    # matched scalar (per K5 discipline — don't match "I don't have"
    # anywhere in a multi-clause sentence).
    clause = _m122_clause_around_claim(sentence, claim_pair)
    if _M122_ABSTENTION_RE.search(clause):
        return "abstention_pattern"

    return None


def tier2_binding_check(response_text, grounding_corpus, user_query=""):
    """For grounded numbers, verify entity binding against registry.

    A number is MISATTRIBUTED if it exists in the grounding corpus AND
    the registry knows its canonical entity, but the sentence context
    doesn't mention that entity or any of its aliases.

    M122 A2: two guards added prior to flag emission:
      1. Bare-scalar guard — skip strip when registry entity has zero
         non-stopword overlap with ±3-word window around the matched
         scalar. Fixes T82 (5% → m114 collision on qwen stream ordering
         sentence).
      2. Abstention-sentence guard — skip strip when the same clause as
         the scalar contains an abstention phrase. Fixes T68 (61% →
         3b_solo collision on "I don't have information about 61%").

    Returns tuple (flags, skips):
        flags: list of MISATTRIBUTED flag dicts (original behavior)
        skips: list of {claim, sentence, skip_reason, expected_entity,
               registry_key} dicts for turns where a guard suppressed a
               strip. Couples with ζ v2.2 schema (Stream A3) via
               `scrub.tier2_binding.skip_reason`.

    Backward compatibility: callers that only unpack the first element
    still get the flag list. `scrub_response` updated to consume both.
    """
    response_pairs = _NUM_UNIT_RE.findall(response_text)
    if not response_pairs:
        return [], []

    corpus_pairs = set(_norm(p) for p in _NUM_UNIT_RE.findall(grounding_corpus))
    if user_query:
        corpus_pairs |= set(_norm(p) for p in _NUM_UNIT_RE.findall(user_query))

    flags = []
    skips = []
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
        registry_key = next(
            (k for k, v in _load_registry().items() if v is entry),
            "")

        if not any(name in sentence_lower for name in all_names):
            # M122 A2: evaluate skip guards before emitting a strip flag.
            # M123 A1: record `path="binding"` so alternate-path fires
            #          (conflict, flag) can be distinguished from this
            #          primary binding-check path in the scrub record.
            skip_reason = _m122_evaluate_skip_guards(sentence, pair, entry)
            if skip_reason is not None:
                skips.append({
                    "path": "binding",
                    "claim": pair,
                    "expected_entity": entity,
                    "registry_key": registry_key,
                    "sentence": sentence,
                    "skip_reason": skip_reason,
                    "reason": skip_reason,
                })
                continue

            flags.append({
                "type": "MISATTRIBUTED",
                "claim": pair,
                "expected_entity": entity,
                "registry_key": registry_key,
                "sentence": sentence,
            })

    return flags, skips


# ── Tier 2 registry-value conflict (M120 C) ──────────────────────────
# m120_c_tier2_registry_conflict
#
# Mechanism anchor: M119 T79. Response pre-scrub said "Yes, the 30.4
# TFLOPS measurement matches production usage." Tier 2 binding check
# stripped the sentence because no NAX alias appeared — the claim was
# true (registry `nax.throughput=30.4 TFLOPS` exact match) but the
# strip destroyed it silently.
#
# This check is ADDITIVE to tier2_binding_check. It does NOT strip.
# When the response mentions an entity that has a canonical registry
# value, and the sentence asserts a DIFFERENT numeric value (outside
# tolerance), we surface an inline clarification to the user instead
# of silently mutating the answer. CA recommendation (directive §3.3):
# user-visible flag > silent strip.
#
# Design choices:
#   - Tolerance: 5% relative (conservative; tunable if evidence warrants).
#   - Option A (chosen): inline clarification appended to the sentence —
#     "(measurement registry records different value: X <unit>; flagging
#     for audit)" — preserves the grounded sentence and surfaces the
#     disagreement for audit.
#   - Option B (fallback, not chosen): top-of-response banner. Only used
#     if inline injection fails.
#   - Flags are emitted as tier2_conflict_flagged in the scrub result so
#     callers can persist them to turn JSON for telemetry.

# Relative tolerance for registry-vs-response numeric agreement. Values
# within this band are treated as measurement noise and NOT flagged.
TIER2_CONFLICT_TOLERANCE = 0.05


def _parse_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _unit_matches(response_unit, registry_unit):
    """Loose unit match — same tolerance as _registry_lookup_by_value."""
    ru = (response_unit or "").strip().lower()
    gu = (registry_unit or "").strip().lower()
    if not ru or not gu:
        return False
    return ru == gu or ru in gu or gu in ru


def tier2_registry_conflict_check(response_text, user_query=""):
    """Per-entity registry-value conflict detector (M120 C).

    For each sentence in the response, if the sentence mentions a known
    registry entity (by name or alias) AND contains a number+unit that
    is INCOMPATIBLE with the registry's canonical value for that entity
    (same measurement_type + unit, outside TIER2_CONFLICT_TOLERANCE),
    emit a conflict flag. The caller surfaces the flag to the user as
    an inline clarification; the original sentence is preserved.

    M123 A1 — alternate-path guard extension: M122 A2's bare-scalar +
    abstention guards (shipped on `tier2_binding_check` only) now also
    gate flag emission here. Motivated by M122 C T59 where the conflict
    path emitted a `q4.dequant_penalty=31%` template inline against the
    claim `0%` in the sentence "EAGLE-3 had a 0% acceptance rate on all
    quantized models...". No overlap between q4 content tokens and the
    ±3-word window around `0%` → bare-scalar guard should have fired.
    Under M122 A2 it did not, because the conflict-path call never
    reached `_m122_evaluate_skip_guards`. M123 A1 wires the same helper
    in; guard fires here suppress the flag (and therefore the downstream
    `annotate_conflict_flags` template inline-leak). Skip reason is
    recorded to `tier2_binding_skips` with `path="conflict"` so ζ v2.2
    emission at midas_ui.py can emit it under
    `scrub.tier2_binding.skip_reason` alongside binding-path fires.

    Returns tuple (flags, skips):
        flags: list of REGISTRY_VALUE_CONFLICT dicts (unchanged schema —
               type, sentence_span, claim_value, claim_unit, registry_key,
               registry_entity, registry_value, registry_unit,
               tolerance_exceeded, rel_diff, sentence).
        skips: list of guard-suppressed entries with fields
               {path="conflict", claim, expected_entity, registry_key,
                sentence, skip_reason, reason}. Same shape as the
               binding-path skip records — downstream consumers can
               distinguish by `path`.

    Backward compatibility: callers that only unpack the first element
    still receive the flag list. `scrub_response` updated to consume
    both.
    """
    if not response_text:
        return [], []

    reg = _load_registry()
    if not reg:
        return [], []

    # Build reverse-lookup: entity alias -> list of registry entries.
    # We keyword-match by alias presence in the sentence, then compare
    # each registry value against each number+unit in the sentence.
    alias_to_entries = []  # list of (alias_lower, key, entry)
    for key, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        entity = str(entry.get("entity", "") or "").strip()
        aliases = entry.get("aliases", []) or []
        names = [entity] + list(aliases)
        for name in names:
            name_l = str(name).strip().lower()
            if not name_l:
                continue
            alias_to_entries.append((name_l, key, entry))

    if not alias_to_entries:
        return [], []

    flags = []
    skips = []
    # Split into sentences and track byte offsets for sentence_span.
    sentences = _split_sentences(response_text)
    cursor = 0
    for sent in sentences:
        # Locate this sentence's span in the original text. Start the
        # search from the current cursor to keep spans monotonic when
        # sentences repeat.
        idx = response_text.find(sent, cursor)
        if idx < 0:
            # Fall back to a global search; span accuracy is best-effort.
            idx = response_text.find(sent)
        if idx < 0:
            continue
        span = (idx, idx + len(sent))
        cursor = span[1]

        sent_lower = sent.lower()

        # Which registry entries' entities are mentioned in this sentence?
        candidate_entries = []
        seen_keys = set()
        for alias_l, key, entry in alias_to_entries:
            if key in seen_keys:
                continue
            # Use word-boundary-ish containment: alias must be a substring
            # with non-alphanumeric boundaries to avoid false hits like
            # "ane" inside "analyze".
            i = sent_lower.find(alias_l)
            while i >= 0:
                left_ok = (i == 0) or not sent_lower[i - 1].isalnum()
                right_end = i + len(alias_l)
                right_ok = (right_end >= len(sent_lower)
                            or not sent_lower[right_end].isalnum())
                if left_ok and right_ok:
                    candidate_entries.append((key, entry))
                    seen_keys.add(key)
                    break
                i = sent_lower.find(alias_l, i + 1)

        if not candidate_entries:
            continue

        # Extract all number+unit pairs in the sentence.
        pairs_in_sent = _NUM_UNIT_RE.findall(sent)
        if not pairs_in_sent:
            continue

        # Group candidate registry entries by entity. Within an entity,
        # a claim is considered GROUNDED if ANY of that entity's
        # registry entries (e.g. `nax.throughput` aggregate vs
        # `nax.throughput_per_core`) matches within tolerance on the
        # same unit. Only flag when NO entry for that entity+unit
        # matches — this prevents spurious divergence reports from
        # alternative-measurement-axis rows on the same entity.
        entity_to_candidates = {}
        for key, entry in candidate_entries:
            ent = str(entry.get("entity", "") or "").lower()
            entity_to_candidates.setdefault(ent, []).append((key, entry))

        for pair in pairs_in_sent:
            m = re.match(r"(\d+(?:\.\d+)?)\s*(.*)", pair)
            if not m:
                continue
            claim_val = _parse_float(m.group(1))
            claim_unit = m.group(2).strip()
            if claim_val is None:
                continue

            for ent, entries_for_entity in entity_to_candidates.items():
                unit_compat = []
                for key, entry in entries_for_entity:
                    reg_unit = str(entry.get("unit", "") or "")
                    if not _unit_matches(claim_unit, reg_unit):
                        continue
                    reg_val = _parse_float(entry.get("value"))
                    if reg_val is None:
                        continue
                    denom = max(abs(claim_val), abs(reg_val), 1e-9)
                    rel_diff = abs(claim_val - reg_val) / denom
                    unit_compat.append((key, entry, reg_val, reg_unit,
                                        rel_diff))

                if not unit_compat:
                    continue  # no registry entry for this entity+unit

                # Any within-tolerance match on this entity → grounded,
                # no flag for this (entity, claim) pair.
                if any(rd <= TIER2_CONFLICT_TOLERANCE
                       for _, _, _, _, rd in unit_compat):
                    continue

                # None within tolerance — surface the closest entry as
                # the authoritative reference in the flag, so the audit
                # clarification cites the strongest registry signal.
                unit_compat.sort(key=lambda t: t[4])
                key, entry, reg_val, reg_unit, rel_diff = unit_compat[0]

                # M123 A1: alternate-path guard. The conflict path
                # pre-filters to sentences that already mention the
                # entity by alias, so the binding-path bare-scalar
                # guard (which assumes entity absence) can't be reused
                # verbatim. _m123_evaluate_conflict_path_guards keeps
                # the abstention guard identical and swaps in a
                # topic-disambiguation variant of the bare-scalar guard:
                # fire only when NEITHER the entity tokens NOR the
                # measurement-type tokens appear in the ±3-word window
                # around the claim. T59 fix + preserves M120 C Neuron
                # shape where entity name is at sentence start.
                _registry_key_for_guard = str(key) if key else ""
                _guard_entry = {
                    "entity": entry.get("entity", ""),
                    "aliases": entry.get("aliases", []) or [],
                    "measurement_type": entry.get(
                        "measurement_type", ""),
                    "registry_key": _registry_key_for_guard,
                }
                _skip_reason = _m123_evaluate_conflict_path_guards(
                    sent, pair, _guard_entry)
                if _skip_reason is not None:
                    skips.append({
                        "path": "conflict",
                        "claim": pair,
                        "expected_entity": entry.get("entity", ""),
                        "registry_key": _registry_key_for_guard,
                        "sentence": sent,
                        "skip_reason": _skip_reason,
                        "reason": _skip_reason,
                    })
                    continue

                flags.append({
                    "type": "REGISTRY_VALUE_CONFLICT",
                    "sentence_span": [span[0], span[1]],
                    "claim_value": claim_val,
                    "claim_unit": claim_unit,
                    "registry_key": key,
                    "registry_entity": entry.get("entity", ""),
                    "registry_value": reg_val,
                    "registry_unit": reg_unit,
                    "tolerance_exceeded": True,
                    "rel_diff": rel_diff,
                    "sentence": sent,
                })

    return flags, skips


def annotate_conflict_flags(response_text, conflict_flags):
    """Append inline clarifications for REGISTRY_VALUE_CONFLICT flags.

    Preserves the original sentence (Option A per M120 C directive §2).
    The clarification is appended after the sentence's terminator. If
    multiple conflicts fire on the same sentence, they are coalesced
    into a single parenthetical.

    Returns the annotated response string. If no flags, returns the
    original text unchanged.
    """
    if not response_text or not conflict_flags:
        return response_text

    # Group flags by sentence_span so we append one parenthetical per
    # affected sentence even when multiple registry keys disagree.
    by_span = {}
    for f in conflict_flags:
        if f.get("type") != "REGISTRY_VALUE_CONFLICT":
            continue
        span = f.get("sentence_span")
        if not span or len(span) != 2:
            continue
        key = (span[0], span[1])
        by_span.setdefault(key, []).append(f)

    if not by_span:
        return response_text

    # Insert clarifications from the end backward so earlier spans stay
    # valid while we mutate.
    out = response_text
    for (start, end), flags in sorted(by_span.items(), key=lambda kv: kv[0][0],
                                      reverse=True):
        parts = []
        for f in flags:
            reg_val = f.get("registry_value")
            reg_unit = f.get("registry_unit", "")
            reg_key = f.get("registry_key", "")
            # Format the registry value cleanly (drop trailing .0 on ints).
            if isinstance(reg_val, float) and reg_val.is_integer():
                reg_val_str = str(int(reg_val))
            else:
                reg_val_str = str(reg_val)
            parts.append(
                f"{reg_key}={reg_val_str} {reg_unit}".strip())
        clarification = (
            " (measurement registry records different value: "
            + "; ".join(parts)
            + "; flagging for audit)"
        )
        out = out[:end] + clarification + out[end:]
    return out


# ── Tier 2 narrative-drift check (M118 C) ────────────────────────────
# m118_c_tier2_narrative_drift
#
# Mechanism anchor: M117 K4 NEW_scrub_under_detection. Four pilot
# instances — T24 (DFlash narrative), T30 (NSIRD_ANECompilerService_*
# path fabrication), T33 (ZinAneTd<N> template name), T35 (xnu_kernel
# memory path citation). Pattern: scrub Tier 2 binding-check catches
# numeric misattribution; authoritative prose without numeric markers
# passes through unchanged. Distinct from M116 A outbound guard — M116 A
# fires on grounding *absence*; this fires when grounding IS present but
# the narrative paraphrases drift from it semantically.
#
# Design (M118 Stream C directive):
#   Option β primary — embed each specific-claim sentence via the shared
#   CoreML MiniLM embedder (LocalMemoryStore.emb_model); compute max
#   cosine vs grounding-corpus sentence embeddings.
#   Thresholds (starting values; tune via test evidence):
#     <0.45            → DRIFT (strip sentence)
#     0.45 ≤ sim < 0.60 → ambiguous — Option α fallback (LLM judge) if
#                         available; else conservative PASS (defer to
#                         other tiers). Ambiguous-without-fallback
#                         intentionally PASSES to avoid over-strip per K2.
#     ≥0.60            → PASS
#
# Specific-claim markers (sentence is a candidate for drift-checking):
#   - filesystem path shape (/ or contains .py/.md/.hwx/.json/.mlpackage)
#   - quoted string / backticked code
#   - CamelCase proper-noun-looking token (>= 2 caps transitions, e.g.
#     `ZinAneTd`, `ANECompilerService`, `IOSurfaceSharedEvent`)
#   - All-caps token of length >= 4 (e.g. `KILLED_SPEED`, `NSIRD`)
#   Sentences without any marker skip the drift check (preserves
#   conversational prose from false-positive strip).

_SHARED_EMBEDDER = None
_SHARED_EMBEDDER_LOAD_TRIED = False


def _get_shared_embedder():
    """Return the CoreML MiniLM embedder shared with LocalMemoryStore.

    Returns None if unavailable (scrub falls back to no-op drift check
    rather than raising — drift check is best-effort, other tiers keep
    firing).
    """
    global _SHARED_EMBEDDER, _SHARED_EMBEDDER_LOAD_TRIED
    if _SHARED_EMBEDDER_LOAD_TRIED:
        return _SHARED_EMBEDDER
    _SHARED_EMBEDDER_LOAD_TRIED = True
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _mem_dir = str(_Path(__file__).resolve().parent.parent / "memory")
        if _mem_dir not in _sys.path:
            _sys.path.insert(0, _mem_dir)
        try:
            from coreml_embedder import maybe_load_coreml_embedder
        except ImportError:
            from phantom_memory.coreml_embedder import (
                maybe_load_coreml_embedder,
            )
        _SHARED_EMBEDDER = maybe_load_coreml_embedder()
        # Fallback to CPU SentenceTransformer if CoreML unavailable.
        if _SHARED_EMBEDDER is None:
            try:
                from sentence_transformers import SentenceTransformer
                _SHARED_EMBEDDER = SentenceTransformer(
                    "all-MiniLM-L6-v2", device="cpu")
            except Exception:
                _SHARED_EMBEDDER = None
    except Exception:
        _SHARED_EMBEDDER = None
    return _SHARED_EMBEDDER


# Specific-claim markers.
_PATH_TOKEN_RE = re.compile(
    r"(?:/[\w\-.@]+){2,}|[\w\-]+\.(?:py|md|hwx|json|mlpackage|mlmodelc|"
    r"dylib|plist|kext|c|cpp|m|h|sh|txt|jsonl)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"`[^`]+`|\"[^\"]+\"|'[^']+'")
# CamelCase: two or more uppercase transitions (e.g. ZinAneTd,
# ANECompilerService).
_CAMEL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*\b|"
                       r"\b(?:[A-Z]+[a-z]*){2,}[A-Za-z0-9]*\b")
_ALLCAPS_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")


def _has_specific_claim(sentence: str) -> bool:
    """Does this sentence contain a specific-claim marker?"""
    if _PATH_TOKEN_RE.search(sentence):
        return True
    if _QUOTED_RE.search(sentence):
        return True
    if _CAMEL_RE.search(sentence):
        return True
    if _ALLCAPS_RE.search(sentence):
        return True
    return False


def _split_corpus_sentences(corpus: str):
    """Sentence-split grounding corpus. Keeps list/bullet lines as their
    own sentences so path citations inside recalled memories are
    individually addressable by cosine."""
    if not corpus:
        return []
    lines = [ln.strip() for ln in corpus.split("\n") if ln.strip()]
    out = []
    for ln in lines:
        parts = _SENTENCE_RE.split(ln)
        out.extend(p.strip() for p in parts if p.strip())
    return out


def _embed_texts(embedder, texts):
    """Best-effort embed. Returns (N, D) float32 or None."""
    if embedder is None or not texts:
        return None
    try:
        import numpy as _np
        arr = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _np.asarray(arr, dtype=_np.float32)
    except Exception:
        return None


def tier2_narrative_drift_check(
    response_text: str,
    grounding_corpus: str,
    *,
    low_threshold: float = 0.50,
    high_threshold: float = 0.60,
    llm_judge=None,
):
    # m118_c_tier2_narrative_drift thresholds — tuned from pilot evidence
    # (see vault/agent_reports/m118_c_scrub_narrative_drift.md §thresholds).
    # Starting directive values were 0.45 / 0.60. M117 pilot T33 sent-1
    # (`ZinAneTd<N>` fab) at 0.488 and T35 xnu_kernel citation at 0.4545
    # sat in the 0.45-0.60 ambiguous band and default-PASSed under the
    # starting thresholds → 2/4 pilot replay. Raising low → 0.50 lets
    # these land in the DRIFT band while leaving T36 (0.575 correct) and
    # T4 regression sentence (0.676) above the high threshold safe. High
    # threshold held at 0.60.
    """Embedding-similarity claim-vs-grounding drift check.

    Returns (verdict, diagnostic) where:
      verdict = "DRIFT" if any claim sentence flagged
              | "PASS" otherwise (no eligible claims, or all above high
                threshold, or embedder unavailable)
      diagnostic = dict with flags (per-sentence), stripped, skipped
                   counts, and latency.

    Caller (scrub_response) also consumes `diagnostic["flags"]` via
    strip_flagged_sentences to remove the drift-flagged sentences.
    """
    t0 = time.time()
    diagnostic = {
        "flags": [],
        "checked": 0,
        "skipped_no_marker": 0,
        "embedder_available": False,
        "thresholds": {"low": low_threshold, "high": high_threshold},
        "latency_ms": 0,
    }

    if not response_text or not grounding_corpus:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    sentences = _split_sentences(response_text)
    if not sentences:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    claim_sentences = []
    for s in sentences:
        if _has_specific_claim(s):
            claim_sentences.append(s)
        else:
            diagnostic["skipped_no_marker"] += 1
    diagnostic["checked"] = len(claim_sentences)
    if not claim_sentences:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    embedder = _get_shared_embedder()
    if embedder is None:
        # Best-effort: no embedder ⇒ cannot run drift check. Conservative
        # PASS — other tiers still fire. Surfaced in diagnostic.
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic
    diagnostic["embedder_available"] = True

    corpus_sents = _split_corpus_sentences(grounding_corpus)
    if not corpus_sents:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    claim_emb = _embed_texts(embedder, claim_sentences)
    corpus_emb = _embed_texts(embedder, corpus_sents)
    if claim_emb is None or corpus_emb is None:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    # Cosine via matmul (embeddings are L2-normalized already).
    try:
        import numpy as _np
        sims = claim_emb @ corpus_emb.T  # (C, G)
        max_sims = sims.max(axis=1) if sims.size else _np.zeros(
            (len(claim_sentences),), dtype=_np.float32)
    except Exception:
        diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
        return "PASS", diagnostic

    # m118_c_tier2_narrative_drift: precompute a lowercase grounding
    # string for Option γ narrow token-absence lookups (engaged only for
    # sentences in the ambiguous 0.50-0.60 band).
    _grounding_lower = grounding_corpus.lower()

    for sent, sim in zip(claim_sentences, max_sims.tolist()):
        if sim < low_threshold:
            diagnostic["flags"].append({
                "type": "NARRATIVE_DRIFT",
                "tier": "2_narrative",
                "max_cosine": float(sim),
                "threshold": float(low_threshold),
                "sentence": sent,
            })
        elif sim < high_threshold:
            # Ambiguous band. Option α fallback if judge supplied.
            judged_drift = False
            if llm_judge is not None:
                try:
                    judged = llm_judge(sent, grounding_corpus)
                except Exception:
                    judged = None
                if judged == "DRIFT":
                    judged_drift = True
                    diagnostic["flags"].append({
                        "type": "NARRATIVE_DRIFT",
                        "tier": "2_narrative_alpha",
                        "max_cosine": float(sim),
                        "threshold": float(low_threshold),
                        "judge": "llm_drift",
                        "sentence": sent,
                    })
            if not judged_drift:
                # Option γ narrow — cheap token-absence check. Scan the
                # sentence for specific-claim tokens (paths, camel/all-
                # caps names, backtick-quoted). If every such token is
                # missing from the grounding corpus, flag as DRIFT. This
                # catches `ZinAneTd<N>`, `stage1_agent_a_xnu_kernel_
                # memory.md` and similar fabrications whose cosine sits
                # in the ambiguous band due to surface-word overlap with
                # unrelated grounding sentences.
                #
                # Interim per M118 Stream C K-handling: if β alone
                # under-covers, ship γ narrow; full fix deferred to
                # M120 if still insufficient.
                tokens = []
                tokens.extend(_PATH_TOKEN_RE.findall(sent))
                for m in _CAMEL_RE.finditer(sent):
                    tokens.append(m.group(0))
                for m in _ALLCAPS_RE.finditer(sent):
                    tokens.append(m.group(0))
                for m in _QUOTED_RE.finditer(sent):
                    tokens.append(m.group(0).strip("`\"'"))
                # De-noise: drop short tokens and common English words
                # that survived the claim regexes.
                tokens = [t for t in tokens if len(t) >= 5]
                tokens = list({t.lower() for t in tokens})
                if tokens:
                    missing = [t for t in tokens
                               if t not in _grounding_lower]
                    if missing and len(missing) == len(tokens):
                        diagnostic["flags"].append({
                            "type": "NARRATIVE_DRIFT",
                            "tier": "2_narrative_gamma",
                            "max_cosine": float(sim),
                            "threshold": float(low_threshold),
                            "missing_tokens": missing[:8],
                            "sentence": sent,
                        })
            # Otherwise ⇒ conservative PASS on ambiguous sentences.
        # sim >= high_threshold ⇒ PASS

    verdict = "DRIFT" if diagnostic["flags"] else "PASS"
    diagnostic["latency_ms"] = int((time.time() - t0) * 1000)
    return verdict, diagnostic


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


# m118_c_tier2_narrative_drift: reuse the M116 A abstain-string constant
# so we can skip the drift check when the upstream already abstained.
# Defensive import — if confabulation_shape_detector isn't reachable,
# skip-by-abstain simply doesn't trigger.
def _m116_abstain_message():
    try:
        from confabulation_shape_detector import ABSTAIN_MESSAGE
        return ABSTAIN_MESSAGE
    except Exception:
        return None


def scrub_response(response_text, grounding_corpus,
                   user_query="", tools_called=None,
                   narrative_drift_enabled=True,
                   narrative_drift_llm_judge=None,
                   canonical_atoms=None):
    """Run Tier 0 + Tier 1 + Tier 2 checks on a response.

    Returns dict:
        flags: list of all flags
        cleaned_response: response with flagged sentences removed (or None)
        tier0_flags: FAB flags (fabricated tool claims)
        tier1_flags: NOGROUND flags
        tier2_flags: MISATTRIBUTED flags
        tier2_narrative_flags: NARRATIVE_DRIFT flags (m118_c)
        tier2_narrative_verdict: DRIFT/PASS/SKIPPED_ABSTAIN (m118_c)
        tier2_narrative_diagnostic: dict with cosine stats (m118_c)
        latency_ms: total scrub time
        sentences_stripped: count

    M125.3 Stream B additions:
      scrub_mechanism_fired: list of tier names whose flag list is
          non-empty at post-gate time (P1 attribution).
      canonical_atom_skips: list of dicts {tier, sentence, atom_id,
          atom_text, overlap_tokens, overlap_count} — per-flag
          suppression events from the P2 scrub-relevance gate.

    New parameter:
      canonical_atoms — optional list of {"id","text","type",...}
        dicts representing canonical_atom rows in the prompt's
        retrieved context. When provided AND a flagged sentence
        shares ≥2 non-stopword content tokens with at least one
        canonical_atom's text, the flag is suppressed (P2 gate).
        Purpose: prevent scrub tiers (narrative-drift especially)
        from destroying a response that cites retrieved canonical
        content. Does NOT suppress Tier 0 (fabricated tool claims)
        — those flags are model-fab, not relevance-dependent.
    """
    t0 = time.time()

    tier0 = tier0_fab_check(response_text, tools_called)
    if TIER1_ENABLED:
        tier1 = tier1_grounding_check(response_text, grounding_corpus, user_query)
    else:
        tier1 = []
    # M122 A2: tier2_binding_check now returns (flags, skips). `skips`
    # is the diagnostic list of bare-scalar / abstention guard fires
    # that suppressed a strip. Each skip is a dict with skip_reason
    # enum {word_overlap_zero, abstention_pattern}. Consumed via
    # scrub.tier2_binding.skip_reason in turn JSON per Stream A3 wire-up.
    tier2, tier2_binding_skip_reasons = tier2_binding_check(
        response_text, grounding_corpus, user_query)

    # m118_c_tier2_narrative_drift
    tier2_narrative = []
    tier2_narr_verdict = "PASS"
    tier2_narr_diag = {}
    abstain_msg = _m116_abstain_message()
    if (narrative_drift_enabled
            and response_text
            and (abstain_msg is None or response_text.strip() != (
                abstain_msg or "").strip())):
        try:
            tier2_narr_verdict, tier2_narr_diag = tier2_narrative_drift_check(
                response_text, grounding_corpus,
                llm_judge=narrative_drift_llm_judge,
            )
            tier2_narrative = list(tier2_narr_diag.get("flags", []))
        except Exception as _nd_err:
            tier2_narr_diag = {"error": str(_nd_err)}
            tier2_narr_verdict = "PASS"
    elif abstain_msg is not None and response_text and (
            response_text.strip() == abstain_msg.strip()):
        tier2_narr_verdict = "SKIPPED_ABSTAIN"

    tier3 = tier3_phrase_repetition_check(response_text)

    # m120_c_tier2_registry_conflict: additive, non-stripping. Surfaces
    # registry-value disagreements to the user via inline clarification.
    # M123 A1 — now returns (flags, skips) per alternate-path guard
    # extension. `skips` carries path="conflict" entries that merge with
    # the binding-path skips for ζ v2.2 emission.
    tier2_conflict_skip_reasons = []
    try:
        _conflict_result = tier2_registry_conflict_check(
            response_text, user_query)
        if isinstance(_conflict_result, tuple):
            tier2_conflict, tier2_conflict_skip_reasons = _conflict_result
        else:  # defensive: pre-M123 single-return shape
            tier2_conflict = _conflict_result or []
    except Exception:
        tier2_conflict = []
        tier2_conflict_skip_reasons = []

    # M123 A1: merge binding-path and conflict-path skips into a single
    # list of {path, reason, ...} dicts. Empty list if neither guard
    # fired. Downstream consumers (midas_ui.py turn-record emit) can
    # filter by `path` or treat the list uniformly.
    tier2_binding_skip_reasons = list(tier2_binding_skip_reasons or [])
    tier2_binding_skip_reasons.extend(tier2_conflict_skip_reasons or [])

    # M125.3 Stream B P2: scrub-relevance gate.
    # If any flagged sentence shares ≥2 non-stopword content tokens with
    # a canonical_atom row from the prompt's retrieved context, suppress
    # that flag. The model cited retrieved canonical content; scrub
    # should not destroy it. Does NOT apply to tier0 (fabricated tool
    # claim — not a grounding-relevance issue).
    # Phase 0 anchor: T33 had 5 canonical_atom rows in prompt (prefix-
    # cache content), model produced a 328-char response, scrub tier
    # (narrative-drift) stripped 100%. Gate would recover that turn.
    canonical_atom_skips = []
    _relevance_gate_enabled = (
        canonical_atoms is not None and len(canonical_atoms) > 0)

    def _filter_by_relevance(flags, tier_name):
        """Return (kept_flags, skips) after applying the P2 gate.

        tier0 is exempt (fabrication, not relevance-dependent).
        """
        if not _relevance_gate_enabled:
            return flags, []
        if tier_name == "tier0":
            return flags, []
        kept = []
        skips = []
        for _flag in flags:
            _sent = _flag.get("sentence") or ""
            if not _sent:
                kept.append(_flag)
                continue
            _sent_toks = set(_m122_content_tokens(_sent))
            if not _sent_toks:
                kept.append(_flag)
                continue
            _match = None
            _match_overlap = None
            for _atom in canonical_atoms:
                _atext = (_atom or {}).get("text") or ""
                if not _atext:
                    continue
                _atoks = set(_m122_content_tokens(_atext))
                _shared = _sent_toks & _atoks
                if len(_shared) >= 2:
                    _match = _atom
                    _match_overlap = _shared
                    break
            if _match is not None:
                skips.append({
                    "tier": tier_name,
                    "sentence": _sent,
                    "atom_id": (_match or {}).get("id", ""),
                    "atom_text": ((_match or {}).get("text") or "")[:200],
                    "overlap_tokens": sorted(list(_match_overlap))[:10],
                    "overlap_count": len(_match_overlap),
                    "skip_reason": "canonical_atom_relevance",
                })
            else:
                kept.append(_flag)
        return kept, skips

    tier1_kept, _s1 = _filter_by_relevance(tier1, "tier1")
    tier2_kept, _s2 = _filter_by_relevance(tier2, "tier2")
    tier2_narrative_kept, _s2n = _filter_by_relevance(
        tier2_narrative, "tier2_narrative")
    tier3_kept, _s3 = _filter_by_relevance(tier3, "tier3")
    canonical_atom_skips = _s1 + _s2 + _s2n + _s3

    # Preserve pre-gate counts for attribution; apply gate to the
    # sets that drive stripping.
    tier1_pre_gate = tier1
    tier2_pre_gate = tier2
    tier2_narrative_pre_gate = tier2_narrative
    tier3_pre_gate = tier3
    tier1 = tier1_kept
    tier2 = tier2_kept
    tier2_narrative = tier2_narrative_kept
    tier3 = tier3_kept

    all_flags = tier0 + tier1 + tier2 + tier2_narrative + tier3

    # M125.3 Stream B P1: scrub-tier attribution. Two signals:
    #   scrub_mechanism_fired      — post-gate tiers that actually
    #                                 contributed to stripping. Closes
    #                                 the Phase 0 gap where total_flags=1
    #                                 appeared with empty tier arrays.
    #   scrub_mechanism_pre_gate   — tiers that would have fired absent
    #                                 the P2 relevance gate. Under the
    #                                 common (no-atoms) path this is
    #                                 identical to scrub_mechanism_fired;
    #                                 under the gate path it tells you
    #                                 which layer the gate silenced.
    _mechanism_fired = []
    if tier0:
        _mechanism_fired.append("tier0")
    if tier1:
        _mechanism_fired.append("tier1")
    if tier2:
        _mechanism_fired.append("tier2")
    if tier2_narrative:
        _mechanism_fired.append("tier2_narrative")
    if tier3:
        _mechanism_fired.append("tier3")
    _mechanism_pre_gate = []
    if tier0:
        _mechanism_pre_gate.append("tier0")  # tier0 exempt; always same
    if tier1_pre_gate:
        _mechanism_pre_gate.append("tier1")
    if tier2_pre_gate:
        _mechanism_pre_gate.append("tier2")
    if tier2_narrative_pre_gate:
        _mechanism_pre_gate.append("tier2_narrative")
    if tier3_pre_gate:
        _mechanism_pre_gate.append("tier3")

    # m118_c_tier2_narrative_drift: additional 40%-stripped safety gate
    # per directive §2. If narrative-drift alone flagged >40% of the
    # claim sentences, escalate to abstain by clearing cleaned output.
    # M125.3 P2: use post-gate tier2_narrative so the relevance gate
    # actually suppresses overstrip when atoms justify the content.
    _narrative_sentences = {f["sentence"] for f in tier2_narrative
                            if f.get("sentence")}
    _checked = tier2_narr_diag.get("checked", 0)
    _narrative_overstrip = (
        _checked > 0
        and len(_narrative_sentences) / max(_checked, 1) > 0.40
    )

    cleaned = strip_flagged_sentences(response_text, all_flags)
    if _narrative_overstrip:
        cleaned = None  # → caller falls back to abstain per §2

    # m120_c_tier2_registry_conflict: annotate the cleaned response with
    # inline clarifications. Option A — preserve sentence, append audit
    # note — rather than silently stripping true claims. See M119 T79
    # for the mechanism anchor.
    if cleaned and tier2_conflict:
        cleaned = annotate_conflict_flags(cleaned, tier2_conflict)

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
        "tier2_narrative_flags": tier2_narrative,
        "tier2_narrative_verdict": tier2_narr_verdict,
        "tier2_narrative_diagnostic": tier2_narr_diag,
        "narrative_overstrip_triggered": _narrative_overstrip,
        # m120_c_tier2_registry_conflict: additive, non-stripping flags.
        # Persisted to turn JSON by the caller as
        # scrub.tier2_conflict_flagged.
        "tier2_conflict_flagged": tier2_conflict,
        # M122 A2: skipped strips with diagnostic skip_reason per entry.
        # Empty list when no skips fired. Key matches the Stream A3
        # wire-up stub in midas_ui.py (emitted as
        # scrub.tier2_binding.skip_reason in turn JSON v2.2).
        "tier2_binding_skip_reasons": tier2_binding_skip_reasons,
        "tier3_flags": tier3,
        "cleaned_response": cleaned,
        "total_flags": len(all_flags),
        "sentences_stripped": sentences_stripped,
        "original_response_chars": len(response_text),
        "cleaned_response_chars": len(cleaned) if cleaned else 0,
        "latency_ms": latency_ms,
        # M125.3 Stream B P1 — scrub-tier attribution. Post-P2-gate.
        "scrub_mechanism_fired": _mechanism_fired,
        # M125.3 Stream B P1 — pre-gate attribution. Which tier
        # would have fired absent the P2 relevance gate. Identical
        # to scrub_mechanism_fired on the no-atoms path.
        "scrub_mechanism_pre_gate": _mechanism_pre_gate,
        # M125.3 Stream B P1 — pre-gate counts for methodology audit.
        # When the P2 gate suppresses flags, these tell you which
        # tier would have fired absent the relevance gate.
        "tier1_flags_pre_gate_count": len(tier1_pre_gate),
        "tier2_flags_pre_gate_count": len(tier2_pre_gate),
        "tier2_narrative_flags_pre_gate_count": len(tier2_narrative_pre_gate),
        "tier3_flags_pre_gate_count": len(tier3_pre_gate),
        # M125.3 Stream B P2 — scrub-relevance gate skips.
        "canonical_atom_skips": canonical_atom_skips,
        "canonical_atom_skip_count": len(canonical_atom_skips),
        "canonical_atoms_considered": (
            len(canonical_atoms) if canonical_atoms else 0),
    }
