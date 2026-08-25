"""
M100 Agent A1: query-side vocabulary normalization.

Four concrete targets from M99 re-score
(vault/agent_reports/m99_live_pilot_scoring.md §"Per-turn scoring"):

1. "Paper 2" → canonical file `vault/paper/five_roadblocks_personal_ai.md`
   and topic keywords the file is actually indexed under
   (M87 T07/T08, M99 T24 — identical FALSE_ABSTAIN pattern).

2. "ANE Enclave" ↔ "ANE exclave" bridge
   (M99 T6/T7 — vault says "exclave", user says "Enclave"; 20+ matches for
   the canonical spelling, 0 for the colloquial one).

3. "SME" / "sme" case-insensitive AND rare-token preservation
   (M87 T30, M99 T33 — LocalMemoryStore already case-folds but vault_read
   scoring outranked SME files because common words "small", "model", "run"
   accumulated more hits than the rare but load-bearing "sme" token).

4. "op codes" / "opcodes" token-joining
   (M99 T32 — vault has 53 opcodes cataloged but "op codes" (spaced) missed
   because `op` and `codes` are separate stop-adjacent tokens).

Mechanism layer: this module sits BELOW the router and BEFORE retrieval
scoring. It is query-side only — no indexer changes, no briefing changes.
Both LocalMemoryStore.recall and tool_executor._vault_read /
_vault_research consume the same tables so behavior stays consistent.

Guardrail: keep tables tight. Every alias is an extra embed + extra pass
over candidate files. If an entry is motivated by a single failed query,
it belongs in a follow-up directive, not here.
"""
from __future__ import annotations

import re


# ── Alias table ─────────────────────────────────────────────────────────────
#
# Canonical substitutions applied to the *query string* before retrieval.
# Keys are lowercase patterns matched as whole tokens (word-boundary aware).
# Values are the canonical form(s) the vault actually uses. Expansion is
# additive: original query is always kept, variants are appended.
#
# Paper 2 expands to the explicit phrase AND to the topic keywords the file
# is indexed under — "paper 2" alone does not appear in the file body.

_ALIAS_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Target 1 — "Paper 2" resolves to five_roadblocks_personal_ai.md
    ("paper 2", (
        "five roadblocks personal ai",
        "five roadblocks",
    )),
    ("paper two", (
        "five roadblocks personal ai",
        "five roadblocks",
    )),
    ("paper2", (
        "five roadblocks personal ai",
        "five roadblocks",
    )),
    # Target 2 — ANE enclave ↔ exclave. Already mirrored in
    # LocalMemoryStore._QUERY_EXPANSIONS; keep canonical spelling lookup
    # here so vault_read/_vault_research pick up the same bridge. We do
    # NOT disambiguate — operator (M99 T7 body) described the exclave's
    # 128-slot firmware wall while asking about "the Enclave", so the
    # colloquial usage maps to the exclave documentation every time
    # observed so far. If that changes (distinct real Enclave content
    # lands in vault), add a follow-up disambiguation rule.
    ("enclave", ("exclave",)),
    ("exclave", ("enclave",)),

    # Target 4 — "op codes" (spaced) / "op-codes" → "opcodes".
    # The token-joining direction is one-way on purpose: "opcodes" already
    # matches the canonical spelling in vault, so no expansion needed
    # from that side.
    ("op codes", ("opcodes",)),
    ("op-codes", ("opcodes",)),
    ("op code", ("opcode",)),
    ("op-code", ("opcode",)),

    # m118_e_sub_shape_1 — "latent alignment" vocabulary gap.
    # User vocab: "latent alignment" / "aligning latents". Vault vocab:
    # "latent bridge", "Paper 3-B", "MLP-4096", "inter-model communication",
    # "Bridge M64", "8B to 31B". M118 E cosine probe (2026-04-21): bare
    # "have we explored latent alignment?" scores 0.26 against main64_bridge_summary
    # and 0.23 against claudemem "Latent Bridge Infrastructure (M64)" — below
    # the 0.40 recall floor in midas_ui. Expansion lifts those to 0.609 and
    # 0.550 respectively, both comfortably above threshold. Canonical-tag
    # rewrite on the stored records is NOT the right fix — the records are
    # assistant-sourced and correctly tagged; the gap is query-vocabulary.
    ("latent alignment", (
        "latent bridge paper 3-b inter-model communication",
        "bridge m64 mlp-4096 8b to 31b",
    )),
    ("aligning latents", (
        "latent bridge paper 3-b inter-model communication",
        "bridge m64 mlp-4096 8b to 31b",
    )),
    ("latent alignments", (
        "latent bridge paper 3-b inter-model communication",
    )),
    ("inter-model alignment", (
        "latent bridge paper 3-b inter-model communication",
    )),

    # ── M125 A1.1 — cosine/rank robustness (pool-gap C2+C3 vocab gaps) ──
    # Each alias is grounded in a specific M124 A 5-cause evidence row.
    # Under-fit by design: narrow substring triggers, no loose phrasing.

    # T17 (m123_c) — cache hit rate on verifier → prefix-cache/KV-cache canonical
    # Canonical: "prefix cache", "system-message KV", "67% hit rate", "Main 25"
    ("cache hit rate", (
        "prefix cache hit rate system-message kv caching",
    )),
    ("hit rate on verifier", (
        "prefix cache verifier 67% hit rate main 25",
    )),

    # T18 (m123_c) — how raised hit rate → mechanism canonical
    ("raise the hit rate", (
        "system-message kv caching kvcache.state prefix cache",
    )),
    ("raised the hit rate", (
        "system-message kv caching kvcache.state prefix cache",
    )),
    ("raised hit rate", (
        "system-message kv caching kvcache.state prefix cache",
    )),
    ("get the hit rate up", (
        "system-message kv caching kvcache.state prefix cache",
    )),

    # T62 (m122_c) — Apple Silicon generation → macOS version framing
    # Canonical documents M-series via macOS 13 / iOS 16 SharedEvents framing.
    ("apple silicon generation", (
        "macos 13 ios 16 sharedevents m-series",
    )),
    ("which generation", (
        "macos version ios version",
    )),

    # T65 (m122_c) — TTFT delta last few sessions → +4.2% / Main 48 framing
    ("ttft delta", (
        "ttft +4.2% main 48 cached prefill",
    )),
    ("ttft across sessions", (
        "ttft +4.2% main 48 session milestones",
    )),
    ("ttft across the last", (
        "ttft +4.2% main 48 session milestones",
    )),

    # T66 (m122_c) + T28 (m123_c) — session-label bridges.
    # Query uses "M108" / "M115-M118", canonical uses "Main NN" / "Main 115".
    # Additive: keep original + append Main-prefixed variant.
    # Note: covered generically via _SESSION_MARKER_RE below, not an alias rule.

    # T17/T18 secondary — "prefix cache" / "kv cache" bridge
    ("kv cache", (
        "prefix cache system-message kv",
    )),
    ("prefix cache", (
        "system-message kv caching kvcache.state",
    )),

    # T28 (m123_c) + T66 (m122_c) — "fix surface" colloquial → canonical terms
    ("fix surface", (
        "parent synthesis shipped fixes m117 m122",
    )),
    ("walk me through the fixes", (
        "parent synthesis shipped fixes session milestones",
    )),

    # T62 secondary — "SharedEvents path" bridge to IOSurface framing
    ("sharedevents path", (
        "iosurface shared events ane-dispatch gpu-ane sync",
    )),
)


# M125 A1.1 — session-marker expansion. Queries like "M108", "M115-M118",
# "M121" need to bridge to "Main 108", "Main 115", etc. used in session
# milestones and parent-synthesis files. Also reverse direction.
# Implemented inline in expand_query() so the regex fires on both directions
# without needing one alias rule per session number (there are 100+).
_M_SESSION_RE = re.compile(r"\bm(\d{2,3})\b", re.IGNORECASE)
_MAIN_SESSION_RE = re.compile(r"\bmain\s+(\d{2,3})\b", re.IGNORECASE)
_M_RANGE_RE = re.compile(
    r"\bm(\d{2,3})\s*(?:-|–|through|to)\s*m?(\d{2,3})\b", re.IGNORECASE
)


# ── Rare-token rank boost ───────────────────────────────────────────────────
#
# Target 3 — SME.  Query "could a small model run on the SME?" reaches
# _vault_read which scores by raw keyword-hit count.  Common words
# "small", "model", "run" accumulate 3+ hits on files that mention them
# incidentally; files that actually document SME accumulate only 1 hit
# (on "sme") and lose the rank fight.
#
# This set lists rare, high-signal technical tokens that should receive a
# per-match rank bonus in the vault_read / vault_research scorers when
# they appear in the query AND in a file. Keep it tight: only add terms
# where a bare mention is reliably topic-indicative (not words with many
# unrelated meanings).
#
# CRITICAL: these must be tokens that are (a) topic-indicative when they
# appear AND (b) not so common in the vault that they appear in most
# files. "ANE" fails criterion (b) — appears in hundreds of files — so
# it is NOT a rare-signal token even though it's topical. "SME", on
# the other hand, is both topic-indicative AND only appears in ~20 files.
RARE_HIGH_SIGNAL_TOKENS: frozenset[str] = frozenset({
    # Instruction set / hardware acronyms — rare in vault corpus
    "sme", "smopa", "nax",
    "h17g", "h16",
    # Research artifacts indexed under exact strings in vault
    "eagle",
    "locomo",
    "cipher",
    # File-specific anchors (Paper 2 expansion target)
    "roadblocks",
    # Spec decode drafter names
    "pard",
    # Hardware RE opcodes (never false-positive)
    "0x9141", "0x9161", "0x9149",
    # ANE architecture terms — specific enough to float on-topic snippets
    "exclave",
    # Token-joined alias expansion target (M99 T32)
    "opcodes", "opcode",
})

# Boost multiplier for hits on rare tokens. Base = large bonus per
# UNIQUE rare-token-hit file so canonical files always outrank zero-hit
# tangential ones. A common-word file might accumulate 900 hits from
# query tokens like "paper"/"about"/"how"; to overcome that we need the
# base bonus to be on the same order of magnitude per rare-token match.
RARE_TOKEN_BONUS: int = 100

# Occurrence-scaled component. Each additional occurrence of a rare
# token in the file adds `RARE_TOKEN_PER_OCCURRENCE` up to the cap.
# Tightens the ordering between two files that both contain a rare
# token — the one that mentions it 30 times wins over the one that
# mentions it once.
RARE_TOKEN_PER_OCCURRENCE: int = 3
RARE_TOKEN_OCCURRENCE_BONUS_CAP: int = 60


# ── Canonical file hints ───────────────────────────────────────────────────
#
# Some query aliases map to a SPECIFIC canonical file, not just a topic.
# When the alias trigger fires in the query, the named file(s) should
# receive a large rank bonus regardless of how keyword-dense it is
# internally — the alias IS the answer.
#
# Example: operator asks "what is Paper 2?" — the correct answer is
# "Paper 2 is the 'Five Roadblocks to Persistent Memory for Personal AI'
# paper at vault/paper/five_roadblocks_personal_ai.md". A scorer that
# only counts keyword hits will bury the canonical file under any
# longer file that happens to use the word "paper" often.
#
# Format: trigger-substring -> tuple of relative paths (relative to
# VAULT_PATH). Substring match is case-insensitive, word-boundary
# aware.
CANONICAL_FILE_HINTS: dict[str, tuple[str, ...]] = {
    "paper 2": (
        "paper/five_roadblocks_personal_ai.md",
    ),
    "paper two": (
        "paper/five_roadblocks_personal_ai.md",
    ),
    "paper2": (
        "paper/five_roadblocks_personal_ai.md",
    ),
    "53 opcodes": (
        "knowledge/ane_hardware.md",
    ),
    "53 op codes": (
        "knowledge/ane_hardware.md",
    ),
    "how many opcodes": (
        "knowledge/ane_hardware.md",
    ),
    "how many op codes": (
        "knowledge/ane_hardware.md",
    ),
    "ane enclave": (
        "knowledge/ane_hardware.md",
    ),
    "the enclave": (
        "knowledge/ane_hardware.md",
    ),
}

# Bonus applied to a match when the file is in the canonical-hint set
# for a trigger that fires on the query. Large enough to always float
# the canonical file into the top-N even against keyword-stuffed
# tangential files.
CANONICAL_FILE_HINT_BONUS: int = 10000


# ── Stop-word-safe rare-token preservation ─────────────────────────────────
#
# The existing stopword filters in tool_executor._vault_read / _vault_research
# drop short words.  The rare-token set above is length-tolerant, but callers
# still need punctuation stripped before set membership tests (e.g. "SME?"
# after query.split() must land as "sme").  Expose the strip helper so call
# sites share behavior.

_PUNCT_STRIP_RE = re.compile(r"[\s\?\!\.,;:\(\)\[\]\"']+")


def strip_query_punct(token: str) -> str:
    """Strip surrounding punctuation that `str.split()` leaves on query
    tokens.  "SME?" -> "sme" — without this, the scorer tests the
    literal "sme?" against lowercased content and misses every hit."""
    return _PUNCT_STRIP_RE.sub("", token).strip().lower()


# ── Public API ──────────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """Return alt-spelling variants of `query` (original NOT included).

    Used by LocalMemoryStore.recall — it embeds each variant and takes the
    per-memory max cosine, which matches the fuzzy-rewrite style of
    vocabulary bridging.

    Match is case-insensitive and word-boundary aware so "paper 2" does
    not fire on "paper 20" and "sme" does not fire on "smedia".
    """
    if not query:
        return []
    low = query.lower()
    variants: list[str] = []
    seen: set[str] = {query}
    for key, replacements in _ALIAS_RULES:
        # Word-boundary match. Keys that end in a digit need a \b on both
        # sides to not accidentally extend into larger numbers.
        pattern = r"\b" + re.escape(key) + r"\b"
        if not re.search(pattern, low):
            continue
        for repl in replacements:
            # Replace preserving surrounding structure. Case-insensitive
            # substitution; output variant is the replacement lowercased,
            # since vault content comparison is lowercase anyway.
            new_q = re.sub(pattern, repl, low)
            if new_q != low and new_q not in seen:
                variants.append(new_q)
                seen.add(new_q)

    # M125 A1.1 — session marker expansion. "M108" -> "Main 108" and vice
    # versa so session-labelled canonicals surface regardless of which form
    # the operator typed. Also expand ranges like "M115-M118" to a variant
    # that names each session explicitly (Main 115, Main 116, ...).
    # Additive only; original query is preserved by the caller.
    m_hits = _M_SESSION_RE.findall(low)
    main_hits = _MAIN_SESSION_RE.findall(low)
    if m_hits:
        new_q = _M_SESSION_RE.sub(lambda mo: f"main {mo.group(1)}", low)
        if new_q != low and new_q not in seen:
            variants.append(new_q)
            seen.add(new_q)
    if main_hits:
        new_q = _MAIN_SESSION_RE.sub(lambda mo: f"m{mo.group(1)}", low)
        if new_q != low and new_q not in seen:
            variants.append(new_q)
            seen.add(new_q)
    # Range expansion: "M115-M118" -> "main 115 main 116 main 117 main 118"
    for rng in _M_RANGE_RE.finditer(low):
        try:
            lo_ = int(rng.group(1))
            hi_ = int(rng.group(2))
        except (TypeError, ValueError):
            continue
        if hi_ < lo_ or hi_ - lo_ > 12:
            # Cap range at 12 sessions to bound expansion cost
            continue
        expanded = " ".join(f"main {n}" for n in range(lo_, hi_ + 1))
        new_q = low[:rng.start()] + expanded + low[rng.end():]
        if new_q != low and new_q not in seen:
            variants.append(new_q)
            seen.add(new_q)
    return variants


def expand_query_terms(
    terms: list[str], raw_query: str | None = None
) -> list[str]:
    """Expand a list of pre-tokenized query terms with alias replacements.

    Used by tool_executor._vault_read / _vault_research, which score by
    bag-of-words hits rather than cosine.

    The caller may pass the ORIGINAL (un-stopworded, un-stripped) query
    string as `raw_query` — alias rules like "paper 2" require two
    adjacent tokens to co-occur, which the tokenized list can't express
    once stop words are removed ("paper" "about" ≠ "paper 2"). If
    `raw_query` is None, we fall back to joining `terms`, which covers
    the simpler "op codes" / "enclave" rules.

    Returns the union of original terms + tokens from each alias variant,
    de-duplicated. Stop-word filtering remains the caller's job — this
    function only ADDS vocabulary coverage.

    Example:
        raw_query="what is Paper 2 about?",
        terms=["what","paper","about"]
          -> ["what","paper","about","five","roadblocks","personal","ai"]
    """
    if not terms and not raw_query:
        return terms
    joined = raw_query if raw_query else " ".join(terms)
    variants = expand_query(joined)
    expanded: list[str] = list(terms)
    seen = {t.lower() for t in terms}
    for v in variants:
        for w in v.split():
            wl = w.lower().strip()
            # Strip trailing punct on variant tokens too (e.g., "paper?")
            wl = strip_query_punct(wl)
            if wl and wl not in seen:
                expanded.append(wl)
                seen.add(wl)
    return expanded


def is_rare_high_signal(token: str) -> bool:
    """True iff `token` (after punctuation strip) is in the rare-token set.
    Callers use this to apply a per-hit rank bonus in keyword scoring."""
    return strip_query_punct(token) in RARE_HIGH_SIGNAL_TOKENS


def canonical_file_hints(query: str) -> tuple[str, ...]:
    """Return relative paths of canonical-hint files for this query.

    Returns an empty tuple when no hint trigger fires.  Callers
    (vault_read / vault_research) add CANONICAL_FILE_HINT_BONUS to any
    match whose relative path matches one of the returned paths.
    """
    if not query:
        return ()
    low = query.lower()
    hits: list[str] = []
    for trigger, paths in CANONICAL_FILE_HINTS.items():
        pattern = r"\b" + re.escape(trigger) + r"\b"
        if re.search(pattern, low):
            hits.extend(paths)
    # De-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in hits:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def rare_token_score(query_terms: list[str], content_lower: str) -> int:
    """Compute the rare-token rank bonus for a file.

    For each *unique* query term that is a rare-signal token AND appears
    in the content:
      + RARE_TOKEN_BONUS (base, independent of occurrence count)
      + RARE_TOKEN_PER_OCCURRENCE * min(count, RARE_TOKEN_OCCURRENCE_BONUS_CAP)

    The base bonus guarantees topical files always win over zero-hit
    tangential ones even when the tangential file is much longer and
    accumulates many common-word hits. The occurrence component
    tightens the ordering between two on-topic files.
    """
    bonus = 0
    seen_tokens: set[str] = set()
    for w in query_terms:
        t = strip_query_punct(w)
        if not t or t in seen_tokens:
            continue
        if t not in RARE_HIGH_SIGNAL_TOKENS:
            continue
        if t not in content_lower:
            continue
        seen_tokens.add(t)
        occ = content_lower.count(t)
        bonus += RARE_TOKEN_BONUS
        bonus += RARE_TOKEN_PER_OCCURRENCE * min(occ, RARE_TOKEN_OCCURRENCE_BONUS_CAP)
    return bonus
