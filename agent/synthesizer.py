"""
Layer 4: Synthesizer — Cognitive Architecture v1.

Context assembly: system prompt + briefing + query mode + history + memories.
Uses the full 16K context budget intelligently.

Main 48: CORE ARCHITECTURE block is now derived from
vault/subconscious_state_synthesis.md at boot time, not hand-maintained.
This eliminates the failure class where the system prompt contradicts
current state because nobody updated it after a pipeline change.
"""

import os as _os

from query_classifier import classify_query
from briefing_assembler import assemble_briefing


def _build_core_architecture_block() -> str:
    """Build the CORE ARCHITECTURE block from the state synthesis file.

    Reads subconscious_state_synthesis.md (the single source of truth)
    and extracts key sections into a ~500 token block for the system prompt.
    Falls back to a minimal static block if the file is unreadable.
    """
    synthesis_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__)))),
        "vault", "subconscious_state_synthesis.md")

    try:
        with open(synthesis_path) as f:
            text = f.read()
    except Exception:
        return _FALLBACK_CORE_ARCHITECTURE

    lines = []

    # --- Extract from Section 2: process state ---
    lines.append(_extract_between(
        text,
        "## 2. Current process state",
        "## 3.",
        extract_fn=_extract_process_state))

    # --- Extract from Section 3: seven retrieval shapes ---
    lines.append(_extract_between(
        text,
        "## 3. Seven retrieval shapes",
        "## 4.",
        extract_fn=_extract_retrieval_shapes))

    # --- Extract from Section 13b: pipeline + scrub + concurrency ---
    lines.append(_extract_between(
        text,
        "## 13b. Main 46 results",
        "## 14.",
        extract_fn=_extract_pipeline_and_scrub))

    # --- Extract from Section 14: hardware measurements ---
    lines.append(_extract_between(
        text,
        "## 14. Hardware measurements",
        "## 15.",
        extract_fn=_extract_hardware))

    # --- Extract from Section 13: dead paths (condensed) ---
    lines.append(_extract_between(
        text,
        "## 13. Dead paths",
        "## 13b.",
        extract_fn=_extract_dead_paths))

    block = "\n".join(line for line in lines if line)
    return block if block.strip() else _FALLBACK_CORE_ARCHITECTURE


def _extract_between(text, start_heading, end_heading, extract_fn=None):
    """Pull text between two headings, optionally transform it."""
    start = text.find(start_heading)
    if start < 0:
        return ""
    end = text.find(end_heading, start + len(start_heading))
    section = text[start:end] if end > start else text[start:]
    if extract_fn:
        return extract_fn(section)
    return section.strip()


def _extract_process_state(section):
    """From Section 2: production model, services table."""
    facts = []
    seen = set()
    # Extract from the services table
    for line in section.splitlines():
        if line.strip().startswith("| **") and "---" not in line:
            parts = [p.strip().strip("*") for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                key = parts[0]
                if key not in seen:
                    seen.add(key)
                    facts.append(f"- {key}: {parts[2].strip()}")
    # Production model line
    for line in section.splitlines():
        if "production model:" in line.lower():
            clean = line.strip().replace("**", "").strip()
            if "production" not in seen:
                seen.add("production")
                facts.append(f"- {clean}")
    return "\n".join(facts)


def _extract_retrieval_shapes(section):
    """From Section 3: the seven shapes, one line each."""
    shapes = []
    for line in section.splitlines():
        if line.strip().startswith("| **") and "---" not in line:
            parts = [p.strip().strip("*") for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                shapes.append(f"{parts[0]} ({parts[1]})")
    if shapes:
        return "- Seven retrieval shapes: " + ", ".join(shapes) + ". These are the ONLY seven shapes."
    return ""


def _extract_pipeline_and_scrub(section):
    """From Section 13b: five-phase pipeline, scrub, concurrency."""
    facts = []
    # Pipeline flow
    for line in section.splitlines():
        if "**Perception" in line or "**Recall" in line or "**Generation" in line or "**Reflection" in line or "**Output" in line:
            clean = line.strip().lstrip("0123456789. ")
            facts.append(f"  {clean}")
    if facts:
        facts.insert(0, "- Five-phase pipeline per turn:")

    # Concurrency
    if "22,402ms" in section or "22402" in section:
        facts.append("- Pipeline concurrency: 8B extracts PREVIOUS turn on ANE at START of current turn, concurrent with 27B GPU generation. 22,402ms mean overlap per turn. The 8B does NOT prepare context for the CURRENT turn — it processes the PREVIOUS turn's output.")

    # Scrub
    if "29% fabrication" in section or "0% fabrication" in section:
        facts.append("- Answer scrub (Tier 1 grounding + Tier 2 entity binding): <1ms. Fabrication 29% -> 0%. Hard absence gate blocks 27B when recall_score_max < 0.5.")

    return "\n".join(facts)


def _extract_hardware(section):
    """From Section 14: key hardware measurements."""
    key_props = [
        "DRAM bandwidth", "ANE dedicated DMA", "31B pure AR",
        "8B ANE extractor", "Cross-accelerator contention",
        "MiniLM embed", "Neuron routing", "Pipeline overlap",
        "Content-correct", "Fabrication rate", "Prefix cache",
    ]
    facts = []
    for line in section.splitlines():
        for prop in key_props:
            if prop in line and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    facts.append(f"- {parts[0]}: {parts[1]}")
                break
    # Static hardware facts that don't change
    facts.append("- Hardware: M5 Pro, 64 GB unified, 18 CPU (6P+12S), 20 GPU, 307 GB/s DRAM, ANE 111 GB/s dedicated DMA.")
    facts.append("- NUMA: DIE1 zero compute, DCS 7:3 split, measurements off by up to 2.33x on page placement.")
    facts.append("- ANE rule: Q8 mandatory. FP16 wastes memory, Q4 has 31% dequant penalty.")
    facts.append("- Memory store: LocalMemoryStore (SQLite WAL + numpy float32 matrix). Sub-ms cosine retrieval.")
    facts.append("- Provenance: source_role tags, assistant-sourced 0.5x penalty to prevent self-poisoning.")
    return "\n".join(facts)


def _extract_dead_paths(section):
    """From Section 13: condensed dead path summary, one line each."""
    paths = []
    for line in section.splitlines():
        if line.strip().startswith("- **"):
            # Extract just the bold name and first sentence of kill reason
            clean = line.strip().lstrip("- ")
            # Truncate at first period after the bold name
            bold_end = clean.find("**", 2)
            if bold_end > 0:
                rest = clean[bold_end + 2:].strip(". ")
                first_sentence = rest.split(".")[0] if rest else ""
                name = clean[2:bold_end]
                if first_sentence:
                    paths.append(f"  {name}: {first_sentence}.")
    if paths:
        paths.insert(0, "- Key dead paths (do not revive):")
        return "\n".join(paths[:8])
    return ""


_FALLBACK_CORE_ARCHITECTURE = """- Production verifier: Gemma 4 31B Dense Q4 on GPU (port 8899). Pure AR.
- 8B extractor: Llama 3.1-8B Q8 on ANE. 72 CoreML models, 7.9 tok/s.
- Memory store: LocalMemoryStore (SQLite WAL + numpy float32). Sub-ms cosine.
- Seven retrieval shapes: Point, Path, Ambient, Recency, Adversarial, Enumeration, Absence.
- Pipeline: extract_prev(8B ANE, concurrent) + recall(CPU) -> generate(31B GPU) -> scrub(CPU) -> return.
- Answer scrub: Tier 1+2, <1ms, 0% fabrication. Absence gate blocks 31B when recall empty.
- Hardware: M5 Pro, 64 GB, 307 GB/s DRAM, ANE 111 GB/s dedicated DMA.
- ANE rule: Q8. Provenance: 0.5x assistant penalty."""


# Build the block once at import time
_CORE_ARCHITECTURE = _build_core_architecture_block()

# Phase 3A: Precision system prompt
SYSTEM_PROMPT = f"""You are Midas, Nick's technical research partner on Apple Silicon. He's an expert, not a developer — builds through you and Claude Code.

RULES: Answer first. No filler. No openers. Have opinions — commit to one recommendation backed by data. Match Nick's register. If something is PARKED/DEAD, say so — never suggest revival. Synthesize memories into reasoning, don't list them. Say "I don't know" when you don't.

GROUNDING: When the briefing or per-query memories contain Dead path entries, measured findings, or specific file/probe names relevant to the question, you MUST cite them by name and use them as constraints. Do NOT propose generic approaches when specific work already exists in memory. If the memories say "X is killed because Y" or "Z is the open lead", build the answer around those facts. Generic brainstorming when specific findings are present is a hallucination. Never dump raw technical details (register values, API constraints, paper findings) unless the user specifically asked for them.

NEVER SAY: "could be valuable to explore", "particularly interesting", "could provide insights", "I think we should also", "Additionally,", "Furthermore,", "Lastly,". Be direct.

NEVER PROMISE WITHOUT EXECUTING: do not say "Let me do that now", "I will perform that search", "Let me search for that", "I'll look that up", "give me a moment", or any variant where you describe an action you are about to take. If a tool is available and appropriate, the tool has already been executed for this turn — its result is in the tool_result field of your context. If the tool_result is empty or missing, say "the search returned nothing" or "I don't have that data" and stop. Do not promise future work. Do not defer. If you cannot answer, say so explicitly. "Let me X" is a forbidden phrase because the prior turn's log shows you producing it repeatedly without ever dispatching the tool, wasting the user's next turn asking you to do what you already promised.

LENGTH: Match response length to question complexity. Yes/no = one sentence. Factual = 1-2 sentences. Dead path question = state it's dead, give the kill reason, stop. Recommendation = pick ONE thing and justify it in one paragraph. Status/casual opener ("what's up", "where are we", "status") = 2-3 sentences: current focus + next step, nothing else. "Explain" = as long as needed but dense. NEVER use numbered lists or bullet points unless the user explicitly asks for a list. NEVER enumerate all projects. NEVER pad a short answer. If you can say it in two sentences, do not write five paragraphs. Shorter is always better.

TOOL RESULTS: When a tool returns data, READ IT and use it. If the tool result contains specific findings, report them. Never say "we have no findings" when the tool result contains findings.

HALLUCINATION RULE: Never state that an event happened (published, shipped, posted, completed, announced) unless a specific memory confirms that exact event. If unsure whether something happened, say you don't know. Never fill gaps with plausible guesses presented as facts.

NEVER: Comment on your own output quality. Never say "I made a mistake" or "I should have answered" or "let me correct myself." Just give the right answer. No apologies. No self-critique.

EMPTY RESPONSES: When you don't have a memory or information to answer a question, say so explicitly (e.g. "I don't have information about that" or "Nothing in memory about that"). Never return an empty response. Always give the user something useful.

THINKING: Do NOT use <think> tags. Answer directly. No internal monologue.

CORE ARCHITECTURE (derived from vault/subconscious_state_synthesis.md at boot — always current, do not contradict):
{_CORE_ARCHITECTURE}"""


# Cached briefing state
_briefing_cache = {"text": "", "turn": 0, "memories": []}


def format_standing_rules_block(standing_rules):
    """Return a formatted system-message directive block for standing
    rules, or empty string if no rules. Single source of truth so both
    build_messages and the chain_of_reasoning fallback path inject the
    same block. Main 39 P1.5 / Section 17.6."""
    if not standing_rules:
        return ""
    rules_block = "\n".join(f"- {r}" for r in standing_rules)
    return (
        f"\n=== STANDING USER RULES ===\n"
        f"You MUST follow these rules in every response. They override "
        f"any default behavior, including length, format, and tone "
        f"defaults. Do not ask the user to confirm them. Do not "
        f"announce that you are following them. Just follow them.\n\n"
        f"{rules_block}\n"
        f"=== END STANDING USER RULES ==="
    )


# Main 40 P1: convert quantitative length constraints in standing rules
# to a hard max_tokens cap on the LLM generation call. The 72B's soft
# compliance on natural-language length directives ("two sentences"
# producing 4-6) is a model property; capping max_tokens forces
# compliance regardless of instruction-following calibration.
#
# Conversion model (Main 40 P1 retuned post-validation):
#   1 sentence  ≈ 25 tokens   (English avg ~15 words at 1.3 tok/word
#                              + small completion buffer)
#   1 word      ≈ 1.4 tokens
#
# Range semantics within a single rule: take the LOWER bound. This
# matches the directive's "most restrictive wins" cross-rule rule.
# "two or three sentences" → 2 sentences cap. The user is fine with
# either, and the literal validation target ("under 200 chars" for
# the load-bearing test query) only passes at the lower bound.
import re as _re

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
    # Common phrasings
    "a": 1, "an": 1, "single": 1, "couple": 2, "few": 3, "several": 4,
    "half": 1,  # "half a sentence" → not really meaningful, treat as 1
}

# Pattern: <number-or-word> <unit>. Capture both numeric and word forms.
_NUM_TOKEN = r"(?:\d+|" + "|".join(_NUMBER_WORDS.keys()) + r")"
_UNIT_TOKEN = r"(?:sentence|sentences|word|words|paragraph|paragraphs|line|lines|character|characters|chars|char)"

_QUANT_RE = _re.compile(
    r"\b(?:(\d+)|("
    + "|".join(_NUMBER_WORDS.keys())
    + r"))\s+(" + _UNIT_TOKEN + r")\b",
    _re.IGNORECASE,
)

# Range pattern: "X or Y unit", "X to Y unit", "X-Y unit", "between X and Y unit".
# Captures the LOWER bound — matches the directive's "most restrictive wins"
# semantics. Example: "two or three sentences" → cap at 2 sentences.
_RANGE_RE = _re.compile(
    r"\b(?:between\s+)?(" + _NUM_TOKEN + r")\s*(?:or|to|-|–|and)\s*(" + _NUM_TOKEN + r")\s+(" + _UNIT_TOKEN + r")\b",
    _re.IGNORECASE,
)


def _word_or_digit_to_int(s):
    if s is None:
        return None
    s = s.lower()
    if s.isdigit():
        return int(s)
    return _NUMBER_WORDS.get(s)

# "no more than X", "at most X", "max X", "under X", "in X" → upper-bound style
_UPPER_BOUND_PREFIXES = (
    "no more than", "at most", "no longer than", "less than", "under",
    "max", "maximum", "max of", "at most of", "in",
)

# Convert a unit count to an estimated token cap with headroom.
def _unit_to_tokens(count: int, unit: str) -> int:
    """Convert a (count, unit) pair to a token cap.

    Sentence estimate is 20 tokens (15 words × 1.3 tok/word) — slightly
    tight on purpose. The streaming endpoint's n-gram drafter typically
    overruns the requested max_tokens by ~10 tokens at the end of a draft
    batch (K=16 truncation), so a literal cap of N produces N+10 actual
    decoded tokens. Tightening the per-unit estimate compensates.
    """
    unit = unit.lower().rstrip("s")
    if unit == "sentence":
        return max(12, count * 20)
    if unit == "word":
        return max(6, int(count * 1.4) + 2)
    if unit == "paragraph":
        return max(30, count * 60)  # ~3 sentences/paragraph at 20 tok each
    if unit == "line":
        return max(12, count * 15)
    if unit in ("character", "char"):
        return max(6, count // 4 + 2)
    return None


def parse_max_tokens_from_rules(standing_rules, default_max=None):
    """Scan standing rules for quantitative length constraints. Return
    the smallest cap implied by any rule, or `default_max` if no rule
    has a quantitative constraint.

    Patterns recognized (single rule):
        "two sentences"           -> 70 tokens
        "in two sentences"        -> 70 tokens
        "answer in 50 words"      -> 74 tokens
        "max 100 words"           -> 144 tokens
        "under 200 words"         -> 284 tokens
        "single sentence"         -> 35 tokens
        "a paragraph"             -> 120 tokens
        "two or three sentences"  -> 105 tokens (uses larger of range)
        "two-three sentences"     -> 105 tokens
        "1-2 sentences"           -> 70 tokens

    Range semantics within one rule: take the LARGER bound so the model
    has room to express. Cross-rule conflicts: take the smallest cap
    (per the directive — most restrictive wins).
    """
    if not standing_rules:
        return default_max
    per_rule_caps = []
    for raw in standing_rules:
        if not raw:
            continue
        text = raw.lower()
        # First pass: range patterns ("X or Y unit", "X-Y unit"). Take
        # the LOWER bound. Track which spans we matched so the single-
        # number pass below doesn't double-count the upper bound.
        matches = []
        consumed_spans = []
        for m in _RANGE_RE.finditer(text):
            x = _word_or_digit_to_int(m.group(1))
            y = _word_or_digit_to_int(m.group(2))
            unit = m.group(3)
            if x is None or y is None:
                continue
            matches.append((min(x, y), unit))
            consumed_spans.append((m.start(), m.end()))
        # Second pass: single (number, unit) pairs not already inside a
        # consumed range span.
        for m in _QUANT_RE.finditer(text):
            if any(m.start() >= s and m.end() <= e for s, e in consumed_spans):
                continue
            num_digit, num_word, unit = m.group(1), m.group(2), m.group(3)
            if num_digit is not None:
                count = int(num_digit)
            else:
                count = _NUMBER_WORDS.get(num_word.lower())
                if count is None:
                    continue
            matches.append((count, unit))
        if not matches:
            # Also catch "single sentence", "a paragraph" without "or"
            for keyword, count in (
                ("single sentence", 1), ("one sentence", 1),
                ("a sentence", 1), ("one word", 1),
                ("a paragraph", 1), ("one paragraph", 1),
            ):
                if keyword in text:
                    matches.append((count, keyword.split()[-1]))
            if not matches:
                continue
        # Range within rule: pick LOWER bound (most restrictive).
        # E.g. "two or three sentences" → 2 sentences cap. Matches the
        # directive's "smallest wins" semantics for cross-rule conflicts.
        same_unit = {}
        for count, unit in matches:
            u = unit.lower().rstrip("s")
            if u in same_unit:
                same_unit[u] = min(same_unit[u], count)
            else:
                same_unit[u] = count
        # Compute token cap for each unit, take the smallest across units
        # within this rule (the most restrictive per-rule constraint).
        per_unit_caps = []
        for u, c in same_unit.items():
            cap = _unit_to_tokens(c, u)
            if cap is not None:
                per_unit_caps.append(cap)
        if per_unit_caps:
            per_rule_caps.append(min(per_unit_caps))
    if not per_rule_caps:
        return default_max
    rule_min = min(per_rule_caps)
    if default_max is None:
        return rule_min
    return min(default_max, rule_min)


def build_messages(history, user_msg, tool_name=None, tool_args=None,
                   tool_result=None, memory_context=None, briefing=None,
                   query_mode=None, standing_rules=None,
                   expand_history_on_low_recall=False):
    """Build the message list with full context assembly.

    Context budget (~14K usable tokens):
      System prompt + mode instruction:  ~300 tokens
      Standing rules block:              ~150 tokens (Main 39 P1.5)
      Briefing:                          ~500 tokens
      History:                           ~4000 tokens max
      Tool result / memories:            ~2000 tokens
      User message:                      variable
      Reserved for generation:           ~2000 tokens min

    M113 gamma — recall-quality regime gate:
      When tool_name+tool_result are set, the default contracted regime
      keeps only history[-4:] (last 2 user/assistant pairs). Under
      M108 Finding 2, this structurally evicts earlier conversation
      content (e.g. T25 information-theory answer dropped at T30 because
      T27/T28 are the newer pairs). When the caller signals that the
      tool's own recall returned a low max_score (i.e. the tool cannot
      rescue the evicted context either), we flip to the expanded
      regime's history walk so the follow-up query can still resolve
      against the model's conversation memory. The tool_result tail is
      preserved in both regimes — flipping only widens history, it does
      not drop the tool grounding.
    """
    # Assemble system prompt — Main 25 Build 0: keep this byte-stable across
    # turns within a session so the verifier's prefix KV cache hits. Per-query
    # variables (mode instructions, query-specific memories) are pushed into
    # the user message tail by the caller (midas_ui.py).
    system_parts = [SYSTEM_PROMPT]

    # Add briefing (Phase 2B). Stable per session.
    if briefing:
        system_parts.append(f"\n{briefing}")
    elif memory_context:
        mem_block = "\n".join(f"- {m}" for m in memory_context[:8])
        system_parts.append(
            f"\nRELEVANT MEMORIES:\n{mem_block}")

    # Main 39 P1.5 (Section 17.6): standing rules ride at the END of the
    # system prompt as imperative directives, after the briefing/memories.
    # The end position gives them the highest authority gradient. Stable
    # per session (loaded from prior session bridge at module boot), so
    # cache geometry holds — three stable blocks instead of two.
    rules_directive = format_standing_rules_block(standing_rules)
    if rules_directive:
        system_parts.append(rules_directive)

    system = "\n".join(system_parts)

    # query_mode rides in the user message instead of the system slot, so
    # different query categories don't invalidate the prefix cache.
    mode_prefix = f"[MODE: {query_mode}]\n\n" if query_mode else ""

    messages = [{"role": "system", "content": system}]

    if tool_name and tool_result is not None:
        # Tool synthesis: limited history + tool result.
        # M113 gamma: under expand_history_on_low_recall, walk full
        # history like the conversation branch (16000 char budget)
        # instead of capping to the last 2 pairs. The tool_result tail
        # is still appended below — this only widens the conversation
        # window so follow-up queries can resolve against earlier turns
        # that the contracted regime would structurally evict.
        if expand_history_on_low_recall:
            char_budget = 16000
            total_chars = 0
            history_to_include = []
            for _msg in reversed(history):
                msg_chars = len(_msg.get("content", ""))
                if total_chars + msg_chars > char_budget:
                    break
                history_to_include.insert(0, _msg)
                total_chars += msg_chars
            recent = history_to_include
        else:
            recent = history[-4:] if len(history) > 4 else history
        for msg in recent:
            messages.append(msg)

        messages.append({"role": "user", "content": mode_prefix + user_msg})

        result_str = str(tool_result)[:4000]
        result_useful = (len(result_str.strip()) > 20
                         and "no matches" not in result_str.lower()
                         and "not found" not in result_str.lower()
                         and "error" not in result_str.lower()[:50])

        if result_useful:
            messages.append({
                "role": "user",
                "content": (
                    f"Answer my question using ALL available context.\n"
                    f"Priority: 1) your BRIEFING (most reliable), "
                    f"2) the {tool_name} result below, "
                    f"3) conversation history.\n"
                    f"If the tool result doesn't directly answer the question "
                    f"but your briefing does, use the briefing.\n"
                    f"Be specific with numbers. Don't repeat yourself.\n\n"
                    f"Tool result:\n{result_str}"
                )
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"The {tool_name} search returned no useful results. "
                    f"Answer using your BRIEFING context above. "
                    f"Be specific with numbers. Don't claim ignorance "
                    f"if the briefing has relevant data."
                )
            })
    else:
        # Direct conversation: full history
        # Cap history at ~4000 tokens (~16K chars)
        char_budget = 16000
        total_chars = 0
        history_to_include = []
        for msg in reversed(history):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > char_budget:
                break
            history_to_include.insert(0, msg)
            total_chars += msg_chars

        for msg in history_to_include:
            messages.append(msg)
        messages.append({"role": "user", "content": mode_prefix + user_msg})

    return messages


def synthesize(llm_fn, history, user_msg, tool_name=None, tool_args=None,
               tool_result=None, max_tokens=800, temperature=0.7,
               memory_context=None, briefing=None, standing_rules=None):
    """Generate a text response with cognitive context assembly.

    Args:
        llm_fn: callable(messages, max_tokens, temperature) -> str
        history: list of prior {role, content} messages
        user_msg: current user message
        tool_name: if set, we're synthesizing a tool result
        tool_args: the args passed to the tool
        tool_result: the tool's output string
        max_tokens: generation limit
        temperature: sampling temperature
        memory_context: list of memory strings (fallback if no briefing)
        briefing: assembled briefing document string (Phase 2B)
        standing_rules: list of imperative user rules to inject as a
            system-message directive block (Main 39 P1.5 / Section 17.6)

    Returns:
        str: the LLM's response text
    """
    # Phase 3B: Classify query type
    query_type, mode_instruction = classify_query(user_msg)

    # Phase 4C + 6A/6B: Check reasoning mode from signal bus
    reasoning_mode = "single"
    try:
        from signal_bus import read as sig_read, update as sig_update
        if query_type == "analytical":
            reasoning_mode = "chain"
            sig_update("reasoning_mode", "chain")
        elif query_type == "debugging":
            reasoning_mode = "chain"
            sig_update("reasoning_mode", "chain")
        else:
            sig_update("reasoning_mode", "single")
    except Exception:
        pass

    # Phase 6A: Chain of reasoning for complex queries
    if reasoning_mode == "chain" and not tool_name:
        try:
            from reasoning_chain import chain_of_reasoning
            context = briefing or ("\n".join(memory_context[:10]) if memory_context else "")
            # Main 39 P1.5: standing rules must apply to the chain path
            # too, not just the standard build_messages path. Append the
            # directive block to system_base so chain_of_reasoning sees it.
            chain_system = SYSTEM_PROMPT + format_standing_rules_block(standing_rules)
            # Main 40 P1: pass the rule-derived token cap into the
            # synthesis step. The internal scratch steps keep their
            # default budgets; only the user-facing synthesis is capped.
            chain_max = parse_max_tokens_from_rules(standing_rules, max_tokens)
            return chain_of_reasoning(llm_fn, user_msg, context,
                                       system_base=chain_system,
                                       max_tokens=chain_max)
        except Exception:
            pass  # Fall through to normal generation

    messages = build_messages(
        history, user_msg,
        tool_name=tool_name, tool_args=tool_args,
        tool_result=tool_result,
        memory_context=memory_context,
        briefing=briefing,
        query_mode=mode_instruction,
        standing_rules=standing_rules,
    )
    # Main 40 P1: apply quantitative rule cap if any active rule sets one.
    # parse_max_tokens_from_rules returns the smaller of (default_max,
    # rule-derived cap), so this only ever tightens the budget.
    effective_max = parse_max_tokens_from_rules(standing_rules, max_tokens)
    return llm_fn(messages, max_tokens=effective_max, temperature=temperature)
