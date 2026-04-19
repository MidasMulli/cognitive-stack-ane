"""Main 35 Part 2A — Domain-agnostic perception prompt for the 8B layer.

Replaces the technical-extraction prompt that was used at paper time
(see vault/paper/m1_3b_memory.md S4 addendum + m5_ablation.md
load-bearing finding). The old prompt asked for "discrete facts"
with rules tuned to technical claims, which (a) made the 8B redundant
with the CPU FactExtractor regex patterns and (b) actively hurt
LoCoMo where conversational extraction is needed.

The new prompt:
  - Domain-agnostic (no Apple Silicon, no ISDA, no specific vocabulary)
  - 8 categories: fact, speculation, correction, decision, preference,
    plan, relationship, context
  - Asks for confidence level + speaker + topic on every extraction
  - Output is JSONL — one record per line — for easy parsing

Designed to be the *primary* extraction layer; the CPU FactExtractor
becomes the structured fast-path that annotates and supplements.
"""

PERCEPTION_PROMPT = """You are an always-on perception system that processes everything the user reads, writes, and discusses. Your job is to notice and extract MEANING from any text, regardless of domain.

For each piece of meaningful information, output one JSON record per line. Categories:

- fact          — something stated as currently true
- speculation   — a possibility, hypothesis, or open question
- correction    — an update or contradiction of an earlier belief
- decision      — a choice made about what to do or not do
- preference    — how the speaker likes things done, what they value
- plan          — an intended future action or commitment
- relationship  — a connection between people, concepts, or topics
- context       — why something matters; what prompted it

For EACH extraction, output exactly one line of JSON with these fields:
{{"type": "...", "content": "one complete sentence", "confidence": "certain|likely|speculative|implied", "speaker": "human|assistant|system", "topic": "<short topic tag>"}}

Rules:
1. Extract from ANY domain. Technical measurements, personal details, strategic thinking, casual observations, code references, emotional reactions — all of it.
2. One record per claim. Never combine two facts.
3. Use the speaker's words verbatim where possible. Preserve numbers exactly.
4. If a statement is hedged ("might", "I think", "possibly"), set confidence to "likely" or "speculative" accordingly.
5. If the speaker isn't explicit but is clearly implied, use confidence "implied".
6. Skip pleasantries, filler, meta-conversation about the chat interface, and content that's already extracted from an earlier message.
7. Your value is BREADTH. Catch things that pattern-matching would miss: decisions, preferences, soft claims, corrections.
8. Output ONLY the JSON lines. No preamble, no headers, no explanatory prose.

TEXT:
---
{text}
---

EXTRACTIONS:
"""


PERCEPTION_PROMPT_LITE = """Extract every meaningful piece of information from this text as a JSONL list. Each line:
{{"type": "fact|speculation|correction|decision|preference|plan|relationship|context", "content": "one sentence", "confidence": "certain|likely|speculative|implied", "speaker": "human|assistant|system", "topic": "tag"}}

Output JSON only, no preamble. Skip pleasantries and filler.

TEXT:
{text}

EXTRACTIONS:
"""


def render(text: str, lite: bool = False) -> str:
    p = PERCEPTION_PROMPT_LITE if lite else PERCEPTION_PROMPT
    return p.format(text=text)


def parse_extractions(raw: str) -> list[dict]:
    """Parse JSONL output from the model. Tolerant of stray prose."""
    import json as _json
    out = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        if s.endswith(","):
            s = s[:-1]
        try:
            rec = _json.loads(s)
            if isinstance(rec, dict) and "type" in rec and "content" in rec:
                out.append(rec)
        except Exception:
            continue
    return out
