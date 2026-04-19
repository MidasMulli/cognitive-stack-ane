"""M73 Track B P1/P2: static query-type dispatch for 31B sampling.

Classify a user message into one of 10 query types (heuristic first pass)
and emit the dispatch bundle from the M73 lever matrix.

P1 consumes `sample_profile.temperature` only; the rest of the schema is
surfaced for downstream consumers (P2: retrieval_modifiers; future:
synthetic_layers, steering_vector, multi_model_mode).
"""

import re
from typing import Optional

QUERY_TYPES = (
    "canonical_lookup", "historical", "decision", "code_gen",
    "synthesis", "conceptual", "debugging", "planning",
    "brainstorm", "chit_chat",
)

_SAMPLE_PROFILES = {
    "canonical_lookup": dict(temperature=0.0, min_p=0.1,  repetition_penalty=1.05, max_tokens=150,  best_of_n=1),
    "historical":       dict(temperature=0.0, min_p=0.1,  repetition_penalty=1.05, max_tokens=250,  best_of_n=1),
    "decision":         dict(temperature=0.0, min_p=0.1,  repetition_penalty=1.05, max_tokens=200,  best_of_n=1),
    "code_gen":         dict(temperature=0.1, min_p=0.05, repetition_penalty=1.00, max_tokens=800,  best_of_n=1),
    "synthesis":        dict(temperature=0.5, min_p=0.05, repetition_penalty=1.20, max_tokens=1500, best_of_n=3),
    "conceptual":       dict(temperature=0.5, min_p=0.05, repetition_penalty=1.20, max_tokens=1200, best_of_n=3),
    "debugging":        dict(temperature=0.3, min_p=0.05, repetition_penalty=1.05, max_tokens=2000, best_of_n=2),
    "planning":         dict(temperature=0.4, min_p=0.05, repetition_penalty=1.20, max_tokens=2000, best_of_n=2),
    "brainstorm":       dict(temperature=0.9, min_p=0.02, repetition_penalty=1.20, max_tokens=1500, best_of_n=5),
    "chit_chat":        dict(temperature=0.7, min_p=0.05, repetition_penalty=1.20, max_tokens=300,  best_of_n=1),
}

_RETRIEVAL_MODIFIERS = {
    "canonical_lookup": dict(canonical_boost=True,  absence_gate_sensitivity="strict", scope_rerank_weight=1.0),
    "historical":       dict(canonical_boost=True,  absence_gate_sensitivity="strict", scope_rerank_weight=1.0),
    "decision":         dict(canonical_boost=True,  absence_gate_sensitivity="strict", scope_rerank_weight=1.0),
    "code_gen":         dict(canonical_boost=False, absence_gate_sensitivity="normal", scope_rerank_weight=0.8),
    "synthesis":        dict(canonical_boost=False, absence_gate_sensitivity="loose",  scope_rerank_weight=0.5),
    "conceptual":       dict(canonical_boost=False, absence_gate_sensitivity="loose",  scope_rerank_weight=0.5),
    "debugging":        dict(canonical_boost=False, absence_gate_sensitivity="normal", scope_rerank_weight=0.7),
    "planning":         dict(canonical_boost=False, absence_gate_sensitivity="normal", scope_rerank_weight=0.5),
    "brainstorm":       dict(canonical_boost=False, absence_gate_sensitivity="loose",  scope_rerank_weight=0.3),
    "chit_chat":        dict(canonical_boost=False, absence_gate_sensitivity="normal", scope_rerank_weight=0.5),
}

# Priority order: most specific pattern first so generic "what is" doesn't
# shadow "what is broken" (debugging) or "what if we" (brainstorm).
_PATTERNS = [
    ("code_gen",    [r'\b(write|add|emit|generate)\b.{0,40}\b(function|class|script|module|method|loop|test|patch|fixture|decorator|endpoint)\b',
                     r'\b(implement|refactor|port|rewrite|convert)\b',
                     r'\bcode (for|that|to|up)\b',
                     r'\bin (python|c\+\+|rust|go|swift|objective[- ]c|metal)\b',
                     r'^```']),
    ("debugging",   [r'\b(bug|crash|traceback|exception|stack ?trace)\b',
                     r'\b(error|fail(ed|ing)?|broken|not working|doesn[\']?t work|wedged|stuck|hang(s|ing)?)\b',
                     r'\bwhy (isn[\']?t|is it not|won[\']?t)\b']),
    ("brainstorm",  [r'\b(brainstorm|ideation|possibilities|what if we)\b',
                     r'\bideas (for|about)\b', r'\bways to\b', r'\bexplore options\b']),
    ("planning",    [r'\b(plan|roadmap|timeline|milestones|next steps|strategy)\b',
                     r'\bhow should we\b', r'\bwhat are the steps\b']),
    ("synthesis",   [r'\b(summar(?:iz|is)(?:e|ing|ed)|synthesi[sz]e|consolidate|overview|digest)\b',
                     r'\btie together\b', r'\bcombine (the|all)\b']),
    ("historical",  [r'\b(last session|previous session|yesterday|last week|earlier today)\b',
                     r'\bwhen did (we|you|i)\b', r'\bwhat did we\b', r'\bhave we (ever |already )?\b',
                     r'\bm\d{2,3}\b', r'\bmain\s*\d{2,3}\b']),
    ("decision",    [r'\bshould (i|we)\b', r'\bwhich is (better|right|correct)\b',
                     r'\b(is|are) (this|that|it) (the |worth )\b',
                     r'\bworth\s+\w+ing\b', r'\bought (to|we)\b']),
    ("conceptual",  [r'\bwhy\b', r'\bhow (does|do) .* (work|operate|behave)\b',
                     r'\bhow (does|do|is|are) .* (relate|related|connect|connected|affect|affects|depend|depends|interact|interacts)\b',
                     r'\bexplain\b', r'\bwhat[\']?s the relationship\b',
                     r'\bdifference between\b', r'\bintuition behind\b',
                     r'\bwalk me through\b', r'\bstep (me )?through\b', r'\btake me through\b']),
    ("canonical_lookup", [r'\b(what is|what was|what[\']?s|how many|how much|how fast|how big|value of|size of|at what)\b']),
]

_CHIT_CHAT_MAX_WORDS = 4


def classify_query_type(message: str) -> str:
    if not message or not message.strip():
        return "chit_chat"
    lower = message.lower().strip()
    words = lower.split()

    if len(words) <= _CHIT_CHAT_MAX_WORDS and not any(
            re.search(p, lower) for _, pats in _PATTERNS for p in pats):
        return "chit_chat"

    for qtype, patterns in _PATTERNS:
        for p in patterns:
            if re.search(p, lower):
                return qtype

    if lower.endswith("?"):
        return "canonical_lookup"
    return "chit_chat"


def _apply_meta_overrides(profile: dict, meta_flags: Optional[dict]) -> dict:
    # absence: NOT a sampling decision. Handled by absence_guard's
    # skip-generation path (midas_ui :2896+). dispatch() surfaces
    # force_abstain=True in meta_flags; temperature stays at base profile.
    # contradiction: low-but-nonzero temp breaks degenerate loops on
    # contradiction-resolution prompts.
    # TODO(M74): ambiguity override pending query_reformulation + hypothesis
    # synthetic layer infra; currently a no-op.
    if not meta_flags:
        return profile
    out = dict(profile)
    if meta_flags.get("contradiction"):
        out["temperature"] = min(out.get("temperature", 0.0), 0.2)
    return out


def dispatch(message: str, meta_flags: Optional[dict] = None) -> dict:
    qtype = classify_query_type(message)
    sample_profile = _apply_meta_overrides(_SAMPLE_PROFILES[qtype], meta_flags)
    out_flags = dict(meta_flags) if meta_flags else {}
    if meta_flags and meta_flags.get("absence"):
        out_flags["force_abstain"] = True
    return {
        "query_type": qtype,
        "sample_profile": sample_profile,
        "retrieval_modifiers": dict(_RETRIEVAL_MODIFIERS[qtype]),
        "meta_flags": out_flags,
        "source": "m73_p15_static",
    }


if __name__ == "__main__":
    tests = [
        "What is the ANE decode tok/s for Llama-8B Q8?",
        "Why did the bridge-LoRA training crash last night?",
        "Write a Python function that parses train_log.jsonl",
        "Should we ship Track B P1 this session?",
        "The server is broken, /api/chat returns 500",
        "What's next for Paper 3?",
        "Brainstorm some ideas for the diverse corpus",
        "Summarize M72 findings",
        "hey",
        "What did we measure in M48?",
        "Explain the 1.7-dim consensus finding",
        "How does the dispatch work?",
    ]
    for m in tests:
        d = dispatch(m)
        sp = d["sample_profile"]
        print(f"{d['query_type']:20s} T={sp['temperature']:.2f} maxtok={sp['max_tokens']:<5} | {m}")
