"""Single source of truth for the grounding corpus union.

Main 40 P3: extracted from `tools/flag_anomalies.py` so the live writer
in `midas_ui.py` and the offline analyzers (`flag_anomalies`,
`session_quality_report`, future detectors) build identical grounding
context for any given turn. When a new grounding source appears (e.g.,
research result, fetched URL, file read tool result), update this
module once and every consumer picks it up automatically.

Two interfaces:

- `build_grounding_corpus(turn)`: takes the per-turn JSON dict shape
  (`{input, routing, retrieval, context, generation, ...}`) used by
  `flag_anomalies`. Convenience wrapper.

- `build_grounding_corpus_from_parts(mem_ctx_text, briefing_text,
  tool_result_preview)`: takes the raw components used by the live
  writer in `midas_ui.py` where the per-turn dict isn't fully
  assembled at detection time. This is the lower-level function;
  `build_grounding_corpus` calls it.

Both produce identical output for equivalent inputs.

Sources currently merged:
  - mem_ctx_text          — recalled memories list
  - briefing_text         — assembled briefing block
  - tool_result_preview   — most recent tool result (4000 chars)
  - SYSTEM_PROMPT         — CORE ARCHITECTURE block (Main 41)
  - measurement_registry  — canonical measurement values (Main 41)
"""

# Cache the system prompt and registry values as grounding sources.
# These are static per session, so loading once is fine.
_SYSTEM_PROMPT_CACHE = None
_REGISTRY_VALUES_CACHE = None


def _load_system_prompt():
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE
    try:
        import importlib, sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        from synthesizer import SYSTEM_PROMPT
        _SYSTEM_PROMPT_CACHE = SYSTEM_PROMPT
    except Exception:
        _SYSTEM_PROMPT_CACHE = ""
    return _SYSTEM_PROMPT_CACHE


def _load_registry_values():
    global _REGISTRY_VALUES_CACHE
    if _REGISTRY_VALUES_CACHE is not None:
        return _REGISTRY_VALUES_CACHE
    try:
        import json
        reg_path = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
                       / "data" / "measurement_registry.json")
        with open(reg_path) as f:
            reg = json.load(f)
        parts = []
        # M104: mirror of M102 narrative_retrieval fix. Some registry entries
        # are bare scalars (float/bool/str) written by pipeline tools (e.g.
        # tools/m96/m96_analyze.py) that bypass the dict schema. Skip them
        # here so a single malformed row can't abort grounding-corpus
        # assembly with `'float' object has no attribute 'get'`. Root cause
        # + full catalog in vault/agent_reports/m102_narrative_retrieval_fix.md.
        for key, entry in reg.items():
            if not isinstance(entry, dict):
                continue
            val = entry.get("value", "")
            unit = entry.get("unit", "")
            parts.append(f"{val} {unit}")
            parts.append(f"{val}{unit}")  # no-space variant for matching
        _REGISTRY_VALUES_CACHE = " ".join(parts)
    except Exception:
        _REGISTRY_VALUES_CACHE = ""
    return _REGISTRY_VALUES_CACHE


def build_grounding_corpus_from_parts(mem_ctx_text=None,
                                       briefing_text=None,
                                       tool_result_preview=None) -> str:
    """Return a single string containing the union of all grounded
    text the model saw for a turn. Empty inputs are tolerated.
    """
    parts = []
    if mem_ctx_text:
        for m in mem_ctx_text:
            if m is not None:
                parts.append(str(m))
    if briefing_text:
        parts.append(str(briefing_text))
    if tool_result_preview:
        parts.append(str(tool_result_preview))
    # Main 41: include system prompt (CORE ARCHITECTURE block) and
    # measurement registry values. Without these, the NOGROUND detector
    # false-flags canonical measurements cited from the system prompt
    # or registry injection.
    parts.append(_load_system_prompt())
    parts.append(_load_registry_values())
    return " ".join(parts)


def build_grounding_corpus(turn: dict) -> str:
    """Per-turn JSON dict interface. Used by `flag_anomalies` and
    other offline analyzers that work from logged turn files."""
    if not turn:
        return ""
    ctx = turn.get("context", {}) or {}
    rt = turn.get("routing", {}) or {}
    return build_grounding_corpus_from_parts(
        mem_ctx_text=ctx.get("mem_ctx_text"),
        briefing_text=ctx.get("briefing_text"),
        tool_result_preview=rt.get("tool_result_preview"),
    )
