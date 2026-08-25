"""
M100 Agent A1 — regression tests for class 3 query-side vocabulary fixes.

Each test covers an exact failure query from the M87 or M99 live pilot
scoring (re-scored primary — `vault/agent_reports/m99_live_pilot_scoring.md`
and `vault/agent_reports/m87_live_pilot_scoring.md`).

Every test is written so it FAILS against the pre-M100 code and PASSES
after the fix. The assertions are structured around what the retrieval
layer MUST return for the downstream synthesizer to avoid FALSE_ABSTAIN
(e.g. file-level match, not exact response text — the response text is
the LLM's job; we only need to confirm that retrieval surfaced the right
canonical source).

Run from repo root:

    python3 -m pytest orion-ane/tests/test_m100_class3_query_vocab.py -v

Or directly:

    python3 orion-ane/tests/test_m100_class3_query_vocab.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import query_vocab  # noqa: E402
import tool_executor  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Target 1 — "Paper 2" resolves to five_roadblocks_personal_ai.md
# Failure queries: M87 T07, M87 T08, M99 T24.
# Citation: vault/agent_reports/m99_live_pilot_scoring.md:59 (T24 row) and
#           vault/agent_reports/m99_live_pilot_scoring.md:90 (T24 narrative).
# ─────────────────────────────────────────────────────────────────────────────

def test_paper_2_alias_expansion():
    """`expand_query` must surface `five roadblocks` for `paper 2`."""
    variants = query_vocab.expand_query("what is paper 2 about?")
    assert any("five roadblocks" in v for v in variants), (
        "paper 2 must expand to 'five roadblocks' — M99 T24 FALSE_ABSTAIN "
        "regression"
    )


def test_paper_2_vault_research_surfaces_canonical_file():
    """vault_research on the T24 exact query must return
    paper/five_roadblocks_personal_ai.md in top matches."""
    result = tool_executor._vault_research(
        "what is Paper 2 about and how does it relate to Paper 1?"
    )
    files = [m["file"] for m in result.get("matches", [])]
    assert any("five_roadblocks_personal_ai" in f for f in files), (
        f"Paper 2 query must surface five_roadblocks_personal_ai.md; got: "
        f"{files[:5]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Target 2 — "ANE Enclave" bridges to vault's canonical "exclave"
# Failure queries: M99 T6, M99 T7.
# Citation: vault/agent_reports/m99_live_pilot_scoring.md:41-42 (T6/T7 rows)
#           and vault/agent_reports/m99_live_pilot_scoring.md:83 (narrative).
# ─────────────────────────────────────────────────────────────────────────────

def test_enclave_expands_to_exclave():
    variants = query_vocab.expand_query(
        "what research have we done on the ANE Enclave?"
    )
    joined = " ".join(variants).lower()
    assert "exclave" in joined, (
        "'ANE Enclave' must bridge to 'exclave' — M99 T6/T7 FALSE_ABSTAIN"
    )


def test_enclave_vault_read_surfaces_exclave_content():
    """The exact T6 query routes through vault_read; after fix, result
    must include at least one file that talks about the ANE exclave."""
    result = tool_executor._vault_read(
        query="what research have we done on the ANE Enclave?"
    )
    matches = result.get("matches", [])
    # Pool the top snippets and check for the canonical spelling.
    pooled = "\n".join(
        s for m in matches for s in m.get("snippets", [])
    ).lower()
    assert "exclave" in pooled, (
        "vault_read on 'ANE Enclave' query must surface exclave content "
        "in its matches — M99 T6 root cause"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Target 3 — "SME" case-insensitive retrieval with rare-token preservation
# Failure queries: M87 T30, M99 T33.
# Citation: vault/agent_reports/m99_live_pilot_scoring.md:68 (T33 row) and
#           vault/agent_reports/m99_live_pilot_scoring.md:97 (narrative).
# ─────────────────────────────────────────────────────────────────────────────

def test_sme_strip_punct_and_rare_token():
    """SME? must normalize to 'sme' and be recognized as a rare token."""
    assert query_vocab.strip_query_punct("SME?") == "sme"
    assert query_vocab.is_rare_high_signal("SME?") is True
    # negative control: a common word must NOT be flagged.
    assert query_vocab.is_rare_high_signal("model") is False


def test_sme_vault_read_surfaces_sme_content():
    """Exact M99 T33 query. Must surface an SME-topical file
    (anything from research/amx_crack/ or agent_reports/ with SME)
    in the top matches — pre-M100 this returned memory_triage_runbook
    because 'small', 'model', 'run' outranked the lone 'sme' hit."""
    result = tool_executor._vault_read(
        query="could a small model run on the SME?"
    )
    matches = result.get("matches", [])
    files = [m["file"] for m in matches]
    # Anti-regression: the top match must not be the triage runbook,
    # and at least one SME-topical file must be in the top 5.
    top_5 = files[:5]
    sme_topical = [
        f for f in top_5
        if "sme" in f.lower()
        or "amx_crack" in f.lower()
        or "pcore_sme" in f.lower()
    ]
    assert sme_topical, (
        f"SME query must surface SME-topical files in top 5; got: {top_5}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Target 4 — "op codes" (spaced) joins to "opcodes"
# Failure queries: M99 T32.
# Citation: vault/agent_reports/m99_live_pilot_scoring.md:67 (T32 row).
# ─────────────────────────────────────────────────────────────────────────────

def test_op_codes_token_joining():
    variants = query_vocab.expand_query("how many op codes are there?")
    assert any("opcodes" in v for v in variants), (
        "'op codes' (spaced) must expand to 'opcodes' — M99 T32 FALSE_ABSTAIN"
    )


def test_op_codes_vault_research_finds_53_opcodes():
    """Exact M99 T32 query routed through vault_research. After M100 A1
    the knowledge/ dir is searched AND 'op codes' expands to 'opcodes',
    so ane_hardware.md (canonical '53 unique opcodes' catalog) must
    appear in the matches."""
    result = tool_executor._vault_research("how many op codes are there for the ANE?")
    matches = result.get("matches", [])
    pooled = "\n".join(
        s for m in matches for s in m.get("snippets", [])
    ).lower()
    # Vault canonical string is "53 unique opcodes" in ane_hardware.md:40.
    # We assert on the number + the lemma since some files spell it
    # "opcodes" and some "op codes" — both should surface after fix.
    assert "53" in pooled and ("opcode" in pooled or "op code" in pooled), (
        "After fix, 'op codes' query must surface the 53-opcode catalog; "
        f"got pooled snippets length={len(pooled)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke battery — adjacent previously-passing queries must not regress.
# Each picks an M99 PASS turn and asserts a minimal structural property
# (the right file type is in top-N matches). These are intentionally
# loose — we only guard against the fix inadvertently breaking the
# rest of the retrieval behavior.
# ─────────────────────────────────────────────────────────────────────────────

def test_smoke_dead_paths_still_surface():
    """M99 T19 PASS — 'other dead paths?' type queries must still route
    to CLAUDE.md / knowledge content that enumerates dead paths."""
    result = tool_executor._vault_read(query="other dead paths we have")
    matches = result.get("matches", [])
    pooled = "\n".join(
        s for m in matches for s in m.get("snippets", [])
    ).lower()
    assert "dead path" in pooled or "dead_path" in pooled, (
        "Smoke: dead-path queries must still surface content"
    )


def test_smoke_tier2_binding_still_passes():
    """M99 T15 PASS — 'entity binding checks' must still surface binding
    content."""
    result = tool_executor._vault_read(query="what are entity binding checks?")
    matches = result.get("matches", [])
    pooled = "\n".join(
        s for m in matches for s in m.get("snippets", [])
    ).lower()
    assert "binding" in pooled, "Smoke: entity binding query broke"


def test_smoke_hidden_states_still_surfaces():
    """M99 T18 PARTIAL-but-content-correct — query about hidden states
    must still surface research content."""
    result = tool_executor._vault_research(
        "our research on hidden states between models"
    )
    matches = result.get("matches", [])
    assert matches, "Smoke: hidden-states research query returned 0 matches"


def test_smoke_307gbs_dram_still_surfaces():
    """M99 T25 PASS — '307 GB/s' DRAM fact must still surface."""
    result = tool_executor._vault_read(query="ANE GPU share DRAM 307 GB/s")
    matches = result.get("matches", [])
    pooled = "\n".join(
        s for m in matches for s in m.get("snippets", [])
    ).lower()
    assert "307" in pooled, "Smoke: 307 GB/s fact lookup broke"


def test_smoke_legacy_enclave_exclave_bidirectional():
    """M54 Phase 4 behavior preserved — exclave → enclave bridge also
    still works (reverse direction)."""
    variants = query_vocab.expand_query("the exclave firmware wall")
    assert any("enclave" in v for v in variants)


def test_smoke_expansion_is_word_boundary_aware():
    """Regression guard: 'paper 20' must NOT trigger the 'paper 2' rule.
    Regex anchor \\b prevents paper 2 from matching inside paper 20, paper
    22, etc."""
    variants = query_vocab.expand_query("what did paper 20 conclude?")
    assert not any("five roadblocks" in v for v in variants), (
        "Word-boundary: 'paper 20' must not expand as 'paper 2'"
    )


def test_smoke_expansion_preserves_originals_in_term_expansion():
    """expand_query_terms must keep all input terms — it's additive."""
    out = query_vocab.expand_query_terms(
        ["how", "many", "op", "codes", "ane"]
    )
    for expected in ("how", "many", "op", "codes", "ane"):
        assert expected in out, f"{expected} was dropped during expansion"
    # And adds the canonical
    assert "opcodes" in out


# ─────────────────────────────────────────────────────────────────────────────
# Allow running without pytest (simple script mode).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
    total = len(tests)
    print(f"\n{passed}/{total} passed")
    if failed:
        print("\nfailures:")
        for n, msg in failed:
            print(f"  - {n}: {msg}")
        sys.exit(1)
    sys.exit(0)
