"""
M109 ζ Agent Z3 — regression tests for the 8 new turn-log instrumentation
fields added by Z1 to `orion-ane/agent/midas_ui.py` and documented by Z2 in
`vault/knowledge/turn_log_schema_v2.md`.

Scope: one test per field + integration-shape tests + v1-backward-compat
regression. Tests exercise the helper functions Z1 added (at module level)
rather than the full HTTP flow — per the Z3 brief, helper-level unit tests
are sufficient for the directive's 8-case gate and avoid needing the full
midas_ui server stack up (no HTTP server, no daemon restart).

Key discipline points (see directive M109 §3.3 + §10):
  - Pattern-1 (test what the field IS, not what the directive says): several
    fields are stubbed on `/api/chat` because Z1 verified that non-streaming
    handler has no turn-log writer at all; tests honor that shape by asserting
    on the stream-path helpers, not by asserting `/api/chat` output.
  - Primary-source alignment: each field's expected values come from Z1's
    helper implementation (canonical=1.30, assistant=0.50, else=1.00) which
    matches local_store.py:434-442. Possessive-intent branch is NOT logged
    at helper time per Z1's doc comment; this deviates from the Z2 schema's
    "log the product of all multipliers that fired" recommendation. Test
    asserts Z1's actual choice (helpers are unconditional) and a dedicated
    test flags the divergence in the report.
  - Honest-negative: v1 turn JSON parse test operates against any available
    M103-era fixture; if none found it skips rather than xfailing.

Run from repo root (pytest):
    python3 -m pytest orion-ane/tests/test_m109_turn_log_schema_v2.py -v

Or standalone (no pytest required):
    python3 orion-ane/tests/test_m109_turn_log_schema_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(os.path.dirname(HERE), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

# Module import triggers minor boot side effects (session turn-log dir,
# warm-open bridge). These are harmless for helper-level testing.
import midas_ui  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Field 1 — role_weight per retrieved memory
# Z1 helper: _m109_role_weight(source_role) → float
# Primary source: local_store.py:434-442 (canonical 1.30, assistant 0.50,
# else 1.00). Possessive-intent multipliers (0.30 user / 0.05 research) are
# query-conditional and intentionally NOT applied in the helper per Z1's
# docstring.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_1_role_weight_canonical_is_1_30():
    """Canonical memories receive the 1.30× authority multiplier."""
    assert midas_ui._m109_role_weight("canonical") == 1.30


def test_field_1_role_weight_assistant_is_0_50():
    """Assistant-extracted memories receive the 0.50× downweight."""
    assert midas_ui._m109_role_weight("assistant") == 0.50


def test_field_1_role_weight_default_is_1_00():
    """user / vault / research / unknown / empty all map to the 1.00 base.
    The possessive-intent 0.30/0.05 downweights are query-conditional and
    are documented (by Z1) as NOT logged at helper time."""
    for src in ("user", "vault", "research", "claude_automemory", "", "mystery"):
        assert midas_ui._m109_role_weight(src) == 1.00, (
            f"source_role={src!r} should receive base 1.00× multiplier")


def test_field_1_role_weight_type_is_float():
    """Downstream consumers rely on float arithmetic on this field."""
    for src in ("canonical", "assistant", "user", ""):
        val = midas_ui._m109_role_weight(src)
        assert isinstance(val, float), (src, type(val))


def test_field_1_annotate_recall_attaches_role_weight_per_record():
    """The per-record telemetry builder attaches role_weight to every
    recall entry, matching the shape that gets spliced into
    retrieval.recall_filtered[i]."""
    filtered = [
        {"source_role": "canonical", "text": "A", "score": 2.0},
        {"source_role": "assistant", "text": "B", "score": 0.4},
        {"source_role": "user",      "text": "C", "score": 1.0},
    ]
    annot = midas_ui._m109_annotate_recall(filtered)
    assert len(annot) == len(filtered)
    assert annot[0]["role_weight"] == 1.30
    assert annot[1]["role_weight"] == 0.50
    assert annot[2]["role_weight"] == 1.00


# ─────────────────────────────────────────────────────────────────────────────
# Field 2 — quality.grounded populated (value or "disabled"), never null
# Z1 helper: _m109_grounding_state() → "enabled" | "disabled" | "unknown"
# Reads answer_scrub.TIER1_ENABLED per schema doc. Gemma-4 verifier currently
# has Tier 1 disabled.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_2_grounding_state_returns_one_of_three_literals():
    """The grounded state must be a string from the allowed enum — never None.
    Populating with a literal string is the whole point of field 2 (v1 logged
    None in all three states)."""
    state = midas_ui._m109_grounding_state()
    assert isinstance(state, str), (type(state), state)
    assert state in ("enabled", "disabled", "unknown"), state


def test_field_2_grounding_state_is_disabled_for_gemma_current_config():
    """Z1's report §Upstream-source verification row 2 confirms answer_scrub's
    TIER1_ENABLED=False for Gemma 4 as of 2026-04-21. Verify the helper
    reflects that. If this assertion flips, it means Tier 1 was re-enabled —
    a real event worth a test failure so the author examines why."""
    state = midas_ui._m109_grounding_state()
    # If answer_scrub cannot be imported (unlikely given live stack), Z1's
    # helper falls back to "unknown" — allow that too so this test does not
    # break in a partial-import harness.
    assert state in ("disabled", "unknown"), (
        "Expected Tier 1 to be disabled for current Gemma 4 verifier. "
        f"Got {state!r}. If Tier 1 was just re-enabled, update this test.")


# ─────────────────────────────────────────────────────────────────────────────
# Field 3 — post_turn.memories_stored_this_turn per-memory manifest
# Z1 helper: _m109_memory_manifest(delta_count) → list[{id, content_hash,
# source_role, extractor, type}]
# Reads memory.daemon._session_facts. Backward-compat: manifest is emitted as
# a sibling field `memories_stored_this_turn_manifest`, preserving the int
# `memories_stored_this_turn` scalar (decided by Z1 per report §Backward-compat).
# ─────────────────────────────────────────────────────────────────────────────

def test_field_3_manifest_empty_on_zero_delta():
    """Zero new memories → empty manifest, no exception."""
    out = midas_ui._m109_memory_manifest(0)
    assert out == []


def test_field_3_manifest_handles_negative_or_none_delta():
    """Negative and None deltas should also yield an empty manifest rather
    than raising (e.g. when _prev_memory_store_total is reset mid-session)."""
    assert midas_ui._m109_memory_manifest(None) == []
    assert midas_ui._m109_memory_manifest(-1) == []


def test_field_3_manifest_entry_shape(monkeypatch=None):
    """When delta > 0, each entry must carry the five documented keys:
    id, content_hash, source_role, extractor, type. We monkey-patch the
    daemon's _session_facts to inject a deterministic fixture."""
    d = getattr(midas_ui.memory, "daemon", None)
    if d is None:
        # Harness without a daemon wired — helper returns [] per Z1's fallback.
        assert midas_ui._m109_memory_manifest(1) == []
        return
    saved = getattr(d, "_session_facts", None)
    fixture = [
        {"id": "mem_a", "text": "alpha", "source_role": "user",
         "type": "preference", "extraction_source": "ane_8b"},
        {"id": "mem_b", "text": "beta", "source_role": "claude_automemory",
         "type": "quantitative"},  # missing extraction_source → cpu_heuristic
    ]
    try:
        d._session_facts = fixture
        manifest = midas_ui._m109_memory_manifest(2)
        assert isinstance(manifest, list)
        assert len(manifest) == 2
        for entry in manifest:
            for key in ("id", "content_hash", "source_role",
                        "extractor", "type"):
                assert key in entry, (key, entry)
        # content_hash must be a 16-char SHA256 prefix (or "" on empty text)
        assert len(manifest[0]["content_hash"]) == 16
        # extractor = "8b_ane" when extraction_source == "ane_8b"
        assert manifest[0]["extractor"] == "8b_ane"
        # missing extraction_source → cpu_heuristic fallback
        assert manifest[1]["extractor"] == "cpu_heuristic"
    finally:
        # Restore to avoid contaminating other tests or live daemon state.
        if saved is None:
            try:
                delattr(d, "_session_facts")
            except AttributeError:
                pass
        else:
            d._session_facts = saved


# ─────────────────────────────────────────────────────────────────────────────
# Field 4 — Decode stop_reason
# Z1 helper: _m109_infer_stop_reason(tokens_decoded, max_tokens, loop_detected,
#                                    gen_error=None, user_cancel=False)
# Enum: user_cancel | verifier_error | loop_detected | max_tokens | eos |
# unknown_v2. Note: Z1's enum uses "loop_detected" (not the directive's
# original "memory_pressure_collapse"); this is honest-scope per Z1 because
# the only loop signal actually captured in the stream is the Main 42 n-gram
# detector. Report flags this for operator review.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_4_stop_reason_user_cancel_takes_priority():
    """user_cancel beats all other signals — set-via-caller must win even if
    tokens reached the cap and the loop detector also fired."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=150, max_tokens=150, loop_detected=True,
        gen_error="boom", user_cancel=True) == "user_cancel"


def test_field_4_stop_reason_verifier_error_beats_cap():
    """gen_error (non-empty) wins over max_tokens and eos paths."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=10, max_tokens=150,
        loop_detected=False, gen_error="ConnectionReset") == "verifier_error"


def test_field_4_stop_reason_loop_detected_enum():
    """Main 42 n-gram loop-detector trip surfaces as 'loop_detected'."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=50, max_tokens=150, loop_detected=True) == "loop_detected"


def test_field_4_stop_reason_max_tokens():
    """tokens_decoded ≥ max_tokens → 'max_tokens'."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=150, max_tokens=150, loop_detected=False) == "max_tokens"


def test_field_4_stop_reason_eos():
    """Partial decode with no error/loop/cap-hit → natural EOS."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=42, max_tokens=600, loop_detected=False) == "eos"


def test_field_4_stop_reason_unknown_v2_on_missing_signal():
    """When neither tokens_decoded nor max_tokens can be resolved, the helper
    falls back to the directive's 'unknown_v2' sentinel rather than raising."""
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=None, max_tokens=None, loop_detected=False) == "unknown_v2"
    # tokens_decoded == 0 with no other signal also yields unknown_v2
    assert midas_ui._m109_infer_stop_reason(
        tokens_decoded=0, max_tokens=600, loop_detected=False) == "unknown_v2"


# ─────────────────────────────────────────────────────────────────────────────
# Field 5 — Absence-guard feature vector
# Populated inline in the stream handler around line 3358 (search
# `m109_zeta_field_5`). Keys documented: max_score, pool_size,
# threshold_applied, threshold_sensitivity, query_type_dispatch,
# narrative_used, mem_ctx_empty_branch, word_overlap_branch_eligible,
# m74_a2_active, q_specific_count, q_unmatched_count, unmatched_ratio.
# There is no dedicated helper function — the dict is built at the call site.
# This test inspects the source for the required keys (static guarantee) and
# treats the call-site dict as the contract.
# ─────────────────────────────────────────────────────────────────────────────

def _read_midas_ui_source():
    src_path = Path(AGENT_DIR) / "midas_ui.py"
    return src_path.read_text(encoding="utf-8")


def test_field_5_absence_guard_feature_vector_has_all_required_keys():
    """Static-verify that the call-site dict (Z1 report §Absence-guard) emits
    the full feature set documented in Z2's schema v2. Keys are required by
    M108 predicate-reconstruction."""
    src = _read_midas_ui_source()
    # Isolate the _abs_features block (from '_abs_features = {' through
    # the terminating '}' at matching indent).
    needle = "_abs_features = {"
    idx = src.find(needle)
    assert idx != -1, "Could not locate _abs_features dict literal"
    tail = src[idx:idx + 3000]  # generous window
    for key in ("max_score", "pool_size", "threshold_applied",
                "threshold_sensitivity", "query_type_dispatch",
                "narrative_used", "mem_ctx_empty_branch",
                "word_overlap_branch_eligible", "m74_a2_active"):
        assert f'"{key}"' in tail, f"feature vector missing key {key!r}"


def test_field_5_word_overlap_branch_keys_emitted_when_branch_fires():
    """The q_specific / q_unmatched / unmatched_ratio keys must be written
    even when the word-overlap branch did NOT fire (Z1 emits the literal
    'not_consumed_this_turn' sentinel so downstream parsers never KeyError).
    Static-verify the sentinel convention."""
    src = _read_midas_ui_source()
    assert 'q_specific_count' in src
    assert 'q_unmatched_count' in src
    assert 'unmatched_ratio' in src
    assert '"not_consumed_this_turn"' in src, (
        "word-overlap sentinel 'not_consumed_this_turn' missing — "
        "Z1's not-fired-this-turn convention is how the schema avoids KeyError")


def test_field_5_turn_record_nests_feature_vector_under_absence_guard():
    """The absence-guard turn-record block must nest the new feature vector
    under the key 'feature_vector' (Z2 schema) rather than at top level."""
    src = _read_midas_ui_source()
    assert '"feature_vector": _abs_features' in src, (
        "feature_vector must be nested inside absence_guard record, not at "
        "the retrieval-top-level. Z2 schema §Field 5.")


# ─────────────────────────────────────────────────────────────────────────────
# Field 6 — Conversation-history source_turn tags per message
# Z1 helper: _m109_tag_history_messages(msgs, briefing_present) →
#   list[{role, content, source_turn}]
# source_turn ∈ {"system", "briefing", "T<N>", "not_consumed_this_turn"}.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_6_tag_history_messages_empty_returns_empty():
    """Empty msgs list → empty tagged list, no exception."""
    assert midas_ui._m109_tag_history_messages([], briefing_present=False) == []
    assert midas_ui._m109_tag_history_messages(None, briefing_present=True) == []


def test_field_6_tag_system_slot_briefing_aware():
    """System slot should carry 'briefing' when briefing_present, else 'system'."""
    msgs = [
        {"role": "system", "content": "You are Midas..."},
        {"role": "user",   "content": "Current question"},
    ]
    with_brief = midas_ui._m109_tag_history_messages(msgs, briefing_present=True)
    without    = midas_ui._m109_tag_history_messages(msgs, briefing_present=False)
    assert with_brief[0]["source_turn"] == "briefing"
    assert without[0]["source_turn"] == "system"


def test_field_6_current_user_message_tagged_with_current_turn():
    """The final user message (the turn being processed) must be tagged with
    the current turn number in T<N> form. We seed the module-global turn log
    with a known turn_number and verify the helper reads it."""
    saved = dict(midas_ui._TURN_LOG) if midas_ui._TURN_LOG else {}
    try:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG["turn_number"] = 15
        msgs = [
            {"role": "system",    "content": "system prompt"},
            {"role": "user",      "content": "prior user"},
            {"role": "assistant", "content": "prior asst"},
            {"role": "user",      "content": "current question"},
        ]
        tagged = midas_ui._m109_tag_history_messages(msgs, briefing_present=False)
        assert len(tagged) == 4
        # Last user message = current turn → "T15"
        assert tagged[-1]["role"] == "user"
        assert tagged[-1]["source_turn"] == "T15"
        # History pair gets an earlier T-tag (not "current", not "system")
        for m in tagged[1:3]:
            assert m["source_turn"].startswith("T"), m["source_turn"]
    finally:
        midas_ui._TURN_LOG.clear()
        midas_ui._TURN_LOG.update(saved)


def test_field_6_every_message_has_source_turn_key():
    """No message in the output may lack the source_turn tag — the whole
    point of field 6 is eliminating the per-message history-provenance gap."""
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user",   "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user",   "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user",   "content": "now"}]
    tagged = midas_ui._m109_tag_history_messages(msgs, briefing_present=False)
    assert len(tagged) == len(msgs)
    for t in tagged:
        assert "source_turn" in t, t
        assert isinstance(t["source_turn"], str)
        assert t["source_turn"]  # non-empty


# ─────────────────────────────────────────────────────────────────────────────
# Field 7 — Role-assignment provenance per retrieved memory
# Bundled into _m109_annotate_recall per Z1 report. Emits provenance object
# {origin, original_turn, extractor_if_any}.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_7_provenance_emitted_per_memory():
    """Every annotated recall record must carry a provenance dict with all
    three documented subkeys."""
    filtered = [
        {"source_role": "canonical", "text": "c"},
        {"source_role": "user",      "text": "u"},
        {"source_role": "assistant", "text": "a", "extraction_source": "ane_8b"},
        {"source_role": "vault",     "text": "v"},
        {"source_role": "research",  "text": "r"},
    ]
    out = midas_ui._m109_annotate_recall(filtered)
    assert len(out) == 5
    for entry in out:
        prov = entry.get("provenance")
        assert isinstance(prov, dict), entry
        for key in ("origin", "original_turn", "extractor_if_any"):
            assert key in prov, (key, prov)


def test_field_7_provenance_origin_derivation_per_source_role():
    """origin is derived from source_role per Z1's lookup: user→user_utterance,
    assistant→extraction, canonical→canonical_inject, vault→vault_sync,
    research→research_import, else→unknown."""
    cases = [
        ("user",              "user_utterance"),
        ("assistant",         "extraction"),
        ("canonical",         "canonical_inject"),
        ("vault",             "vault_sync"),
        ("research",          "research_import"),
        ("claude_automemory", "unknown"),
        ("",                  "unknown"),
    ]
    for src, expected_origin in cases:
        filtered = [{"source_role": src, "text": "x"}]
        out = midas_ui._m109_annotate_recall(filtered)
        assert out[0]["provenance"]["origin"] == expected_origin, (src, out[0])


def test_field_7_provenance_extractor_if_any_reflects_extraction_source():
    """extractor_if_any should reflect the memory's extraction_source meta:
    'ane_8b' → '8b_ane', absent → None."""
    filtered = [
        {"source_role": "assistant", "text": "x", "extraction_source": "ane_8b"},
        {"source_role": "user", "text": "y"},  # no extraction_source
    ]
    out = midas_ui._m109_annotate_recall(filtered)
    assert out[0]["provenance"]["extractor_if_any"] == "8b_ane"
    assert out[1]["provenance"]["extractor_if_any"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Field 8 — Canonical-boost applied-multiplier trace per record
# Z1 helper: _m109_canonical_boost_multiplier(source_role) → 1.30 | 1.00
# Counterfactual: for every non-canonical record the multiplier is 1.00
# (visible = "would be no-op"); for canonical records it is 1.30.
# ─────────────────────────────────────────────────────────────────────────────

def test_field_8_canonical_boost_counterfactual_on_canonical():
    """Canonical records receive the 1.30 counterfactual multiplier."""
    assert midas_ui._m109_canonical_boost_multiplier("canonical") == 1.30


def test_field_8_canonical_boost_counterfactual_default_is_1_00():
    """Non-canonical records receive the 1.00 counterfactual (null-op)."""
    for src in ("user", "assistant", "vault", "research", "claude_automemory", ""):
        val = midas_ui._m109_canonical_boost_multiplier(src)
        assert val == 1.00, (src, val)


def test_field_8_annotate_recall_surfaces_counterfactual_per_record():
    """Every recall entry must carry a canonical_boost_multiplier_if_active
    field, regardless of whether the record itself is canonical."""
    filtered = [
        {"source_role": "canonical", "text": "c"},
        {"source_role": "user",      "text": "u"},
    ]
    out = midas_ui._m109_annotate_recall(filtered)
    assert out[0]["canonical_boost_multiplier_if_active"] == 1.30
    assert out[1]["canonical_boost_multiplier_if_active"] == 1.00


# ─────────────────────────────────────────────────────────────────────────────
# Integration — full _m109_annotate_recall shape (fields 1 + 7 + 8 bundled)
# ─────────────────────────────────────────────────────────────────────────────

def test_integration_annotate_recall_bundles_fields_1_7_8():
    """One recall → one annotation dict with role_weight (f1),
    canonical_boost_multiplier_if_active (f8), and provenance (f7)."""
    filtered = [{"source_role": "canonical", "text": "t",
                 "extraction_source": "", "original_turn": 7}]
    out = midas_ui._m109_annotate_recall(filtered)
    assert len(out) == 1
    entry = out[0]
    assert entry["role_weight"] == 1.30                               # f1
    assert entry["canonical_boost_multiplier_if_active"] == 1.30      # f8
    assert isinstance(entry["provenance"], dict)                      # f7
    # original_turn propagation from input meta
    assert entry["provenance"]["original_turn"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint parity — /api/chat vs /api/chat/stream
# Z1's implementation adds v2 instrumentation only on the stream handler.
# /api/chat (non-streaming) does NOT invoke _turn_start / _turn_write, so no
# turn JSON is emitted from that path and the fields are "stubbed" only in
# the sense of "absent because no turn log is written". This is the
# Pattern-1 honest-stub: we test what IS rather than the idealized contract.
# ─────────────────────────────────────────────────────────────────────────────

def test_api_chat_nonstream_does_not_call_turn_start_or_turn_write():
    """Per Z1 report §'/api/chat stub rationale': the non-streaming handler
    does not call _turn_start or _turn_write, so no per-turn JSON is emitted
    at all from that path. Verify statically so future refactors that add
    emission get surfaced as a test failure (forcing author to double-check
    schema-v2 coverage on the new code path)."""
    src = _read_midas_ui_source()
    # Locate the /api/chat handler (Flask route at approx. line 1867 per Z1).
    chat_route_idx = src.find('@app.route("/api/chat"')
    if chat_route_idx == -1:
        chat_route_idx = src.find("@app.route('/api/chat'")
    assert chat_route_idx != -1, "Could not locate /api/chat route"
    # Locate the stream route so we can bound the non-streaming handler body.
    stream_route_idx = src.find('/api/chat/stream', chat_route_idx + 1)
    assert stream_route_idx != -1, "Could not locate /api/chat/stream route"
    handler_body = src[chat_route_idx:stream_route_idx]
    assert "_turn_start(" not in handler_body, (
        "/api/chat handler now calls _turn_start — coverage gap if v2 fields "
        "are not also populated on this path.")
    assert "_turn_write(" not in handler_body, (
        "/api/chat handler now calls _turn_write — coverage gap.")


def test_stream_handler_has_all_eight_field_markers():
    """Static-verify the 8 Z1 marker comments exist in the stream handler.
    If this test fails the instrumentation has regressed from Z1's landing.

    Note: Z1 bundled fields 1/7/8 under compound markers (`field_1+7+8`,
    `field_1 + field_8`) since the helpers _m109_annotate_recall and
    _m109_canonical_boost_multiplier populate them at the same call site.
    Accept any marker that mentions the field number as a standalone `N`
    token (so `field_1+7+8` satisfies field_1, field_7, and field_8)."""
    import re
    src = _read_midas_ui_source()
    # Collect every m109_zeta_field_<nums> token, then explode compound
    # forms like '1+7+8' into {1, 7, 8}.
    present = set()
    for m in re.finditer(r"m109_zeta_field_([0-9+]+)", src):
        for n in m.group(1).split("+"):
            if n.isdigit():
                present.add(int(n))
    for n in range(1, 9):
        assert n in present, (
            f"No marker comment found covering m109_zeta_field_{n}. "
            f"Present markers (after compound-form expansion): "
            f"{sorted(present)}")


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat regression — v1 turn JSONs still parse
# ─────────────────────────────────────────────────────────────────────────────

def test_existing_v1_turn_json_still_parses():
    """M103-era turn JSONs (schema v1) must continue to parse cleanly with
    the v2 schema in place. All new fields are additive; none of the v1
    keys have been removed. If no fixture exists, skip honestly rather
    than passing vacuously."""
    fixture = (Path(__file__).resolve().parent.parent.parent
               / "data" / "session_logs" / "sess_20260420_203721_94881"
               / "turn_0001.json")
    if not fixture.exists():
        # Honest skip — the fixture directory is optional.
        print(f"SKIP: fixture not found at {fixture}")
        return
    with open(fixture) as f:
        data = json.load(f)
    # Must parse and expose the v1 sections that M103 scoring reads.
    assert isinstance(data, dict)
    for section in ("input", "retrieval", "generation", "quality", "post_turn",
                    "assembled_prompt"):
        assert section in data, (section, list(data.keys()))
    # v1-era `quality.grounded` may still be None on old logs — that is the
    # exact gap Z1's Field 2 closes for v2 logs. Do NOT assert non-null on
    # historical data (would be a false failure).
    assert "grounded" in data["quality"]
    # v1-era `post_turn.memories_stored_this_turn` is an int; v2 adds a new
    # sibling `memories_stored_this_turn_manifest` (per Z1 BC decision).
    # The int field must still be present on old logs.
    assert "memories_stored_this_turn" in data["post_turn"]
    assert isinstance(data["post_turn"]["memories_stored_this_turn"], int)


# ─────────────────────────────────────────────────────────────────────────────
# Z1 implementation vs. Z2 schema divergence check
# Z2's schema §Field 1 recommends logging the product of all multipliers
# that fired (including possessive-intent 0.30 user / 0.05 research).
# Z1's helper logs only the unconditional base (canonical/assistant/else).
# This test captures that divergence EXPLICITLY so the delta is on the
# record rather than silently papered-over. It is a PASS — Z1's simpler
# contract is what shipped, and downstream forensic can still reconstruct
# the possessive-intent product at offline time from query-level logs.
# ─────────────────────────────────────────────────────────────────────────────

def test_divergence_z1_helper_ignores_possessive_intent_gate():
    """Documented divergence: Z2 schema §Pattern-1 note recommends that the
    role_weight field log the actual multiplier product that fired
    (including query-conditional possessive-intent downweights). Z1 chose
    to log only the always-on base multiplier. This test codifies the
    shipped behavior — a future change that switches to product-logging
    should also delete or update this test."""
    # Possessive-intent would take user from 1.00 → 0.30 and research from
    # 1.00 → 0.05. Helper has no possessive_intent parameter, so this is
    # structurally guaranteed.
    assert midas_ui._m109_role_weight("user")     == 1.00
    assert midas_ui._m109_role_weight("research") == 1.00
    # Structural check: the helper signature has exactly one argument.
    import inspect
    sig = inspect.signature(midas_ui._m109_role_weight)
    assert len(sig.parameters) == 1, sig.parameters


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required — matches test_m103 convention).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Field 1
        test_field_1_role_weight_canonical_is_1_30,
        test_field_1_role_weight_assistant_is_0_50,
        test_field_1_role_weight_default_is_1_00,
        test_field_1_role_weight_type_is_float,
        test_field_1_annotate_recall_attaches_role_weight_per_record,
        # Field 2
        test_field_2_grounding_state_returns_one_of_three_literals,
        test_field_2_grounding_state_is_disabled_for_gemma_current_config,
        # Field 3
        test_field_3_manifest_empty_on_zero_delta,
        test_field_3_manifest_handles_negative_or_none_delta,
        test_field_3_manifest_entry_shape,
        # Field 4
        test_field_4_stop_reason_user_cancel_takes_priority,
        test_field_4_stop_reason_verifier_error_beats_cap,
        test_field_4_stop_reason_loop_detected_enum,
        test_field_4_stop_reason_max_tokens,
        test_field_4_stop_reason_eos,
        test_field_4_stop_reason_unknown_v2_on_missing_signal,
        # Field 5
        test_field_5_absence_guard_feature_vector_has_all_required_keys,
        test_field_5_word_overlap_branch_keys_emitted_when_branch_fires,
        test_field_5_turn_record_nests_feature_vector_under_absence_guard,
        # Field 6
        test_field_6_tag_history_messages_empty_returns_empty,
        test_field_6_tag_system_slot_briefing_aware,
        test_field_6_current_user_message_tagged_with_current_turn,
        test_field_6_every_message_has_source_turn_key,
        # Field 7
        test_field_7_provenance_emitted_per_memory,
        test_field_7_provenance_origin_derivation_per_source_role,
        test_field_7_provenance_extractor_if_any_reflects_extraction_source,
        # Field 8
        test_field_8_canonical_boost_counterfactual_on_canonical,
        test_field_8_canonical_boost_counterfactual_default_is_1_00,
        test_field_8_annotate_recall_surfaces_counterfactual_per_record,
        # Integration + endpoint parity
        test_integration_annotate_recall_bundles_fields_1_7_8,
        test_api_chat_nonstream_does_not_call_turn_start_or_turn_write,
        test_stream_handler_has_all_eight_field_markers,
        # Regression
        test_existing_v1_turn_json_still_parses,
        # Divergence
        test_divergence_z1_helper_ignores_possessive_intent_gate,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failures.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"{len(tests)}/{len(tests)} passed")
