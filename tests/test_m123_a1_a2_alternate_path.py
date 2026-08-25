"""M123 Stream A1 — A2 alternate-pathway gap close.

Authoritative spec:
    vault/directives/in_progress/2026-04-22T20-48-04_m123_m123-fix-completion-m122-findings-and-se.md §3.1
    vault/agent_reports/m122_a2_tier2_refinement.md §3 (binding-path guard surface)
    vault/agent_reports/m122_c_pilot_synthesis.md §2.5 (T59 alternate-path finding)

Mechanism anchor:
    M122 C T59 — pre-scrub response was FACTUALLY CORRECT
    ("EAGLE-3 had a 0% acceptance rate on all quantized models, including
    Q3 and Q4"). Tier 2 binding-check stripped the sentence and Tier 2
    registry-conflict-check emitted a `q4.dequant_penalty=31%` template
    inline via `annotate_conflict_flags`. The stripped first sentence
    displaced the span offsets, so the template text leaked into the
    middle of the second sentence. Result: novel `inline_confab_guard_
    leak` shape (M122 C K14 fire).

    The M122 A2 bare-scalar + abstention guards wrap `tier2_binding_
    check` only. `tier2_registry_conflict_check` emits flags without
    guard evaluation — the alternate path this stream closes.

Fix shipped (M123 A1):
  1. `tier2_registry_conflict_check` now returns (flags, skips) and
     runs a topic-disambiguation variant of the bare-scalar guard
     (plus unchanged abstention guard) before emitting a flag. Helper:
     `_m123_evaluate_conflict_path_guards` (answer_scrub.py). Duplicate
     of M122 A2's helper with adapted semantics because the conflict
     path pre-filters to sentences that mention the entity — binding-
     path's entity-absence bare-scalar check fires on every conflict-
     path call by construction.
  2. `tier2_binding_skip_reasons` list (emitted to ζ v2.2 via
     `scrub.tier2_binding.skip_reason`) now carries dicts with a
     `path` field: `"binding"` for M122 A2 fires, `"conflict"` for the
     new M123 A1 fires. Backward-compatible: existing `skip_reason`
     field still populated on every entry.

Test coverage:
  T59 replay                — conflict flag suppressed, no inline template leak
  T82 regression            — binding-path guard still preserves correct answer
  T68 regression            — abstention guard still preserves abstention
  Honest-abstention variant — abstention phrase on conflict-path claim suppresses flag
  M108/M117 corpus          — 5 historical binding-path cases still preserved
  Neuron throughput case    — M120 C conflict flag still fires when entity is topic
  M120 C NAX agreement      — no flag when registry agrees with claim
  Schema annotation         — skip entries carry {path, reason, skip_reason}

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m123_a1_a2_alternate_path.py

Registry values produced:
    m123.a1.verdict
    m123.a1.alternate_paths_covered
    m123.a1.t59_inline_leak_eliminated
    m123.a1.binding_path_regression_preserved
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from answer_scrub import (  # noqa: E402
    _m122_evaluate_skip_guards,
    _m123_evaluate_conflict_path_guards,
    annotate_conflict_flags,
    scrub_response,
    tier2_binding_check,
    tier2_registry_conflict_check,
)


# ──────────────────────────────────────────────────────────────────────
# Assertion helpers
# ──────────────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0
_FAILURES = []


def _check(label, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {label}")
    else:
        _FAIL += 1
        _FAILURES.append((label, detail))
        print(f"  FAIL  {label}: {detail}")


# ──────────────────────────────────────────────────────────────────────
# Test 1: T59 replay — conflict flag suppressed, template not leaked
# ──────────────────────────────────────────────────────────────────────

def test_t59_replay_no_inline_template_leak():
    """M122 C T59 smoking gun: EAGLE-3 0% acceptance response had a
    conflict flag emitted for `q4.dequant_penalty=31%` against the
    claim `0%`. Inline annotation leaked the template text into the
    response body.

    Under M123 A1, the conflict-path topic-disambiguation guard fires
    (registry tokens {q4, 4, bit, dequant, penalty} do not overlap the
    ±5-word window {eagle, acceptance, rate, all, quantized} around
    `0%`), suppressing the flag and preventing the inline template
    leak."""
    print("\n[Test 1] T59 replay — no inline template leak")
    response = (
        "EAGLE-3 had a 0% acceptance rate on all quantized models, "
        "including Q3 and Q4.")

    conflict_flags, conflict_skips = tier2_registry_conflict_check(
        response)
    # The q4.dequant_penalty flag should be suppressed.
    q4_flags = [f for f in conflict_flags
                if f.get("registry_key") == "q4.dequant_penalty"]
    _check("T59 q4.dequant_penalty conflict flag suppressed",
           len(q4_flags) == 0,
           f"expected 0 q4.dequant_penalty flags, got {q4_flags}")

    # Skip must be recorded with path="conflict".
    q4_skips = [s for s in conflict_skips
                if s.get("registry_key") == "q4.dequant_penalty"]
    _check("T59 q4.dequant_penalty skip recorded on conflict path",
           len(q4_skips) >= 1,
           f"expected ≥1 q4 skip, got {q4_skips}")
    if q4_skips:
        _check("T59 skip path annotation = conflict",
               q4_skips[0].get("path") == "conflict",
               f"got path={q4_skips[0].get('path')}")
        _check("T59 skip_reason = word_overlap_zero",
               q4_skips[0].get("skip_reason") == "word_overlap_zero",
               f"got skip_reason={q4_skips[0].get('skip_reason')}")
        # Backward-compat field.
        _check("T59 skip carries legacy `reason` field",
               q4_skips[0].get("reason") == q4_skips[0].get("skip_reason"),
               f"got reason={q4_skips[0].get('reason')}")

    # Full scrub_response path: no inline template in cleaned output.
    result = scrub_response(response, grounding_corpus="",
                            user_query="EAGLE-3 acceptance rate?")
    cleaned = result.get("cleaned_response") or ""
    _check("T59 cleaned response has NO inline conflict template",
           "measurement registry records different value" not in cleaned,
           f"inline template leaked: {cleaned!r}")
    _check("T59 skip_reason list present in scrub result",
           isinstance(result.get("tier2_binding_skip_reasons"), list),
           f"got {type(result.get('tier2_binding_skip_reasons'))}")
    skip_reasons = result.get("tier2_binding_skip_reasons", []) or []
    # Alternate-path entry with path="conflict" should be in the list.
    conflict_path_entries = [s for s in skip_reasons
                             if isinstance(s, dict)
                             and s.get("path") == "conflict"]
    _check("T59 ζ v2.2 skip_reason list includes path=conflict entry",
           len(conflict_path_entries) >= 1,
           f"got skip_reasons={skip_reasons}")


# ──────────────────────────────────────────────────────────────────────
# Test 2: T82 regression — binding-path guard still fires cleanly
# ──────────────────────────────────────────────────────────────────────

def test_t82_binding_path_regression_preserved():
    """M122 A2 T82: correct answer preservation via bare-scalar guard
    on the binding path. M123 A1 must not regress this — the binding
    path's guard semantics are unchanged, and its skip entries now
    carry `path="binding"` for downstream distinction."""
    print("\n[Test 2] T82 binding-path regression preserved")
    response = (
        "The root cause was a stream ordering issue in "
        "`qwen_spec_decode_server.py`, which accounted for 19/20 "
        "failures and dropped the clean rate to 5% before being fixed.")
    grounding = (
        "Main 42 Closing Report: session started with a 5% clean rate "
        "on the 20-turn diagnostic. qwen_spec_decode_server.py stream "
        "ordering fix. 19/20 failures on pre-fix battery.")

    flags, skips = tier2_binding_check(response, grounding)
    _check("T82 binding-path 0 flags",
           len(flags) == 0,
           f"expected 0 flags, got {flags}")
    _check("T82 binding-path 1 skip",
           len(skips) == 1,
           f"expected 1 skip, got {skips}")
    if skips:
        _check("T82 skip path annotation = binding",
               skips[0].get("path") == "binding",
               f"got path={skips[0].get('path')}")
        _check("T82 skip_reason = word_overlap_zero",
               skips[0].get("skip_reason") == "word_overlap_zero",
               f"got skip_reason={skips[0].get('skip_reason')}")

    result = scrub_response(response, grounding)
    cleaned = result.get("cleaned_response") or ""
    _check("T82 correct-answer sentence preserved",
           "qwen_spec_decode_server.py" in cleaned
           and "19/20 failures" in cleaned
           and "5%" in cleaned,
           f"cleaned={cleaned!r}")
    _check("T82 sentences_stripped = 0",
           result.get("sentences_stripped", 0) == 0,
           f"got sentences_stripped={result.get('sentences_stripped')}")


# ──────────────────────────────────────────────────────────────────────
# Test 3: T68 regression — abstention guard on binding path preserved
# ──────────────────────────────────────────────────────────────────────

def test_t68_abstention_path_regression_preserved():
    """M122 A2 T68: honest-abstention preservation via abstention
    guard on the binding path. M123 A1 must not regress this."""
    print("\n[Test 3] T68 abstention-path regression preserved")
    response = (
        "I don't have information about an extraction recall of 61% "
        "or which model tier that corresponds to in the briefing or "
        "provided memory.")
    grounding = "3B solo 61% extraction recall. Tier 1 boundary."

    flags, skips = tier2_binding_check(response, grounding)
    _check("T68 binding-path 0 flags",
           len(flags) == 0,
           f"expected 0 flags, got {flags}")
    _check("T68 binding-path 1 skip with abstention_pattern",
           len(skips) == 1
           and skips[0].get("skip_reason") == "abstention_pattern",
           f"got skips={skips}")
    if skips:
        _check("T68 skip path annotation = binding",
               skips[0].get("path") == "binding",
               f"got path={skips[0].get('path')}")

    result = scrub_response(response, grounding)
    cleaned = result.get("cleaned_response") or ""
    _check("T68 abstention sentence preserved",
           "don't have information" in cleaned and "61%" in cleaned,
           f"cleaned={cleaned!r}")


# ──────────────────────────────────────────────────────────────────────
# Test 4: New — abstention guard on conflict path
# ──────────────────────────────────────────────────────────────────────

def test_abstention_guard_on_conflict_path():
    """Abstention sentences that happen to include a claim matching a
    registry entity should have conflict flags suppressed. Example:
    "I don't have Neuron measurements near 42 tok/s, but maybe in an
    older run." — the claim `42 tok/s` would conflict with
    `neuron.throughput=1064`, but the abstention phrase in the same
    clause as the claim should suppress the flag."""
    print("\n[Test 4] Abstention guard fires on conflict path")
    response = (
        "I don't have Neuron measurements near 42 tok/s in memory.")

    conflict_flags, conflict_skips = tier2_registry_conflict_check(
        response)
    neuron_flags = [f for f in conflict_flags
                    if f.get("registry_key") == "neuron.throughput"]
    _check("Abstention on conflict path suppresses neuron flag",
           len(neuron_flags) == 0,
           f"expected 0 neuron flags, got {neuron_flags}")

    abstention_skips = [s for s in conflict_skips
                        if s.get("path") == "conflict"
                        and s.get("skip_reason") == "abstention_pattern"]
    _check("Abstention skip recorded on conflict path",
           len(abstention_skips) >= 1,
           f"expected ≥1 abstention skip, got {conflict_skips}")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Schema — skip list is list-of-dicts with path annotation
# ──────────────────────────────────────────────────────────────────────

def test_skip_reason_schema_path_annotated():
    """ζ v2.2 extension: `scrub.tier2_binding.skip_reason` is a list of
    dicts, each with `path` ∈ {binding, flag, conflict}, `reason` /
    `skip_reason` ∈ {word_overlap_zero, abstention_pattern, other}.
    Backward compatibility with M122 A2's string-list interpretation
    is preserved because consumers index the list of dicts by the
    legacy `skip_reason` key."""
    print("\n[Test 5] Skip list schema — path-annotated dicts")

    # Compose a response that fires both paths: binding-path T82 and
    # conflict-path T59 happen on distinct grounding, so synthesize a
    # single response that triggers both paths.
    response_binding = (
        "The clean rate dropped to 5% in an unrelated qwen run.")
    grounding_binding = (
        "m114 pilot 5% mechanism count. qwen stream ordering fix.")

    result_binding = scrub_response(response_binding, grounding_binding)
    sk_binding = result_binding.get("tier2_binding_skip_reasons", [])
    _check("Binding-path skip list is list",
           isinstance(sk_binding, list),
           f"got {type(sk_binding)}")
    if sk_binding:
        e = sk_binding[0]
        _check("Binding-path entry is dict",
               isinstance(e, dict), f"got {type(e)}")
        _check("Binding-path entry has `path` key",
               isinstance(e, dict) and "path" in e,
               f"entry={e}")
        _check("Binding-path entry has legacy `skip_reason` key",
               isinstance(e, dict) and "skip_reason" in e,
               f"entry={e}")

    # Conflict path via T59.
    response_conflict = (
        "EAGLE-3 had a 0% acceptance rate on all quantized models, "
        "including Q3 and Q4.")
    result_conflict = scrub_response(response_conflict, grounding_corpus="")
    sk_conflict = result_conflict.get("tier2_binding_skip_reasons", [])
    conflict_paths = [e for e in sk_conflict
                      if isinstance(e, dict) and e.get("path") == "conflict"]
    _check("Conflict-path skip entry present with path=conflict",
           len(conflict_paths) >= 1,
           f"got {sk_conflict}")


# ──────────────────────────────────────────────────────────────────────
# Test 6: M108/M117 corpus — binding-path preservation holds
# ──────────────────────────────────────────────────────────────────────

def test_m108_m117_corpus_preserved_under_a1():
    """M122 A2 regression corpus (5 historical bare-scalar over-strip
    cases). M123 A1 binding path is unchanged so preservation must
    hold identically; this test re-asserts the M122 A2 invariant from
    the M123 A1 test file for completeness."""
    print("\n[Test 6] M108/M117 corpus preserved under M123 A1")
    cases = [
        ("T60 EAGLE-3 0% quantized dead path",
         "It had 0% acceptance on both Q3 and Q4 quantized 70B models "
         "because quantization destroys the hidden states required for "
         "the architecture to function.",
         "0% acceptance EAGLE-3 70B Q3 Q4 quantized."),
        ("T58 1B drafter 0-1% unique acceptance",
         "speculative decoding for Apple Silicon is currently limited "
         "to N-gram in production, as the ANE 1B drafter path was "
         "killed due to a 0-1% unique acceptance rate.",
         "ANE 1B drafter speculative decoding 1% unique acceptance."),
        ("T39 0% EAGLE quantization dead path",
         "EAGLE-3 is killed for this stack: quantization destroys the "
         "hidden states needed for the speculative draft, yielding 0% "
         "acceptance on both Q3 and Q4 variants.",
         "EAGLE-3 dead on quantized 70B. 0% acceptance Q3 Q4."),
        ("T6 50.2 tok/s benchmark pipeline efficient",
         "No, we do not have true concurrency for Subconscious yet; "
         "the 50.2 tok/s benchmark proves the pipeline is efficient "
         "(C ops + cross-layer fusion).",
         "Llama-1B 50.2 tok/s combined stack C ops fusion."),
        ("T20 0% contamination isolation fixes",
         "By implementing five system isolation fixes—including adding "
         "topic classification to the memory schema, introducing "
         "provenance tags, and a 1.5x topic boost—errors dropped from "
         "37.5% to 0% contamination.",
         "System isolation 37.5% 0% contamination 1.5x topic boost."),
    ]
    pass_count = 0
    for label, resp, gc in cases:
        flags, skips = tier2_binding_check(resp, gc)
        if skips:
            pass_count += 1
            _check(f"[{label}] binding-path path=binding",
                   all(s.get("path") == "binding" for s in skips),
                   f"got paths={[s.get('path') for s in skips]}")
        elif not flags:
            pass_count += 1
    _check("M108/M117 corpus: 5 historical cases preserved",
           pass_count >= 5,
           f"got {pass_count}/5 preserved")
    return pass_count


# ──────────────────────────────────────────────────────────────────────
# Test 7: Conflict path still fires for true conflicts
# ──────────────────────────────────────────────────────────────────────

def test_conflict_path_true_positive_still_fires():
    """Canonical M120 C regression case: "NAX delivers 40 TFLOPS in
    the profiled workload" — claim diverges from registry
    (`nax.throughput=30.4 TFLOPS`). The conflict-path guard has
    window-overlap on `nax` (entity alias in local window), so it does
    NOT fire; the flag is emitted as M120 C intended."""
    print("\n[Test 7] Conflict path true-positive still fires")
    response = "NAX delivers 40 TFLOPS in the profiled workload."
    conflict_flags, conflict_skips = tier2_registry_conflict_check(
        response)
    nax_flags = [f for f in conflict_flags
                 if f.get("registry_key") == "nax.throughput"]
    _check("NAX 40 vs 30.4 TFLOPS conflict flag emitted",
           len(nax_flags) == 1,
           f"expected 1 nax.throughput flag, got {conflict_flags}")
    _check("No conflict-path skip on NAX flag",
           all(s.get("registry_key") != "nax.throughput"
               for s in conflict_skips),
           f"got skips={conflict_skips}")


# ──────────────────────────────────────────────────────────────────────
# Test 8: Neuron topic case (M120 C test 2) — flag preserved
# ──────────────────────────────────────────────────────────────────────

def test_conflict_path_neuron_topic_case():
    """M120 C: "The Neuron model runs at 7.9 tok/s on the ANE" — entity
    name `Neuron` sits at sentence start, 4 words from the claim. The
    M123 A1 conflict-path guard widens the window to ±5 so this case
    is correctly preserved as a true conflict (vs the T59 case where
    the registry entity is secondary and the window at ±5 has no
    overlap)."""
    print("\n[Test 8] Neuron topic-case conflict flag preserved")
    response = "The Neuron model runs at 7.9 tok/s on the ANE."
    conflict_flags, _ = tier2_registry_conflict_check(response)
    neuron_flags = [f for f in conflict_flags
                    if f.get("registry_key") == "neuron.throughput"]
    _check("Neuron conflict flag still emitted (topic-case)",
           len(neuron_flags) >= 1,
           f"expected neuron.throughput flag, got {conflict_flags}")


# ──────────────────────────────────────────────────────────────────────
# Test 9: Helper — _m123_evaluate_conflict_path_guards correctness
# ──────────────────────────────────────────────────────────────────────

def test_m123_helper_correctness():
    """Unit-level checks on the conflict-path guard helper."""
    print("\n[Test 9] _m123_evaluate_conflict_path_guards correctness")
    entry_q4 = {
        "entity": "q4",
        "aliases": ["q4", "4-bit"],
        "measurement_type": "dequant_penalty",
    }
    # T59: topic-mismatch fires.
    reason = _m123_evaluate_conflict_path_guards(
        "EAGLE-3 had a 0% acceptance rate on all quantized models, "
        "including Q3 and Q4.",
        "0%", entry_q4)
    _check("T59 helper → word_overlap_zero",
           reason == "word_overlap_zero",
           f"got reason={reason}")

    # Neuron topic case: overlap on entity name → no skip.
    entry_neuron = {
        "entity": "neuron",
        "aliases": ["neuron", "classifier", "80m"],
        "measurement_type": "throughput",
    }
    reason = _m123_evaluate_conflict_path_guards(
        "The Neuron model runs at 7.9 tok/s on the ANE.",
        "7.9 tok/s", entry_neuron)
    _check("Neuron topic-case helper → None (no skip)",
           reason is None,
           f"got reason={reason}")

    # Abstention in same clause → abstention_pattern.
    reason = _m123_evaluate_conflict_path_guards(
        "I don't have Neuron measurements near 42 tok/s in memory.",
        "42 tok/s", entry_neuron)
    _check("Abstention-in-clause helper → abstention_pattern",
           reason == "abstention_pattern",
           f"got reason={reason}")


# ──────────────────────────────────────────────────────────────────────
# Runner + registry emission
# ──────────────────────────────────────────────────────────────────────

def emit_registry_values(corpus_pass_count):
    """Print registry keys for dict-envelope persistence."""
    print("\n[Registry values for m123.a1]")
    verdict = "shipped" if _FAIL == 0 else "shipped_with_gaps"
    print(f"  m123.a1.verdict = {verdict!r}")
    # Alternate paths covered: binding (pre-existing) + conflict (new)
    # = 1 new path wired. tier2_flags/tier2_conflict were the two
    # emission sites surfaced in the directive; binding_check is
    # already guarded. "Covered" here counts NEW sites guarded by M123.
    print(f"  m123.a1.alternate_paths_covered = 1")
    print(f"  m123.a1.t59_inline_leak_eliminated = {_FAIL == 0}")
    print(f"  m123.a1.binding_path_regression_preserved = "
          f"{corpus_pass_count >= 5}")


def main():
    print("=" * 60)
    print("M123 Stream A1 — A2 alternate-pathway gap close tests")
    print("=" * 60)

    try:
        test_t59_replay_no_inline_template_leak()
        test_t82_binding_path_regression_preserved()
        test_t68_abstention_path_regression_preserved()
        test_abstention_guard_on_conflict_path()
        test_skip_reason_schema_path_annotated()
        corpus_pass_count = test_m108_m117_corpus_preserved_under_a1()
        test_conflict_path_true_positive_still_fires()
        test_conflict_path_neuron_topic_case()
        test_m123_helper_correctness()
    except Exception as e:
        print(f"\nFAIL (exception): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        corpus_pass_count = 0

    print("\n" + "=" * 60)
    print(f"Passed: {_PASS}")
    print(f"Failed: {_FAIL}")
    if _FAILURES:
        print("\nFailures:")
        for label, detail in _FAILURES:
            print(f"  - {label}: {detail}")
    emit_registry_values(corpus_pass_count)
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
