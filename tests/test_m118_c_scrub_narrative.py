"""M118 Stream C — scrub Tier 2 narrative-drift extension.

Authoritative spec:
    vault/agent_reports/m117_parent_synthesis.md (K4 NEW_scrub_under_detection)
    vault/agent_reports/m117_a3_scoring_and_persistence.md §K4 + T24/T30/T33/T35
    vault/agent_reports/m118_c_scrub_narrative_drift.md

Mechanism anchor: M117 K4 pilot evidence. scrub Tier 2 binding-check
catches numeric misattribution but authoritative prose without numeric
markers passes through. Distinct from M116 A outbound guard which fires
on grounding *absence*; this fires when grounding IS present but the
narrative paraphrases drift.

Tests:
    Pilot replays — T24 (DFlash), T30 (NSIRD path fab), T33 (ZinAneTd
    template fab), T35 (xnu_kernel path citation — this one IS real per
    live-vault spot-check; retained as regression because wording drifts
    from the grounding). Regressions — T4 (ane-dispatch content-correct),
    T36 (Gemma footprint), paraphrase case.

Run:
    ~/.mlx-env/bin/python3 orion-ane/tests/test_m118_c_scrub_narrative.py

Registry values produced by this suite:
    m118.c.verdict                                    : shipped | shipped_narrow | deferred
    m118.c.detection_approach                         : option_beta_primary | option_beta_plus_alpha_fallback | option_gamma_narrow | deferred
    m118.c.pilot_replay_pass_count                    : int (out of 4)
    m118.c.regression_pass_count                      : int (out of 3)

m118_c_tier2_narrative_drift
"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_AGENT_DIR = os.path.join(_REPO_ROOT, "orion-ane", "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from answer_scrub import (  # noqa: E402
    scrub_response,
    tier2_narrative_drift_check,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures — pulled from data/session_logs/sess_20260421_175653_85335/
# turn_NNNN.json. We materialize (query, response, grounding) triples
# directly so the suite can run without the live session log present.
# ──────────────────────────────────────────────────────────────────────

SESS = "sess_20260421_175653_85335"
_TURN_DIR = os.path.join(_REPO_ROOT, "data", "session_logs", SESS)


def _load_turn(n):
    """Return (query, response, grounding_corpus) from the pilot log.

    Falls back to hard-coded text if the file is missing so the suite
    still runs post-cleanup.
    """
    p = os.path.join(_TURN_DIR, f"turn_{n:04d}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    q = (d.get("input") or {}).get("query", "")
    r = (d.get("generation") or {}).get("response_text", "")
    ctx = d.get("context") or {}
    mem = ctx.get("mem_ctx_text") or ""
    brf = ctx.get("briefing_text") or ""
    if isinstance(mem, list):
        mem = "\n".join(str(x) for x in mem)
    if isinstance(brf, list):
        brf = "\n".join(str(x) for x in brf)
    return q, r, (mem + "\n" + brf)


# ── Hard-coded fallbacks (M117 pilot content) ────────────────────────

T24_FALLBACK = (
    "what is dflash?",
    (
        "DFlash is a performance acceleration tool that does NOT "
        "support Gemma 4 due to a missing trained draft checkpoint, "
        "the lack of a custom MLX target adapter for RotatingKVCache "
        "rollback (sliding window layers), and missing cache "
        "verification logic for hybrid attention. It currently leaves "
        "a 44% speed gap compared to Qwen 27B, which achieved "
        "14.3 tok/s (recorded as `dflash.qwen27b_tok_s`) before "
        "receiving a `KILLED_SPEED` verdict."
    ),
    # Grounding that does NOT contain DFlash, Qwen 27B 14.3 tok/s,
    # KILLED_SPEED, or RotatingKVCache. Realistic M117 briefing excerpt.
    (
        "CURRENT SESSION CONTEXT: Gemma 4 31B Q4 is the production "
        "verifier on GPU at 17.5 tok/s. N-gram drafter disabled "
        "(phrase loops, Main 43). Stop tokens [1, 106, 50]. "
        "The ane-compiler ships Llama-1B at 50.2 tok/s. "
        "8B Q8 on ANE at 7.9 tok/s via ane_server_8b.py."
    ),
)

T30_FALLBACK = (
    "what sort of files does the aned generate?",
    (
        "`aned` is a thin orchestration daemon; the actual file "
        "generation occurs via the `ANECompilerService`. During "
        "compilation, it creates temporary files in "
        "`/private/var/folders/zz/.../T/com.apple.aned/TemporaryItems/"
        "NSIRD_ANECompilerService_*/`, specifically: "
        "- `net.plist`: Network plist (espresso IR translation output). "
        "- `net_aux.json`: Auxiliary network info. "
        "- `model.src`: Source model copy (cached alongside `.hwx`)."
    ),
    (
        "ane-compiler wraps aned for compilation. emit_linear_hwx() "
        "emits conv/inner_product .hwx directly. Dispatches are via "
        "IOKit to ANE firmware. 72 CoreML models, 7.2 GB."
    ),
)

T33_FALLBACK = (
    "what kind of files does the ane compiler generate?",
    (
        "The ANE compiler generates code via `ZinAneTd<N>` template "
        "specializations, with specific codegen for each hardware "
        "version. Production artifacts from the compiler include "
        "`build_8b_q8.py` and `build_8b_attn_fused.py`, which are "
        "used to produce the 72 CoreML models (7.2 GB) currently "
        "running in the 8B ANE server."
    ),
    (
        "ane-compiler/build_8b_q8.py emits the 72 CoreML models "
        "(7.2 GB) consumed by ane_server_8b.py. ane-compiler also "
        "ships Llama-1B at 50.2 tok/s and GPT-2 at 229 tok/s."
    ),
)

T35_FALLBACK = (
    "what issues have we faced with memory pressure?",
    (
        "We have focused on kernel-side memory pressure thresholds "
        "and the resulting system behavior. Specifically, research in "
        "`agent_reports/stage1_agent_a_xnu_kernel_memory.md` "
        "investigated the `memory_pressure -l` transitions (Normal, "
        "Warning, Critical, Urgent) and their corresponding XNU "
        "kernel thresholds defined in `bsd/sys/kern_memorystatus.h` "
        "and `osfmk/vm/vm_pageout.c`. This work specifically "
        "addressed the mechanics of the compressor, footprint "
        "attribution, and the Jetsam idle-exit cascade."
    ),
    # Grounding does NOT cite the xnu_kernel path or bsd/sys headers.
    # Spot-checked: path IS present in live vault but not in the
    # grounding offered on this turn — drift against what the model
    # was handed.
    (
        "No System Reboots — Operator standing rule. System reboots "
        "are unacceptably costly. Every cognitive-stack action must "
        "consider memory headroom before launching. gate=YELLOW "
        "threshold at single-daemon RSS > 40 GB on 64 GB box."
    ),
)

T4_FALLBACK = (
    "what is the current status of the ane-dispatch project specifically",
    (
        "ane-dispatch is the first open-source demonstration of "
        "IOSurfaceSharedEvent-based cross-accelerator signaling "
        "through the aned macOS XPC service. It enables client "
        "processes to send CoreML-compiled .mlmodelc bundles to "
        "aned, which then manages firmware interface tasks including "
        "program slot allocation (128 slots per device), weight "
        "memory mapping to IOSurface, and dispatch queueing."
    ),
    (
        "ane-dispatch (published): Direct ANE dispatch + "
        "SharedEvents. 37% faster than CoreML. "
        "IOSurfaceSharedEvent cross-accelerator signaling through "
        "the aned macOS XPC service. CoreML-compiled .mlmodelc "
        "bundles sent to aned. Program slot allocation 128 slots "
        "per device. Weight memory mapped to IOSurface. "
        "Dispatch queueing handled by aned."
    ),
)

T36_FALLBACK = (
    "what's the memory footprint of Gemma 4?",
    "Gemma 4 31B Dense Q4 has a memory footprint of 17.6 GB.",
    (
        "CANONICAL: Gemma 4 31B Q4 production verifier on GPU. "
        "17.6 GB model footprint. Pure AR at 17.5 tok/s. "
        "Swapped Main 52 from Qwen 27B."
    ),
)

# Paraphrase regression — response says the same thing in different
# words. Should PASS (stay above 0.60 threshold).
T_PARAPHRASE = (
    "how fast is ane-compiler on llama 1b?",
    (
        "On Llama-1B quantized to Q8, ane-compiler reaches "
        "approximately 50 tokens per second using 25 dispatches "
        "fused with the C backend."
    ),
    (
        "ane-compiler achieves 50.2 tok/s on Llama-1B with 25d+C "
        "fusion. CPU 0.4ms, ANE 15.5ms, lm_head 4.6ms."
    ),
)


def _replay(label, triple, expected_verdict, registry):
    if triple is None:
        print(f"[{label}] no fixture — skipping")
        registry["skipped"].append(label)
        return None
    query, response, grounding = triple
    verdict, diag = tier2_narrative_drift_check(response, grounding)
    scrub_out = scrub_response(response, grounding, user_query=query,
                               tools_called=[])
    observed = "DRIFT" if scrub_out["tier2_narrative_flags"] else "PASS"
    ok = observed == expected_verdict
    print(f"[{label}] expect={expected_verdict} observed={observed} "
          f"direct={verdict} "
          f"flags={len(scrub_out['tier2_narrative_flags'])} "
          f"checked={diag.get('checked')} "
          f"embedder={diag.get('embedder_available')} "
          f"narr_diag_lat={diag.get('latency_ms')}ms "
          f"→ {'PASS' if ok else 'FAIL'}")
    if ok:
        registry["pass"].append(label)
    else:
        registry["fail"].append((label, expected_verdict, observed, diag))
    return scrub_out, diag


def main():
    print("── M118 Stream C — Tier 2 narrative-drift replay ─────────")
    registry = {"pass": [], "fail": [], "skipped": []}

    # Pilot replays (expect DRIFT)
    print("\n-- Pilot replays (expect DRIFT) --")
    _replay("T24_dflash_narrative",
            _load_turn(24) or T24_FALLBACK,
            "DRIFT", registry)
    _replay("T30_nsird_path_fab",
            _load_turn(30) or T30_FALLBACK,
            "DRIFT", registry)
    _replay("T33_zinanetd_template",
            _load_turn(33) or T33_FALLBACK,
            "DRIFT", registry)
    _replay("T35_xnu_kernel_path",
            _load_turn(35) or T35_FALLBACK,
            "DRIFT", registry)

    pilot_pass = sum(1 for k in registry["pass"]
                     if k.startswith(("T24", "T30", "T33", "T35")))

    # Regressions (expect PASS)
    print("\n-- Regressions (expect PASS) --")
    _replay("T4_ane_dispatch_correct",
            _load_turn(4) or T4_FALLBACK,
            "PASS", registry)
    _replay("T36_gemma_footprint",
            _load_turn(36) or T36_FALLBACK,
            "PASS", registry)
    _replay("paraphrase_llama_1b_tps",
            T_PARAPHRASE,
            "PASS", registry)

    regression_pass = sum(1 for k in registry["pass"]
                          if k.startswith(("T4", "T36", "paraphrase")))

    # ── M116 A coordination check ────────────────────────────────
    try:
        from confabulation_shape_detector import ABSTAIN_MESSAGE
        abstain = ABSTAIN_MESSAGE
    except Exception:
        abstain = (
            "I don't have grounded support for that answer. Nothing in "
            "memory, recalled context, or a dispatched tool call backs "
            "the claim I was about to make. Ask a more specific "
            "question or point me to a source and I'll retry."
        )
    print("\n-- M116 A coordination --")
    out = scrub_response(abstain, "unrelated grounding", user_query="x",
                         tools_called=[])
    skip_ok = out["tier2_narrative_verdict"] == "SKIPPED_ABSTAIN"
    print(f"[m116_a_coord] verdict={out['tier2_narrative_verdict']} "
          f"→ {'PASS' if skip_ok else 'FAIL'}")
    if skip_ok:
        registry["pass"].append("m116_a_coord")
    else:
        registry["fail"].append(("m116_a_coord", "SKIPPED_ABSTAIN",
                                 out["tier2_narrative_verdict"], {}))

    # ── Summary ──────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────")
    print(f"pilot pass:      {pilot_pass}/4")
    print(f"regression pass: {regression_pass}/3")
    print(f"m116_a coord:    {'PASS' if skip_ok else 'FAIL'}")
    print(f"total pass:  {len(registry['pass'])}")
    print(f"total fail:  {len(registry['fail'])}")
    print(f"skipped:     {len(registry['skipped'])}")
    if registry["fail"]:
        print("FAILURES:")
        for f in registry["fail"]:
            print(f"  {f}")

    print("\n── Registry deltas ──────────────────────────────────────")
    if pilot_pass == 4 and regression_pass == 3:
        verdict = "shipped"
        approach = "option_beta_primary"
    elif pilot_pass >= 3 and regression_pass >= 2:
        verdict = "shipped_narrow"
        approach = "option_beta_primary"
    else:
        verdict = "deferred"
        approach = "deferred"
    print(f"m118.c.verdict: {verdict}")
    print(f"m118.c.detection_approach: {approach}")
    print(f"m118.c.pilot_replay_pass_count: {pilot_pass}")
    print(f"m118.c.regression_pass_count: {regression_pass}")

    # Exit nonzero only if hard-expected invariants break
    if len(registry["fail"]) and not (pilot_pass >= 3 and
                                      regression_pass >= 2):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
