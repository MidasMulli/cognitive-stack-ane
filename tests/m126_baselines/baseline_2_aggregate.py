#!/usr/bin/env python3
"""
M126 Baselines Stream C — Aggregate + attribution analysis.

Reads baseline_2_scored.jsonl and emits:
- Aggregate strict / generous / partial / fail counts + rates
- Per-class strict rates (skip routing per taxonomy gap)
- ground_truth_in_grep_hits_pct (excluding memory_store/operator_knowledge/
  model_parametric/ambiguous source locations — grep can't find those)
- zero_hit_turn_count
- truncation_count

Writes aggregate summary as JSON + builds the vault report markdown.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCORED_PATH = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_2_scored.jsonl")
AGGREGATE_JSON = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_2_aggregate.json")
REPORT_PATH = Path("/Users/midas/Desktop/cowork/vault/agent_reports/m126_baselines_c_model_grep.md")

NON_GREPPABLE_SOURCES = {"memory_store", "operator_knowledge", "model_parametric", "ambiguous", "none", "", None}


def ground_truth_in_grep(record: dict) -> str:
    """Return 'hit' | 'miss' | 'na' for this scored turn."""
    src = record.get("ground_truth_source_location")
    if src in NON_GREPPABLE_SOURCES or record.get("ground_truth_ambiguous"):
        return "na"
    # src may be a vault-relative path with :line or a file:anchor style
    # Extract the leading file path (strip after first ':')
    path_part = src.split(":")[0].strip() if src else ""
    if not path_part:
        return "na"
    # Normalize: grep_file_list entries are absolute paths; ground truth may be
    # relative ("vault/..." or "CLAUDE.md") or absolute. Match loosely.
    grep_files = record.get("grep_file_list", [])
    # Build a set of basenames + full paths for matching
    for gf in grep_files:
        # Match if path_part appears as substring or basename equality
        if path_part in gf:
            return "hit"
        if os.path.basename(path_part) == os.path.basename(gf):
            return "hit"
    return "miss"


def main() -> int:
    scored = []
    with open(SCORED_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scored.append(json.loads(line))

    total = len(scored)
    ambiguous = [r for r in scored if r.get("ground_truth_ambiguous")]
    scorable = [r for r in scored if not r.get("ground_truth_ambiguous")]
    ambiguous_count = len(ambiguous)
    n_scorable = len(scorable)

    # Aggregate verdict counts on scorable only (strict-rate denominator)
    verdict_counts = Counter(r["verdict"] for r in scorable)
    # Full counts (including ambiguous)
    verdict_counts_all = Counter(r["verdict"] for r in scored)

    strict = verdict_counts["strict"]
    generous = verdict_counts["generous"]
    partial = verdict_counts["partial"]
    fail = verdict_counts["fail"]

    strict_rate = strict / n_scorable if n_scorable else 0.0
    generous_rate = (strict + generous) / n_scorable if n_scorable else 0.0
    fail_rate = fail / n_scorable if n_scorable else 0.0

    # Per-class strict rates (excluding routing)
    by_class = defaultdict(list)
    for r in scorable:
        qc = r.get("query_class") or "other"
        by_class[qc].append(r)

    per_class_strict = {}
    for qc, rows in by_class.items():
        if qc == "routing":
            continue  # taxonomy gap per directive
        strict_n = sum(1 for r in rows if r["verdict"] == "strict")
        per_class_strict[qc] = {
            "strict_rate": strict_n / len(rows) if rows else 0.0,
            "strict_n": strict_n,
            "total": len(rows),
            "generous_n": sum(1 for r in rows if r["verdict"] == "generous"),
            "partial_n": sum(1 for r in rows if r["verdict"] == "partial"),
            "fail_n": sum(1 for r in rows if r["verdict"] == "fail"),
        }

    # Ground-truth coverage: only on scorable turns with greppable sources
    greppable_scorable = [r for r in scorable if r.get("ground_truth_source_location") not in NON_GREPPABLE_SOURCES]
    gt_hits = [r for r in greppable_scorable if ground_truth_in_grep(r) == "hit"]
    gt_in_grep_pct = (len(gt_hits) / len(greppable_scorable) * 100) if greppable_scorable else 0.0

    # Zero-hit and truncation counts (on all 135)
    zero_hit = sum(1 for r in scored if r["grep_hit_count"] == 0)
    truncation = sum(1 for r in scored if r["truncated"])

    # Fabrication count (on scorable)
    fabrication_n = sum(1 for r in scorable if r.get("fabrication"))
    fabrication_rate = fabrication_n / n_scorable if n_scorable else 0.0
    severity_dist = Counter(r.get("severity", 0) for r in scorable)

    aggregate = {
        "total_turns": total,
        "ambiguous_count": ambiguous_count,
        "n_scorable": n_scorable,
        "verdict_counts_scorable": dict(verdict_counts),
        "verdict_counts_all": dict(verdict_counts_all),
        "strict_pass_rate": strict_rate,
        "generous_pass_rate": generous_rate,
        "fail_rate": fail_rate,
        "per_class_strict_rates": per_class_strict,
        "ground_truth_in_grep_hits_pct": gt_in_grep_pct,
        "ground_truth_greppable_denominator": len(greppable_scorable),
        "ground_truth_hit_count": len(gt_hits),
        "zero_hit_turn_count": zero_hit,
        "truncation_count": truncation,
        "fabrication_count": fabrication_n,
        "fabrication_rate": fabrication_rate,
        "severity_distribution": dict(severity_dist),
    }

    with open(AGGREGATE_JSON, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Build vault report
    lines = []
    lines.append("# M126 Baselines — Stream C: Baseline 2 (Gemma 4 31B Q4 + raw vault grep)")
    lines.append("")
    lines.append(f"**Directive:** `vault/directives/in_progress/2026-04-24T00-46-26_m126_baselines_open_m126-baselines-comparative-measurement-f.md` §3.3")
    lines.append(f"**Harness:** `orion-ane/tests/m126_baselines/baseline_2_model_grep.py`")
    lines.append(f"**Corpus:** `data/m126_baselines/corpus_v1.jsonl` (135 records)")
    lines.append(f"**Rubric:** `data/m126_baselines/rubric_v1.md` (hash `e9595e49432db6783da6be54ac0a35c24ea1dbdd94b0f37c76fbb9a281fb551f`)")
    lines.append(f"**Verifier:** Gemma 4 31B Q4 on `:8899`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Total turns: **{total}**")
    lines.append(f"- Ambiguous bucket: **{ambiguous_count}** (excluded from strict-rate denominator)")
    lines.append(f"- Scorable: **{n_scorable}**")
    lines.append("")
    lines.append("| Verdict | Count (scorable) | Rate |")
    lines.append("|---|---|---|")
    lines.append(f"| strict | {strict} | {strict_rate*100:.2f}% |")
    lines.append(f"| generous (includes strict) | {strict+generous} | {generous_rate*100:.2f}% |")
    lines.append(f"| partial | {partial} | {partial/n_scorable*100:.2f}% |")
    lines.append(f"| fail | {fail} | {fail_rate*100:.2f}% |")
    lines.append("")
    lines.append(f"**Fabrication count:** {fabrication_n} ({fabrication_rate*100:.2f}%)")
    lines.append(f"**Severity distribution:** {dict(severity_dist)}")
    lines.append("")
    lines.append("## Per-class strict rates (routing excluded per taxonomy gap)")
    lines.append("")
    lines.append("| Class | N | strict | generous | partial | fail | strict_rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for qc in sorted(per_class_strict.keys()):
        s = per_class_strict[qc]
        lines.append(f"| {qc} | {s['total']} | {s['strict_n']} | {s['generous_n']} | {s['partial_n']} | {s['fail_n']} | {s['strict_rate']*100:.2f}% |")
    lines.append("")
    lines.append("## Ground-truth coverage in grep hits")
    lines.append("")
    lines.append(f"- Greppable scorable turns (ground-truth source is a vault path): **{len(greppable_scorable)}**")
    lines.append(f"- Turns where ground-truth source appeared in `grep_file_list`: **{len(gt_hits)}**")
    lines.append(f"- `ground_truth_in_grep_hits_pct`: **{gt_in_grep_pct:.2f}%**")
    lines.append("")
    lines.append("Excluded from coverage analysis (non-greppable sources): `memory_store`, `operator_knowledge`, `model_parametric`, `ambiguous`, `none`, or ambiguous flag set.")
    lines.append("")
    lines.append("## Truncation + zero-hit distribution")
    lines.append("")
    lines.append(f"- **truncation_count** (4000-char cap hit): **{truncation}** / {total} ({truncation/total*100:.1f}%)")
    lines.append(f"- **zero_hit_turn_count** (grep returned no files): **{zero_hit}** / {total} ({zero_hit/total*100:.1f}%)")
    lines.append("")
    lines.append("## K-outcomes")
    lines.append("")
    lines.append(f"- **K7** grep-context truncated: expected for broad queries; observed **{truncation}** turns hit 4000-char cap")
    lines.append(f"- **K8** grep zero hits: observed **{zero_hit}** turns returned no grep files (keyword extraction stripped all tokens OR no vault match)")
    lines.append(f"- **K9** keyword extraction misses ground-truth: manifests as `ground_truth_in_grep_hits_pct < 100%` on greppable turns. Observed: **{100 - gt_in_grep_pct:.2f}%** of greppable turns where grep did not find the canonical source — this is the basic-retrieval limitation the directive isolates. Short tokens (<4 chars like version numbers 'M59', 'Q4', '1.01') and domain-adjacent paraphrase (query uses synonyms absent from vault text) both contribute.")
    lines.append(f"- **K10** Track 2 leak: **none** — vault tree does not contain `track2`/`derivatives`/`platform_x`; path-filter active in harness as safety. No halts triggered.")
    lines.append("")
    lines.append("## Registry values (proposed)")
    lines.append("")
    lines.append("```")
    lines.append(f"m126_baselines.c.strict_pass_rate = {strict_rate:.4f}")
    lines.append(f"m126_baselines.c.per_class_strict_rates = {json.dumps({k: round(v['strict_rate'],4) for k,v in per_class_strict.items()})}")
    lines.append(f"m126_baselines.c.ground_truth_in_grep_hits_pct = {gt_in_grep_pct:.2f}")
    lines.append(f"m126_baselines.c.truncation_count = {truncation}")
    lines.append(f"m126_baselines.c.zero_hit_turn_count = {zero_hit}")
    lines.append("```")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Per-turn response JSONs: `data/m126_baselines/baseline_2_responses/turn_<turn_id>.json`")
    lines.append("- Scored JSONL: `data/m126_baselines/baseline_2_scored.jsonl`")
    lines.append("- Aggregate JSON: `data/m126_baselines/baseline_2_aggregate.json`")
    lines.append("- Scoring method: LLM-as-judge (Gemma 4 on :8899) gated on 10-turn spot-check; rule-based fallback if spot-check fails. Scorer per-turn recorded in `scored.jsonl`.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    sys.stdout.write(f"[done] aggregate -> {AGGREGATE_JSON}\n")
    sys.stdout.write(f"[done] report -> {REPORT_PATH}\n")

    # Print summary
    sys.stdout.write("\n=== SUMMARY ===\n")
    sys.stdout.write(f"Total turns: {total}\n")
    sys.stdout.write(f"Scorable: {n_scorable} (ambiguous: {ambiguous_count})\n")
    sys.stdout.write(f"Strict: {strict} ({strict_rate*100:.2f}%)\n")
    sys.stdout.write(f"Generous (incl strict): {strict+generous} ({generous_rate*100:.2f}%)\n")
    sys.stdout.write(f"Fail: {fail} ({fail_rate*100:.2f}%)\n")
    sys.stdout.write(f"Ground-truth in grep hits: {gt_in_grep_pct:.2f}% (n={len(greppable_scorable)})\n")
    sys.stdout.write(f"Zero-hit turns: {zero_hit}\n")
    sys.stdout.write(f"Truncated turns: {truncation}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
