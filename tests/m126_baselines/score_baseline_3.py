#!/usr/bin/env python3
"""
M126 Baselines — Stream D scorer.

Mirrors score_baseline_1.py (same rubric + same LLM-judge methodology). Adds
Stream D-specific analysis:
  - ground_truth_in_top5 coverage (overall + per-class)
  - top-5 miss distribution by query class
  - aggregate + per-class strict rates

Rubric source: /Users/midas/Desktop/cowork/data/m126_baselines/rubric_v1.md
Rubric SHA:    e9595e49432db6783da6be54ac0a35c24ea1dbdd94b0f37c76fbb9a281fb551f
"""

import hashlib
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines")
RESP_DIR = DATA_DIR / "baseline_3_responses"
OUT_JSONL = DATA_DIR / "baseline_3_scored.jsonl"
RUBRIC_PATH = DATA_DIR / "rubric_v1.md"
RUBRIC_EXPECTED_SHA = "e9595e49432db6783da6be54ac0a35c24ea1dbdd94b0f37c76fbb9a281fb551f"

ENDPOINT = "http://127.0.0.1:8899/v1/chat/completions"
JUDGE_MODEL_ID = "gemma-4-31b-it-4bit"

M125_2F_SESSION = Path("/Users/midas/Desktop/cowork/data/session_logs/sess_20260423_134018_69617")
PILOT_LOG = Path("/Users/midas/Desktop/cowork/vault/agent_reports/m125_2_f_pilot_log.md")

JUDGE_SYSTEM = """You are a strict evaluation rubric judge. You apply the following rubric VERBATIM to (query, response) pairs and return a single verdict.

RUBRIC (frozen, M117+ lineage):

strict = response is correct, grounded, specific (cites numbers/entities/paths where applicable), non-hedged. Zero fabrication. No refusal when the answer was in the grounding material.

generous = response is useful and directionally correct — partially correct OR correct-but-hedged OR correct via honest abstention when the answer was NOT in the grounding material.

partial = partially correct but significant specificity gaps (right entity, wrong number) OR partial failure (correct start, drifts off-topic). Partial is NOT a pass condition.

fail = incorrect / fabrication / refused-when-answer-was-in-material / template-leak / non-response.

Additional context you need:
- If the response ABSTAINS ("I don't have information about that") and the ground-truth answer is NOT trivially known to a general-knowledge LLM, classify as generous (honest abstention).
- If the response ABSTAINS but the ground-truth answer IS general knowledge the model should know from training, classify as fail (unjustified refusal).
- If the response fabricates facts not in the ground truth (wrong numbers, invented file paths, invented entity relationships), classify as fail AND set fabrication=true.
- A fabrication flag can also be set on a strict/generous verdict if the response fabricates SIDE claims while still answering the main question correctly.

Return ONLY a single JSON object on one line, no prose, no markdown, no commentary:
{"verdict":"strict|generous|partial|fail","fabrication":true|false,"hallucination_severity":"none|mild|severe","evidence":"<=200 chars from response supporting verdict"}"""


def verify_rubric_sha():
    h = hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()
    if h != RUBRIC_EXPECTED_SHA:
        raise RuntimeError(f"Rubric SHA drift: got {h!r} expected {RUBRIC_EXPECTED_SHA!r}")


def call_judge(query: str, response: str, ground_truth: str, retries: int = 3) -> dict:
    user_msg = (
        f"QUERY: {query}\n\n"
        f"RESPONSE TO JUDGE:\n{response}\n\n"
        f"GROUND TRUTH ANSWER: {ground_truth if ground_truth else '(none — honest abstention is correct)'}\n\n"
        "Apply the rubric. Return the JSON object only."
    )
    payload = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "model": JUDGE_MODEL_ID,
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"].strip()
            m = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", content, re.DOTALL)
            if not m:
                last_err = f"no-json in content: {content[:200]!r}"
                time.sleep(1)
                continue
            parsed = json.loads(m.group(0))
            if parsed.get("verdict") not in ("strict", "generous", "partial", "fail"):
                last_err = f"bad verdict: {parsed!r}"
                time.sleep(1)
                continue
            return parsed
        except Exception as e:
            last_err = repr(e)
            time.sleep(2)
    raise RuntimeError(f"judge call failed after {retries} retries: {last_err}")


def load_pilot_log_responses():
    rows = {}
    txt = PILOT_LOG.read_text()
    rx = re.compile(r"^\| T(\d+) \| (.+?) \| (.+?) \| (strict|generous|partial|fail) \| (\S+) \| ", re.MULTILINE)
    for m in rx.finditer(txt):
        rows[int(m.group(1))] = {
            "query": m.group(2).strip(),
            "verdict": m.group(4),
            "query_class": m.group(5).strip(),
        }
    for n, rec in rows.items():
        p = M125_2F_SESSION / f"turn_{n:04d}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rec["response_text"] = d.get("generation", {}).get("response_text", "")
    return rows


def run_judge_validation(pilot_rows: dict, picks: list[int]):
    matches = 0
    total = 0
    detail = []
    for n in picks:
        rec = pilot_rows.get(n)
        if not rec or not rec.get("response_text"):
            detail.append({"t": n, "skipped": True})
            continue
        gt_hint = "(use rubric judgment based on the response text itself)"
        try:
            j = call_judge(rec["query"], rec["response_text"], gt_hint)
        except Exception as e:
            detail.append({"t": n, "error": repr(e)})
            continue
        total += 1
        match = (j["verdict"] == rec["verdict"])
        if match:
            matches += 1
        detail.append({
            "t": n, "pilot_verdict": rec["verdict"], "judge_verdict": j["verdict"],
            "match": match, "evidence": j.get("evidence", "")[:120],
        })
    return matches, total, detail


def score_all():
    response_files = sorted(RESP_DIR.glob("turn_*.json"))
    if len(response_files) != 135:
        print(f"[WARN] expected 135 response files, found {len(response_files)}")
    out = OUT_JSONL.open("w")
    written = 0
    for p in response_files:
        rec = json.loads(p.read_text())
        turn_id = rec["turn_id"]
        meta = rec.get("corpus_meta", {})
        query = rec["query"]
        response = rec.get("response", "")
        gt = meta.get("ground_truth_answer")
        ambig = meta.get("ground_truth_ambiguous", False)
        qclass = meta.get("query_class")
        gt_in_top5 = rec.get("ground_truth_in_top5", False)
        top5_scores = rec.get("top5_cosine_scores", [])

        if ambig:
            row = {
                "turn_id": turn_id,
                "query_class": qclass,
                "ground_truth_ambiguous": True,
                "verdict": "ambiguous",
                "evidence_snippet": response[:200],
                "fabrication_flag": False,
                "hallucination_severity": "none",
                "scoring_method": "ambiguous_bucket",
                "ground_truth_in_top5": gt_in_top5,
                "top1_cosine": top5_scores[0] if top5_scores else None,
            }
        else:
            try:
                j = call_judge(query, response, gt or "")
                row = {
                    "turn_id": turn_id,
                    "query_class": qclass,
                    "ground_truth_ambiguous": False,
                    "verdict": j["verdict"],
                    "evidence_snippet": (j.get("evidence") or response[:200])[:200],
                    "fabrication_flag": bool(j.get("fabrication", False)),
                    "hallucination_severity": j.get("hallucination_severity", "none"),
                    "scoring_method": "llm_judge_gemma_31b_q4",
                    "ground_truth_in_top5": gt_in_top5,
                    "top1_cosine": top5_scores[0] if top5_scores else None,
                }
            except Exception as e:
                row = {
                    "turn_id": turn_id,
                    "query_class": qclass,
                    "ground_truth_ambiguous": False,
                    "verdict": "fail",
                    "evidence_snippet": f"JUDGE_ERROR: {e!r}",
                    "fabrication_flag": False,
                    "hallucination_severity": "none",
                    "scoring_method": "judge_error_default_fail",
                    "ground_truth_in_top5": gt_in_top5,
                    "top1_cosine": top5_scores[0] if top5_scores else None,
                }
        out.write(json.dumps(row) + "\n")
        written += 1
        if written % 20 == 0:
            print(f"[score] {written}/{len(response_files)}")
    out.close()
    print(f"[score] wrote {written} verdicts -> {OUT_JSONL}")


def aggregate():
    counts = Counter()
    class_counts = Counter()
    class_strict = Counter()
    class_gt_in_top5_scorable = Counter()
    class_gt_in_top5_hits = Counter()
    class_top5_misses = Counter()   # miss = scorable AND gt_in_top5 False
    fabr = 0
    scorable = 0
    ambig_count = 0
    total = 0
    gt_in_top5_hits_scorable = 0
    with OUT_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            total += 1
            v = r["verdict"]
            counts[v] += 1
            qc = r.get("query_class") or "unknown"
            if v == "ambiguous":
                ambig_count += 1
                continue
            scorable += 1
            class_counts[qc] += 1
            if v == "strict":
                class_strict[qc] += 1
            if r.get("fabrication_flag"):
                fabr += 1
            # Ground-truth-in-top5 metrics only count scorable turns.
            class_gt_in_top5_scorable[qc] += 1
            if r.get("ground_truth_in_top5"):
                gt_in_top5_hits_scorable += 1
                class_gt_in_top5_hits[qc] += 1
            else:
                class_top5_misses[qc] += 1

    strict = counts["strict"]
    generous = counts["generous"]
    partial = counts["partial"]
    fail = counts["fail"]
    print("=" * 60)
    print(f"Aggregate (N_total={total}, N_scorable={scorable}, N_ambiguous={ambig_count})")
    if scorable:
        print(f"  strict: {strict} ({strict/scorable*100:.1f}%)")
        print(f"  generous: {generous} ({generous/scorable*100:.1f}%)")
        print(f"  partial: {partial} ({partial/scorable*100:.1f}%)")
        print(f"  fail: {fail} ({fail/scorable*100:.1f}%)")
        print(f"  strict_pass_rate: {strict/scorable:.4f}")
        print(f"  generous_pass_rate: {(strict+generous)/scorable:.4f}")
        print(f"  zero_hallucination_rate: {1 - fabr/scorable:.4f}  (fabrications: {fabr}/{scorable})")
        print(f"  ground_truth_in_top5_pct (scorable): {gt_in_top5_hits_scorable/scorable:.4f} "
              f"({gt_in_top5_hits_scorable}/{scorable})")
    print()
    print("Per-class strict rates (of scorable in that class):")
    per_class_strict = {}
    for cls in sorted(class_counts):
        n = class_counts[cls]
        s = class_strict[cls]
        rate = s / n if n else 0.0
        per_class_strict[cls] = rate
        print(f"  {cls}: {s}/{n} = {rate:.3f}")

    print()
    print("Ground-truth-in-top5 per class (scorable):")
    per_class_gt = {}
    for cls in sorted(class_gt_in_top5_scorable):
        n = class_gt_in_top5_scorable[cls]
        h = class_gt_in_top5_hits[cls]
        rate = h / n if n else 0.0
        per_class_gt[cls] = rate
        print(f"  {cls}: {h}/{n} = {rate:.3f}")

    print()
    print("Top-5 miss distribution (scorable turns where gt NOT in top5):")
    miss_total = sum(class_top5_misses.values())
    per_class_miss_share = {}
    for cls in sorted(class_top5_misses):
        n = class_top5_misses[cls]
        share = n / miss_total if miss_total else 0.0
        per_class_miss_share[cls] = {"count": n, "share_of_misses": share}
        print(f"  {cls}: {n} misses ({share*100:.1f}% of misses)")

    return {
        "total": total, "scorable": scorable, "ambiguous": ambig_count,
        "strict": strict, "generous": generous, "partial": partial, "fail": fail,
        "strict_pass_rate": strict / scorable if scorable else 0.0,
        "generous_pass_rate": (strict + generous) / scorable if scorable else 0.0,
        "zero_hallucination_rate": 1 - fabr / scorable if scorable else 0.0,
        "ground_truth_in_top5_pct": gt_in_top5_hits_scorable / scorable if scorable else 0.0,
        "ground_truth_in_top5_hits": gt_in_top5_hits_scorable,
        "per_class_counts": dict(class_counts),
        "per_class_strict_counts": dict(class_strict),
        "per_class_strict_rates": per_class_strict,
        "per_class_gt_in_top5_scorable": dict(class_gt_in_top5_scorable),
        "per_class_gt_in_top5_hits": dict(class_gt_in_top5_hits),
        "per_class_gt_in_top5_rates": per_class_gt,
        "per_class_top5_miss_counts": dict(class_top5_misses),
        "per_class_top5_miss_share": per_class_miss_share,
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    verify_rubric_sha()
    print(f"[ok] rubric SHA verified: {RUBRIC_EXPECTED_SHA}")

    if mode in ("validate", "all"):
        pilot = load_pilot_log_responses()
        picks = [22, 26, 29, 30, 36, 37, 41, 44, 45, 46, 47, 50, 52, 53, 54, 56]
        picks = [p for p in picks if p in pilot and pilot[p].get("response_text")][:10]
        print(f"[validate] running judge on {len(picks)} pilot turns: {picks}")
        matches, total, detail = run_judge_validation(pilot, picks)
        print(f"[validate] judge matched {matches}/{total} pilot verdicts")
        for d in detail:
            print(f"  T{d.get('t')}: {d}")
        vstate = {"matches": matches, "total": total, "detail": detail}
        (DATA_DIR / "baseline_3_judge_validation.json").write_text(json.dumps(vstate, indent=2))
        if matches < total:
            print(f"[validate] WARN: {matches}/{total} — review baseline_3_judge_validation.json")
        else:
            print("[validate] PASS 10/10 — proceeding with LLM judge")

    if mode in ("score", "all"):
        score_all()

    if mode in ("aggregate", "all"):
        agg = aggregate()
        (DATA_DIR / "baseline_3_aggregate.json").write_text(json.dumps(agg, indent=2))
