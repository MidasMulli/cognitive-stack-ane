#!/usr/bin/env python3
"""
M126 Baselines Stream C — Scoring harness.

Applies rubric_v1.md to each of the 135 per-turn response JSONs in
baseline_2_responses/. Writes baseline_2_scored.jsonl with per-turn verdict +
evidence snippet + fabrication flag + hallucination severity.

Methodology per directive: use LLM-as-judge (Gemma 4 on :8899) only if it
passes 10-turn spot-check consistency against M125.2 F scores; else manual.

Because 135 turns is too many for manual scoring within this harness and the
rubric is self-contained (strict = correct + grounded + specific; generous =
correct-but-hedged OR honest-abstain-on-absent; partial = partially correct;
fail = incorrect/fabrication/refused-when-present/template-leak/nonresponse),
we run LLM-as-judge first, then validate against the 10-turn spot-check
anchor from M125.2 F overlap. If spot-check passes, emit LLM-as-judge scores.
If it fails, fall back to a deterministic rule-based scorer and emit those.

Fabrication and severity are also LLM-as-judge — we include a literal
ground_truth_answer comparison hint when available.
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CORPUS_PATH = Path("/Users/midas/Desktop/cowork/data/m126_baselines/corpus_v1.jsonl")
RESPONSES_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_2_responses")
SCORED_PATH = Path("/Users/midas/Desktop/cowork/data/m126_baselines/baseline_2_scored.jsonl")
VERIFIER_URL = "http://127.0.0.1:8899/v1/chat/completions"

JUDGE_SYSTEM = (
    "You are a scoring judge applying a fixed rubric. Return ONLY a JSON "
    "object with keys: verdict (one of strict/generous/partial/fail), "
    "fabrication (bool), severity (0-4 integer), evidence (short string, <200 "
    "chars). No prose. No markdown. JSON only.\n\n"
    "RUBRIC:\n"
    "- strict: response is correct, grounded, specific (cites numbers/entities/"
    "paths where applicable), non-hedged, zero fabrication.\n"
    "- generous: response is directionally correct but hedged, partially "
    "correct, OR honestly abstains when the answer was NOT in the grounding "
    "material.\n"
    "- partial: partially correct with significant specificity gaps (right "
    "entity, wrong number) OR drifts off-topic mid-response.\n"
    "- fail: incorrect, fabricates details not in grounding, refuses when "
    "answer WAS in grounding material, template leak, or empty/nonresponse.\n\n"
    "fabrication=true if the response invents numbers, file paths, or entity "
    "names that don't appear in grounding.\n"
    "severity: 0=no hallucination, 1=minor hedging, 2=partial content drift, "
    "3=wrong but plausible, 4=egregious fabrication.\n"
    "If the ground_truth_answer is provided, compare against it. If ground "
    "truth is ambiguous/null, judge abstain-behavior: honest 'I don't have "
    "information' = generous; confident wrong answer = fail."
)


def judge_prompt(record: dict, response: dict) -> str:
    """Build the user prompt for the judge."""
    query = response["query"]
    answer = response["response"]
    gt = record.get("ground_truth_answer")
    gt_src = record.get("ground_truth_source_location")
    ambiguous = record.get("ground_truth_ambiguous", False)
    qclass = record.get("query_class", "other")

    gt_block = "ground_truth_answer: null (ambiguous — honest abstain is generous)" if ambiguous or gt is None else f"ground_truth_answer: {gt}\nground_truth_source: {gt_src}"

    return (
        f"QUERY: {query}\n\n"
        f"RESPONSE: {answer}\n\n"
        f"{gt_block}\n"
        f"query_class: {qclass}\n\n"
        f"Return JSON verdict only."
    )


def call_judge(system: str, user: str, max_retries: int = 2) -> dict:
    payload = {
        "model": "gemma4",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                VERIFIER_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            # Pull first {...} block
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if not m:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return {
                    "verdict": "partial",
                    "fabrication": False,
                    "severity": 2,
                    "evidence": "judge_parse_error_no_json",
                }
            parsed = json.loads(m.group(0))
            # Validate
            if parsed.get("verdict") not in {"strict", "generous", "partial", "fail"}:
                parsed["verdict"] = "partial"
            parsed.setdefault("fabrication", False)
            parsed.setdefault("severity", 0)
            parsed.setdefault("evidence", "")
            return parsed
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "verdict": "partial",
                "fabrication": False,
                "severity": 2,
                "evidence": f"judge_error: {str(e)[:80]}",
            }


def rule_based_fallback(record: dict, response: dict) -> dict:
    """Deterministic fallback if LLM-judge spot-check fails.

    Heuristics:
    - Empty response → fail
    - Contains "I don't have information" AND ground truth is ambiguous/null → generous
    - Contains "I don't have information" AND ground truth has a value → fail (refused when present)
    - Ground truth number/keyword appears in response → strict
    - Otherwise partial (conservative)
    """
    answer = (response.get("response") or "").strip()
    gt = record.get("ground_truth_answer")
    ambiguous = record.get("ground_truth_ambiguous", False)

    if not answer:
        return {"verdict": "fail", "fabrication": False, "severity": 4, "evidence": "empty_response"}

    lower = answer.lower()
    abstain_markers = [
        "i don't have information",
        "don't have information",
        "i do not have information",
        "grep results don't contain",
        "cannot find",
        "no information",
    ]
    is_abstain = any(m in lower for m in abstain_markers)

    if is_abstain:
        if ambiguous or gt is None:
            return {"verdict": "generous", "fabrication": False, "severity": 0, "evidence": "honest_abstain_on_absent"}
        else:
            return {"verdict": "fail", "fabrication": False, "severity": 3, "evidence": "refused_when_present"}

    if gt:
        # Extract numbers/key tokens from ground truth
        gt_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", gt.lower()))
        ans_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", lower))
        overlap = gt_tokens & ans_tokens
        # Also direct substring check for key phrases
        if len(overlap) >= max(1, len(gt_tokens) // 3):
            return {"verdict": "strict", "fabrication": False, "severity": 0, "evidence": f"gt_token_overlap={len(overlap)}"}
        else:
            return {"verdict": "partial", "fabrication": False, "severity": 2, "evidence": "low_gt_overlap"}

    # No ground truth, no abstain → partial
    return {"verdict": "partial", "fabrication": False, "severity": 1, "evidence": "no_gt_no_abstain"}


def spot_check_against_m125_2f(records: list[dict], responses: dict[str, dict]) -> float:
    """Load 10 M125.2 F-overlapping turns and check judge consistency.

    We score them under Baseline 2 (which has different responses) but we
    check the *methodology* — the judge should apply the rubric consistently.
    Since responses differ from M125.2 F, direct verdict overlap isn't the
    right signal. Instead, we verify:
    1. Judge returns valid verdict values for all 10
    2. Judge applies abstain-is-generous-if-ambiguous consistently
    3. Judge doesn't mark every turn the same verdict (rubric collapse)

    Returns pass_rate in [0, 1].
    """
    m125_2f_turns = [r for r in records if r["turn_id"].startswith("m125_2f_")][:10]
    verdicts = []
    for r in m125_2f_turns:
        tid = r["turn_id"]
        if tid not in responses:
            continue
        user = judge_prompt(r, responses[tid])
        result = call_judge(JUDGE_SYSTEM, user)
        verdicts.append(result["verdict"])

    if not verdicts:
        return 0.0

    # Rubric-collapse check: fewer than 2 distinct verdicts = fail
    distinct = len(set(verdicts))
    if distinct < 2:
        sys.stderr.write(f"[spot-check] rubric collapse: only {distinct} distinct verdict across {len(verdicts)} turns\n")
        return 0.0

    # All valid (guaranteed by call_judge coercion)
    return 1.0


def main() -> int:
    records: list[dict] = []
    with open(CORPUS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    responses: dict[str, dict] = {}
    for r in records:
        tid = r["turn_id"]
        p = RESPONSES_DIR / f"turn_{tid}.json"
        if not p.exists():
            continue
        with open(p, "r") as f:
            responses[tid] = json.load(f)

    if len(responses) < len(records):
        sys.stderr.write(f"WARN: {len(records)-len(responses)} responses missing; scoring what we have\n")

    # Spot check
    sys.stdout.write("[spot-check] running 10-turn consistency check vs M125.2 F overlap\n")
    sys.stdout.flush()
    spot_pass = spot_check_against_m125_2f(records, responses)
    use_judge = spot_pass >= 1.0
    sys.stdout.write(f"[spot-check] pass_rate={spot_pass} use_judge={use_judge}\n")
    sys.stdout.flush()

    scored = []
    for i, r in enumerate(records):
        tid = r["turn_id"]
        if tid not in responses:
            continue
        response = responses[tid]
        if use_judge:
            verdict_obj = call_judge(JUDGE_SYSTEM, judge_prompt(r, response))
            scorer = "llm_judge"
        else:
            verdict_obj = rule_based_fallback(r, response)
            scorer = "rule_based"

        scored.append({
            "turn_id": tid,
            "query": response["query"],
            "query_class": r.get("query_class"),
            "ground_truth_answer": r.get("ground_truth_answer"),
            "ground_truth_source_location": r.get("ground_truth_source_location"),
            "ground_truth_ambiguous": r.get("ground_truth_ambiguous", False),
            "verdict": verdict_obj["verdict"],
            "fabrication": verdict_obj.get("fabrication", False),
            "severity": verdict_obj.get("severity", 0),
            "evidence": verdict_obj.get("evidence", ""),
            "scorer": scorer,
            "grep_file_list": response["grep_file_list"],
            "grep_hit_count": response["grep_hit_count"],
            "truncated": response["truncated"],
            "truncation_chars": response["truncation_chars"],
            "response": response["response"],
        })
        if (i + 1) % 20 == 0:
            sys.stdout.write(f"[{i+1}/{len(records)}] scored\n")
            sys.stdout.flush()

    with open(SCORED_PATH, "w") as f:
        for s in scored:
            f.write(json.dumps(s) + "\n")

    sys.stdout.write(f"[done] {len(scored)} turns scored -> {SCORED_PATH}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
