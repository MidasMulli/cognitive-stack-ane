#!/usr/bin/env python3
"""
M126 Baselines — Stream E scorer (Full Subconscious).

Two phases:
 1. Judge validation — 10 pilot-log turns from M125.2 F (authoritative verdicts);
    must achieve 10/10 match, else fall back to per-turn llm-judge-with-flag.
 2. Score — apply the judge to all 135 subconscious_responses/turn_*.json files,
    emit subconscious_scored.jsonl + subconscious_aggregate.json.

Rubric source: /Users/midas/Desktop/cowork/data/m126_baselines/rubric_v1.md (SHA256 e9595e...).
Pilot log: vault/agent_reports/m125_2_f_pilot_log.md
Reference: orion-ane/tests/m126_baselines/score_baseline_1.py
"""

import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path("/Users/midas/Desktop/cowork/data/m126_baselines")
RESP_DIR = DATA_DIR / "subconscious_responses"
OUT_JSONL = DATA_DIR / "subconscious_scored.jsonl"
AGG_JSON = DATA_DIR / "subconscious_aggregate.json"
JUDGE_VALIDATION = DATA_DIR / "subconscious_judge_validation.json"
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


def load_pilot_log_responses() -> dict:
    """Map pilot turn number -> {query, verdict, query_class, response_text}."""
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


def run_judge_validation(pilot_rows: dict, picks: list[int]) -> tuple[int, int, list]:
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
        turn_id = rec["turn_id_in_corpus"]
        meta = rec.get("corpus_meta", {})
        query = rec["corpus_query"]
        response = rec.get("midas_response", "") or ""
        gt = meta.get("ground_truth_answer")
        ambig = meta.get("ground_truth_ambiguous", False)
        qclass = meta.get("query_class")

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
                }
        out.write(json.dumps(row) + "\n")
        written += 1
        if written % 20 == 0:
            print(f"[score] {written}/{len(response_files)}")
    out.close()
    print(f"[score] wrote {written} verdicts -> {OUT_JSONL}")


def aggregate():
    from collections import Counter
    counts = Counter()
    class_counts = Counter()
    class_strict = Counter()
    class_generous = Counter()
    fabr = 0
    scorable = 0
    ambig_count = 0
    total = 0
    with OUT_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            total += 1
            v = r["verdict"]
            counts[v] += 1
            if v == "ambiguous":
                ambig_count += 1
                continue
            scorable += 1
            qc = r.get("query_class") or "unknown"
            class_counts[qc] += 1
            if v == "strict":
                class_strict[qc] += 1
            if v in ("strict", "generous"):
                class_generous[qc] += 1
            if r.get("fabrication_flag"):
                fabr += 1
    strict = counts["strict"]
    generous = counts["generous"]
    partial = counts["partial"]
    fail = counts["fail"]
    result = {
        "total": total, "scorable": scorable, "ambiguous": ambig_count,
        "strict": strict, "generous": generous, "partial": partial, "fail": fail,
        "strict_pass_rate": strict / scorable if scorable else 0.0,
        "generous_pass_rate": (strict + generous) / scorable if scorable else 0.0,
        "zero_hallucination_rate": 1 - fabr / scorable if scorable else 0.0,
        "per_class_counts": dict(class_counts),
        "per_class_strict": dict(class_strict),
        "per_class_generous": dict(class_generous),
        "per_class_strict_rates": {
            c: (class_strict[c] / class_counts[c]) for c in class_counts if class_counts[c] > 0
        },
    }
    print("=" * 60)
    print(f"Stream E aggregate (N_total={total}, N_scorable={scorable}, N_ambiguous={ambig_count})")
    print(f"  strict: {strict} ({strict/scorable*100:.1f}% of scorable)" if scorable else "  n/a")
    print(f"  generous(+strict): {strict+generous} ({(strict+generous)/scorable*100:.1f}%)")
    print(f"  partial: {partial} | fail: {fail}")
    print(f"  strict_pass_rate: {result['strict_pass_rate']:.4f}")
    print(f"  generous_pass_rate: {result['generous_pass_rate']:.4f}")
    print(f"  zero_hallucination_rate: {result['zero_hallucination_rate']:.4f}  (fabrications: {fabr}/{scorable})")
    print("Per-class strict rates:")
    for cls in sorted(class_counts):
        n = class_counts[cls]
        s = class_strict[cls]
        print(f"  {cls}: {s}/{n} = {s/n:.3f}")
    return result


def compute_m125_2f_delta():
    """For turns whose corpus turn_id starts with 'm125_2f_', compare Stream E verdicts
    to pilot log verdicts on the same corresponding T<n>."""
    pilot = load_pilot_log_responses()
    same = 0
    improved = 0
    regressed = 0
    overlap_n = 0
    e_strict_overlap = 0
    pilot_strict_overlap = 0
    rank = {"fail": 0, "partial": 1, "generous": 2, "strict": 3, "ambiguous": None}
    detail = []
    with OUT_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            tid = r["turn_id"]
            if not tid.startswith("m125_2f_"):
                continue
            m = re.match(r"m125_2f_t(\d+)$", tid)
            if not m:
                continue
            tnum = int(m.group(1))
            pilot_rec = pilot.get(tnum)
            if not pilot_rec:
                continue
            overlap_n += 1
            p_verdict = pilot_rec["verdict"]
            e_verdict = r["verdict"]
            if e_verdict == "ambiguous":
                continue  # exclude
            if e_verdict == "strict":
                e_strict_overlap += 1
            if p_verdict == "strict":
                pilot_strict_overlap += 1
            if p_verdict == e_verdict:
                same += 1
            else:
                pr = rank.get(p_verdict)
                er = rank.get(e_verdict)
                if pr is not None and er is not None:
                    if er > pr:
                        improved += 1
                    elif er < pr:
                        regressed += 1
            detail.append({"t": tnum, "pilot": p_verdict, "stream_e": e_verdict})
    e_strict_rate = (e_strict_overlap / overlap_n) if overlap_n else 0.0
    pilot_strict_rate = (pilot_strict_overlap / overlap_n) if overlap_n else 0.0
    delta_pp = (e_strict_rate - pilot_strict_rate) * 100.0
    return {
        "overlap_n": overlap_n,
        "same_verdict": same,
        "improved": improved,
        "regressed": regressed,
        "stream_e_strict_rate_on_overlap": e_strict_rate,
        "pilot_m125_2f_strict_rate_on_overlap": pilot_strict_rate,
        "delta_pp_vs_m125_2f": delta_pp,
        "detail_sample": detail[:20],
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    verify_rubric_sha()
    print(f"[ok] rubric SHA verified: {RUBRIC_EXPECTED_SHA}")

    if mode in ("validate", "all"):
        pilot = load_pilot_log_responses()
        picks = [22, 26, 29, 30, 36, 37, 40, 41, 44, 45, 46, 47, 50, 52, 53, 54, 55, 56, 58]
        picks = [p for p in picks if p in pilot and pilot[p].get("response_text")][:10]
        print(f"[validate] running judge on {len(picks)} pilot turns: {picks}")
        matches, total, detail = run_judge_validation(pilot, picks)
        print(f"[validate] judge matched {matches}/{total} pilot verdicts")
        for d in detail:
            print(f"  T{d.get('t')}: {d}")
        vstate = {"matches": matches, "total": total, "detail": detail}
        JUDGE_VALIDATION.write_text(json.dumps(vstate, indent=2))
        if matches < total:
            print(f"[validate] WARN: {matches}/{total} != 10/10 — proceeding with judge but note fallback limit")
        else:
            print("[validate] PASS 10/10 — proceeding with LLM judge")

    if mode in ("score", "all"):
        score_all()

    if mode in ("aggregate", "all"):
        agg = aggregate()
        delta = compute_m125_2f_delta()
        agg["m125_2f_overlap_analysis"] = delta
        AGG_JSON.write_text(json.dumps(agg, indent=2))
        print("[m125_2f_delta]")
        print(json.dumps(delta, indent=2))
