"""M121 Stream B — Tag-index extension tests.

Scope: verify that the builder now emits `published_repo` + `standing_rule`
tag categories, that the Enumeration dispatcher round-trips them end-to-end
via enumerate_by_tag(), and that pre-existing tag categories (dead_path,
measurement, decision, plan, correction, preference, open_lead) still work.

Canonical sources:
  published_repo — CLAUDE.md "Public repos current as of Main 26: orion-ane,
                   ane-compiler, ngram-engine, subconscious"
  standing_rule  — vault/knowledge/standing_rules.md A (rules 1-8) +
                   B (B1-B12) sections = 20 rules total

Run:
  ~/.mlx-env/bin/python3 orion-ane/tests/test_m121_b_tag_index.py

Exit code 0 = all pass, non-zero = failure (with per-test trace).
"""
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENT = os.path.normpath(os.path.join(_HERE, "..", "agent"))
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

from enumeration_retrieval import (  # noqa: E402
    shape_dispatch_signal,
    enumerate_by_tag,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TAG_INDEX_PATH = _REPO_ROOT / "data" / "tag_index.json"


def _load_tag_index() -> dict:
    return json.loads(_TAG_INDEX_PATH.read_text(encoding="utf-8"))


# ── published_repo category ────────────────────────────────────────────────

def test_published_repo_present_in_index():
    """tag_index.json contains a published_repo key with ≥4 entries
    (CLAUDE.md Main 26 canonical: orion-ane, ane-compiler, ngram-engine,
    subconscious)."""
    idx = _load_tag_index()
    assert "published_repo" in idx, "published_repo missing from tag_index.json"
    recs = idx["published_repo"]
    assert len(recs) >= 4, f"expected ≥4 published repos, got {len(recs)}"


def test_published_repo_canonical_entities():
    """All four Main 26 canonical repo names are present."""
    idx = _load_tag_index()
    entities = {r.get("entity") for r in idx["published_repo"]}
    expected = {"orion-ane", "ane-compiler", "ngram-engine", "subconscious"}
    missing = expected - entities
    assert not missing, f"missing canonical repos: {missing}"


def test_published_repo_session_label_canonical():
    """Every published_repo record points back to CLAUDE.md as source."""
    idx = _load_tag_index()
    for r in idx["published_repo"]:
        assert r.get("session_label") == "CLAUDE.md", \
            f"non-canonical source on {r.get('entity')}: {r.get('session_label')}"


# ── standing_rule category ─────────────────────────────────────────────────

def test_standing_rule_present_in_index():
    """tag_index.json contains standing_rule key with the 8 A-rules +
    12 B-rules = 20 entries (per vault/knowledge/standing_rules.md M83)."""
    idx = _load_tag_index()
    assert "standing_rule" in idx, "standing_rule missing from tag_index.json"
    recs = idx["standing_rule"]
    assert len(recs) == 20, f"expected 20 rules (8 A + 12 B), got {len(recs)}"


def test_standing_rule_a_section_complete():
    """Rules 1-8 (A section) all present."""
    idx = _load_tag_index()
    a_entities = {r.get("entity") for r in idx["standing_rule"]
                  if r.get("section") == "A"}
    expected = {f"Rule {n}" for n in range(1, 9)}
    missing = expected - a_entities
    assert not missing, f"missing A-section rules: {missing}"


def test_standing_rule_b_section_complete():
    """B1-B12 (B section) all present."""
    idx = _load_tag_index()
    b_entities = {r.get("entity") for r in idx["standing_rule"]
                  if r.get("section") == "B"}
    expected = {f"B{n}" for n in range(1, 13)}
    missing = expected - b_entities
    assert not missing, f"missing B-section rules: {missing}"


# ── T88 end-to-end: dispatcher → tag index → records ──────────────────────

def test_t88_end_to_end_returns_repo_list():
    """T88 gold-path: query 'enumerate all published repos under my name'
    dispatches to Enumeration, resolves to published_repo tag, and returns
    the 4 canonical repo records."""
    q = "enumerate all published repos under my name"

    # Step 1: shape classifier fires
    sig = shape_dispatch_signal(q)
    assert sig is not None, "T88 query did not dispatch to any shape"
    assert sig["shape"] == "enumeration"
    assert sig["tag"] == "published_repo"
    assert sig["backed"] is True, "published_repo tag not backed — M121 B gap still open"

    # Step 2: enumerate_by_tag returns records (not None — K9 gap closed)
    result = enumerate_by_tag(q)
    assert result is not None, "enumerate_by_tag returned None — backing gap still open"
    assert result["tag"] == "published_repo"
    assert result["count"] >= 4, f"expected ≥4 records, got {result['count']}"

    # Step 3: formatted output lists the canonical repos
    joined = "\n".join(result["records"])
    for repo in ("orion-ane", "ane-compiler", "ngram-engine", "subconscious"):
        assert repo in joined, f"canonical repo {repo!r} missing from T88 output:\n{joined}"


def test_standing_rules_end_to_end():
    """End-to-end standing_rule: 'what are the standing rules?' returns
    20 rule records (8 A + 12 B)."""
    q = "what are the standing rules?"
    result = enumerate_by_tag(q)
    assert result is not None, "standing_rule query returned None"
    assert result["tag"] == "standing_rule"
    assert result["count"] == 20, f"expected 20 rules, got {result['count']}"


# ── Edge: non-existent / malformed tag ─────────────────────────────────────

def test_nonexistent_tag_clean_empty_response():
    """Query shape matches enumeration but references no known tag —
    returns None cleanly (no crash) per existing enumerate_by_tag
    contract."""
    q = "list all purple elephants"  # no known tag keyword matches
    sig = shape_dispatch_signal(q)
    # Shape classifier returns None when no tag keyword matches
    assert sig is None, f"unexpected shape match on non-tag query: {sig}"
    result = enumerate_by_tag(q)
    assert result is None, f"unexpected result on non-tag query: {result}"


def test_query_with_empty_tag_bucket():
    """If a query matches the enumeration intent + a tag keyword but the
    tag bucket is empty, enumerate_by_tag returns None (per existing
    contract: `if not records: return None`). This is tested via a fresh
    tag that isn't in the index — use the same non-existent-tag path."""
    q = "enumerate every unicorn"  # 'enumerate' + no tag keyword match
    result = enumerate_by_tag(q)
    assert result is None


# ── Regression: existing tags still work ───────────────────────────────────

def test_regression_dead_path():
    """Pre-existing dead_path tag still routes + returns records."""
    q = "list all dead paths"
    result = enumerate_by_tag(q)
    assert result is not None, "dead_path regression: returned None"
    assert result["tag"] == "dead_path"
    assert result["count"] > 0


def test_regression_measurement():
    """Pre-existing measurement tag still routes + returns records."""
    q = "how many measurements do we have?"
    result = enumerate_by_tag(q)
    assert result is not None, "measurement regression: returned None"
    assert result["tag"] == "measurement"
    assert result["count"] > 0


def test_regression_decision():
    """Pre-existing decision tag still routes + returns records."""
    q = "list every decision"
    result = enumerate_by_tag(q)
    assert result is not None, "decision regression: returned None"
    assert result["tag"] == "decision"
    assert result["count"] > 0


def test_regression_open_lead():
    """Pre-existing open_lead tag still routes + returns records."""
    q = "list every open lead"
    result = enumerate_by_tag(q)
    assert result is not None, "open_lead regression: returned None"
    assert result["tag"] == "open_lead"
    assert result["count"] > 0


def test_regression_all_prior_categories_nonempty():
    """All seven pre-M121 tag categories still have non-zero records."""
    idx = _load_tag_index()
    prior_tags = ("dead_path", "decision", "measurement", "preference",
                  "plan", "correction", "open_lead")
    for tag in prior_tags:
        assert tag in idx, f"{tag} disappeared from tag_index.json"
        assert len(idx[tag]) > 0, f"{tag} bucket now empty"


# ── Runner ─────────────────────────────────────────────────────────────────

_TESTS = [
    test_published_repo_present_in_index,
    test_published_repo_canonical_entities,
    test_published_repo_session_label_canonical,
    test_standing_rule_present_in_index,
    test_standing_rule_a_section_complete,
    test_standing_rule_b_section_complete,
    test_t88_end_to_end_returns_repo_list,
    test_standing_rules_end_to_end,
    test_nonexistent_tag_clean_empty_response,
    test_query_with_empty_tag_bucket,
    test_regression_dead_path,
    test_regression_measurement,
    test_regression_decision,
    test_regression_open_lead,
    test_regression_all_prior_categories_nonempty,
]


def _run() -> int:
    failed = 0
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
