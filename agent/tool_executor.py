"""
Layer 3: Tool Executor.

Takes (tool_name, args_dict) from the router and executes it.
No LLM calls. Pure function dispatch.

Returns a plain-text result string ready to hand to the synthesizer.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime

# ── Dependencies from existing agent infrastructure ──────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
PLAYBOOK_PATH = os.path.join(VAULT_PATH, "midas/playbook.md")
CLAUDE_INBOX = os.path.join(VAULT_PATH, "midas/claude-inbox.md")

# Lazy-init singletons (set by agent.py at boot)
_memory = None
_browser = None


def set_memory(bridge):
    global _memory
    _memory = bridge


def set_browser(bridge):
    global _browser
    _browser = bridge


# ── Vault (read-only) ───────────────────────────────────────────────────────

def _useful_len(text: str) -> int:
    """Main 55 P1a: measure 'useful' content length — strip YAML frontmatter
    and whitespace-only lines before counting."""
    if not text:
        return 0
    # Strip leading YAML frontmatter ( --- ... --- )
    t = text.lstrip()
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + 4:]
    # Collapse whitespace
    t = "\n".join(line for line in t.splitlines() if line.strip())
    return len(t.strip())


# Main 55 P1d: scope qualifiers for dead-path queries. When a user asks
# about dead paths AND names a subsystem, re-rank results post-retrieval
# so the scope-matching rows float to the top. The CLAUDE.md dead-paths
# table has rows scoped to each subsystem; naive keyword scoring returns
# adjacent-but-wrong-scope rows (T2 failure: spec-decode dead paths when
# the question asked for KV-cache dead paths).
_DEADPATH_TRIGGERS = ("dead path", "dead paths", "killed path", "parked path")
_DEADPATH_SCOPES = {
    "kv cache":    ("kv cache", "kv-cache", "kvcache", "iosurface", "slc cache",
                    "slc optimization", "cache-swap", "cache swap",
                    "ctrl2", "cache hint"),
    "spec decode": ("spec decode", "speculative", "drafter", "eagle",
                    "n-gram", "ngram", "pard", "early exit", "self-spec"),
    "ane":         ("ane ", "neural engine", "_anevirtual", "ane attention",
                    "ane 1b", "ane slot", "ane standalone", "ane bandwidth",
                    "espresso", "aned"),
    "slc":         ("slc ", "cache hint", "ctrl2", "dcs ", "way mask"),
    "hardware":    ("tb5", "dma", "mie", "emte", "n1 wireless", "amc ",
                    "media engine", "amx", "gpu concurrent", "metal concurrent"),
    "agent":       ("midas", "router", "layer1", "layer2", "layer3",
                    "tool executor", "playbook"),
    "retrieval":   ("recall", "retrieval", "embedding", "chromadb",
                    "localmemorystore", "memory store"),
}


def _scope_rerank(matches: list, query: str) -> list:
    """Main 55 P1d: if query is a scope-qualified dead-path question,
    re-rank matches so snippets mentioning the scope's marker terms
    dominate. Non-destructive: if no scope markers match anywhere,
    returns matches unchanged.
    """
    q_low = query.lower()
    if not any(t in q_low for t in _DEADPATH_TRIGGERS):
        return matches
    scope_terms: tuple = ()
    for scope_key, terms in _DEADPATH_SCOPES.items():
        if scope_key in q_low:
            scope_terms = terms
            break
    if not scope_terms:
        return matches
    rescored = []
    for m in matches:
        bonus = 0
        for s in m.get("snippets", []):
            s_low = s.lower()
            for term in scope_terms:
                if term in s_low:
                    bonus += 3
        m = dict(m)  # shallow copy — don't mutate caller's dict
        m["_score"] = m.get("_score", 0) + bonus
        rescored.append(m)
    rescored.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return rescored


def _vault_read(path: str = "", query: str = "") -> dict:
    """Read vault content. Prefers snippet-based search over full-file dumps.

    Three modes:
      query="search term" → multi-file snippet search (best for specific questions)
      path="Roadmap.md" → smart read (extract active sections, skip completed/dead)
      path="" → vault structure listing
    """
    import glob as globmod

    # Main 46: targeted file search. When both path and query are
    # provided, search within the specific file for the query terms.
    # Used by session-indexed retrieval to search session_milestones.md.
    if query and path:
        target = os.path.join(VAULT_PATH, path)
        if os.path.isfile(target):
            with open(target, "r") as f:
                content = f.read()
            # Find the section matching the query. Support OR-expanded
            # multi-session queries from _session_query_args P1b: split
            # on " OR " and match if ANY term appears in the section.
            sections = content.split("\n## ")
            query_lower = query.lower()
            or_terms = [t.strip() for t in query_lower.split(" or ")
                        if t.strip()]
            if not or_terms:
                or_terms = [query_lower]
            matched_sections = []
            for sec in sections:
                sec_low = sec.lower()
                if any(t in sec_low for t in or_terms):
                    matched_sections.append("## " + sec if not sec.startswith("#") else sec)
            if matched_sections:
                result_text = "\n\n".join(s.strip() for s in matched_sections)
                first_result = {"query": query, "file": path,
                                "matches": [{"file": path, "snippets": [result_text[:3000]]}]}
                # Main 55 P1a: widen-on-sparse fallback. If the targeted
                # file read returned thin content, ALSO fire a broad
                # vault_search and merge the broader hits. "Thin" = <500
                # useful chars (frontmatter + whitespace stripped).
                widen_on_sparse = _useful_len(result_text) < 500
                # Main 57 P3: widen-on-topic-qualifier. Even when the
                # section match is substantive (≥500 chars), if the
                # query contains topic terms beyond the session anchor
                # (e.g., "Main 54 mechanism audit findings" — audit
                # detail lives in agent_reports/, not session_milestones
                # summary), also fire a cross-dir search. Trigger fires
                # only when path is the session_milestones default from
                # router._session_query_args so user-supplied path reads
                # still return exact content without noise.
                widen_on_topic = False
                if (path == "knowledge/session_milestones.md"
                        and not widen_on_sparse):
                    # Strip the session anchor and stopwords, count
                    # remaining content words. 2+ means the user is
                    # asking for specific detail, not a summary.
                    _anchor_re = re.compile(
                        r'\b(?:main|session|m)\s*\d{1,3}\b',
                        re.IGNORECASE)
                    _topic_stop = {
                        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to',
                        'for', 'from', 'with', 'by', 'is', 'are', 'was',
                        'were', 'be', 'been', 'have', 'has', 'had', 'do',
                        'does', 'did', 'what', 'which', 'who', 'how',
                        'why', 'when', 'where', 'that', 'this', 'it',
                        'its', 'we', 'our', 'they', 'them', 'you',
                        'your', 'my', 'say', 'said', 'says', 'tell',
                        'show', 'list', 'about', 'any', 'some', 'all',
                        'and', 'or', 'but', 'so', 'not', 'no', 'yes',
                        'just', 'only', 'also',
                    }
                    _q_stripped = _anchor_re.sub(" ", query.lower())
                    _topic_words = [
                        w for w in re.findall(r'\w+', _q_stripped)
                        if w not in _topic_stop and not w.isdigit()
                        and len(w) > 2
                    ]
                    widen_on_topic = len(_topic_words) >= 2
                if widen_on_sparse or widen_on_topic:
                    reason = ("sparse" if widen_on_sparse
                              else f"topic-qualifier ({len(_topic_words)} terms)")
                    print(f"[vault_read] {reason} path+query hit "
                          f"({_useful_len(result_text)} chars) on {path}, "
                          f"widening to cross-dir search", flush=True)
                    broader = _vault_read(path="", query=query)
                    broader_matches = broader.get("matches", [])
                    if broader_matches:
                        seen = {path}
                        merged = first_result["matches"][:]
                        for bm in broader_matches:
                            if bm.get("file") not in seen:
                                merged.append(bm)
                                seen.add(bm.get("file"))
                        first_result["matches"] = merged[:8]
                        first_result["_widened"] = True
                        first_result["_widen_reason"] = reason
                        print(f"[vault_read] widen hit: +{len(merged) - 1} files "
                              f"from cross-dir search", flush=True)
                return first_result
            # Main 55 P1c: path+query gave ZERO matches — fall back to
            # cross-directory search rather than abstaining. T5 failure
            # mode: path='knowledge/session_milestones.md' but the answer
            # lived in agent_reports/.
            print(f"[vault_read] no section match for '{query}' in {path}; "
                  f"falling back to cross-dir vault_search", flush=True)
            fallback = _vault_read(path="", query=query)
            fallback_matches = fallback.get("matches", [])
            if fallback_matches:
                print(f"[vault_read] fallback hit: {len(fallback_matches)} files "
                      f"from cross-dir search", flush=True)
                return {"query": query, "file": path,
                        "matches": fallback_matches,
                        "_widened_from_path_miss": True,
                        "note": f"path '{path}' had no match; widened to cross-dir"}
            return {"query": query, "file": path, "matches": [],
                    "note": f"No section matching '{query}' in {path}"}

    if query:
        # Multi-file keyword search with stopword filtering and
        # knowledge/ directory boost. Main 40 fix: the old code
        # dropped numbers ≤2 chars ("38" from "main 38") and ranked
        # by snippet count rather than keyword coverage, causing
        # session-identifier queries to match the wrong session.
        _STOP = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
            'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'not',
            'so', 'if', 'than', 'that', 'this', 'it', 'its', 'we',
            'our', 'they', 'them', 'what', 'which', 'who', 'how',
            'why', 'when', 'where', 'all', 'each', 'every', 'some',
            'too', 'very', 'just', 'about', 'also', 'any', 'up',
            'out', 'off', 'list', 'explain', 'describe', 'tell',
            'show', 'give', 'me', 'us', 'you', 'your', 'my',
        }
        # Keep numbers (even short ones like "38") and meaningful words
        query_words = [w.lower() for w in query.split()
                       if w.lower() not in _STOP and len(w) > 0]
        if not query_words:
            query_words = [w.lower() for w in query.split()[:4]]

        matches = []
        for md_file in globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True):
            rel = os.path.relpath(md_file, VAULT_PATH)
            if rel.startswith("memory/") or rel.startswith("subconscious/"):
                continue
            try:
                with open(md_file, "r") as f:
                    content = f.read()
                content_lower = content.lower()
                # Score by keyword coverage
                hits = sum(1 for w in query_words if w in content_lower)
                if hits == 0:
                    continue
                # Boost knowledge/ files (curated authoritative), deprioritize
                # research/ files (exploratory, often tangential). Main 41:
                # Ray-Ban research files were outranking ANE canonical data.
                if rel.startswith("knowledge/"):
                    score = hits + 3
                elif rel.startswith("research/"):
                    score = hits - 1
                else:
                    score = hits
                # Main 57 P3: filename-match boost. When query words appear
                # in the file's basename, the file's topic is explicitly
                # about that anchor. Without this, `agent_reports/main54_
                # phase1_mechanism_inventory.md` scores only 2 (hits alone)
                # while `knowledge/enricher_architecture.md` scores 5 (1 hit +
                # 3 knowledge boost) despite being off-topic. Results
                # observed in M57 live validation T15/T17 where M54 audit
                # content was ranked out of the top-8 by tangential
                # knowledge/ files. +5 per word match in basename.
                _fn_base = os.path.basename(rel).lower().replace("_", " ").replace("-", " ")
                fn_hits = sum(1 for w in query_words if len(w) > 1 and w in _fn_base)
                if fn_hits:
                    score += fn_hits * 5
                lines = content.split("\n")
                snippets = []
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(w in line_lower for w in query_words):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end]).strip()
                        if len(snippet) > 20:
                            snippets.append(snippet)
                if snippets:
                    matches.append({
                        "file": rel,
                        "snippets": snippets[:5],
                        "_score": score,
                    })
            except Exception:
                continue
        # Sort by score (keyword coverage + knowledge boost)
        matches.sort(key=lambda m: m.get("_score", 0), reverse=True)
        # Main 55 P1d: scope-qualified dead-path rerank.
        matches = _scope_rerank(matches, query)
        # Cap total output to ~3000 chars
        result_matches = []
        total_len = 0
        for m in matches[:8]:
            trimmed_snippets = []
            for s in m["snippets"]:
                if total_len + len(s) > 3000:
                    break
                trimmed_snippets.append(s)
                total_len += len(s)
            if trimmed_snippets:
                result_matches.append({"file": m["file"], "snippets": trimmed_snippets})
        return {"query": query, "matches": result_matches}

    if not path:
        structure = {}
        for md_file in sorted(globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True)):
            rel = os.path.relpath(md_file, VAULT_PATH)
            if rel.startswith("memory/"):
                continue
            parts = rel.split("/")
            d = structure
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = os.path.getsize(md_file)
        return {"vault_path": VAULT_PATH, "structure": structure}

    full_path = os.path.join(VAULT_PATH, path)
    if os.path.isdir(full_path):
        files = [f for f in sorted(os.listdir(full_path)) if f.endswith(".md")]
        return {"directory": path, "files": files}
    if not os.path.exists(full_path):
        # Main 55 P1c: file-not-found → fall back to cross-dir search using
        # the path stem as a query. Derive keywords from the path basename.
        stem = os.path.splitext(os.path.basename(path))[0]
        fallback_q = stem.replace("_", " ").replace("-", " ")
        if fallback_q:
            print(f"[vault_read] file not found: {path}; falling back to "
                  f"vault_search(query='{fallback_q}')", flush=True)
            fallback = _vault_read(path="", query=fallback_q)
            if fallback.get("matches"):
                return {"query": fallback_q, "requested_path": path,
                        "matches": fallback["matches"],
                        "_widened_from_missing_file": True}
        return {"error": f"File not found: {path}"}
    try:
        with open(full_path, "r") as f:
            content = f.read()

        # Smart truncation: for large files, extract active/relevant sections
        if len(content) > 3000:
            lines = content.split("\n")
            # Keep: title, "Active NOW", "Near-term", "Production", current sections
            # Skip: "Completed", "Dead Paths", long historical sections
            kept = []
            skip_section = False
            for line in lines:
                # Detect section headers
                if line.startswith("## Completed") or line.startswith("## Dead"):
                    skip_section = True
                    kept.append(line)
                    kept.append("(see full file for details)")
                    continue
                if line.startswith("## ") and skip_section:
                    skip_section = False
                if not skip_section:
                    kept.append(line)

            content = "\n".join(kept)
            if len(content) > 4000:
                content = content[:4000] + "\n\n[... truncated ...]"

        return {"file": path, "content": content}
    except Exception as e:
        return {"error": str(e)}


def _vault_research(query: str) -> dict:
    """Deep research search — searches ANE RE files, agent reports, session logs.

    Unlike vault_read which searches summaries, this searches the detailed
    research files where specific findings, measurements, and opcodes live.
    Larger output budget (6000 chars) for technical depth.

    M54 Phase 2.3: when the query is possessive intent ("our X", "we have"),
    downweight files in research/ subdirectory by 0.1x. The research/ folder
    is overwhelmingly external project notes (Orion, Meta Ray-Ban, etc.),
    not internal capabilities. Without this filter, vault_research returns
    Orion-heavy content for "our LoRA pipeline" queries and the model
    attributes external research as our capability.
    """
    import glob as globmod

    if not query:
        return {"error": "query required"}

    query_words = [w.lower() for w in query.split() if len(w) > 2]
    if not query_words:
        return {"error": "query too short"}

    # M54 Phase 2.4: possessive-intent detection. Distinguish CAPABILITY
    # questions ("do we have X", "is X our pipeline") from KNOWLEDGE
    # questions ("what do we know about X", "have we researched X").
    # Knowledge questions SHOULD surface external research notes — that's
    # exactly what they're asking for. Capability questions should NOT.
    q_low = query.lower()
    capability_markers = [
        "our ", "we have", "we've", "do we have", "do we use",
        "did we build", "did we ship", "did we deploy", "are we using",
        "are we running", "have we built", "have we deployed",
    ]
    knowledge_markers = [
        "do we know", "what do we know", "have we researched",
        "have we explored", "have we investigated", "have we read",
        "what have we found", "have we documented", "have we studied",
    ]
    has_capability = any(m in q_low for m in capability_markers)
    has_knowledge = any(m in q_low for m in knowledge_markers)
    # Knowledge intent dominates: if user asks "what do we know about X",
    # don't downweight external research even if "our" appears.
    possessive = has_capability and not has_knowledge

    # Search deep research directories that vault_read skips
    search_dirs = [
        os.path.join(VAULT_PATH, "ane-reverse"),
        os.path.join(VAULT_PATH, "agent_reports"),
        os.path.join(VAULT_PATH, "slc-probe"),
        os.path.join(VAULT_PATH, "research"),
    ]
    # Also search CLAUDE_reference.md and CLAUDE_session_log.md
    extra_files = [
        os.path.join(VAULT_PATH, "CLAUDE_reference.md"),
        os.path.join(VAULT_PATH, "CLAUDE_session_log.md"),
    ]

    matches = []

    # Search directory trees
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for md_file in globmod.glob(os.path.join(search_dir, "**/*.md"), recursive=True):
            try:
                with open(md_file, "r") as f:
                    content = f.read()
                content_lower = content.lower()
                if not any(w in content_lower for w in query_words):
                    continue
                rel = os.path.relpath(md_file, VAULT_PATH)
                lines = content.split("\n")
                snippets = []
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(w in line_lower for w in query_words):
                        start = max(0, i - 3)
                        end = min(len(lines), i + 4)
                        snippet = "\n".join(lines[start:end]).strip()
                        if len(snippet) > 20:
                            snippets.append(snippet)
                if snippets:
                    # Score by keyword density + filename relevance bonus
                    score = sum(1 for s in snippets for w in query_words if w in s.lower())
                    # Boost files whose names match query words
                    fname_lower = os.path.basename(md_file).lower()
                    score += sum(5 for w in query_words if w in fname_lower)
                    # M54 Phase 2.4: possessive-intent filter, tightened.
                    # Match recall-layer's 0.05x. Detect external-project
                    # content beyond research/ folder via content markers
                    # (arXiv IDs, third-party project names, explicit
                    # "external" / "prior art" labels). Catches files like
                    # agent_reports/ane_frontier_deep_dive.md that are
                    # internal write-ups OF external research — these
                    # were the Q06 leak source.
                    if possessive:
                        rel_norm = rel.replace("\\", "/")
                        is_external = False
                        # Filename / path heuristics. relpath has no
                        # leading slash, so check both startswith and
                        # mid-path for nested cases.
                        if (rel_norm.startswith("research/")
                                or "/research/" in rel_norm):
                            is_external = True
                        elif any(t in fname_lower for t in (
                                "frontier", "prior_art", "prior-art",
                                "deep_dive", "deep-dive", "external_",
                                "third_party", "competitor")):
                            is_external = True
                        else:
                            # Content heuristics: snippet-level external
                            # markers. arXiv IDs, "Murai Labs", "third-party",
                            # "external project", "prior art" all signal that
                            # the discussed system is NOT ours.
                            joined = " ".join(snippets[:3]).lower()
                            ext_signals = sum(1 for sig in (
                                "arxiv:", "arxiv ", "murai labs",
                                "third-party", "third party", "external project",
                                "prior art", "competing", "rival project",
                                "(external)", "(third-party)") if sig in joined)
                            # Two or more signals → external
                            if ext_signals >= 2:
                                is_external = True
                        if is_external:
                            score = score * 0.05
                    matches.append({"file": rel, "snippets": snippets[:8], "score": score})
            except Exception:
                continue

    # Search extra files
    for extra in extra_files:
        if not os.path.exists(extra):
            continue
        try:
            with open(extra, "r") as f:
                content = f.read()
            content_lower = content.lower()
            if not any(w in content_lower for w in query_words):
                continue
            rel = os.path.relpath(extra, VAULT_PATH)
            lines = content.split("\n")
            snippets = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(w in line_lower for w in query_words):
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    snippet = "\n".join(lines[start:end]).strip()
                    if len(snippet) > 20:
                        snippets.append(snippet)
            if snippets:
                score = sum(1 for s in snippets for w in query_words if w in s.lower())
                matches.append({"file": rel, "snippets": snippets[:8], "score": score})
        except Exception:
            continue

    # Sort by relevance score
    matches.sort(key=lambda m: m["score"], reverse=True)

    # Build output with 6000 char budget (2x vault_read)
    result_matches = []
    total_len = 0
    for m in matches[:10]:
        trimmed = []
        for s in m["snippets"]:
            if total_len + len(s) > 6000:
                break
            trimmed.append(s)
            total_len += len(s)
        if trimmed:
            result_matches.append({"file": m["file"], "snippets": trimmed, "score": m["score"]})

    # Extract headlines from top matches for the 70B
    headlines = []
    for m in result_matches[:3]:
        for s in m["snippets"][:2]:
            # First line of each snippet is usually a heading or key finding
            first_line = s.split("\n")[0].strip().lstrip("#").strip()
            if len(first_line) > 10 and len(first_line) < 200:
                headlines.append(f"[{m['file']}] {first_line}")

    return {"query": query, "key_findings": headlines[:5],
            "matches": result_matches, "files_searched": len(matches)}


def _vault_insight(topic: str) -> dict:
    import re
    result = {"topic": topic, "vault_context": [], "memory_context": []}

    key_files = ["HOME.md", "Roadmap.md", "Decision Log.md", "Infrastructure Map.md"]
    projects_dir = os.path.join(VAULT_PATH, "projects", "active")
    if os.path.isdir(projects_dir):
        for f in os.listdir(projects_dir):
            if f.endswith(".md"):
                key_files.append(f"projects/active/{f}")
    domain_dir = os.path.join(VAULT_PATH, "domain")
    if os.path.isdir(domain_dir):
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if f.endswith(".md"):
                    key_files.append(os.path.relpath(os.path.join(root, f), VAULT_PATH))

    vault_hits = []
    for rel_path in key_files:
        full = os.path.join(VAULT_PATH, rel_path)
        if not os.path.exists(full):
            continue
        try:
            with open(full, "r") as f:
                content = f.read()
            if topic.lower() in content.lower() or any(
                w.lower() in content.lower() for w in topic.split() if len(w) > 3
            ):
                lines = content.split("\n")
                relevant = []
                for i, line in enumerate(lines):
                    if any(w.lower() in line.lower() for w in topic.split() if len(w) > 3):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        relevant.append("\n".join(lines[start:end]))
                if relevant:
                    vault_hits.append({"file": rel_path, "excerpts": relevant[:3]})
        except Exception:
            continue

    result["vault_context"] = vault_hits[:5]

    if _memory and _memory._started:
        memories = _memory.recall(topic, n_results=10)
        result["memory_context"] = memories.get("results", [])
        result["total_memories"] = memories.get("total_memories", 0)

    vault_entities = set()
    for hit in vault_hits:
        for excerpt in hit.get("excerpts", []):
            for match in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', excerpt):
                if len(match) > 3:
                    vault_entities.add(match)
    memory_entities = set()
    for mem in result.get("memory_context", []):
        for ent in mem.get("entities", []):
            memory_entities.add(ent)
    overlap = vault_entities & memory_entities
    if overlap:
        result["cross_references"] = list(overlap)

    return result


# ── Scanner ──────────────────────────────────────────────────────────────────

def _scan_digest(mode: str = "latest", top_n: int = 10) -> dict:
    try:
        from scanner import Scanner
        scanner = Scanner()
    except ImportError:
        return {"error": "Scanner module not available"}

    if mode == "latest":
        items = scanner.get_latest_candidates(min(top_n, 5))
        lines = []
        for i, item in enumerate(items, 1):
            title = (item.get("title") or "")[:150]
            source = item.get("source", "")
            lines.append(f"{i}. {title} ({source})")
        return {"summary": f"{len(items)} candidates:\n" + "\n".join(lines)}
    elif mode == "unreviewed":
        return {"mode": "unreviewed", "scans": scanner.get_unreviewed(), "count": len(scanner.get_unreviewed())}
    elif mode == "clear":
        unreviewed = scanner.get_unreviewed()
        all_items, seen_ids = [], set()
        for scan_id in unreviewed:
            scan_path = os.path.join(VAULT_PATH, "midas/scans/candidates", f"{scan_id}.json")
            try:
                with open(scan_path) as f:
                    data = json.load(f)
                for source_data in data.get("sources", {}).values():
                    for item in source_data.get("items", []):
                        item_id = item.get("id", item.get("title", ""))
                        if item_id not in seen_ids:
                            seen_ids.add(item_id)
                            all_items.append(item)
            except Exception:
                continue
        all_items.sort(key=lambda x: x.get("relevance", 0) + x.get("score", 0) / 1000, reverse=True)
        lines = []
        for i, item in enumerate(all_items[:min(top_n, 5)], 1):
            title = (item.get("title") or "")[:150]
            source = item.get("source", "")
            lines.append(f"{i}. {title} ({source})")
        return {"summary": f"Processed {len(unreviewed)} scans, {len(all_items)} unique items. Top {min(top_n, 5)}:\n" + "\n".join(lines)}
    elif mode == "stats":
        return {"mode": "stats", **scanner.get_calibration_stats()}
    else:
        return {"error": f"Unknown mode: {mode}"}


# ── Playbook ─────────────────────────────────────────────────────────────────

import re as _re

_PLAYBOOK_SECTIONS = {
    "scan_schedule": "## Scan Schedule",
    "what_works": "## What Works",
    "what_doesnt": "## What Doesn't Work",
    "high_signal": "## High-Signal Sources",
    "self_eval": "## Self-Eval",
    "improvement_queue": "## Improvement Queue",
    "lessons": "## Lessons Learned",
    "voice": "## Voice & Growth",
}

def _playbook(section: str, action: str, content: str = "") -> dict:
    if action == "read":
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
            if section == "full":
                return {"playbook": text}
            marker = _PLAYBOOK_SECTIONS.get(section)
            if not marker:
                return {"error": f"Unknown section: {section}. Valid: {list(_PLAYBOOK_SECTIONS.keys())}"}
            idx = text.find(marker)
            if idx == -1:
                return {"error": f"Section '{marker}' not found"}
            start = idx + len(marker)
            rest = text[start:]
            end = len(rest)
            for boundary in ["\n## ", "\n---"]:
                pos = rest.find(boundary)
                if pos != -1 and pos < end:
                    end = pos
            return {"section": section, "content": rest[:end].strip()}
        except FileNotFoundError:
            return {"error": "Playbook not found"}

    if action in ("append", "replace"):
        if not content:
            return {"error": "content required"}
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
        except FileNotFoundError:
            return {"error": "Playbook not found"}
        marker = _PLAYBOOK_SECTIONS.get(section)
        if not marker:
            return {"error": f"Unknown section: {section}"}
        idx = text.find(marker)
        if idx == -1:
            return {"error": f"Section '{marker}' not found"}
        start = idx + len(marker)
        rest = text[start:]
        end = len(rest)
        for boundary in ["\n## ", "\n---"]:
            pos = rest.find(boundary)
            if pos != -1 and pos < end:
                end = pos
        old_section = rest[:end]
        new_section = (old_section.rstrip() + "\n" + content + "\n") if action == "append" else ("\n" + content + "\n")
        text = text[:start] + new_section + rest[end:]
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        if "*Last updated:" in text:
            text = _re.sub(r'\*Last updated:.*\*', f'*Last updated: {today} (auto)*', text)
        with open(PLAYBOOK_PATH, "w") as f:
            f.write(text)
        return {"status": "updated", "section": section, "action": action}

    return {"error": f"Unknown action: {action}"}


# ── Claude Inbox ─────────────────────────────────────────────────────────────

def _message_claude(message: str, priority: str = "medium", context: str = "") -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {timestamp}\n**Priority:** {priority}\n**From:** Midas\n\n{message}\n"
    if context:
        entry += f"\n**Context:**\n```\n{context[:2000]}\n```\n"
    entry += "\n---\n"
    os.makedirs(os.path.dirname(CLAUDE_INBOX), exist_ok=True)
    if not os.path.exists(CLAUDE_INBOX):
        with open(CLAUDE_INBOX, "w") as f:
            f.write("# Claude Inbox\n\nMessages from Midas for Claude to review.\n\n---\n")
    with open(CLAUDE_INBOX, "a") as f:
        f.write(entry)
    return {"status": "sent", "timestamp": timestamp, "priority": priority}


# ── Self-Observation Tools ────────────────────────────────────────────────

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.mlx-env/bin/python3")

def _self_test(mode: str) -> dict:
    """Run stress test, parse results, return structured summary.
    Feeds failures back into correction log automatically via --json."""
    if mode in ("hardcore", "full", "deep", "stress", "all"):
        script = "live_stress_test.py"
        timeout = 600
    else:
        script = "test_router.py"
        timeout = 30

    # For live_stress_test, use --json to get machine-readable output
    if script == "live_stress_test.py":
        cmd = f"{PYTHON} {script} --json"
        try:
            out = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=AGENT_DIR,
            )
            # --json mode outputs JSON to stdout
            parsed = json.loads(out.stdout)
            summary_parts = [f"{parsed['pass']}/{parsed['total']} pass"]
            if parsed["warn"] > 0:
                summary_parts.append(f"{parsed['warn']} warn")
            if parsed["fail"] > 0:
                summary_parts.append(f"{parsed['fail']} fail")
            summary_parts.append(f"in {parsed['duration']}s")

            # Comparison to last run
            comp = parsed.get("comparison")
            if comp:
                if comp["regressed"]:
                    summary_parts.append(f"REGRESSION: was {comp['prev_pass']}/{comp['prev_total']}")
                elif comp["delta_pass"] > 0:
                    summary_parts.append(f"improved +{comp['delta_pass']} from last run")
                else:
                    summary_parts.append("no regressions")

            weaknesses = [t for t in parsed.get("tests", []) if t["status"] != "pass"]
            result = {
                "summary": ". ".join(summary_parts),
                "total": parsed["total"],
                "pass": parsed["pass"],
                "warn": parsed["warn"],
                "fail": parsed["fail"],
                "duration": parsed["duration"],
            }
            if weaknesses:
                result["weaknesses"] = [
                    f"{w['id']}: {w['detail'][:80]}" for w in weaknesses[:5]
                ]
            return result
        except subprocess.TimeoutExpired:
            return {"error": f"Stress test timed out after {timeout}s"}
        except (json.JSONDecodeError, KeyError) as e:
            return {"error": f"Failed to parse test output: {e}"}
    else:
        # Light test — just run test_router.py and capture output
        try:
            out = subprocess.run(
                f"{PYTHON} {script}", shell=True, capture_output=True,
                text=True, timeout=timeout, cwd=AGENT_DIR,
            )
            # Extract result line like "53/53 (100%) — ALL PASS"
            lines = out.stdout.strip().split("\n")
            result_line = ""
            for line in reversed(lines):
                if "/" in line and "%" in line:
                    # Strip ANSI codes
                    clean = _re.sub(r'\033\[[0-9;]*m', '', line).strip()
                    result_line = clean
                    break
            return {"summary": result_line or "Test complete", "output": out.stdout[-1000:]}
        except subprocess.TimeoutExpired:
            return {"error": f"Test timed out after {timeout}s"}


def _launch_heartbeat() -> dict:
    """Launch the Heartbeat monitoring dashboard if not already running."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8423/api/all", timeout=2)
        return {"status": "already running", "url": "http://localhost:8423"}
    except Exception:
        pass
    heartbeat_path = os.path.join(os.path.dirname(__file__), "heartbeat.py")
    if not os.path.exists(heartbeat_path):
        return {"error": "heartbeat.py not found"}
    subprocess.Popen(
        [sys.executable, heartbeat_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait briefly for it to start
    import time
    for _ in range(10):
        time.sleep(0.5)
        try:
            urllib.request.urlopen("http://localhost:8423/api/all", timeout=1)
            # Open in browser on macOS
            subprocess.Popen(["open", "http://localhost:8423"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"status": "launched", "url": "http://localhost:8423"}
        except Exception:
            continue
    return {"error": "Heartbeat started but not responding yet. Check http://localhost:8423"}


def _brain_snapshot(scope: str) -> dict:
    """Return current agent state for synthesis."""
    from feedback_loop import get_last_decision, get_session_stats

    if scope == "last":
        last = get_last_decision()
        if not last:
            return {"summary": "No routing decisions logged yet this session."}
        layer = "L1" if last.get("l1") else ("L2" if last.get("l2") else "conversation")
        tool = last.get("final", "?")
        msg = last.get("msg", "?")[:80]
        return {
            "summary": f"Last decision: '{msg}' -> {tool} via {layer}",
            "input": msg,
            "layer": layer,
            "tool": tool,
            "l1_match": last.get("l1"),
            "l2_category": last.get("l2"),
        }
    else:
        stats = get_session_stats()
        total = stats.get("total_decisions", 0)
        parts = [f"{total} decisions"]
        parts.append(f"{stats.get('l1_count', 0)} L1, {stats.get('l2_count', 0)} L2, {stats.get('conv_count', 0)} conversation")
        if stats.get("total_corrections", 0) > 0:
            parts.append(f"{stats['total_corrections']} corrections")
        else:
            parts.append("zero corrections")
        if stats.get("accuracy_pct") is not None:
            parts.append(f"accuracy {stats['accuracy_pct']}%")
        if stats.get("avg_route_ms") is not None:
            parts.append(f"avg route {stats['avg_route_ms']}ms")

        last_stress = stats.get("last_stress_result")
        if last_stress:
            parts.append(f"last stress test: {last_stress['pass']}/{last_stress['total']} ({last_stress.get('ts', '?')[:16]})")

        return {
            "summary": ". ".join(parts),
            **stats,
        }


def _self_improve(mode: str) -> dict:
    """Run router_improver.py, return analysis and proposals."""
    try:
        out = subprocess.run(
            f"{PYTHON} router_improver.py --auto", shell=True,
            capture_output=True, text=True, timeout=60, cwd=AGENT_DIR,
        )
        report_out = subprocess.run(
            f"{PYTHON} router_improver.py --report", shell=True,
            capture_output=True, text=True, timeout=30, cwd=AGENT_DIR,
        )
        # Strip ANSI codes from report
        report = _re.sub(r'\033\[[0-9;]*m', '', report_out.stdout).strip()
        return {"summary": report, "auto_output": out.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "Improver timed out"}


# Note: domain-specific entity-lookup pipeline tools were removed in Main 26
# housekeeping. The Subconscious memory recall path handles entity questions
# directly via multi_path_recall.


# ── Main dispatch ────────────────────────────────────────────────────────────

def execute(tool_name: str, args: dict) -> str:
    """Execute a tool and return plain-text result.

    Argument validation happens here — reject obviously bad calls
    before touching any backend.
    """
    # Validation
    if tool_name == "memory_recall" and not args.get("query", "").strip():
        return "Error: Empty recall query. Provide a specific search term."
    if tool_name == "browse_search" and not args.get("query", "").strip():
        return "Error: Empty search query. Provide a specific query."
    if tool_name == "browse_navigate" and not args.get("url", "").startswith("http"):
        return f"Error: Invalid URL: {args.get('url', '')}. Must start with http(s)."
    try:
        result = _dispatch(tool_name, args)
    except Exception as e:
        result = {"error": str(e)}

    # Flatten to plain text — 9B regurgitates JSON, so prefer text
    if isinstance(result, dict):
        if "summary" in result and len(result) <= 2:
            return result["summary"]
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2)
    return str(result)


def _dispatch(name: str, args: dict) -> dict:
    """Route tool name to handler. Returns dict."""
    # Main 46: normalize truncated tool names from L2 LLM routing.
    # The 27B occasionally returns "vault" instead of "vault_read",
    # "memory" instead of "memory_recall", etc.
    _TOOL_ALIASES = {
        "vault": "vault_read",
        "memory": "memory_recall",
        "browse": "browse_search",
        "research": "vault_research",
    }
    name = _TOOL_ALIASES.get(name, name)

    # Memory
    if name == "memory_ingest":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.ingest(args.get("role", "user"), args.get("text", ""))
    if name == "memory_recall":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.recall(args.get("query", ""), args.get("n_results", 5), args.get("type_filter", ""))
    if name == "memory_stats":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.stats()
    if name == "memory_insights":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.get_insights()

    # Vault
    if name == "vault_read":
        return _vault_read(args.get("path", ""), args.get("query", ""))
    if name == "vault_research":
        return _vault_research(args.get("query", ""))
    if name == "vault_insight":
        return _vault_insight(args.get("topic", ""))

    # Browser — auto-launch Chrome headless if not already connected
    if name.startswith("browse_") and not _browser:
        try:
            from browser import BrowserBridge
            _b = BrowserBridge()
            if _b.is_available():  # triggers auto-launch if Chrome not running
                _b.connect()
                set_browser(_b)
                print("[tool_executor] browser auto-connected on demand", flush=True)
        except Exception as _be:
            print(f"[tool_executor] browser auto-connect failed: {_be}", flush=True)
    if name.startswith("browse_") and not _browser:
        return {"error": "Browser not available. Chrome could not be launched."}
    if name == "browse_navigate":
        return _browser.navigate(args.get("url", ""), args.get("wait", 2))
    if name == "browse_read":
        return _browser.read_page(args.get("selector", "body"), args.get("max_length", 5000))
    if name == "browse_click":
        return _browser.click(args.get("selector", ""))
    if name == "browse_type":
        return _browser.type_text(args.get("selector", ""), args.get("text", ""))
    if name == "browse_js":
        return _browser.run_js(args.get("expression", ""))
    if name == "browse_search":
        return _browser.search(args.get("query", ""), args.get("max_results", 5))
    if name == "browse_x_feed":
        # Main 62 pilot-fix 4: multi-handle ("difference between @X and @Y")
        # support. If router emits a `handles` list of 2+, fetch each and
        # merge. Back-compat: if `handles` absent, fall through to the
        # original single-handle call.
        handles = args.get("handles")
        if isinstance(handles, list) and len(handles) >= 2:
            merged_posts = []
            fetched = []
            errors = []
            for h in handles:
                try:
                    r = _browser.scan_x_feed(args.get("count", 5), h)
                    fetched.append(h)
                    if isinstance(r, dict) and r.get("posts"):
                        for p in r["posts"]:
                            # Tag each post with its source handle so
                            # the synthesizer can differentiate.
                            if isinstance(p, dict) and "handle" not in p:
                                p = {**p, "handle": h}
                            merged_posts.append(p)
                    elif isinstance(r, dict):
                        # Preserve any non-posts fields (errors, meta) per-handle
                        errors.append({"handle": h, "result": r})
                except Exception as e:
                    errors.append({"handle": h, "error": str(e)})
            return {
                "posts": merged_posts,
                "handles_fetched": fetched,
                "multi_handle": True,
                "per_handle_errors": errors,
            }
        return _browser.scan_x_feed(args.get("count", 5), args.get("handle"))
    if name == "browse_tabs":
        return {"tabs": _browser.get_tabs()}

    # Scanner
    if name == "scan_digest":
        return _scan_digest(args.get("mode", "latest"), args.get("top_n", 10))

    # Playbook
    if name == "playbook_update":
        return _playbook(args.get("section", "full"), args.get("action", "read"), args.get("content", ""))

    # Claude inbox
    if name == "message_claude":
        return _message_claude(args.get("message", ""), args.get("priority", "medium"), args.get("context", ""))

    # Heartbeat dashboard
    if name == "heartbeat":
        return _launch_heartbeat()

    # Self-test
    if name == "self_test":
        return _self_test(args.get("mode", "light"))
    if name == "brain_snapshot":
        return _brain_snapshot(args.get("scope", "session"))
    if name == "self_improve":
        return _self_improve(args.get("mode", "analyze"))

    # Shell
    if name == "shell":
        cmd = args.get("command", "")
        try:
            out = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
                cwd="/Users/midas/Desktop/cowork"
            )
            return {"stdout": out.stdout[-2000:] if out.stdout else "", "stderr": out.stderr[-500:] if out.stderr else "", "returncode": out.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "command timed out (30s)"}

    # Research probe — run commands, save findings to vault
    if name == "research_probe":
        return _research_probe(args.get("task", ""), args.get("commands", []),
                               args.get("tag", "general"))

    return {"error": f"unknown tool: {name}"}


def _research_probe(task: str, commands: list, tag: str = "general") -> dict:
    """Execute research probe commands and save findings to vault.

    Args:
        task: description of what we're investigating
        commands: list of shell commands to run
        tag: category tag (ane, silicon, memory, performance)
    """
    from datetime import datetime

    results = []
    for cmd in commands[:10]:  # cap at 10 commands
        try:
            out = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=60,
                cwd="/Users/midas/Desktop/cowork"
            )
            results.append({
                "command": cmd,
                "stdout": out.stdout[-3000:] if out.stdout else "",
                "stderr": out.stderr[-500:] if out.stderr else "",
                "returncode": out.returncode,
            })
        except subprocess.TimeoutExpired:
            results.append({"command": cmd, "error": "timeout (60s)"})
        except Exception as e:
            results.append({"command": cmd, "error": str(e)})

    # Build structured report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    report_lines = [
        f"# Research Probe: {task}",
        f"Tag: {tag} | Time: {timestamp}",
        f"Commands: {len(commands)} | Results: {len(results)}",
        "",
    ]
    for r in results:
        report_lines.append(f"## `{r['command']}`")
        if r.get("error"):
            report_lines.append(f"ERROR: {r['error']}")
        else:
            report_lines.append(f"Return code: {r['returncode']}")
            if r["stdout"]:
                report_lines.append(f"```\n{r['stdout'][:2000]}\n```")
            if r["stderr"]:
                report_lines.append(f"stderr: {r['stderr'][:500]}")
        report_lines.append("")

    report = "\n".join(report_lines)

    # Save to vault
    probe_dir = os.path.join(VAULT_PATH, "research", "findings", "observed")
    os.makedirs(probe_dir, exist_ok=True)
    filename = f"probe-{tag}-{timestamp}.md"
    filepath = os.path.join(probe_dir, filename)
    with open(filepath, "w") as f:
        f.write(report)

    return {
        "task": task,
        "commands_run": len(results),
        "saved_to": f"research/findings/observed/{filename}",
        "summary": report[:2000],
    }
