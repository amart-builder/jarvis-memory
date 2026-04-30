#!/usr/bin/env python3
"""LongMemEval adapter for jarvis-memory v1.1.

Per pre-registered protocol (docs/eval/longmemeval-v1.1-protocol.md):
- Test set: longmemeval_s_cleaned.json (NOT oracle).
- Single-shot per question, temperature=0.
- Question classifier (regex/heuristic) — not reading question_type.
- OMEGA's 5 prompt templates verbatim.
- OMEGA's triple fan-out retrieval recipe.
- AR1: PPR damping α=0.5 (HippoRAG-2 paper value).
- AR2: PPR seed broadening (noun phrases, not only proper nouns).
- AR3: counting enumeration (already in OMEGA's MULTISESSION prompt).
- Per-question isolation via group_id=lme_q_<id> + label :LMETestEpisode.

Usage:
    JARVIS_LME_ANSWERER=opus python scripts/run_longmemeval.py \\
        --output runs/lme_opus_v1.1.jsonl

    # Validate on 10 stratified questions:
    JARVIS_LME_ANSWERER=opus python scripts/run_longmemeval.py \\
        --output runs/lme_opus_validate.jsonl --validate

Resume:
    Re-running with the same --output skips already-answered question_ids.

Cost note: full 500 × 1 answerer ≈ $30-50 in API; ~6-10hr wall time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import resource
from pathlib import Path


def _load_env_file() -> None:
    """Load repo-local .env values for direct serial runner invocations."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _bump_fd_limit_if_low(target: int = 8192) -> None:
    """Raise the open-file ulimit to ``target`` if it's lower.

    macOS defaults to 256 file descriptors per process. The adapter
    holds simultaneous handles on Chroma's HNSW segments, Neo4j's bolt
    pool, sentence-transformer model files, the cross-encoder, plus
    transient TLS sockets for OpenAI's batched extraction calls. 256
    blows up mid-run with "Too many open files" — observed on Mini at
    ingest_observations time on 2026-04-28. Bumping to 8192 has no
    runtime cost (it's just a kernel-side cap) and self-heals any host
    that didn't pre-set it via shell ulimit or launchd.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= target:
            return
        new_soft = min(target, hard) if hard != resource.RLIM_INFINITY else target
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):
        # If the kernel refuses (e.g., target > hard cap on a locked-down
        # box), don't crash — the run might still succeed; if it doesn't,
        # the operator gets the same OSError as before.
        pass


_bump_fd_limit_if_low()
_load_env_file()

import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("scripts.run_longmemeval")

# Make our scripts package importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.longmemeval.classifier import (  # noqa: E402
    ABSTENTION_FILTER,
    CONTEXT_BUDGET_CHARS,
    CONTEXT_BUDGET_MIN_HITS,
    COUNTING_K_FLOOR,
    FILTER_CONFIG,
    K_FLOORS,
    channel_weights,
    classify,
    classify_lme_intent,
    is_counting_question,
)
from scripts.longmemeval.answer_scaffold import build_answer_scaffold  # noqa: E402
from scripts.longmemeval.evidence_ledger import build_evidence_ledger  # noqa: E402
from scripts.longmemeval.prompts import (  # noqa: E402
    answer_to_str,
    format_session_for_prompt,
    format_session_text,
    parse_longmemeval_date,
    render_ms_count_prompt,
    render_ms_extract_prompt,
    render_prompt,
)


# ── Constants ─────────────────────────────────────────────────────────


# Worker isolation env vars (Codex's parallel harness sets these per-worker;
# default to single-tenant values if unset). Each worker thus gets its own
# Neo4j label, Chroma collection, Chroma path, and group_id prefix — eliminates
# cross-worker contention on shared HNSW indexes and Cypher writes.
LME_NEO4J_LABEL: str = os.getenv("JARVIS_LME_NEO4J_LABEL", "LMETestEpisode")
LME_OBSERVATION_LABEL: str = LME_NEO4J_LABEL.replace("LMETestEpisode", "LMETestObservation", 1)
LME_CHROMA_COLLECTION: str = os.getenv("JARVIS_LME_CHROMA_COLLECTION", "jarvis_lme_v1")
LME_GROUP_PREFIX: str = os.getenv("JARVIS_LME_GROUP_PREFIX", "lme_q")
LME_AGENT_ID: str = "benchmark-longmemeval"

# Stage 5 v2 Phase 8 (2026-04-28): typed observation extraction. When
# JARVIS_LME_OBSERVATIONS=1, each haystack session also gets a per-session
# gpt-4o-mini extraction pass that produces typed/concise observations
# stored as :LMETestObservation_<worker_suffix> nodes. Retrieval surfaces
# these BEFORE chronological notes in the prompt. Disabled by default —
# Phase 6 baseline behavior is preserved unless this is explicitly on.
LME_OBSERVATIONS_ENABLED: bool = os.getenv("JARVIS_LME_OBSERVATIONS", "0") == "1"
LME_OBSERVATION_TOP_K: int = int(os.getenv("JARVIS_LME_OBSERVATION_TOP_K", "10"))
# When set, extractions are cached in this SQLite DB — re-runs hit the
# cache instead of re-calling gpt-4o-mini. Empty/unset → no caching.
LME_EXTRACTION_CACHE_PATH: str = os.getenv("JARVIS_LME_EXTRACTION_CACHE_PATH", "").strip()
# Phase 11 (2026-04-29): deterministic evidence ledgers. Experimental and
# opt-in while the target-40 plan is still under review. Enable with
# JARVIS_LME_EVIDENCE_LEDGER=1; default-off preserves exact Phase 10 parity.
LME_EVIDENCE_LEDGER_ENABLED: bool = os.getenv("JARVIS_LME_EVIDENCE_LEDGER", "0") == "1"
LME_ANSWER_SCAFFOLD_ENABLED: bool = (
    os.getenv("JARVIS_LME_ANSWER_SCAFFOLD", os.getenv("JARVIS_LME_EVIDENCE_LEDGER", "0")) == "1"
)

DEFAULT_DATASET: Path = Path("data/longmemeval/longmemeval_s_cleaned.json")
DEFAULT_ORACLE: Path = Path("data/longmemeval/longmemeval_oracle.json")
DEFAULT_OUTPUT: Path = Path("runs/lme_run.jsonl")

# Stage 0: deterministic seeding constant. Used for OpenAI ``seed=`` arg
# (gpt-4o + gpt-4.1 honor it for reproducible decoding at temperature=0)
# AND as Python's random seed at module load. PYTHONHASHSEED is enforced
# by re-exec'ing in main() if the env var isn't already 42.
RUN_SEED: int = 42

# Apply Python random seed at import time so any downstream module that
# samples on import (e.g. embedding init) sees a deterministic state.
random.seed(RUN_SEED)

# Stoplist for AR2 (PPR seed broadening). Common English words that
# should NOT seed a graph walk — they appear too often.
_AR2_STOPLIST: set[str] = {
    "what", "when", "where", "which", "while", "with", "would", "have",
    "this", "that", "than", "then", "they", "them", "there", "these",
    "those", "from", "their", "your", "yours", "mine", "much", "many",
    "more", "most", "less", "some", "such", "since", "still", "thing",
    "things", "time", "times", "year", "years", "month", "months", "week",
    "weeks", "day", "days", "long", "ago", "after", "before", "between",
    "about", "ever", "very", "often", "into", "over", "under", "also",
    "been", "were", "been", "does", "did", "doing", "done", "had", "has",
    "having", "tell", "told", "said", "say", "saying", "asked", "answer",
    "good", "best", "first", "last", "next", "now", "currently", "current",
    "recent", "recently", "still", "yet", "anymore", "longer", "started",
    "begin", "began", "begun", "made", "make", "making", "took", "take",
    "taking", "taken", "give", "giving", "gave", "given", "going", "gone",
    "want", "wanting", "wanted", "needs", "need", "needing", "needed",
    "tried", "trying", "tries", "remember", "remembering", "remembered",
    "recall", "recalls", "recalled", "remind", "reminding", "reminded",
    "spend", "spent", "spending", "find", "finding", "found",
    "find", "lost", "lose", "losing", "shows", "show", "showing", "showed",
    "play", "plays", "played", "playing",
}


# ── AR1 + AR2: PPR overrides (monkey-patch) ───────────────────────────


def apply_ppr_overrides() -> None:
    """Apply pre-registered protocol additions AR1 + AR2 to PPR.

    AR1: damping α=0.85 → 0.5 (HippoRAG-2 paper value, spreads
    activation further across the graph for multi-hop).

    AR2: PPR seed broadening — extract noun phrases (lowercase common
    nouns ≥4 chars not in stoplist) in addition to the existing proper-
    noun extraction. So "how often do I exercise" seeds PPR on
    "exercise" instead of returning [].

    Applied via monkey-patch — no production code touched. The
    overrides revert when this process exits.
    """
    from jarvis_memory.search import ppr as ppr_mod

    _orig_extract = ppr_mod._extract_query_entities
    _orig_ppr = ppr_mod.personalized_pagerank

    def broadened_extract(query: str) -> list[str]:
        # Original proper-noun seeds first (preserves prior behavior).
        seeds = list(_orig_extract(query))
        seen = set(seeds)
        # Broaden with lowercase common nouns ≥4 chars.
        for word in re.findall(r"\b[a-z]{4,}\b", query.lower()):
            if word in seen or word in _AR2_STOPLIST:
                continue
            seeds.append(word)
            seen.add(word)
        return seeds

    def ppr_with_alpha(query, **kwargs):
        kwargs.setdefault("damping", 0.5)
        return _orig_ppr(query, **kwargs)

    ppr_mod._extract_query_entities = broadened_extract
    ppr_mod.personalized_pagerank = ppr_with_alpha


# ── Stage 2: list-extraction post-processing for MS counting ──────────


_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s+",  # "- foo", "* foo", "• foo", "1. foo", "1) foo"
    re.MULTILINE,
)
_TOTAL_LINE_RE = re.compile(
    r"(?im)^\s*(?:total|count|answer)\s*[:\-=]\s*(\d+)\b",
)


def maybe_append_total_line(hypothesis: str, category: str, counting: bool) -> str:
    """Defensive: ensure MS counting answers end with 'Total: N'.

    Stage 2 prompt rule says the FINAL line MUST be "Total: N". gpt-4.1
    follows the rule most of the time but occasionally enumerates a list
    and forgets the total line. When that happens, count list items and
    append the total ourselves so the judge sees a clean number.

    Only applies to ``multi-session`` category AND counting questions —
    the rule is in the MS prompt and the judge looks for a number on
    these questions specifically. Pure function — never mutates input.

    Args:
        hypothesis: Raw LLM output.
        category: Predicted category. Skip unless multi-session.
        counting: ``is_counting_question`` result. Skip unless True.

    Returns:
        Possibly-augmented hypothesis. If the answer already has a
        "Total: N" line, returns input unchanged.
    """
    if category != "multi-session" or not counting:
        return hypothesis
    if not hypothesis or not hypothesis.strip():
        return hypothesis

    # Already has a total/count/answer-N line? Leave alone.
    if _TOTAL_LINE_RE.search(hypothesis):
        return hypothesis

    # Count enumerated list items (bullet- or numbered-list lines).
    items = _LIST_ITEM_RE.findall(hypothesis)
    if len(items) < 2:
        # Need at least 2 list items to be confident this is a count answer
        # — single bullet might be incidental to a non-counting answer.
        return hypothesis

    return f"{hypothesis.rstrip()}\n\nTotal: {len(items)}"


# ── Stage 1: confidence-based abstention guard ────────────────────────


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z][a-zA-Z]+)\b")

# Common sentence-initial words that look like proper nouns when
# capitalized but aren't entity references. Don't trigger abstention
# guard on these.
_ABSTENTION_GUARD_STOPWORDS: set[str] = {
    "How", "What", "When", "Where", "Why", "Who", "Which",
    "Did", "Do", "Does", "Is", "Are", "Was", "Were",
    "Have", "Has", "Had", "Will", "Would", "Could", "Should", "Can",
    "Tell", "Remind", "Give", "List", "Find", "Show",
    "The", "This", "That", "These", "Those",
    "And", "But", "Or", "If", "Then", "Also", "Just",
    "I", "You", "We", "They", "It", "He", "She",
    "Note", "Notes",  # collide with our [Note N] convention
}


def _extract_question_proper_nouns(question: str) -> list[str]:
    """Pick out likely entity references from a question string.

    Heuristic: capitalized words ≥3 chars, deduped (case-insensitive),
    minus a stoplist of sentence-initial Wh-words and modals that get
    accidentally capitalized.
    """
    seen: set[str] = set()
    out: list[str] = []
    for w in _PROPER_NOUN_RE.findall(question):
        if w in _ABSTENTION_GUARD_STOPWORDS:
            continue
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


# Calibrated against the baseline run distribution. 0.30 was at the
# median for SS-user (0.32) and KU (0.34) — would trigger on ~27% of
# the run, mostly false positives. 0.20 matches OMEGA's own
# ABSTENTION_FILTER.min_rel and only fires on the bottom ~10% of
# retrieval-confidence cases — where the guard's value is real.
_ABSTENTION_THRESHOLD: float = 0.20
_ABSTENTION_RULE_TEMPLATE: str = (
    "ABSTENTION RULE — read carefully:\n"
    "Retrieval found NO session that specifically mentions {entities}. "
    "If the notes below do not contain specific information about "
    "{entities}, your answer MUST say you don't have enough information. "
    "Do not guess or hallucinate.\n\n"
)


def maybe_build_abstention_prefix(
    *,
    question: str,
    hits: list[dict[str, Any]],
    top_score: float,
    threshold: float = _ABSTENTION_THRESHOLD,
) -> Optional[str]:
    """Decide whether to prepend an abstention rule, return the text or None.

    Triggers when BOTH conditions hold:
      1. ``top_score < threshold`` — retrieval confidence is weak.
      2. The question contains a proper noun that does NOT appear
         (case-insensitively) in any retrieved hit's content.

    The prepended block tells the LLM to abstain rather than hallucinate.
    Targets the abstention false-negative failure mode: 4 ``_abs``
    questions in the baseline run hallucinated answers when the truth
    was "not enough information".

    Returns ``None`` when the guard does NOT fire — caller skips the
    prepend, prompt is unchanged. Pure function; never mutates inputs.
    """
    if top_score >= threshold:
        return None
    nouns = _extract_question_proper_nouns(question)
    if not nouns:
        return None

    haystack = " ".join((h.get("content") or "") for h in hits).lower()
    missing = [n for n in nouns if n.lower() not in haystack]
    if not missing:
        return None

    # If multiple entities are missing, name up to two — keeps the
    # injected text short and concrete.
    label = " or ".join(repr(m) for m in missing[:2])
    return _ABSTENTION_RULE_TEMPLATE.format(entities=label)


# ── Stage 1: per-category prompt-context budget ───────────────────────


def _hit_score(h: dict[str, Any]) -> float:
    """Pick the best score field on a hit (composite_score > score > similarity)."""
    for k in ("composite_score", "score", "similarity"):
        v = h.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _hit_date_key(h: dict[str, Any]) -> str:
    """Stable date key for prompt-order sorting (oldest first)."""
    return str(h.get("referenced_date") or h.get("created_at") or "")


def trim_to_context_budget(
    hits: list[dict[str, Any]],
    category: str,
    budget_chars: Optional[int] = None,
    min_hits: int = CONTEXT_BUDGET_MIN_HITS,
) -> list[dict[str, Any]]:
    """Drop lowest-scored hits until cumulative content fits the budget.

    Categories absent from ``CONTEXT_BUDGET_CHARS`` skip the trim and
    return the input unchanged (in date-sorted order). Categories
    present (currently SS only) get aggressive trimming because the
    answer to a single-session question is concentrated in 1 session
    and surplus context just distracts the LLM.

    Hits are first re-sorted by score (highest first) so the trim
    drops bottom-of-the-rerank items, then re-sorted by date for the
    prompt's "[Note N]" ordering convention.

    A hard floor of ``min_hits`` is enforced — even if the first hit
    alone exceeds budget, we keep the top ``min_hits`` so the LLM
    isn't handed an empty context.

    Args:
        hits: Hit dicts as returned by ``retrieve_with_omega_recipe``.
            Order doesn't matter (we re-sort).
        category: Predicted category — keys ``CONTEXT_BUDGET_CHARS``.
        budget_chars: Explicit override. ``None`` looks up the per-
            category default; if category isn't in the budget dict,
            no trimming happens. Test/debug knob.
        min_hits: Floor on result size, never trim below this.

    Returns:
        New list of hits, date-sorted ascending (oldest first), with
        cumulative ``content`` chars ≤ budget when budget applied.
        Pure function — never mutates input list or its dicts.
    """
    if not hits:
        return []
    if budget_chars is None:
        cfg_budget = CONTEXT_BUDGET_CHARS.get(category)
        if cfg_budget is None:
            # Category opted out of trimming — return date-sorted copy.
            return sorted(hits, key=_hit_date_key)
        budget_chars = cfg_budget

    by_score = sorted(hits, key=_hit_score, reverse=True)
    kept: list[dict[str, Any]] = []
    total = 0
    for h in by_score:
        c = len(h.get("content") or "")
        if kept and len(kept) >= min_hits and total + c > budget_chars:
            break
        kept.append(h)
        total += c

    kept.sort(key=_hit_date_key)
    return kept


def apply_adaptive_filter(
    hits: list[dict[str, Any]],
    *,
    category: str,
    counting: bool,
) -> list[dict[str, Any]]:
    """Apply OMEGA-style per-category filtering.

    Phase 9: multi-session counting is the exception. The Phase 8 error
    atlas showed repeated cases where a gold session was retrieved
    upstream, then dropped solely because its score fell below
    ``min_rel``. Counting questions prefer recall: the two-pass MS
    prompt can enumerate and dedupe, but it cannot recover a missing
    event.
    """
    cfg = FILTER_CONFIG.get(category, FILTER_CONFIG["single-session-user"])
    min_rel = float(cfg["min_rel"])
    min_res = int(cfg["min_res"])
    max_res = int(cfg["max_res"])

    if category == "multi-session" and counting:
        return hits[:max_res]

    above = [h for h in hits if _hit_score(h) >= min_rel]
    if len(above) >= min_res:
        return above[:max_res]
    return hits[:max(min_res, len(hits))][:max_res]


# ── Stage 0: gold-session retrieval diagnostics ───────────────────────


def _extract_session_id(uuid: str, group_id: str) -> str:
    """Recover the bare LongMemEval session_id from our UUID format.

    Ingestion writes UUIDs as ``f"{group_id}__{i:03d}_{session_id}"``
    (the ``i:03d`` was added in commit f9aa28c to handle 13/500 questions
    in s_cleaned where the same session_id appears twice in
    haystack_session_ids — Chroma rejects duplicate IDs in one batch).

    LongMemEval session_ids themselves can contain underscores
    (e.g. ``answer_4be1b6b4_2``) so we can't naively split on ``_``.
    Strip the known prefix shape, then strip exactly the 3-digit index
    that follows.
    """
    prefix = f"{group_id}__"
    if not uuid.startswith(prefix):
        return uuid
    rest = uuid[len(prefix):]
    if len(rest) >= 4 and rest[:3].isdigit() and rest[3] == "_":
        return rest[4:]
    return rest


def compute_retrieval_diagnostics(
    hits: list[dict[str, Any]],
    answer_session_ids: list[str],
    group_id: str,
) -> dict[str, Any]:
    """Compute gold-session retrieval diagnostics for one question.

    Stage 0 instrumentation. Tracks where each oracle ``answer_session_id``
    ranks in the final hit list that gets fed to the LLM. Diagnostics are
    purely observational — they are NOT used to alter generation.

    The returned dict is logged verbatim into the JSONL row so per-stage
    runs can be diff'd to see which interventions improved retrieval vs
    which improved generation.

    Args:
        hits: Final hit list (post-filter, post-sort, post-recency-boost)
            sent into the prompt. Order matters — rank 1 = top.
        answer_session_ids: Oracle's ground-truth session IDs for this Q.
        group_id: Per-question namespace prefix used in UUIDs.

    Returns:
        Dict with:
          - ``gold_session_ids``: sorted oracle IDs (for the row)
          - ``gold_count``: how many gold sessions exist
          - ``gold_ranks``: {session_id: 1-based rank, or -1 if missing}
          - ``gold_in_top{5,10,20,50}``: count of gold IDs at-or-above rank K
          - ``gold_in_pool``: count of gold IDs anywhere in the hit list
          - ``all_gold_in_top5``: every gold ID is ranked ≤5
          - ``any_gold_in_top5``: at least one gold ID is ranked ≤5
          - ``candidate_pool_size``: ``len(hits)``
    """
    gold_set = set(answer_session_ids)
    gold_count = len(gold_set)
    ranks: dict[str, int] = {sid: -1 for sid in gold_set}

    for rank, h in enumerate(hits, start=1):
        uid = str(h.get("uuid") or h.get("id") or "")
        sid = _extract_session_id(uid, group_id)
        if sid in gold_set and ranks[sid] == -1:
            ranks[sid] = rank

    found_ranks = [r for r in ranks.values() if r > 0]

    def _at_or_below(k: int) -> int:
        return sum(1 for r in found_ranks if r <= k)

    # Stage 5 Phase 2: capture the ordered list of final UUIDs that
    # entered the prompt. Lets us diff retrieval order between runs (e.g.
    # for parallelism parity check) and label each hit by gold-or-not.
    final_uuids = [str(h.get("uuid") or h.get("id") or "") for h in hits]

    return {
        "gold_session_ids": sorted(gold_set),
        "gold_count": gold_count,
        "gold_ranks": ranks,
        "gold_in_top5": _at_or_below(5),
        "gold_in_top10": _at_or_below(10),
        "gold_in_top20": _at_or_below(20),
        "gold_in_top50": _at_or_below(50),
        "gold_in_pool": len(found_ranks),
        "all_gold_in_top5": gold_count > 0 and all(0 < r <= 5 for r in ranks.values()),
        "any_gold_in_top5": any(0 < r <= 5 for r in ranks.values()),
        "candidate_pool_size": len(hits),
        "final_hit_uuids": final_uuids,
    }


# ── Stage 5 Phase 2: per-stage retrieval pipeline diagnostics ────────


def compute_pipeline_diagnostics(
    stage_snapshots: dict[str, list[str]],
    answer_session_ids: list[str],
    group_id: str,
) -> dict[str, Any]:
    """Compute per-stage gold-session ranks across the retrieval pipeline.

    Stage 5 Phase 2 (Atlas/Codex amendment to old Stage 5 plan): the
    existing ``compute_retrieval_diagnostics`` only records the FINAL
    post-everything rank, which led to a dangerous misread (Codex caught:
    ``docs/eval/codex-stage5-review.md``). We thought Bucket C gold sessions
    were below the prompt cap; they were already in the prompt, the model
    just wasn't using them.

    This function records ranks at EVERY pipeline stage so we can label
    each still-wrong question by failure mode:

      - Not retrieved anywhere: gold ranks all -1 across every stage.
        Fix: improve channels (BM25 boost, query rewrite).

      - Retrieved then dropped: gold has a positive rank in early stages
        but -1 later. Tells us exactly which cap dropped it.
        Fix: widen that cap surgically, or adjust filter.

      - Visible but ignored: gold present in final_chrono but model still
        wrong. Fix: salience (evidence packet, two-lane temporal).

    Args:
        stage_snapshots: dict {stage_name: [uuid, uuid, ...]} populated by
            ``retrieve_with_omega_recipe`` at each stage. Stage names:
            ``expanded_primary``, ``raw_secondary``, ``merged_pre_rerank``,
            ``pure_vec``, ``pure_kw``, ``weighted_rerank``, ``temporal_boost``,
            ``filtered``, ``final_chrono``.
        answer_session_ids: oracle gold sessions for this question.
        group_id: per-question UUID prefix.

    Returns:
        Dict ``{"pipeline_stage_ranks": {stage_name: {sid: rank, ...}}}``
        — ranks are 1-based, -1 if missing at that stage. ``stage_pool_sizes``
        is included so we can see at a glance "Stage X had N candidates."
    """
    gold_set = set(answer_session_ids)

    per_stage_ranks: dict[str, dict[str, int]] = {}
    per_stage_sizes: dict[str, int] = {}

    for stage_name, uuids in stage_snapshots.items():
        ranks_at_stage: dict[str, int] = {sid: -1 for sid in gold_set}
        for rank, uid in enumerate(uuids, start=1):
            sid = _extract_session_id(str(uid), group_id)
            if sid in gold_set and ranks_at_stage[sid] == -1:
                ranks_at_stage[sid] = rank
        per_stage_ranks[stage_name] = ranks_at_stage
        per_stage_sizes[stage_name] = len(uuids)

    return {
        "pipeline_stage_ranks": per_stage_ranks,
        "pipeline_stage_sizes": per_stage_sizes,
    }


# ── Resume / output handling ──────────────────────────────────────────


def load_done_question_ids(output_path: Path) -> set[str]:
    """Read existing JSONL output to find already-answered question_ids.

    Skips rows with an ``error`` field — those need retry, not skip.
    """
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                # Failed row — skip from "done" set so it gets retried.
                continue
            qid = row.get("question_id")
            if qid:
                done.add(qid)
    return done


# ── Ingestion ─────────────────────────────────────────────────────────


def ingest_question_haystack(
    *,
    driver: Any,
    chroma_collection: Any,
    question_data: dict,
    group_id: str,
) -> int:
    """Ingest the question's haystack sessions as :LMETestEpisode nodes.

    Wipes any prior LMETestEpisode rows for this group_id first
    (idempotent — supports re-runs after a mid-run crash). Each session
    becomes ONE node:
      - content = role-prefixed concat of turns (OMEGA recipe)
      - referenced_date = ISO from haystack_dates[i]
      - group_id = unique-per-question lme_q_<id>
      - agent_id = "benchmark-longmemeval"
      - namespace = :LMETestEpisode

    Returns the number of sessions ingested.
    """
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    session_dates = question_data["haystack_dates"]

    # Wipe stale nodes from a prior run of this exact question.
    with driver.session() as db:
        db.run(
            f"MATCH (n:{LME_NEO4J_LABEL} {{group_id: $gid}}) DETACH DELETE n",
            gid=group_id,
        )

    # Drop any prior Chroma rows for this group.
    try:
        chroma_collection.delete(where={"group_id": group_id})
    except Exception:
        # Collection may be fresh — ignore.
        pass

    ids_batch: list[str] = []
    docs_batch: list[str] = []
    meta_batch: list[dict[str, Any]] = []
    n_ingested = 0

    for i, (turns, sid, raw_date) in enumerate(zip(sessions, session_ids, session_dates)):
        content = format_session_text(turns)
        if not content.strip():
            continue
        ref_date = parse_longmemeval_date(raw_date)
        # Use a unique per-question UUID. Include the haystack INDEX
        # because LongMemEval has a few questions (13/500 in s_cleaned)
        # where the same session_id appears twice in haystack_session_ids
        # — Chroma rejects upserts with duplicate IDs in one batch.
        uid = f"{group_id}__{i:03d}_{sid}"

        with driver.session() as db:
            db.run(
                f"""
                CREATE (n:{LME_NEO4J_LABEL} {{
                    uuid: $uid,
                    content: $content,
                    group_id: $gid,
                    memory_type: 'session_summary',
                    episode_type: 'session_summary',
                    referenced_date: $ref_date,
                    created_at: datetime($created_at),
                    t_created: datetime($created_at),
                    importance: 0.5,
                    lifecycle_status: 'active',
                    access_count: 0,
                    agent_id: $agent_id,
                    note_index: $idx
                }})
                """,
                uid=uid,
                content=content,
                gid=group_id,
                ref_date=ref_date,
                created_at=ref_date if "T" in ref_date else "2024-01-01T00:00:00",
                agent_id=LME_AGENT_ID,
                idx=i,
            )

        ids_batch.append(uid)
        docs_batch.append(content)
        meta_batch.append({
            "wing": group_id,        # eval.py uses `wing` for group_id
            "group_id": group_id,
            "memory_type": "session_summary",
            "referenced_date": ref_date,
            "created_at": ref_date if "T" in ref_date else "2024-01-01T00:00:00",
            "note_index": i,
        })
        n_ingested += 1

    if ids_batch:
        chroma_collection.upsert(ids=ids_batch, documents=docs_batch, metadatas=meta_batch)

    # Stage 5 v2 Phase 8 — typed observation extraction.
    # Adds a small per-session gpt-4o-mini call that produces structured
    # rows {type, key, value, date, details}. Stored as :LMETestObservation
    # nodes linked back to their Episode via EVIDENCED_BY. Retrieval will
    # surface them BEFORE chronological notes in the prompt. Additive only:
    # if extraction fails, the question still runs from raw sessions.
    if LME_OBSERVATIONS_ENABLED and ids_batch:
        try:
            ingest_observations(
                driver=driver,
                chroma_collection=chroma_collection,
                episode_ids=ids_batch,
                episode_docs=docs_batch,
                episode_metas=meta_batch,
                group_id=group_id,
            )
        except Exception as e:  # noqa: BLE001
            # Phase 8 must not block the question pipeline. Log and move on.
            logger.warning("observation ingestion failed (graceful skip): %s", e)

    return n_ingested


def ingest_observations(
    *,
    driver: Any,
    chroma_collection: Any,
    episode_ids: list[str],
    episode_docs: list[str],
    episode_metas: list[dict[str, Any]],
    group_id: str,
) -> int:
    """Phase 8 — extract and persist typed observations for each session.

    Calls gpt-4o-mini per session (parallel, max 4 concurrent) to produce
    a small list of typed observations. Each observation becomes one
    :LMETestObservation node connected to its source :LMETestEpisode via
    EVIDENCED_BY. Observations also go into the same Chroma collection
    so the retrieval pipeline can surface them via vector search.

    Returns the count of observation nodes created.
    """
    from scripts.longmemeval.extract import (
        ExtractionCache,
        extract_observations_batch,
    )

    # Wipe any prior observations for this group_id (idempotent re-runs).
    with driver.session() as db:
        db.run(
            f"MATCH (n:{LME_OBSERVATION_LABEL} {{group_id: $gid}}) DETACH DELETE n",
            gid=group_id,
        )

    # Build (session_id, text, date) tuples for the batch extractor.
    # Use the Episode UUID as the session_id so we can map back later.
    sessions: list[tuple[str, str, str]] = []
    for uid, doc, meta in zip(episode_ids, episode_docs, episode_metas):
        sessions.append((uid, doc, meta.get("referenced_date", "")))

    # Optional extraction cache. Identical inputs → identical observations,
    # so a cache hit is byte-for-byte equivalent to re-running OpenAI.
    cache = None
    if LME_EXTRACTION_CACHE_PATH:
        try:
            cache = ExtractionCache(LME_EXTRACTION_CACHE_PATH)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not open extraction cache at %s: %s",
                           LME_EXTRACTION_CACHE_PATH, e)
            cache = None

    obs_by_episode = extract_observations_batch(
        sessions=sessions, max_workers=4, cache=cache,
    )

    if cache is not None:
        logger.info("ingest_observations: %s", cache.stats_summary())

    n_obs = 0
    obs_ids: list[str] = []
    obs_docs: list[str] = []
    obs_metas: list[dict[str, Any]] = []

    for episode_uid, observations in obs_by_episode.items():
        if not observations:
            continue
        episode_meta = next(
            (m for u, m in zip(episode_ids, episode_metas) if u == episode_uid),
            None,
        )
        if episode_meta is None:
            continue

        for j, obs in enumerate(observations):
            obs_uid = f"{episode_uid}__obs_{j:02d}"
            obs_text = obs.render_line()  # human-readable single line for vector search
            obs_dict = obs.to_dict()

            # Create the Observation node + EVIDENCED_BY edge to its Episode.
            with driver.session() as db:
                db.run(
                    f"""
                    MATCH (e:{LME_NEO4J_LABEL} {{uuid: $eid}})
                    CREATE (o:{LME_OBSERVATION_LABEL} {{
                        uuid: $oid,
                        group_id: $gid,
                        agent_id: $agent_id,
                        memory_type: 'observation',
                        episode_type: 'observation',
                        obs_type: $otype,
                        obs_key: $okey,
                        obs_value: $ovalue,
                        obs_date: $odate,
                        obs_details: $odetails,
                        content: $content,
                        referenced_date: $ref_date,
                        created_at: datetime($created_at),
                        t_created: datetime($created_at),
                        importance: 0.5,
                        lifecycle_status: 'active',
                        access_count: 0,
                        source_episode_uid: $eid
                    }})
                    CREATE (o)-[:EVIDENCED_BY]->(e)
                    """,
                    eid=episode_uid,
                    oid=obs_uid,
                    gid=group_id,
                    agent_id=LME_AGENT_ID,
                    otype=obs_dict["type"],
                    okey=obs_dict["key"],
                    ovalue=obs_dict["value"],
                    odate=obs_dict["date"] or "",
                    odetails=obs_dict["details"],
                    content=obs_text,
                    ref_date=episode_meta.get("referenced_date", ""),
                    created_at=episode_meta.get("created_at", "2024-01-01T00:00:00"),
                )

            obs_ids.append(obs_uid)
            obs_docs.append(obs_text)
            obs_metas.append({
                "wing": group_id,
                "group_id": group_id,
                "memory_type": "observation",
                "obs_type": obs_dict["type"],
                "obs_key": obs_dict["key"],
                "source_episode_uid": episode_uid,
                "referenced_date": episode_meta.get("referenced_date", ""),
                "created_at": episode_meta.get("created_at", "2024-01-01T00:00:00"),
            })
            n_obs += 1

    if obs_ids:
        chroma_collection.upsert(ids=obs_ids, documents=obs_docs, metadatas=obs_metas)

    logger.info("Phase 8: ingested %d observations across %d episodes for %s",
                n_obs, len(episode_ids), group_id)
    return n_obs


# ── Retrieval (OMEGA's triple fan-out + classifier-driven K) ──────────


def lme_weighted_rerank(
    candidates: list[dict],
    *,
    pure_vec_hits: list[dict],
    pure_kw_hits: list[Any],
    vec_weight: float,
    kw_weight: float,
    fallback_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[dict]:
    """OMEGA-style channel-weighted RRF rerank.

    Score each candidate by its rank in three sources:
      1. Pure vector channel (weight = ``vec_weight``)
      2. Pure keyword channel (weight = ``kw_weight``)
      3. The original fused list (weight = ``fallback_weight``)

    A hit's score is the sum of ``weight / (rank + rrf_k)`` over the
    sources where it appears. Candidates not in any source get 0.0 and
    sort to the bottom. The list is mutated in-place (each hit gets
    ``_lme_weighted_score`` set) and returned sorted descending.

    ``pure_vec_hits`` items are dicts with ``uuid`` or ``id`` keys.
    ``pure_kw_hits`` items are :class:`jarvis_memory.search.keyword.Hit`
    (objects with ``.id``) — keyword search returns these directly.
    """
    vec_rank: dict[str, int] = {}
    for r, h in enumerate(pure_vec_hits):
        hid = h.get("uuid") or h.get("id")
        if hid is not None and hid not in vec_rank:
            vec_rank[hid] = r

    kw_rank: dict[str, int] = {}
    for r, h in enumerate(pure_kw_hits):
        hid = getattr(h, "id", None)
        if hid is not None and hid not in kw_rank:
            kw_rank[hid] = r

    composite_rank: dict[str, int] = {}
    for r, h in enumerate(candidates):
        hid = h.get("uuid") or h.get("id")
        if hid is not None and hid not in composite_rank:
            composite_rank[hid] = r

    for h in candidates:
        hid = h.get("uuid") or h.get("id")
        if hid is None:
            h["_lme_weighted_score"] = 0.0
            continue
        s = 0.0
        if hid in vec_rank:
            s += vec_weight / (vec_rank[hid] + rrf_k)
        if hid in kw_rank:
            s += kw_weight / (kw_rank[hid] + rrf_k)
        if hid in composite_rank:
            s += fallback_weight / (composite_rank[hid] + rrf_k)
        h["_lme_weighted_score"] = s

    candidates.sort(key=lambda h: h.get("_lme_weighted_score", 0.0), reverse=True)
    return candidates


def retrieve_with_omega_recipe(
    *,
    query: str,
    group_id: str,
    category: str,
    counting: bool,
    driver: Any,
    embedding_store: Any,
    chroma_collection: Any,
    question_date: Optional[str] = None,
    stage_snapshots: Optional[dict[str, list[str]]] = None,
) -> list[dict[str, Any]]:
    """OMEGA-style retrieval: triple fan-out + per-category K floor.

    Returns enriched-hit dicts sorted CHRONOLOGICALLY by referenced_date
    (oldest first), as required by the prompt rules ("higher note
    numbers are more recent").

    Stage 4D: when ``question_date`` is provided, applies OMEGA's
    query expansion (counting cues + resolved relative dates + entity
    extraction) to the primary retrieval call, and boosts hits whose
    ``referenced_date`` falls inside the inferred temporal window.
    Targets the dominant Stage 1.5 failure mode where gold sessions
    were retrieved but ranked too low for date-anchored questions.
    """
    from jarvis_memory.scoring import scored_search
    from scripts.longmemeval.temporal_anchor import (
        expand_query, infer_temporal_range_anchored, hit_in_temporal_window,
    )

    # Stage 5 Phase 2: optional per-stage UUID snapshots for pipeline
    # diagnostics. ``_snap(stage, hits)`` is a no-op when the caller
    # didn't pass ``stage_snapshots``. Pure side-effect — never alters
    # retrieval logic.
    #
    # Note: ``pure_kw`` from ``jarvis_memory.search.keyword`` returns
    # ``Hit`` dataclass instances, while every other stage returns plain
    # dicts. We extract UUID via attribute-or-key pattern so both work.
    def _snap(stage_name: str, hits_list: list) -> None:
        if stage_snapshots is None:
            return

        def _uuid_of(h: Any) -> str:
            # dict-style hit
            if isinstance(h, dict):
                return str(h.get("uuid") or h.get("id") or "")
            # dataclass / object-style hit (e.g. Hit from search.keyword)
            return str(
                getattr(h, "uuid", None)
                or getattr(h, "id", None)
                or ""
            )

        stage_snapshots[stage_name] = [_uuid_of(h) for h in hits_list]

    # K floor per OMEGA recipe: counting=45, multi/temporal=25, default=20.
    if counting:
        k = COUNTING_K_FLOOR
    else:
        k = K_FLOORS.get(category, 20)

    # Stage 4D: OMEGA-style query expansion + anchored temporal range.
    # Expansion is purely additive (original query stays in the string),
    # so RRF fusion still benefits from the user's exact phrasing on
    # any keyword/lexical channel.
    expanded_query = expand_query(query, question_date) if question_date else query
    temporal_range: Optional[tuple[str, str]] = (
        infer_temporal_range_anchored(query, question_date)
        if question_date else None
    )

    def _vector_search_fn(q: str, n: int) -> list[dict]:
        """Bind chroma to the per-question collection.

        Phase 8 note: when `JARVIS_LME_OBSERVATIONS=1`, the same Chroma
        collection holds both `session_summary` rows AND `observation`
        rows. The raw-session channel must filter by `memory_type` so
        observations don't leak into the [Note N] block (they get
        surfaced separately via `retrieve_observations`).
        """
        try:
            res = chroma_collection.query(
                query_texts=[q],
                n_results=min(n, 100),
                where={
                    "$and": [
                        {"group_id": {"$eq": group_id}},
                        {"memory_type": {"$eq": "session_summary"}},
                    ]
                },
            )
        except Exception:
            return []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out: list[dict] = []
        for uid, doc, dist, meta in zip(ids, docs, dists, metas):
            similarity = max(0.0, 1.0 - float(dist))  # cosine distance → similarity
            out.append({
                "uuid": uid,
                "id": uid,
                "content": doc,
                "similarity": similarity,
                "score": similarity,
                "group_id": (meta or {}).get("group_id", group_id),
                "memory_type": (meta or {}).get("memory_type", "session_summary"),
                "referenced_date": (meta or {}).get("referenced_date", ""),
                "created_at": (meta or {}).get("created_at", ""),
                "note_index": (meta or {}).get("note_index", 0),
            })
        return out

    # OMEGA does triple fan-out — primary + secondary unfiltered + tertiary
    # with raw query. We approximate with TWO calls. Stage 4D: pass the
    # OMEGA-EXPANDED query (counting cues + dates + entities) to the
    # primary call, and the RAW query to the secondary call. This mirrors
    # OMEGA's secondary+tertiary pattern (line 982-1011 of their script).
    primary = scored_search(
        query=expanded_query,
        group_id=group_id,
        namespace=LME_NEO4J_LABEL,
        limit=k,
        driver=driver,
        embedding_store=embedding_store,
        vector_search_fn=_vector_search_fn,
        include_expansion=True,
    )
    _snap("expanded_primary", primary)
    seen_ids = {h.get("uuid") or h.get("id") for h in primary}

    secondary = scored_search(
        query=query,
        group_id=group_id,
        namespace=LME_NEO4J_LABEL,
        limit=k,
        driver=driver,
        embedding_store=embedding_store,
        vector_search_fn=_vector_search_fn,
        include_expansion=False,  # raw query
    )
    _snap("raw_secondary", secondary)
    for h in secondary:
        hid = h.get("uuid") or h.get("id")
        if hid and hid not in seen_ids:
            primary.append(h)
            seen_ids.add(hid)
    _snap("merged_pre_rerank", primary)

    # Stage 1.5: OMEGA-style channel re-weighting. ``scored_search`` fuses
    # vec/keyword with EQUAL weights via RRF; OMEGA's recipe applies
    # per-category × per-intent multipliers BEFORE fusion (e.g. KU prefers
    # keyword 1.4× vector 0.8×, NAVIGATIONAL slams keyword 2.0× vector
    # 0.1×). We can't unwind scored_search's internal RRF, so we run pure
    # vector + pure keyword side-by-side at higher recall to get clean
    # per-channel rank positions, then re-score every candidate by
    # weighted-RRF and re-sort. Hits found only via expansion/PPR (not in
    # either pure channel) keep their position via a small fallback term
    # on the existing fused rank — preserves OMEGA's "triple fan-out" idea
    # while letting the channel weights actually bite.
    intent = classify_lme_intent(query)
    vec_w, kw_w = channel_weights(category, intent)

    # Per-channel pure ranks at higher recall depth (k * 2) so re-rank has
    # signal beyond the cutoff.
    pool_n = max(k * 2, 60)
    pure_vec = _vector_search_fn(query, pool_n)
    _snap("pure_vec", pure_vec)
    try:
        from jarvis_memory.search.keyword import keyword_search
        pure_kw = keyword_search(
            query=query,
            k=pool_n,
            namespace=LME_NEO4J_LABEL,
            driver=driver,
            include_pages=False,  # LME has no pages — only episode hits matter
        )
    except Exception:
        # Keyword channel unreachable (Neo4j hiccup, fulltext-index miss,
        # whatever) → fall back to weighted vector only. Better than crashing.
        pure_kw = []
    _snap("pure_kw", pure_kw)

    primary = lme_weighted_rerank(
        primary,
        pure_vec_hits=pure_vec,
        pure_kw_hits=pure_kw,
        vec_weight=vec_w,
        kw_weight=kw_w,
    )
    _snap("weighted_rerank", primary)

    # Stage 4D: temporal-window boost. Hits whose ``referenced_date``
    # falls inside the inferred window get a 1.5× multiplier on their
    # weighted score — pushes date-relevant sessions toward the top
    # without dropping anything (which would destroy recall on edge
    # cases). Failure analysis on Stage 1.5 still-wrongs showed gold
    # sessions for date-anchored questions are RETRIEVED but ranked
    # 11-20, where the LLM ignores them. This boost lifts them.
    if temporal_range is not None:
        for h in primary:
            ref_date = str(h.get("referenced_date") or h.get("created_at") or "")
            if ref_date and hit_in_temporal_window(ref_date, temporal_range):
                h["_lme_weighted_score"] = (
                    float(h.get("_lme_weighted_score", 0.0)) * 1.5
                )
        primary.sort(
            key=lambda h: h.get("_lme_weighted_score", 0.0),
            reverse=True,
        )
    _snap("temporal_boost", primary)

    # Apply OMEGA's adaptive filter (per-category min_rel / min_res / max_res).
    # Phase 9 keeps top candidates for MS counting instead of threshold-dropping
    # low-score events that the two-pass enumerator can still use.
    kept = apply_adaptive_filter(
        primary,
        category=category,
        counting=counting,
    )
    _snap("filtered", kept)

    # Recency boost for knowledge-update — OMEGA recipe (line 945).
    if category == "knowledge-update" and kept:
        cfg = FILTER_CONFIG.get(category, FILTER_CONFIG["single-session-user"])
        max_res = int(cfg["max_res"])
        # Sort by note_index ascending so we know the oldest/newest.
        with_idx = sorted(
            kept, key=lambda h: int(h.get("note_index") or 0)
        )
        n = len(with_idx)
        if n > 1:
            for i, h in enumerate(with_idx):
                # Stage 3 (2026-04-27): bumped recency multiplier 0.5 → 0.8
                # so the freshest fact dominates more aggressively for KU.
                # Linearly scale 1.0× (oldest) to 1.8× (newest).
                frac = i / (n - 1)
                h["_kept_score"] = _hit_score(h) * (1.0 + 0.8 * frac)
            kept = sorted(with_idx, key=lambda h: h["_kept_score"], reverse=True)[:max_res]

    # Sort by referenced_date ascending (oldest → newest) for the prompt.
    kept.sort(key=_hit_date_key)
    _snap("final_chrono", kept)
    return kept


# ── Generation ────────────────────────────────────────────────────────


_OPUS_MODEL: str = "claude-opus-4-7"
_GPT4O_MODEL: str = "gpt-4o-2024-08-06"
_GPT41_MODEL: str = "gpt-4.1"


def call_llm(*, answerer: str, prompt: str, max_tokens: int) -> str:
    """Single-shot LLM generation. Temperature=0. No best-of-N.

    Supports three answerers:
      - opus  → Anthropic claude-opus-4-7 via ANTHROPIC_API_KEY
      - gpt4o → OpenAI gpt-4o-2024-08-06 via OPENAI_API_KEY
      - gpt41 → OpenAI gpt-4.1 via OPENAI_API_KEY

    Errors are logged and return an empty string — the runner records
    a failure row for that question and moves on.
    """
    if answerer == "opus":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic")
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=_OPUS_MODEL,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    if answerer in ("gpt4o", "gpt41"):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai SDK not installed; pip install openai")
        # Postmortem 2026-04-28: explicit per-request timeout. The OpenAI
        # Python client has NO default request timeout — when OpenAI's edge
        # closes a TCP connection mid-stream (CLOSE_WAIT half-close), the
        # client sits in recv() forever. We saw 4 workers wedged for 90+
        # minutes on a single benchmark run because of this. 120s covers
        # the 99th percentile of gpt-4.1 long-prompt completions; anything
        # past that is safer to retry than wait on. max_retries=2 lets the
        # SDK recycle the connection on transient failures.
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=120.0,
            max_retries=2,
        )
        model = _GPT4O_MODEL if answerer == "gpt4o" else _GPT41_MODEL
        # Stage 0: pass ``seed=`` for reproducibility. OpenAI honors this on
        # gpt-4o + gpt-4.1 — combined with temperature=0 it shaves run-to-run
        # noise so we can tell signal from variance when iterating.
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            seed=RUN_SEED,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    raise ValueError(f"Unknown answerer: {answerer!r} (use opus|gpt4o|gpt41)")


# ── Phase 8: observation retrieval ────────────────────────────────────


def retrieve_observations(
    *,
    chroma_collection: Any,
    query: str,
    group_id: str,
    top_k: int = 10,
) -> list[dict]:
    """Vector-search the per-question Chroma collection for observations.

    Phase 8: observations live in the same collection as raw sessions but
    are tagged ``memory_type="observation"`` in their metadata. We filter
    by group + type and return up to ``top_k`` ranked by cosine similarity.

    Returns a list of dicts with ``content`` (the rendered observation
    line, e.g. ``"- fact: user_age = 32"``) and ``similarity``. Empty list
    on any error — Phase 8 is additive, not load-bearing.
    """
    if not query or top_k <= 0:
        return []
    try:
        res = chroma_collection.query(
            query_texts=[query],
            n_results=min(top_k, 100),
            where={
                "$and": [
                    {"group_id": {"$eq": group_id}},
                    {"memory_type": {"$eq": "observation"}},
                ]
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("observation retrieval failed (graceful skip): %s", e)
        return []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out: list[dict] = []
    for uid, doc, dist, meta in zip(ids, docs, dists, metas):
        similarity = max(0.0, 1.0 - float(dist))
        out.append({
            "uuid": uid,
            "content": doc,
            "similarity": similarity,
            "obs_type": (meta or {}).get("obs_type", ""),
            "obs_key": (meta or {}).get("obs_key", ""),
            "referenced_date": (meta or {}).get("referenced_date", ""),
            "source_episode_uid": (meta or {}).get("source_episode_uid", ""),
        })
    return out


def format_observations_block(observations: list[dict]) -> str:
    """Render a compact "Structured evidence" block to prepend to the prompt.

    The lines are already in render_line() format (e.g.
    ``- event: yoga_class = 5th class [2023-06-12] — vinyasa style``).
    """
    if not observations:
        return ""
    lines = [obs.get("content", "").strip() for obs in observations if obs.get("content")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return (
        "[Structured evidence (extracted observations from prior sessions)]\n"
        + "\n".join(lines)
        + "\n[End structured evidence]"
    )


# ── Per-question pipeline ─────────────────────────────────────────────


def run_one_question(
    *,
    q: dict,
    answerer: str,
    driver: Any,
    embedding_store: Any,
    chroma_collection: Any,
    oracle_answer_session_ids: Optional[list[str]] = None,
    oracle_category: Optional[str] = None,
) -> dict:
    """Execute the full pipeline for a single LongMemEval question.

    Returns the JSONL row to write. On crash, returns a row with
    ``hypothesis=""`` and ``error=<traceback>`` so the run continues.

    When ``oracle_answer_session_ids`` is provided (Stage 0 diagnostics
    mode), retrieval-quality stats are computed from ``hits`` BEFORE
    generation and attached to the row under ``"diagnostics"``. The
    oracle data does NOT alter the prompt or generation in any way —
    it's purely observational.

    When ``oracle_category`` is provided (Stage 1 ``--use-oracle-categories``),
    that label is used as the category instead of running the heuristic
    classifier. The classifier still runs as a shadow so we can log its
    prediction (``shadow_classifier_label``) for later analysis. This
    leaks oracle ``question_type`` into routing — fine for benchmark-
    competitive runs, NOT fine for the production-honest baseline.
    """
    qid = q["question_id"]
    # Defensive defaults so the error row in the except block always
    # carries readable category info even if setup raises before the
    # classifier runs.
    category = "unknown"
    classifier_rule = ""
    shadow_classifier_label = ""
    category_source = "classifier"
    counting = False
    diagnostics: Optional[dict] = None
    n_sessions = 0
    hits: list[dict[str, Any]] = []
    n_hits_pre_trim = 0
    top_score = 0.0
    max_tokens = 1024
    abstention_fired = False
    total_line_appended = False
    # Stage 1.5 defaults — populated after classification succeeds.
    lme_intent = "NEUTRAL"
    lme_channel_weights: tuple[float, float] = (1.0, 1.0)
    # Stage 4A defaults — populated only on multi-session counting questions.
    ms_two_pass_used = False
    ms_pass1_chars = 0
    # Stage 4D defaults — populated when question has a date anchor.
    lme_temporal_window: Optional[tuple[str, str]] = None
    lme_query_expanded = False
    # Stage 5 v2 Phase 8 default — populated when JARVIS_LME_OBSERVATIONS=1.
    n_observations = 0
    # Phase 11 default — populated when the deterministic evidence ledger is on.
    n_evidence_ledger_lines = 0
    # Phase 11.1 default — populated when deterministic answer scaffolds are on.
    n_answer_scaffold_rows = 0

    t0 = time.time()
    try:
        question = q["question"]
        raw_qdate = q.get("question_date", "")
        qdate = parse_longmemeval_date(raw_qdate) if raw_qdate else ""

        group_id = f"{LME_GROUP_PREFIX}_{qid}"
        classification = classify(question)
        shadow_classifier_label = classification.label
        classifier_rule = classification.rule
        if oracle_category:
            category = oracle_category
            category_source = "oracle"
        else:
            category = classification.label
            category_source = "classifier"
        counting = is_counting_question(question)

        cfg = FILTER_CONFIG.get(category, FILTER_CONFIG["single-session-user"])
        max_tokens = int(cfg["max_tokens"])

        # Stage 1.5: capture intent + channel weights so they show up in
        # the JSONL row alongside category. Recomputed inside
        # ``retrieve_with_omega_recipe`` (cheap regex scan); the duplicate
        # call avoids threading another return value through the function.
        lme_intent = classify_lme_intent(question)
        lme_channel_weights = channel_weights(category, lme_intent)

        # Stage 4D: capture temporal-anchor diagnostics. Same dup-pattern
        # as Stage 1.5 — both are pure-regex helpers, microsecond cost,
        # cleaner than threading return values.
        from scripts.longmemeval.temporal_anchor import (
            expand_query, infer_temporal_range_anchored,
        )
        if qdate:
            lme_temporal_window = infer_temporal_range_anchored(question, qdate)
            lme_query_expanded = expand_query(question, qdate) != question

        # 1. Ingest haystack into isolated namespace.
        n_sessions = ingest_question_haystack(
            driver=driver,
            chroma_collection=chroma_collection,
            question_data=q,
            group_id=group_id,
        )

        # 2. Retrieve via OMEGA recipe (with Stage 4D temporal anchor).
        # Stage 5 Phase 2: collect per-stage UUID snapshots for pipeline
        # diagnostics when oracle is available. ``stage_snapshots`` stays
        # None for non-diagnostic runs so the snapshot is a true no-op.
        stage_snapshots: Optional[dict[str, list[str]]] = (
            {} if oracle_answer_session_ids is not None else None
        )
        hits = retrieve_with_omega_recipe(
            query=question,
            group_id=group_id,
            category=category,
            counting=counting,
            driver=driver,
            embedding_store=embedding_store,
            chroma_collection=chroma_collection,
            question_date=qdate,
            stage_snapshots=stage_snapshots,
        )

        # 3. Confidence diagnostics ONLY — do not suppress max_tokens
        # or truncate hits based on our scoring scale. OMEGA's 0.20
        # threshold is calibrated to their similarity score; our RRF
        # composite + cosine similarity live on different scales, so
        # a hard cutoff would over-abstain. Instead we trust the
        # prompt rule "If the question cannot be answered ... say so"
        # — gpt-4o / Opus / gpt-4.1 all honor it reliably.
        if hits:
            top_score = max(
                float(h.get("similarity") or h.get("score") or 0.0)
                for h in hits
            )
        else:
            top_score = 0.0

        # 3.4. Stage 1: per-category prompt-context budget. Drops
        # lowest-scored hits until cumulative content fits — keeps the
        # prompt focused on the most relevant sessions and avoids
        # distract-the-LLM failures on single-session questions.
        n_hits_pre_trim = len(hits)
        hits = trim_to_context_budget(hits, category)

        # 3.5. (Optional) gold-session retrieval diagnostics — Stage 0.
        # Computed on the final hits list (what enters the prompt) so we
        # answer "did the LLM see the gold session?" cleanly. Pure
        # observation; never feeds into prompt or generation. Computed
        # BEFORE generation so a generation crash still preserves the
        # retrieval signal in the error row.
        if oracle_answer_session_ids is not None:
            diagnostics = compute_retrieval_diagnostics(
                hits=hits,
                answer_session_ids=oracle_answer_session_ids,
                group_id=group_id,
            )
            # Stage 5 Phase 2: pipeline-stage rank diagnostics. Tells us
            # WHERE in the pipeline gold drops, not just whether it
            # entered the prompt. Critical for designing surgical fixes.
            if stage_snapshots is not None:
                diagnostics.update(
                    compute_pipeline_diagnostics(
                        stage_snapshots=stage_snapshots,
                        answer_session_ids=oracle_answer_session_ids,
                        group_id=group_id,
                    )
                )

        # 4. Format sessions block (shared across single-pass + two-pass).
        raw_sessions_text = "\n\n".join(
            format_session_for_prompt(
                content=h.get("content", ""),
                date_str=str(h.get("referenced_date") or h.get("created_at") or ""),
                index=i + 1,  # 1-indexed for [Note N] readability
            )
            for i, h in enumerate(hits)
        )
        prefix_blocks: list[str] = []

        # 4.0a. Phase 11.1 — deterministic answer scaffolds for cases where
        # GPT-4.1 needs structured bookkeeping more than more prose.
        if LME_ANSWER_SCAFFOLD_ENABLED:
            scaffold_block, n_answer_scaffold_rows = build_answer_scaffold(
                hits=hits,
                question=question,
                category=category,
            )
            if scaffold_block:
                prefix_blocks.append(scaffold_block)

        # 4.0. Phase 11 — deterministic evidence ledger. Most Phase 10
        # failures already had gold sessions in the final prompt, but the
        # prompt was ~200k chars. The ledger lifts compact, question-matching
        # user/assistant turns above the raw notes so answer synthesis has a
        # smaller working surface. Raw notes remain available below.
        if LME_EVIDENCE_LEDGER_ENABLED:
            ledger_block, n_evidence_ledger_lines = build_evidence_ledger(
                hits=hits,
                question=question,
                category=category,
            )
            if ledger_block:
                prefix_blocks.append(ledger_block)

        # 4.1. Stage 5 v2 Phase 8 — prepend a structured-evidence block of
        # extracted observations BEFORE the chronological notes. The model
        # gets both: a compact list of typed facts ("- fact: user_age = 32
        # [2023-04-11]") at the top, then the full session transcripts
        # below. This is the move AgentMemory uses to beat OMEGA on
        # multi-session: the LLM sees ENUMERATED facts plus their evidence,
        # instead of having to derive them from paragraph-long quotes.
        # Additive: failures fall through silently to the original block.
        if LME_OBSERVATIONS_ENABLED:
            obs_hits = retrieve_observations(
                chroma_collection=chroma_collection,
                query=question,
                group_id=group_id,
                top_k=LME_OBSERVATION_TOP_K,
            )
            n_observations = len(obs_hits)
            obs_block = format_observations_block(obs_hits)
            if obs_block:
                prefix_blocks.append(obs_block)

        sessions_text = "\n\n".join([*prefix_blocks, raw_sessions_text])

        # 4.5. Stage 1: confidence-based abstention guard. When retrieval
        # is weak AND the question names an entity nobody mentions, prepend
        # an explicit abstention rule. Targets 4 _abs questions in the
        # baseline that hallucinated answers instead of saying "not enough".
        abstention_prefix = maybe_build_abstention_prefix(
            question=question,
            hits=hits,
            top_score=top_score,
        )
        abstention_fired = abstention_prefix is not None

        # 5. Generate. Stage 4A: multi-session counting questions use a
        # two-pass extract-then-count flow — the single-pass MS prompt
        # under-counts because the model prunes during enumeration. Split
        # into a high-recall extract pass + a high-precision count pass.
        # Failure analysis on Stage 1.5 still-wrongs found 0/17 retrieval
        # misses, confirming this is purely a generation problem.
        ms_two_pass_used = (category == "multi-session" and counting)
        if ms_two_pass_used:
            pass1_prompt = render_ms_extract_prompt(
                sessions=sessions_text,
                question=question,
                question_date=qdate,
            )
            pass1_hyp = call_llm(
                answerer=answerer,
                prompt=pass1_prompt,
                max_tokens=max_tokens,
            )
            ms_pass1_chars = len(pass1_hyp)

            pass2_prompt = render_ms_count_prompt(
                sessions=sessions_text,
                candidate_list=pass1_hyp,
                question=question,
                question_date=qdate,
            )
            if abstention_prefix is not None:
                pass2_prompt = abstention_prefix + pass2_prompt
            hypothesis = call_llm(
                answerer=answerer,
                prompt=pass2_prompt,
                max_tokens=max_tokens,
            )
            # Stage 5 Phase 2: prompt hash on the FINAL prompt that
            # produced the hypothesis (pass-2 for MS two-pass).
            final_prompt_for_hash = pass2_prompt
        else:
            prompt = render_prompt(
                category=category,
                sessions=sessions_text,
                question=question,
                question_date=qdate,
            )
            if abstention_prefix is not None:
                prompt = abstention_prefix + prompt
            hypothesis = call_llm(
                answerer=answerer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            final_prompt_for_hash = prompt

        # Stage 5 Phase 2: hash the final prompt + record it on
        # diagnostics. Lets us diff prompt content between runs (parity
        # check, regression bisect) without storing the full prompt.
        if oracle_answer_session_ids is not None:
            import hashlib
            diagnostics["prompt_hash"] = hashlib.sha256(
                final_prompt_for_hash.encode("utf-8")
            ).hexdigest()[:16]
            diagnostics["prompt_chars"] = len(final_prompt_for_hash)

        # 5.5. Stage 2: defensive total-line post-process for MS counting
        # questions when the LLM forgot the "Total: N" final line. No-op
        # for any other category or non-counting MS question.
        hypothesis_pre_post = hypothesis
        hypothesis = maybe_append_total_line(hypothesis, category, counting)
        total_line_appended = hypothesis != hypothesis_pre_post

        elapsed = time.time() - t0
        row = {
            "question_id": qid,
            "hypothesis": hypothesis,
            "predicted_category": category,
            "category_source": category_source,
            "shadow_classifier_label": shadow_classifier_label,
            "classifier_rule": classifier_rule,
            "counting": counting,
            "lme_intent": lme_intent,
            "lme_channel_weights": [
                round(lme_channel_weights[0], 3),
                round(lme_channel_weights[1], 3),
            ],
            "n_sessions_ingested": n_sessions,
            "n_hits_used": len(hits),
            "n_hits_pre_trim": n_hits_pre_trim,
            "top_score": round(top_score, 4),
            "abstention_fired": abstention_fired,
            "total_line_appended": total_line_appended,
            "ms_two_pass_used": ms_two_pass_used,
            "ms_pass1_chars": ms_pass1_chars,
            "lme_temporal_window": (
                list(lme_temporal_window) if lme_temporal_window else None
            ),
            "lme_query_expanded": lme_query_expanded,
            "n_observations": n_observations,
            "n_evidence_ledger_lines": n_evidence_ledger_lines,
            "n_answer_scaffold_rows": n_answer_scaffold_rows,
            "max_tokens": max_tokens,
            "answerer": answerer,
            # Stage 0: ``seed`` arg is honored only for OpenAI answerers
            # (gpt4o, gpt41). Anthropic Messages API has no seed param so
            # opus runs are NOT bit-reproducible (temperature=0 only).
            "seed_honored": answerer in ("gpt4o", "gpt41"),
            "elapsed_sec": round(elapsed, 2),
        }
        if diagnostics is not None:
            row["diagnostics"] = diagnostics
        return row
    except Exception:
        elapsed = time.time() - t0
        # Preserve every observable that was captured before the crash —
        # if retrieval succeeded but generation failed, these fields
        # still contain real data and aid debugging.
        err_row: dict[str, Any] = {
            "question_id": qid,
            "hypothesis": "",
            "predicted_category": category,
            "category_source": category_source,
            "shadow_classifier_label": shadow_classifier_label,
            "classifier_rule": classifier_rule,
            "counting": counting,
            "lme_intent": lme_intent,
            "lme_channel_weights": [
                round(lme_channel_weights[0], 3),
                round(lme_channel_weights[1], 3),
            ],
            "n_sessions_ingested": n_sessions,
            "n_hits_used": len(hits),
            "n_hits_pre_trim": n_hits_pre_trim,
            "top_score": round(top_score, 4),
            "abstention_fired": abstention_fired,
            "total_line_appended": total_line_appended,
            "ms_two_pass_used": ms_two_pass_used,
            "ms_pass1_chars": ms_pass1_chars,
            "lme_temporal_window": (
                list(lme_temporal_window) if lme_temporal_window else None
            ),
            "lme_query_expanded": lme_query_expanded,
            "n_observations": n_observations,
            "n_evidence_ledger_lines": n_evidence_ledger_lines,
            "n_answer_scaffold_rows": n_answer_scaffold_rows,
            "max_tokens": max_tokens,
            "answerer": answerer,
            "seed_honored": answerer in ("gpt4o", "gpt41"),
            "elapsed_sec": round(elapsed, 2),
            "error": traceback.format_exc(),
        }
        if diagnostics is not None:
            err_row["diagnostics"] = diagnostics
        return err_row


# ── Setup ─────────────────────────────────────────────────────────────


def setup_resources():
    """Connect to Neo4j + ChromaDB + EmbeddingStore. Returns the trio."""
    from neo4j import GraphDatabase
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    from jarvis_memory.config import (
        CHROMADB_PATH,
        EMBEDDING_MODEL,
        NEO4J_PASSWORD,
        NEO4J_URI,
        NEO4J_USER,
    )
    from jarvis_memory.embeddings import EmbeddingStore

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Isolated Chroma collection — separate from prod jarvis_memories.
    # Per-worker Chroma path via JARVIS_LME_CHROMA_PATH env var (set by the
    # parallel harness) eliminates HNSW write contention when multiple workers
    # run the adapter concurrently. Falls back to the shared path for solo runs.
    chroma_path = os.getenv("JARVIS_LME_CHROMA_PATH", CHROMADB_PATH)
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try:
        collection = chroma_client.get_collection(LME_CHROMA_COLLECTION, embedding_function=ef)
    except Exception:
        collection = chroma_client.create_collection(
            name=LME_CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    embedding_store = EmbeddingStore()  # uses prod Chroma — required for scoring helpers
    return driver, embedding_store, collection


def write_run_summary(output_path: Path) -> Path:
    """Read the JSONL output and write a ``<output>_summary.json`` file.

    Stage 0 reporting. Aggregates per-category processed/errored counts
    and (when diagnostics rows are present) gold-session retrieval stats
    so we can compare runs without re-parsing 500 lines by hand. Judge
    scoring is NOT in this summary — it runs separately and produces
    its own ``.eval-results-*`` file.
    """
    if not output_path.exists():
        raise FileNotFoundError(f"output JSONL missing: {output_path}")

    n_total = 0
    n_errored = 0
    cat_total: Counter[str] = Counter()
    cat_errored: Counter[str] = Counter()
    elapsed_total = 0.0
    answerer_seen: set[str] = set()
    # Stage 1 aggregates
    abstention_fired_total = 0
    category_source_counts: Counter[str] = Counter()
    pretrim_total = 0
    pretrim_n = 0
    posttrim_total = 0
    posttrim_n = 0

    diag_n = 0
    diag_n_abstention = 0  # rows whose oracle has no answer_session_ids
    diag_all_top5 = 0
    diag_any_top5 = 0
    diag_any_top10 = 0
    diag_any_pool = 0
    diag_by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "all_top5": 0, "any_top5": 0, "any_top10": 0, "any_pool": 0}
    )

    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            cat = row.get("predicted_category", "unknown")
            cat_total[cat] += 1
            if row.get("error"):
                n_errored += 1
                cat_errored[cat] += 1
            elapsed_total += float(row.get("elapsed_sec") or 0.0)
            ans = row.get("answerer")
            if ans:
                answerer_seen.add(ans)
            if row.get("abstention_fired"):
                abstention_fired_total += 1
            cs = row.get("category_source")
            if cs:
                category_source_counts[cs] += 1
            pre = row.get("n_hits_pre_trim")
            post = row.get("n_hits_used")
            if isinstance(pre, int) and pre > 0:
                pretrim_total += pre
                pretrim_n += 1
            if isinstance(post, int) and post > 0:
                posttrim_total += post
                posttrim_n += 1

            d = row.get("diagnostics")
            if isinstance(d, dict):
                if (d.get("gold_count") or 0) <= 0:
                    # Abstention question (oracle has no answer_session_ids)
                    # — track separately so empty-gold rows don't dilute the
                    # retrieval-quality aggregates.
                    diag_n_abstention += 1
                    continue
                diag_n += 1
                if d.get("all_gold_in_top5"):
                    diag_all_top5 += 1
                if d.get("any_gold_in_top5"):
                    diag_any_top5 += 1
                if (d.get("gold_in_top10") or 0) > 0:
                    diag_any_top10 += 1
                if (d.get("gold_in_pool") or 0) > 0:
                    diag_any_pool += 1

                bucket = diag_by_cat[cat]
                bucket["n"] += 1
                if d.get("all_gold_in_top5"):
                    bucket["all_top5"] += 1
                if d.get("any_gold_in_top5"):
                    bucket["any_top5"] += 1
                if (d.get("gold_in_top10") or 0) > 0:
                    bucket["any_top10"] += 1
                if (d.get("gold_in_pool") or 0) > 0:
                    bucket["any_pool"] += 1

    def _pct(num: int, denom: int) -> Optional[float]:
        return round(num / denom, 4) if denom > 0 else None

    def _avg(t: int, n: int) -> Optional[float]:
        return round(t / n, 2) if n > 0 else None

    summary: dict[str, Any] = {
        "output_path": str(output_path),
        "answerer": sorted(answerer_seen),
        "n_total": n_total,
        "n_errored": n_errored,
        "elapsed_sec_total": round(elapsed_total, 2),
        "predicted_categories": dict(cat_total),
        "errored_by_category": dict(cat_errored),
        "abstention_fired_total": abstention_fired_total,
        "abstention_fired_pct": _pct(abstention_fired_total, n_total),
        "category_source": dict(category_source_counts),
        "avg_hits_pre_trim": _avg(pretrim_total, pretrim_n),
        "avg_hits_used": _avg(posttrim_total, posttrim_n),
    }
    if diag_n > 0 or diag_n_abstention > 0:
        summary["diagnostics"] = {
            "n_questions": diag_n,
            "n_abstention": diag_n_abstention,
            "all_gold_in_top5_pct": _pct(diag_all_top5, diag_n),
            "any_gold_in_top5_pct": _pct(diag_any_top5, diag_n),
            "any_gold_in_top10_pct": _pct(diag_any_top10, diag_n),
            "any_gold_in_pool_pct": _pct(diag_any_pool, diag_n),
            "by_predicted_category": {
                cat: {
                    "n": v["n"],
                    "all_top5_pct": _pct(v["all_top5"], v["n"]),
                    "any_top5_pct": _pct(v["any_top5"], v["n"]),
                    "any_top10_pct": _pct(v["any_top10"], v["n"]),
                    "any_pool_pct": _pct(v["any_pool"], v["n"]),
                }
                for cat, v in diag_by_cat.items()
            },
        }

    # ``runs/foo.jsonl`` → ``runs/foo.summary.json``.
    summary_path = output_path.parent / f"{output_path.stem}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary_path


def stratified_subset(dataset: list[dict], n_per_cat: int = 2) -> list[dict]:
    """Pick a stratified validation subset across all 6 categories."""
    by_cat: dict[str, list[dict]] = {}
    for q in dataset:
        if q["question_id"].endswith("_abs"):
            # one abstention question per pass too
            by_cat.setdefault("_abs", []).append(q)
        else:
            by_cat.setdefault(q["question_type"], []).append(q)
    out: list[dict] = []
    for cat, qs in by_cat.items():
        out.extend(qs[:n_per_cat])
    return out


# ── Main ──────────────────────────────────────────────────────────────


def main():
    # Stage 0: re-exec with PYTHONHASHSEED=42 if it isn't already set so
    # set/dict iteration order is deterministic across runs. Re-exec is
    # the only way — once Python is up, hash randomization is locked.
    # Skip when running under pytest (PYTEST_CURRENT_TEST is the canonical
    # marker pytest sets per-test); re-execing the test runner would be a
    # surprising side effect.
    running_under_pytest = "PYTEST_CURRENT_TEST" in os.environ
    if not running_under_pytest and os.environ.get("PYTHONHASHSEED") != str(RUN_SEED):
        os.environ["PYTHONHASHSEED"] = str(RUN_SEED)
        os.execvp(sys.executable, [sys.executable] + sys.argv)
        return 0  # unreachable; execvp replaces the process

    parser = argparse.ArgumentParser(description="LongMemEval adapter for jarvis-memory v1.1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path (resume-safe).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to longmemeval_s_cleaned.json (or oracle for diagnostics).")
    parser.add_argument("--oracle-path", type=Path, default=DEFAULT_ORACLE,
                        help="Path to longmemeval_oracle.json. Read only when --diagnostics is set.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Stage 0: load oracle answer_session_ids and log retrieval-quality "
                             "diagnostics per question. Pure observation — does NOT alter prompts "
                             "or generation. Adds ~0 cost.")
    parser.add_argument("--use-oracle-categories", action="store_true",
                        help="Stage 1: read question_type directly from the oracle dataset and "
                             "use it as the category, bypassing the heuristic classifier. The "
                             "classifier still runs as a shadow so its prediction is logged for "
                             "later analysis. Leaks oracle into routing — fine for benchmark-"
                             "competitive runs, NOT for the production-honest baseline.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many NEW questions (after resume).")
    parser.add_argument("--validate", action="store_true",
                        help="Run on a stratified 14-question subset (2 per category + 2 abs).")
    parser.add_argument("--question-id", type=str, default=None,
                        help="Run only this question_id (for debugging).")
    parser.add_argument("--answerer", type=str, default=None,
                        help="Override JARVIS_LME_ANSWERER env var. opus|gpt4o|gpt41.")
    args = parser.parse_args()

    answerer = args.answerer or os.environ.get("JARVIS_LME_ANSWERER", "")
    if answerer not in ("opus", "gpt4o", "gpt41"):
        print("ERROR: set JARVIS_LME_ANSWERER=opus|gpt4o|gpt41 (or pass --answerer)",
              file=sys.stderr)
        return 2

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    # Apply pre-registered AR1 + AR2 PPR overrides BEFORE any retrieval runs.
    apply_ppr_overrides()

    print(f"Loading dataset: {args.dataset}")
    with args.dataset.open() as f:
        dataset = json.load(f)
    print(f"  Loaded {len(dataset)} questions")

    # Optional oracle data — Stage 0 (diagnostics) and Stage 1
    # (use-oracle-categories) both source from the same file but populate
    # independent maps. Load once if either flag is set.
    oracle_answer_session_ids_by_qid: dict[str, list[str]] = {}
    oracle_category_by_qid: dict[str, str] = {}
    need_oracle = args.diagnostics or args.use_oracle_categories
    if need_oracle:
        if not args.oracle_path.exists():
            flag = "--diagnostics" if args.diagnostics else "--use-oracle-categories"
            print(f"ERROR: {flag} set but oracle missing: {args.oracle_path}",
                  file=sys.stderr)
            return 2
        print(f"Loading oracle: {args.oracle_path}")
        with args.oracle_path.open() as f:
            oracle = json.load(f)
        for o in oracle:
            qid = o.get("question_id")
            if not qid:
                continue
            if args.diagnostics:
                sids = o.get("answer_session_ids") or []
                if sids:
                    oracle_answer_session_ids_by_qid[qid] = list(sids)
            if args.use_oracle_categories:
                qtype = o.get("question_type")
                if qtype:
                    oracle_category_by_qid[qid] = qtype
        if args.diagnostics:
            print(f"  Indexed {len(oracle_answer_session_ids_by_qid)} oracle answer-session lists")
        if args.use_oracle_categories:
            print(f"  Indexed {len(oracle_category_by_qid)} oracle category labels")

    if args.question_id:
        dataset = [q for q in dataset if q["question_id"] == args.question_id]
        if not dataset:
            print(f"ERROR: question_id {args.question_id} not in dataset", file=sys.stderr)
            return 2
    elif args.validate:
        dataset = stratified_subset(dataset, n_per_cat=2)
        print(f"  Validation subset: {len(dataset)} questions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_question_ids(args.output)
    if done_ids:
        print(f"Resume: {len(done_ids)} questions already answered in {args.output}")

    todo = [q for q in dataset if q["question_id"] not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"To process: {len(todo)} questions with answerer={answerer}")

    if not todo:
        print("Nothing to do — output already complete.")
        return 0

    print("Setting up Neo4j + Chroma + embedding store...")
    driver, embedding_store, chroma_collection = setup_resources()

    n_done = 0
    n_failed = 0
    start = time.time()

    try:
        with args.output.open("a") as out:
            for q in todo:
                row = run_one_question(
                    q=q,
                    answerer=answerer,
                    driver=driver,
                    embedding_store=embedding_store,
                    chroma_collection=chroma_collection,
                    oracle_answer_session_ids=oracle_answer_session_ids_by_qid.get(q["question_id"]),
                    oracle_category=oracle_category_by_qid.get(q["question_id"]),
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_done += 1
                if row.get("error"):
                    n_failed += 1
                    print(f"  [{n_done}/{len(todo)}] {row['question_id']}  ERROR ({row['elapsed_sec']:.1f}s)")
                else:
                    cat = row.get("predicted_category", "?")
                    hyp_preview = (row.get("hypothesis") or "")[:80].replace("\n", " ")
                    print(f"  [{n_done}/{len(todo)}] {row['question_id']:30s} "
                          f"cat={cat:25s} t={row['elapsed_sec']:5.1f}s  → {hyp_preview}")
    finally:
        driver.close()

    elapsed = time.time() - start
    print(f"\nDone. {n_done} processed, {n_failed} failed, {elapsed:.0f}s total.")
    print(f"Output: {args.output}")

    # Stage 0: write per-run summary alongside the JSONL.
    try:
        summary_path = write_run_summary(args.output)
        print(f"Summary: {summary_path}")
    except Exception as e:  # noqa: BLE001 — summary is best-effort
        print(f"WARN: failed to write run summary: {e}", file=sys.stderr)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
