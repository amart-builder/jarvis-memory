"""Unit tests for the LongMemEval adapter — testable pieces only.

The adapter has integration paths (Neo4j, Chroma, LLM) we don't mock —
those are validated by the live --validate run, not unit tests. This
module covers:
  - resume / JSONL parsing
  - stratified sampling
  - AR2 seed-broadening behavior (via monkey-patch verification)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Resume: load_done_question_ids ────────────────────────────────────


def test_load_done_question_ids_empty_when_missing(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    assert load_done_question_ids(tmp_path / "nope.jsonl") == set()


def test_load_done_question_ids_reads_existing(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "a"}) + "\n"
        + json.dumps({"question_id": "q2", "hypothesis": "b"}) + "\n"
    )
    assert load_done_question_ids(p) == {"q1", "q2"}


def test_load_done_question_ids_tolerates_blank_lines(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1"}) + "\n"
        + "\n"
        + json.dumps({"question_id": "q2"}) + "\n"
    )
    assert load_done_question_ids(p) == {"q1", "q2"}


def test_load_done_question_ids_tolerates_bad_line(tmp_path):
    from scripts.run_longmemeval import load_done_question_ids
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1"}) + "\n"
        + "{bad json\n"
        + json.dumps({"question_id": "q2"}) + "\n"
    )
    # q1 is read; the bad line is skipped; q2 is read.
    assert load_done_question_ids(p) == {"q1", "q2"}


# ── stratified_subset ─────────────────────────────────────────────────


def test_stratified_subset_picks_n_per_category():
    from scripts.run_longmemeval import stratified_subset
    data = []
    for cat in ("temporal-reasoning", "multi-session", "knowledge-update",
                "single-session-user", "single-session-assistant",
                "single-session-preference"):
        for i in range(5):
            data.append({"question_id": f"{cat}_{i}", "question_type": cat})
    out = stratified_subset(data, n_per_cat=2)
    # 6 cats × 2 = 12 questions
    assert len(out) == 12
    cats = [q["question_type"] for q in out]
    for cat in ("temporal-reasoning", "multi-session", "knowledge-update",
                "single-session-user", "single-session-assistant",
                "single-session-preference"):
        assert cats.count(cat) == 2


def test_stratified_subset_includes_abstention():
    """_abs questions form their own bucket so validation hits abstention."""
    from scripts.run_longmemeval import stratified_subset
    data = [
        {"question_id": f"q_{i}", "question_type": "multi-session"} for i in range(3)
    ] + [
        {"question_id": f"abs_{i}_abs", "question_type": "multi-session"} for i in range(3)
    ]
    out = stratified_subset(data, n_per_cat=2)
    abs_ids = [q["question_id"] for q in out if q["question_id"].endswith("_abs")]
    assert len(abs_ids) == 2


# ── AR2 seed-broadening ───────────────────────────────────────────────


def test_apply_ppr_overrides_broadens_seeds():
    """After apply_ppr_overrides, common nouns ≥4 chars seed the PPR.

    Test by hijacking the patched _extract_query_entities directly.
    """
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("how often do I exercise")
        # Original (proper-noun-only) returns []. Broadened returns
        # at least "exercise" — possibly "often" too if not in stoplist.
        assert "exercise" in seeds, f"AR2 didn't seed common noun; got {seeds}"
    finally:
        # Revert patches so other tests aren't affected.
        ppr_mod._extract_query_entities = original_extract
        # Also revert the PPR function patch (apply_ppr_overrides patches both).
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_preserves_proper_noun_seeds():
    """Broadening shouldn't drop the original proper-noun extraction."""
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("decisions in Catalyst that affected Astack")
        # Both proper nouns should survive (lowercase form per existing code).
        assert "catalyst" in seeds
        assert "astack" in seeds
    finally:
        ppr_mod._extract_query_entities = original_extract
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_skips_stoplist():
    """AR2 must not seed stoplist words like 'this', 'have', 'when'."""
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides, _AR2_STOPLIST

    original_extract = ppr_mod._extract_query_entities

    try:
        apply_ppr_overrides()
        seeds = ppr_mod._extract_query_entities("when have I been doing this")
        for word in seeds:
            assert word not in _AR2_STOPLIST, f"stoplist word leaked: {word}"
    finally:
        ppr_mod._extract_query_entities = original_extract
        from jarvis_memory.search.ppr import personalized_pagerank as orig_ppr_fn
        ppr_mod.personalized_pagerank = orig_ppr_fn


def test_apply_ppr_overrides_sets_damping_to_05():
    """AR1: PPR damping defaults to 0.5 after overrides applied.

    Test by inspecting the wrapper's closure: ``ppr_with_alpha`` is a
    closure that holds the original PPR function. We inspect it by
    monkey-patching the original at the source FIRST, then applying
    overrides — so the captured closure sees our spy.
    """
    from jarvis_memory.search import ppr as ppr_mod
    from scripts.run_longmemeval import apply_ppr_overrides

    original_extract = ppr_mod._extract_query_entities
    original_ppr = ppr_mod.personalized_pagerank
    captured: dict = {}

    def spy(query, **kwargs):
        captured["damping"] = kwargs.get("damping")
        return []

    try:
        # Replace the source PPR with our spy BEFORE applying overrides.
        ppr_mod.personalized_pagerank = spy
        apply_ppr_overrides()  # captures `spy` as `_orig_ppr` in its closure
        # Now call the wrapper — it must pass damping=0.5 to spy.
        ppr_mod.personalized_pagerank("any query", driver=None)
        assert captured["damping"] == 0.5, f"AR1: got damping={captured.get('damping')}"
    finally:
        ppr_mod._extract_query_entities = original_extract
        ppr_mod.personalized_pagerank = original_ppr
