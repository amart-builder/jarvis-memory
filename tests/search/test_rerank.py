"""Cross-encoder rerank tests.

Mocks the model so tests don't download ~568MB of weights or require a
network. The rerank module must:

  * Reorder candidates by cross-encoder score (descending).
  * Add a ``rerank_score`` key to each returned dict.
  * Fail open: return inputs unchanged when the model is missing or
    inference raises.
  * Respect the ``JARVIS_RERANK=0`` env flag.
  * Pick the right text field across episode / page / legacy shapes.
"""
from __future__ import annotations

from typing import Any

import pytest

import jarvis_memory.search.rerank as rerank_mod
from jarvis_memory.search.rerank import rerank, reset_model_cache, _extract_text


# ── Fake reranker model ─────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, doc_id: int, score: float):
        self.doc_id = doc_id
        self.score = score
        self.rank = None


class _FakeRanked:
    def __init__(self, results):
        self._results = results

    def __iter__(self):
        return iter(self._results)


class _FakeModel:
    """Scores each doc by token overlap with the query — deterministic."""

    def __init__(self, raise_on_rank: bool = False):
        self.raise_on_rank = raise_on_rank
        self.calls: list[tuple[str, list[str]]] = []

    def rank(self, query: str, docs, doc_ids=None):
        self.calls.append((query, list(docs)))
        if self.raise_on_rank:
            raise RuntimeError("synthetic rerank failure")
        q_tokens = set(query.lower().split())
        results = []
        for i, d in enumerate(docs):
            d_tokens = set(d.lower().split())
            score = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
            results.append(_FakeResult(doc_id=i if doc_ids is None else doc_ids[i], score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return _FakeRanked(results)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset module-level caches between tests so they don't leak state."""
    reset_model_cache()
    monkeypatch.setenv("JARVIS_RERANK", "1")
    yield
    reset_model_cache()


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(rerank_mod, "_get_model", lambda *_a, **_k: model)
    return model


# ── Tests ──────────────────────────────────────────────────────────────


def test_rerank_reorders_by_query_relevance(fake_model):
    """Doc with stronger query overlap should be ranked first regardless of input order."""
    candidates = [
        {"uuid": "a", "content": "auth provider configuration is unrelated"},
        {"uuid": "b", "content": "navi pricing setup decision finalized today"},
        {"uuid": "c", "content": "infrastructure tuning notes"},
    ]
    out = rerank("navi pricing decision", candidates)

    assert out[0]["uuid"] == "b", f"expected 'b' first, got order: {[c['uuid'] for c in out]}"
    assert "rerank_score" in out[0]
    assert out[0]["rerank_score"] > out[-1]["rerank_score"]


def test_rerank_preserves_original_dict_data(fake_model):
    """Reranker must not mutate or drop existing fields on the candidates."""
    candidates = [
        {"uuid": "x", "content": "alpha beta", "group_id": "navi", "score": 0.5, "custom": [1, 2]},
    ]
    out = rerank("alpha", candidates)

    assert out[0]["uuid"] == "x"
    assert out[0]["group_id"] == "navi"
    assert out[0]["score"] == 0.5
    assert out[0]["custom"] == [1, 2]
    assert "rerank_score" in out[0]


def test_rerank_does_not_mutate_input(fake_model):
    """Returned dicts must be copies — caller's list and dicts unchanged."""
    candidates = [
        {"uuid": "a", "content": "navi pricing"},
        {"uuid": "b", "content": "auth setup"},
    ]
    snapshot = [dict(c) for c in candidates]
    rerank("navi", candidates)
    assert candidates == snapshot, "rerank() mutated the input dicts/list"


def test_rerank_falls_open_when_model_missing(monkeypatch):
    """When the model can't be loaded, return input unchanged with no rerank_score key."""
    monkeypatch.setattr(rerank_mod, "_get_model", lambda *_a, **_k: None)
    candidates = [{"uuid": "a", "content": "x"}]
    out = rerank("anything", candidates)

    assert out is candidates  # same list, unchanged
    assert "rerank_score" not in out[0]


def test_rerank_falls_open_when_inference_raises(monkeypatch):
    """Reranker exceptions must not break the retrieval pipeline."""
    bad_model = _FakeModel(raise_on_rank=True)
    monkeypatch.setattr(rerank_mod, "_get_model", lambda *_a, **_k: bad_model)
    candidates = [{"uuid": "a", "content": "x"}, {"uuid": "b", "content": "y"}]

    out = rerank("query", candidates)
    assert out is candidates  # untouched
    assert "rerank_score" not in out[0]


def test_rerank_respects_disable_env_flag(monkeypatch, fake_model):
    """JARVIS_RERANK=0 must short-circuit — no model call, no rerank_score."""
    monkeypatch.setenv("JARVIS_RERANK", "0")
    candidates = [
        {"uuid": "a", "content": "navi pricing"},
        {"uuid": "b", "content": "auth provider"},
    ]
    out = rerank("navi", candidates)

    assert out is candidates
    assert fake_model.calls == [], "model.rank() called even though JARVIS_RERANK=0"


def test_rerank_handles_empty_inputs(fake_model):
    """Empty candidates / empty query → return input unchanged, no model call."""
    assert rerank("query", []) == []
    assert rerank("", [{"uuid": "a", "content": "x"}]) == [{"uuid": "a", "content": "x"}]
    assert rerank("   ", [{"uuid": "a", "content": "x"}]) == [{"uuid": "a", "content": "x"}]
    assert fake_model.calls == []


def test_extract_text_prefers_content_then_compiled_truth_then_summary():
    """Field-priority order must be content → compiled_truth → name → summary → fact."""
    assert _extract_text({"content": "real", "summary": "fallback"}) == "real"
    assert _extract_text({"compiled_truth": "page-text", "summary": "fallback"}) == "page-text"
    assert _extract_text({"name": "n", "summary": "s"}) == "n"
    assert _extract_text({"summary": "s", "fact": "f"}) == "s"
    assert _extract_text({"fact": "f"}) == "f"
    assert _extract_text({}) == ""
    # Non-strings ignored
    assert _extract_text({"content": None, "summary": "real"}) == "real"
    # Whitespace-only treated as empty
    assert _extract_text({"content": "   ", "summary": "real"}) == "real"


def test_rerank_uses_compiled_truth_for_page_records(fake_model):
    """Page nodes have no 'content' field — fall through to compiled_truth."""
    candidates = [
        {"uuid": "ep-1", "content": "alpha bravo"},
        {"uuid": "page:x", "compiled_truth": "alpha bravo charlie delta"},
    ]
    out = rerank("alpha bravo", candidates)
    # page:x has the same overlap fraction (alpha+bravo = full overlap of query),
    # so it ties with ep-1. We only assert the model saw both texts.
    assert len(fake_model.calls) == 1
    _, docs = fake_model.calls[0]
    assert "alpha bravo" in docs[0]
    assert "alpha bravo charlie delta" in docs[1]
