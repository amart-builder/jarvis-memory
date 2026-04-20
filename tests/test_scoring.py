"""Tests for the composite scoring module."""
import math
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from jarvis_memory.scoring import (
    composite_score,
    score_results,
    scored_search,
    _compute_recency,
    TYPE_BOOST,
    PERSISTENT_TYPES,
)


class TestComputeRecency:
    """Tests for recency decay calculation."""

    def test_none_returns_default(self):
        assert _compute_recency(None) == 0.5

    def test_invalid_string_returns_default(self):
        assert _compute_recency("not-a-date") == 0.5

    def test_recent_has_high_recency(self):
        now = datetime.now(timezone.utc).isoformat()
        recency = _compute_recency(now)
        assert recency > 0.99

    def test_old_has_low_recency(self):
        old = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        recency = _compute_recency(old)
        assert recency < 0.1

    def test_half_life_point(self):
        """At exactly half_life_days, recency should be ~0.5."""
        half_life = 90.0
        at_half = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        recency = _compute_recency(at_half, half_life_days=half_life)
        assert abs(recency - 0.5) < 0.05

    def test_datetime_object_input(self):
        now = datetime.now(timezone.utc)
        recency = _compute_recency(now)
        assert recency > 0.99

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime.now()
        recency = _compute_recency(naive)
        assert 0.0 <= recency <= 1.0


class TestCompositeScore:
    """Tests for the composite scoring function."""

    def test_perfect_score(self):
        """High similarity, recent, important decision = high score."""
        score = composite_score(
            semantic_similarity=1.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            importance=1.0,
            access_count=10,
            memory_type="decision",
        )
        assert score > 1.0  # access boost pushes above 1.0

    def test_minimum_score(self):
        """Zero similarity, old, low importance = near-zero score."""
        old = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
        score = composite_score(
            semantic_similarity=0.0,
            created_at=old,
            importance=0.0,
            access_count=0,
            memory_type="cancellation",
        )
        assert score < 0.1

    def test_persistent_type_decay_floor(self):
        """Decisions should never decay below 40% recency."""
        very_old = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
        score = composite_score(
            semantic_similarity=0.8,
            created_at=very_old,
            memory_type="decision",
        )
        # The recency component should be at least 0.4 × 0.30 = 0.12
        assert score > 0.5

    def test_non_persistent_type_decays_fully(self):
        """Events should decay without a floor."""
        very_old = (datetime.now(timezone.utc) - timedelta(days=1000)).isoformat()
        score_event = composite_score(
            semantic_similarity=0.5,
            created_at=very_old,
            memory_type="event",
        )
        score_decision = composite_score(
            semantic_similarity=0.5,
            created_at=very_old,
            memory_type="decision",
        )
        assert score_decision > score_event

    def test_access_boost_capped(self):
        """Access boost should cap at 1.5×."""
        base = composite_score(
            semantic_similarity=0.8,
            access_count=0,
            memory_type="fact",
        )
        boosted = composite_score(
            semantic_similarity=0.8,
            access_count=100,  # way above cap
            memory_type="fact",
        )
        assert abs(boosted / base - 1.5) < 0.01

    def test_all_type_boosts_defined(self):
        """Every type in TYPE_BOOST should produce a valid score."""
        for mem_type, boost in TYPE_BOOST.items():
            score = composite_score(
                semantic_similarity=0.7,
                memory_type=mem_type,
            )
            assert 0.0 <= score <= 1.5, f"Bad score for {mem_type}: {score}"


class TestScoreResults:
    """Tests for batch result re-scoring."""

    def test_sorts_descending(self):
        results = [
            {"score": 0.3, "metadata": {"type": "fact"}},
            {"score": 0.9, "metadata": {"type": "decision"}},
            {"score": 0.6, "metadata": {"type": "event"}},
        ]
        scored = score_results(results)
        scores = [r["composite_score"] for r in scored]
        assert scores == sorted(scores, reverse=True)

    def test_adds_composite_score_key(self):
        results = [{"score": 0.5, "metadata": {}}]
        scored = score_results(results)
        assert "composite_score" in scored[0]

    def test_empty_results(self):
        assert score_results([]) == []

    def test_flat_metadata(self):
        """Should work when metadata is flat (no nested dict)."""
        results = [{"score": 0.7, "type": "insight", "importance": 0.9}]
        scored = score_results(results)
        assert "composite_score" in scored[0]


# ── Run 3: scored_search internals ──────────────────────────────────────


class TestScoredSearchInternals:
    """Rewritten scored_search — external contract locked; internals testable.

    These tests drive ``scored_search`` with pure-Python substitutes for
    the retriever functions so we never need a real Neo4j/Chroma stack.
    """

    def _fake_vector(self, id_score_map: dict[str, float]):
        """Return a ``vector_search_fn`` that emits a ranked list from a map."""

        ranked = sorted(id_score_map.items(), key=lambda kv: -kv[1])

        def _fn(_q: str, n: int):
            return [
                {"id": uid, "uuid": uid, "similarity": sim, "metadata": {}}
                for uid, sim in ranked[:n]
            ]

        return _fn

    def test_empty_query_returns_empty(self):
        assert scored_search("") == []
        assert scored_search("   ") == []

    def test_rrf_is_used_internally(self, monkeypatch):
        """Monkeypatch the RRF combiner and assert scored_search calls it."""
        called = {"n": 0}

        import jarvis_memory.scoring as scoring_mod
        from jarvis_memory.search import rrf as rrf_mod

        original_rrf = rrf_mod.reciprocal_rank_fusion

        def _spy(rankings, k=60):
            called["n"] += 1
            return original_rrf(rankings, k=k)

        monkeypatch.setattr(rrf_mod, "reciprocal_rank_fusion", _spy)
        # Make sure scored_search uses the same name we patched.
        monkeypatch.setattr(
            scoring_mod, "_legacy_scored_search",
            lambda **_kw: [],  # if the search falls back to legacy, RRF wouldn't fire
        )

        vector = self._fake_vector({"a": 0.9, "b": 0.8})
        out = scored_search(
            "what is the architecture of jarvis",
            limit=5,
            vector_search_fn=vector,
            expand_fn=lambda q, n: [q],  # skip expansion fan-out
        )
        assert called["n"] >= 1
        # We asked for limit=5 and had two candidates; both should come back.
        ids = [r.get("uuid") or r.get("id") for r in out]
        assert set(ids) == {"a", "b"}

    def test_intent_routing_fires(self, monkeypatch):
        """Entity intents must trigger expand, temporal intents must not."""
        import jarvis_memory.scoring as scoring_mod

        expand_log: list[str] = []

        def _spy_expand(q: str, n: int):
            expand_log.append(q)
            return [q]

        vector = self._fake_vector({"a": 0.9})

        # Entity intent (proper noun + long enough query).
        expand_log.clear()
        scored_search(
            "tell me about Foundry Ventures overview and strategy",
            limit=3,
            vector_search_fn=vector,
            expand_fn=_spy_expand,
        )
        assert expand_log, "entity intent should have called expand_fn"

        # Temporal intent (date phrase).
        expand_log.clear()
        scored_search(
            "show notes from last week",
            limit=3,
            vector_search_fn=vector,
            expand_fn=_spy_expand,
        )
        assert not expand_log, (
            "temporal intent should NOT trigger expand_fn (costs + noise)"
        )

    def test_group_id_filter_preserved(self):
        """Hits with a different group_id must be filtered out."""
        vector = lambda q, n: [
            {"id": "a", "uuid": "a", "similarity": 0.9, "metadata": {"group_id": "alpha"}},
            {"id": "b", "uuid": "b", "similarity": 0.8, "metadata": {"group_id": "beta"}},
            {"id": "c", "uuid": "c", "similarity": 0.7, "metadata": {"group_id": "alpha"}},
        ]
        out = scored_search(
            "something",
            group_id="alpha",
            vector_search_fn=vector,
            expand_fn=lambda q, n: [q],
        )
        ids = {r.get("uuid") for r in out}
        assert ids == {"a", "c"}

    def test_limit_respected(self):
        vector = lambda q, n: [
            {"id": f"u{i}", "uuid": f"u{i}", "similarity": 1.0 - i * 0.01, "metadata": {}}
            for i in range(20)
        ]
        out = scored_search(
            "anything",
            limit=4,
            vector_search_fn=vector,
            expand_fn=lambda q, n: [q],
        )
        assert len(out) == 4

    def test_legacy_env_var_routes_to_composite(self, monkeypatch):
        """JARVIS_SEARCH_LEGACY=1 must route through the Run 1 composite path."""
        monkeypatch.setenv("JARVIS_SEARCH_LEGACY", "1")

        vector = self._fake_vector({"a": 0.9, "b": 0.5})
        out = scored_search(
            "anything at all here",
            limit=5,
            vector_search_fn=vector,
            expand_fn=lambda q, n: [q],
        )
        # Legacy path must still produce results with composite_score
        # attached (that's what the Run 1 baseline exposes).
        assert out
        assert all("composite_score" in r for r in out)

    def test_signature_accepts_all_documented_params(self):
        """Accept every locked kw arg without raising; result may be empty."""
        # Minimal no-op retrievers so we don't need a real driver.
        out = scored_search(
            "q",
            group_id="g",
            room="r",
            hall="h",
            memory_type="fact",
            as_of="2025-01-01",
            limit=3,
            vector_search_fn=lambda q, n: [],
            expand_fn=lambda q, n: [q],
        )
        assert isinstance(out, list)
