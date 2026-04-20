"""Tests for the composite scoring module."""
import math
from datetime import datetime, timezone, timedelta

from jarvis_memory.scoring import (
    composite_score,
    score_results,
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
