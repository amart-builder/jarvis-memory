"""Tests for the pure-Python reciprocal rank fusion combiner."""
from __future__ import annotations

import pytest

from jarvis_memory.search.rrf import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    """Core RRF semantics — inputs, outputs, ordering."""

    def test_basic_two_rankers_fused(self):
        """Docs present in both rankers score higher than singletons."""
        r1 = ["a", "b", "c"]
        r2 = ["b", "a", "d"]
        fused = reciprocal_rank_fusion([r1, r2], k=60)
        # Every non-empty input doc appears in output exactly once.
        ids = [doc_id for doc_id, _ in fused]
        assert set(ids) == {"a", "b", "c", "d"}
        assert len(ids) == len(set(ids))
        # a and b are in both rankers — they must come above c (only r1)
        # and d (only r2).
        scores = dict(fused)
        assert scores["a"] > scores["c"]
        assert scores["a"] > scores["d"]
        assert scores["b"] > scores["c"]
        assert scores["b"] > scores["d"]

    def test_empty_inputs_returns_empty(self):
        """No rankers and empty rankers both yield empty results."""
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[]], k=60) == []
        assert reciprocal_rank_fusion([[], [], []], k=60) == []

    def test_single_ranker_passthrough_ordering(self):
        """With one ranker, ordering matches input ordering."""
        fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
        assert [doc_id for doc_id, _ in fused] == ["x", "y", "z"]
        # And scores are strictly decreasing (1/(k+1) > 1/(k+2) > ...).
        scores = [s for _, s in fused]
        assert scores[0] > scores[1] > scores[2]

    def test_ranker_dedupe_within_one_list(self):
        """Duplicate ids inside one ranker are counted only once."""
        fused = reciprocal_rank_fusion([["a", "a", "a", "b"]], k=60)
        ids = [doc_id for doc_id, _ in fused]
        # a should appear once and get rank-1 credit only.
        assert ids == ["a", "b"]
        # The score for "a" is 1/(60+1); for "b" it is 1/(60+2) —
        # so a > b despite the dupes.
        scores = dict(fused)
        assert scores["a"] > scores["b"]
        # Confirm no double-counting: score for "a" equals one-hit-at-rank-1.
        assert scores["a"] == pytest.approx(1.0 / 61)

    def test_tie_breaks_alphabetically(self):
        """Docs with identical RRF score come back in deterministic order."""
        # Two single-doc rankers, each putting one doc at rank 1.
        fused = reciprocal_rank_fusion([["m"], ["z"], ["a"]], k=60)
        # All three have identical scores (1/61). Alphabetic tie-break.
        scores = dict(fused)
        assert scores["m"] == scores["z"] == scores["a"]
        assert [doc_id for doc_id, _ in fused] == ["a", "m", "z"]

    def test_stability_across_ranker_order(self):
        """Same rankers passed in different order yield same fused set/scores."""
        r1 = ["a", "b", "c"]
        r2 = ["c", "b", "a"]
        r3 = ["d", "a"]
        fused_abc = reciprocal_rank_fusion([r1, r2, r3], k=60)
        fused_cba = reciprocal_rank_fusion([r3, r2, r1], k=60)
        assert dict(fused_abc) == dict(fused_cba)

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=0)
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-1)

    def test_none_entries_ignored(self):
        """None or empty-string doc ids inside a ranker are skipped."""
        fused = reciprocal_rank_fusion([[None, "a", "", "b"]], k=60)
        ids = [doc_id for doc_id, _ in fused]
        assert ids == ["a", "b"]

    def test_none_ranker_tolerated(self):
        """An entire ``None`` ranker in the outer list is treated as empty."""
        fused = reciprocal_rank_fusion([None, ["a", "b"]], k=60)  # type: ignore[list-item]
        assert [doc_id for doc_id, _ in fused] == ["a", "b"]

    def test_lower_k_weights_top_rank_more(self):
        """A smaller k makes the top-of-list position more decisive."""
        r_a_first = ["a", "b"]
        r_b_first = ["b", "a"]
        # With very small k, disagreements at rank 1 dominate.
        fused_small = dict(reciprocal_rank_fusion([r_a_first, r_b_first], k=1))
        # Both docs still tie because they each get one rank-1 + one rank-2.
        assert fused_small["a"] == fused_small["b"]
        # But with an asymmetric setup, small-k amplifies the rank-1 voter.
        fused_small = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=1))
        fused_large = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=1000))
        # In both, a > b; the *ratio* shrinks as k grows.
        assert fused_small["a"] > fused_small["b"]
        assert fused_large["a"] > fused_large["b"]
        assert (fused_small["a"] / fused_small["b"]) > (
            fused_large["a"] / fused_large["b"]
        )
