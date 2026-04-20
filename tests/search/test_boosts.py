"""Tests for the post-RRF compiled-truth + backlink boosts."""
from __future__ import annotations

import math

import pytest

from jarvis_memory.pages import Page
from jarvis_memory.search.boosts import (
    BoostConfig,
    apply_boosts,
    backlink_boost,
    compiled_truth_boost,
)


# ── compiled_truth_boost ────────────────────────────────────────────────


class TestCompiledTruthBoost:
    def test_rich_truth_multiplies(self):
        page = Page(
            slug="foundry",
            domain="company",
            compiled_truth="Foundry is an AI infrastructure company focused on ...",
            created_at="",
            updated_at="",
        )
        out = compiled_truth_boost("foundry", 1.0, {"foundry": page})
        assert out == pytest.approx(1.2)

    def test_short_truth_does_not_boost(self):
        page = Page(
            slug="stub",
            domain="concept",
            compiled_truth="TBD",
            created_at="",
            updated_at="",
        )
        out = compiled_truth_boost("stub", 1.0, {"stub": page})
        assert out == 1.0

    def test_missing_lookup_is_noop(self):
        assert compiled_truth_boost("any", 0.5, None) == 0.5
        assert compiled_truth_boost("any", 0.5, {}) == 0.5

    def test_string_compiled_truth_also_works(self):
        """Accept a raw string in the lookup — convenience for callers."""
        lookup = {"page-a": "x" * 40}  # long enough
        out = compiled_truth_boost("page-a", 2.0, lookup)
        assert out == pytest.approx(2.4)


# ── backlink_boost ──────────────────────────────────────────────────────


class TestBacklinkBoost:
    def test_higher_degree_more_boost(self):
        low = backlink_boost("a", 1.0, {"a": 1})
        high = backlink_boost("a", 1.0, {"a": 50})
        assert high > low
        # Formula: base + log(1+d) * 0.1.
        assert low == pytest.approx(1.0 + math.log(2) * 0.1)
        assert high == pytest.approx(1.0 + math.log(51) * 0.1)

    def test_zero_or_missing_degree_is_noop(self):
        assert backlink_boost("a", 1.0, {"a": 0}) == 1.0
        assert backlink_boost("a", 1.0, {"b": 5}) == 1.0
        assert backlink_boost("a", 1.0, None) == 1.0
        assert backlink_boost("a", 1.0, {}) == 1.0


# ── apply_boosts end-to-end ─────────────────────────────────────────────


class TestApplyBoosts:
    def test_resorts_after_boost(self):
        """A lower-ranked doc with rich page + high degree can overtake."""
        fused = [("a", 1.0), ("b", 0.9)]
        # "b" has a rich Page with high degree; it should move above "a".
        pages = {
            "b": Page(
                slug="b",
                domain="company",
                compiled_truth="x" * 100,
                created_at="",
                updated_at="",
            )
        }
        degrees = {"b": 20}
        out = apply_boosts(fused, page_lookup=pages, in_degree_lookup=degrees)
        assert [doc_id for doc_id, _ in out][0] == "b"

    def test_preserves_order_without_any_boost_signals(self):
        """With no pages or degrees, ``apply_boosts`` returns the same ranking."""
        fused = [("a", 0.9), ("b", 0.5), ("c", 0.3)]
        out = apply_boosts(fused, page_lookup=None, in_degree_lookup=None)
        assert out == fused

    def test_empty_fused_returns_empty(self):
        assert apply_boosts([]) == []

    def test_alphabetic_tiebreak_on_equal_boosted_scores(self):
        fused = [("zeta", 1.0), ("alpha", 1.0)]
        out = apply_boosts(fused)
        # No boosts applied → identical scores → alphabetic order wins.
        assert [doc_id for doc_id, _ in out] == ["alpha", "zeta"]

    def test_custom_config_respected(self):
        pages = {
            "a": Page(
                slug="a",
                domain="topic",
                compiled_truth="x" * 100,
                created_at="",
                updated_at="",
            )
        }
        cfg = BoostConfig(compiled_truth_factor=2.0, backlink_weight=0.5)
        out = apply_boosts([("a", 1.0)], page_lookup=pages, config=cfg)
        assert out[0][1] == pytest.approx(2.0)  # 1.0 * 2.0
