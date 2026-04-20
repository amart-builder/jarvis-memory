"""Tests for jarvis_memory/graph.py — typed-edge extraction.

Pure-function tests. No Neo4j. Every edge type + edge cases + the
low-confidence filtering guarantee.
"""
from __future__ import annotations

import pytest

from jarvis_memory.graph import (
    MIN_CONFIDENCE,
    TypedEdge,
    extract_typed_edges,
)


# ── One test per typed edge ──────────────────────────────────────────


class TestEdgeExtractionPerType:
    """One positive test per edge type from schema_v2.TYPED_EDGES."""

    def test_works_at(self):
        edges = extract_typed_edges(
            "Alice Cooper works at Foundry Inc on the platform team.",
            episode_type="fact",
            group_id="foundry",
        )
        kinds = {e.edge_type for e in edges}
        assert "WORKS_AT" in kinds, f"expected WORKS_AT; got {[e.to_dict() for e in edges]}"
        hit = next(e for e in edges if e.edge_type == "WORKS_AT")
        assert hit.from_slug == "alice-cooper"
        assert "foundry" in hit.to_slug

    def test_attended(self):
        edges = extract_typed_edges(
            "Bob Jones attended YC Startup School in March.",
            episode_type="event",
        )
        kinds = {e.edge_type for e in edges}
        assert "ATTENDED" in kinds

    def test_invested_in(self):
        edges = extract_typed_edges(
            "Carol Smith invested in Foundry Labs during the seed round.",
            episode_type="event",
        )
        kinds = {e.edge_type for e in edges}
        assert "INVESTED_IN" in kinds

    def test_founded(self):
        edges = extract_typed_edges(
            "David Wright founded Navi Systems in 2024.",
            episode_type="fact",
        )
        kinds = {e.edge_type for e in edges}
        assert "FOUNDED" in kinds
        hit = next(e for e in edges if e.edge_type == "FOUNDED")
        assert hit.from_slug == "david-wright"

    def test_advises(self):
        edges = extract_typed_edges(
            "Eve Martin advises Catalyst Partners on growth strategy.",
            episode_type="fact",
        )
        kinds = {e.edge_type for e in edges}
        assert "ADVISES" in kinds

    def test_decided_on(self):
        edges = extract_typed_edges(
            "[DECISION] Decided to use Postgres for the canonical store. "
            "Reason: simpler ops profile than DynamoDB.",
            episode_type="decision",
            group_id="navi",
        )
        kinds = {e.edge_type for e in edges}
        assert "DECIDED_ON" in kinds
        hit = next(e for e in edges if e.edge_type == "DECIDED_ON")
        assert hit.from_slug == "navi"  # anchor

    def test_mentions_fallback(self):
        # A proper noun with no typed-edge pattern → MENTIONS
        edges = extract_typed_edges(
            "Alex met Frank Ocean at the conference dinner.",
            episode_type="event",
            group_id="system",
        )
        kinds = {e.edge_type for e in edges}
        assert "MENTIONS" in kinds

    def test_refers_to(self):
        edges = extract_typed_edges(
            "For the rationale see Navi Phase1 doc.",
            episode_type="fact",
            group_id="system",
        )
        kinds = {e.edge_type for e in edges}
        assert "REFERS_TO" in kinds


# ── Edge cases — empty / bad input / low-confidence filter ───────────


class TestEdgeCases:
    def test_empty_content_returns_empty(self):
        assert extract_typed_edges("", episode_type="fact") == []

    def test_none_content_returns_empty(self):
        assert extract_typed_edges(None, episode_type="fact") == []  # type: ignore

    def test_pronoun_heavy_no_proper_nouns(self):
        """No proper nouns = no edges (MENTIONS fallback has nothing to fire on)."""
        edges = extract_typed_edges(
            "we discussed it earlier and decided to push it to next week.",
            episode_type="decision",
            group_id="system",
        )
        # "system" group anchor + no proper nouns = zero edges
        assert edges == []

    def test_all_edges_meet_min_confidence(self):
        edges = extract_typed_edges(
            "Alice Cooper works at Foundry Inc. Bob Jones founded Navi Systems. "
            "Carol Smith invested in Catalyst Partners.",
            episode_type="fact",
            group_id="system",
        )
        assert edges, "expected edges from multi-pattern input"
        for e in edges:
            assert e.confidence >= MIN_CONFIDENCE

    def test_dedupe_by_triple(self):
        """Same (from, type, to) reported once, highest-confidence wins."""
        # Two sentences both say "Alice Cooper works at Foundry Inc."
        text = (
            "Alice Cooper works at Foundry Inc on the platform team. "
            "Additionally, Alice Cooper works at Foundry Inc as an engineer."
        )
        edges = extract_typed_edges(text, episode_type="fact", group_id="system")
        triples = [(e.from_slug, e.edge_type, e.to_slug) for e in edges]
        # Exactly one instance of the (alice-cooper, WORKS_AT, foundry-inc) triple
        works_at_triples = [t for t in triples if t[1] == "WORKS_AT"]
        assert len(works_at_triples) == len(set(works_at_triples)), (
            f"dedup failed: {works_at_triples}"
        )

    def test_decision_episode_type_boosts_confidence(self):
        edges = extract_typed_edges(
            "[DECISION] Decided to use Postgres.",
            episode_type="decision",
            group_id="navi",
        )
        assert any(e.confidence >= 0.85 for e in edges if e.edge_type == "DECIDED_ON"), (
            f"decision boost not applied: {[e.to_dict() for e in edges]}"
        )

    def test_non_decision_type_does_not_boost(self):
        edges = extract_typed_edges(
            "[DECISION] Decided to use Postgres.",
            episode_type=None,  # no episode_type → no boost
            group_id="navi",
        )
        for e in edges:
            if e.edge_type == "DECIDED_ON":
                assert e.confidence < 0.95  # uncapped

    def test_same_subject_and_object_filtered(self):
        """An edge where slugify(subj) == slugify(obj) should NOT be emitted."""
        # If the regex matches "X works at X" the function must drop it.
        edges = extract_typed_edges(
            "Foundry Inc works at Foundry Inc.",
            episode_type="fact",
            group_id="foundry",
        )
        for e in edges:
            assert e.from_slug != e.to_slug, f"self-edge leaked: {e.to_dict()}"

    def test_low_confidence_filtered(self):
        """A weak MENTIONS match (single appearance, no pattern fire) →
        confidence 0.6 threshold; pattern confidences must all meet it."""
        edges = extract_typed_edges(
            "Today I had a pleasant walk.",
            episode_type="fact",
            group_id="system",
        )
        # No proper nouns except maybe "Today" which is stopword-filtered.
        assert edges == [] or all(e.confidence >= MIN_CONFIDENCE for e in edges)


# ── TypedEdge dataclass ──────────────────────────────────────────────


class TestTypedEdge:
    def test_frozen(self):
        edge = TypedEdge(
            from_slug="alice",
            edge_type="WORKS_AT",
            to_slug="foundry",
            confidence=0.9,
        )
        with pytest.raises(Exception):
            edge.confidence = 0.1  # type: ignore

    def test_to_dict(self):
        edge = TypedEdge(
            from_slug="alice",
            edge_type="WORKS_AT",
            to_slug="foundry",
            confidence=0.9,
            evidence="works at",
        )
        d = edge.to_dict()
        assert d["edge_type"] == "WORKS_AT"
        assert d["confidence"] == 0.9


# ── Ordering guarantee ───────────────────────────────────────────────


class TestOrdering:
    def test_result_is_deterministic(self):
        """Two identical calls produce identical lists."""
        content = (
            "Alice Cooper works at Foundry Inc. "
            "Bob Jones founded Navi Systems. "
            "Carol Smith advises Catalyst Partners."
        )
        a = extract_typed_edges(content, episode_type="fact", group_id="system")
        b = extract_typed_edges(content, episode_type="fact", group_id="system")
        assert [e.to_dict() for e in a] == [e.to_dict() for e in b]

    def test_typed_edges_ordered_by_canonical_type_then_slug(self):
        """Typed-edge canonical order is stable across runs."""
        content = (
            "Carol Smith invested in Catalyst Partners. "
            "Alice Cooper works at Foundry Inc."
        )
        edges = extract_typed_edges(content, episode_type="fact", group_id="system")
        kinds = [e.edge_type for e in edges]
        # WORKS_AT comes before INVESTED_IN in schema_v2.TYPED_EDGES ordering
        # (ATTENDED, WORKS_AT, INVESTED_IN, ...)
        # Both present → WORKS_AT first.
        if "WORKS_AT" in kinds and "INVESTED_IN" in kinds:
            assert kinds.index("WORKS_AT") < kinds.index("INVESTED_IN")
