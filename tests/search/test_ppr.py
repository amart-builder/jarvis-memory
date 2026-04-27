"""Personalized PageRank channel — multi-hop graph retrieval (B1).

Mocks Neo4j so tests run without a live DB. The PPR module must:
  * Extract proper-noun candidates from a query.
  * Match candidates to ``:Page`` slugs (case-insensitive).
  * Build a NetworkX subgraph from typed edges + EVIDENCED_BY.
  * Run ``nx.pagerank`` with personalization seeded at matched Pages.
  * Project mass back to Episode UUIDs only.
  * Fail open: empty result on any error or empty graph.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis_memory.search.ppr import (
    DEFAULT_DAMPING,
    DEFAULT_ITERATIONS,
    _build_subgraph,
    _extract_query_entities,
    _match_entities_to_pages,
    personalized_pagerank,
)


# ── Query entity extraction ────────────────────────────────────────────


def test_extract_query_entities_returns_lowercased_first_words():
    """Multi-word names ('Jacob Martin') reduce to first token for slug matching."""
    out = _extract_query_entities("decisions in Catalyst that affected Astack")
    # Catalyst and Astack are non-initial proper nouns — both should fire.
    assert "catalyst" in out
    assert "astack" in out


def test_extract_query_entities_empty_when_no_proper_nouns():
    out = _extract_query_entities("how does it work")
    assert out == []


def test_extract_query_entities_dedups():
    out = _extract_query_entities("Foundry decisions affected Foundry pricing")
    assert out.count("foundry") <= 1


# ── Entity → Page slug matching ────────────────────────────────────────


def _driver_with_slugs(slug_rows: list[dict]):
    """Build a driver mock whose Cypher returns the supplied slug rows."""

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def run(self, query, **_):
            return iter(slug_rows)

    driver = MagicMock()
    driver.session.return_value = _Session()
    return driver


def test_match_entities_to_pages_returns_slug_list():
    driver = _driver_with_slugs([{"slug": "foundry"}, {"slug": "catalyst"}])
    out = _match_entities_to_pages(driver, ["foundry", "catalyst"])
    assert out == ["foundry", "catalyst"]


def test_match_entities_to_pages_empty_input():
    driver = _driver_with_slugs([])
    assert _match_entities_to_pages(driver, []) == []


def test_match_entities_to_pages_handles_driver_error():
    """Cypher failures must not bubble up — fail open with []."""
    bad_session = MagicMock()
    bad_session.__enter__ = MagicMock(return_value=bad_session)
    bad_session.__exit__ = MagicMock(return_value=False)
    bad_session.run = MagicMock(side_effect=RuntimeError("boom"))
    driver = MagicMock()
    driver.session.return_value = bad_session
    assert _match_entities_to_pages(driver, ["x"]) == []


# ── Subgraph build + PPR end-to-end ────────────────────────────────────


class _FixtureSession:
    """Mock Neo4j session that dispatches by query content.

    Hand it a tiny canned graph: pages, episodes, edges. Runs every
    query in the build_subgraph pipeline.
    """

    def __init__(self, pages: list[str], episodes: list[str], edges: list[tuple]):
        # edges: list of (rel, a_id, a_kind, b_id, b_kind)
        self._pages = pages
        self._episodes = episodes
        self._edges = edges

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def run(self, query, **_):
        if "MATCH (p:Page)" in query and "RETURN p.slug AS slug" in query and ":Page)" in query.split("RETURN")[0]:
            return iter([{"slug": s} for s in self._pages])
        if "RETURN e.uuid" in query:
            return iter([{"uuid": u} for u in self._episodes])
        # Edge query: we extract the relationship name from the Cypher.
        if "RETURN coalesce(a.slug, a.uuid) AS a_id" in query:
            # Pull the relationship name out of "[:NAME]" for filtering.
            try:
                rel = query.split("[:", 1)[1].split("]", 1)[0]
            except IndexError:
                return iter([])
            rows = []
            for edge_rel, a_id, a_kind, b_id, b_kind in self._edges:
                if edge_rel != rel:
                    continue
                rows.append(
                    {
                        "a_id": a_id,
                        "a_kind": a_kind,
                        "b_id": b_id,
                        "b_kind": b_kind,
                    }
                )
            return iter(rows)
        return iter([])


def _make_fixture_driver(pages, episodes, edges, *, page_match_slugs=None):
    """Build a driver whose session() returns a FixtureSession but also
    handles the ``_match_entities_to_pages`` lookup."""
    fixture = _FixtureSession(pages, episodes, edges)
    match_slugs = list(page_match_slugs or [])

    class _RouterSession:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def run(self, query, **params):
            if "WHERE toLower(p.slug)" in query:
                # entity → page lookup
                return iter([{"slug": s} for s in match_slugs])
            return fixture.run(query, **params)

    driver = MagicMock()
    driver.session.return_value = _RouterSession()
    return driver


def test_build_subgraph_includes_pages_and_episodes():
    driver = _make_fixture_driver(
        pages=["foundry", "navi"],
        episodes=["ep-1", "ep-2"],
        edges=[("EVIDENCED_BY", "foundry", "page", "ep-1", "ep")],
    )
    G = _build_subgraph(driver)
    assert "page:foundry" in G
    assert "page:navi" in G
    assert "ep:ep-1" in G
    assert "ep:ep-2" in G
    assert G.has_edge("page:foundry", "ep:ep-1")


def test_build_subgraph_drops_unknown_endpoints():
    """Edges referencing nodes not in the loaded set are skipped — keeps
    PPR from leaking mass to phantoms when an edge is partially loaded."""
    driver = _make_fixture_driver(
        pages=["foundry"],
        episodes=["ep-1"],
        edges=[("FOUNDED", "foundry", "page", "missing-page", "page")],
    )
    G = _build_subgraph(driver)
    assert "page:foundry" in G
    assert "page:missing-page" not in G
    assert not G.has_edge("page:foundry", "page:missing-page")


def test_personalized_pagerank_returns_episodes_only():
    """PPR mass on Pages must NOT appear in the final result — only Episodes."""
    driver = _make_fixture_driver(
        pages=["foundry", "navi"],
        episodes=["ep-foundry", "ep-navi", "ep-shared"],
        edges=[
            ("EVIDENCED_BY", "foundry", "page", "ep-foundry", "ep"),
            ("EVIDENCED_BY", "foundry", "page", "ep-shared", "ep"),
            ("EVIDENCED_BY", "navi", "page", "ep-navi", "ep"),
            ("EVIDENCED_BY", "navi", "page", "ep-shared", "ep"),
            ("INVESTED_IN", "navi", "page", "foundry", "page"),  # Page-Page edge
        ],
        page_match_slugs=["foundry", "navi"],
    )

    out = personalized_pagerank(
        "decisions in Foundry that affected Navi",
        driver=driver,
    )
    uuids = [u for u, _ in out]
    # All three episodes should appear; none of the page slugs should.
    assert "ep-foundry" in uuids
    assert "ep-navi" in uuids
    assert "ep-shared" in uuids
    for uid in uuids:
        assert not uid.startswith("page:"), "PPR must project to Episode UUIDs only"


def test_personalized_pagerank_returns_empty_for_unknown_entities():
    """No matching Page slugs → empty (no PPR seed → nothing to compute)."""
    driver = _make_fixture_driver(
        pages=["foundry"],
        episodes=["ep-1"],
        edges=[],
        page_match_slugs=[],  # no matches found
    )
    out = personalized_pagerank("about UnknownThing and AlsoUnknown", driver=driver)
    assert out == []


def test_personalized_pagerank_returns_empty_for_no_proper_nouns():
    """Lowercase query → no entity candidates → empty result."""
    driver = _make_fixture_driver(pages=["foundry"], episodes=["ep-1"], edges=[])
    out = personalized_pagerank("how does it work", driver=driver)
    assert out == []


def test_personalized_pagerank_fails_open_on_driver_error():
    """A Neo4j explosion must not break retrieval — return [] silently."""
    bad_session = MagicMock()
    bad_session.__enter__ = MagicMock(return_value=bad_session)
    bad_session.__exit__ = MagicMock(return_value=False)
    bad_session.run = MagicMock(side_effect=RuntimeError("network gone"))
    driver = MagicMock()
    driver.session.return_value = bad_session

    out = personalized_pagerank("decisions in Foundry that affected Navi", driver=driver)
    assert out == []


def test_personalized_pagerank_empty_query():
    driver = _make_fixture_driver(pages=["foundry"], episodes=["ep-1"], edges=[])
    assert personalized_pagerank("", driver=driver) == []
    assert personalized_pagerank("   ", driver=driver) == []


def test_personalized_pagerank_no_driver():
    assert personalized_pagerank("Foundry stuff", driver=None) == []


# ── Hyperparameter sanity ──────────────────────────────────────────────


def test_default_hyperparameters_are_reasonable():
    assert 0 < DEFAULT_DAMPING < 1, "damping must be in (0, 1)"
    assert DEFAULT_ITERATIONS >= 10, "PPR needs enough iterations to converge"
