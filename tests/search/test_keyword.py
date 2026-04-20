"""Tests for the Neo4j keyword retriever — mocked driver + live-skippable integration."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from jarvis_memory.search.keyword import (
    Hit,
    _escape_for_lucene,
    _tokenize,
    keyword_search,
)


def _neo4j_reachable() -> bool:
    try:
        from jarvis_memory.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        from neo4j import GraphDatabase

        d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        d.verify_connectivity()
        d.close()
        return True
    except Exception:
        return False


NEO4J_LIVE = _neo4j_reachable()


class _FakeSession:
    """Minimal Neo4j session stub. Records queries and returns preset rows."""

    def __init__(self, scripts):
        # scripts: list of (predicate, rows_or_exception) tuples. The
        # predicate receives the Cypher string; first match wins.
        self.scripts = scripts
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, cypher, **params):
        self.calls.append({"cypher": cypher, "params": params})
        for pred, payload in self.scripts:
            if pred(cypher):
                if isinstance(payload, Exception):
                    raise payload
                return _FakeResult(payload)
        # Default: empty iter.
        return _FakeResult([])


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeDriver:
    def __init__(self, scripts):
        self._scripts = scripts

    def session(self):
        return _FakeSession(self._scripts)


class TestTokenize:
    def test_short_tokens_dropped(self):
        assert _tokenize("a bb ccc dddd") == ["ccc", "dddd"]

    def test_lowercases(self):
        assert _tokenize("Foundry Atlas") == ["foundry", "atlas"]


class TestEscape:
    def test_strips_lucene_special_chars(self):
        assert "+" not in _escape_for_lucene("jarvis+memory")
        assert "[" not in _escape_for_lucene("foo[bar]")
        # Multiple specials → stripped to at most single spaces around.
        assert _escape_for_lucene("a:b/c") == "a b c"

    def test_empty_input(self):
        assert _escape_for_lucene("") == ""


class TestKeywordSearchUnits:
    """Behavioral unit tests using a fake driver."""

    def test_empty_query_returns_empty(self):
        assert keyword_search("", driver=_FakeDriver([])) == []
        assert keyword_search("   ", driver=_FakeDriver([])) == []

    def test_no_driver_returns_empty(self):
        assert keyword_search("jarvis memory", driver=None) == []

    def test_fulltext_happy_path_episode_label(self):
        """Fulltext index returns episode hits mapped to Hit objects."""
        fulltext_rows = [
            {"id": "uuid-1", "score": 4.2, "n": {"uuid": "uuid-1", "content": "..."}},
            {"id": "uuid-2", "score": 2.1, "n": {"uuid": "uuid-2", "content": "..."}},
        ]
        driver = _FakeDriver(
            [
                (lambda c: "db.index.fulltext.queryNodes" in c and "node.uuid" in c, fulltext_rows),
                # Page channel still gets called; give it an empty result.
                (lambda c: "node.slug AS id" in c, []),
            ]
        )

        hits = keyword_search("foundry memory", k=5, driver=driver)
        assert len(hits) == 2
        assert all(isinstance(h, Hit) for h in hits)
        # Preserve score-descending order (already sorted by Neo4j; our
        # final sort preserves it).
        assert hits[0].id == "uuid-1"
        assert hits[0].score == pytest.approx(4.2)
        assert hits[0].source == "keyword:episode"

    def test_contains_fallback_when_fulltext_missing(self):
        """When the fulltext index call raises, fall back to CONTAINS scoring."""
        fallback_rows = [
            {
                "id": "uuid-10",
                "content": "Foundry memory is the knowledge graph",
                "node": {"uuid": "uuid-10", "content": "Foundry memory ..."},
            },
            {
                "id": "uuid-11",
                "content": "Unrelated text with only memory",
                "node": {"uuid": "uuid-11", "content": "Unrelated ..."},
            },
        ]
        driver = _FakeDriver(
            [
                (
                    lambda c: "db.index.fulltext.queryNodes" in c and "node.uuid" in c,
                    RuntimeError("no such index"),
                ),
                (lambda c: "CONTAINS" in c and "LIMIT $k" in c, fallback_rows),
                # Pages channel: fulltext fails too → skipped.
                (lambda c: "node.slug AS id" in c, RuntimeError("no page index")),
            ]
        )
        hits = keyword_search("foundry memory", k=5, driver=driver)
        # Both rows should be kept (each has at least one token overlap);
        # uuid-10 has two overlaps ⇒ higher score.
        assert [h.id for h in hits] == ["uuid-10", "uuid-11"]
        assert hits[0].score > hits[1].score

    def test_namespace_param_scopes_label(self):
        """Passing namespace='TestEpisode' must make the query filter on that label."""
        captured: list[dict] = []

        class RecordingSession:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def run(self_inner, cypher, **params):
                captured.append({"cypher": cypher, "params": params})
                # Return an empty fake result.
                return _FakeResult([])

        class RecordingDriver:
            def session(self_inner):
                return RecordingSession()

        keyword_search(
            "foundry", k=3, namespace="TestEpisode", driver=RecordingDriver()
        )
        # Either the label must show up as a Cypher label literal or as
        # a bound parameter value — both are valid implementations.
        episode_calls = [
            c for c in captured if "node.uuid AS id" in c["cypher"] or "Episode" in c["cypher"]
        ]
        assert episode_calls, f"no episode query issued: {captured}"
        in_params = any(
            c.get("params", {}).get("label") == "TestEpisode" for c in episode_calls
        )
        in_cypher = any("TestEpisode" in c["cypher"] for c in episode_calls)
        assert in_params or in_cypher, (
            f"namespace='TestEpisode' did not reach Neo4j; saw: {episode_calls}"
        )

    def test_pages_can_be_excluded(self):
        """include_pages=False must not query the page fulltext index."""
        page_calls: list[str] = []

        class PageTrackingSession:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def run(self_inner, cypher, **params):
                if "node.slug AS id" in cypher:
                    page_calls.append(cypher)
                return _FakeResult([])

        class PageTrackingDriver:
            def session(self_inner):
                return PageTrackingSession()

        keyword_search(
            "foundry",
            k=3,
            driver=PageTrackingDriver(),
            include_pages=False,
        )
        assert page_calls == []

    def test_result_cap_respects_k(self):
        """When more hits return than k, the output is capped."""
        rows = [
            {"id": f"u{i}", "score": 10 - i, "n": {"uuid": f"u{i}"}}
            for i in range(8)
        ]
        driver = _FakeDriver(
            [
                (lambda c: "db.index.fulltext.queryNodes" in c and "node.uuid" in c, rows),
                (lambda c: "node.slug AS id" in c, []),
            ]
        )
        hits = keyword_search("foundry memory", k=3, driver=driver)
        assert len(hits) == 3
        # And they're the three best scores (10, 9, 8).
        assert [h.score for h in hits] == [10.0, 9.0, 8.0]


# ── Live Neo4j smoke test — skipped unless a live cluster is reachable ──


@pytest.mark.skipif(not NEO4J_LIVE, reason="Neo4j not reachable")
def test_live_keyword_search_on_empty_namespace():
    """Against a live cluster, an isolated label returns zero hits cleanly."""
    from jarvis_memory.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # Use a namespace that definitely doesn't exist — guaranteed zero.
        hits = keyword_search(
            "foundry memory atlas",
            k=3,
            namespace="RunThreeTestEpisode_NoSuchLabel",
            driver=drv,
            include_pages=False,
        )
        assert hits == []
    finally:
        drv.close()
