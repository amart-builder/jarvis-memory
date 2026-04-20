"""Tests for jarvis_memory/orphans.py — graph-orphan detection.

Live Neo4j tests use the isolated :RunTwoTestPage label so they never
touch production data.
"""
from __future__ import annotations

import socket
from contextlib import contextmanager

import pytest

from jarvis_memory.config import NEO4J_URI
from jarvis_memory.orphans import find_orphans, main as orphans_main


# ── Unit tests against fake driver ───────────────────────────────────


class _FakeResult:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class _FakeSession:
    def __init__(self, records, calls):
        self.records = records
        self.calls = calls

    def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult(self.records)


class _FakeDriver:
    def __init__(self, records):
        self.records = records
        self.calls: list = []

    @contextmanager
    def session(self):
        yield _FakeSession(self.records, self.calls)


class TestFindOrphansUnit:
    def test_returns_empty_when_no_pages(self):
        driver = _FakeDriver(records=[])
        grouped = find_orphans(driver=driver)
        assert grouped == {}

    def test_groups_by_domain(self):
        records = [
            {"p": {"slug": "a", "domain": "person", "compiled_truth": "", "created_at": "", "updated_at": ""}},
            {"p": {"slug": "b", "domain": "person", "compiled_truth": "", "created_at": "", "updated_at": ""}},
            {"p": {"slug": "c", "domain": "company", "compiled_truth": "", "created_at": "", "updated_at": ""}},
        ]
        driver = _FakeDriver(records=records)
        grouped = find_orphans(driver=driver)
        assert set(grouped.keys()) == {"person", "company"}
        assert len(grouped["person"]) == 2
        assert len(grouped["company"]) == 1

    def test_domain_filter_passed_to_cypher(self):
        records = [
            {"p": {"slug": "a", "domain": "person", "compiled_truth": "", "created_at": "", "updated_at": ""}},
        ]
        driver = _FakeDriver(records=records)
        find_orphans(domain="person", driver=driver)
        assert driver.calls, "expected Cypher to be issued"
        q, params = driver.calls[0]
        assert "p.domain = $domain" in q
        assert params["domain"] == "person"

    def test_query_excludes_evidenced_by_in_orphan_definition(self):
        """A Page with an inbound EVIDENCED_BY edge should still be counted
        as orphan if it has no typed inbound edges."""
        driver = _FakeDriver(records=[])
        find_orphans(driver=driver)
        q = driver.calls[0][0]
        # The Cypher must use the typed-edge alternation, NOT include EVIDENCED_BY
        assert "EVIDENCED_BY" not in q.split("NOT EXISTS")[1] if "NOT EXISTS" in q else True
        # Typed edges should appear in the NOT EXISTS clause
        assert "WORKS_AT" in q
        assert "MENTIONS" in q


# ── Live Neo4j integration tests (isolated namespace) ────────────────


def _neo4j_reachable() -> bool:
    try:
        rest = NEO4J_URI.split("://", 1)[1]
        host, _, port_str = rest.partition(":")
        port = int(port_str.split("/")[0]) if port_str else 7687
    except Exception:
        return False
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


NEO4J_LIVE = _neo4j_reachable()
neo4j_required = pytest.mark.skipif(
    not NEO4J_LIVE, reason="Neo4j unreachable"
)

TEST_LABEL = "RunTwoTestPage"


@pytest.fixture
def live_driver():
    from neo4j import GraphDatabase
    from jarvis_memory.config import NEO4J_USER, NEO4J_PASSWORD

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    try:
        with driver.session() as db:
            db.run(f"MATCH (p:{TEST_LABEL}) DETACH DELETE p")
    except Exception:  # pragma: no cover
        pass
    driver.close()


@neo4j_required
class TestFindOrphansLive:
    def test_page_with_no_inbound_is_orphan(self, live_driver):
        from jarvis_memory.pages import put_page

        put_page("live-orphan-a", "person", driver=live_driver, label=TEST_LABEL)
        grouped = find_orphans(driver=live_driver, label=TEST_LABEL)
        slugs = {p.slug for pages in grouped.values() for p in pages}
        assert "live-orphan-a" in slugs

    def test_page_with_typed_inbound_not_orphan(self, live_driver):
        """Create A → WORKS_AT → B; B is not orphan, A is (has no inbound)."""
        from jarvis_memory.pages import put_page

        put_page("live-person-a", "person", driver=live_driver, label=TEST_LABEL)
        put_page("live-company-b", "company", driver=live_driver, label=TEST_LABEL)
        with live_driver.session() as db:
            db.run(
                f"""
                MATCH (a:{TEST_LABEL} {{slug: 'live-person-a'}})
                MATCH (b:{TEST_LABEL} {{slug: 'live-company-b'}})
                MERGE (a)-[:WORKS_AT {{confidence: 0.9}}]->(b)
                """
            )
        grouped = find_orphans(driver=live_driver, label=TEST_LABEL)
        slugs = {p.slug for pages in grouped.values() for p in pages}
        assert "live-company-b" not in slugs
        # A has no inbound typed edge, so it IS orphan
        assert "live-person-a" in slugs

    def test_evidenced_by_alone_still_orphan(self, live_driver):
        """A Page whose only inbound edge is EVIDENCED_BY is still orphan."""
        from jarvis_memory.pages import put_page

        put_page("live-timeline-only", "concept", driver=live_driver, label=TEST_LABEL)
        # Manually create a fake Episode + EVIDENCED_BY edge
        with live_driver.session() as db:
            db.run(
                f"""
                MERGE (e:Episode {{uuid: 'fake-ep-uuid-for-orphans-test'}})
                WITH e
                MATCH (p:{TEST_LABEL} {{slug: 'live-timeline-only'}})
                MERGE (p)-[:EVIDENCED_BY {{at: datetime()}}]->(e)
                """
            )
            # cleanup the fake episode at teardown
            pass
        grouped = find_orphans(driver=live_driver, label=TEST_LABEL)
        slugs = {p.slug for pages in grouped.values() for p in pages}
        assert "live-timeline-only" in slugs
        # Clean up the fake episode
        with live_driver.session() as db:
            db.run("MATCH (e:Episode {uuid: 'fake-ep-uuid-for-orphans-test'}) DETACH DELETE e")

    def test_domain_filter_limits_results(self, live_driver):
        from jarvis_memory.pages import put_page

        put_page("live-filter-a", "person", driver=live_driver, label=TEST_LABEL)
        put_page("live-filter-b", "company", driver=live_driver, label=TEST_LABEL)
        grouped = find_orphans(domain="person", driver=live_driver, label=TEST_LABEL)
        # Only person bucket
        assert set(grouped.keys()) <= {"person"}
        slugs = {p.slug for pages in grouped.values() for p in pages}
        assert "live-filter-a" in slugs
        assert "live-filter-b" not in slugs
