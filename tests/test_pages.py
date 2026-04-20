"""Tests for jarvis_memory/pages.py — Page CRUD + timeline.

Unit tests run against a fake driver (no Neo4j required) and assert the
Cypher we dispatch + the in-memory Page dataclass shape. Integration
tests use the isolated ``:RunTwoTestPage`` label so they don't collide
with production data; skipped when Neo4j is unreachable.
"""
from __future__ import annotations

import pytest
import socket
from contextlib import contextmanager
from unittest.mock import MagicMock

from jarvis_memory.config import NEO4J_URI
from jarvis_memory.pages import (
    Page,
    append_timeline_entry,
    count_pages,
    get_page,
    is_valid_slug,
    list_pages,
    put_page,
    slugify,
    COMPILED_TRUTH_MAX_CHARS,
)


# ── Slug helpers (pure, no driver) ───────────────────────────────────


class TestSlugify:
    def test_simple(self):
        assert slugify("Foundry") == "foundry"

    def test_spaces(self):
        assert slugify("Foundry Inc.") == "foundry-inc"

    def test_punctuation(self):
        assert slugify("X Æ A-12") == "x-a-12"

    def test_empty(self):
        assert slugify("") == ""

    def test_trims_leading_trailing_hyphens(self):
        assert slugify("  Alice Cooper  ") == "alice-cooper"

    def test_clamps_to_80_chars(self):
        long = "x" * 200
        assert len(slugify(long)) <= 80


class TestIsValidSlug:
    def test_valid(self):
        assert is_valid_slug("foundry") is True
        assert is_valid_slug("alice-cooper") is True
        assert is_valid_slug("x-ae-a-12") is True

    def test_empty(self):
        assert is_valid_slug("") is False

    def test_none(self):
        assert is_valid_slug(None) is False  # type: ignore

    def test_starts_with_hyphen(self):
        assert is_valid_slug("-foo") is False

    def test_contains_upper(self):
        assert is_valid_slug("Foundry") is False

    def test_contains_space(self):
        assert is_valid_slug("foo bar") is False

    def test_too_long(self):
        assert is_valid_slug("a" * 81) is False


# ── Page dataclass ───────────────────────────────────────────────────


class TestPageDataclass:
    def test_from_record_full(self):
        page = Page.from_record(
            {
                "slug": "foundry",
                "domain": "company",
                "compiled_truth": "summary",
                "created_at": "2026-04-20T00:00:00+00:00",
                "updated_at": "2026-04-20T00:00:00+00:00",
            }
        )
        assert page.slug == "foundry"
        assert page.domain == "company"

    def test_from_record_partial_defaults_empty(self):
        page = Page.from_record({"slug": "alice"})
        assert page.slug == "alice"
        assert page.domain == ""
        assert page.compiled_truth == ""

    def test_to_dict_roundtrip(self):
        d = {
            "slug": "foundry",
            "domain": "company",
            "compiled_truth": "x",
            "created_at": "t",
            "updated_at": "t",
        }
        assert Page.from_record(d).to_dict() == d


# ── Fake driver for unit tests ───────────────────────────────────────


class _FakeResult:
    def __init__(self, record=None):
        self._record = record

    def single(self):
        return self._record

    def __iter__(self):
        # Treat result as iterable of records; for list_pages it returns rows
        if self._record is None:
            return iter([])
        if isinstance(self._record, list):
            return iter(self._record)
        return iter([self._record])


class _FakeSession:
    """Mimics a neo4j session; every .run logs the query + params."""

    def __init__(self, calls: list[tuple[str, dict]], record_provider=None):
        self.calls = calls
        self._record_provider = record_provider or (lambda q, p: None)

    def run(self, query: str, **params):
        self.calls.append((query.strip(), params))
        return _FakeResult(self._record_provider(query, params))


class _FakeDriver:
    def __init__(self, record_provider=None):
        self.calls: list[tuple[str, dict]] = []
        self._record_provider = record_provider

    @contextmanager
    def session(self):
        yield _FakeSession(self.calls, self._record_provider)


# ── Unit tests — put_page / get_page / append_timeline_entry ─────────


class TestPutPage:
    def test_upserts_with_merge(self):
        driver = _FakeDriver(
            record_provider=lambda q, p: {
                "p": {
                    "slug": p["slug"],
                    "domain": p["domain"],
                    "compiled_truth": p.get("compiled_truth", ""),
                    "created_at": "now",
                    "updated_at": "now",
                }
            }
        )
        page = put_page("foundry", "company", driver=driver)
        assert page is not None
        assert page.slug == "foundry"
        assert page.domain == "company"
        # MERGE semantics in the issued query
        assert any("MERGE" in q for q, _ in driver.calls)

    def test_rejects_invalid_slug(self):
        driver = _FakeDriver()
        assert put_page("Foundry Inc.", "company", driver=driver) is None
        assert driver.calls == [], "invalid slug should not hit the driver"

    def test_clamps_compiled_truth(self):
        driver = _FakeDriver(
            record_provider=lambda q, p: {
                "p": {
                    "slug": p["slug"],
                    "domain": p["domain"],
                    "compiled_truth": p["compiled_truth"],
                    "created_at": "now",
                    "updated_at": "now",
                }
            }
        )
        long_truth = "x" * (COMPILED_TRUTH_MAX_CHARS + 100)
        page = put_page("alice", "person", compiled_truth=long_truth, driver=driver)
        assert page is not None
        assert len(page.compiled_truth) == COMPILED_TRUTH_MAX_CHARS

    def test_none_truth_preserves_existing(self):
        """With compiled_truth=None the query must not overwrite on match."""
        driver = _FakeDriver(
            record_provider=lambda q, p: {
                "p": {
                    "slug": p["slug"],
                    "domain": p["domain"],
                    "compiled_truth": "existing",
                    "created_at": "now",
                    "updated_at": "now",
                }
            }
        )
        page = put_page("bob", "person", compiled_truth=None, driver=driver)
        assert page is not None
        # Query should not contain a SET on compiled_truth for ON MATCH
        q = driver.calls[0][0]
        assert "ON MATCH SET" in q
        # When compiled_truth is None, we only touch domain coalesce + updated_at
        assert "p.compiled_truth = $compiled_truth" not in q


class TestGetPage:
    def test_returns_none_when_missing(self):
        driver = _FakeDriver(record_provider=lambda q, p: None)
        assert get_page("ghost", driver=driver) is None

    def test_returns_page_when_found(self):
        driver = _FakeDriver(
            record_provider=lambda q, p: {
                "p": {
                    "slug": "foundry",
                    "domain": "company",
                    "compiled_truth": "x",
                    "created_at": "now",
                    "updated_at": "now",
                }
            }
        )
        page = get_page("foundry", driver=driver)
        assert page is not None
        assert page.slug == "foundry"

    def test_empty_slug_returns_none(self):
        assert get_page("", driver=_FakeDriver()) is None


class TestAppendTimelineEntry:
    def test_success_returns_true(self):
        driver = _FakeDriver(
            record_provider=lambda q, p: {"r": {"at": "now", "summary": p.get("summary", "")}}
        )
        ok = append_timeline_entry(
            "foundry",
            "episode-uuid",
            at="2026-04-20T00:00:00+00:00",
            summary="meeting notes",
            driver=driver,
        )
        assert ok is True
        q = driver.calls[0][0]
        assert "EVIDENCED_BY" in q
        assert "MERGE" in q

    def test_empty_slug_returns_false(self):
        assert append_timeline_entry("", "uuid", driver=_FakeDriver()) is False

    def test_empty_episode_returns_false(self):
        assert append_timeline_entry("foundry", "", driver=_FakeDriver()) is False


class TestListPages:
    def test_list_with_domain_filter(self):
        driver = _FakeDriver(
            record_provider=lambda q, p: [
                {
                    "p": {
                        "slug": f"slug-{i}",
                        "domain": p.get("domain", "company"),
                        "compiled_truth": "",
                        "created_at": "",
                        "updated_at": "",
                    }
                }
                for i in range(3)
            ]
        )
        pages = list_pages(domain="company", driver=driver)
        assert len(pages) == 3
        assert all(isinstance(p, Page) for p in pages)

    def test_list_without_filter(self):
        driver = _FakeDriver(record_provider=lambda q, p: [])
        pages = list_pages(driver=driver)
        assert pages == []


class TestCountPages:
    def test_count_returns_int(self):
        driver = _FakeDriver(record_provider=lambda q, p: {"n": 42})
        assert count_pages(driver=driver) == 42

    def test_count_zero_when_empty(self):
        driver = _FakeDriver(record_provider=lambda q, p: {"n": 0})
        assert count_pages(driver=driver) == 0


# ── Integration tests against live Neo4j (isolated namespace) ────────


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
    not NEO4J_LIVE,
    reason="Neo4j unreachable — skipping live Page tests",
)

TEST_LABEL = "RunTwoTestPage"


@pytest.fixture
def live_driver():
    """Neo4j driver fixture — tears down all :RunTwoTestPage nodes after each test."""
    from neo4j import GraphDatabase
    from jarvis_memory.config import NEO4J_USER, NEO4J_PASSWORD

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    yield driver
    # Teardown: wipe anything we created.
    try:
        with driver.session() as db:
            db.run(f"MATCH (p:{TEST_LABEL}) DETACH DELETE p")
    except Exception:  # pragma: no cover — best-effort cleanup
        pass
    driver.close()


@neo4j_required
class TestLivePages:
    def test_put_then_get(self, live_driver):
        page = put_page("live-foundry", "company", driver=live_driver, label=TEST_LABEL)
        assert page is not None
        assert page.slug == "live-foundry"
        fetched = get_page("live-foundry", driver=live_driver, label=TEST_LABEL)
        assert fetched is not None
        assert fetched.domain == "company"

    def test_put_updates_compiled_truth(self, live_driver):
        put_page("live-alice", "person", "v1", driver=live_driver, label=TEST_LABEL)
        put_page("live-alice", "person", "v2", driver=live_driver, label=TEST_LABEL)
        page = get_page("live-alice", driver=live_driver, label=TEST_LABEL)
        assert page is not None
        assert page.compiled_truth == "v2"

    def test_list_pages_with_domain_filter(self, live_driver):
        put_page("live-co-1", "company", driver=live_driver, label=TEST_LABEL)
        put_page("live-co-2", "company", driver=live_driver, label=TEST_LABEL)
        put_page("live-p-1", "person", driver=live_driver, label=TEST_LABEL)
        companies = list_pages(domain="company", driver=live_driver, label=TEST_LABEL)
        persons = list_pages(domain="person", driver=live_driver, label=TEST_LABEL)
        assert len(companies) == 2
        assert len(persons) == 1
