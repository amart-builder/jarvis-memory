"""Tests for jarvis_memory/doctor.py — entity-layer health checks.

Each of the four checks (schema_v2_present, page_completeness,
edge_validity, orphan_count_reasonable) has at least one unit test and
(where feasible) a live integration test.
"""
from __future__ import annotations

import socket
from contextlib import contextmanager

import pytest

from jarvis_memory.config import NEO4J_URI
from jarvis_memory.doctor import (
    FAIL,
    PASS,
    WARN,
    check_edge_validity,
    check_orphan_count_reasonable,
    check_page_completeness,
    check_schema_v2_present,
    run_health_checks,
)


# ── Fake driver helpers ──────────────────────────────────────────────


class _FakeResult:
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self):
        return self.records[0] if self.records else None


class _FakeSession:
    def __init__(self, dispatch):
        self.dispatch = dispatch
        self.last_query = ""

    def run(self, query, **params):
        self.last_query = query
        return self.dispatch(query, params)


class _FakeDriver:
    def __init__(self, dispatch):
        self.dispatch = dispatch

    @contextmanager
    def session(self):
        yield _FakeSession(self.dispatch)


# ── schema_v2_present ────────────────────────────────────────────────


class TestSchemaV2Check:
    def test_returns_pass_when_all_present(self):
        def dispatch(query, params):
            if "SHOW CONSTRAINTS" in query:
                return _FakeResult([{"name": "page_slug_unique"}])
            if "SHOW INDEXES" in query:
                return _FakeResult([{"name": "page_compiled_truth_fulltext"}])
            return _FakeResult([])

        result = check_schema_v2_present(driver=_FakeDriver(dispatch))
        assert result["status"] == PASS

    def test_returns_fail_when_missing_constraint(self):
        def dispatch(query, params):
            if "SHOW CONSTRAINTS" in query:
                return _FakeResult([])  # no constraints
            if "SHOW INDEXES" in query:
                return _FakeResult([{"name": "page_compiled_truth_fulltext"}])
            return _FakeResult([])

        result = check_schema_v2_present(driver=_FakeDriver(dispatch))
        assert result["status"] == FAIL
        assert "page_slug_unique" in result["detail"]
        assert "migrate_to_v2" in result["fix_hint"]

    def test_returns_fail_on_introspection_error(self):
        def dispatch(query, params):
            raise RuntimeError("connection dropped")

        result = check_schema_v2_present(driver=_FakeDriver(dispatch))
        assert result["status"] == FAIL


# ── page_completeness ────────────────────────────────────────────────


class TestPageCompletenessCheck:
    def test_no_pages_returns_pass(self):
        def dispatch(query, params):
            return _FakeResult([{"n": 0}])

        result = check_page_completeness(driver=_FakeDriver(dispatch))
        assert result["status"] == PASS
        assert "0 pages" in result["detail"]

    def test_high_completeness_returns_pass(self):
        state = {"call": 0}

        def dispatch(query, params):
            state["call"] += 1
            # first call → total, second call → filled
            return _FakeResult([{"n": 10 if state["call"] == 1 else 5}])

        result = check_page_completeness(driver=_FakeDriver(dispatch))
        # 50% filled > 25% threshold
        assert result["status"] == PASS

    def test_low_completeness_returns_warn(self):
        state = {"call": 0}

        def dispatch(query, params):
            state["call"] += 1
            return _FakeResult([{"n": 100 if state["call"] == 1 else 5}])  # 5%

        result = check_page_completeness(driver=_FakeDriver(dispatch))
        assert result["status"] == WARN
        assert "compiled_truth" in result["fix_hint"]


# ── edge_validity ────────────────────────────────────────────────────


class TestEdgeValidityCheck:
    def test_no_dangling_edges_returns_pass(self):
        def dispatch(query, params):
            return _FakeResult([{"n": 0}])

        result = check_edge_validity(driver=_FakeDriver(dispatch))
        assert result["status"] == PASS

    def test_dangling_edges_return_fail(self):
        def dispatch(query, params):
            return _FakeResult([{"n": 3}])

        result = check_edge_validity(driver=_FakeDriver(dispatch))
        assert result["status"] == FAIL
        assert "3" in result["detail"]


# ── orphan_count_reasonable ──────────────────────────────────────────


class TestOrphanCountCheck:
    def test_zero_pages_returns_pass(self):
        """orphan_count_reasonable delegates to count_pages + find_orphans.
        Wire both to return empty."""
        state = {"call": 0}

        def dispatch(query, params):
            state["call"] += 1
            # Order: count_pages first, then find_orphans
            if "count(p) AS n" in query:
                return _FakeResult([{"n": 0}])
            return _FakeResult([])

        result = check_orphan_count_reasonable(driver=_FakeDriver(dispatch))
        assert result["status"] == PASS

    def test_low_orphan_ratio_returns_pass(self):
        def dispatch(query, params):
            if "count(p) AS n" in query:
                return _FakeResult([{"n": 100}])
            # find_orphans — return 5 records
            return _FakeResult(
                [
                    {"p": {"slug": f"s{i}", "domain": "concept", "compiled_truth": "", "created_at": "", "updated_at": ""}}
                    for i in range(5)
                ]
            )

        result = check_orphan_count_reasonable(driver=_FakeDriver(dispatch))
        assert result["status"] == PASS  # 5% < 10%

    def test_high_orphan_ratio_returns_warn(self):
        def dispatch(query, params):
            if "count(p) AS n" in query:
                return _FakeResult([{"n": 10}])
            return _FakeResult(
                [
                    {"p": {"slug": f"s{i}", "domain": "concept", "compiled_truth": "", "created_at": "", "updated_at": ""}}
                    for i in range(2)
                ]
            )  # 20% orphan

        result = check_orphan_count_reasonable(driver=_FakeDriver(dispatch))
        assert result["status"] == WARN

    def test_extreme_orphan_ratio_returns_fail(self):
        def dispatch(query, params):
            if "count(p) AS n" in query:
                return _FakeResult([{"n": 4}])
            return _FakeResult(
                [
                    {"p": {"slug": f"s{i}", "domain": "concept", "compiled_truth": "", "created_at": "", "updated_at": ""}}
                    for i in range(3)
                ]
            )  # 75% orphan

        result = check_orphan_count_reasonable(driver=_FakeDriver(dispatch))
        assert result["status"] == FAIL


# ── Orchestrator ─────────────────────────────────────────────────────


class TestRunHealthChecks:
    def test_overall_pass_when_all_pass(self):
        def dispatch(query, params):
            if "SHOW CONSTRAINTS" in query:
                return _FakeResult([{"name": "page_slug_unique"}])
            if "SHOW INDEXES" in query:
                return _FakeResult([{"name": "page_compiled_truth_fulltext"}])
            if "count(p) AS n" in query:
                return _FakeResult([{"n": 0}])
            return _FakeResult([])

        report = run_health_checks(driver=_FakeDriver(dispatch))
        assert report["overall"] == PASS
        assert "schema_v2_present" in report["checks"]
        assert "page_completeness" in report["checks"]
        assert "edge_validity" in report["checks"]
        assert "orphan_count_reasonable" in report["checks"]

    def test_fast_mode_skips_orphans(self):
        def dispatch(query, params):
            if "SHOW CONSTRAINTS" in query:
                return _FakeResult([{"name": "page_slug_unique"}])
            if "SHOW INDEXES" in query:
                return _FakeResult([{"name": "page_compiled_truth_fulltext"}])
            return _FakeResult([{"n": 0}])

        report = run_health_checks(driver=_FakeDriver(dispatch), fast=True)
        assert "orphan_count_reasonable" not in report["checks"]

    def test_overall_fail_when_any_fail(self):
        def dispatch(query, params):
            if "SHOW CONSTRAINTS" in query:
                return _FakeResult([])  # missing
            if "SHOW INDEXES" in query:
                return _FakeResult([])
            return _FakeResult([{"n": 0}])

        report = run_health_checks(driver=_FakeDriver(dispatch))
        assert report["overall"] == FAIL


# ── Live Neo4j sanity tests ──────────────────────────────────────────


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


@neo4j_required
def test_schema_check_on_live_neo4j():
    """Live Neo4j — either the schema is present (post-migration) or missing.
    Either outcome is acceptable; the call should just not crash."""
    from neo4j import GraphDatabase
    from jarvis_memory.config import NEO4J_USER, NEO4J_PASSWORD

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        result = check_schema_v2_present(driver=driver)
    finally:
        driver.close()
    assert result["status"] in {PASS, FAIL}
    assert "check" in result
