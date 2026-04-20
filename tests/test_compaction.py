"""Tests for the Run 3 dream-cycle compaction phases.

The three new phases in ``compaction.CompactionEngine`` are read-only
scanners that surface fix queues. We don't spin up a live Neo4j cluster
here — a small ``_FakeDriver`` supplies the rows each phase queries.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from jarvis_memory.compaction import CompactionEngine


# ── Tiny Neo4j driver double ────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Reads from a script: list of (predicate, rows_or_exception)."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def close(self):
        return None

    def run(self, cypher, **params):
        self.calls.append({"cypher": cypher, "params": params})
        for pred, payload in self.scripts:
            if pred(cypher):
                if isinstance(payload, Exception):
                    raise payload
                return _FakeResult(payload)
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, scripts):
        self.scripts = scripts

    def session(self):
        return _FakeSession(self.scripts)


def _engine_from(scripts) -> CompactionEngine:
    e = CompactionEngine.__new__(CompactionEngine)
    e._driver = _FakeDriver(scripts)
    e._owns_driver = False
    e._graphiti = None
    e._embed_store = None
    return e


# ── _fix_citations ──────────────────────────────────────────────────────


class TestFixCitations:
    def test_flags_broken_citations(self):
        # Three episodes; only the middle two have bad citations.
        rows = [
            {"uuid": "ok-1", "content": "normal content with no citation patterns"},
            {"uuid": "bad-1", "content": "see details [cite: ]"},  # empty cite
            {"uuid": "bad-2", "content": "see episode abc and move on"},  # stub ref
        ]
        engine = _engine_from([
            (lambda c: "MATCH (n:Episode)" in c and "n.content" in c, rows),
        ])
        report = engine._fix_citations()
        assert report["scanned"] == 3
        assert report["broken_count"] == 2
        flagged_uuids = {item["uuid"] for item in report["queue"]}
        assert flagged_uuids == {"bad-1", "bad-2"}

    def test_runtime_budget_under_two_minutes(self):
        """Even on a long corpus, the phase should complete quickly."""
        # 500 rows, none of which match — exercise the full loop.
        rows = [
            {"uuid": f"u{i}", "content": f"content with no issue {i}"}
            for i in range(500)
        ]
        engine = _engine_from([
            (lambda c: "MATCH (n:Episode)" in c, rows),
        ])
        start = time.monotonic()
        report = engine._fix_citations()
        elapsed = time.monotonic() - start
        # 2 minutes is 120 s; mock path should finish well under 1 s.
        assert elapsed < 120, f"phase took {elapsed:.2f}s"
        assert report["scanned"] == 500
        assert report["broken_count"] == 0

    def test_no_driver_returns_error_marker(self):
        engine = CompactionEngine.__new__(CompactionEngine)
        engine._driver = None
        engine._owns_driver = False
        engine._graphiti = None
        engine._embed_store = None
        report = engine._fix_citations()
        assert report["scanned"] == 0
        assert report["broken_count"] == 0


# ── _report_orphans ─────────────────────────────────────────────────────


class TestReportOrphans:
    def test_counts_grouped_by_domain(self, monkeypatch):
        """Use monkeypatch to replace find_orphans with a canned result."""
        from jarvis_memory.pages import Page

        fake_orphans = {
            "company": [
                Page(
                    slug="foundry",
                    domain="company",
                    compiled_truth="",
                    created_at="",
                    updated_at="",
                ),
                Page(
                    slug="rivian",
                    domain="company",
                    compiled_truth="",
                    created_at="",
                    updated_at="",
                ),
            ],
            "concept": [
                Page(
                    slug="rag",
                    domain="concept",
                    compiled_truth="",
                    created_at="",
                    updated_at="",
                ),
            ],
        }

        import jarvis_memory.orphans as orphans_mod

        monkeypatch.setattr(orphans_mod, "find_orphans", lambda **_kw: fake_orphans)

        engine = _engine_from([])  # no Neo4j needed because find_orphans is mocked
        report = engine._report_orphans()
        assert report["total_orphans"] == 3
        assert report["by_domain"] == {"company": 2, "concept": 1}
        assert "foundry" in report["sample"]["company"]


# ── _reconcile_stale_edges ──────────────────────────────────────────────


class TestReconcileStaleEdges:
    def test_returns_rows_and_counts(self):
        rows = [
            {"page_slug": "foundry", "episode_uuid": ""},
            {"page_slug": "rivian", "episode_uuid": ""},
        ]
        engine = _engine_from([
            (lambda c: "MATCH (p:Page)-[r:EVIDENCED_BY]->(e)" in c, rows),
        ])
        report = engine._reconcile_stale_edges()
        assert report["stale_edges"] == 2
        assert report["sample"][0]["page_slug"] == "foundry"

    def test_no_stale_edges(self):
        engine = _engine_from([
            (lambda c: "MATCH (p:Page)" in c, []),
        ])
        report = engine._reconcile_stale_edges()
        assert report["stale_edges"] == 0
        assert report["sample"] == []


# ── run_dream_cycle aggregator ──────────────────────────────────────────


class TestRunDreamCycle:
    def test_aggregates_all_three(self, monkeypatch):
        """run_dream_cycle must call each phase and return all three reports."""
        import jarvis_memory.orphans as orphans_mod

        monkeypatch.setattr(orphans_mod, "find_orphans", lambda **_kw: {})

        engine = _engine_from([
            (lambda c: "MATCH (n:Episode)" in c, []),
            (lambda c: "MATCH (p:Page)-[r:EVIDENCED_BY]" in c, []),
        ])
        out = engine.run_dream_cycle()
        assert set(out.keys()) >= {"fix_citations", "orphans", "stale_edges"}
