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


# ── Postmortem 2026-04-28: daily_digest semantic-dedup safeguards ───────


class _FakeEmbedStore:
    """Test double for EmbeddingStore. Tracks search() invocation count."""

    def __init__(self, returns: list[dict] | None = None, sleep_per_call: float = 0.0):
        self.calls = 0
        self._returns = returns or []
        self._sleep = sleep_per_call

    def health_check(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5):
        self.calls += 1
        if self._sleep:
            time.sleep(self._sleep)
        return list(self._returns)


def _engine_with_embed(embed_store, scripts):
    e = CompactionEngine.__new__(CompactionEngine)
    e._driver = _FakeDriver(scripts)
    e._owns_driver = False
    e._graphiti = None
    e._embed_store = embed_store
    return e


class TestDailyDigestSafeguards:
    """Guards introduced after the 2026-04-28 API runaway postmortem.

    The daily_digest Pass-2 loop calls embed_store.search() once per
    remaining memory, which encodes the query through MiniLM each call.
    These tests pin the safety net: the feature flag actually skips the
    loop, and the wall-clock timeout aborts before going forever.
    """

    @staticmethod
    def _mems(n: int) -> list[dict]:
        return [
            {
                "uuid": f"m-{i}",
                "content": f"unique content number {i}",
                "name": "",
                "memory_type": "fact",
            }
            for i in range(n)
        ]

    def test_flag_disabled_skips_pass_2_entirely(self, monkeypatch):
        """JARVIS_SEMANTIC_DEDUP=0 must NOT call embed_store.search()."""
        monkeypatch.setattr("jarvis_memory.compaction.SEMANTIC_DEDUP_ENABLED", False)

        embed = _FakeEmbedStore()
        rows = self._mems(50)
        engine = _engine_with_embed(embed, [
            (lambda c: "MATCH (n)" in c and "compaction_daily_run IS NULL" in c, rows),
        ])

        report = engine.daily_digest()
        assert embed.calls == 0, f"semantic dedup must be skipped; saw {embed.calls} embed calls"
        # Pass 1 (exact hash dedup) should still run normally
        assert "merged_count" in report

    def test_timeout_aborts_long_loop(self, monkeypatch):
        """Wall-clock cap aborts Pass 2 mid-loop instead of running forever."""
        # Make sure flag is on so we exercise Pass 2
        monkeypatch.setattr("jarvis_memory.compaction.SEMANTIC_DEDUP_ENABLED", True)
        # 1-second cap, 0.4s per search → cap trips after ~3 calls
        monkeypatch.setattr("jarvis_memory.compaction.SEMANTIC_DEDUP_TIMEOUT_SEC", 1)

        embed = _FakeEmbedStore(sleep_per_call=0.4)
        rows = self._mems(20)
        engine = _engine_with_embed(embed, [
            (lambda c: "MATCH (n)" in c and "compaction_daily_run IS NULL" in c, rows),
        ])

        start = time.monotonic()
        report = engine.daily_digest()
        elapsed = time.monotonic() - start

        # Should bail well under the 20-call worst case (20*0.4 = 8s)
        assert elapsed < 4.0, f"timeout did not fire; loop ran {elapsed:.1f}s"
        # Some calls happen before the cap trips, but not all 20
        assert 1 <= embed.calls < 20, f"unexpected call count: {embed.calls}"
        assert "merged_count" in report

    def test_default_flag_runs_pass_2(self, monkeypatch):
        """Sanity: with the flag on (default) and a reasonable timeout, Pass 2
        actually invokes embed_store.search()."""
        monkeypatch.setattr("jarvis_memory.compaction.SEMANTIC_DEDUP_ENABLED", True)
        monkeypatch.setattr("jarvis_memory.compaction.SEMANTIC_DEDUP_TIMEOUT_SEC", 60)

        embed = _FakeEmbedStore(returns=[])
        rows = self._mems(3)
        engine = _engine_with_embed(embed, [
            (lambda c: "MATCH (n)" in c and "compaction_daily_run IS NULL" in c, rows),
        ])
        engine.daily_digest()
        assert embed.calls == 3, f"expected one search per remaining memory; saw {embed.calls}"
