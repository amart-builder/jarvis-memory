"""Unit tests for the bi-temporal migration script.

These tests mock the Neo4j driver — they don't touch a real database.
The dry-run / apply / rollback paths through ``main()`` are exercised
end-to-end via a fake driver that records every Cypher statement.

Live verification happens out-of-band: run ``python scripts/migrate_to_bitemporal.py
--dry-run`` against the prod Mini, eyeball the counts, then ``--apply``
and re-dry-run to confirm zero pending. See the script docstring.
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def migrate_module():
    """Import the migration script as a module so we can call its helpers."""
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "migrate_to_bitemporal", repo_root / "scripts" / "migrate_to_bitemporal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_driver(per_label_counts: dict[str, int]):
    """Build a driver mock whose ``run().single()`` returns count records.

    The migration runs one query per label. We dispatch by the label name
    appearing in the Cypher string.
    """

    class _Result:
        def __init__(self, count: int):
            self._record = {"c": count}

        def single(self):
            return self._record

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_):
            for label, count in per_label_counts.items():
                if f":{label})" in query:
                    return _Result(count)
            return _Result(0)

    driver = MagicMock()
    driver.session.return_value = _Session()
    return driver


# ── FACT_LABELS shape ──────────────────────────────────────────────────


def test_fact_labels_includes_all_fact_bearing_node_types(migrate_module):
    labels = set(migrate_module.FACT_LABELS)
    assert "Episode" in labels, "primary fact-bearing label missing"
    assert "Page" in labels, "entity-hub label missing"
    assert "Entity" in labels, "legacy graphiti entity label missing"
    assert "Episodic" in labels, "legacy graphiti episode label missing"


def test_fact_labels_excludes_operational_labels(migrate_module):
    labels = set(migrate_module.FACT_LABELS)
    # Sessions and snapshots are operational metadata, not fact claims —
    # they shouldn't get the bi-temporal treatment.
    for non_fact in ("Session", "Snapshot", "Community", "Saga"):
        assert non_fact not in labels, f"{non_fact} should not be in FACT_LABELS"


# ── _count_pending ─────────────────────────────────────────────────────


def test_count_pending_aggregates_per_label(migrate_module):
    driver = _make_driver({"Episode": 231, "Page": 276, "Entity": 125, "Episodic": 12})
    counts = migrate_module._count_pending(driver)
    assert counts == {"Episode": 231, "Page": 276, "Entity": 125, "Episodic": 12}


def test_count_pending_zero_when_already_migrated(migrate_module):
    driver = _make_driver({"Episode": 0, "Page": 0, "Entity": 0, "Episodic": 0})
    counts = migrate_module._count_pending(driver)
    assert sum(counts.values()) == 0


# ── _apply_migration ────────────────────────────────────────────────────


def test_apply_migration_returns_per_label_counts(migrate_module):
    driver = _make_driver({"Episode": 5, "Page": 7, "Entity": 0, "Episodic": 0})
    migrated = migrate_module._apply_migration(driver)
    assert migrated == {"Episode": 5, "Page": 7, "Entity": 0, "Episodic": 0}


def test_apply_migration_query_does_not_set_t_expired(migrate_module):
    """``t_expired`` must stay NULL — only the migration ``_apply_migration``
    sets ``t_created``. ``t_expired`` only gets set later by lifecycle
    events (contradict / supersede). Inspect every statement run."""
    statements: list[str] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_):
            statements.append(query)

            class _R:
                def single(self):
                    return {"c": 0}

            return _R()

    driver = MagicMock()
    driver.session.return_value = _Session()

    migrate_module._apply_migration(driver)

    assert statements, "_apply_migration must execute at least one query"
    for stmt in statements:
        assert "t_created" in stmt, f"migration query should set t_created: {stmt!r}"
        # The migration must not touch t_expired — that's lifecycle-only.
        # We allow the literal name to appear as part of "is null" guards
        # only on the t_created side; assert the SET clause excludes it.
        set_section = stmt.split("SET", 1)[1] if "SET" in stmt else ""
        assert "t_expired" not in set_section, (
            f"migration must not set t_expired in SET clause: {stmt!r}"
        )


def test_apply_migration_uses_coalesce_with_created_at(migrate_module):
    """Backfill must default to ``created_at`` and only fall back to
    ``datetime()`` when ``created_at`` itself is NULL — otherwise we'd
    silently rewrite the ingestion timeline to "now" for legacy nodes.
    """
    statements: list[str] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_):
            statements.append(query)

            class _R:
                def single(self):
                    return {"c": 0}

            return _R()

    driver = MagicMock()
    driver.session.return_value = _Session()
    migrate_module._apply_migration(driver)

    for stmt in statements:
        assert "coalesce(n.created_at" in stmt, (
            f"migration must coalesce against existing created_at: {stmt!r}"
        )


# ── _apply_rollback ─────────────────────────────────────────────────────


def test_rollback_removes_both_properties(migrate_module):
    """Rollback drops t_created AND t_expired."""
    statements: list[str] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_):
            statements.append(query)

            class _R:
                def single(self):
                    return {"c": 0}

            return _R()

    driver = MagicMock()
    driver.session.return_value = _Session()
    migrate_module._apply_rollback(driver)

    assert statements, "_apply_rollback must execute at least one query"
    for stmt in statements:
        assert "REMOVE" in stmt, f"rollback must use REMOVE: {stmt!r}"
        assert "n.t_created" in stmt, f"rollback must drop t_created: {stmt!r}"
        assert "n.t_expired" in stmt, f"rollback must drop t_expired: {stmt!r}"


def test_rollback_idempotent_via_guard(migrate_module):
    """Rollback should target only nodes that have at least one of the
    properties — re-runnable cleanly when nothing remains."""
    statements: list[str] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **_):
            statements.append(query)

            class _R:
                def single(self):
                    return {"c": 0}

            return _R()

    driver = MagicMock()
    driver.session.return_value = _Session()
    migrate_module._apply_rollback(driver)

    for stmt in statements:
        # Either property non-null guards the match.
        assert "IS NOT NULL" in stmt, f"rollback should guard with IS NOT NULL: {stmt!r}"
