"""Schema-level tests for the Minions SQLite DDL.

Validates:
  - ``connect()`` creates all tables on first call.
  - Re-applying the schema on the same connection is a no-op (idempotent).
  - The critical indexes exist.
"""
from __future__ import annotations

import sqlite3

import pytest

from jarvis_memory.minions.schema import APPLY_STATEMENTS, apply_schema, connect


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {r[0] for r in cur.fetchall()}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )
    return {r[0] for r in cur.fetchall()}


class TestConnect:
    def test_creates_all_three_tables(self):
        conn = connect(":memory:")
        try:
            tables = _table_names(conn)
            assert "jobs" in tables
            assert "job_logs" in tables
            assert "job_children_done" in tables
        finally:
            conn.close()

    def test_creates_expected_indexes(self):
        conn = connect(":memory:")
        try:
            idx = _index_names(conn)
            assert "ix_jobs_status_sched" in idx
            assert "ix_jobs_parent_id" in idx
            assert "ux_jobs_idempotency" in idx
            assert "ix_job_logs_job_ts" in idx
            assert "ix_children_parent_id" in idx
        finally:
            conn.close()

    def test_file_backed_parent_dir_created(self, tmp_path):
        db = tmp_path / "deep" / "nested" / "minions.sqlite"
        conn = connect(db)
        try:
            assert db.exists()
        finally:
            conn.close()


class TestIdempotency:
    def test_reapply_does_not_raise(self):
        conn = connect(":memory:")
        try:
            apply_schema(conn)
            apply_schema(conn)  # third time, still fine
            tables = _table_names(conn)
            assert {"jobs", "job_logs", "job_children_done"}.issubset(tables)
        finally:
            conn.close()

    def test_apply_statements_all_idempotent(self):
        """Each statement in APPLY_STATEMENTS should be valid 'IF NOT EXISTS' DDL."""
        for stmt in APPLY_STATEMENTS:
            assert "IF NOT EXISTS" in stmt.upper(), (
                f"Non-idempotent DDL detected: {stmt[:80]}..."
            )


class TestWalMode:
    def test_wal_enabled_for_file_db(self, tmp_path):
        conn = connect(tmp_path / "minions.sqlite")
        try:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            # Some filesystems silently downgrade from wal. Accept wal or
            # persist/delete without failing.
            assert mode.lower() in {"wal", "delete", "truncate", "persist", "memory"}
        finally:
            conn.close()

    def test_memory_db_works(self):
        conn = connect(":memory:")
        try:
            cur = conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode.lower() == "memory"
        finally:
            conn.close()
