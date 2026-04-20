"""SQLite DDL for the Minions job queue.

Three tables:
  jobs              — job rows (id, name, params, status, attempts, ...).
  job_logs          — structured log lines per job (for post-mortems).
  job_children_done — inbox of completed child-job results for a parent to
                      collect (enables parent-child flows without the parent
                      having to poll every child).

Indexes:
  ix_jobs_status_sched   — dominant claim query (status + scheduled_at).
  ix_jobs_parent_id      — parent-child traversal and cascade cancel.
  ix_jobs_idempotency    — uniqueness check on (name, idempotency_key).
  ix_job_logs_job_ts     — per-job log ordering.
  ix_children_parent_id  — collect children for a parent.

The DDL is idempotent (IF NOT EXISTS everywhere). Calling ``connect(db_path)``
repeatedly against the same file is safe.

SQLite is stdlib — no new dependencies. WAL journal mode is enabled so
readers don't block the single writer (claim path).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union


# DDL statements applied in order by ``connect`` / ``apply_schema``. Each
# must be idempotent (IF NOT EXISTS).
APPLY_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id                 TEXT PRIMARY KEY,
        name               TEXT NOT NULL,
        queue              TEXT NOT NULL DEFAULT 'default',
        params             TEXT NOT NULL DEFAULT '{}',
        status             TEXT NOT NULL DEFAULT 'pending',
        attempts           INTEGER NOT NULL DEFAULT 0,
        max_attempts       INTEGER NOT NULL DEFAULT 3,
        priority           INTEGER NOT NULL DEFAULT 0,
        created_at         TEXT NOT NULL,
        scheduled_at       TEXT NOT NULL,
        claimed_at         TEXT,
        completed_at       TEXT,
        failed_at          TEXT,
        failure_reason     TEXT,
        result             TEXT,
        parent_id          TEXT,
        depth              INTEGER NOT NULL DEFAULT 0,
        idempotency_key    TEXT,
        timeout_seconds    INTEGER NOT NULL DEFAULT 60,
        worker_id          TEXT,
        trusted            INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_logs (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id  TEXT NOT NULL,
        ts      TEXT NOT NULL,
        level   TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_children_done (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id     TEXT NOT NULL,
        child_id      TEXT NOT NULL,
        result_json   TEXT,
        completed_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_jobs_status_sched ON jobs(status, scheduled_at)",
    "CREATE INDEX IF NOT EXISTS ix_jobs_parent_id ON jobs(parent_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_idempotency ON jobs(name, idempotency_key) WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_job_logs_job_ts ON job_logs(job_id, ts)",
    "CREATE INDEX IF NOT EXISTS ix_children_parent_id ON job_children_done(parent_id)",
]


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply all DDL statements idempotently to an open connection."""
    cur = conn.cursor()
    for stmt in APPLY_STATEMENTS:
        cur.execute(stmt)
    conn.commit()


def connect(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open a SQLite connection and apply the schema.

    - Ensures the parent directory exists for file-backed databases.
    - Enables WAL journal mode (except for ``:memory:``) so readers don't
      block the single writer on the claim path.
    - Sets ``row_factory`` to ``sqlite3.Row`` for dict-style access.
    - ``detect_types`` is off; all timestamps are stored as ISO-8601 strings.

    Pass ``':memory:'`` for an ephemeral in-memory database (tests).
    """
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        db_str,
        isolation_level=None,
        timeout=30.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    if db_str != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    apply_schema(conn)
    return conn
