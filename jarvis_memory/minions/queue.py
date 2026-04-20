"""MinionQueue — SQLite-backed deterministic job queue.

Exposes the state-machine around the ``jobs`` table defined in ``schema.py``.
Callers never touch SQL directly — they go through ``submit``, ``claim``,
``complete``, ``fail``, ``cancel``, ``stall_sweep``, ``get``, ``list``.

Concurrency model
-----------------
The claim path uses ``BEGIN IMMEDIATE`` to acquire a reserved-write lock
before selecting & updating pending rows. SQLite's serializable isolation
guarantees no two workers can claim the same job. The scope of the immediate
transaction is kept short (one SELECT + one UPDATE) so writer contention is
minimal. WAL journal mode (enabled in ``schema.connect``) keeps readers
unblocked during the claim.

Idempotency
-----------
A non-null ``idempotency_key`` is unique within a given ``name``. Resubmit
against the same ``(name, idempotency_key)`` returns the existing ``job_id``
instead of inserting a duplicate. Implemented via a partial-unique index so
jobs without a key aren't constrained.

Timeouts
--------
``timeout_seconds`` is stored on the row for the worker to enforce at
execution time. The queue itself does not enforce timeouts — that's the
worker's job. ``stall_sweep`` returns claimed jobs whose lease has aged out
(claimed_at + timeout_seconds*STALL_MULTIPLIER < now) so a supervisor can
reclaim or dead-letter them.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .schema import connect as open_connection
from .types import (
    CLAIMABLE_STATES,
    ClaimResult,
    Job,
    JobStatus,
    TERMINAL_STATES,
)

logger = logging.getLogger(__name__)

# How many times the claimed lease can elapse before stall_sweep considers
# the job stuck. 2x means: a 60s job is stale if its lease started >120s ago.
STALL_MULTIPLIER = 2

# Safety ceiling on parent-child depth to prevent runaway job trees.
MAX_JOB_DEPTH = 8

# Per-parent child cap (also a runaway-prevention belt).
DEFAULT_CHILD_CAP = 256


def _utcnow_iso() -> str:
    """UTC ISO-8601 with microseconds (monotonic ordering-friendly)."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_job(row: sqlite3.Row) -> Job:
    """Decode a row into an immutable ``Job`` dataclass."""
    params_raw = row["params"] if row["params"] is not None else "{}"
    result_raw = row["result"]
    try:
        params = json.loads(params_raw) if params_raw else {}
    except json.JSONDecodeError:
        params = {}
    try:
        result = json.loads(result_raw) if result_raw else None
    except json.JSONDecodeError:
        result = None
    return Job(
        id=row["id"],
        name=row["name"],
        queue=row["queue"],
        params=params,
        status=row["status"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        priority=row["priority"],
        created_at=row["created_at"],
        scheduled_at=row["scheduled_at"],
        claimed_at=row["claimed_at"],
        completed_at=row["completed_at"],
        failed_at=row["failed_at"],
        failure_reason=row["failure_reason"],
        result=result,
        parent_id=row["parent_id"],
        depth=row["depth"],
        idempotency_key=row["idempotency_key"],
        timeout_seconds=row["timeout_seconds"],
        worker_id=row["worker_id"],
        trusted=bool(row["trusted"]),
    )


class MinionQueue:
    """Thin state-machine wrapper over the ``jobs`` table.

    Instantiate with a path to a SQLite file (or ``:memory:`` for tests).
    Safe for single-process multi-threaded use: the underlying SQLite
    connection uses ``check_same_thread=False`` and a per-instance
    ``threading.Lock`` serializes all writes from this object's threads,
    which together with ``BEGIN IMMEDIATE`` gives us cross-process
    serializability too.

    ``close()`` should be called on shutdown. Usable as a context manager.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "data/minions.sqlite",
        *,
        conn: Optional[sqlite3.Connection] = None,
    ):
        self.db_path = str(db_path)
        if conn is not None:
            self._conn = conn
        else:
            self._conn = open_connection(self.db_path)
        self._lock = threading.RLock()

    # ── context manager / lifecycle ─────────────────────────────────────

    def __enter__(self) -> MinionQueue:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 — idempotent close
                pass

    # ── internal helpers ────────────────────────────────────────────────

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Cursor]:
        """BEGIN IMMEDIATE transaction. Commits on success, rolls back on error."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE")
                yield cur
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise

    @contextmanager
    def _deferred(self) -> Iterator[sqlite3.Cursor]:
        """Regular (deferred) transaction for read-heavy paths."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN")
                yield cur
                self._conn.commit()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    pass
                raise

    # ── submit ──────────────────────────────────────────────────────────

    def submit(
        self,
        name: str,
        params: Optional[dict[str, Any]] = None,
        *,
        queue: str = "default",
        priority: int = 0,
        parent_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        timeout_seconds: int = 60,
        max_attempts: int = 3,
        scheduled_at: Optional[str] = None,
        trusted: bool = False,
    ) -> str:
        """Enqueue a job. Returns the job_id.

        If ``(name, idempotency_key)`` already exists, returns the existing
        job_id without inserting a duplicate. ``parent_id`` links the job
        to a parent; ``depth`` is computed as ``parent.depth + 1`` and
        capped at ``MAX_JOB_DEPTH``.
        """
        if not name or not isinstance(name, str):
            raise ValueError("job name must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        params = params or {}
        params_json = json.dumps(params, default=str, sort_keys=True)
        now_iso = _utcnow_iso()
        scheduled = scheduled_at or now_iso

        depth = 0
        if parent_id is not None:
            parent = self.get(parent_id)
            if parent is None:
                raise ValueError(f"parent_id {parent_id!r} not found")
            depth = parent.depth + 1
            if depth > MAX_JOB_DEPTH:
                raise ValueError(
                    f"max job depth {MAX_JOB_DEPTH} exceeded (parent depth {parent.depth})"
                )
            child_count = self._count_children(parent_id)
            if child_count >= DEFAULT_CHILD_CAP:
                raise ValueError(
                    f"parent {parent_id} already has {child_count} children (cap {DEFAULT_CHILD_CAP})"
                )

        # Idempotency shortcut — return existing id if a live row matches.
        if idempotency_key is not None:
            existing = self._find_by_idempotency(name, idempotency_key)
            if existing is not None:
                return existing

        job_id = str(uuid.uuid4())
        try:
            with self._immediate() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (
                        id, name, queue, params, status, attempts, max_attempts,
                        priority, created_at, scheduled_at, parent_id, depth,
                        idempotency_key, timeout_seconds, trusted
                    ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        name,
                        queue,
                        params_json,
                        max_attempts,
                        priority,
                        now_iso,
                        scheduled,
                        parent_id,
                        depth,
                        idempotency_key,
                        timeout_seconds,
                        1 if trusted else 0,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            # Another submitter raced us on the idempotency key. Re-read.
            if idempotency_key is None:
                raise
            existing = self._find_by_idempotency(name, idempotency_key)
            if existing is None:
                raise RuntimeError(
                    f"idempotency race: insert failed but no existing row found ({exc})"
                ) from exc
            return existing
        return job_id

    def _find_by_idempotency(self, name: str, key: str) -> Optional[str]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id FROM jobs WHERE name = ? AND idempotency_key = ? LIMIT 1",
                (name, key),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def _count_children(self, parent_id: str) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT count(*) AS n FROM jobs WHERE parent_id = ?",
                (parent_id,),
            )
            row = cur.fetchone()
            return int(row["n"]) if row else 0

    # ── claim ───────────────────────────────────────────────────────────

    def claim(
        self,
        queue: str = "default",
        *,
        limit: int = 1,
        worker_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> ClaimResult:
        """Claim up to ``limit`` pending jobs for execution.

        The returned jobs are now in the ``claimed`` state with
        ``claimed_at`` / ``worker_id`` populated. Attempts is incremented.
        Jobs are selected in (priority DESC, scheduled_at ASC, created_at ASC)
        order. Only jobs with ``scheduled_at <= now`` are eligible.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        with self._immediate() as cur:
            cur.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'pending'
                  AND queue = ?
                  AND scheduled_at <= ?
                ORDER BY priority DESC, scheduled_at ASC, created_at ASC
                LIMIT ?
                """,
                (queue, now_iso, limit),
            )
            rows = cur.fetchall()
            claimed_jobs: list[Job] = []
            for row in rows:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'claimed',
                        claimed_at = ?,
                        worker_id = ?,
                        attempts = attempts + 1
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now_iso, worker_id, row["id"]),
                )
                if cur.rowcount == 1:
                    cur.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],))
                    updated = cur.fetchone()
                    claimed_jobs.append(_row_to_job(updated))
        return ClaimResult(jobs=claimed_jobs, worker_id=worker_id)

    # ── complete / fail / cancel ────────────────────────────────────────

    def complete(self, job_id: str, result: Optional[dict[str, Any]] = None) -> Job:
        """Mark ``job_id`` as complete. If it has a parent, write to the child_done inbox."""
        result_json = json.dumps(result, default=str, sort_keys=True) if result is not None else None
        now_iso = _utcnow_iso()
        with self._immediate() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = 'complete',
                    completed_at = ?,
                    result = ?
                WHERE id = ? AND status = 'claimed'
                """,
                (now_iso, result_json, job_id),
            )
            if cur.rowcount != 1:
                raise ValueError(f"cannot complete job {job_id!r}: not in claimed state")
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            job = _row_to_job(row)
            # Parent inbox: if the job had a parent, record the result for collection.
            if job.parent_id:
                cur.execute(
                    """
                    INSERT INTO job_children_done (parent_id, child_id, result_json, completed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (job.parent_id, job.id, result_json, now_iso),
                )
        return job

    def fail(
        self,
        job_id: str,
        reason: str,
        *,
        retriable: bool = True,
    ) -> Job:
        """Mark a claimed job as failed. If retriable and attempts < max_attempts,
        return it to pending (exponential backoff scheduled_at). Otherwise mark
        terminally failed (``dead`` if retries exhausted, ``failed`` otherwise)."""
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        with self._immediate() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"job {job_id!r} not found")
            if row["status"] != "claimed":
                raise ValueError(
                    f"cannot fail job {job_id!r}: status is {row['status']!r} not 'claimed'"
                )

            attempts = row["attempts"]
            max_attempts = row["max_attempts"]
            if retriable and attempts < max_attempts:
                # Exponential backoff: 2^(attempts-1) * 5s capped at 1h.
                delay_sec = min(3600, (2 ** max(0, attempts - 1)) * 5)
                retry_at = (now_dt + timedelta(seconds=delay_sec)).isoformat()
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending',
                        claimed_at = NULL,
                        worker_id = NULL,
                        scheduled_at = ?,
                        failure_reason = ?
                    WHERE id = ?
                    """,
                    (retry_at, reason[:500], job_id),
                )
            else:
                terminal = JobStatus.DEAD.value if retriable else JobStatus.FAILED.value
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = ?,
                        failed_at = ?,
                        failure_reason = ?
                    WHERE id = ?
                    """,
                    (terminal, now_iso, reason[:500], job_id),
                )
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            updated = cur.fetchone()
        return _row_to_job(updated)

    def cancel(self, job_id: str, *, cascade: bool = True) -> int:
        """Cancel a job and (by default) all its descendants.

        Returns number of rows transitioned to ``cancelled``. Jobs already in
        a terminal state are left alone.
        """
        now_iso = _utcnow_iso()
        cancelled_count = 0
        with self._immediate() as cur:
            # Gather descendant ids BFS.
            to_cancel: list[str] = [job_id]
            if cascade:
                frontier = [job_id]
                while frontier:
                    placeholders = ",".join("?" * len(frontier))
                    cur.execute(
                        f"SELECT id FROM jobs WHERE parent_id IN ({placeholders})",
                        frontier,
                    )
                    children = [r["id"] for r in cur.fetchall()]
                    to_cancel.extend(children)
                    frontier = children
            # Cancel non-terminal rows only.
            terminal_placeholders = ",".join("?" * len(TERMINAL_STATES))
            placeholders = ",".join("?" * len(to_cancel))
            params = list(to_cancel) + list(TERMINAL_STATES)
            cur.execute(
                f"""
                UPDATE jobs
                SET status = 'cancelled',
                    failed_at = ?,
                    failure_reason = 'cancelled'
                WHERE id IN ({placeholders})
                  AND status NOT IN ({terminal_placeholders})
                """,
                (now_iso, *params),
            )
            cancelled_count = cur.rowcount
        return cancelled_count

    # ── stall_sweep ─────────────────────────────────────────────────────

    def stall_sweep(self, *, now: Optional[datetime] = None) -> list[Job]:
        """Return claimed jobs whose lease has expired.

        Does NOT mutate state — returning the list lets the caller decide
        whether to fail, requeue, or ignore. A worker with lease-renewal
        will update ``claimed_at`` before the lease expires.
        """
        now_dt = now or datetime.now(timezone.utc)
        with self._deferred() as cur:
            cur.execute(
                """
                SELECT * FROM jobs
                WHERE status = 'claimed'
                  AND claimed_at IS NOT NULL
                """,
            )
            rows = cur.fetchall()
            stalled: list[Job] = []
            for row in rows:
                claimed_at = datetime.fromisoformat(row["claimed_at"])
                timeout = row["timeout_seconds"]
                expiry = claimed_at + timedelta(seconds=timeout * STALL_MULTIPLIER)
                if now_dt > expiry:
                    stalled.append(_row_to_job(row))
        return stalled

    def renew_lease(self, job_id: str, *, worker_id: str) -> bool:
        """Extend a claimed job's lease — update ``claimed_at`` to now if the
        caller still owns the job. Returns ``True`` on success."""
        now_iso = _utcnow_iso()
        with self._immediate() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET claimed_at = ?
                WHERE id = ? AND status = 'claimed' AND worker_id = ?
                """,
                (now_iso, job_id, worker_id),
            )
            return cur.rowcount == 1

    # ── introspection ───────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            return _row_to_job(row) if row else None

    def list(
        self,
        *,
        status: Optional[str] = None,
        queue: Optional[str] = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status/queue."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if queue is not None:
            clauses.append("queue = ?")
            params.append(queue)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            return [_row_to_job(r) for r in cur.fetchall()]

    def collect_children_done(self, parent_id: str) -> list[dict[str, Any]]:
        """Pop all completed-child entries from the inbox for ``parent_id``.

        Each entry is a dict: ``{"child_id", "result", "completed_at"}``.
        After this call, the inbox for ``parent_id`` is empty.
        """
        with self._immediate() as cur:
            cur.execute(
                "SELECT id, child_id, result_json, completed_at FROM job_children_done WHERE parent_id = ? ORDER BY id",
                (parent_id,),
            )
            rows = cur.fetchall()
            ids = [r["id"] for r in rows]
            out: list[dict[str, Any]] = []
            for r in rows:
                raw = r["result_json"]
                try:
                    result = json.loads(raw) if raw else None
                except json.JSONDecodeError:
                    result = None
                out.append({
                    "child_id": r["child_id"],
                    "result": result,
                    "completed_at": r["completed_at"],
                })
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur.execute(
                    f"DELETE FROM job_children_done WHERE id IN ({placeholders})",
                    ids,
                )
        return out

    def log(self, job_id: str, level: str, message: str) -> None:
        """Append a structured log entry for a job."""
        with self._immediate() as cur:
            cur.execute(
                "INSERT INTO job_logs (job_id, ts, level, message) VALUES (?, ?, ?, ?)",
                (job_id, _utcnow_iso(), level.upper(), message[:2000]),
            )

    def get_logs(self, job_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT ts, level, message FROM job_logs WHERE job_id = ? ORDER BY ts ASC LIMIT ?",
                (job_id, limit),
            )
            return [
                {"ts": r["ts"], "level": r["level"], "message": r["message"]}
                for r in cur.fetchall()
            ]


# Re-exports for typing imports elsewhere.
__all__ = [
    "ClaimResult",
    "Job",
    "JobStatus",
    "MinionQueue",
    "CLAIMABLE_STATES",
    "TERMINAL_STATES",
]
