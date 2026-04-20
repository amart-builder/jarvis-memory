"""Type declarations for the Minions queue.

All dataclasses are frozen so they can be passed across threads safely and
so accidentally mutating a Job instance raises instead of silently corrupting
an in-memory cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class JobStatus(str, Enum):
    """Canonical job states. Sub-classing str keeps SQLite serialization trivial."""

    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETE = "complete"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


CLAIMABLE_STATES: frozenset[str] = frozenset({JobStatus.PENDING.value})

TERMINAL_STATES: frozenset[str] = frozenset({
    JobStatus.COMPLETE.value,
    JobStatus.FAILED.value,
    JobStatus.DEAD.value,
    JobStatus.CANCELLED.value,
})


@dataclass(frozen=True)
class Job:
    """Complete row from the ``jobs`` table.

    Mirrors the schema in ``schema.py``. ``params`` and ``result`` are
    decoded JSON (dicts); ``trusted`` is a bool (INTEGER 0/1 in storage).
    """

    id: str
    name: str
    queue: str
    params: dict[str, Any]
    status: str
    attempts: int
    max_attempts: int
    priority: int
    created_at: str
    scheduled_at: str
    claimed_at: Optional[str]
    completed_at: Optional[str]
    failed_at: Optional[str]
    failure_reason: Optional[str]
    result: Optional[dict[str, Any]]
    parent_id: Optional[str]
    depth: int
    idempotency_key: Optional[str]
    timeout_seconds: int
    worker_id: Optional[str]
    trusted: bool


@dataclass(frozen=True)
class ClaimResult:
    jobs: list[Job]
    worker_id: str


@dataclass(frozen=True)
class SubmitOptions:
    queue: str = "default"
    priority: int = 0
    parent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    timeout_seconds: int = 60
    max_attempts: int = 3
    scheduled_at: Optional[str] = None
    trusted: bool = False
    depth: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)
