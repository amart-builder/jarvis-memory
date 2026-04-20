"""Minions — SQLite-native deterministic job queue.

Public surface:
  MinionQueue   — state-machine wrapper around the ``jobs`` table.
  MinionWorker  — daemon that claims and runs jobs via registered handlers.
  register_handler / get_handler — handler registry for named jobs.

All three are safe to import without side effects. Built-in handlers
(``echo``, ``compact_daily``, ``compact_weekly``) register themselves
automatically when ``jarvis_memory.minions.handlers.builtin`` is imported.

The shell handler is gated behind ``GBRAIN_ALLOW_SHELL_JOBS=1`` and is
NOT imported here — callers wanting shell jobs must import
``jarvis_memory.minions.handlers.shell`` explicitly.
"""
from __future__ import annotations

from .queue import (
    CLAIMABLE_STATES,
    ClaimResult,
    Job,
    JobStatus,
    MinionQueue,
    TERMINAL_STATES,
)
from .handlers import (
    get_handler,
    list_handlers,
    register_handler,
    unregister_handler,
)

# MinionWorker is imported lazily inside get_worker() to avoid pulling in
# signal/subprocess machinery at import time (e.g., on the REST hot path
# where we only want MinionQueue).

__all__ = [
    "CLAIMABLE_STATES",
    "ClaimResult",
    "Job",
    "JobStatus",
    "MinionQueue",
    "TERMINAL_STATES",
    "register_handler",
    "get_handler",
    "unregister_handler",
    "list_handlers",
    "get_worker",
]


def get_worker(*args, **kwargs):
    """Lazy accessor for ``MinionWorker`` (deferred import)."""
    from .worker import MinionWorker

    return MinionWorker(*args, **kwargs)
