"""Shared fixtures for Minions tests.

Critical isolation rules (see task packet):
  - Queue tests use ``:memory:`` SQLite or a temp file in ``tests/tmp/``.
    NEVER write to ``~/Atlas/jarvis-memory/data/minions.sqlite``.
  - Audit tests monkeypatch ``GBRAIN_AUDIT_DIR`` to a tmp path.
    NEVER write to real ``~/Atlas/jarvis-memory/audit/``.
  - Shell handler tests exercise only trivial commands (``echo``, ``true``,
    ``false``, ``sleep 0.1``). NEVER run destructive shell commands.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis_memory.minions.queue import MinionQueue
from jarvis_memory.minions.handlers import _clear_registry_for_tests


@pytest.fixture
def queue_mem() -> MinionQueue:
    """In-memory MinionQueue — fresh per test, auto-closed."""
    q = MinionQueue(":memory:")
    try:
        yield q
    finally:
        q.close()


@pytest.fixture
def queue_file(tmp_path) -> MinionQueue:
    """File-backed MinionQueue in pytest's ``tmp_path``. Auto-closed."""
    db = tmp_path / "minions.sqlite"
    q = MinionQueue(db)
    try:
        yield q
    finally:
        q.close()


@pytest.fixture(autouse=True)
def _clean_handler_registry():
    """Wipe the handler registry before every test so registrations don't leak.

    The built-in handlers re-register themselves on import, but tests that
    import ``builtin`` explicitly trigger that in a controlled way.
    """
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


@pytest.fixture(autouse=True)
def _audit_dir_isolation(monkeypatch, tmp_path):
    """Redirect audit writes to a per-test tmp directory.

    If a test doesn't touch the audit module, this is a no-op. If it does,
    the file lands in ``tmp_path/audit/`` instead of the real home dir.
    """
    monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
    yield


@pytest.fixture
def shell_enabled(monkeypatch):
    """Enable ``GBRAIN_ALLOW_SHELL_JOBS=1`` for the duration of a test."""
    monkeypatch.setenv("GBRAIN_ALLOW_SHELL_JOBS", "1")
    yield


@pytest.fixture
def shell_disabled(monkeypatch):
    """Ensure ``GBRAIN_ALLOW_SHELL_JOBS`` is unset."""
    monkeypatch.delenv("GBRAIN_ALLOW_SHELL_JOBS", raising=False)
    yield
