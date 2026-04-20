"""MinionWorker tests — drive run_once() deterministically against :memory: SQLite."""
from __future__ import annotations

import threading
import time

import pytest

from jarvis_memory.minions.handlers import register_handler
from jarvis_memory.minions.queue import MinionQueue
from jarvis_memory.minions.worker import MinionWorker


@pytest.fixture
def q():
    """File-backed queue so the worker's executor thread can see the rows
    a test thread inserts. :memory: on one thread isn't visible to another
    thread's separate connection — but MinionQueue is per-instance so both
    sides are sharing the same connection, which makes :memory: safe here.
    Using tmp_path too to double-check.
    """
    qi = MinionQueue(":memory:")
    try:
        yield qi
    finally:
        qi.close()


class TestRunOnce:
    def test_empty_queue_returns_zero(self, q):
        register_handler("noop", lambda p: {"ok": True})
        w = MinionWorker(q, queue="default")
        assert w.run_once() == 0

    def test_claim_execute_complete_roundtrip(self, q):
        register_handler("echo2", lambda p: {"got": p})
        job_id = q.submit("echo2", {"x": 42})
        w = MinionWorker(q, queue="default")
        assert w.run_once() == 1
        job = q.get(job_id)
        assert job.status == "complete"
        assert job.result == {"got": {"x": 42}}

    def test_missing_handler_fails_nonretriable(self, q):
        job_id = q.submit("unregistered_job", {}, max_attempts=5)
        w = MinionWorker(q, queue="default")
        w.run_once()
        job = q.get(job_id)
        # No handler → non-retriable → failed (not dead) on first fail.
        assert job.status == "failed"
        assert "no handler" in job.failure_reason.lower()


class TestHandlerErrors:
    def test_handler_exception_fails_retriable(self, q):
        def boomer(params):
            raise RuntimeError("boom")

        register_handler("boomer", boomer)
        job_id = q.submit("boomer", {}, max_attempts=3)
        w = MinionWorker(q, queue="default")
        w.run_once()
        job = q.get(job_id)
        assert job.status == "pending"  # retriable, still attempts left
        assert "RuntimeError" in job.failure_reason

    def test_handler_exhausts_retries_goes_dead(self, q):
        def boomer(params):
            raise RuntimeError("boom")

        register_handler("boomer2", boomer)
        job_id = q.submit("boomer2", {}, max_attempts=1)
        w = MinionWorker(q, queue="default")
        w.run_once()
        # Force scheduled_at to past so retry would be claimable — but
        # max_attempts=1 means dead on first failure.
        job = q.get(job_id)
        assert job.status == "dead"


class TestTimeout:
    def test_timeout_marks_job_failed(self, q):
        def slow(params):
            time.sleep(5)
            return {"done": True}

        register_handler("slowpoke", slow)
        job_id = q.submit("slowpoke", {}, timeout_seconds=1, max_attempts=1)
        w = MinionWorker(q, queue="default")
        w.run_once()
        job = q.get(job_id)
        assert job.status == "dead"  # max_attempts=1 → dead immediately
        assert job.failure_reason == "timeout"

    def test_fast_handler_under_timeout_completes(self, q):
        def fast(params):
            time.sleep(0.05)
            return {"ms": 50}

        register_handler("fast", fast)
        job_id = q.submit("fast", {}, timeout_seconds=5)
        w = MinionWorker(q, queue="default")
        w.run_once()
        assert q.get(job_id).status == "complete"


class TestGracefulShutdown:
    def test_stop_event_exits_run_loop(self, q):
        register_handler("tick", lambda p: {"tick": True})
        w = MinionWorker(q, queue="default", idle_poll_interval=0.05)
        thread = threading.Thread(target=w.run, daemon=True)
        thread.start()
        time.sleep(0.15)
        w.stop()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


class TestHandlerSignature:
    def test_handler_with_job_param_is_passed_job(self, q):
        captured: dict = {}

        def takes_job(params, job=None):
            captured["job_id"] = job.id if job else None
            return {"seen": True}

        register_handler("with_job", takes_job)
        job_id = q.submit("with_job", {})
        w = MinionWorker(q, queue="default")
        w.run_once()
        assert captured["job_id"] == job_id
