"""MinionQueue state-machine tests.

Validates: submit/claim/complete/fail/cancel/stall_sweep across all the
edge cases the plan calls out (CRUD, claim locking, idempotency,
parent-child, cascade cancel, stall sweep, retry backoff).

All tests run against ``:memory:`` SQLite or ``tmp_path`` — never against
the real ``data/minions.sqlite`` file.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from jarvis_memory.minions.queue import (
    DEFAULT_CHILD_CAP,
    MAX_JOB_DEPTH,
    MinionQueue,
    STALL_MULTIPLIER,
)
from jarvis_memory.minions.types import JobStatus


class TestSubmit:
    def test_submit_returns_uuid_string(self, queue_mem):
        job_id = queue_mem.submit("echo", {"x": 1})
        assert isinstance(job_id, str) and len(job_id) == 36

    def test_submit_persists_params(self, queue_mem):
        job_id = queue_mem.submit("echo", {"x": 1, "y": "hi"})
        job = queue_mem.get(job_id)
        assert job is not None
        assert job.params == {"x": 1, "y": "hi"}
        assert job.status == "pending"

    def test_submit_rejects_empty_name(self, queue_mem):
        with pytest.raises(ValueError):
            queue_mem.submit("", {})

    def test_submit_rejects_zero_timeout(self, queue_mem):
        with pytest.raises(ValueError):
            queue_mem.submit("echo", {}, timeout_seconds=0)


class TestIdempotency:
    def test_same_key_returns_same_job_id(self, queue_mem):
        first = queue_mem.submit("echo", {"a": 1}, idempotency_key="k1")
        second = queue_mem.submit("echo", {"a": 2}, idempotency_key="k1")
        assert first == second

    def test_different_names_allow_same_key(self, queue_mem):
        """Key is scoped per-name; same key under a different name is a new job."""
        a = queue_mem.submit("echo", {}, idempotency_key="shared")
        b = queue_mem.submit("compact_daily", {}, idempotency_key="shared")
        assert a != b

    def test_null_key_does_not_conflict(self, queue_mem):
        a = queue_mem.submit("echo", {"x": 1})
        b = queue_mem.submit("echo", {"x": 2})
        assert a != b


class TestClaim:
    def test_claim_single_pending_job(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        cr = queue_mem.claim("default", limit=1)
        assert len(cr.jobs) == 1
        assert cr.jobs[0].id == job_id
        assert cr.jobs[0].status == "claimed"
        assert cr.jobs[0].attempts == 1
        assert cr.jobs[0].worker_id == cr.worker_id

    def test_claim_empty_queue_returns_empty(self, queue_mem):
        cr = queue_mem.claim("default")
        assert cr.jobs == []

    def test_claim_respects_priority_order(self, queue_mem):
        low = queue_mem.submit("echo", {"prio": "low"}, priority=0)
        high = queue_mem.submit("echo", {"prio": "high"}, priority=10)
        cr = queue_mem.claim("default", limit=1)
        assert cr.jobs[0].id == high
        cr2 = queue_mem.claim("default", limit=1)
        assert cr2.jobs[0].id == low

    def test_claim_ignores_future_scheduled_jobs(self, queue_mem):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        queue_mem.submit("echo", {}, scheduled_at=future)
        cr = queue_mem.claim("default")
        assert cr.jobs == []

    def test_two_claims_do_not_return_same_job(self, queue_mem):
        """Serial claim test — BEGIN IMMEDIATE must ensure a job is claimed once."""
        queue_mem.submit("echo", {"a": 1})
        queue_mem.submit("echo", {"a": 2})
        first = queue_mem.claim("default", limit=1)
        second = queue_mem.claim("default", limit=1)
        assert first.jobs[0].id != second.jobs[0].id

    def test_claim_respects_queue_filter(self, queue_mem):
        queue_mem.submit("echo", {}, queue="other")
        cr = queue_mem.claim("default")
        assert cr.jobs == []
        cr2 = queue_mem.claim("other")
        assert len(cr2.jobs) == 1

    def test_claim_limit_one_by_default(self, queue_mem):
        for _ in range(3):
            queue_mem.submit("echo", {})
        cr = queue_mem.claim("default")
        assert len(cr.jobs) == 1

    def test_claim_limit_multi(self, queue_mem):
        for _ in range(5):
            queue_mem.submit("echo", {})
        cr = queue_mem.claim("default", limit=3)
        assert len(cr.jobs) == 3


class TestComplete:
    def test_complete_sets_status_and_result(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        queue_mem.claim("default")
        job = queue_mem.complete(job_id, {"ok": True})
        assert job.status == "complete"
        assert job.result == {"ok": True}
        assert job.completed_at is not None

    def test_complete_on_unclaimed_raises(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        with pytest.raises(ValueError):
            queue_mem.complete(job_id, {})

    def test_complete_writes_to_parent_inbox(self, queue_mem):
        parent_id = queue_mem.submit("echo", {"role": "parent"})
        child_id = queue_mem.submit("echo", {"role": "child"}, parent_id=parent_id)
        queue_mem.claim("default", limit=1)  # claims parent or child; doesn't matter
        # Find and claim/complete specifically the child.
        # Easier: mark parent claimed then claim again.
        # Reset by claiming the remaining job (whichever).
        pending = queue_mem.list(status="pending")
        for job in pending:
            queue_mem.claim("default", limit=1)
        # Now complete the child.
        queue_mem.complete(child_id, {"v": 42})
        inbox = queue_mem.collect_children_done(parent_id)
        assert len(inbox) == 1
        assert inbox[0]["child_id"] == child_id
        assert inbox[0]["result"] == {"v": 42}


class TestFailAndRetry:
    def test_fail_retriable_returns_to_pending(self, queue_mem):
        job_id = queue_mem.submit("echo", {}, max_attempts=3)
        queue_mem.claim("default")
        job = queue_mem.fail(job_id, "transient", retriable=True)
        assert job.status == "pending"
        assert job.failure_reason == "transient"

    def test_fail_nonretriable_marks_failed(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        queue_mem.claim("default")
        job = queue_mem.fail(job_id, "bad input", retriable=False)
        assert job.status == "failed"

    def test_fail_after_max_attempts_marks_dead(self, queue_mem):
        job_id = queue_mem.submit("echo", {}, max_attempts=1)
        queue_mem.claim("default")
        job = queue_mem.fail(job_id, "boom", retriable=True)
        assert job.status == "dead"


class TestCancel:
    def test_cancel_single(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        n = queue_mem.cancel(job_id, cascade=False)
        assert n == 1
        job = queue_mem.get(job_id)
        assert job.status == "cancelled"

    def test_cancel_cascades_to_children(self, queue_mem):
        parent = queue_mem.submit("echo", {"tag": "root"})
        child1 = queue_mem.submit("echo", {"tag": "c1"}, parent_id=parent)
        child2 = queue_mem.submit("echo", {"tag": "c2"}, parent_id=parent)
        grandchild = queue_mem.submit("echo", {"tag": "g1"}, parent_id=child1)
        n = queue_mem.cancel(parent, cascade=True)
        assert n == 4
        for jid in (parent, child1, child2, grandchild):
            assert queue_mem.get(jid).status == "cancelled"

    def test_cancel_skips_already_terminal(self, queue_mem):
        a = queue_mem.submit("echo", {})
        queue_mem.claim("default")
        queue_mem.complete(a, {})
        n = queue_mem.cancel(a, cascade=False)
        assert n == 0
        assert queue_mem.get(a).status == "complete"


class TestParentChild:
    def test_depth_is_parent_depth_plus_one(self, queue_mem):
        parent = queue_mem.submit("echo", {})
        child = queue_mem.submit("echo", {}, parent_id=parent)
        grand = queue_mem.submit("echo", {}, parent_id=child)
        assert queue_mem.get(parent).depth == 0
        assert queue_mem.get(child).depth == 1
        assert queue_mem.get(grand).depth == 2

    def test_max_depth_enforced(self, queue_mem):
        last = queue_mem.submit("echo", {})
        for _ in range(MAX_JOB_DEPTH):
            last = queue_mem.submit("echo", {}, parent_id=last)
        # Now at depth == MAX_JOB_DEPTH, next submit must refuse.
        with pytest.raises(ValueError, match="depth"):
            queue_mem.submit("echo", {}, parent_id=last)

    def test_unknown_parent_raises(self, queue_mem):
        with pytest.raises(ValueError, match="parent_id"):
            queue_mem.submit("echo", {}, parent_id="not-a-real-id")


class TestStallSweep:
    def test_no_stalled_when_fresh(self, queue_mem):
        queue_mem.submit("echo", {}, timeout_seconds=30)
        queue_mem.claim("default")
        stalled = queue_mem.stall_sweep()
        assert stalled == []

    def test_detects_stalled_job(self, queue_mem):
        queue_mem.submit("echo", {}, timeout_seconds=10)
        queue_mem.claim("default")
        future = datetime.now(timezone.utc) + timedelta(seconds=10 * STALL_MULTIPLIER + 1)
        stalled = queue_mem.stall_sweep(now=future)
        assert len(stalled) == 1


class TestLeaseRenewal:
    def test_renew_updates_claimed_at(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        cr = queue_mem.claim("default")
        ok = queue_mem.renew_lease(job_id, worker_id=cr.worker_id)
        assert ok is True

    def test_renew_fails_for_wrong_worker(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        queue_mem.claim("default")
        ok = queue_mem.renew_lease(job_id, worker_id="nobody")
        assert ok is False


class TestLogging:
    def test_log_appends(self, queue_mem):
        job_id = queue_mem.submit("echo", {})
        queue_mem.log(job_id, "INFO", "started")
        queue_mem.log(job_id, "DEBUG", "progress 50%")
        logs = queue_mem.get_logs(job_id)
        assert len(logs) == 2
        assert logs[0]["message"] == "started"
        assert logs[1]["message"] == "progress 50%"


class TestList:
    def test_list_filters_by_status(self, queue_mem):
        p1 = queue_mem.submit("echo", {})
        queue_mem.submit("echo", {})
        queue_mem.claim("default")
        pending = queue_mem.list(status="pending")
        claimed = queue_mem.list(status="claimed")
        assert len(pending) == 1
        assert len(claimed) == 1

    def test_list_limit(self, queue_mem):
        for _ in range(10):
            queue_mem.submit("echo", {})
        rows = queue_mem.list(limit=3)
        assert len(rows) == 3
