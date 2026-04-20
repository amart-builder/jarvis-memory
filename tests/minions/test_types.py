"""Type declarations for Minions queue — enum + dataclass invariants."""
from __future__ import annotations

import dataclasses

import pytest

from jarvis_memory.minions.types import (
    CLAIMABLE_STATES,
    ClaimResult,
    Job,
    JobStatus,
    SubmitOptions,
    TERMINAL_STATES,
)


class TestJobStatus:
    def test_all_expected_states_present(self):
        expected = {"pending", "claimed", "complete", "failed", "dead", "cancelled"}
        assert {s.value for s in JobStatus} == expected

    def test_is_string_enum(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PENDING == "pending"  # str-enum equality

    def test_claimable_and_terminal_are_disjoint(self):
        assert CLAIMABLE_STATES.isdisjoint(TERMINAL_STATES)

    def test_terminal_contains_expected_states(self):
        assert {"complete", "failed", "dead", "cancelled"}.issubset(TERMINAL_STATES)


class TestJobImmutability:
    def _make_job(self):
        return Job(
            id="abc",
            name="echo",
            queue="default",
            params={"x": 1},
            status="pending",
            attempts=0,
            max_attempts=3,
            priority=0,
            created_at="2026-04-20T00:00:00+00:00",
            scheduled_at="2026-04-20T00:00:00+00:00",
            claimed_at=None,
            completed_at=None,
            failed_at=None,
            failure_reason=None,
            result=None,
            parent_id=None,
            depth=0,
            idempotency_key=None,
            timeout_seconds=60,
            worker_id=None,
            trusted=False,
        )

    def test_job_is_frozen(self):
        job = self._make_job()
        with pytest.raises(dataclasses.FrozenInstanceError):
            job.status = "claimed"  # type: ignore[misc]

    def test_claim_result_is_frozen(self):
        cr = ClaimResult(jobs=[], worker_id="w1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cr.worker_id = "w2"  # type: ignore[misc]

    def test_submit_options_defaults(self):
        opts = SubmitOptions()
        assert opts.queue == "default"
        assert opts.priority == 0
        assert opts.max_attempts == 3
        assert opts.timeout_seconds == 60
        assert opts.trusted is False
