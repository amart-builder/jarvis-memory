"""Tests for the lifecycle state machine — transition validation only.

These tests validate the state machine logic without requiring a Neo4j connection.
Integration tests with Neo4j should be run separately.
"""
from unittest.mock import MagicMock

from jarvis_memory.lifecycle import (
    DEFAULT_STATUS,
    EXPIRING_STATES,
    LIFECYCLE_STATES,
    VALID_TRANSITIONS,
    MemoryLifecycle,
)


class TestLifecycleStateMachine:
    """Tests for lifecycle state definitions and transition rules."""

    def test_all_states_have_transitions(self):
        """Every state should appear in the transition map."""
        for state in LIFECYCLE_STATES:
            assert state in VALID_TRANSITIONS, f"State '{state}' missing from VALID_TRANSITIONS"

    def test_deleted_is_terminal(self):
        """Deleted state should have no valid transitions."""
        assert VALID_TRANSITIONS["deleted"] == set()

    def test_merged_only_goes_to_deleted(self):
        assert VALID_TRANSITIONS["merged"] == {"deleted"}

    def test_superseded_only_goes_to_deleted(self):
        assert VALID_TRANSITIONS["superseded"] == {"deleted"}

    def test_active_can_reach_all_non_active(self):
        """Active should be able to transition to every other state except itself."""
        active_targets = VALID_TRANSITIONS["active"]
        # Should include all the main states
        assert "confirmed" in active_targets
        assert "outdated" in active_targets
        assert "archived" in active_targets
        assert "deleted" in active_targets
        assert "contradicted" in active_targets

    def test_archived_can_be_restored(self):
        """Archived memories should be restorable to active."""
        assert "active" in VALID_TRANSITIONS["archived"]

    def test_contradicted_can_be_revalidated(self):
        """Contradicted memories should be restorable to active."""
        assert "active" in VALID_TRANSITIONS["contradicted"]

    def test_outdated_can_be_revalidated(self):
        """Outdated memories should be restorable to active."""
        assert "active" in VALID_TRANSITIONS["outdated"]

    def test_default_status_is_active(self):
        assert DEFAULT_STATUS == "active"

    def test_no_self_transitions(self):
        """No state should be able to transition to itself."""
        for state, targets in VALID_TRANSITIONS.items():
            assert state not in targets, f"State '{state}' can transition to itself"

    def test_transition_targets_are_valid_states(self):
        """All transition targets should be valid lifecycle states."""
        for state, targets in VALID_TRANSITIONS.items():
            for target in targets:
                assert target in LIFECYCLE_STATES, (
                    f"Transition {state} -> {target}: '{target}' is not a valid state"
                )

    def test_eight_states_total(self):
        assert len(LIFECYCLE_STATES) == 8


# ── Bi-temporal expiration on lifecycle transition ──────────────────────


def _capture_cypher_driver():
    """Build a driver mock that records every Cypher query and returns
    a successful single-record result. Returns ``(driver, captures)``
    where ``captures`` is a list of ``(query, params)`` tuples.
    """
    captures: list[tuple[str, dict]] = []

    class _Result:
        def single(self):
            return {"uuid": "fake-uuid"}

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            captures.append((query, params))
            return _Result()

    driver = MagicMock()
    driver.session.return_value = _Session()
    return driver, captures


class TestBitemporalExpiration:
    """``transition()`` writes ``t_expired`` + ``valid_to`` only on expiring states."""

    def test_expiring_states_includes_contradicted_superseded_deleted(self):
        assert EXPIRING_STATES == frozenset({"contradicted", "superseded", "deleted"})

    def test_expiring_states_excludes_soft_states(self):
        # Soft transitions: memory might still be true, just not in active rotation.
        for soft in ("outdated", "archived", "confirmed"):
            assert soft not in EXPIRING_STATES, (
                f"{soft!r} should NOT close the bi-temporal window"
            )

    def test_contradicted_transition_sets_t_expired(self):
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)

        ok = lc.transition("uuid-1", "active", "contradicted", metadata={"by": "uuid-2"})

        assert ok, "transition should succeed when driver returns a record"
        assert captures, "expected at least one Cypher query"
        query, _ = captures[0]
        assert "n.t_expired = coalesce(n.t_expired, datetime())" in query, (
            "contradicted transition must set t_expired"
        )
        assert "n.valid_to = coalesce(n.valid_to, datetime())" in query, (
            "contradicted transition must close event-time window too"
        )

    def test_superseded_transition_sets_t_expired(self):
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "superseded", metadata={"by": "uuid-2"})

        query, _ = captures[0]
        assert "n.t_expired = coalesce(n.t_expired, datetime())" in query

    def test_deleted_transition_sets_t_expired(self):
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "deleted")

        query, _ = captures[0]
        assert "n.t_expired" in query

    def test_archived_transition_does_not_set_t_expired(self):
        """Archived = no longer active, but might still be true. Don't expire."""
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "archived")

        query, _ = captures[0]
        assert "t_expired" not in query, (
            f"archived must NOT close the bi-temporal window; got: {query!r}"
        )
        assert "valid_to" not in query, (
            f"archived must NOT pin event-time end; got: {query!r}"
        )

    def test_outdated_transition_does_not_set_t_expired(self):
        """Outdated = needs revalidation. Don't expire — fact may still be true."""
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "outdated")

        query, _ = captures[0]
        assert "t_expired" not in query

    def test_confirmed_transition_does_not_set_t_expired(self):
        """Confirmed = high confidence, still valid. Definitely don't expire."""
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "confirmed")

        query, _ = captures[0]
        assert "t_expired" not in query

    def test_expiring_uses_coalesce_to_preserve_prior_valid_to(self):
        """If a memory already has an explicit ``valid_to`` (set via
        ``set_validity()``), a later contradiction must NOT overwrite it.
        The pre-existing event-time end is the correct truth."""
        driver, captures = _capture_cypher_driver()
        lc = MemoryLifecycle(driver=driver)
        lc.transition("uuid-1", "active", "contradicted")

        query, _ = captures[0]
        # The coalesce ensures: if n.valid_to is already set, it wins.
        assert "coalesce(n.valid_to, datetime())" in query, (
            "valid_to must be preserved when already set"
        )
        assert "coalesce(n.t_expired, datetime())" in query, (
            "t_expired must be preserved when already set "
            "(supports retroactive expiration via set_validity)"
        )
