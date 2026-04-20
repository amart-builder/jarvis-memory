"""Tests for the lifecycle state machine — transition validation only.

These tests validate the state machine logic without requiring a Neo4j connection.
Integration tests with Neo4j should be run separately.
"""
from jarvis_memory.lifecycle import VALID_TRANSITIONS, LIFECYCLE_STATES, DEFAULT_STATUS


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
