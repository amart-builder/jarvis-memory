"""Tests for conversation persistence — sessions, episodes, and snapshots.

Tests validate the logic without requiring a Neo4j connection.
EpisodeRecorder.should_record() and SnapshotManager.format_snapshot_for_injection()
are pure functions that can be tested directly.

Integration tests with Neo4j should be run separately.
"""
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from jarvis_memory.conversation import (
    EpisodeRecorder,
    SnapshotManager,
    _SIGNIFICANT_KEYWORDS,
)


class TestEpisodeShouldRecord:
    """Tests for the episode significance heuristic."""

    def test_empty_content_rejected(self):
        assert EpisodeRecorder.should_record("") is False

    def test_none_content_rejected(self):
        assert EpisodeRecorder.should_record(None) is False

    def test_short_content_rejected(self):
        """Content under EPISODE_MIN_LENGTH (50 chars) should be filtered."""
        assert EpisodeRecorder.should_record("too short") is False

    def test_long_but_no_keywords_rejected(self):
        """Long content without significance keywords should be filtered."""
        content = "x" * 100  # long enough, but no keywords
        assert EpisodeRecorder.should_record(content) is False

    def test_decision_keyword_accepted(self):
        content = "We decided to use Neo4j as the primary graph database for this project."
        assert EpisodeRecorder.should_record(content) is True

    def test_plan_keyword_accepted(self):
        content = "The plan is to implement conversation persistence across both devices first."
        assert EpisodeRecorder.should_record(content) is True

    def test_completed_keyword_accepted(self):
        content = "Successfully completed the migration of all scoring functions from MemClawz."
        assert EpisodeRecorder.should_record(content) is True

    def test_blocked_keyword_accepted(self):
        content = "We are blocked on the Neo4j connectivity issue between MacBook Pro and Mac Mini."
        assert EpisodeRecorder.should_record(content) is True

    def test_because_keyword_accepted(self):
        content = "Chose hooks-based architecture because MCP tool invocation rate is only 50-80%."
        assert EpisodeRecorder.should_record(content) is True

    def test_architecture_keyword_accepted(self):
        content = "The architecture uses a three-level hierarchy: Project, Session, and Episode nodes."
        assert EpisodeRecorder.should_record(content) is True

    def test_case_insensitive(self):
        content = "DECIDED to go with the shared brain approach for cross-device continuity."
        assert EpisodeRecorder.should_record(content) is True

    def test_shipped_keyword_accepted(self):
        content = "We shipped the first version of the jarvis-memory package with all 28 MCP tools."
        assert EpisodeRecorder.should_record(content) is True

    def test_error_keyword_accepted(self):
        content = "Got an error when trying to connect to Neo4j remotely — firewall might be blocking port 7687."
        assert EpisodeRecorder.should_record(content) is True

    def test_milestone_keyword_accepted(self):
        content = "Reached the milestone of having all conversation persistence classes implemented and tested."
        assert EpisodeRecorder.should_record(content) is True


class TestSignificantKeywords:
    """Tests for the keyword list itself."""

    def test_keywords_not_empty(self):
        assert len(_SIGNIFICANT_KEYWORDS) > 0

    def test_decision_keywords_present(self):
        assert "decided" in _SIGNIFICANT_KEYWORDS
        assert "decision" in _SIGNIFICANT_KEYWORDS
        assert "chose" in _SIGNIFICANT_KEYWORDS

    def test_plan_keywords_present(self):
        assert "plan" in _SIGNIFICANT_KEYWORDS
        assert "next step" in _SIGNIFICANT_KEYWORDS

    def test_completion_keywords_present(self):
        assert "completed" in _SIGNIFICANT_KEYWORDS
        assert "shipped" in _SIGNIFICANT_KEYWORDS
        assert "deployed" in _SIGNIFICANT_KEYWORDS

    def test_blocker_keywords_present(self):
        assert "blocked" in _SIGNIFICANT_KEYWORDS
        assert "error" in _SIGNIFICANT_KEYWORDS
        assert "bug" in _SIGNIFICANT_KEYWORDS

    def test_context_keywords_present(self):
        assert "because" in _SIGNIFICANT_KEYWORDS
        assert "key insight" in _SIGNIFICANT_KEYWORDS

    def test_architecture_keywords_present(self):
        assert "architecture" in _SIGNIFICANT_KEYWORDS
        assert "schema" in _SIGNIFICANT_KEYWORDS
        assert "database" in _SIGNIFICANT_KEYWORDS


class TestSnapshotFormatting:
    """Tests for snapshot → context block formatting."""

    def test_empty_snapshot_returns_empty(self):
        assert SnapshotManager.format_snapshot_for_injection({}) == ""
        assert SnapshotManager.format_snapshot_for_injection(None) == ""

    def test_basic_snapshot_includes_task(self):
        snapshot = {
            "_device": "macbook-pro",
            "task": "Building shared brain architecture",
            "status": "in_progress",
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "Building shared brain architecture" in result
        assert "in_progress" in result
        assert "macbook-pro" in result

    def test_snapshot_includes_completed_items(self):
        snapshot = {
            "_device": "mac-mini",
            "task": "Test task",
            "status": "completed",
            "completed": ["Built scoring engine", "Wrote classifier"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "Built scoring engine" in result
        assert "Wrote classifier" in result

    def test_snapshot_includes_next_steps(self):
        snapshot = {
            "_device": "mac-mini",
            "task": "Test task",
            "status": "in_progress",
            "next_steps": ["Deploy to Mac Mini", "Run integration tests"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "Deploy to Mac Mini" in result
        assert "Run integration tests" in result

    def test_snapshot_includes_blockers(self):
        snapshot = {
            "_device": "macbook-pro",
            "task": "Test task",
            "status": "blocked",
            "blockers": ["Neo4j not reachable from MacBook Pro"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "Neo4j not reachable" in result

    def test_snapshot_includes_key_decisions(self):
        snapshot = {
            "_device": "mac-mini",
            "task": "Test task",
            "status": "in_progress",
            "key_decisions": ["Using hooks instead of relying solely on MCP tool invocation"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "hooks instead of relying" in result

    def test_snapshot_includes_files_modified(self):
        snapshot = {
            "_device": "macbook-pro",
            "task": "Test task",
            "status": "in_progress",
            "files_modified": ["server.py", "conversation.py"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "server.py" in result
        assert "conversation.py" in result

    def test_full_snapshot_formatting(self):
        """A complete snapshot should produce a well-structured context block."""
        snapshot = {
            "_device": "macbook-pro",
            "_session_id": "abc12345-1234-1234-1234-123456789abc",
            "_session_started": "2026-04-04T10:00:00+00:00",
            "task": "Implement cross-device session continuity",
            "status": "in_progress",
            "completed": ["Built conversation.py", "Updated hooks"],
            "in_progress": ["Adding MCP dispatch handlers"],
            "next_steps": ["Write tests", "Deploy"],
            "key_decisions": ["Shared Neo4j on Mac Mini"],
            "blockers": [],
            "files_modified": ["conversation.py", "server.py"],
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)

        # Should have section headers
        assert "Session State" in result
        assert "Task:" in result
        assert "Status:" in result

    def test_snapshot_missing_device_uses_unknown(self):
        snapshot = {
            "task": "Some task",
            "status": "in_progress",
        }
        result = SnapshotManager.format_snapshot_for_injection(snapshot)
        assert "unknown" in result


class _FakeDriver:
    """Minimal Neo4j driver stub for EpisodeRecorder.record_episode.

    Records every Cypher statement for assertion. Returns a single-record
    result on MATCH queries so the recorder's group_id lookup succeeds.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    @contextmanager
    def session(self):
        fake = _FakeSession(self.calls)
        yield fake

    def close(self):  # pragma: no cover - not used by the test
        pass


class _FakeSession:
    def __init__(self, log: list[tuple[str, dict]]):
        self._log = log

    def run(self, query: str, **params):
        self._log.append((query.strip().split("\n", 1)[0], params))
        return _FakeResult(params)

    def close(self):  # pragma: no cover
        pass


class _FakeResult:
    def __init__(self, params: dict):
        self._params = params

    def single(self):
        # Return a record that looks like {"gid": "system"} for group_id lookups.
        class R:
            def __getitem__(self_inner, key):
                return "system"
        return R()


class TestDetectLayerWarning:
    """Run 1: detect_layer warning hook on the write path."""

    def test_detect_layer_warning_fires(self, caplog):
        """Agent-ops content at conf > 0.7 → WARNING log; episode still persists."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        # Use content that is clearly ops-flavored and meets should_record
        # length + keyword bar. "default to" + "always" + significant keyword.
        content = (
            "User prefers voice input over typed prompts. Always respond with "
            "bullet points, never paragraphs. Default to Claude Sonnet unless "
            "the decision calls for deeper reasoning."
        )

        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="preference",
                group_id="system",
            )

        # Episode still persisted (non-None uuid, Cypher write issued).
        assert ep_id is not None
        assert any("CREATE (e:Episode" in q for q, _ in driver.calls), (
            f"expected an Episode CREATE call; got {[q for q, _ in driver.calls]}"
        )

        # Warning emitted with our structured key.
        msgs = [rec.message for rec in caplog.records if rec.levelno >= logging.WARNING]
        assert any("possible_mis_routed_write" in m for m in msgs), (
            f"expected mis-routed-write warning, got {msgs!r}"
        )

    def test_no_warning_for_world_knowledge(self, caplog):
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        content = (
            "[DECISION] Chose Neo4j over Weaviate for jarvis-memory. WHY: better "
            "graph traversal ergonomics. IMPACT: unblocks typed-edge design."
        )
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="system",
            )

        assert ep_id is not None
        assert not any(
            "possible_mis_routed_write" in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
        )

    def test_none_episode_type_does_not_raise(self):
        """Legacy callers may omit episode_type — classifier.classify_memory fills in."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = (
            "Shipped the rebrand preview to staging. We decided to keep the "
            "old domain live for 48 hours as a fallback."
        )
        ep_id = recorder.record_episode(
            session_id="test-session",
            content=content,
            episode_type=None,
            group_id="navi",
        )
        assert ep_id is not None
