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
        # Store the FULL stripped query so tests can assert on multi-line
        # patterns (e.g. [r:WORKS_AT] appears on line 3 of edge MERGE).
        self._log.append((query.strip(), params))
        return _FakeResult(params)

    def close(self):  # pragma: no cover
        pass


class _FakeResult:
    def __init__(self, params: dict):
        self._params = params

    def single(self):
        # Return a record that looks generic enough for group_id lookup
        # AND for Page.from_record upserts (which expects props dict).
        class R:
            def __init__(self_inner, params):
                self_inner._params = params
                # Synthesize a page-ish prop bag for any 'p' lookup.
                self_inner._page_props = {
                    "slug": params.get("slug") or params.get("from_slug") or "x",
                    "domain": params.get("domain") or params.get("from_domain") or "",
                    "compiled_truth": params.get("compiled_truth", "") or "",
                    "created_at": "t",
                    "updated_at": "t",
                }

            def __getitem__(self_inner, key):
                if key in ("gid", "n"):
                    return "system" if key == "gid" else 0
                if key == "p":
                    return self_inner._page_props
                if key == "r":
                    return {"at": "t", "summary": ""}
                return "system"

        return R(self._params)


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


# ── Run 2: Page + typed-edge maintenance on record_episode ───────────


class TestPageMaintenance:
    """Run 2: record_episode now maintains Pages + typed edges.

    Assertions inspect the Cypher issued by the recorder against a
    fake driver — integration tests (live Neo4j) live elsewhere.
    """

    def test_page_merge_on_first_mention(self):
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        content = (
            "[FACT] Alice Cooper works at Foundry Inc. We decided to bring "
            "her onto the platform team this quarter."
        )
        ep_id = recorder.record_episode(
            session_id="test-session",
            content=content,
            episode_type="fact",
            group_id="foundry",
        )
        assert ep_id is not None

        # Expect at least one MERGE (p:Page ... slug: ...) for an entity.
        page_merges = [q for q, _ in driver.calls if "MERGE (p:Page {slug:" in q]
        assert page_merges, f"expected Page MERGE; got {[q[:80] for q, _ in driver.calls]}"

    def test_typed_edge_materialized(self):
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        recorder.record_episode(
            session_id="test-session",
            content=(
                "[FACT] Alice Cooper works at Foundry Inc on the platform team. "
                "Important relationship; decided to prioritize onboarding."
            ),
            episode_type="fact",
            group_id="foundry",
        )
        # create_edges_in_tx issues a MERGE with the edge type embedded.
        edge_mergers = [q for q, _ in driver.calls if "[r:WORKS_AT]" in q]
        assert edge_mergers, (
            f"expected WORKS_AT edge MERGE; got types: "
            f"{[q[:120] for q, _ in driver.calls if 'MERGE' in q]}"
        )

    def test_timeline_append_on_every_referenced_page(self):
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        recorder.record_episode(
            session_id="test-session",
            content=(
                "[FACT] Alice Cooper works at Foundry Inc. Important context: "
                "we decided to keep her reporting line flat for at least six months."
            ),
            episode_type="fact",
            group_id="foundry",
        )
        # EVIDENCED_BY edge creation happens via append_timeline_entry.
        evidenced_calls = [q for q, _ in driver.calls if "[r:EVIDENCED_BY]" in q]
        assert evidenced_calls, (
            f"expected EVIDENCED_BY MERGE; got: {[q[:80] for q, _ in driver.calls]}"
        )

    def test_zero_entity_refs_still_writes_episode(self):
        """No proper nouns in content → no Page work, but the episode still writes."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        content = (
            "completed today's refactor; shipped to staging. "
            "decided to roll forward on the plan rather than revert."
        )
        ep_id = recorder.record_episode(
            session_id="test-session",
            content=content,
            episode_type="fact",
            group_id="system",
        )
        assert ep_id is not None
        # Still performed the Episode CREATE.
        assert any("CREATE (e:Episode" in q for q, _ in driver.calls)

    def test_none_episode_type_works(self):
        """Back-compat: episode_type=None goes through classifier.classify_memory
        then Page maintenance. Must not raise."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        ep_id = recorder.record_episode(
            session_id="test-session",
            content="[FACT] Alice Cooper works at Foundry Inc. Decided to hire her.",
            episode_type=None,
            group_id="foundry",
        )
        assert ep_id is not None

    def test_page_maintenance_failure_does_not_lose_episode(self):
        """If Page maintenance raises, the episode is still returned."""
        driver = _FakeDriver()

        # Monkey-patch _maintain_pages_for_episode to explode.
        recorder = EpisodeRecorder(driver=driver)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated page-maintenance failure")

        recorder._maintain_pages_for_episode = _boom  # type: ignore[assignment]

        ep_id = recorder.record_episode(
            session_id="test-session",
            content=(
                "[FACT] Alice Cooper works at Foundry Inc. Decided to hire her "
                "onto the platform team. Important precedent."
            ),
            episode_type="fact",
            group_id="foundry",
        )
        # Episode still persisted (we caught the exception).
        assert ep_id is not None

    def test_ephemeral_content_skips_page_creation(self):
        """A session-ephemeral-looking episode should NOT seed Pages."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)

        recorder.record_episode(
            session_id="test-session",
            content=(
                "earlier in this conversation we decided to push it to next "
                "week — just making sure you remember the plan."
            ),
            episode_type="ephemeral",
            group_id="system",
        )
        page_merges = [q for q, _ in driver.calls if "MERGE (p:Page {slug:" in q]
        assert page_merges == [], (
            f"expected zero Page MERGEs for ephemeral content; got {page_merges}"
        )


# ── Run 2: performance guard ────────────────────────────────────────


def test_record_episode_page_maintenance_under_50ms():
    """Page maintenance (in-memory, fake-driver) adds < 50 ms overhead.

    The fake driver short-circuits every .run() call in microseconds, so
    the measured number is extraction + orchestration, not DB latency.
    Per spec §B5 the real DB-backed budget is 50 ms; our fake-driver
    budget is a tight overhead proxy.
    """
    import time

    driver = _FakeDriver()
    recorder = EpisodeRecorder(driver=driver)

    content = (
        "[FACT] Alice Cooper works at Foundry Inc on the platform team. "
        "Bob Jones founded Navi Systems last year. "
        "Carol Smith advises Catalyst Partners. Decided to document these."
    )

    start = time.perf_counter()
    ep_id = recorder.record_episode(
        session_id="test-session",
        content=content,
        episode_type="fact",
        group_id="system",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert ep_id is not None
    # Generous envelope — test sanity check, not a hard SLA.
    assert elapsed_ms < 50.0, f"record_episode took {elapsed_ms:.2f}ms, expected < 50ms"


# ── Run 4: OperationContext-driven abuse audit on record_episode ─────


class TestTrustBoundaryAudit:
    """Run 4: ctx=OperationContext(remote=True) + abuse heuristic → WARNING.

    All tests verify LOG-ONLY behavior: episodes are still persisted, calls
    never raise due to the audit, and local/None ctx skips audit entirely.
    """

    def test_no_audit_when_ctx_is_none(self, caplog):
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = "rm -rf / is a dangerous command that we decided to never run."
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="system",
                ctx=None,
            )
        assert ep_id is not None
        assert not any(
            "possible_abusive_remote_write" in rec.message
            for rec in caplog.records
        )

    def test_no_audit_when_ctx_is_local(self, caplog):
        from jarvis_memory.operation_context import OperationContext

        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = "rm -rf / — we decided never to run this in production."
        ctx = OperationContext.for_cli("alex")
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="system",
                ctx=ctx,
            )
        assert ep_id is not None
        assert not any(
            "possible_abusive_remote_write" in rec.message
            for rec in caplog.records
        )

    def test_remote_ctx_with_shell_payload_warns(self, caplog):
        from jarvis_memory.operation_context import OperationContext

        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = (
            "We decided to never ship this — rm -rf / would nuke production. "
            "The plan is to block any writes that look like shell payloads."
        )
        ctx = OperationContext.for_mcp("malicious-agent")
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="system",
                ctx=ctx,
            )
        # Episode still persisted — LOG-ONLY means we don't refuse.
        assert ep_id is not None
        # Warning fired.
        warnings = [
            rec for rec in caplog.records
            if "possible_abusive_remote_write" in rec.message
        ]
        assert len(warnings) == 1
        text = warnings[0].message
        assert "malicious-agent" in text
        assert "mcp" in text
        assert "rm" in text  # pattern name surfaced in reasons

    def test_remote_ctx_with_nonstandard_group_id_warns(self, caplog):
        from jarvis_memory.operation_context import OperationContext

        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = "We decided to ship v2 of the scoring algorithm next sprint."
        ctx = OperationContext.for_rest("10.0.0.5")
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="Weird Group With Spaces!",
                ctx=ctx,
            )
        assert ep_id is not None
        warnings = [
            rec for rec in caplog.records
            if "possible_abusive_remote_write" in rec.message
        ]
        assert len(warnings) == 1
        assert "non_canonical_shape" in warnings[0].message

    def test_remote_ctx_with_large_content_warns(self, caplog):
        from jarvis_memory.operation_context import OperationContext

        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        # Mix a significant keyword with a lot of filler so should_record passes.
        content = "decided to paste: " + ("A" * 12_000)
        ctx = OperationContext.for_mcp("bulk-writer")
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="system",
                ctx=ctx,
            )
        assert ep_id is not None
        warnings = [
            rec for rec in caplog.records
            if "possible_abusive_remote_write" in rec.message
        ]
        assert len(warnings) == 1
        assert "content_over_10kb" in warnings[0].message

    def test_remote_ctx_clean_content_does_not_warn(self, caplog):
        from jarvis_memory.operation_context import OperationContext

        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = (
            "Decided to migrate the scoring weights from a flat config to a "
            "tiered structure keyed by memory_type. Plan is to ship next sprint."
        )
        ctx = OperationContext.for_mcp("trusted-agent")
        with caplog.at_level(logging.WARNING, logger="jarvis_memory.conversation"):
            ep_id = recorder.record_episode(
                session_id="test-session",
                content=content,
                episode_type="decision",
                group_id="jarvis-memory",
                ctx=ctx,
            )
        assert ep_id is not None
        # No abuse warning.
        assert not any(
            "possible_abusive_remote_write" in rec.message
            for rec in caplog.records
        )

    def test_record_episode_signature_backward_compatible(self):
        """Callers that don't know about ctx must still work (positional + keyword)."""
        driver = _FakeDriver()
        recorder = EpisodeRecorder(driver=driver)
        content = "Decided to keep the legacy API shape for backward compat."
        # Call WITHOUT ctx — pre-Run-4 signature.
        ep_id = recorder.record_episode(
            session_id="s",
            content=content,
            episode_type="decision",
            group_id="system",
        )
        assert ep_id is not None
