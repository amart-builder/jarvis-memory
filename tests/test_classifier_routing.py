"""Tests for classifier.detect_layer — three-layer routing classifier.

Pure-Python. No DB. Spec §5-6.
"""
from __future__ import annotations

import pytest

from jarvis_memory.classifier import detect_layer


# ── world_knowledge ────────────────────────────────────────────────────


class TestWorldKnowledge:
    """Default bucket: facts, decisions, plans, events about the world."""

    def test_decision_type_and_content(self):
        layer, conf = detect_layer(
            "[DECISION] Chose Clerk over Auth0 because native MCP support.",
            episode_type="decision",
        )
        assert layer == "world_knowledge"
        assert conf > 0.0

    def test_fact_type_with_proper_noun(self):
        layer, conf = detect_layer(
            "Navi's Postgres host lives in us-east-1. Schema v3 shipped 2026-03-01.",
            episode_type="fact",
        )
        assert layer == "world_knowledge"
        # World_knowledge with strong content should have positive confidence.
        assert conf > 0.0

    def test_plan_type(self):
        layer, _ = detect_layer(
            "[PLAN] Build the SPV carry report by end of sprint. Owner: Alex.",
            episode_type="plan",
        )
        assert layer == "world_knowledge"

    def test_event_type_meeting(self):
        layer, _ = detect_layer(
            "Met with Marcus about Foundry LP portal launch timeline.",
            episode_type="event",
        )
        assert layer == "world_knowledge"

    def test_no_type_concrete_entity(self):
        """Neutral prose with a proper noun defaults to world_knowledge."""
        layer, _ = detect_layer("Alex shipped jarvis-memory v0.1 on Monday.")
        assert layer == "world_knowledge"

    def test_empty_content_defaults(self):
        """Empty content should not raise and should default to world_knowledge."""
        layer, conf = detect_layer("")
        assert layer == "world_knowledge"
        assert conf == 0.0

    def test_none_episode_type_ok(self):
        """episode_type=None is legal and handled without raising."""
        layer, _ = detect_layer("Deployed jarvis-memory to the Mac Mini.", episode_type=None)
        assert layer == "world_knowledge"


# ── agent_operations ──────────────────────────────────────────────────


class TestAgentOperations:
    """Preferences, config, always/never directives, response formatting."""

    def test_user_preference_explicit(self):
        layer, conf = detect_layer(
            "User prefers voice input over typed prompts for long-form drafts."
        )
        assert layer == "agent_operations"
        assert conf > 0.7

    def test_alex_preference(self):
        layer, conf = detect_layer("Alex prefers short standup summaries with bullet points.")
        assert layer == "agent_operations"
        assert conf > 0.5

    def test_always_never_directive(self):
        layer, conf = detect_layer(
            "Always respond in first person for emails drafted in Alex's voice. Never use em dashes."
        )
        assert layer == "agent_operations"
        assert conf > 0.7

    def test_default_to_behavior(self):
        layer, conf = detect_layer(
            "Default to Claude Sonnet 4.5 unless the task requires Opus reasoning depth."
        )
        assert layer == "agent_operations"
        assert conf > 0.5

    def test_explicit_preference_type_low_signal_content(self):
        """If episode_type=preference, trust the caller."""
        layer, conf = detect_layer("Keep it short.", episode_type="preference")
        assert layer == "agent_operations"
        # Explicit type → high base confidence even with sparse content.
        assert conf >= 0.85

    def test_config_type(self):
        layer, _ = detect_layer(
            "Turn off autoformat for markdown files.", episode_type="config"
        )
        assert layer == "agent_operations"

    def test_claude_should_directive(self):
        layer, conf = detect_layer(
            "Claude should always cite the spec file before writing code."
        )
        assert layer == "agent_operations"
        assert conf > 0.5

    def test_auto_memory_reference(self):
        layer, _ = detect_layer(
            "Add this rule to .claude/settings.json and keep auto-memory scoped to project tasks."
        )
        assert layer == "agent_operations"


# ── session_ephemeral ─────────────────────────────────────────────────


class TestSessionEphemeral:
    """References to the current conversation / pronoun-heavy / [TEMP]."""

    def test_this_conversation(self):
        layer, conf = detect_layer("In this conversation, we've been debugging the webhook.")
        assert layer == "session_ephemeral"
        assert conf > 0.5

    def test_just_now(self):
        layer, conf = detect_layer("What I said just now about the timeout was wrong.")
        assert layer == "session_ephemeral"
        assert conf > 0.5

    def test_explicit_temp_tag(self):
        layer, conf = detect_layer("[TEMP] Holding this here until the review finishes.")
        assert layer == "session_ephemeral"
        assert conf > 0.5

    def test_pronoun_heavy_no_entity(self):
        """Short, pronoun-dense prose with no proper noun should look ephemeral."""
        layer, conf = detect_layer("it is what i think we should do about them now")
        assert layer == "session_ephemeral"
        assert conf >= 0.7

    def test_as_we_discussed(self):
        layer, _ = detect_layer("As we discussed earlier in this thread, the fix is idempotent.")
        assert layer == "session_ephemeral"

    def test_ephemeral_type(self):
        layer, conf = detect_layer(
            "Random scratch thought while reading the diff.", episode_type="ephemeral"
        )
        assert layer == "session_ephemeral"
        assert conf >= 0.85


# ── edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Behavior-under-stress tests. None must raise."""

    def test_ambiguous_short_text_defaults_world(self):
        """Ambiguous 2-3 word input should fall back to world_knowledge with low conf."""
        layer, conf = detect_layer("Okay.")
        assert layer == "world_knowledge"
        assert conf <= 0.5

    def test_no_llm_call_on_hot_path(self, monkeypatch):
        """detect_layer must not instantiate anthropic — heuristics only."""
        import jarvis_memory.classifier as c

        def boom(*a, **kw):
            raise AssertionError("anthropic.Anthropic should not be called from detect_layer")

        monkeypatch.setattr(c, "classify_with_llm", boom)
        # Run every branch — none should invoke the LLM.
        detect_layer("User prefers markdown.")
        detect_layer("[TEMP] scratch")
        detect_layer("[DECISION] Chose Clerk", episode_type="decision")
        detect_layer("")

    def test_confidence_in_range(self):
        for content, ep in [
            ("User prefers bullet lists.", None),
            ("In this conversation, we will now do X.", None),
            ("[DECISION] Chose Clerk.", "decision"),
            ("", None),
            ("Random.", None),
        ]:
            _, conf = detect_layer(content, episode_type=ep)
            assert 0.0 <= conf <= 0.95

    def test_return_type_shape(self):
        result = detect_layer("Alex shipped.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        layer, conf = result
        assert layer in {"world_knowledge", "agent_operations", "session_ephemeral"}
        assert isinstance(conf, float)

    def test_non_string_content_does_not_crash(self):
        # Some callers might pass numbers/None by mistake. Must not raise.
        layer, _ = detect_layer(None)  # type: ignore[arg-type]
        assert layer == "world_knowledge"
        layer2, _ = detect_layer(12345)  # type: ignore[arg-type]
        assert layer2 == "world_knowledge"
