"""Rule-based query intent classifier — one test per class × ≥ 2 each."""
from __future__ import annotations

import pytest

from jarvis_memory.search.intent import classify


class TestEntity:
    @pytest.mark.parametrize(
        "query",
        [
            "What does Foundry do?",  # proper noun mid-sentence
            "tell me about Foundry Ventures",  # two-token cap phrase
            "Notes on Rivian",  # proper noun after prep
        ],
    )
    def test_proper_noun_queries_classify_as_entity(self, query):
        assert classify(query) == "entity"

    def test_entity_stopwords_do_not_trigger(self):
        """A capitalized tag like 'Decision' must NOT be an entity."""
        # This is actually an event (keyword match) — but the key is
        # that we did NOT return "entity".
        assert classify("Decision") == "event"


class TestTemporal:
    @pytest.mark.parametrize(
        "query",
        [
            "what happened yesterday",
            "decisions last week",  # temporal dominates over event
            "anything from since 2024-01-01",
            "today's updates",
            "show notes before 2025-04-01",
        ],
    )
    def test_temporal_phrases_dominate(self, query):
        assert classify(query) == "temporal"


class TestEvent:
    @pytest.mark.parametrize(
        "query",
        [
            "find the handoff",
            "what decisions about auth",
            "meeting with Foundry",
            "milestone review notes",
        ],
    )
    def test_event_vocabulary_triggers_event_intent(self, query):
        assert classify(query) == "event"


class TestGeneral:
    @pytest.mark.parametrize(
        "query",
        [
            "what is the architecture",
            "how does it work",
            "explain the memory system",
            "",
            "   ",
        ],
    )
    def test_fallback_to_general(self, query):
        assert classify(query) == "general"


class TestPriorityOrder:
    """Make sure the documented priority (multi_hop > temporal > event > entity > general) holds."""

    def test_multi_hop_wins_over_temporal(self):
        # Two entities + temporal phrase. PPR-channel is more
        # important here than recency slicing.
        assert classify("decisions in Catalyst that affected Astack last week") == "multi_hop"

    def test_temporal_wins_over_event(self):
        # Both "decisions" (event) and "last week" (temporal) present.
        assert classify("decisions last week") == "temporal"

    def test_event_wins_over_entity(self):
        # "meeting" (event) + "Foundry" (entity) — event wins.
        assert classify("meeting about Foundry") == "event"

    def test_entity_wins_over_general(self):
        # Neither temporal nor event — just an entity mention.
        assert classify("show me Foundry overview") == "entity"


class TestMultiHop:
    """Multi-hop intent fires for relationship/causal queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "decisions in Catalyst that affected Astack",  # 2 entities
            "what led to the Foundry pricing change",  # 1 entity + assoc
            "trail from Navi to Foundry",  # 2 entities + path word
            "how Jacob influenced the Foundry direction",  # 1 entity + assoc
            "connection between Navi and Catalyst",  # 2 entities + connector
        ],
    )
    def test_relationship_queries_classify_as_multi_hop(self, query):
        assert classify(query) == "multi_hop"

    def test_single_entity_factoid_is_not_multi_hop(self):
        """One entity, no associative phrase — should stay ``entity``."""
        assert classify("what does Foundry do") == "entity"

    def test_single_entity_with_event_word_stays_event(self):
        """Multi-hop requires either ≥2 entities OR ≥1 entity + assoc.
        Just an event word + entity goes to ``event``."""
        assert classify("meeting with Foundry") == "event"

    def test_associative_phrase_alone_does_not_trigger_multi_hop(self):
        """An associative phrase with NO entity falls through to general
        — PPR has nothing to seed at."""
        assert classify("what led to it") == "general"
