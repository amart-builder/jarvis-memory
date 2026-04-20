"""Tests for the memory classifier module."""
from jarvis_memory.classifier import (
    classify_heuristic,
    classify_memory,
    MEMORY_TYPES,
    _KEYWORD_MAP,
)


class TestClassifyHeuristic:
    """Tests for keyword-based classification."""

    def test_decision_keywords(self):
        assert classify_heuristic("We decided to use Graphiti") == "decision"
        assert classify_heuristic("The team agreed on the architecture") == "decision"

    def test_preference_keywords(self):
        assert classify_heuristic("Alex prefers voice input") == "preference"
        assert classify_heuristic("His favorite tool is Wispr") == "preference"

    def test_procedure_keywords(self):
        assert classify_heuristic("How to deploy the memory server") == "procedure"
        assert classify_heuristic("Steps to configure Neo4j") == "procedure"

    def test_relationship_keywords(self):
        assert classify_heuristic("Alex works at Edge Fund") == "relationship"
        assert classify_heuristic("Contact email is alex@edge-fund.io") == "relationship"

    def test_event_keywords(self):
        assert classify_heuristic("MemClawz v7 launched yesterday") == "event"
        assert classify_heuristic("The migration completed successfully") == "event"

    def test_insight_keywords(self):
        assert classify_heuristic("We learned that MCP reliability is 50-80%") == "insight"
        assert classify_heuristic("Key takeaway from the analysis") == "insight"

    def test_goal_keywords(self):
        assert classify_heuristic("The main goal for this quarter is shipping v2") == "goal"

    def test_constraint_keywords(self):
        assert classify_heuristic("Dispatch cannot run parallel sessions") == "constraint"

    def test_correction_keywords(self):
        assert classify_heuristic("Correction: the port is 3500, not 3000") == "correction"

    def test_no_match_returns_none(self):
        assert classify_heuristic("The quick brown fox") is None
        assert classify_heuristic("") is None

    def test_case_insensitive(self):
        assert classify_heuristic("DECIDED to use Graphiti") == "decision"


class TestClassifyMemory:
    """Tests for the full classification pipeline."""

    def test_heuristic_match(self):
        assert classify_memory("We decided to fork Graphiti") == "decision"

    def test_no_match_defaults_to_fact(self):
        assert classify_memory("The sky is blue") == "fact"

    def test_no_llm_by_default(self):
        """Should not call LLM unless explicitly requested."""
        result = classify_memory("Some ambiguous text about things", use_llm=False)
        assert result == "fact"

    def test_all_keyword_maps_have_valid_types(self):
        """Every keyword map entry should reference a valid MEMORY_TYPES key."""
        for mem_type in _KEYWORD_MAP:
            assert mem_type in MEMORY_TYPES, f"Keyword map references unknown type: {mem_type}"
