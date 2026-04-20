"""Tests for schema_v2 module — pure-data assertions.

The migration script runs against a live Neo4j instance. These tests
assert only that the planned statement list is shaped correctly and
that the idempotency helpers work. Neither requires Neo4j.
"""
from __future__ import annotations

import pytest

from jarvis_memory import schema_v2


class TestApplyStatements:
    """APPLY_STATEMENTS contains the expected DDL."""

    def test_statements_list_is_non_empty(self):
        assert schema_v2.APPLY_STATEMENTS, "APPLY_STATEMENTS must be non-empty"

    def test_statements_list_length(self):
        # v1 of Run 2: uniqueness + fulltext = 2 statements.
        assert len(schema_v2.APPLY_STATEMENTS) == 2

    def test_uniqueness_constraint_named(self):
        joined = "\n".join(schema_v2.APPLY_STATEMENTS)
        assert "page_slug_unique" in joined
        assert "CREATE CONSTRAINT" in joined
        assert "REQUIRE p.slug IS UNIQUE" in joined

    def test_fulltext_index_named(self):
        joined = "\n".join(schema_v2.APPLY_STATEMENTS)
        assert "page_compiled_truth_fulltext" in joined
        assert "CREATE FULLTEXT INDEX" in joined
        assert "compiled_truth" in joined

    def test_page_label_in_every_statement(self):
        for stmt in schema_v2.APPLY_STATEMENTS:
            assert ":Page" in stmt, f"missing :Page in {stmt!r}"


class TestRollbackStatements:
    """ROLLBACK_STATEMENTS undo the APPLY_STATEMENTS."""

    def test_rollback_is_non_empty(self):
        assert schema_v2.ROLLBACK_STATEMENTS

    def test_rollback_length_matches_apply(self):
        # One drop per create (constraint + index).
        assert len(schema_v2.ROLLBACK_STATEMENTS) == len(schema_v2.APPLY_STATEMENTS)

    def test_rollback_drops_constraint(self):
        joined = "\n".join(schema_v2.ROLLBACK_STATEMENTS)
        assert "DROP CONSTRAINT" in joined
        assert "page_slug_unique" in joined

    def test_rollback_drops_index(self):
        joined = "\n".join(schema_v2.ROLLBACK_STATEMENTS)
        assert "DROP INDEX" in joined
        assert "page_compiled_truth_fulltext" in joined

    def test_rollback_uses_if_exists(self):
        for stmt in schema_v2.ROLLBACK_STATEMENTS:
            assert "IF EXISTS" in stmt, f"rollback must be idempotent: {stmt!r}"


class TestIdempotencyDetection:
    """``is_migration_complete`` + ``planned_changes`` short-circuit re-runs."""

    def test_migration_incomplete_when_empty(self):
        assert schema_v2.is_migration_complete(set(), set()) is False

    def test_migration_incomplete_missing_index(self):
        constraints = {schema_v2.PAGE_SLUG_UNIQUE}
        indexes: set[str] = set()
        assert schema_v2.is_migration_complete(constraints, indexes) is False

    def test_migration_incomplete_missing_constraint(self):
        constraints: set[str] = set()
        indexes = {schema_v2.PAGE_COMPILED_TRUTH_FTS}
        assert schema_v2.is_migration_complete(constraints, indexes) is False

    def test_migration_complete_with_both(self):
        constraints = {schema_v2.PAGE_SLUG_UNIQUE}
        indexes = {schema_v2.PAGE_COMPILED_TRUTH_FTS}
        assert schema_v2.is_migration_complete(constraints, indexes) is True

    def test_planned_changes_all_when_empty(self):
        planned = schema_v2.planned_changes(set(), set())
        assert len(planned) == 2

    def test_planned_changes_none_when_complete(self):
        constraints = {schema_v2.PAGE_SLUG_UNIQUE}
        indexes = {schema_v2.PAGE_COMPILED_TRUTH_FTS}
        planned = schema_v2.planned_changes(constraints, indexes)
        assert planned == []

    def test_planned_changes_partial(self):
        constraints = {schema_v2.PAGE_SLUG_UNIQUE}
        indexes: set[str] = set()
        planned = schema_v2.planned_changes(constraints, indexes)
        assert len(planned) == 1
        assert "FULLTEXT INDEX" in planned[0]


class TestEdgeVocabulary:
    """The reserved typed-edge names are stable."""

    def test_eight_typed_edges(self):
        assert len(schema_v2.TYPED_EDGES) == 8

    @pytest.mark.parametrize(
        "name",
        [
            "ATTENDED",
            "WORKS_AT",
            "INVESTED_IN",
            "FOUNDED",
            "ADVISES",
            "DECIDED_ON",
            "MENTIONS",
            "REFERS_TO",
        ],
    )
    def test_expected_edge_names(self, name):
        assert name in schema_v2.TYPED_EDGES

    def test_evidence_edge_name(self):
        assert schema_v2.EVIDENCE_EDGE == "EVIDENCED_BY"

    def test_typed_edges_all_upper(self):
        for edge in schema_v2.TYPED_EDGES:
            assert edge == edge.upper(), f"edge label must be upper-case: {edge!r}"
