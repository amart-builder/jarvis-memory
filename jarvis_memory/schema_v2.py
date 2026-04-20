"""Schema v2 — Page node + typed-edge knowledge graph schema.

This module holds the Neo4j schema additions for Run 2 (Entity Layer).
It is intentionally a pure data module — no DB connection, no side
effects — so tests can assert the planned statements without touching
Neo4j, and `scripts/migrate_to_v2.py` can consume the same list and
apply it idempotently.

Schema added
------------
* **Node label** ``:Page``
  - ``slug`` (unique) — canonical entity identifier (e.g. ``"foundry"``).
  - ``domain`` — category bucket (``person``, ``company``, ``project``,
    ``concept``, ``system``, ``topic`` — free-form for v1).
  - ``compiled_truth`` — current agent-authored summary (``<= 2000`` chars).
  - ``created_at`` / ``updated_at`` — ISO timestamps.

* **Uniqueness constraint** on ``Page.slug``: ``page_slug_unique``.

* **Full-text index** on ``Page.compiled_truth``: ``page_compiled_truth_fulltext``.

* **Edge types** (pure relationship labels — Neo4j doesn't require explicit
  creation, but we reserve names here so edge creation in ``graph.py`` is
  type-checked against a canonical list):

  Typed edges (``Episode -> Page`` or ``Page -> Page``):
    ``ATTENDED``, ``WORKS_AT``, ``INVESTED_IN``, ``FOUNDED``, ``ADVISES``,
    ``DECIDED_ON``, ``MENTIONS``, ``REFERS_TO``.

  Evidence edge (``Page -> Episode``):
    ``EVIDENCED_BY`` (append-only timeline).

The Cypher statements below are the exact DDL we want applied. Tests
assert the list shape; the migration script applies them idempotently.
"""
from __future__ import annotations

from typing import Final

# ── Node label name (for isolated-namespace testing) ────────────────────
PAGE_LABEL: Final[str] = "Page"

# ── Constraint + index names (stable, referenced by migration + doctor) ─
PAGE_SLUG_UNIQUE: Final[str] = "page_slug_unique"
PAGE_COMPILED_TRUTH_FTS: Final[str] = "page_compiled_truth_fulltext"

# ── Edge type vocabulary ─────────────────────────────────────────────────
TYPED_EDGES: Final[tuple[str, ...]] = (
    "ATTENDED",
    "WORKS_AT",
    "INVESTED_IN",
    "FOUNDED",
    "ADVISES",
    "DECIDED_ON",
    "MENTIONS",
    "REFERS_TO",
)

EVIDENCE_EDGE: Final[str] = "EVIDENCED_BY"

# ── Apply statements (idempotent DDL) ────────────────────────────────────
# Each string is a single Cypher statement. ``IF NOT EXISTS`` makes the
# constraint creation a no-op when re-run. The full-text index creation
# uses a safety pattern: check SHOW INDEXES first via the migration
# script (Neo4j 5 doesn't accept IF NOT EXISTS on CREATE FULLTEXT INDEX
# in all minor versions, so we guard from the script side instead).
APPLY_STATEMENTS: Final[list[str]] = [
    # Uniqueness constraint on Page.slug
    (
        f"CREATE CONSTRAINT {PAGE_SLUG_UNIQUE} IF NOT EXISTS "
        f"FOR (p:{PAGE_LABEL}) REQUIRE p.slug IS UNIQUE"
    ),
    # Full-text index on Page.compiled_truth
    (
        f"CREATE FULLTEXT INDEX {PAGE_COMPILED_TRUTH_FTS} IF NOT EXISTS "
        f"FOR (p:{PAGE_LABEL}) ON EACH [p.compiled_truth]"
    ),
]

# ── Rollback statements (drop only the v2 schema; episodes untouched) ───
ROLLBACK_STATEMENTS: Final[list[str]] = [
    f"DROP INDEX {PAGE_COMPILED_TRUTH_FTS} IF EXISTS",
    f"DROP CONSTRAINT {PAGE_SLUG_UNIQUE} IF EXISTS",
]


# ── Expected names (for doctor + idempotency detection) ─────────────────
EXPECTED_CONSTRAINTS: Final[frozenset[str]] = frozenset({PAGE_SLUG_UNIQUE})
EXPECTED_INDEXES: Final[frozenset[str]] = frozenset({PAGE_COMPILED_TRUTH_FTS})


def is_migration_complete(
    existing_constraints: set[str],
    existing_indexes: set[str],
) -> bool:
    """Return True iff all v2 constraints + indexes already exist.

    Pure helper: takes sets of names (from ``SHOW CONSTRAINTS`` /
    ``SHOW INDEXES``) and checks whether everything we'd add is already
    there. Used by the migration script to decide whether to print
    "0 changes needed" on re-run.
    """
    return (
        EXPECTED_CONSTRAINTS.issubset(existing_constraints)
        and EXPECTED_INDEXES.issubset(existing_indexes)
    )


def planned_changes(
    existing_constraints: set[str],
    existing_indexes: set[str],
) -> list[str]:
    """Return the subset of APPLY_STATEMENTS that would actually do work.

    Used by ``--dry-run`` to show just the delta.
    """
    planned: list[str] = []
    if PAGE_SLUG_UNIQUE not in existing_constraints:
        planned.append(APPLY_STATEMENTS[0])
    if PAGE_COMPILED_TRUTH_FTS not in existing_indexes:
        planned.append(APPLY_STATEMENTS[1])
    return planned
