"""Migrate Neo4j to the bi-temporal data model (A3 — v1.1 roadmap).

Adds two new properties to fact-bearing nodes:

* ``t_created`` (datetime) — when WE first recorded this fact (ingestion
  time start). Distinct from event time (``valid_from`` / ``valid_to``).
  Backfills to ``coalesce(n.created_at, datetime())`` so existing data
  joins the bi-temporal model immediately.
* ``t_expired`` (datetime, nullable) — when WE stopped trusting this
  fact (ingestion time end). Stays NULL for active facts; set by
  ``contradict_memory`` / ``supersede_memory`` going forward (see A3.2).

Why the second timeline matters: the existing ``valid_from`` /
``valid_to`` model says "the fact was true in the world from X to Y."
That's *event* time. The chief-of-staff workflow ("what did we
believe on date X?") needs the orthogonal *ingestion* time.
Bi-temporal preserves both — we never overwrite history, we *expire*
old beliefs. Lossless audit trail.

Targets: ``:Episode``, ``:Page``, ``:Episodic`` (legacy Graphiti),
``:Entity`` (legacy Graphiti). Skips ``:Session`` / ``:Snapshot`` /
``:Community`` / ``:Saga`` — those are operational metadata, not
fact-bearing claims.

Usage
-----
    python scripts/migrate_to_bitemporal.py                  # apply
    python scripts/migrate_to_bitemporal.py --dry-run        # plan, no writes
    python scripts/migrate_to_bitemporal.py --rollback       # remove t_created / t_expired
    python scripts/migrate_to_bitemporal.py --json           # machine-readable

Idempotent. Safe to run repeatedly — only sets ``t_created`` on nodes
where it's currently NULL. ``t_expired`` is never set during migration
(it gets set later by lifecycle events).

Exit codes
----------
    0 — success (applied / nothing to apply / rollback)
    1 — connection or runtime error
    2 — bad arguments
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


def _load_env_file() -> None:
    """Load ``.env`` from the repo root before reading config.

    Same pattern as ``migrate_to_v2.py`` — without this, a subprocess
    that didn't source ``.env`` hits the localhost defaults.
    """
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), val)


_load_env_file()


# Labels that carry fact claims and therefore need bi-temporal tracking.
# Ordered by expected node count descending; doesn't matter for correctness.
FACT_LABELS = ("Episode", "Page", "Entity", "Episodic")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the bi-temporal data model to Neo4j.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes; do not write.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Remove t_created and t_expired properties from all fact-bearing nodes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout.",
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="Neo4j bolt URI (defaults to NEO4J_URI env).",
    )
    return parser.parse_args()


def _count_pending(driver) -> dict[str, int]:
    """Count fact-bearing nodes per label that still lack t_created."""
    pending: dict[str, int] = {}
    with driver.session() as sess:
        for label in FACT_LABELS:
            rec = sess.run(
                f"MATCH (n:{label}) WHERE n.t_created IS NULL RETURN count(n) AS c"
            ).single()
            pending[label] = int(rec["c"]) if rec else 0
    return pending


def _count_present(driver) -> dict[str, int]:
    """Count nodes per label that already have t_created (post-migration view)."""
    present: dict[str, int] = {}
    with driver.session() as sess:
        for label in FACT_LABELS:
            rec = sess.run(
                f"MATCH (n:{label}) WHERE n.t_created IS NOT NULL RETURN count(n) AS c"
            ).single()
            present[label] = int(rec["c"]) if rec else 0
    return present


def _apply_migration(driver) -> dict[str, int]:
    """Set ``t_created = coalesce(created_at, datetime())`` on each label.

    ``t_expired`` is intentionally *not* set — it stays NULL for active
    facts. Returns per-label counts of nodes touched.
    """
    migrated: dict[str, int] = {}
    with driver.session() as sess:
        for label in FACT_LABELS:
            rec = sess.run(
                f"""
                MATCH (n:{label})
                WHERE n.t_created IS NULL
                SET n.t_created = coalesce(n.created_at, datetime())
                RETURN count(n) AS c
                """
            ).single()
            migrated[label] = int(rec["c"]) if rec else 0
    return migrated


def _apply_rollback(driver) -> dict[str, int]:
    """Remove ``t_created`` and ``t_expired`` from all fact-bearing nodes.

    Used to back out the migration cleanly. Each ``REMOVE`` is a no-op on
    nodes that don't have the property, so this is also idempotent.
    """
    rolled_back: dict[str, int] = {}
    with driver.session() as sess:
        for label in FACT_LABELS:
            rec = sess.run(
                f"""
                MATCH (n:{label})
                WHERE n.t_created IS NOT NULL OR n.t_expired IS NOT NULL
                REMOVE n.t_created, n.t_expired
                RETURN count(n) AS c
                """
            ).single()
            rolled_back[label] = int(rec["c"]) if rec else 0
    return rolled_back


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    action = payload.get("action", "")
    print(f"migrate_to_bitemporal: {action}")
    pending = payload.get("pending") or {}
    present = payload.get("present") or {}
    migrated = payload.get("migrated") or {}
    rolled_back = payload.get("rolled_back") or {}

    if pending:
        total = sum(pending.values())
        print(f"  pending: {total} node(s) need t_created backfilled")
        for label, count in pending.items():
            if count:
                print(f"    {label}: {count}")
    if present:
        total = sum(present.values())
        print(f"  present: {total} node(s) already have t_created")
    if migrated:
        total = sum(migrated.values())
        print(f"  migrated: {total} node(s) backfilled")
        for label, count in migrated.items():
            if count:
                print(f"    {label}: {count}")
    if rolled_back:
        total = sum(rolled_back.values())
        print(f"  rolled_back: {total} node(s) reverted")
        for label, count in rolled_back.items():
            if count:
                print(f"    {label}: {count}")
    if note := payload.get("note"):
        print(f"  note: {note}")


def main() -> int:
    args = _parse_args()
    if args.dry_run and args.rollback:
        print("error: --dry-run and --rollback are mutually exclusive", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    # Neo4j 5.x emits a NotificationCategory.UNRECOGNIZED warning the first
    # time a query references a property name the DB has never indexed
    # (``t_created`` here). That's expected — the property *literally*
    # doesn't exist yet on any node. Silence the spam so the dry-run
    # output stays readable.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        from neo4j import GraphDatabase  # type: ignore

        from jarvis_memory.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    except Exception as e:  # pragma: no cover — import failure path
        print(f"error: import failed: {e}", file=sys.stderr)
        return 1

    uri = args.uri or NEO4J_URI
    try:
        driver = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"error: neo4j unreachable at {uri}: {e}", file=sys.stderr)
        return 1

    try:
        if args.rollback:
            present_before = _count_present(driver)
            rolled_back = _apply_rollback(driver)
            payload: dict[str, Any] = {
                "action": "rollback",
                "labels": list(FACT_LABELS),
                "present_before": present_before,
                "rolled_back": rolled_back,
            }
            _emit(payload, args.json)
            return 0

        pending = _count_pending(driver)
        present = _count_present(driver)
        total_pending = sum(pending.values())

        if args.dry_run:
            payload = {
                "action": "dry-run",
                "labels": list(FACT_LABELS),
                "pending": pending,
                "present": present,
            }
            if total_pending == 0:
                payload["note"] = "0 changes needed — bi-temporal model already applied"
            _emit(payload, args.json)
            return 0

        if total_pending == 0:
            payload = {
                "action": "apply",
                "labels": list(FACT_LABELS),
                "pending": pending,
                "present": present,
                "migrated": {label: 0 for label in FACT_LABELS},
                "note": "0 changes needed — bi-temporal model already applied",
            }
            _emit(payload, args.json)
            return 0

        migrated = _apply_migration(driver)
        payload = {
            "action": "apply",
            "labels": list(FACT_LABELS),
            "pending_before": pending,
            "present_before": present,
            "migrated": migrated,
        }
        _emit(payload, args.json)
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
