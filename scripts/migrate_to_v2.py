"""Apply the v2 entity-layer schema to Neo4j.

Idempotent. Safe to run twice.

Usage
-----
    python scripts/migrate_to_v2.py                  # apply
    python scripts/migrate_to_v2.py --dry-run        # print plan, no writes
    python scripts/migrate_to_v2.py --rollback       # drop v2 schema
    python scripts/migrate_to_v2.py --json           # machine-readable output

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
    """Load ``.env`` from the repo root into ``os.environ`` before the
    migrator reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD via
    :mod:`jarvis_memory.config`.

    Without this, a user who populated ``.env`` via ``client-install.sh``
    still hits the config defaults (``bolt://localhost:7687``) when this
    script runs as a subprocess that never sourced ``.env`` in its shell.
    Uses ``setdefault`` so parent-shell exports win.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the Run 2 entity-layer schema to Neo4j.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned statements; do not write.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Drop the v2 schema (DROP CONSTRAINT / DROP INDEX).",
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


def _introspect(driver) -> tuple[set[str], set[str]]:
    """Return ``(constraints, indexes)`` known to the Neo4j instance."""
    constraints: set[str] = set()
    indexes: set[str] = set()
    with driver.session() as sess:
        for rec in sess.run("SHOW CONSTRAINTS YIELD name"):
            constraints.add(rec["name"])
        for rec in sess.run("SHOW INDEXES YIELD name"):
            indexes.add(rec["name"])
    return constraints, indexes


def _apply(driver, statements: list[str]) -> list[dict[str, Any]]:
    """Run each statement; collect per-statement status."""
    results: list[dict[str, Any]] = []
    with driver.session() as sess:
        for stmt in statements:
            try:
                sess.run(stmt)
                results.append({"statement": stmt, "status": "ok"})
            except Exception as e:  # noqa: BLE001 — report all failures
                results.append(
                    {
                        "statement": stmt,
                        "status": "error",
                        "error": str(e),
                    }
                )
    return results


def _emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    # Human-readable fallback.
    action = payload.get("action", "")
    print(f"migrate_to_v2: {action}")
    planned = payload.get("planned", [])
    applied = payload.get("applied", [])
    if "planned_count" in payload:
        print(f"  planned: {payload['planned_count']} statement(s)")
    for stmt in planned:
        print(f"    PLAN: {stmt}")
    for res in applied:
        marker = "OK" if res.get("status") == "ok" else "FAIL"
        print(f"    [{marker}] {res['statement'][:100]}")
        if res.get("error"):
            print(f"      error: {res['error']}")
    if payload.get("note"):
        print(f"  note: {payload['note']}")


def main() -> int:
    args = _parse_args()
    if args.dry_run and args.rollback:
        print("error: --dry-run and --rollback are mutually exclusive", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    # Lazy import so the module is importable for dry-run argparse tests
    # without a live Neo4j driver. We still need it for real runs.
    try:
        from neo4j import GraphDatabase  # type: ignore

        from jarvis_memory import schema_v2
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
            # Apply rollback unconditionally; each statement is IF EXISTS.
            applied = _apply(driver, schema_v2.ROLLBACK_STATEMENTS)
            payload: dict[str, Any] = {
                "action": "rollback",
                "planned_count": len(schema_v2.ROLLBACK_STATEMENTS),
                "planned": schema_v2.ROLLBACK_STATEMENTS,
                "applied": applied,
            }
            _emit(payload, args.json)
            return 0 if all(r["status"] == "ok" for r in applied) else 1

        constraints, indexes = _introspect(driver)
        planned = schema_v2.planned_changes(constraints, indexes)

        if args.dry_run:
            if not planned:
                payload = {
                    "action": "dry-run",
                    "planned_count": 0,
                    "planned": [],
                    "note": "0 changes needed — v2 schema already present",
                }
            else:
                payload = {
                    "action": "dry-run",
                    "planned_count": len(planned),
                    "planned": planned,
                }
            _emit(payload, args.json)
            return 0

        if not planned:
            payload = {
                "action": "apply",
                "planned_count": 0,
                "planned": [],
                "applied": [],
                "note": "0 changes needed — v2 schema already present",
            }
            _emit(payload, args.json)
            return 0

        applied = _apply(driver, planned)
        payload = {
            "action": "apply",
            "planned_count": len(planned),
            "planned": planned,
            "applied": applied,
        }
        _emit(payload, args.json)
        return 0 if all(r["status"] == "ok" for r in applied) else 1
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
