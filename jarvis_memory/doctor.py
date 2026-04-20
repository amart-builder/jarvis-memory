"""Health checks for the Run 2 entity layer.

``run_health_checks()`` returns a dict keyed by check name. Each check
returns ``{status: PASS|WARN|FAIL, detail, fix_hint}``.

Checks
------
``schema_v2_present``
    The expected constraints + full-text index exist in Neo4j.

``page_completeness``
    Percentage of Pages whose ``compiled_truth`` is non-empty. WARN
    below 25% (low population by policy — authors haven't written
    summaries yet).

``edge_validity``
    No dangling ``EVIDENCED_BY`` edges pointing at episodes that no
    longer exist. (Episodes are never deleted in prod, so this should
    be FAIL only on true corruption.)

``orphan_count_reasonable``
    ``find_orphans`` returns < 10% of total pages. WARN above 10%,
    FAIL above 25%.

CLI
---
    python -m jarvis_memory.doctor
    python -m jarvis_memory.doctor --json
    python -m jarvis_memory.doctor --fast   # skip the expensive orphan query
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional

from . import schema_v2
from .pages import PAGE_LABEL, count_pages
from .orphans import find_orphans

logger = logging.getLogger(__name__)

# Check-status constants.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _status(check: str, status: str, detail: str, fix_hint: str = "") -> dict:
    return {
        "check": check,
        "status": status,
        "detail": detail,
        "fix_hint": fix_hint,
    }


def check_schema_v2_present(
    *,
    driver=None,
    tx=None,
) -> dict:
    """All expected constraints + indexes exist."""
    try:
        if tx is not None:
            constraints = {r["name"] for r in tx.run("SHOW CONSTRAINTS YIELD name")}
            indexes = {r["name"] for r in tx.run("SHOW INDEXES YIELD name")}
        else:
            if driver is None:
                return _status(
                    "schema_v2_present",
                    FAIL,
                    "neither driver nor tx provided",
                    "pass driver=... or tx=...",
                )
            with driver.session() as sess:
                constraints = {r["name"] for r in sess.run("SHOW CONSTRAINTS YIELD name")}
                indexes = {r["name"] for r in sess.run("SHOW INDEXES YIELD name")}
    except Exception as e:
        return _status(
            "schema_v2_present",
            FAIL,
            f"introspection failed: {e}",
            "verify Neo4j is reachable and the user has SHOW CONSTRAINTS rights",
        )

    missing_constraints = schema_v2.EXPECTED_CONSTRAINTS - constraints
    missing_indexes = schema_v2.EXPECTED_INDEXES - indexes

    if not missing_constraints and not missing_indexes:
        return _status(
            "schema_v2_present",
            PASS,
            f"all {len(schema_v2.EXPECTED_CONSTRAINTS)} constraints + "
            f"{len(schema_v2.EXPECTED_INDEXES)} indexes present",
        )

    return _status(
        "schema_v2_present",
        FAIL,
        f"missing_constraints={sorted(missing_constraints)} "
        f"missing_indexes={sorted(missing_indexes)}",
        "run `python scripts/migrate_to_v2.py`",
    )


def check_page_completeness(
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
    warn_threshold: float = 0.25,
) -> dict:
    """Fraction of Pages with non-empty compiled_truth."""
    try:
        runner = tx if tx is not None else None
        if runner is None:
            if driver is None:
                return _status(
                    "page_completeness",
                    FAIL,
                    "neither driver nor tx provided",
                )
            with driver.session() as sess:
                total = sess.run(
                    f"MATCH (p:{label}) RETURN count(p) AS n"
                ).single()["n"]
                filled = sess.run(
                    f"MATCH (p:{label}) WHERE p.compiled_truth IS NOT NULL "
                    "AND p.compiled_truth <> '' RETURN count(p) AS n"
                ).single()["n"]
        else:
            total = runner.run(
                f"MATCH (p:{label}) RETURN count(p) AS n"
            ).single()["n"]
            filled = runner.run(
                f"MATCH (p:{label}) WHERE p.compiled_truth IS NOT NULL "
                "AND p.compiled_truth <> '' RETURN count(p) AS n"
            ).single()["n"]
    except Exception as e:
        return _status(
            "page_completeness",
            FAIL,
            f"query failed: {e}",
        )

    total = int(total or 0)
    filled = int(filled or 0)
    if total == 0:
        return _status(
            "page_completeness",
            PASS,
            "0 pages — nothing to measure",
        )
    ratio = filled / total
    detail = f"{filled}/{total} pages have compiled_truth ({ratio:.0%})"
    if ratio >= warn_threshold:
        return _status("page_completeness", PASS, detail)
    return _status(
        "page_completeness",
        WARN,
        detail,
        "consider authoring compiled_truth for under-populated pages",
    )


def check_edge_validity(
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> dict:
    """No dangling EVIDENCED_BY relationships — a referential-integrity check.

    EVIDENCED_BY edges always go Page → Episode; if the Episode doesn't
    exist anymore the edge is dangling. We count edges whose target
    doesn't carry the Episode label.
    """
    query = f"""
        MATCH (p:{label})-[r:EVIDENCED_BY]->(e)
        WHERE NOT (e:Episode)
        RETURN count(r) AS n
    """
    try:
        if tx is not None:
            row = tx.run(query).single()
            bad = int(row["n"]) if row else 0
        else:
            if driver is None:
                return _status(
                    "edge_validity",
                    FAIL,
                    "neither driver nor tx provided",
                )
            with driver.session() as sess:
                row = sess.run(query).single()
                bad = int(row["n"]) if row else 0
    except Exception as e:
        return _status(
            "edge_validity",
            FAIL,
            f"query failed: {e}",
        )

    if bad == 0:
        return _status(
            "edge_validity",
            PASS,
            "0 dangling EVIDENCED_BY edges",
        )
    return _status(
        "edge_validity",
        FAIL,
        f"{bad} EVIDENCED_BY edges point at non-Episode nodes",
        "investigate episode deletions; run integrity sweep",
    )


def check_orphan_count_reasonable(
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
    warn_ratio: float = 0.10,
    fail_ratio: float = 0.25,
) -> dict:
    """Orphan count as a fraction of total Pages."""
    try:
        total = count_pages(driver=driver, tx=tx, label=label)
        grouped = find_orphans(driver=driver, tx=tx, label=label)
        orphan_count = sum(len(v) for v in grouped.values())
    except Exception as e:
        return _status(
            "orphan_count_reasonable",
            FAIL,
            f"query failed: {e}",
        )

    if total == 0:
        return _status(
            "orphan_count_reasonable",
            PASS,
            "0 pages — nothing to measure",
        )
    ratio = orphan_count / total
    detail = f"{orphan_count}/{total} pages orphan ({ratio:.0%})"
    if ratio >= fail_ratio:
        return _status(
            "orphan_count_reasonable",
            FAIL,
            detail,
            "run `python -m jarvis_memory.orphans` and resolve the backlog",
        )
    if ratio >= warn_ratio:
        return _status(
            "orphan_count_reasonable",
            WARN,
            detail,
            "review the orphan list and decide which pages need typed edges",
        )
    return _status("orphan_count_reasonable", PASS, detail)


# ── Orchestrator ─────────────────────────────────────────────────────


def run_health_checks(
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
    fast: bool = False,
) -> dict[str, Any]:
    """Run all checks, return a combined dict.

    Args:
        driver: Neo4j driver.
        tx: In-progress transaction.
        label: Page label (override for tests).
        fast: Skip orphan_count check (the expensive one).

    Returns:
        ``{
            "overall": PASS|WARN|FAIL,  # worst status wins
            "checks": { name: check_result, ... },
            "summary": {"pass": N, "warn": N, "fail": N},
        }``
    """
    checks: dict[str, dict] = {}
    checks["schema_v2_present"] = check_schema_v2_present(driver=driver, tx=tx)
    checks["page_completeness"] = check_page_completeness(driver=driver, tx=tx, label=label)
    checks["edge_validity"] = check_edge_validity(driver=driver, tx=tx, label=label)
    if not fast:
        checks["orphan_count_reasonable"] = check_orphan_count_reasonable(
            driver=driver, tx=tx, label=label
        )

    summary = {"pass": 0, "warn": 0, "fail": 0}
    for c in checks.values():
        summary[c["status"].lower()] += 1

    if summary["fail"] > 0:
        overall = FAIL
    elif summary["warn"] > 0:
        overall = WARN
    else:
        overall = PASS

    return {
        "overall": overall,
        "checks": checks,
        "summary": summary,
    }


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health-check the entity layer.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument("--fast", action="store_true", help="Skip the orphan query.")
    return parser.parse_args()


def _format_text(report: dict[str, Any]) -> str:
    lines = [f"overall: {report['overall']}"]
    lines.append(
        f"  pass={report['summary']['pass']} "
        f"warn={report['summary']['warn']} "
        f"fail={report['summary']['fail']}"
    )
    for name, c in report["checks"].items():
        lines.append(f"  [{c['status']}] {name}: {c['detail']}")
        if c.get("fix_hint"):
            lines.append(f"      hint: {c['fix_hint']}")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    try:
        from neo4j import GraphDatabase  # type: ignore

        from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    except Exception as e:  # pragma: no cover
        print(f"error: import failed: {e}", file=sys.stderr)
        return 1

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"error: neo4j unreachable at {NEO4J_URI}: {e}", file=sys.stderr)
        return 1

    try:
        report = run_health_checks(driver=driver, fast=args.fast)
    finally:
        driver.close()

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
