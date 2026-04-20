"""Orphan Page detection.

A *graph-orphan* is a Page with zero inbound typed edges (any of the
8 types from ``schema_v2.TYPED_EDGES``). ``EVIDENCED_BY`` does NOT
count — a Page can have a rich timeline but still be graph-orphaned
if nothing points to it through a semantic relation.

Returned grouped by ``domain`` so callers can spot whole categories
where the graph is sparse.

CLI
---
    python -m jarvis_memory.orphans                 # grouped by domain, text
    python -m jarvis_memory.orphans --domain person # filter to one domain
    python -m jarvis_memory.orphans --json          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional

from .pages import Page
from .schema_v2 import PAGE_LABEL, TYPED_EDGES

logger = logging.getLogger(__name__)


def find_orphans(
    domain: Optional[str] = None,
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> dict[str, list[Page]]:
    """Return Pages grouped by domain with zero inbound typed edges.

    Args:
        domain: Optional domain filter. When set, result contains at most
            that one key.
        driver: Neo4j driver (required if ``tx`` not given).
        tx: An existing Neo4j transaction to run in.
        label: Page label (defaults to the canonical ``:Page``; tests
            may override to ``:RunTwoTestPage``).

    Returns:
        ``{domain: [Page, ...]}`` — empty dict when nothing orphaned.
        Pages whose domain prop is empty bucket under the empty string.
    """
    # Build the edge-type match list. We explicitly exclude EVIDENCED_BY
    # so a Page's own timeline doesn't disqualify it from being orphan.
    # ``NOT EXISTS`` with a typed-label alternation.
    typed_alt = "|".join(TYPED_EDGES)

    params: dict[str, Any] = {}
    where_clauses: list[str] = [
        f"NOT EXISTS {{ MATCH ()-[:{typed_alt}]->(p) }}",
    ]
    if domain:
        where_clauses.append("p.domain = $domain")
        params["domain"] = domain

    where = " AND ".join(where_clauses)
    query = f"""
        MATCH (p:{label})
        WHERE {where}
        RETURN p
        ORDER BY p.domain ASC, p.slug ASC
    """

    def _collect(runner) -> list[Page]:
        rows = runner.run(query, **params)
        return [Page.from_record(dict(r["p"])) for r in rows]

    try:
        if tx is not None:
            pages = _collect(tx)
        else:
            if driver is None:
                raise ValueError("find_orphans requires driver or tx")
            with driver.session() as sess:
                pages = _collect(sess)
    except Exception as e:
        logger.error(f"find_orphans failed: {e}")
        return {}

    grouped: dict[str, list[Page]] = {}
    for p in pages:
        grouped.setdefault(p.domain or "", []).append(p)
    return grouped


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Pages with zero inbound typed edges.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Filter to a single domain (e.g., 'person', 'company').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Machine-readable output.",
    )
    return parser.parse_args()


def _format_text(grouped: dict[str, list[Page]]) -> str:
    if not grouped:
        return "orphans: 0"
    lines = []
    total = sum(len(v) for v in grouped.values())
    lines.append(f"orphans: {total}")
    for domain in sorted(grouped.keys()):
        pages = grouped[domain]
        lines.append(f"  [{domain or '(none)'}] {len(pages)}")
        for p in pages:
            lines.append(f"    - {p.slug}")
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
        grouped = find_orphans(domain=args.domain, driver=driver)
    finally:
        driver.close()

    if args.json:
        # JSON: [{"domain": "...", "slug": "...", "compiled_truth": "...", ...}, ...]
        arr: list[dict[str, Any]] = []
        for domain, pages in sorted(grouped.items()):
            for p in pages:
                arr.append(p.to_dict())
        print(json.dumps(arr, indent=2, default=str))
    else:
        print(_format_text(grouped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
