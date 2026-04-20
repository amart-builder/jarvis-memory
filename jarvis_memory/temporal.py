"""Temporal fact management — validity windows and timelines.

Facts have lifespans. "Max works on Orion" may have been true in 2025
but ended in 2026-02. This module adds valid_from/valid_to properties
to Neo4j nodes and provides temporal query capabilities.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


def set_validity(
    driver,
    memory_id: str,
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
) -> dict[str, Any]:
    """Set temporal validity bounds on a memory node.

    Args:
        driver: Neo4j driver instance.
        memory_id: UUID of the memory node.
        valid_from: ISO datetime string when the fact became true.
                    If None, uses the node's created_at.
        valid_to: ISO datetime string when the fact stopped being true.
                  If None, means the fact is still current.

    Returns:
        Dict with success status and updated values.
    """
    try:
        with driver.session() as db:
            set_clauses = []
            params: dict[str, Any] = {"uuid": memory_id}

            if valid_from is not None:
                set_clauses.append("n.valid_from = datetime($valid_from)")
                params["valid_from"] = valid_from
            else:
                # Default valid_from to created_at if not already set
                set_clauses.append(
                    "n.valid_from = CASE WHEN n.valid_from IS NULL "
                    "THEN coalesce(n.created_at, datetime()) ELSE n.valid_from END"
                )

            if valid_to is not None:
                set_clauses.append("n.valid_to = datetime($valid_to)")
                params["valid_to"] = valid_to

            set_str = ", ".join(set_clauses)
            result = db.run(
                f"""
                MATCH (n) WHERE n.uuid = $uuid
                SET {set_str}
                RETURN n.uuid AS uuid, n.valid_from AS valid_from, n.valid_to AS valid_to
                """,
                **params,
            )
            record = result.single()
            if record is None:
                return {"success": False, "error": f"Memory {memory_id} not found"}

            return {
                "success": True,
                "memory_id": memory_id,
                "valid_from": str(record["valid_from"]) if record["valid_from"] else None,
                "valid_to": str(record["valid_to"]) if record["valid_to"] else None,
            }

    except Exception as e:
        logger.error(f"Failed to set validity on {memory_id}: {e}")
        return {"success": False, "error": str(e)}


def get_timeline(
    driver,
    entity: str,
    group_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get chronological fact history for an entity or topic.

    Searches across node names, content, and summaries for mentions
    of the entity, ordered by valid_from (or created_at if not set).

    Args:
        driver: Neo4j driver instance.
        entity: Entity name or topic to search for.
        group_id: Optional project filter.
        limit: Maximum results.

    Returns:
        List of dicts with uuid, content, valid_from, valid_to, memory_type.
    """
    try:
        with driver.session() as db:
            where_clauses = [
                "(toLower(coalesce(n.name, '')) CONTAINS toLower($entity) "
                "OR toLower(coalesce(n.content, '')) CONTAINS toLower($entity) "
                "OR toLower(coalesce(n.summary, '')) CONTAINS toLower($entity) "
                "OR toLower(coalesce(n.fact, '')) CONTAINS toLower($entity))"
            ]
            params: dict[str, Any] = {"entity": entity, "limit": limit}

            if group_id:
                where_clauses.append("n.group_id = $group_id")
                params["group_id"] = group_id

            where_str = " AND ".join(where_clauses)

            result = db.run(
                f"""
                MATCH (n)
                WHERE {where_str}
                  AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed', 'outdated']
                RETURN n.uuid AS uuid,
                       coalesce(n.content, n.name, n.summary, '') AS content,
                       n.valid_from AS valid_from,
                       n.valid_to AS valid_to,
                       n.created_at AS created_at,
                       coalesce(n.memory_type, n.episode_type, 'fact') AS memory_type,
                       n.lifecycle_status AS status
                ORDER BY coalesce(n.valid_from, n.created_at) ASC
                LIMIT $limit
                """,
                **params,
            )

            timeline = []
            for record in result:
                entry = {
                    "uuid": record["uuid"],
                    "content": record["content"][:200],  # Truncate for readability
                    "memory_type": record["memory_type"],
                    "status": record["status"] or "active",
                    "valid_from": str(record["valid_from"]) if record["valid_from"] else str(record["created_at"]) if record["created_at"] else None,
                    "valid_to": str(record["valid_to"]) if record["valid_to"] else None,
                    "is_current": record["valid_to"] is None,
                }
                timeline.append(entry)

            return timeline

    except Exception as e:
        logger.error(f"Failed to get timeline for '{entity}': {e}")
        return []


def filter_by_date(
    results: list[dict[str, Any]],
    as_of: str,
) -> list[dict[str, Any]]:
    """Filter a result set to facts valid at a specific date.

    Args:
        results: List of result dicts (must have valid_from/valid_to keys
                 or created_at in their metadata).
        as_of: ISO date string to filter by.

    Returns:
        Filtered list containing only facts valid at the given date.
    """
    try:
        target = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid as_of date: {as_of}, returning unfiltered")
        return results

    filtered = []
    for r in results:
        # Try to get validity bounds from various locations
        vf = r.get("valid_from") or r.get("created_at")
        vt = r.get("valid_to")

        # Parse valid_from
        if vf:
            try:
                vf_dt = datetime.fromisoformat(str(vf).replace("Z", "+00:00"))
                if vf_dt > target:
                    continue  # Fact didn't exist yet
            except (ValueError, TypeError):
                pass  # Can't parse, include by default

        # Parse valid_to
        if vt:
            try:
                vt_dt = datetime.fromisoformat(str(vt).replace("Z", "+00:00"))
                if vt_dt < target:
                    continue  # Fact had already ended
            except (ValueError, TypeError):
                pass  # Can't parse, include by default

        filtered.append(r)

    return filtered
