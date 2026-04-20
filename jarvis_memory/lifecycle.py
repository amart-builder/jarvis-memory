"""Memory lifecycle management — 8-state machine with transition validation.

Ported from MemClawz v7, adapted for Graphiti's Neo4j-based storage.
Instead of Qdrant set_payload operations, we update node properties via Cypher.

States: active → confirmed → archived → deprecated → superseded → merged → disputed → deleted
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, STALE_THRESHOLD_DAYS

logger = logging.getLogger(__name__)

# 8 lifecycle states
LIFECYCLE_STATES: set[str] = {
    "active",       # default — actively used memory
    "confirmed",    # high confidence — accessed multiple times, still valid
    "outdated",     # potentially stale — needs validation
    "archived",     # preserved but not actively used
    "contradicted", # contradicted by newer information
    "merged",       # combined into another memory during compaction
    "superseded",   # replaced by newer version (linked)
    "deleted",      # soft-deleted, excluded from search
}

# Valid state transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    "active":       {"confirmed", "outdated", "archived", "contradicted", "merged", "superseded", "deleted"},
    "confirmed":    {"outdated", "archived", "superseded", "deleted"},
    "outdated":     {"archived", "deleted", "active"},     # can be re-validated
    "archived":     {"active", "deleted"},                  # can be restored
    "contradicted": {"deleted", "active"},                  # can be re-validated
    "merged":       {"deleted"},                            # terminal
    "superseded":   {"deleted"},                            # terminal
    "deleted":      set(),                                  # final state
}

# Default status for new memories
DEFAULT_STATUS = "active"


class MemoryLifecycle:
    """Memory lifecycle management with status tracking and transitions.

    Uses Neo4j (via Graphiti's graph) to store lifecycle state as node properties.
    Compatible with both EpisodicNode and EntityNode types.
    """

    def __init__(self, driver=None):
        """Initialize with an optional Neo4j driver.

        If no driver is provided, creates one from config.
        Pass the Graphiti client's driver for shared connection pooling.
        """
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self._owns_driver = True

    def close(self):
        """Close the Neo4j driver if we own it."""
        if self._owns_driver and self._driver:
            self._driver.close()

    def transition(
        self,
        memory_id: str,
        from_status: str,
        to_status: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Validate and execute a status transition.

        Args:
            memory_id: UUID of the memory node to transition.
            from_status: Expected current status (for optimistic concurrency).
            to_status: Target status.
            metadata: Optional additional metadata to store with the transition.

        Returns:
            True if transition succeeded, False if invalid or failed.
        """
        # Validate transition is allowed
        if from_status not in VALID_TRANSITIONS:
            logger.warning(f"Invalid from_status: {from_status}")
            return False

        if to_status not in VALID_TRANSITIONS.get(from_status, set()):
            logger.warning(f"Invalid transition: {from_status} -> {to_status}")
            return False

        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (n)
                    WHERE n.uuid = $uuid AND coalesce(n.lifecycle_status, 'active') = $from_status
                    SET n.lifecycle_status = $to_status,
                        n.lifecycle_updated_at = datetime(),
                        n.lifecycle_metadata = $metadata
                    RETURN n.uuid AS uuid
                    """,
                    uuid=memory_id,
                    from_status=from_status,
                    to_status=to_status,
                    metadata=str(metadata or {}),
                )

                record = result.single()
                if record is None:
                    logger.warning(
                        f"Memory {memory_id} not found or status mismatch "
                        f"(expected {from_status})"
                    )
                    return False

            logger.info(f"Memory {memory_id} transitioned: {from_status} -> {to_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to transition memory {memory_id}: {e}")
            return False

    def get_status(self, memory_id: str) -> str:
        """Get the current lifecycle status of a memory.

        Returns DEFAULT_STATUS if the node has no lifecycle_status property.
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (n) WHERE n.uuid = $uuid
                    RETURN coalesce(n.lifecycle_status, 'active') AS status
                    """,
                    uuid=memory_id,
                )
                record = result.single()
                if record is None:
                    logger.warning(f"Memory {memory_id} not found")
                    return DEFAULT_STATUS
                return record["status"]

        except Exception as e:
            logger.error(f"Failed to get status for {memory_id}: {e}")
            return DEFAULT_STATUS

    def bulk_check_outdated(
        self,
        threshold_days: int = STALE_THRESHOLD_DAYS,
        group_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find active memories older than threshold that may be outdated.

        Args:
            threshold_days: Days after which a memory is considered stale.
            group_id: Optional project group to filter by.

        Returns:
            List of dicts with uuid, content preview, age_days, memory_type.
        """
        threshold_date = datetime.now(timezone.utc) - timedelta(days=threshold_days)

        try:
            with self._driver.session() as session:
                query = """
                    MATCH (n)
                    WHERE coalesce(n.lifecycle_status, 'active') = 'active'
                      AND n.created_at < $threshold
                """
                params: dict[str, Any] = {"threshold": threshold_date.isoformat()}

                if group_id:
                    query += " AND n.group_id = $group_id"
                    params["group_id"] = group_id

                query += """
                    RETURN n.uuid AS uuid,
                           left(coalesce(n.content, n.name, ''), 100) AS preview,
                           duration.between(n.created_at, datetime()).days AS age_days,
                           coalesce(n.memory_type, 'unknown') AS memory_type,
                           coalesce(n.group_id, 'unknown') AS group_id
                    ORDER BY age_days DESC
                    LIMIT 500
                """

                result = session.run(query, **params)
                candidates = []
                for record in result:
                    candidates.append({
                        "uuid": record["uuid"],
                        "preview": record["preview"],
                        "age_days": record["age_days"],
                        "memory_type": record["memory_type"],
                        "group_id": record["group_id"],
                    })

            logger.info(f"Found {len(candidates)} memories older than {threshold_days} days")
            return candidates

        except Exception as e:
            logger.error(f"Failed to check for outdated memories: {e}")
            return []

    def confirm(self, memory_id: str) -> bool:
        """Mark a memory as confirmed (high confidence)."""
        current = self.get_status(memory_id)
        if current == "confirmed":
            return True
        return self.transition(memory_id, current, "confirmed")

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark old memory as superseded by a new one, with link metadata."""
        old_status = self.get_status(old_id)
        return self.transition(
            old_id,
            old_status,
            "superseded",
            metadata={"superseded_by": new_id},
        )

    def contradict(self, memory_id: str, contradicting_id: str) -> bool:
        """Mark a memory as contradicted by another."""
        current = self.get_status(memory_id)
        return self.transition(
            memory_id,
            current,
            "contradicted",
            metadata={"contradicted_by": contradicting_id},
        )

    def restore(self, memory_id: str) -> bool:
        """Restore an archived or contradicted memory back to active."""
        current = self.get_status(memory_id)
        if current not in ("archived", "contradicted", "outdated"):
            logger.warning(f"Cannot restore from status '{current}'")
            return False
        return self.transition(memory_id, current, "active")

    def get_lifecycle_stats(self, group_id: Optional[str] = None) -> dict[str, int]:
        """Get counts of memories by lifecycle status.

        Args:
            group_id: Optional project group to filter by.

        Returns:
            Dict mapping status -> count.
        """
        stats: dict[str, int] = {}

        try:
            with self._driver.session() as session:
                query = """
                    MATCH (n)
                    WHERE True
                """
                params: dict[str, Any] = {}

                if group_id:
                    query += " AND n.group_id = $group_id"
                    params["group_id"] = group_id

                query += """
                    RETURN coalesce(n.lifecycle_status, 'active') AS status,
                           count(n) AS cnt
                """

                result = session.run(query, **params)
                for record in result:
                    stats[record["status"]] = record["cnt"]

        except Exception as e:
            logger.error(f"Failed to get lifecycle stats: {e}")

        return stats

    def bulk_archive_stale(
        self,
        threshold_days: int = STALE_THRESHOLD_DAYS,
        group_id: Optional[str] = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Archive all active memories older than threshold.

        Args:
            threshold_days: Days threshold for staleness.
            group_id: Optional project group filter.
            dry_run: If True, just count — don't actually transition.

        Returns:
            Dict with 'count' of affected memories and 'archived' list if not dry_run.
        """
        candidates = self.bulk_check_outdated(threshold_days, group_id)

        if dry_run:
            return {"count": len(candidates), "dry_run": True, "candidates": candidates}

        archived = []
        for c in candidates:
            if self.transition(c["uuid"], "active", "archived"):
                archived.append(c["uuid"])

        return {"count": len(archived), "dry_run": False, "archived": archived}
