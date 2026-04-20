"""3-tier memory compaction engine.

Ported from MemClawz v6/v7, adapted for Graphiti's Neo4j graph storage.

Three tiers:
  1. Session compaction — deduplicate within a single session's memories
  2. Daily digest — merge similar memories from past 24h into summaries
  3. Weekly merge — consolidate daily digests into long-term memories

Key improvements over MemClawz:
  - Idempotency: each compaction run is tagged with a run_id to prevent double-processing
  - Uses Graphiti's temporal edges for merge provenance instead of flat metadata
  - Group-aware: can compact per-project or globally
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from .config import (
    COMPACTION_DEDUP_DAILY,
    COMPACTION_DEDUP_WEEKLY,
    COMPACTION_SESSION_MAX,
    CLASSIFIER_MODEL,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)
from .classifier import classify_memory

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Generate a content hash for deduplication."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class CompactionEngine:
    """3-tier memory compaction with idempotency and group isolation.

    v2: Supports semantic dedup via ChromaDB EmbeddingStore when available.
    Falls back to hash-only dedup if no embedding store provided.
    """

    def __init__(self, driver=None, graphiti_client=None, embedding_store=None):
        """Initialize with Neo4j driver and optional embedding store.

        Args:
            driver: Neo4j driver instance. If None, creates from config.
            graphiti_client: Optional Graphiti client (legacy, unused in v2).
            embedding_store: Optional EmbeddingStore for semantic dedup.
        """
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self._owns_driver = True

        self._graphiti = graphiti_client
        self._embed_store = embedding_store

    def close(self):
        if self._owns_driver and self._driver:
            self._driver.close()

    # ── Tier 1: Session Compaction ─────────────────────────────────────

    def compact_session(
        self,
        session_id: str,
        group_id: Optional[str] = None,
        max_memories: int = COMPACTION_SESSION_MAX,
    ) -> dict[str, Any]:
        """Deduplicate memories within a single session.

        Finds exact and near-duplicate memories (by content hash) created
        during this session and merges them, keeping the most recent version.

        Args:
            session_id: Identifier for the session to compact.
            group_id: Optional project group filter.
            max_memories: Maximum memories to process per session.

        Returns:
            Dict with stats: total_checked, duplicates_found, merged_count.
        """
        run_id = f"session-{session_id}-{uuid.uuid4().hex[:8]}"

        try:
            with self._driver.session() as db:
                # Get all memories from this session
                query = """
                    MATCH (n:EpisodicNode)
                    WHERE n.source_description CONTAINS $session_id
                      AND n.compaction_run_id IS NULL
                """
                params: dict[str, Any] = {"session_id": session_id}

                if group_id:
                    query += " AND n.group_id = $group_id"
                    params["group_id"] = group_id

                query += """
                    RETURN n.uuid AS uuid, n.content AS content, n.created_at AS created_at
                    ORDER BY n.created_at DESC
                    LIMIT $limit
                """
                params["limit"] = max_memories

                result = db.run(query, **params)
                memories = [dict(r) for r in result]

            if not memories:
                return {"run_id": run_id, "total_checked": 0, "duplicates_found": 0, "merged_count": 0}

            # Group by content hash to find duplicates
            hash_groups: dict[str, list[dict]] = {}
            for mem in memories:
                h = _content_hash(mem.get("content", ""))
                hash_groups.setdefault(h, []).append(mem)

            duplicates_found = 0
            merged_count = 0

            for h, group in hash_groups.items():
                if len(group) < 2:
                    continue

                duplicates_found += len(group) - 1

                # Keep the most recent, mark others as merged
                keeper = group[0]  # already sorted by created_at DESC
                for dup in group[1:]:
                    self._mark_merged(dup["uuid"], keeper["uuid"], run_id)
                    merged_count += 1

            # Tag all processed memories with run_id
            self._tag_run(run_id, [m["uuid"] for m in memories])

            return {
                "run_id": run_id,
                "total_checked": len(memories),
                "duplicates_found": duplicates_found,
                "merged_count": merged_count,
            }

        except Exception as e:
            logger.error(f"Session compaction failed: {e}")
            return {"run_id": run_id, "error": str(e)}

    # ── Tier 2: Daily Digest ───────────────────────────────────────────

    def daily_digest(
        self,
        group_id: Optional[str] = None,
        similarity_threshold: float = COMPACTION_DEDUP_DAILY,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """Create daily digest by merging similar memories from the past day.

        Uses content hashing for exact dedup and optionally vector similarity
        for near-duplicate detection (if Graphiti client available).

        Args:
            group_id: Optional project group filter.
            similarity_threshold: Cosine similarity above which memories are merged.
            lookback_hours: How far back to look.

        Returns:
            Dict with stats.
        """
        run_id = f"daily-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        try:
            with self._driver.session() as db:
                query = """
                    MATCH (n)
                    WHERE n.created_at >= $cutoff
                      AND coalesce(n.lifecycle_status, 'active') = 'active'
                      AND n.compaction_daily_run IS NULL
                """
                params: dict[str, Any] = {"cutoff": cutoff.isoformat()}

                if group_id:
                    query += " AND n.group_id = $group_id"
                    params["group_id"] = group_id

                query += """
                    RETURN n.uuid AS uuid, n.content AS content,
                           coalesce(n.name, '') AS name,
                           coalesce(n.memory_type, 'fact') AS memory_type
                    ORDER BY n.created_at
                    LIMIT 500
                """

                result = db.run(query, **params)
                memories = [dict(r) for r in result]

            if not memories:
                return {"run_id": run_id, "total_checked": 0, "merged_count": 0}

            # Exact dedup by content hash
            hash_groups: dict[str, list[dict]] = {}
            for mem in memories:
                h = _content_hash(mem.get("content", "") or mem.get("name", ""))
                hash_groups.setdefault(h, []).append(mem)

            merged_count = 0
            merged_uuids = set()  # Track already-merged to avoid double-merge

            # Pass 1: Exact hash dedup (fast path)
            for h, group in hash_groups.items():
                if len(group) < 2:
                    continue
                keeper = group[0]
                for dup in group[1:]:
                    self._mark_merged(dup["uuid"], keeper["uuid"], run_id)
                    merged_uuids.add(dup["uuid"])
                    merged_count += 1

            # Pass 2: Semantic dedup via ChromaDB (if available)
            semantic_merged = 0
            if self._embed_store and self._embed_store.health_check():
                remaining = [m for m in memories if m["uuid"] not in merged_uuids]
                for i, mem in enumerate(remaining):
                    if mem["uuid"] in merged_uuids:
                        continue
                    content = mem.get("content", "") or mem.get("name", "")
                    if not content:
                        continue
                    # Search for similar memories
                    similar = self._embed_store.search(query=content, limit=5)
                    for s in similar:
                        if s["id"] == mem["uuid"] or s["id"] in merged_uuids:
                            continue
                        if s["similarity"] >= similarity_threshold:
                            # Check it's in our working set
                            if any(m2["uuid"] == s["id"] for m2 in remaining):
                                self._mark_merged(s["id"], mem["uuid"], run_id)
                                merged_uuids.add(s["id"])
                                merged_count += 1
                                semantic_merged += 1

            if semantic_merged > 0:
                logger.info(f"Daily digest: {semantic_merged} semantic merges (threshold {similarity_threshold})")

            # Tag daily run
            with self._driver.session() as db:
                uuids = [m["uuid"] for m in memories]
                db.run(
                    """
                    UNWIND $uuids AS uid
                    MATCH (n) WHERE n.uuid = uid
                    SET n.compaction_daily_run = $run_id
                    """,
                    uuids=uuids,
                    run_id=run_id,
                )

            return {
                "run_id": run_id,
                "total_checked": len(memories),
                "merged_count": merged_count,
            }

        except Exception as e:
            logger.error(f"Daily digest failed: {e}")
            return {"run_id": run_id, "error": str(e)}

    # ── Tier 3: Weekly Merge ───────────────────────────────────────────

    def weekly_merge(
        self,
        group_id: Optional[str] = None,
        similarity_threshold: float = COMPACTION_DEDUP_WEEKLY,
    ) -> dict[str, Any]:
        """Consolidate memories from the past week.

        Higher dedup threshold (0.92) — only merges very similar memories.

        Args:
            group_id: Optional project group filter.
            similarity_threshold: Threshold for near-duplicate detection.

        Returns:
            Dict with stats.
        """
        run_id = f"weekly-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        try:
            with self._driver.session() as db:
                query = """
                    MATCH (n)
                    WHERE n.created_at >= $cutoff
                      AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                      AND n.compaction_weekly_run IS NULL
                """
                params: dict[str, Any] = {"cutoff": cutoff.isoformat()}

                if group_id:
                    query += " AND n.group_id = $group_id"
                    params["group_id"] = group_id

                query += """
                    RETURN n.uuid AS uuid, n.content AS content,
                           coalesce(n.name, '') AS name
                    ORDER BY n.created_at
                    LIMIT 1000
                """

                result = db.run(query, **params)
                memories = [dict(r) for r in result]

            if not memories:
                return {"run_id": run_id, "total_checked": 0, "merged_count": 0}

            # Content hash dedup (stricter threshold)
            hash_groups: dict[str, list[dict]] = {}
            for mem in memories:
                h = _content_hash(mem.get("content", "") or mem.get("name", ""))
                hash_groups.setdefault(h, []).append(mem)

            merged_count = 0
            for h, group in hash_groups.items():
                if len(group) < 2:
                    continue
                keeper = group[0]
                for dup in group[1:]:
                    self._mark_merged(dup["uuid"], keeper["uuid"], run_id)
                    merged_count += 1

            # Tag weekly run
            with self._driver.session() as db:
                uuids = [m["uuid"] for m in memories]
                db.run(
                    """
                    UNWIND $uuids AS uid
                    MATCH (n) WHERE n.uuid = uid
                    SET n.compaction_weekly_run = $run_id
                    """,
                    uuids=uuids,
                    run_id=run_id,
                )

            return {
                "run_id": run_id,
                "total_checked": len(memories),
                "merged_count": merged_count,
            }

        except Exception as e:
            logger.error(f"Weekly merge failed: {e}")
            return {"run_id": run_id, "error": str(e)}

    # ── Helpers ────────────────────────────────────────────────────────

    def _mark_merged(self, dup_uuid: str, keeper_uuid: str, run_id: str):
        """Mark a memory as merged, linking it to the keeper."""
        try:
            with self._driver.session() as db:
                db.run(
                    """
                    MATCH (dup) WHERE dup.uuid = $dup_uuid
                    SET dup.lifecycle_status = 'merged',
                        dup.lifecycle_updated_at = datetime(),
                        dup.merged_into = $keeper_uuid,
                        dup.compaction_run_id = $run_id
                    """,
                    dup_uuid=dup_uuid,
                    keeper_uuid=keeper_uuid,
                    run_id=run_id,
                )
                # Create a MERGED_INTO relationship for graph traversal
                db.run(
                    """
                    MATCH (dup) WHERE dup.uuid = $dup_uuid
                    MATCH (keeper) WHERE keeper.uuid = $keeper_uuid
                    MERGE (dup)-[:MERGED_INTO {run_id: $run_id, at: datetime()}]->(keeper)
                    """,
                    dup_uuid=dup_uuid,
                    keeper_uuid=keeper_uuid,
                    run_id=run_id,
                )
        except Exception as e:
            logger.error(f"Failed to mark {dup_uuid} as merged into {keeper_uuid}: {e}")

    def _tag_run(self, run_id: str, uuids: list[str]):
        """Tag memories with a compaction run ID for idempotency."""
        try:
            with self._driver.session() as db:
                db.run(
                    """
                    UNWIND $uuids AS uid
                    MATCH (n) WHERE n.uuid = uid
                    SET n.compaction_run_id = $run_id
                    """,
                    uuids=uuids,
                    run_id=run_id,
                )
        except Exception as e:
            logger.error(f"Failed to tag compaction run {run_id}: {e}")

    def get_compaction_status(self) -> dict[str, Any]:
        """Get compaction health metrics."""
        try:
            with self._driver.session() as db:
                result = db.run("""
                    MATCH (n)
                    RETURN
                        count(CASE WHEN n.compaction_run_id IS NOT NULL THEN 1 END) AS session_compacted,
                        count(CASE WHEN n.compaction_daily_run IS NOT NULL THEN 1 END) AS daily_compacted,
                        count(CASE WHEN n.compaction_weekly_run IS NOT NULL THEN 1 END) AS weekly_compacted,
                        count(CASE WHEN n.lifecycle_status = 'merged' THEN 1 END) AS total_merged,
                        count(n) AS total_nodes
                """)
                record = result.single()
                return dict(record) if record else {}
        except Exception as e:
            logger.error(f"Failed to get compaction status: {e}")
            return {"error": str(e)}

    # ── Run 3: dream-cycle hygiene phases ──────────────────────────────
    #
    # These run *read-only*. They surface fix queues and drift metrics
    # so later review (human or agent) can act. Auto-fixing production
    # data is deferred (spec Run 3 Decision Log, 2026-04-20).
    _CITATION_PATTERNS = [
        # "[cite:uuid]" or "[cite:slug]" — must have a value.
        re.compile(r"\[cite:\s*\]"),
        # Dangling bracket: "[cite: something\n" with no close bracket.
        re.compile(r"\[cite:[^\]\n]{0,120}$", re.MULTILINE),
        # "see episode UUID" where UUID is obviously malformed (less than
        # 6 chars after the prefix — UUID-shaped ids are far longer).
        re.compile(r"see episode\s+[^\s]{1,5}(?:\W|$)", re.IGNORECASE),
    ]

    def _fix_citations(self, session=None, *, sample_limit: int = 500) -> dict[str, Any]:
        """Scan episode content for broken citation patterns.

        Dream-cycle phase 1 of 3. Returns a fix queue (does NOT mutate
        episode content). Runtime budget: ≤ 2 min on current corpus size
        (spec Run 3 §"Constraints"); scanning is bounded by ``sample_limit``.

        Args:
            session: Optional existing Neo4j session. When None, we open
                one from ``self._driver``.
            sample_limit: Maximum episodes to scan in a single pass.

        Returns:
            ``{"scanned": int, "broken_count": int, "queue": list, "error": str?}``.
            Each queue entry: ``{uuid, issues, content_preview}``.
        """
        import time

        started = time.monotonic()
        try:
            if self._driver is None:
                return {"scanned": 0, "broken_count": 0, "queue": [], "error": "no driver"}

            owns = session is None
            sess = session or self._driver.session()
            try:
                result = sess.run(
                    """
                    MATCH (n:Episode)
                    WHERE n.content IS NOT NULL
                      AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                    RETURN n.uuid AS uuid, n.content AS content
                    ORDER BY n.created_at DESC
                    LIMIT $lim
                    """,
                    lim=sample_limit,
                )
                rows = [dict(r) for r in result]
            finally:
                if owns and hasattr(sess, "close"):
                    sess.close()

            queue: list[dict[str, Any]] = []
            for row in rows:
                content = row.get("content") or ""
                issues: list[str] = []
                for pattern in self._CITATION_PATTERNS:
                    if pattern.search(content):
                        issues.append(pattern.pattern)
                if issues:
                    queue.append(
                        {
                            "uuid": row.get("uuid"),
                            "issues": issues,
                            "content_preview": content[:160],
                        }
                    )

            runtime_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "dream-cycle fix_citations: scanned=%d broken=%d runtime_ms=%d",
                len(rows),
                len(queue),
                runtime_ms,
            )
            return {
                "scanned": len(rows),
                "broken_count": len(queue),
                "queue": queue,
                "runtime_ms": runtime_ms,
            }
        except Exception as e:
            logger.error(f"dream-cycle fix_citations failed: {e}")
            return {"scanned": 0, "broken_count": 0, "queue": [], "error": str(e)}

    def _report_orphans(self, session=None) -> dict[str, Any]:
        """Invoke orphans.find_orphans and surface counts for trending.

        Dream-cycle phase 2 of 3. Read-only — we do not auto-delete
        orphan Pages; we log them so later review can decide.
        """
        import time

        started = time.monotonic()
        try:
            # Late import to avoid a circular dep at module-load time.
            from .orphans import find_orphans

            driver = self._driver
            # find_orphans manages its own session under the hood; if the
            # caller passed one we reuse it via the ``tx`` parameter.
            if session is not None:
                grouped = find_orphans(tx=session)
            else:
                grouped = find_orphans(driver=driver)

            counts = {domain: len(pages) for domain, pages in grouped.items()}
            total = sum(counts.values())
            runtime_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "dream-cycle report_orphans: total=%d by_domain=%s runtime_ms=%d",
                total,
                counts,
                runtime_ms,
            )
            return {
                "total_orphans": total,
                "by_domain": counts,
                "sample": {
                    domain: [p.slug for p in pages[:5]]
                    for domain, pages in grouped.items()
                },
                "runtime_ms": runtime_ms,
            }
        except Exception as e:
            logger.error(f"dream-cycle report_orphans failed: {e}")
            return {"total_orphans": 0, "by_domain": {}, "error": str(e)}

    def _reconcile_stale_edges(self, session=None) -> dict[str, Any]:
        """Find EVIDENCED_BY edges that reference missing Episodes.

        Dream-cycle phase 3 of 3. Read-only — surfaces the ids of any
        edge pointing at a deleted/missing Episode so later review can
        archive or clean them. Runtime budget ≤ 2 min.
        """
        import time

        started = time.monotonic()
        try:
            if self._driver is None:
                return {"stale_edges": 0, "sample": [], "error": "no driver"}

            owns = session is None
            sess = session or self._driver.session()
            try:
                # EVIDENCED_BY edge with a NULL destination end means
                # the Episode was deleted but the Page still points at it.
                # In Neo4j 5 a MATCH can't produce a NULL endpoint, so
                # we look for edges whose target has a missing uuid.
                result = sess.run(
                    """
                    MATCH (p:Page)-[r:EVIDENCED_BY]->(e)
                    WHERE e.uuid IS NULL
                       OR coalesce(e.lifecycle_status, 'active') = 'deleted'
                    RETURN p.slug AS page_slug, coalesce(e.uuid, '') AS episode_uuid
                    LIMIT 200
                    """,
                )
                rows = [dict(r) for r in result]
            finally:
                if owns and hasattr(sess, "close"):
                    sess.close()

            runtime_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "dream-cycle reconcile_stale_edges: stale=%d runtime_ms=%d",
                len(rows),
                runtime_ms,
            )
            return {
                "stale_edges": len(rows),
                "sample": rows[:25],
                "runtime_ms": runtime_ms,
            }
        except Exception as e:
            logger.error(f"dream-cycle reconcile_stale_edges failed: {e}")
            return {"stale_edges": 0, "sample": [], "error": str(e)}

    def run_dream_cycle(self) -> dict[str, Any]:
        """Run all three read-only dream-cycle phases in sequence.

        Called from the daily compaction cron (``scripts/run_compaction.py``)
        right after ``daily_digest``. Total runtime budget ≤ 6 minutes
        (≤ 2 min per phase) so the combined daily run stays under the
        extended 25-minute budget in spec Run 3.
        """
        report: dict[str, Any] = {}
        report["fix_citations"] = self._fix_citations()
        report["orphans"] = self._report_orphans()
        report["stale_edges"] = self._reconcile_stale_edges()
        return report
