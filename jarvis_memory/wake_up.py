"""Token-aware context loading — layered memory stack.

Layer 0 (Identity):        ~100 tokens — project name + current status
Layer 1 (Essential Story): ~500 tokens — top memories grouped by room
Layer 2 (Session Context):  on-demand  — via continue_session
Layer 3 (Deep Search):      on-demand  — via scored_search

wake_up() returns Layer 0 + Layer 1 as a pre-formatted context block.
Total cost: ~600 tokens (1-2% of context window).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from .config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    WAKE_UP_LAYER1_MAX_ITEMS, WAKE_UP_LAYER1_MAX_TOKENS,
)
from .scoring import composite_score
from .rooms import detect_room, get_hall

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def generate_layer0(group_id: str, driver=None) -> str:
    """Layer 0 — Identity: project name + latest session status.

    Args:
        group_id: Project identifier.
        driver: Optional Neo4j driver for fetching latest session info.

    Returns:
        Formatted identity block (~100 tokens).
    """
    lines = [f"Project: {group_id}"]

    if driver:
        try:
            with driver.session() as db:
                result = db.run(
                    """
                    MATCH (s:Session {group_id: $gid})
                    WHERE s.status IN ['completed', 'interrupted', 'handoff']
                    RETURN s.task_summary AS task, s.status AS status,
                           s.device AS device, s.ended_at AS ended_at
                    ORDER BY s.started_at DESC LIMIT 1
                    """,
                    gid=group_id,
                )
                record = result.single()
                if record:
                    lines.append(f"Last session: {record['task'] or 'untitled'} ({record['status']})")
                    lines.append(f"Device: {record['device'] or 'unknown'}")
                    if record["ended_at"]:
                        lines.append(f"Ended: {str(record['ended_at'])[:19]}")
        except Exception as e:
            logger.warning(f"Layer 0 session lookup failed: {e}")

    return "\n".join(lines)


def generate_layer1(
    store,
    group_id: str,
    driver=None,
    max_items: int = WAKE_UP_LAYER1_MAX_ITEMS,
    max_tokens: int = WAKE_UP_LAYER1_MAX_TOKENS,
) -> str:
    """Layer 1 — Essential Story: top memories from last 30 days, grouped by room.

    Args:
        store: EmbeddingStore instance (for semantic retrieval).
        group_id: Project identifier.
        driver: Neo4j driver for fetching full memory data.
        max_items: Maximum number of memories to include.
        max_tokens: Target token budget.

    Returns:
        Formatted essential context block (~500 tokens).
    """
    if not store or not store.health_check():
        return _generate_layer1_fallback(driver, group_id, max_items)

    # Get recent memories from ChromaDB for this wing
    try:
        # Search for broadly relevant recent memories
        results = store.search(
            query=f"project {group_id} recent decisions plans progress",
            limit=max_items * 2,  # Oversample for scoring
            where_filter={"wing": group_id},
        )

        if not results:
            return _generate_layer1_fallback(driver, group_id, max_items)

        # Score and sort
        scored_items = []
        for r in results:
            meta = r.get("metadata", {})
            cs = composite_score(
                semantic_similarity=r.get("similarity", 0.5),
                created_at=meta.get("created_at"),
                importance=0.8,
                access_count=0,
                memory_type=meta.get("memory_type", "fact"),
            )
            scored_items.append({
                "id": r["id"],
                "score": cs,
                "room": meta.get("room", "general"),
                "hall": meta.get("hall", "context"),
                "type": meta.get("memory_type", "fact"),
            })

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        top_items = scored_items[:max_items]

        # Fetch content from Neo4j
        if driver:
            uuids = [item["id"] for item in top_items]
            content_map = _fetch_content(driver, uuids)
        else:
            content_map = {}

        # Group by room
        rooms: dict[str, list[str]] = {}
        for item in top_items:
            room = item["room"]
            content = content_map.get(item["id"], "")
            if content:
                # Truncate individual items
                short = content[:120].strip()
                if len(content) > 120:
                    short += "..."
                rooms.setdefault(room, []).append(f"[{item['type']}] {short}")

        # Format grouped by room
        lines = ["## Key Context (last 30 days)"]
        token_count = _estimate_tokens(lines[0])

        for room, items in sorted(rooms.items()):
            room_header = f"\n**{room}:**"
            token_count += _estimate_tokens(room_header)
            if token_count > max_tokens:
                break
            lines.append(room_header)

            for item in items:
                token_count += _estimate_tokens(item)
                if token_count > max_tokens:
                    break
                lines.append(f"- {item}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Layer 1 generation failed: {e}")
        return _generate_layer1_fallback(driver, group_id, max_items)


def _generate_layer1_fallback(
    driver,
    group_id: str,
    max_items: int,
) -> str:
    """Fallback Layer 1 using Neo4j text search when ChromaDB unavailable."""
    if not driver:
        return "No recent context available."

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with driver.session() as db:
            result = db.run(
                """
                MATCH (n)
                WHERE (n:EntityNode OR n:EpisodicNode OR n:Episode)
                  AND n.group_id = $gid
                  AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                  AND n.created_at >= $cutoff
                RETURN coalesce(n.content, n.name, n.summary, '') AS text,
                       coalesce(n.memory_type, n.episode_type, 'fact') AS memory_type
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                gid=group_id,
                cutoff=cutoff,
                limit=max_items,
            )

            lines = ["## Key Context (last 30 days)"]
            for record in result:
                text = (record["text"] or "")[:120]
                if text:
                    lines.append(f"- [{record['memory_type']}] {text}")

            return "\n".join(lines) if len(lines) > 1 else "No recent context available."

    except Exception as e:
        logger.warning(f"Layer 1 fallback failed: {e}")
        return "No recent context available."


def _fetch_content(driver, uuids: list[str]) -> dict[str, str]:
    """Fetch content for a list of memory UUIDs from Neo4j."""
    try:
        with driver.session() as db:
            result = db.run(
                """
                UNWIND $uuids AS uid
                MATCH (n) WHERE n.uuid = uid
                RETURN n.uuid AS uuid,
                       coalesce(n.content, n.name, n.summary, n.fact, '') AS text
                """,
                uuids=uuids,
            )
            return {record["uuid"]: record["text"] for record in result}
    except Exception:
        return {}


def wake_up(
    store,
    driver,
    group_id: str,
) -> dict[str, Any]:
    """Generate full wake-up context (Layer 0 + Layer 1).

    Args:
        store: EmbeddingStore instance.
        driver: Neo4j driver instance.
        group_id: Project identifier.

    Returns:
        Dict with:
            context: Formatted context string
            token_estimate: Rough token count
            layers_loaded: Which layers were included
    """
    layer0 = generate_layer0(group_id, driver)
    layer1 = generate_layer1(store, group_id, driver)

    context = f"{layer0}\n\n{layer1}"
    tokens = _estimate_tokens(context)

    return {
        "context": context,
        "token_estimate": tokens,
        "layers_loaded": ["layer0_identity", "layer1_essential"],
        "group_id": group_id,
    }
