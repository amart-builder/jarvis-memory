"""Composite relevance scoring for memory retrieval.

Ported from MemClawz v6/v7, adapted for Graphiti's result format.

Score formula:
  score = (w_semantic * similarity + w_recency * decay + w_importance * weight) * access_boost

Key differences from MemClawz:
  - Works with Graphiti EntityNode/EpisodicNode results instead of Qdrant payloads
  - Uses Graphiti's created_at datetime objects directly (no ISO string parsing needed)
  - Integrates with Graphiti's group_id for project isolation
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .config import W_SEMANTIC, W_RECENCY, W_IMPORTANCE, HALF_LIFE_DAYS

# Base importance by memory type
TYPE_BOOST: dict[str, float] = {
    # Core types (from MemClawz v6)
    "decision": 1.0,
    "preference": 0.95,
    "relationship": 0.9,
    "insight": 0.9,
    "procedure": 0.85,
    "fact": 0.8,
    "event": 0.7,
    # Action cycle types (from MemClawz v7)
    "intention": 0.75,
    "plan": 0.85,
    "commitment": 0.9,
    "action": 0.8,
    "outcome": 0.85,
    "cancellation": 0.6,
    # Extended types
    "goal": 0.9,
    "constraint": 0.85,
    "hypothesis": 0.7,
    "observation": 0.65,
    "question": 0.6,
    "answer": 0.75,
    "correction": 0.85,
    "meta": 0.5,
}

# Types that resist decay (floor at 40% recency)
PERSISTENT_TYPES: set[str] = {"decision", "preference", "relationship", "commitment", "goal", "constraint"}


def composite_score(
    semantic_similarity: float,
    created_at: str | datetime | None = None,
    importance: float = 0.8,
    access_count: int = 0,
    memory_type: str = "fact",
    *,
    w_semantic: float = W_SEMANTIC,
    w_recency: float = W_RECENCY,
    w_importance: float = W_IMPORTANCE,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Compute composite relevance score for a memory.

    Args:
        semantic_similarity: Cosine similarity from vector/graph search (0-1).
        created_at: ISO 8601 string or datetime object of memory creation.
        importance: Base importance (0-1).
        access_count: Number of times this memory has been accessed.
        memory_type: One of the classified memory types.
        w_semantic: Weight for semantic similarity component.
        w_recency: Weight for recency component.
        w_importance: Weight for importance component.
        half_life_days: Half-life for exponential decay in days.

    Returns:
        Composite score between 0 and ~1.5 (access boost can push above 1.0).
    """
    # --- Recency decay ---
    recency = _compute_recency(created_at, half_life_days)

    # Persistent types don't decay below floor
    if memory_type in PERSISTENT_TYPES:
        recency = max(recency, 0.4)

    # --- Type-based importance ---
    type_weight = TYPE_BOOST.get(memory_type, 0.8)
    weighted_importance = importance * type_weight

    # --- Access frequency boost (capped at 1.5×) ---
    access_boost = min(1.0 + (access_count * 0.05), 1.5)

    # --- Composite ---
    score = (
        w_semantic * semantic_similarity
        + w_recency * recency
        + w_importance * weighted_importance
    ) * access_boost

    return min(max(score, 0.0), 1.5)


def score_results(
    results: list[dict[str, Any]],
    similarity_key: str = "score",
) -> list[dict[str, Any]]:
    """Re-score a list of search results using composite scoring.

    Works with both Graphiti search results and raw dicts. Adds 'composite_score'
    to each result and returns sorted descending.

    Args:
        results: List of result dicts. Each should have at minimum a similarity
                 score. Metadata can be nested under 'metadata' or flat.
        similarity_key: Key name for the semantic similarity score.

    Returns:
        Same list with 'composite_score' added, sorted descending.
    """
    scored = []
    for r in results:
        # Support both nested metadata and flat keys
        meta = r.get("metadata", r)
        sim = r.get(similarity_key, meta.get(similarity_key, 0.5))

        # Importance: check enriched, then top-level, then default
        importance = (
            meta.get("importance")
            or r.get("importance")
            or 0.8
        )

        # Timestamp: check multiple possible keys
        created_at = (
            meta.get("created_at")
            or meta.get("extracted_at")
            or meta.get("date")
            or r.get("created_at")
        )

        cs = composite_score(
            semantic_similarity=sim,
            created_at=created_at,
            importance=importance,
            access_count=meta.get("access_count", 0),
            memory_type=meta.get("memory_type", meta.get("type", "fact")),
        )
        r["composite_score"] = cs
        scored.append(r)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def _compute_recency(
    created: str | datetime | None,
    half_life_days: float = HALF_LIFE_DAYS,
) -> float:
    """Compute recency factor using exponential decay.

    Accepts both ISO date strings and datetime objects (for Graphiti compatibility).
    """
    if created is None:
        return 0.5

    now = datetime.now(timezone.utc)

    if isinstance(created, datetime):
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max((now - created).days, 0)
    elif isinstance(created, str):
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            age_days = max((now - created_dt).days, 0)
        except (ValueError, TypeError):
            return 0.5
    else:
        return 0.5

    return math.exp(-0.693 * age_days / half_life_days)
