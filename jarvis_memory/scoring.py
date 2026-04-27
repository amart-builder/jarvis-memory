"""Relevance scoring for memory retrieval.

Run 1 baseline: composite-weighted scoring (semantic × recency ×
importance × access). Kept here as :func:`composite_score` /
:func:`score_results` because ``test_scoring.py`` tests those
names directly (signature lock).

Run 3 addition: :func:`scored_search` — the hybrid retrieval entry
point. It composes the new ``jarvis_memory.search`` stack:

    1. :func:`~jarvis_memory.search.intent.classify` picks a route
       (entity / temporal / event / general).
    2. Two base retrievers run in parallel: Chroma vector search and
       Neo4j full-text keyword search.
    3. For entity/general intents we also ask
       :func:`~jarvis_memory.search.expansion.expand` for cheap
       Haiku-written query variants and run the vector retriever over
       each of them.
    4. :func:`~jarvis_memory.search.rrf.reciprocal_rank_fusion` fuses
       every ranked list into one consensus order (k=60).
    5. :func:`~jarvis_memory.search.boosts.apply_boosts` re-ranks using
       Page.compiled_truth presence (×1.2) and Page in-degree
       (log(1+d)·0.1 additive).
    6. Post-fusion filters (group_id, room, hall, memory_type, as_of)
       are applied on the enriched node properties, preserving the
       legacy API contract.

``JARVIS_SEARCH_LEGACY=1`` flips back to the Run 1 composite path for
A/B comparison.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from .config import W_SEMANTIC, W_RECENCY, W_IMPORTANCE, HALF_LIFE_DAYS

logger = logging.getLogger(__name__)

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


# ══════════════════════════════════════════════════════════════════════
# Run 3 — hybrid scored_search
# ══════════════════════════════════════════════════════════════════════

# Public signature is LOCKED — spec Run 3 §"Must-not-break flows" + plan
# C1. Internal plumbing may change freely; external contract may not.
VectorSearchFn = Callable[[str, int], list[dict[str, Any]]]
"""(query, n_results) -> list of hits with 'id' / 'uuid' + optional 'similarity'."""


def _should_use_legacy() -> bool:
    """``JARVIS_SEARCH_LEGACY=1`` forces the Run 1 composite path."""
    return os.environ.get("JARVIS_SEARCH_LEGACY", "").strip() == "1"


def scored_search(
    query: str,
    *,
    group_id: Optional[str] = None,
    room: Optional[str] = None,
    hall: Optional[str] = None,
    memory_type: Optional[str] = None,
    as_of: Optional[str] = None,
    seen_as_of: Optional[str] = None,
    limit: int = 10,
    driver=None,
    embedding_store=None,
    namespace: Optional[str] = None,
    vector_search_fn: Optional[VectorSearchFn] = None,
    expand_fn: Optional[Callable[[str, int], list[str]]] = None,
    include_expansion: Optional[bool] = None,
    boost_config=None,
) -> list[dict[str, Any]]:
    """Hybrid RRF-based scored search.

    This is the internal function behind the ``/api/v2/scored_search``
    REST endpoint and the ``scored_search`` MCP tool. The response shape
    returned by those surfaces is frozen — this function returns a list
    of enriched-node dicts that the caller wraps in the locked envelope.

    Args:
        query: User query. Empty/whitespace → empty list.
        group_id: Project filter (matched against ``n.group_id``).
        room: Topic filter.
        hall: Category filter.
        memory_type: Specific memory type filter.
        as_of: ISO date — *event-time* validity filter via
            :func:`jarvis_memory.temporal.filter_by_date`. Answers
            "what was true in the world on date X?"
        seen_as_of: ISO date — *ingestion-time* validity filter via
            :func:`jarvis_memory.temporal.filter_by_seen_as_of`. Answers
            "what did we believe on date X?" Composes with ``as_of`` —
            pass both to ask "what did we believe on date X about the
            world on date Y?"
        limit: Maximum results.
        driver: Neo4j driver. When None, the function tries the Chroma
            channel only (keyword_search requires a driver).
        embedding_store: Optional :class:`jarvis_memory.embeddings.EmbeddingStore`.
            Used when ``vector_search_fn`` is not supplied.
        namespace: Neo4j label to scan for episodes. Default ``"Episode"``;
            eval/tests pass ``"TestEpisode"``.
        vector_search_fn: Override for the vector channel — enables the
            eval harness to drive a dedicated Chroma collection.
            ``(query, n_results) -> list[hit dict]``.
        expand_fn: Override for query expansion — defaults to
            :func:`jarvis_memory.search.expansion.expand`.
        include_expansion: Force-enable or force-disable Haiku expansion.
            When ``None`` (default) we auto-decide based on intent +
            query length.
        boost_config: Optional :class:`~jarvis_memory.search.boosts.BoostConfig`.

    Returns:
        List of enriched hit dicts, at most ``limit`` long. Each dict
        contains (where available) ``uuid``, ``content``, ``group_id``,
        ``room``, ``hall``, ``memory_type``, ``created_at``,
        ``similarity``, ``score``, ``composite_score``.

    Never raises on empty input; propagates infrastructure errors
    (Neo4j auth, etc.) the same way the legacy path does.
    """
    if not query or not query.strip():
        return []

    if _should_use_legacy():
        return _legacy_scored_search(
            query=query,
            group_id=group_id,
            room=room,
            hall=hall,
            memory_type=memory_type,
            as_of=as_of,
            seen_as_of=seen_as_of,
            limit=limit,
            driver=driver,
            embedding_store=embedding_store,
            namespace=namespace,
            vector_search_fn=vector_search_fn,
        )

    # Deferred imports keep import graph lightweight when this module is
    # reached from test fixtures that don't touch the search stack.
    from .search.boosts import BoostConfig, apply_boosts
    from .search.expansion import expand as default_expand
    from .search.intent import classify
    from .search.keyword import keyword_search
    from .search.rerank import rerank as cross_encoder_rerank
    from .search.rrf import reciprocal_rank_fusion

    expand_fn = expand_fn or default_expand

    # ── Build per-intent retriever plan ────────────────────────────────
    intent = classify(query)

    # Chroma / vector channel.
    v_over_fetch = max(limit * 3, 20)
    vector_fn = vector_search_fn or _make_default_vector_search_fn(embedding_store)

    def _vector_ranking(q: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
        hits = vector_fn(q, v_over_fetch)
        ids: list[str] = []
        meta: dict[str, dict[str, Any]] = {}
        for h in hits or []:
            hid = str(h.get("id") or h.get("uuid") or "")
            if not hid:
                continue
            ids.append(hid)
            meta[hid] = h
        return ids, meta

    rankings: list[list[str]] = []
    vector_meta: dict[str, dict[str, Any]] = {}

    primary_ids, primary_meta = _vector_ranking(query)
    if primary_ids:
        rankings.append(primary_ids)
        vector_meta.update(primary_meta)

    # Keyword channel (Episode + Page fulltext).
    keyword_hits = []
    if driver is not None:
        try:
            keyword_hits = keyword_search(
                query,
                k=max(limit * 2, 10),
                namespace=namespace,
                driver=driver,
                # Keep pages in the fused list — they contribute to the
                # compiled-truth boost lookup later anyway.
                include_pages=True,
            )
        except Exception as e:  # noqa: BLE001 — keyword is optional
            logger.debug("keyword_search failed (%s); continuing without it", e)
            keyword_hits = []
    if keyword_hits:
        rankings.append([h.id for h in keyword_hits if h.id])

    # Expansion channel — entity + general intents + long-enough queries.
    if include_expansion is None:
        include_expansion = intent in {"entity", "general"} and _looks_expandable(query)
    if include_expansion:
        try:
            variants = expand_fn(query, 2)
        except Exception as e:  # noqa: BLE001 — expansion is optional
            logger.debug("expansion raised (%s); continuing", e)
            variants = [query]
        # Drop the original (already searched) and keep unique variants.
        seen = {query.strip().lower()}
        for v in variants[1:] if variants else []:
            key = (v or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            variant_ids, variant_meta = _vector_ranking(v)
            if variant_ids:
                rankings.append(variant_ids)
                # Fill gaps in meta only (primary meta wins).
                for k, md in variant_meta.items():
                    vector_meta.setdefault(k, md)

    if not rankings:
        # Nothing retrieved from any channel — fall back to legacy composite
        # so a minimal result set is still produced (keeps API contract behavior).
        return _legacy_scored_search(
            query=query,
            group_id=group_id,
            room=room,
            hall=hall,
            memory_type=memory_type,
            as_of=as_of,
            seen_as_of=seen_as_of,
            limit=limit,
            driver=driver,
            embedding_store=embedding_store,
            namespace=namespace,
            vector_search_fn=vector_search_fn,
        )

    # ── Reciprocal rank fusion ─────────────────────────────────────────
    fused = reciprocal_rank_fusion(rankings, k=60)

    # ── Post-RRF boosts ────────────────────────────────────────────────
    fused_ids = [doc_id for doc_id, _ in fused]
    page_lookup, in_degree_lookup = _fetch_boost_lookups(
        fused_ids,
        driver=driver,
    )
    boosted = apply_boosts(
        fused,
        page_lookup=page_lookup,
        in_degree_lookup=in_degree_lookup,
        config=boost_config,
    )

    # ── Enrich + filter ─────────────────────────────────────────────────
    enriched = _enrich_hits(
        boosted,
        vector_meta=vector_meta,
        driver=driver,
        namespace=namespace,
    )
    filtered = _apply_filters(
        enriched,
        group_id=group_id,
        room=room,
        hall=hall,
        memory_type=memory_type,
        as_of=as_of,
        seen_as_of=seen_as_of,
    )

    # ── Cross-encoder rerank (env-gated, fail-open) ────────────────────
    # RRF + boosts decided the candidate set; the cross-encoder gets the
    # final say over order. When ``JARVIS_RERANK=0`` or the model can't
    # load (no network on first call, etc.) the function returns the
    # input unchanged — retrieval never breaks because of the reranker.
    filtered = cross_encoder_rerank(query, filtered)

    # Attach composite_score for downstream consumers that expect that
    # key (wake_up, v1 search shim, etc.) without changing sort order.
    for rec in filtered:
        rec.setdefault("composite_score", rec.get("score", 0.0))

    return filtered[:limit]


# ── Internal helpers ────────────────────────────────────────────────────


def _looks_expandable(query: str) -> bool:
    """Heuristic: only expand queries with more than three tokens.

    Short queries ("status", "foundry") don't benefit from paraphrasing
    — the expansion tends to add noise and costs a network round trip.
    """
    return len([t for t in (query or "").split() if t.strip()]) > 3


def _make_default_vector_search_fn(
    embedding_store,
) -> VectorSearchFn:
    """Return a ``(query, n) -> hits`` wrapper around the shared EmbeddingStore."""

    def _fn(q: str, n: int) -> list[dict[str, Any]]:
        if embedding_store is None or not hasattr(embedding_store, "search"):
            return []
        try:
            hits = embedding_store.search(query=q, limit=n)
        except Exception as e:  # noqa: BLE001
            logger.debug("EmbeddingStore.search failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for h in hits or []:
            out.append(
                {
                    "id": h.get("id") or h.get("uuid"),
                    "uuid": h.get("id") or h.get("uuid"),
                    "similarity": h.get("similarity", 0.7),
                    "metadata": h.get("metadata", {}),
                }
            )
        return out

    return _fn


def _fetch_boost_lookups(
    doc_ids: Iterable[str],
    *,
    driver,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fetch compiled_truth + in-degree for any doc id that maps to a Page.

    Args:
        doc_ids: Candidate ids from the fused RRF ranking. May include
            Page slugs and Episode UUIDs — we look both up.
        driver: Neo4j driver. Without it, both lookups are empty.

    Returns:
        ``(page_lookup, in_degree_lookup)`` keyed by doc id.
    """
    if driver is None:
        return {}, {}

    ids = [i for i in doc_ids if i]
    if not ids:
        return {}, {}

    page_lookup: dict[str, str] = {}
    in_degree_lookup: dict[str, int] = {}

    try:
        with driver.session() as sess:
            records = sess.run(
                """
                UNWIND $ids AS id
                OPTIONAL MATCH (p:Page {slug: id})
                WITH id, p,
                     CASE WHEN p IS NULL THEN 0
                          ELSE size([(x)-[r]->(p) | r]) END AS in_degree
                RETURN id,
                       CASE WHEN p IS NULL THEN '' ELSE coalesce(p.compiled_truth, '') END AS truth,
                       in_degree
                """,
                ids=ids,
            )
            for r in records:
                rid = str(r["id"])
                truth = r["truth"] or ""
                if truth:
                    page_lookup[rid] = truth
                deg = int(r["in_degree"] or 0)
                if deg > 0:
                    in_degree_lookup[rid] = deg
    except Exception as e:  # noqa: BLE001
        logger.debug("boost lookup failed (%s); continuing without boosts", e)
        return {}, {}

    return page_lookup, in_degree_lookup


def _enrich_hits(
    boosted: list[tuple[str, float]],
    *,
    vector_meta: dict[str, dict[str, Any]],
    driver,
    namespace: Optional[str],
) -> list[dict[str, Any]]:
    """Fetch Neo4j node props for each fused hit and attach the RRF score."""
    label = namespace or "Episode"
    enriched: list[dict[str, Any]] = []

    if driver is None:
        # No driver → fall back to whatever vector_meta carries.
        for doc_id, score in boosted:
            meta = vector_meta.get(doc_id, {})
            flattened = dict(meta.get("metadata", {}) or {})
            rec = {
                "uuid": doc_id,
                "id": doc_id,
                "similarity": meta.get("similarity", 0.0),
                "score": score,
                **flattened,
            }
            enriched.append(rec)
        return enriched

    ids = [doc_id for doc_id, _ in boosted]
    if not ids:
        return []

    # One round trip to grab every node at once. We allow both the
    # explicit namespace (TestEpisode) and any node with matching uuid
    # so Page-slug hits (id='page:foo') don't silently drop — the match
    # below uses uuid first then slug.
    try:
        with driver.session() as sess:
            rows = sess.run(
                f"""
                UNWIND $ids AS id
                OPTIONAL MATCH (n:{label} {{uuid: id}})
                OPTIONAL MATCH (p:Page {{slug: id}})
                RETURN id, n, p
                """,
                ids=ids,
            )
            node_map: dict[str, dict[str, Any]] = {}
            for r in rows:
                rid = str(r["id"])
                node = r["n"]
                page = r["p"]
                if node is not None:
                    node_map[rid] = dict(node)
                elif page is not None:
                    # Represent a Page hit with slug + compiled_truth so the
                    # REST envelope still carries meaningful content.
                    pd = dict(page)
                    pd.setdefault("uuid", pd.get("slug") or rid)
                    pd.setdefault("content", pd.get("compiled_truth", ""))
                    node_map[rid] = pd
    except Exception as e:  # noqa: BLE001
        logger.debug("enrichment query failed (%s); returning id-only records", e)
        node_map = {}

    for doc_id, score in boosted:
        props = dict(node_map.get(doc_id, {}))
        # Merge vector-channel similarity if Neo4j didn't have one.
        vm = vector_meta.get(doc_id, {})
        if vm:
            props.setdefault("similarity", vm.get("similarity", 0.0))
            for k, v in (vm.get("metadata") or {}).items():
                props.setdefault(k, v)
        props.setdefault("uuid", doc_id)
        props.setdefault("id", doc_id)
        props["score"] = score
        enriched.append(props)

    return enriched


def _apply_filters(
    records: list[dict[str, Any]],
    *,
    group_id: Optional[str],
    room: Optional[str],
    hall: Optional[str],
    memory_type: Optional[str],
    as_of: Optional[str],
    seen_as_of: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Apply the REST-level filters with legacy-compatible semantics.

    ``as_of`` and ``seen_as_of`` are independent. ``as_of`` is event time
    (was the world like this on date X?); ``seen_as_of`` is ingestion
    time (did we believe this on date X?). Pass either, both, or
    neither. See :func:`jarvis_memory.temporal.filter_by_seen_as_of`.
    """
    out = records
    if group_id:
        out = [r for r in out if r.get("group_id") == group_id or r.get("wing") == group_id]
    if room:
        out = [r for r in out if r.get("room") == room]
    if hall:
        out = [r for r in out if r.get("hall") == hall]
    if memory_type:
        out = [
            r
            for r in out
            if r.get("memory_type") == memory_type
            or r.get("episode_type") == memory_type
        ]
    if as_of:
        try:
            from .temporal import filter_by_date

            out = filter_by_date(out, as_of=as_of)
        except Exception as e:  # noqa: BLE001
            logger.debug("filter_by_date failed (%s); skipping temporal filter", e)
    if seen_as_of:
        try:
            from .temporal import filter_by_seen_as_of

            out = filter_by_seen_as_of(out, seen_as_of=seen_as_of)
        except Exception as e:  # noqa: BLE001
            logger.debug("filter_by_seen_as_of failed (%s); skipping ingestion filter", e)
    return out


def _legacy_scored_search(
    *,
    query: str,
    group_id: Optional[str],
    room: Optional[str],
    hall: Optional[str],
    memory_type: Optional[str],
    as_of: Optional[str],
    seen_as_of: Optional[str] = None,
    limit: int,
    driver,
    embedding_store,
    namespace: Optional[str],
    vector_search_fn: Optional[VectorSearchFn],
) -> list[dict[str, Any]]:
    """Run 1 composite path, preserved behind ``JARVIS_SEARCH_LEGACY=1``.

    This is intentionally a thin reproduction of the original api.py logic
    so ``JARVIS_SEARCH_LEGACY=1 python -m jarvis_memory.eval`` returns
    numbers that match the Run 1 baseline within noise.
    """
    from .temporal import filter_by_date

    label = namespace or "Episode"
    results: list[dict[str, Any]] = []

    # Vector channel (prod: EmbeddingStore; eval: caller-supplied).
    vector_fn = vector_search_fn or _make_default_vector_search_fn(embedding_store)
    over_fetch = min(max(limit * 3, 10), 60)
    hits = vector_fn(query, over_fetch)
    for h in hits or []:
        uid = str(h.get("id") or h.get("uuid") or "")
        if not uid:
            continue
        rec: dict[str, Any] = {
            "uuid": uid,
            "id": uid,
            "similarity": h.get("similarity", 0.7),
            **(h.get("metadata") or {}),
        }
        if driver is not None:
            try:
                with driver.session() as sess:
                    node = sess.run(
                        f"MATCH (n:{label} {{uuid: $uid}}) RETURN n",
                        uid=uid,
                    ).single()
                if node and node["n"]:
                    rec.update(dict(node["n"]))
                    rec["similarity"] = h.get("similarity", 0.7)
            except Exception as e:  # noqa: BLE001
                logger.debug("legacy enrich failed for %s: %s", uid, e)
        results.append(rec)

    if as_of:
        try:
            results = filter_by_date(results, as_of=as_of)
        except Exception:
            pass
    if seen_as_of:
        try:
            from .temporal import filter_by_seen_as_of

            results = filter_by_seen_as_of(results, seen_as_of=seen_as_of)
        except Exception:
            pass
    if group_id:
        results = [
            r
            for r in results
            if r.get("group_id") == group_id or r.get("wing") == group_id
        ]
    if room:
        results = [r for r in results if r.get("room") == room]
    if hall:
        results = [r for r in results if r.get("hall") == hall]
    if memory_type:
        results = [
            r
            for r in results
            if r.get("memory_type") == memory_type
            or r.get("episode_type") == memory_type
        ]

    scored = score_results(results, similarity_key="similarity")
    for rec in scored:
        rec["score"] = rec.get("composite_score", rec.get("similarity", 0.0))
    return scored[:limit]
