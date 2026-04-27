"""Personalized PageRank — multi-hop graph retrieval channel (B1).

HippoRAG 2-style retrieval: the graph itself does the reasoning.

  1. Extract entities from the query (proper nouns / Page slugs).
  2. Seed a personalization vector at matching ``:Page`` nodes.
  3. Build a NetworkX subgraph from Neo4j typed edges + EVIDENCED_BY
     edges. (At ~800 nodes the graph fits trivially in memory; rebuilding
     per query stays well under a second.)
  4. Run personalized PageRank (damping 0.85, ~30 iterations).
  5. Map PPR mass back to Episodes — that's the ranking we return into
     the RRF pool as a fourth retrieval channel.

When does this fire? Only when the intent classifier flags a query as
``multi_hop`` — multi-entity queries or queries with associative
language ("led to", "because of", "drove", "affected"). Single-entity
factoid queries don't benefit from PPR; the bi-encoder + keyword
retrievers already nail those.

Cost: ~30ms graph build + ~30ms PPR at current scale. No LLM calls on
the hot path. Failure mode: returns ``[]``, retrieval continues without
the channel — never blocks search.

Reference: Gutiérrez et al., "From RAG to Memory: Non-Parametric
Continual Learning for Large Language Models" (HippoRAG 2, ICML 2025).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Neo4j 5.x emits a notification warning when a query references a
# relationship type with no instances yet (e.g. ``:ATTENDED`` before
# anyone's attended a meeting). PPR loops through the entire typed-edge
# vocabulary on every call, so these warnings fire repeatedly. Silence
# them at module load — same pattern as scripts/migrate_to_bitemporal.py
# and scripts/run_compaction.py.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# PPR hyperparameters. Damping and iteration count from the HippoRAG /
# NetworkX defaults — well-tested across knowledge-graph workloads.
DEFAULT_DAMPING: float = 0.85
DEFAULT_ITERATIONS: int = 30
DEFAULT_LIMIT: int = 50


def personalized_pagerank(
    query: str,
    *,
    driver,
    namespace: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    damping: float = DEFAULT_DAMPING,
    iterations: int = DEFAULT_ITERATIONS,
) -> list[tuple[str, float]]:
    """Return Episode UUIDs ranked by Personalized PageRank score.

    Args:
        query: User query — we extract proper nouns to seed PPR.
        driver: Neo4j driver. Required to load the subgraph.
        namespace: Optional Episode label override (default ``Episode``;
            ``TestEpisode`` for the eval harness).
        limit: Cap on returned episodes.
        damping: PPR damping (alpha). Default 0.85 — standard for
            knowledge-graph PPR.
        iterations: ``max_iter`` for nx.pagerank. 30 is well past
            convergence at our scale.

    Returns:
        List of ``(episode_uuid, ppr_score)`` tuples sorted descending.
        Empty list when:
          - no proper nouns in the query
          - no matching ``:Page`` slugs in the graph
          - networkx not installed (returns silently — fail open)
          - PPR iteration fails (logged warning)
    """
    if not query or not query.strip() or driver is None:
        return []

    try:
        import networkx as nx  # type: ignore
    except ImportError:
        logger.warning("networkx not installed — PPR channel disabled")
        return []

    # 1. Extract candidate entities from the query.
    seeds_text = _extract_query_entities(query)
    if not seeds_text:
        return []

    # 2. Match those entities to actual Page slugs in the graph.
    seed_slugs = _match_entities_to_pages(driver, seeds_text)
    if not seed_slugs:
        return []

    # 3. Build the subgraph (Pages + Episodes + their edges).
    label = namespace or "Episode"
    G = _build_subgraph(driver, episode_label=label)
    if G.number_of_nodes() == 0:
        return []

    # 4. Build personalization vector — uniform mass over matched seeds
    # that actually appear in the loaded graph.
    personalization = {f"page:{s}": 1.0 for s in seed_slugs if f"page:{s}" in G}
    if not personalization:
        return []

    # 5. Run PPR.
    try:
        scores = nx.pagerank(
            G,
            alpha=damping,
            personalization=personalization,
            max_iter=iterations,
        )
    except Exception as e:  # noqa: BLE001 — fail open on any PPR error
        logger.warning("nx.pagerank failed (%s); skipping PPR channel", e)
        return []

    # 6. Project mass onto Episode nodes only.
    ep_scores: list[tuple[str, float]] = [
        (node[len("ep:") :], score)
        for node, score in scores.items()
        if node.startswith("ep:") and score > 0.0
    ]
    ep_scores.sort(key=lambda kv: kv[1], reverse=True)
    return ep_scores[:limit]


# ── Private helpers ─────────────────────────────────────────────────────


def _extract_query_entities(query: str) -> list[str]:
    """Pull proper-noun candidates out of the query.

    Reuses ``graph._extract_proper_nouns`` so the regex stays in one
    place. Returns lowercased candidates so matching against Page slugs
    (lowercased by convention) is straightforward.
    """
    try:
        from jarvis_memory.graph import _extract_proper_nouns

        nouns = _extract_proper_nouns(query)
    except Exception as e:  # noqa: BLE001 — extractor is optional
        logger.debug("entity extraction failed (%s)", e)
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token, _ in nouns:
        norm = token.strip().lower()
        # Skip multi-word tokens for slug matching — Page slugs are
        # single tokens by convention. The first word usually carries
        # the disambiguating signal anyway ("Jacob" out of "Jacob Martin").
        first = norm.split()[0] if norm else ""
        if first and first not in seen:
            seen.add(first)
            out.append(first)
    return out


def _match_entities_to_pages(driver, candidates: list[str]) -> list[str]:
    """Return Page slugs that match any of ``candidates`` (case-insensitive)."""
    if not candidates:
        return []
    try:
        with driver.session() as sess:
            rows = sess.run(
                """
                MATCH (p:Page)
                WHERE toLower(p.slug) IN $candidates
                   OR ANY(c IN $candidates WHERE toLower(p.slug) CONTAINS c)
                RETURN DISTINCT p.slug AS slug
                LIMIT 50
                """,
                candidates=candidates,
            )
            return [r["slug"] for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.debug("entity→page match failed (%s)", e)
        return []


def _build_subgraph(driver, *, episode_label: str = "Episode"):
    """Load Pages + Episodes + their connecting edges into a NetworkX graph.

    Node IDs are namespaced strings (``page:<slug>`` / ``ep:<uuid>``)
    so collisions are impossible even if a slug ever equals a UUID.
    Graph is undirected — PPR mass flows freely in both directions,
    which matches the "associative recall" intent.
    """
    import networkx as nx  # type: ignore

    G = nx.Graph()

    try:
        with driver.session() as sess:
            # Pages
            for r in sess.run("MATCH (p:Page) RETURN p.slug AS slug"):
                slug = r["slug"]
                if slug:
                    G.add_node(f"page:{slug}", kind="page")

            # Episodes — also include the legacy Graphiti label if present.
            # Use OPTIONAL MATCH-style UNION to keep the query simple.
            ep_query = f"""
                MATCH (e:{episode_label})
                RETURN e.uuid AS uuid
                UNION
                MATCH (e:Episodic)
                RETURN e.uuid AS uuid
            """
            for r in sess.run(ep_query):
                uid = r["uuid"]
                if uid:
                    G.add_node(f"ep:{uid}", kind="episode")

            # Typed edges (Page <-> Page or Episode -> Page).
            # We unify on the relationship-name set from schema_v2.
            from jarvis_memory.schema_v2 import EVIDENCE_EDGE, TYPED_EDGES

            typed_set = list(TYPED_EDGES) + [EVIDENCE_EDGE]
            for rel in typed_set:
                for r in sess.run(
                    f"""
                    MATCH (a)-[:{rel}]->(b)
                    WHERE (a:Page OR a:{episode_label} OR a:Episodic)
                      AND (b:Page OR b:{episode_label} OR b:Episodic)
                    RETURN coalesce(a.slug, a.uuid) AS a_id,
                           CASE WHEN a:Page THEN 'page' ELSE 'ep' END AS a_kind,
                           coalesce(b.slug, b.uuid) AS b_id,
                           CASE WHEN b:Page THEN 'page' ELSE 'ep' END AS b_kind
                    """
                ):
                    a = f"{r['a_kind']}:{r['a_id']}" if r["a_id"] else None
                    b = f"{r['b_kind']}:{r['b_id']}" if r["b_id"] else None
                    if a and b and a in G and b in G:
                        G.add_edge(a, b, rel=rel)
    except Exception as e:  # noqa: BLE001 — graph load is best-effort
        logger.warning("PPR subgraph build failed (%s); returning empty graph", e)
        return nx.Graph()

    return G
