"""Neo4j keyword (lexical) search — second retriever for the RRF stack.

This complements the existing Chroma semantic retriever by surfacing docs
whose content literally contains query terms. For the RRF fusion in
``scored_search`` we want the lexical channel to be independent of the
vector channel so their rankings disagree productively.

Implementation notes
--------------------

* **Preferred path: Neo4j full-text index.** When the caller has set up a
  ``CREATE FULLTEXT INDEX`` on ``Episode.content`` or ``Page.compiled_truth``,
  ``db.index.fulltext.queryNodes`` gives BM25-style scoring. Run 2 ships a
  ``page_compiled_truth_fulltext`` index; an Episode fulltext index may or
  may not be present depending on deploy state.
* **Fallback: lexical ``CONTAINS``.** If a fulltext index is unavailable
  (e.g. the synthetic eval namespace uses ``:TestEpisode``, which has no
  fulltext index), we fall back to a ``CONTAINS`` scan with a crude
  token-overlap score. Slower but correct for small namespaces.
* **Namespace parameter.** Tests (and the eval harness) write to an
  isolated label (``:TestEpisode``). Pass ``namespace=...`` to target
  that label; default is the production ``:Episode``.

The public surface is a pure callable that takes a ``driver`` (or the
module creates one from config); tests mock the driver. Returns a list
of :class:`Hit`, which is a small dataclass with ``id``, ``score`` (the
raw lexical score for debugging — RRF only uses the rank order), and
``source`` (which retriever produced it).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default Neo4j labels + index names for the production search surface.
DEFAULT_EPISODE_LABEL = "Episode"
DEFAULT_EPISODE_FULLTEXT_INDEX = "episode_content_fulltext"
DEFAULT_PAGE_LABEL = "Page"
DEFAULT_PAGE_FULLTEXT_INDEX = "page_compiled_truth_fulltext"

# Characters that would break a Lucene query. Stripped defensively so a
# user query like "foo: bar" doesn't crash the driver with a parse error.
_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\\\/]|&&|\|\|')


@dataclass
class Hit:
    """Single keyword-search result."""

    id: str
    score: float = 0.0
    source: str = "keyword"  # "keyword:episode" or "keyword:page" in practice
    metadata: dict[str, Any] = field(default_factory=dict)


def _escape_for_lucene(query: str) -> str:
    """Strip Lucene special chars so user input is safe for fulltext search.

    We don't try to preserve operators — callers that need boolean
    operators should pass the raw query through a different path. Here
    we want a defensive, bag-of-words-ish query.
    """
    return _LUCENE_SPECIAL.sub(" ", query).strip()


def _tokenize(query: str) -> list[str]:
    """Lowercase word tokens >= 3 chars. Used by the CONTAINS fallback."""
    if not query:
        return []
    return [
        t.lower()
        for t in re.findall(r"\b\w{3,}\b", query)
    ]


def keyword_search(
    query: str,
    k: int = 10,
    namespace: Optional[str] = None,
    *,
    driver=None,
    include_pages: bool = True,
    page_label: str = DEFAULT_PAGE_LABEL,
    page_fulltext_index: str = DEFAULT_PAGE_FULLTEXT_INDEX,
    episode_fulltext_index: str = DEFAULT_EPISODE_FULLTEXT_INDEX,
) -> list[Hit]:
    """Lexical search over Neo4j episodes (+ optionally Pages).

    Args:
        query: User query. Lucene special chars are stripped defensively.
        k: Maximum hits to return.
        namespace: Neo4j label to scan. Defaults to ``"Episode"``. Tests
            and the eval harness use ``"TestEpisode"`` for isolation.
        driver: Neo4j ``Driver``. Required. Callers that want a
            best-effort behavior when Neo4j is unreachable should catch
            exceptions themselves; this function lets them bubble up.
        include_pages: When True (default) also searches Page.compiled_truth
            and returns those as hits. Page hits use the Page slug as
            ``id`` so callers can distinguish episode vs page hits by
            checking the ``source`` field.
        page_label: Override the Page node label (defaults ``"Page"``).
        page_fulltext_index: Override the Page fulltext index name.
        episode_fulltext_index: Override the Episode fulltext index name.

    Returns:
        List of :class:`Hit`, best first, capped at ``k``. Empty list on
        empty query or if no driver is available.

    Never raises on empty input. Propagates driver errors — caller owns
    retry/fallback policy.
    """
    if not query or not query.strip():
        return []

    effective_label = namespace or DEFAULT_EPISODE_LABEL
    hits: list[Hit] = []

    # ── 1. Episode channel ────────────────────────────────────────────
    episode_hits = _search_episodes(
        query=query,
        k=k,
        label=effective_label,
        index_name=episode_fulltext_index,
        driver=driver,
    )
    hits.extend(episode_hits)

    # ── 2. Page channel (Run 2 fulltext) ───────────────────────────────
    if include_pages:
        page_hits = _search_pages(
            query=query,
            k=k,
            label=page_label,
            index_name=page_fulltext_index,
            driver=driver,
        )
        hits.extend(page_hits)

    # Sort by score desc, stable tiebreak by id. Cap at k.
    hits.sort(key=lambda h: (-h.score, h.id))
    return hits[:k]


def _search_episodes(
    *,
    query: str,
    k: int,
    label: str,
    index_name: str,
    driver,
) -> list[Hit]:
    """Try Neo4j fulltext first; fall back to ``CONTAINS`` token scan."""
    if driver is None:
        return []

    # Fulltext index attempt. ``db.index.fulltext.queryNodes`` returns
    # nodes with a BM25-style score. Non-existent index raises.
    safe_q = _escape_for_lucene(query)
    if safe_q:
        try:
            with driver.session() as sess:
                records = sess.run(
                    """
                    CALL db.index.fulltext.queryNodes($index, $q)
                    YIELD node, score
                    WHERE $label IN labels(node)
                      AND coalesce(node.lifecycle_status, 'active')
                          IN ['active', 'confirmed']
                    RETURN node.uuid AS id, score, node AS n
                    ORDER BY score DESC
                    LIMIT $k
                    """,
                    index=index_name,
                    q=safe_q,
                    label=label,
                    k=k,
                )
                return [
                    Hit(
                        id=str(r["id"]),
                        score=float(r["score"]),
                        source="keyword:episode",
                        metadata=dict(r["n"]) if r["n"] else {},
                    )
                    for r in records
                    if r["id"]
                ]
        except Exception as e:  # noqa: BLE001 — fall through to CONTAINS
            logger.debug(
                "fulltext query on :%s failed (%s); falling back to CONTAINS",
                label,
                e,
            )

    # CONTAINS fallback. Score = number of distinct tokens that appear
    # in the content, normalized by token count. Cheap and deterministic.
    tokens = _tokenize(query)
    if not tokens:
        return []
    try:
        with driver.session() as sess:
            where_clauses = " OR ".join(
                f"toLower(n.content) CONTAINS $t{i}" for i, _ in enumerate(tokens)
            )
            params: dict[str, Any] = {
                f"t{i}": tok for i, tok in enumerate(tokens)
            }
            params["k"] = k
            cypher = f"""
                MATCH (n:{label})
                WHERE ({where_clauses})
                  AND coalesce(n.lifecycle_status, 'active')
                      IN ['active', 'confirmed']
                RETURN n.uuid AS id, n.content AS content, n AS node
                LIMIT $k * 4
            """
            records = list(sess.run(cypher, **params))
    except Exception as e:  # noqa: BLE001
        logger.warning("keyword CONTAINS fallback failed: %s", e)
        return []

    hits: list[Hit] = []
    for r in records:
        content = (r["content"] or "").lower()
        overlap = sum(1 for tok in tokens if tok in content)
        if overlap == 0:
            continue
        score = overlap / max(len(tokens), 1)
        node = r["node"]
        hits.append(
            Hit(
                id=str(r["id"]) if r["id"] else "",
                score=float(score),
                source="keyword:episode",
                metadata=dict(node) if node else {},
            )
        )
    hits.sort(key=lambda h: (-h.score, h.id))
    return hits[:k]


def _search_pages(
    *,
    query: str,
    k: int,
    label: str,
    index_name: str,
    driver,
) -> list[Hit]:
    """Query the Page.compiled_truth fulltext index (Run 2)."""
    if driver is None:
        return []
    safe_q = _escape_for_lucene(query)
    if not safe_q:
        return []
    try:
        with driver.session() as sess:
            records = sess.run(
                """
                CALL db.index.fulltext.queryNodes($index, $q)
                YIELD node, score
                WHERE $label IN labels(node)
                RETURN node.slug AS id, score, node AS n
                ORDER BY score DESC
                LIMIT $k
                """,
                index=index_name,
                q=safe_q,
                label=label,
                k=k,
            )
            return [
                Hit(
                    id=str(r["id"]),
                    score=float(r["score"]),
                    source="keyword:page",
                    metadata=dict(r["n"]) if r["n"] else {},
                )
                for r in records
                if r["id"]
            ]
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "Page fulltext query failed (index=%s): %s — skipping page hits",
            index_name,
            e,
        )
        return []
