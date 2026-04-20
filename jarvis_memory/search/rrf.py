"""Reciprocal Rank Fusion — combine multiple ranked lists into one.

The RRF formula (Cormack, Clarke, Buettcher 2009):

    rrf_score(d) = sum over all rankers r of 1 / (k + rank_r(d))

where ``rank_r(d)`` is the 1-indexed position of document ``d`` in ranker
``r``'s output (or +inf if the ranker didn't return ``d``). ``k`` is a
smoothing constant; 60 is the literature default and what Run 3 uses.

This module is a pure function — no I/O, no DB, no randomness. It is
small on purpose so the internal ``scored_search`` rewrite can compose it
freely.
"""
from __future__ import annotations

from typing import Iterable

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of document ids into one RRF-scored list.

    Args:
        rankings: Iterable of ranked lists. Each inner list is a sequence
            of document ids (strings), best first. Empty inner lists are
            tolerated and contribute nothing. ``None`` is tolerated in
            place of an inner list and treated as empty.
        k: RRF smoothing constant. Must be positive; lower values make
            top-of-list positions dominate more aggressively. Default 60.

    Returns:
        List of ``(doc_id, score)`` tuples sorted by descending score,
        then by document id (stable tiebreak). Each doc appears at most
        once; duplicate appearances within a single ranker are ignored
        beyond the first occurrence.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    scores: dict[str, float] = {}

    for ranker_list in rankings:
        if not ranker_list:
            continue
        # Dedupe within a single ranker — repeated hits shouldn't compound
        # one ranker's confidence. RRF treats each ranker as one vote per
        # doc; we honor that.
        seen_in_ranker: set[str] = set()
        rank = 0
        for doc_id in ranker_list:
            if doc_id is None:
                continue
            # Coerce to str for safety — callers sometimes hand us
            # numpy/neo4j id objects.
            sid = str(doc_id)
            if not sid:
                continue
            if sid in seen_in_ranker:
                continue
            seen_in_ranker.add(sid)
            rank += 1
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)

    # Sort by descending score, then alphabetically by id for stable
    # ordering on ties.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
