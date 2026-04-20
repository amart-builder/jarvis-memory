"""Post-RRF re-rank boosts driven by the Page knowledge graph.

After ``reciprocal_rank_fusion`` produces a fused list of document ids,
we optionally re-rank via two boosts:

* **compiled_truth_boost(hit, pages) -> score_delta**
  Multiplicative boost for hits whose matching Page has a non-empty
  ``compiled_truth``. Default factor ``1.2x``. Rationale: a Page with a
  rich compiled summary is an established entity worth surfacing.

* **backlink_boost(hit, graph) -> score_delta**
  Additive boost ``log(1 + in_degree) * 0.1`` for hits whose entity is
  well-connected — i.e. many other Pages or Episodes reference it via
  typed edges. Rationale: high in-degree implies centrality.

Both functions are pure. They take pre-computed lookup structures
(Page map + in-degree map) rather than querying Neo4j themselves, so
``scored_search`` can batch its graph I/O once and pass the results in.
``apply_boosts`` composes the two over a fused RRF list.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from ..pages import Page

__all__ = [
    "BoostConfig",
    "apply_boosts",
    "backlink_boost",
    "compiled_truth_boost",
]


# Defaults — tunable per spec Run 3 §"Assumptions" A6.
DEFAULT_COMPILED_TRUTH_FACTOR = 1.2
DEFAULT_BACKLINK_WEIGHT = 0.1
MIN_COMPILED_TRUTH_LEN = 20  # below this, treat as "empty/stub" compiled_truth


@dataclass
class BoostConfig:
    """Tunable parameters for the two post-RRF boosts.

    Kept as a dataclass so test cases and the main search pipeline both
    document their overrides explicitly instead of passing kwargs around.
    """

    compiled_truth_factor: float = DEFAULT_COMPILED_TRUTH_FACTOR
    backlink_weight: float = DEFAULT_BACKLINK_WEIGHT
    min_compiled_truth_len: int = MIN_COMPILED_TRUTH_LEN


def compiled_truth_boost(
    doc_id: str,
    base_score: float,
    page_lookup: dict[str, Page] | dict[str, str] | None,
    *,
    factor: float = DEFAULT_COMPILED_TRUTH_FACTOR,
    min_len: int = MIN_COMPILED_TRUTH_LEN,
) -> float:
    """Return a *new* score after applying the compiled-truth multiplier.

    Args:
        doc_id: Hit identifier. For Page hits (keyword:page) this is a
            Page slug. For Episode hits we look the slug up indirectly
            via the ``page_lookup`` — see below.
        base_score: The pre-boost RRF score to multiply.
        page_lookup: Mapping ``id -> Page`` (or ``id -> compiled_truth``
            string). ``None`` or an empty dict disables the boost.
        factor: Multiplier. Default 1.2x.
        min_len: Minimum compiled_truth length to count as "rich".

    Returns:
        ``base_score * factor`` if ``doc_id`` maps to a Page whose
        compiled_truth is at least ``min_len`` characters long, else
        ``base_score`` unchanged. Negative/zero scores are preserved.
    """
    if not page_lookup:
        return base_score
    entry = page_lookup.get(doc_id)
    if entry is None:
        return base_score

    if isinstance(entry, Page):
        truth = entry.compiled_truth or ""
    elif isinstance(entry, str):
        truth = entry
    else:
        # Unknown shape — don't boost.
        return base_score

    if len(truth.strip()) < min_len:
        return base_score
    return base_score * factor


def backlink_boost(
    doc_id: str,
    base_score: float,
    in_degree_lookup: dict[str, int] | None,
    *,
    weight: float = DEFAULT_BACKLINK_WEIGHT,
) -> float:
    """Return a *new* score after adding the log-scaled backlink boost.

    Args:
        doc_id: Hit identifier.
        base_score: The pre-boost RRF score.
        in_degree_lookup: Mapping ``id -> in_degree`` (number of typed
            edges pointing at the associated Page). ``None``/empty
            disables the boost.
        weight: Multiplier on ``log(1 + in_degree)``. Default 0.1.

    Returns:
        ``base_score + log(1 + in_degree) * weight`` when a positive
        in-degree is available; otherwise ``base_score`` unchanged.
    """
    if not in_degree_lookup:
        return base_score
    degree = in_degree_lookup.get(doc_id, 0)
    if degree <= 0:
        return base_score
    return base_score + math.log(1 + degree) * weight


def apply_boosts(
    fused: Iterable[tuple[str, float]],
    page_lookup: Optional[dict[str, Page] | dict[str, str]] = None,
    in_degree_lookup: Optional[dict[str, int]] = None,
    *,
    config: Optional[BoostConfig] = None,
) -> list[tuple[str, float]]:
    """Apply both boosts to a fused RRF list and resort.

    Args:
        fused: Iterable of ``(doc_id, score)`` pairs (e.g. the output of
            :func:`~jarvis_memory.search.rrf.reciprocal_rank_fusion`).
        page_lookup: ``doc_id -> Page`` or ``doc_id -> compiled_truth``.
            Optional — when omitted, the compiled-truth boost is a no-op.
        in_degree_lookup: ``doc_id -> int``. Optional — when omitted, the
            backlink boost is a no-op.
        config: Override the boost weights. Defaults to ``BoostConfig()``.

    Returns:
        New list of ``(doc_id, boosted_score)`` sorted by descending
        score, with alphabetical tiebreak on ``doc_id``.
    """
    cfg = config or BoostConfig()

    boosted: list[tuple[str, float]] = []
    for doc_id, score in fused:
        s = compiled_truth_boost(
            doc_id,
            score,
            page_lookup,
            factor=cfg.compiled_truth_factor,
            min_len=cfg.min_compiled_truth_len,
        )
        s = backlink_boost(
            doc_id, s, in_degree_lookup, weight=cfg.backlink_weight
        )
        boosted.append((doc_id, s))

    boosted.sort(key=lambda kv: (-kv[1], kv[0]))
    return boosted
