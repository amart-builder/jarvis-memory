"""Search stack for jarvis-memory — RRF hybrid retrieval + intent routing.

Run 3 (2026-04-20-search-upgrade) replaces the composite-weighted scoring
inside ``scored_search`` with:

* ``rrf.reciprocal_rank_fusion`` — fuse ranked lists from multiple
  retrievers using reciprocal rank fusion (k=60 default).
* ``keyword.keyword_search`` — Neo4j full-text query on Episode.content
  and Page.compiled_truth.
* ``boosts.compiled_truth_boost`` / ``boosts.backlink_boost`` — post-RRF
  re-ranking against the Page graph.
* ``intent.classify`` — rule-based router (entity / temporal / event /
  general) deciding which retrievers to invoke.
* ``expansion.expand`` — Haiku-backed multi-query expansion with
  prompt-injection sanitization (in and out).

The public ``scored_search`` contract is unchanged — only internals move.
"""
from __future__ import annotations

from .rrf import reciprocal_rank_fusion
from .keyword import Hit, keyword_search
from .boosts import (
    apply_boosts,
    backlink_boost,
    compiled_truth_boost,
)
from .intent import classify
from .expansion import expand, sanitize_expansion_output, sanitize_query_for_prompt

__all__ = [
    "Hit",
    "apply_boosts",
    "backlink_boost",
    "classify",
    "compiled_truth_boost",
    "expand",
    "keyword_search",
    "reciprocal_rank_fusion",
    "sanitize_expansion_output",
    "sanitize_query_for_prompt",
]
