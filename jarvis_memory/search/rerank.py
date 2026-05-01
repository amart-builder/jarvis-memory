"""Cross-encoder reranking — final relevance pass after RRF + filters.

Bi-encoder retrieval (Chroma + Neo4j fulltext, fused via RRF) is fast but
imprecise. A cross-encoder scores each (query, document) pair *together*
in one transformer forward pass, attending across the full pair, so it
catches subtle relevance the bi-encoder misses. Industry-standard pattern:
retrieve N candidates with the cheap stack, rerank to top-k with the
cross-encoder. Reported lift: +5–15 nDCG@10.

**Where this slots in.** ``scored_search`` calls :func:`rerank` after
``_enrich_hits`` and ``_apply_filters`` — i.e. once the candidate set has
been narrowed to the ``group_id`` / ``room`` / ``hall`` matches with
fetched content. That keeps the cross-encoder from wasting compute on
documents that would have been filtered out anyway.

**Default model.** ``BAAI/bge-reranker-v2-m3`` — Apache-2.0, ~568MB,
multilingual, ~150ms/query at depth-50 on CPU. Override with the
``JARVIS_RERANK_MODEL`` env var (any cross-encoder identifier the
``rerankers`` package recognizes).

**Failure mode.** Reranker unavailable → return the input list unchanged
("fail open"). The retrieval pipeline never fails because of reranker
problems. Cases:
  * ``rerankers`` package not installed
  * Model download fails (no network on first call)
  * Model inference raises

**Env flags.**
  * ``JARVIS_RERANK=0`` — disable reranking, return RRF order. Default ``1``.
  * ``JARVIS_RERANK_MODEL`` — override the model identifier.
  * ``JARVIS_RERANK_DEVICE`` — force a torch device, e.g. ``cpu`` when MPS hangs.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# CPU-friendly Apache-2.0 cross-encoder. v2-m3 = "multilingual,
# multifunctional, multi-granularity" — fastest of the BGE-v2 family
# on CPU and the recommended default for production reranking in 2026.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

_model_lock = threading.Lock()
_model_singleton: Any = None
_load_attempted = False


def _enabled() -> bool:
    """Return True iff ``JARVIS_RERANK`` is not explicitly disabled."""
    return os.environ.get("JARVIS_RERANK", "1").strip() not in {"0", "false", "False", ""}


def _resolve_model_name(override: Optional[str] = None) -> str:
    """Pick the model name from arg → env → default, in that order."""
    if override:
        return override
    return os.environ.get("JARVIS_RERANK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _model_kwargs_from_env() -> dict[str, Any]:
    """Return optional model-loader kwargs from env.

    ``rerankers`` auto-selects MPS on Apple Silicon. Some long-running
    retrieval workloads can hit unhealthy MPS behavior, so operators need
    a runtime way to force CPU without changing retrieval logic.
    """
    kwargs: dict[str, Any] = {}
    device = os.environ.get("JARVIS_RERANK_DEVICE", "").strip()
    if device:
        kwargs["device"] = device
    return kwargs


def _get_model(model_name: Optional[str] = None) -> Any:
    """Lazily load and cache the cross-encoder. Thread-safe.

    Subsequent calls return the cached instance. Repeated load failures
    are not retried (we set a flag the first time the load fails so a
    cold-start outage doesn't get hammered every query).
    """
    global _model_singleton, _load_attempted
    if _model_singleton is not None:
        return _model_singleton
    if _load_attempted:
        # Already tried and failed; don't keep retrying.
        return None

    with _model_lock:
        if _model_singleton is not None:
            return _model_singleton
        if _load_attempted:
            return None
        _load_attempted = True

        try:
            from rerankers import Reranker  # type: ignore
        except ImportError:
            logger.warning(
                "rerankers package not installed — reranking disabled. "
                "Install with `pip install rerankers` to enable."
            )
            return None

        name = _resolve_model_name(model_name)
        try:
            _model_singleton = Reranker(name, **_model_kwargs_from_env())
            logger.info("loaded cross-encoder reranker: %s", name)
        except Exception as e:  # noqa: BLE001 — load is best-effort
            logger.warning(
                "failed to load reranker %r (%s); reranking disabled for this process",
                name,
                e,
            )
            return None

    return _model_singleton


def _extract_text(record: dict[str, Any]) -> str:
    """Pick the best text field to feed the cross-encoder.

    Episode nodes carry their text in ``content``. Page nodes use
    ``compiled_truth``. Some legacy records use ``name`` or ``summary``.
    Falls back to an empty string — the cross-encoder will still rank,
    just with no signal for that doc.
    """
    for key in ("content", "compiled_truth", "name", "summary", "fact"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


_DEFAULT_RERANK_CAP: int = 30
# Truncate doc text to this char count before sending to the cross-encoder.
# BGE-rerankers max out at 512 tokens (~2000 chars) and the relevance
# signal is concentrated near the head of the document. Keeping inputs
# bounded prevents MPS / CUDA buffer-allocation blowups on long
# session-sized documents (5000-10000 chars are common in conversation
# memory). 1500 chars is roughly 375 tokens, well under the model's limit.
_DEFAULT_RERANK_DOC_CHARS: int = 1500


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model_name: Optional[str] = None,
    candidate_cap: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Re-rank ``candidates`` by cross-encoder relevance to ``query``.

    Args:
        query: User query string.
        candidates: List of enriched node dicts as returned by
            ``_enrich_hits``. Each must have a usable text field
            (``content`` / ``compiled_truth`` / ``name`` / ``summary`` /
            ``fact``) — empty texts get zero rerank score.
        model_name: Override the default model. Falls back to
            ``JARVIS_RERANK_MODEL`` env var, then ``DEFAULT_MODEL``.
        candidate_cap: Maximum number of candidates to send to the
            cross-encoder in one ``rank()`` call. Defaults to 30, large
            enough for any typical retrieval depth, small enough to
            avoid MPS/CUDA buffer-allocation blowups with BGE-rerankers
            on long documents. Excess candidates retain their incoming
            RRF order at the tail of the result.

    Returns:
        The same dicts, each augmented with ``rerank_score: float``,
        sorted descending by ``rerank_score``. When the reranker is
        disabled or unavailable, the input list is returned unchanged
        with no ``rerank_score`` added (caller can detect this by key
        presence).

        Pure function in spirit — produces a new list and new dict
        instances; never mutates inputs.
    """
    if not _enabled():
        return candidates
    if not candidates or not query or not query.strip():
        return candidates

    model = _get_model(model_name)
    if model is None:
        return candidates

    cap = candidate_cap if candidate_cap is not None else _DEFAULT_RERANK_CAP
    head = candidates[:cap]
    tail = candidates[cap:]  # preserve RRF order for anything beyond the cap

    docs: list[str] = [_extract_text(c)[:_DEFAULT_RERANK_DOC_CHARS] for c in head]
    doc_ids: list[int] = list(range(len(head)))

    try:
        ranked = model.rank(query=query, docs=docs, doc_ids=doc_ids)
    except Exception as e:  # noqa: BLE001 — never let the reranker break retrieval
        logger.warning("reranker.rank failed (%s); returning input order", e)
        return candidates

    # Map original-index -> cross-encoder score. Iterate ``ranked`` rather
    # than touching ``ranked.results`` directly so we work with both the
    # current ``RankedResults`` shape and any future iterable variant.
    score_by_idx: dict[int, float] = {}
    for r in ranked:
        idx = getattr(r, "doc_id", None)
        if idx is None:
            idx = getattr(r, "document", None)
            idx = getattr(idx, "doc_id", None) if idx is not None else None
        if idx is None:
            continue
        try:
            score = float(getattr(r, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        score_by_idx[int(idx)] = score

    out: list[dict[str, Any]] = []
    for i, c in enumerate(head):
        new_c = dict(c)
        new_c["rerank_score"] = score_by_idx.get(i, 0.0)
        out.append(new_c)

    out.sort(key=lambda c: c["rerank_score"], reverse=True)
    # Append uncapped tail items unchanged — they keep their RRF order
    # and don't appear in the rerank top-K.
    out.extend(tail)
    return out


def reset_model_cache() -> None:
    """Clear the cached model. Test-only — production callers shouldn't need this."""
    global _model_singleton, _load_attempted
    with _model_lock:
        _model_singleton = None
        _load_attempted = False
