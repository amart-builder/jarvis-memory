"""Typed-edge extraction — deterministic, no LLM on the hot path.

``extract_typed_edges(content, episode_type)`` scans text for phrase
patterns that correspond to one of the 8 typed edges in ``schema_v2``
(plus the ``MENTIONS`` fallback) and returns a list of ``TypedEdge``.

Design constraints
------------------
* **Pure function.** No Neo4j, no LLM, no network. Idempotent given
  identical input.
* **Deterministic.** Regex + keyword + episode_type hints. Order and
  grouping are stable so tests can assert exact expected lists.
* **Confidence threshold 0.6.** Edges scored below this are dropped —
  we persist only signals we're reasonably sure about.
* **No false edges.** Prefer to miss a signal over to invent one. Agents
  can always re-read the underlying episode; a spurious ``WORKS_AT``
  poisons the graph.

Edge taxonomy (from ``schema_v2.TYPED_EDGES``)
    ATTENDED, WORKS_AT, INVESTED_IN, FOUNDED, ADVISES, DECIDED_ON,
    MENTIONS, REFERS_TO
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Optional

from .pages import slugify
from .schema_v2 import TYPED_EDGES

logger = logging.getLogger(__name__)

# Minimum confidence to emit an edge. Spec §"Constraints".
MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class TypedEdge:
    """A single typed edge extracted from an episode.

    ``from_slug`` and ``to_slug`` are Page slugs. ``from_slug`` may be the
    speaker/subject inferred from the sentence; when that's unavailable
    the edge is keyed on an anchor slug (see ``anchor``).
    """

    from_slug: str
    edge_type: str
    to_slug: str
    confidence: float
    evidence: str = ""  # the phrase snippet that matched

    def to_dict(self) -> dict:
        return asdict(self)


# ── Proper-noun detection ────────────────────────────────────────────────
#
# A conservative proper-noun pattern: 1–3 capitalized tokens in a row.
# We deliberately skip single-letter tokens and all-caps acronyms (those
# are noise in prose — e.g. "DECISION" in "[DECISION]").
_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})\b"
)

# Common English words that incidentally start a sentence and get flagged
# as proper nouns. We strip these post-match.
_SENTENCE_INITIAL_STOPWORDS = {
    "the", "this", "that", "these", "those", "a", "an",
    "i", "we", "you", "they", "he", "she", "it",
    "and", "but", "or", "so", "yet", "for", "nor",
    "in", "on", "at", "by", "to", "of", "with",
    "after", "before", "during", "while", "since",
    "today", "tomorrow", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "yes", "no", "maybe", "okay", "ok",
    "please", "thanks", "thank",
    "note", "todo", "done",
    "approved", "rejected", "decided", "completed", "shipped", "deployed",
    "plan", "plans", "status", "update", "updates", "context", "notes",
    "alex",  # exclude the owner — not an external entity
    "claude", "openclaw",  # agents, not external entities
    "run", "phase", "task", "week", "month", "year", "day",
    "fact", "decision", "correction", "handoff", "milestone",
    "meeting", "email", "message", "thread",
    "why", "what", "where", "when", "who", "how", "which",
}


def _extract_proper_nouns(text: str) -> list[tuple[str, int]]:
    """Return list of (proper_noun, start_offset) from text.

    Filtered to exclude common sentence-initial noise.
    """
    found: list[tuple[str, int]] = []
    for m in _PROPER_NOUN.finditer(text):
        token = m.group(1)
        head = token.split()[0].lower()
        if head in _SENTENCE_INITIAL_STOPWORDS:
            continue
        # Drop single-char trailing fragments.
        if len(token) < 3:
            continue
        found.append((token, m.start()))
    return found


# ── Per-edge-type phrase patterns ────────────────────────────────────────
#
# Each pattern captures two groups where possible:
#   (1) subject / from — the "who"
#   (2) object / to — the "what"
# When the subject isn't grammatically present, patterns use a placeholder
# ``_anchor`` slug supplied by the caller (derived from episode context).
#
# Patterns are ordered from most-specific to least-specific; earlier
# matches take precedence in ``extract_typed_edges``.

_PATTERNS: dict[str, list[tuple[re.Pattern, float]]] = {
    "WORKS_AT": [
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+works at\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)", re.IGNORECASE | re.UNICODE), 0.9),
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?),?\s+(?:a|an)\s+[\w\s\-]+(?:at|of)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.75),
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:is|was)\s+(?:a|an|the)?\s*[\w\s\-]*employee(?:\s+of|\s+at)\s+([A-Z][\w&\-\.]+)"), 0.8),
    ],
    "ATTENDED": [
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+attended\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.85),
        (re.compile(r"\bmet with\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:at|in|during)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.75),
    ],
    "INVESTED_IN": [
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+invested\s+(?:in|\$[\d,]+\s+in)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.9),
        (re.compile(r"\binvestment\s+(?:from\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?))?\s*(?:in|into)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.75),
    ],
    "FOUNDED": [
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+founded\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.9),
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:is|was)\s+(?:the\s+)?founder\s+of\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.85),
    ],
    "ADVISES": [
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+advises\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.9),
        (re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:is|was)\s+(?:an?\s+)?advisor\s+(?:to|of|for)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)*)"), 0.85),
    ],
    "DECIDED_ON": [
        # Subject implicit — anchor is the episode's group_id page.
        (re.compile(r"\[decision\].*?(?:decided|chose|picked|going with)\s+(?:to\s+use\s+)?([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)?)", re.IGNORECASE | re.DOTALL), 0.85),
        (re.compile(r"\bdecided to (?:use|go with|adopt|switch to)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)?)"), 0.75),
    ],
    "REFERS_TO": [
        # Explicit cross-reference language.
        (re.compile(r"\b(?:see|refer to|cf\.|c\.f\.|reference)\s+([A-Z][\w&\-\.]+(?:\s+[A-Z][\w&\-\.]+)?)"), 0.7),
    ],
}


def _derive_anchor(
    episode_type: Optional[str],
    group_id: Optional[str],
    content: str,
) -> str:
    """Pick an anchor slug for edges whose subject isn't explicit.

    Order of preference:
      1. group_id (the project scope on the episode).
      2. First proper noun in the content.
      3. episode_type-derived slug.
      4. Fallback ``"episode"``.
    """
    if group_id:
        return slugify(group_id)
    nouns = _extract_proper_nouns(content or "")
    if nouns:
        return slugify(nouns[0][0])
    if episode_type:
        return slugify(episode_type)
    return "episode"


def _domain_for_slug(slug: str, edge_type: str, is_subject: bool) -> str:
    """Best-effort domain labeling for a page created from an extracted entity.

    This is heuristic — callers that have stronger signals should overwrite.
    """
    # Edge-type-driven defaults for the *object* side.
    if not is_subject:
        if edge_type in ("WORKS_AT", "INVESTED_IN", "FOUNDED", "ADVISES"):
            return "company"
        if edge_type == "ATTENDED":
            return "event"
        if edge_type == "DECIDED_ON":
            return "concept"
        if edge_type in ("REFERS_TO", "MENTIONS"):
            return "topic"
    # Subject side is usually a person in the WORKS_AT/FOUNDED/ADVISES triad.
    if edge_type in ("WORKS_AT", "FOUNDED", "ADVISES", "INVESTED_IN", "ATTENDED"):
        return "person"
    return "concept"


def extract_typed_edges(
    content: str,
    episode_type: Optional[str] = None,
    group_id: Optional[str] = None,
) -> list[TypedEdge]:
    """Extract typed edges from an episode.

    Args:
        content: Episode body. May be multi-line, may contain tags like
            ``[DECISION]`` / ``[MEETING]``. Pattern matching is
            case-insensitive for the keyword anchors; proper-noun
            capture is case-sensitive.
        episode_type: Caller-supplied or classified type. Used to boost
            confidence on ``DECIDED_ON`` when ``episode_type == "decision"``.
        group_id: Project scope (optional). Used as anchor for edges
            without an explicit subject.

    Returns:
        List of ``TypedEdge`` with confidence ``>= MIN_CONFIDENCE``.
        Deduplicated on ``(from_slug, edge_type, to_slug)``; the highest
        confidence instance wins.

    Never raises on malformed input. Returns ``[]`` on exception.
    """
    if not content:
        return []
    try:
        return _extract_typed_edges_impl(content, episode_type, group_id)
    except Exception as e:  # defensive — pattern bugs shouldn't break writes
        logger.error(f"extract_typed_edges failed: {e}")
        return []


def _extract_typed_edges_impl(
    content: str,
    episode_type: Optional[str],
    group_id: Optional[str],
) -> list[TypedEdge]:
    anchor = _derive_anchor(episode_type, group_id, content)
    et_lower = (episode_type or "").strip().lower()

    raw: list[TypedEdge] = []

    # Boost the DECIDED_ON pattern when episode_type is explicitly decision.
    decision_boost = 0.1 if et_lower == "decision" else 0.0

    # ── Pattern-based extraction (per edge type) ─────────────────────
    for edge_type, patterns in _PATTERNS.items():
        for pattern, base_conf in patterns:
            for m in pattern.finditer(content):
                groups = [g for g in m.groups() if g]
                if edge_type == "DECIDED_ON":
                    # DECIDED_ON: one captured object group; anchor is subject.
                    obj = groups[0] if groups else None
                    if not obj:
                        continue
                    from_slug = anchor
                    to_slug = slugify(obj)
                    if not to_slug or to_slug == from_slug:
                        continue
                    raw.append(
                        TypedEdge(
                            from_slug=from_slug,
                            edge_type=edge_type,
                            to_slug=to_slug,
                            confidence=min(0.95, base_conf + decision_boost),
                            evidence=m.group(0)[:120],
                        )
                    )
                elif edge_type == "REFERS_TO":
                    obj = groups[0] if groups else None
                    if not obj:
                        continue
                    from_slug = anchor
                    to_slug = slugify(obj)
                    if not to_slug or to_slug == from_slug:
                        continue
                    raw.append(
                        TypedEdge(
                            from_slug=from_slug,
                            edge_type=edge_type,
                            to_slug=to_slug,
                            confidence=base_conf,
                            evidence=m.group(0)[:120],
                        )
                    )
                else:
                    # Two-group patterns: subject + object.
                    if len(groups) < 2:
                        continue
                    subj, obj = groups[0], groups[1]
                    from_slug = slugify(subj)
                    to_slug = slugify(obj)
                    if not from_slug or not to_slug or from_slug == to_slug:
                        continue
                    raw.append(
                        TypedEdge(
                            from_slug=from_slug,
                            edge_type=edge_type,
                            to_slug=to_slug,
                            confidence=base_conf,
                            evidence=m.group(0)[:120],
                        )
                    )

    # ── MENTIONS fallback ────────────────────────────────────────────
    # Every proper noun that didn't already appear as an object of a
    # typed edge gets a MENTIONS edge from the anchor. Low confidence
    # relative to typed edges; still persisted when >= 0.6.
    seen_object_slugs = {e.to_slug for e in raw}
    proper_nouns = _extract_proper_nouns(content)
    for noun, _pos in proper_nouns:
        to_slug = slugify(noun)
        if not to_slug or to_slug in seen_object_slugs or to_slug == anchor:
            continue
        # MENTIONS confidence scales with the number of times the noun
        # appears. 1x = 0.6, 2x = 0.7, 3x+ = 0.8.
        count = sum(1 for n, _ in proper_nouns if slugify(n) == to_slug)
        conf = min(0.8, 0.5 + 0.1 * count)
        if conf < MIN_CONFIDENCE:
            continue
        raw.append(
            TypedEdge(
                from_slug=anchor,
                edge_type="MENTIONS",
                to_slug=to_slug,
                confidence=conf,
                evidence=noun,
            )
        )
        seen_object_slugs.add(to_slug)

    # ── Threshold + dedupe ───────────────────────────────────────────
    filtered = [e for e in raw if e.confidence >= MIN_CONFIDENCE]
    # Dedupe on (from, type, to); keep the highest-confidence.
    best: dict[tuple[str, str, str], TypedEdge] = {}
    for e in filtered:
        key = (e.from_slug, e.edge_type, e.to_slug)
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e
    # Stable order: by edge type (canonical), then from_slug, then to_slug.
    edge_type_order = {t: i for i, t in enumerate((*TYPED_EDGES,))}
    return sorted(
        best.values(),
        key=lambda e: (edge_type_order.get(e.edge_type, 99), e.from_slug, e.to_slug),
    )


def create_edges_in_tx(
    tx,
    edges: list[TypedEdge],
    from_label: str = "Page",
    to_label: str = "Page",
) -> int:
    """Materialize edges as Neo4j relationships.

    Creates the endpoint Pages if missing (ambient Page creation per spec),
    then MERGEs one edge per TypedEdge. Uses the caller-supplied ``tx``.

    Returns the number of edges that were processed (not necessarily
    new — MERGE is idempotent).
    """
    if not edges:
        return 0
    count = 0
    for e in edges:
        query = f"""
            MERGE (from:{from_label} {{slug: $from_slug}})
            ON CREATE SET
                from.domain = $from_domain,
                from.compiled_truth = '',
                from.created_at = datetime(),
                from.updated_at = datetime()
            MERGE (to:{to_label} {{slug: $to_slug}})
            ON CREATE SET
                to.domain = $to_domain,
                to.compiled_truth = '',
                to.created_at = datetime(),
                to.updated_at = datetime()
            MERGE (from)-[r:{e.edge_type}]->(to)
            ON CREATE SET
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.created_at = datetime()
            ON MATCH SET
                r.confidence = CASE WHEN r.confidence < $confidence THEN $confidence ELSE r.confidence END,
                r.evidence = coalesce(nullif(r.evidence, ''), $evidence)
            RETURN r
        """
        try:
            tx.run(
                query,
                from_slug=e.from_slug,
                to_slug=e.to_slug,
                from_domain=_domain_for_slug(e.from_slug, e.edge_type, True),
                to_domain=_domain_for_slug(e.to_slug, e.edge_type, False),
                confidence=e.confidence,
                evidence=e.evidence or "",
            )
            count += 1
        except Exception as exc:  # noqa: BLE001 — log, continue
            logger.error(f"create_edge {e} failed: {exc}")
    return count
