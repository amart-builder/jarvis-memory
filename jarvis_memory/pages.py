"""Page CRUD — compiled-truth + append-only timeline per entity.

Every entity referenced in an episode gets a ``:Page`` node:

    (:Page {
        slug: "foundry",              # canonical identifier, unique
        domain: "company",            # person / company / project / concept / ...
        compiled_truth: "Foundry is...", # agent-authored summary (<= 2000 chars)
        created_at: datetime,
        updated_at: datetime
    })

Pages link to Episodes via ``(:Page)-[:EVIDENCED_BY]->(:Episode)`` —
the append-only timeline of supporting evidence. The ``compiled_truth``
is the *current* summary (a mutable pointer); ``EVIDENCED_BY`` is the
immutable evidence trail.

All functions here operate via an existing ``neo4j.Driver``. The caller
owns driver lifecycle. Every mutation is wrapped in a Neo4j transaction
scope; callers can also pass ``tx`` to participate in an outer tx (see
``record_episode`` in ``conversation.py`` for the canonical use case).

Threading
---------
Neo4j ``Driver`` is thread-safe; sessions are not. Each function here
grabs its own session unless ``tx`` is supplied.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .schema_v2 import PAGE_LABEL, EVIDENCE_EDGE

logger = logging.getLogger(__name__)

# Maximum length of Page.compiled_truth. Spec §"Constraints".
COMPILED_TRUTH_MAX_CHARS = 2000

# Slug pattern — what we accept as a canonical entity identifier.
# Conservative: lowercase letters, digits, hyphens.
_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def slugify(name: str) -> str:
    """Convert an arbitrary entity name to a Page slug.

    Lowercases, strips non-word chars, collapses whitespace/punctuation
    to single hyphens. Not guaranteed to round-trip; use only when a
    canonical slug isn't already known.
    """
    if not name:
        return ""
    s = name.strip().lower()
    # Replace anything that isn't letter/digit with a hyphen
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:80]  # keep slugs short


def is_valid_slug(slug: str) -> bool:
    """Cheap validation — used by ``put_page`` to reject garbage inputs."""
    if not slug or not isinstance(slug, str):
        return False
    if len(slug) > 80:
        return False
    return bool(_SLUG_PATTERN.match(slug))


@dataclass
class Page:
    """In-memory view of a :Page node."""

    slug: str
    domain: str
    compiled_truth: str
    created_at: str
    updated_at: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Page":
        """Build from a Neo4j record-as-dict (all props).

        Timestamps may come back as Neo4j DateTime objects — we str() them
        so the dataclass always holds ISO strings.
        """
        return cls(
            slug=record.get("slug") or "",
            domain=record.get("domain") or "",
            compiled_truth=record.get("compiled_truth") or "",
            created_at=str(record.get("created_at") or ""),
            updated_at=str(record.get("updated_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "domain": self.domain,
            "compiled_truth": self.compiled_truth,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_truth(text: Optional[str]) -> str:
    if not text:
        return ""
    if len(text) <= COMPILED_TRUTH_MAX_CHARS:
        return text
    return text[:COMPILED_TRUTH_MAX_CHARS]


def _run_or_fail(runner, query: str, **params) -> Any:
    """Thin wrapper so we can pass either a session or a transaction."""
    return runner.run(query, **params)


def get_page(
    slug: str,
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> Optional[Page]:
    """Return the Page with this slug, or None."""
    if not slug:
        return None

    query = f"MATCH (p:{label} {{slug: $slug}}) RETURN p"
    if tx is not None:
        rec = _run_or_fail(tx, query, slug=slug).single()
        if rec is None:
            return None
        return Page.from_record(dict(rec["p"]))

    if driver is None:
        raise ValueError("get_page requires either driver or tx")
    try:
        with driver.session() as sess:
            rec = sess.run(query, slug=slug).single()
            if rec is None:
                return None
            return Page.from_record(dict(rec["p"]))
    except Exception as e:
        logger.error(f"get_page({slug!r}) failed: {e}")
        return None


def put_page(
    slug: str,
    domain: str,
    compiled_truth: Optional[str] = None,
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> Optional[Page]:
    """Create if missing; otherwise update domain + compiled_truth.

    - If the Page doesn't exist, creates with ``compiled_truth or ""``.
    - If it exists and ``compiled_truth is None``, does NOT touch it
      (this is the "ambient create-on-reference" path — we don't want
      to clobber a real summary with empty).
    - If ``compiled_truth`` is provided, it replaces the existing value
      (clamped to ``COMPILED_TRUTH_MAX_CHARS``).

    Returns the current Page, or None on error.
    """
    if not is_valid_slug(slug):
        logger.warning(f"put_page: invalid slug {slug!r}, skipping")
        return None

    now = _now_iso()
    clamped = _clamp_truth(compiled_truth) if compiled_truth is not None else None

    # MERGE with ON CREATE / ON MATCH semantics.
    if clamped is not None:
        query = f"""
            MERGE (p:{label} {{slug: $slug}})
            ON CREATE SET
                p.domain = $domain,
                p.compiled_truth = $compiled_truth,
                p.created_at = datetime($now),
                p.updated_at = datetime($now)
            ON MATCH SET
                p.domain = $domain,
                p.compiled_truth = $compiled_truth,
                p.updated_at = datetime($now)
            RETURN p
        """
        params = {"slug": slug, "domain": domain, "compiled_truth": clamped, "now": now}
    else:
        # Create if missing; don't touch compiled_truth if present.
        query = f"""
            MERGE (p:{label} {{slug: $slug}})
            ON CREATE SET
                p.domain = $domain,
                p.compiled_truth = '',
                p.created_at = datetime($now),
                p.updated_at = datetime($now)
            ON MATCH SET
                p.domain = coalesce(p.domain, $domain),
                p.updated_at = datetime($now)
            RETURN p
        """
        params = {"slug": slug, "domain": domain, "now": now}

    try:
        if tx is not None:
            rec = _run_or_fail(tx, query, **params).single()
            if rec is None:
                return None
            return Page.from_record(dict(rec["p"]))

        if driver is None:
            raise ValueError("put_page requires either driver or tx")
        with driver.session() as sess:
            rec = sess.run(query, **params).single()
            if rec is None:
                return None
            return Page.from_record(dict(rec["p"]))
    except Exception as e:
        logger.error(f"put_page({slug!r}) failed: {e}")
        return None


def append_timeline_entry(
    slug: str,
    episode_uuid: str,
    at: Optional[str] = None,
    summary: Optional[str] = None,
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> bool:
    """Add ``(:Page)-[:EVIDENCED_BY]->(:Episode)`` with metadata.

    Args:
        slug: Page slug (Page must already exist).
        episode_uuid: Episode UUID (Episode must already exist).
        at: ISO timestamp for when this evidence applies (defaults to now).
        summary: Optional short description of why this episode supports
            this page. Stored as an edge property (not searched).

    Returns True on success.
    """
    if not slug or not episode_uuid:
        return False

    at = at or _now_iso()

    # MERGE the edge so repeated calls for the same (page, episode) pair
    # don't create duplicate edges. We update ``at``/``summary`` on match
    # only if new values are non-empty — conservative on overwrite.
    query = f"""
        MATCH (p:{label} {{slug: $slug}})
        MATCH (e:Episode {{uuid: $euuid}})
        MERGE (p)-[r:{EVIDENCE_EDGE}]->(e)
        ON CREATE SET r.at = datetime($at), r.summary = $summary
        ON MATCH SET
            r.at = coalesce(r.at, datetime($at)),
            r.summary = coalesce(nullif(r.summary, ''), $summary)
        RETURN r
    """
    params = {"slug": slug, "euuid": episode_uuid, "at": at, "summary": summary or ""}

    try:
        if tx is not None:
            result = _run_or_fail(tx, query, **params)
            rec = result.single()
            return rec is not None

        if driver is None:
            raise ValueError("append_timeline_entry requires either driver or tx")
        with driver.session() as sess:
            rec = sess.run(query, **params).single()
            return rec is not None
    except Exception as e:
        logger.error(f"append_timeline_entry({slug!r}, {episode_uuid!r}) failed: {e}")
        return False


def list_pages(
    domain: Optional[str] = None,
    *,
    driver=None,
    tx=None,
    limit: int = 100,
    label: str = PAGE_LABEL,
) -> list[Page]:
    """List Pages, optionally filtered by domain, newest-first by updated_at."""
    if domain:
        query = f"""
            MATCH (p:{label})
            WHERE p.domain = $domain
            RETURN p
            ORDER BY p.updated_at DESC
            LIMIT $lim
        """
        params = {"domain": domain, "lim": limit}
    else:
        query = f"""
            MATCH (p:{label})
            RETURN p
            ORDER BY p.updated_at DESC
            LIMIT $lim
        """
        params = {"lim": limit}

    try:
        if tx is not None:
            rows = _run_or_fail(tx, query, **params)
            return [Page.from_record(dict(r["p"])) for r in rows]

        if driver is None:
            raise ValueError("list_pages requires either driver or tx")
        with driver.session() as sess:
            rows = sess.run(query, **params)
            return [Page.from_record(dict(r["p"])) for r in rows]
    except Exception as e:
        logger.error(f"list_pages failed: {e}")
        return []


def count_pages(
    *,
    driver=None,
    tx=None,
    label: str = PAGE_LABEL,
) -> int:
    """Return total Page count. Used by doctor + orphans tests."""
    query = f"MATCH (p:{label}) RETURN count(p) AS n"
    try:
        if tx is not None:
            rec = _run_or_fail(tx, query).single()
            return int(rec["n"]) if rec else 0
        if driver is None:
            raise ValueError("count_pages requires either driver or tx")
        with driver.session() as sess:
            rec = sess.run(query).single()
            return int(rec["n"]) if rec else 0
    except Exception as e:
        logger.error(f"count_pages failed: {e}")
        return 0
