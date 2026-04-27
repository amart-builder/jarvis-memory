"""Conversation persistence — sessions, episodes, and snapshots.

Three-level hierarchy for cross-device session continuity:
  Project (group_id) → Session → Episodes

Sessions are first-class nodes in Neo4j with CONTINUES_FROM edges
for cross-device chaining. Episodes capture meaningful exchanges.
Snapshots capture structured state for fast context reconstruction.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    DEVICE_ID, SESSION_CHAIN_DEPTH, EPISODE_MIN_LENGTH,
    MAX_EPISODES_PER_SESSION, SNAPSHOT_MAX_SIZE,
)
from .classifier import classify_memory, detect_layer, extract_entity_references
from .graph import extract_typed_edges, create_edges_in_tx
from .pages import put_page, append_timeline_entry, PAGE_LABEL

logger = logging.getLogger(__name__)


# ── Episode significance heuristic keywords ────────────────────────────

_SIGNIFICANT_KEYWORDS = [
    # Decisions
    "decided", "decision", "chose", "agreed", "going with", "let's use",
    "approach", "strategy", "picked",
    # Plans
    "plan", "roadmap", "next step", "phase", "milestone", "will do",
    "todo", "task",
    # Completions
    "done", "finished", "completed", "shipped", "deployed", "merged",
    "implemented", "built", "created", "wrote",
    # Blockers
    "blocked", "issue", "problem", "error", "failed", "bug", "can't",
    "doesn't work", "broken",
    # Context
    "because", "reason", "why", "trade-off", "trade off", "instead of",
    "rather than", "important", "key insight", "learned",
    # Architecture
    "architecture", "design", "schema", "model", "structure", "pattern",
    "api", "endpoint", "database", "migration",
]


class SessionManager:
    """Manages session lifecycle in Neo4j.

    Sessions are conversation-level nodes that group episodes
    and enable cross-device continuity via CONTINUES_FROM edges.
    """

    def __init__(self, driver=None):
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

    def close(self):
        if self._owns_driver and self._driver:
            self._driver.close()

    def create_session(
        self,
        group_id: str,
        device: str = DEVICE_ID,
        task_summary: str = "",
        continues_from: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new session node.

        Args:
            group_id: Project identifier.
            device: Device creating this session ("macbook-pro" or "mac-mini").
            task_summary: One-line description of the session's purpose.
            continues_from: UUID of the previous session (for cross-device linking).

        Returns:
            Dict with session uuid, group_id, device, started_at.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        try:
            with self._driver.session() as db:
                db.run(
                    """
                    CREATE (s:Session {
                        uuid: $uuid,
                        group_id: $group_id,
                        device: $device,
                        started_at: datetime($started_at),
                        ended_at: null,
                        status: 'active',
                        task_summary: $task_summary,
                        continues_from: $continues_from,
                        episode_count: 0
                    })
                    """,
                    uuid=session_id,
                    group_id=group_id,
                    device=device,
                    started_at=now.isoformat(),
                    task_summary=task_summary,
                    continues_from=continues_from or "",
                )

                # Create CONTINUES_FROM edge if linking sessions
                if continues_from:
                    db.run(
                        """
                        MATCH (new:Session {uuid: $new_id})
                        MATCH (prev:Session {uuid: $prev_id})
                        CREATE (new)-[:CONTINUES_FROM {
                            at: datetime($at),
                            device_from: prev.device,
                            device_to: $device
                        }]->(prev)
                        """,
                        new_id=session_id,
                        prev_id=continues_from,
                        at=now.isoformat(),
                        device=device,
                    )

            logger.info(
                f"Session created: {session_id} for {group_id} on {device}"
                + (f" (continues {continues_from[:8]})" if continues_from else "")
            )

            return {
                "uuid": session_id,
                "group_id": group_id,
                "device": device,
                "started_at": now.isoformat(),
                "status": "active",
                "continues_from": continues_from,
            }

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return {"error": str(e)}

    def end_session(
        self,
        session_id: str,
        status: str = "completed",
    ) -> bool:
        """Mark a session as ended.

        Args:
            session_id: UUID of the session to end.
            status: Final status ("completed" or "interrupted").

        Returns:
            True if successful.
        """
        now = datetime.now(timezone.utc)

        try:
            with self._driver.session() as db:
                result = db.run(
                    """
                    MATCH (s:Session {uuid: $uuid})
                    SET s.ended_at = datetime($ended_at),
                        s.status = $status
                    RETURN s.uuid AS uuid
                    """,
                    uuid=session_id,
                    ended_at=now.isoformat(),
                    status=status,
                )
                if result.single() is None:
                    logger.warning(f"Session {session_id} not found")
                    return False

            logger.info(f"Session {session_id} ended with status: {status}")
            return True

        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False

    def get_latest_session(
        self,
        group_id: str,
        include_active: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Get the most recent session for a project.

        Args:
            group_id: Project identifier.
            include_active: Whether to include currently active sessions.

        Returns:
            Session dict or None.
        """
        try:
            with self._driver.session() as db:
                status_filter = "['completed', 'interrupted', 'active']" if include_active else "['completed', 'interrupted']"
                result = db.run(
                    f"""
                    MATCH (s:Session {{group_id: $group_id}})
                    WHERE s.status IN {status_filter}
                    RETURN s
                    ORDER BY s.started_at DESC
                    LIMIT 1
                    """,
                    group_id=group_id,
                )
                record = result.single()
                if record is None:
                    return None

                node = record["s"]
                return {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(node).items()}

        except Exception as e:
            logger.error(f"Failed to get latest session for {group_id}: {e}")
            return None

    def get_session_chain(
        self,
        session_id: str,
        depth: int = SESSION_CHAIN_DEPTH,
    ) -> list[dict[str, Any]]:
        """Follow CONTINUES_FROM edges to get the session chain.

        Args:
            session_id: Starting session UUID.
            depth: How many previous sessions to retrieve.

        Returns:
            List of session dicts, most recent first.
        """
        try:
            with self._driver.session() as db:
                result = db.run(
                    """
                    MATCH path = (s:Session {uuid: $uuid})-[:CONTINUES_FROM*0..""" + str(depth) + """]->(prev:Session)
                    UNWIND nodes(path) AS node
                    WITH DISTINCT node
                    RETURN node
                    ORDER BY node.started_at DESC
                    """,
                    uuid=session_id,
                )
                sessions = []
                for record in result:
                    node = record["node"]
                    sessions.append(
                        {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(node).items()}
                    )
                return sessions

        except Exception as e:
            logger.error(f"Failed to get session chain for {session_id}: {e}")
            return []

    def list_sessions(
        self,
        group_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List recent sessions for a project.

        Args:
            group_id: Project identifier.
            limit: Maximum sessions to return.

        Returns:
            List of session dicts, most recent first.
        """
        try:
            with self._driver.session() as db:
                result = db.run(
                    """
                    MATCH (s:Session {group_id: $group_id})
                    RETURN s
                    ORDER BY s.started_at DESC
                    LIMIT $limit
                    """,
                    group_id=group_id,
                    limit=limit,
                )
                sessions = []
                for record in result:
                    node = record["s"]
                    sessions.append(
                        {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(node).items()}
                    )
                return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions for {group_id}: {e}")
            return []


class EpisodeRecorder:
    """Records conversation episodes within a session.

    Episodes capture meaningful exchanges — decisions, plans, completions,
    blockers, and key context. Trivial exchanges are filtered by the
    should_record() heuristic.
    """

    def __init__(self, driver=None):
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

    def close(self):
        if self._owns_driver and self._driver:
            self._driver.close()

    def record_episode(
        self,
        session_id: str,
        content: str,
        episode_type: Optional[str] = None,
        group_id: Optional[str] = None,
        importance: float = 0.8,
        agent_id: Optional[str] = None,
        ctx=None,  # Run 4: OperationContext | None — see EOF trust-boundary block.
    ) -> Optional[str]:
        """Record a conversation episode.

        Args:
            session_id: Parent session UUID.
            content: The episode content (decision, plan, context, etc.).
            episode_type: Memory type. Auto-classified if not provided.
            group_id: Project group. Inherited from session if not provided.
            importance: Importance score (0-1).
            agent_id: Logical writer identity (``"claude-code"`` /
                ``"openclaw"`` / ``"cron"`` / ``"hooks"`` / etc.). When
                None, derived from ``ctx`` (mcp→claude-code,
                rest→openclaw, cli→cli) or stored as ``"unknown"``.
                Used for "which system wrote this?" filtering and
                cross-system debugging.
            ctx: Optional OperationContext (Run 4 trust boundary). When the
                caller provides a ``ctx`` with ``remote=True`` and the
                content trips an abuse heuristic, a WARNING is logged.
                LOGGED-ONLY — never refuses. ``None`` (default) skips the
                check for backward compatibility.

        Returns:
            Episode UUID if saved, None if filtered or failed.
        """
        # Run 4: trust-boundary audit — log-only, never refuses. Implementation
        # lives at EOF in the `# === TRUST BOUNDARY — RUN 4 ===` block.
        try:
            self._audit_remote_write(ctx, content, group_id, episode_type)
        except Exception as _aud_err:  # noqa: BLE001 — advisory only
            logger.debug(f"trust-boundary audit failed (non-blocking): {_aud_err}")

        if not self.should_record(content):
            logger.debug(f"Episode filtered (not significant enough): {content[:50]}")
            return None

        if episode_type is None:
            episode_type = classify_memory(content)

        # Run 1 routing warning: predict the correct layer and log a
        # structured WARNING if the content looks mis-routed. Never
        # blocks, never raises — just a signal for later audit.
        try:
            layer, layer_conf = detect_layer(content, episode_type)
            if layer != "world_knowledge" and layer_conf > 0.7:
                target = {
                    "agent_operations": "Claude auto-memory (.claude/memory/*.md)",
                    "session_ephemeral": "current session context (no persistence)",
                }.get(layer, layer)
                logger.warning(
                    "possible_mis_routed_write layer=%s confidence=%s episode_type=%s target=%r preview=%r",
                    layer,
                    round(layer_conf, 2),
                    episode_type,
                    target,
                    content[:80],
                )
        except Exception as e:  # noqa: BLE001 — advisory only
            logger.debug(f"detect_layer failed (non-blocking): {e}")

        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        try:
            with self._driver.session() as db:
                # Get group_id from session if not provided
                if group_id is None:
                    result = db.run(
                        "MATCH (s:Session {uuid: $sid}) RETURN s.group_id AS gid",
                        sid=session_id,
                    )
                    record = result.single()
                    group_id = record["gid"] if record else "unknown"

                # Resolve agent_id: explicit > ctx-derived > "unknown".
                # Per-surface defaults: mcp→claude-code (the dominant MCP
                # client today), rest→openclaw, cli→cli. Callers that want
                # something specific should pass it explicitly.
                resolved_agent_id = agent_id
                if resolved_agent_id is None and ctx is not None:
                    src = getattr(ctx, "source", None)
                    resolved_agent_id = {
                        "mcp": "claude-code",
                        "rest": "openclaw",
                        "cli": "cli",
                    }.get(src)
                if resolved_agent_id is None:
                    resolved_agent_id = "unknown"

                # Create the episode node
                db.run(
                    """
                    CREATE (e:Episode {
                        uuid: $uuid,
                        session_id: $session_id,
                        group_id: $group_id,
                        agent_id: $agent_id,
                        content: $content,
                        episode_type: $episode_type,
                        importance: $importance,
                        created_at: datetime($created_at),
                        t_created: datetime($created_at),
                        access_count: 0
                    })
                    """,
                    uuid=episode_id,
                    session_id=session_id,
                    group_id=group_id,
                    agent_id=resolved_agent_id,
                    content=content,
                    episode_type=episode_type,
                    importance=importance,
                    created_at=now.isoformat(),
                )

                # Link to session
                db.run(
                    """
                    MATCH (s:Session {uuid: $session_id})
                    MATCH (e:Episode {uuid: $episode_id})
                    CREATE (s)-[:HAS_EPISODE {order: s.episode_count}]->(e)
                    SET s.episode_count = s.episode_count + 1
                    """,
                    session_id=session_id,
                    episode_id=episode_id,
                )

                # Run 2: Page + typed-edge maintenance.
                # Same session scope as the episode write — if any step
                # raises, the whole session's work (including the episode
                # create above) is discarded. In practice each `db.run`
                # is already auto-committed individually because we're
                # not inside an explicit `begin_transaction` block;
                # still, any exception aborts the flow and we log it.
                try:
                    self._maintain_pages_for_episode(
                        session=db,
                        episode_id=episode_id,
                        content=content,
                        episode_type=episode_type,
                        group_id=group_id,
                        created_at=now.isoformat(),
                    )
                except Exception as page_err:  # noqa: BLE001 — page maint failures must not lose episodes
                    logger.warning(
                        f"Page maintenance failed for {episode_id} (episode persisted): {page_err}"
                    )

            logger.info(f"Episode recorded: [{episode_type}] {content[:60]}...")
            return episode_id

        except Exception as e:
            logger.error(f"Failed to record episode: {e}")
            return None

    def _maintain_pages_for_episode(
        self,
        session,
        episode_id: str,
        content: str,
        episode_type: Optional[str],
        group_id: Optional[str],
        created_at: str,
        page_label: str = PAGE_LABEL,
    ) -> None:
        """Run 2: Page + typed-edge maintenance inside an episode write.

        Called by ``record_episode`` after the Episode node is created.
        Mirrors the spec §2 contract:

        1. ``extract_entity_references(content)`` → EntityRef list.
        2. For each ref, upsert a ``:Page {slug, domain}`` (ambient create).
        3. ``extract_typed_edges(content, episode_type)`` → TypedEdge list.
        4. MERGE each edge (endpoints get ambient Pages if still missing).
        5. ``append_timeline_entry`` on each referenced page, pointing at
           this episode.

        Runs on the *same* Neo4j session as the episode write so the work
        is coherent with the surrounding context. Exceptions propagate
        upward; caller (``record_episode``) logs + continues (episodes
        are authoritative; Page maintenance is augmentation).
        """
        # 1. Entity refs — what does this episode talk about?
        refs = extract_entity_references(content, episode_type)

        # 2. Ambient Page creation for every referenced entity.
        for ref in refs:
            put_page(ref.slug, ref.domain, tx=session, label=page_label)

        # 3. Typed-edge extraction (pure function).
        edges = extract_typed_edges(
            content,
            episode_type=episode_type,
            group_id=group_id,
        )

        # 4. Materialize edges (creates endpoint Pages if missing).
        if edges:
            create_edges_in_tx(session, edges, from_label=page_label, to_label=page_label)

        # 5. Timeline: EVIDENCED_BY each referenced page → this episode.
        #    The union of ref slugs + edge-endpoint slugs is the set of
        #    pages whose timeline should grow by this episode.
        timeline_slugs: set[str] = {ref.slug for ref in refs}
        for e in edges:
            timeline_slugs.add(e.from_slug)
            timeline_slugs.add(e.to_slug)
        summary = content[:120] if content else ""
        for slug in timeline_slugs:
            append_timeline_entry(
                slug,
                episode_id,
                at=created_at,
                summary=summary,
                tx=session,
                label=page_label,
            )

    def get_session_episodes(
        self,
        session_id: str,
        limit: int = MAX_EPISODES_PER_SESSION,
    ) -> list[dict[str, Any]]:
        """Get all episodes for a session, ordered chronologically.

        Args:
            session_id: Session UUID.
            limit: Maximum episodes to return.

        Returns:
            List of episode dicts.
        """
        try:
            with self._driver.session() as db:
                result = db.run(
                    """
                    MATCH (s:Session {uuid: $session_id})-[r:HAS_EPISODE]->(e:Episode)
                    RETURN e
                    ORDER BY r.order ASC
                    LIMIT $limit
                    """,
                    session_id=session_id,
                    limit=limit,
                )
                episodes = []
                for record in result:
                    node = record["e"]
                    episodes.append(
                        {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(node).items()}
                    )
                return episodes

        except Exception as e:
            logger.error(f"Failed to get episodes for session {session_id}: {e}")
            return []

    @staticmethod
    def should_record(content: str) -> bool:
        """Heuristic: is this exchange worth saving as an episode?

        Checks content length and presence of significance keywords.
        """
        if not content or len(content.strip()) < EPISODE_MIN_LENGTH:
            return False

        content_lower = content.lower()
        return any(kw in content_lower for kw in _SIGNIFICANT_KEYWORDS)


class SnapshotManager:
    """Manages session state snapshots.

    Snapshots capture structured state for fast context reconstruction
    when picking up a session on a different device.
    """

    def __init__(self, driver=None):
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._owns_driver = True

    def close(self):
        if self._owns_driver and self._driver:
            self._driver.close()

    def save_snapshot(
        self,
        session_id: str,
        snapshot_data: dict[str, Any],
    ) -> Optional[str]:
        """Save a session state snapshot.

        Args:
            session_id: Session UUID.
            snapshot_data: Structured snapshot with keys like:
                task, status, completed, in_progress, next_steps,
                key_decisions, blockers, files_modified.

        Returns:
            Snapshot UUID if saved.
        """
        snapshot_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Enforce size limit on snapshot content
        snapshot_json = json.dumps(snapshot_data, default=str)
        if len(snapshot_json) > SNAPSHOT_MAX_SIZE:
            logger.warning(f"Snapshot too large ({len(snapshot_json)} chars), truncating")
            # Truncate lists to fit
            for key in ["completed", "in_progress", "next_steps", "key_decisions"]:
                if key in snapshot_data and isinstance(snapshot_data[key], list):
                    while len(json.dumps(snapshot_data, default=str)) > SNAPSHOT_MAX_SIZE and len(snapshot_data[key]) > 1:
                        snapshot_data[key].pop()
            snapshot_json = json.dumps(snapshot_data, default=str)

        try:
            with self._driver.session() as db:
                # Get group_id from session
                result = db.run(
                    "MATCH (s:Session {uuid: $sid}) RETURN s.group_id AS gid",
                    sid=session_id,
                )
                record = result.single()
                group_id = record["gid"] if record else "unknown"

                # Create snapshot node
                db.run(
                    """
                    CREATE (snap:Snapshot {
                        uuid: $uuid,
                        session_id: $session_id,
                        group_id: $group_id,
                        data: $data,
                        created_at: datetime($created_at)
                    })
                    """,
                    uuid=snapshot_id,
                    session_id=session_id,
                    group_id=group_id,
                    data=snapshot_json,
                    created_at=now.isoformat(),
                )

                # Link to session
                db.run(
                    """
                    MATCH (s:Session {uuid: $session_id})
                    MATCH (snap:Snapshot {uuid: $snapshot_id})
                    CREATE (s)-[:HAS_SNAPSHOT {at: datetime($at)}]->(snap)
                    """,
                    session_id=session_id,
                    snapshot_id=snapshot_id,
                    at=now.isoformat(),
                )

            logger.info(f"Snapshot saved for session {session_id}")
            return snapshot_id

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def get_latest_snapshot(
        self,
        group_id: str,
    ) -> Optional[dict[str, Any]]:
        """Get the most recent snapshot for a project.

        Args:
            group_id: Project identifier.

        Returns:
            Parsed snapshot data dict, or None.
        """
        try:
            with self._driver.session() as db:
                result = db.run(
                    """
                    MATCH (s:Session {group_id: $group_id})-[:HAS_SNAPSHOT]->(snap:Snapshot)
                    WHERE s.status IN ['completed', 'interrupted']
                    RETURN snap, s.uuid AS session_id, s.device AS device,
                           s.started_at AS session_started
                    ORDER BY snap.created_at DESC
                    LIMIT 1
                    """,
                    group_id=group_id,
                )
                record = result.single()
                if record is None:
                    return None

                snap_node = record["snap"]
                snapshot_data = json.loads(snap_node["data"])
                snapshot_data["_session_id"] = record["session_id"]
                snapshot_data["_device"] = str(record["device"])
                snapshot_data["_session_started"] = str(record["session_started"])
                return snapshot_data

        except Exception as e:
            logger.error(f"Failed to get latest snapshot for {group_id}: {e}")
            return None

    @staticmethod
    def format_snapshot_for_injection(snapshot: dict[str, Any]) -> str:
        """Format a snapshot as a context block for session injection.

        Args:
            snapshot: Parsed snapshot data dict.

        Returns:
            Formatted string ready for context injection.
        """
        if not snapshot:
            return ""

        lines = [
            "## Session State (from previous session)",
            f"**Device:** {snapshot.get('_device', 'unknown')}",
            f"**Task:** {snapshot.get('task', 'unknown')}",
            f"**Status:** {snapshot.get('status', 'unknown')}",
        ]

        completed = snapshot.get("completed", [])
        if completed:
            lines.append("**Completed:**")
            for item in completed:
                lines.append(f"  - {item}")

        in_progress = snapshot.get("in_progress", [])
        if in_progress:
            lines.append("**In Progress:**")
            for item in in_progress:
                lines.append(f"  - {item}")

        next_steps = snapshot.get("next_steps", [])
        if next_steps:
            lines.append("**Next Steps:**")
            for item in next_steps:
                lines.append(f"  - {item}")

        decisions = snapshot.get("key_decisions", [])
        if decisions:
            lines.append("**Key Decisions:**")
            for item in decisions:
                lines.append(f"  - {item}")

        blockers = snapshot.get("blockers", [])
        if blockers:
            lines.append("**Blockers:**")
            for item in blockers:
                lines.append(f"  - {item}")

        files = snapshot.get("files_modified", [])
        if files:
            lines.append(f"**Files Modified:** {', '.join(files)}")

        lines.append("")
        return "\n".join(lines)


# === TRUST BOUNDARY — RUN 4 ===
#
# Write-side audit for remote-originated episode writes. Logged-only.
# ``EpisodeRecorder._audit_remote_write`` is mounted onto the class at the
# end of this block so the main class body above stays visually clean
# and Run 2's edits land further up without colliding with this block.
#
# Heuristic triggers (any one is enough to emit a warning):
#   1. group_id uses a non-canonical shape (uppercase, spaces, > 64 chars,
#      starts with punctuation).
#   2. content looks like a shell payload (leading ``#!`` shebang, contains
#      ``rm -rf``, ``:(){`` fork-bomb signature, ``wget|curl http``, etc.).
#   3. content is suspiciously large (>10 kB) — possible exfiltration.
#
# The warning is structured so ops can grep it later:
#   logger.warning("possible_abusive_remote_write %s", {...fields...})

import re as _re_run4  # noqa: E402 — trust-boundary block imports at EOF


_GROUP_ID_CANONICAL = _re_run4.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")

_SHELL_PAYLOAD_PATTERNS = [
    _re_run4.compile(r"^#!"),
    _re_run4.compile(r"\brm\s+-rf\s"),
    _re_run4.compile(r":\(\)\s*\{.*:\|:&"),                 # fork bomb
    _re_run4.compile(r"\b(wget|curl)\s+https?://[^\s|]+"),
    _re_run4.compile(r"\$\(.*\)\s*\|?\s*sh\b"),             # $(...) | sh
    _re_run4.compile(r"\beval\s*\("),
    _re_run4.compile(r"\bsudo\s"),
]

_LARGE_CONTENT_BYTES = 10 * 1024


def _looks_like_abusive_content(content: str) -> Optional[str]:
    """Return the reason string if content matches an abuse pattern, else None."""
    if not isinstance(content, str):
        return None
    if len(content.encode("utf-8")) > _LARGE_CONTENT_BYTES:
        return "content_over_10kb"
    for pat in _SHELL_PAYLOAD_PATTERNS:
        if pat.search(content):
            return f"shell_payload_pattern:{pat.pattern}"
    return None


def _looks_like_nonstandard_group_id(group_id: Optional[str]) -> Optional[str]:
    """Return a reason string if group_id looks off, else None."""
    if group_id is None:
        return None
    if not isinstance(group_id, str) or not group_id.strip():
        return "empty_or_non_string"
    if not _GROUP_ID_CANONICAL.fullmatch(group_id):
        return "non_canonical_shape"
    return None


def _resolve_ambient_ctx(explicit_ctx):
    """If the caller didn't pass ``ctx``, try to inherit from the ambient surface.

    MCP server and REST API install contextvars at the start of each request
    (Run 4 trust-boundary blocks). If one is set, we use it. Otherwise ``None``
    (treat as trusted-local).
    """
    if explicit_ctx is not None:
        return explicit_ctx
    # MCP first (more likely to be abusive), then REST.
    try:
        from mcp_server.server import current_mcp_context  # noqa: WPS433 — intentional circ-ok import

        mcp_ctx = current_mcp_context()
        if mcp_ctx is not None:
            return mcp_ctx
    except Exception:  # noqa: BLE001 — defensive
        pass
    try:
        from jarvis_memory.api import current_rest_context  # noqa: WPS433

        rest_ctx = current_rest_context()
        if rest_ctx is not None:
            return rest_ctx
    except Exception:  # noqa: BLE001
        pass
    return None


def _audit_remote_write(
    self: "EpisodeRecorder",
    ctx,
    content: str,
    group_id: Optional[str],
    episode_type: Optional[str],
) -> None:
    """Emit a structured WARNING if ``ctx.remote`` and content/group_id trip a heuristic.

    Never raises — callers wrap in try/except anyway. Never refuses the write.
    """
    ctx = _resolve_ambient_ctx(ctx)
    if ctx is None:
        return
    # Only care about remote ctx.
    if not getattr(ctx, "remote", False):
        return

    reasons: list[str] = []
    content_reason = _looks_like_abusive_content(content)
    if content_reason:
        reasons.append(content_reason)
    group_reason = _looks_like_nonstandard_group_id(group_id)
    if group_reason:
        reasons.append(f"group_id:{group_reason}")

    if not reasons:
        return

    logger.warning(
        "possible_abusive_remote_write %s",
        {
            "event": "possible_abusive_remote_write",
            "caller": getattr(ctx, "caller", None),
            "source": getattr(ctx, "source", None),
            "reasons": reasons,
            "episode_type": episode_type,
            "group_id": group_id,
            "content_preview": (content or "")[:80],
            "content_bytes": len(content.encode("utf-8")) if isinstance(content, str) else 0,
        },
    )


# Mount as a bound method on ``EpisodeRecorder`` so record_episode can call
# ``self._audit_remote_write(...)`` above. Keeping the logic at EOF makes
# the trust-boundary block replaceable / removable as a single unit.
EpisodeRecorder._audit_remote_write = _audit_remote_write  # type: ignore[attr-defined]


__all_run4__ = [
    "_audit_remote_write",
    "_looks_like_abusive_content",
    "_looks_like_nonstandard_group_id",
]

# === END TRUST BOUNDARY — RUN 4 ===
