"""Centralized handoff contract implementation.

All handoff + session-state operations go through this module. Before it
existed, the REST API, MCP server, and PreCompact hook each had their own
slightly-different implementation — the REST path was broken (called a
non-existent ``SessionManager.save_state`` method), the MCP path always
wrote to ``DEFAULT_GROUP_ID`` regardless of request args, and only the
direct-Cypher hook actually produced retrievable ``[HANDOFF]`` Episodes.
Centralizing fixes that drift.

Public surface:
    - ``save_handoff(...)``           write a handoff (snapshot + [HANDOFF] Episode)
    - ``get_latest_handoff(...)``     read the most recent handoff for a group
    - ``save_state_snapshot(...)``    write a session state snapshot (non-terminal)
    - ``list_groups(...)``            return group_id + episode count for every known group
    - ``HandoffResult``               typed return shape

Contract rules enforced here (see HANDOFF_CONTRACT.md for the full list):
    - ``group_id`` is required on every write (empty-string treated as missing).
    - ``session_key`` is optional but recommended; stored on the node for
      cross-surface correlation.
    - ``idempotency_key`` + ``session_id`` + name='session_handoff' are
      checked against the last hour of writes; a duplicate returns the
      existing handoff IDs instead of creating new rows.
    - Every handoff writes BOTH a snapshot AND an ``Episode`` with
      ``memory_type='handoff'`` so it's retrievable via ``get_latest_handoff``.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .config import SNAPSHOT_MAX_SIZE


logger = logging.getLogger("jarvis_memory.handoff")


@dataclass
class HandoffResult:
    """Return shape for save_handoff."""
    snapshot_id: Optional[str]
    episode_id: Optional[str]
    session_id: str
    group_id: str
    idempotent_hit: bool  # True when an existing handoff matched the idempotency_key


class GroupIDRequired(ValueError):
    """Raised when group_id is missing or empty on a write path."""


def _validate_group_id(group_id: Optional[str]) -> str:
    if group_id is None or not group_id.strip():
        raise GroupIDRequired(
            "group_id is required on handoff/state writes. "
            "Pass a project slug (e.g. 'atlas-system', 'navi', 'foundry') "
            "or 'system' for true admin/system-level memory."
        )
    return group_id.strip()


def save_handoff(
    driver,
    *,
    group_id: str,
    task: str,
    next_steps: list[str] | None = None,
    notes: str = "",
    device: str = "unknown",
    session_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    session_key: Optional[str] = None,
    source: str = "unknown",
    create_session_if_missing: bool = True,
) -> HandoffResult:
    """Write a handoff: snapshot + [HANDOFF] Episode.

    Args:
        driver: Neo4j driver.
        group_id: project scope. Required. Raises ``GroupIDRequired`` if empty.
        task: short description of what was being worked on.
        next_steps: list of pickup instructions for the next session.
        notes: free-form handoff notes.
        device: device identifier (e.g. "mac-mini", "vps", "client-alpha").
        session_id: session to attach the handoff to. If omitted:
            - uses the latest session for this group_id, OR
            - creates a fresh session if ``create_session_if_missing=True`` (default),
              OR raises ValueError.
        idempotency_key: optional caller-supplied key. If provided and a
            handoff with the same key exists within the last hour for this
            session_id, this call is a no-op and returns the existing IDs.
        session_key: optional cross-surface correlation id, stored on the
            Episode for later "did this handoff happen?" queries.
        source: free-form source tag ("rest", "mcp", "hook", "cli").
        create_session_if_missing: if the group_id has no sessions, create one.

    Returns:
        ``HandoffResult`` with snapshot_id, episode_id, session_id, group_id,
        idempotent_hit flag.
    """
    from .conversation import SessionManager

    group_id = _validate_group_id(group_id)
    next_steps = next_steps or []

    sm = SessionManager(driver=driver)
    try:
        # Resolve session_id.
        if session_id is None:
            latest = sm.get_latest_session(group_id)
            if latest:
                session_id = latest["uuid"]
            elif create_session_if_missing:
                fresh = sm.create_session(group_id=group_id, device=device)
                session_id = fresh["uuid"]
            else:
                raise ValueError(
                    f"No existing session for group_id={group_id!r} and "
                    f"create_session_if_missing=False."
                )

        # Idempotency check. Scoped by group_id + key (NOT session_id): each
        # successful handoff ends its session with status='handoff', which
        # `get_latest_session` excludes — so a retry after the first write
        # resolves to a NEW session_id, and session-scoped idempotency would
        # miss the prior hit. Keying on group_id keeps dedupe stable across
        # session resolution regardless of what session_id the caller resolves.
        if idempotency_key:
            existing = _find_existing_handoff(
                driver, group_id=group_id, idempotency_key=idempotency_key,
            )
            if existing:
                logger.info(
                    "handoff idempotent hit group=%s key=%s", group_id, idempotency_key,
                )
                return HandoffResult(
                    snapshot_id=existing.get("snapshot_id"),
                    episode_id=existing.get("episode_id"),
                    session_id=existing.get("session_id") or session_id,
                    group_id=group_id,
                    idempotent_hit=True,
                )

        # Build payloads before opening the transaction so a JSON serialization
        # error can't leave the DB in a half-written state.
        snapshot_id = str(uuid.uuid4())
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        snapshot_data = {
            "type": "handoff_snapshot",
            "task": task,
            "status": "handoff",
            "next_steps": next_steps,
            "notes": notes,
            "device": device,
            "session_key": session_key,
            "source": source,
            "idempotency_key": idempotency_key,
        }
        snapshot_json = json.dumps(snapshot_data, default=str)
        if len(snapshot_json) > SNAPSHOT_MAX_SIZE:
            # Trim lists until the JSON fits. Same approach SnapshotManager uses.
            for key in ("next_steps",):
                while (
                    len(json.dumps(snapshot_data, default=str)) > SNAPSHOT_MAX_SIZE
                    and isinstance(snapshot_data.get(key), list)
                    and len(snapshot_data[key]) > 1
                ):
                    snapshot_data[key].pop()
            snapshot_json = json.dumps(snapshot_data, default=str)

        content_lines = [f"[HANDOFF] {task}"]
        if next_steps:
            content_lines.append("Next steps:")
            content_lines.extend(f"  - {s}" for s in next_steps)
        if notes:
            content_lines.append(f"Notes: {notes}")
        content = "\n".join(content_lines)

        # All four writes in one managed transaction — either all land or
        # Neo4j rolls back. Before this was split across three separate
        # driver.session() blocks and could half-write on any failure.
        def _write_atomic(tx):
            tx.run(
                """
                CREATE (snap:Snapshot {
                    uuid: $snap_id,
                    session_id: $session_id,
                    group_id: $group_id,
                    data: $data,
                    idempotency_key: $idempotency_key,
                    created_at: datetime($now)
                })
                """,
                snap_id=snapshot_id,
                session_id=session_id,
                group_id=group_id,
                data=snapshot_json,
                idempotency_key=idempotency_key,
                now=now,
            )
            tx.run(
                """
                MATCH (s:Session {uuid: $session_id})
                MATCH (snap:Snapshot {uuid: $snap_id})
                CREATE (s)-[:HAS_SNAPSHOT {at: datetime($now)}]->(snap)
                """,
                session_id=session_id,
                snap_id=snapshot_id,
                now=now,
            )
            tx.run(
                """
                CREATE (e:Episode {
                    uuid: $ep_id,
                    group_id: $group_id,
                    content: $content,
                    name: $name,
                    memory_type: 'handoff',
                    episode_type: 'outcome',
                    importance: 0.9,
                    access_count: 0,
                    created_at: datetime($now),
                    session_id: $session_id,
                    device: $device,
                    source: $source,
                    idempotency_key: $idempotency_key,
                    session_key: $session_key
                })
                """,
                ep_id=episode_id,
                group_id=group_id,
                content=content,
                name=f"[HANDOFF] {task[:80]}",
                now=now,
                session_id=session_id,
                device=device,
                source=source,
                idempotency_key=idempotency_key,
                session_key=session_key,
            )
            tx.run(
                """
                MATCH (s:Session {uuid: $session_id})
                MATCH (e:Episode {uuid: $ep_id})
                CREATE (s)-[:PRODUCED_HANDOFF {at: datetime($now)}]->(e)
                """,
                session_id=session_id,
                ep_id=episode_id,
                now=now,
            )

        with driver.session() as db:
            db.execute_write(_write_atomic)

        # Mark session as ended/handoff (non-fatal; the handoff itself already
        # landed — if end_session fails the session just stays 'open', which
        # a cleanup job or the next handoff will reconcile).
        try:
            sm.end_session(session_id, status="handoff")
        except Exception as e:
            logger.debug("end_session non-fatal error: %s", e)

        logger.info(
            "handoff written group=%s session=%s episode=%s snapshot=%s",
            group_id, session_id, episode_id, snapshot_id,
        )
        return HandoffResult(
            snapshot_id=snapshot_id,
            episode_id=episode_id,
            session_id=session_id,
            group_id=group_id,
            idempotent_hit=False,
        )
    finally:
        sm.close()


def save_state_snapshot(
    driver,
    *,
    group_id: str,
    task: str,
    status: str = "in_progress",
    completed: list[str] | None = None,
    in_progress: list[str] | None = None,
    next_steps: list[str] | None = None,
    blockers: list[str] | None = None,
    key_decisions: list[str] | None = None,
    files_modified: list[str] | None = None,
    device: str = "unknown",
    session_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    session_key: Optional[str] = None,
    source: str = "unknown",
    create_session_if_missing: bool = True,
) -> dict[str, Any]:
    """Write a session-state snapshot (non-terminal; doesn't end the session).

    Same idempotency + session resolution semantics as ``save_handoff``.
    Does NOT write an Episode — this is pure state, not a retrievable
    decision/outcome. Use ``save_handoff`` when you want a retrievable
    ``[HANDOFF]`` Episode.
    """
    from .conversation import SessionManager, SnapshotManager

    group_id = _validate_group_id(group_id)

    sm = SessionManager(driver=driver)
    try:
        if session_id is None:
            latest = sm.get_latest_session(group_id)
            if latest:
                session_id = latest["uuid"]
            elif create_session_if_missing:
                fresh = sm.create_session(group_id=group_id, device=device)
                session_id = fresh["uuid"]
            else:
                raise ValueError(
                    f"No existing session for group_id={group_id!r} and "
                    f"create_session_if_missing=False."
                )

        if idempotency_key:
            existing = _find_existing_snapshot(
                driver, session_id=session_id, idempotency_key=idempotency_key,
            )
            if existing:
                logger.info(
                    "save_state idempotent hit session=%s key=%s",
                    session_id, idempotency_key,
                )
                return {
                    "snapshot_id": existing["snapshot_id"],
                    "session_id": session_id,
                    "group_id": group_id,
                    "idempotent_hit": True,
                }

        snapshot_data = {
            "type": "session_snapshot",
            "task": task,
            "status": status,
            "completed": completed or [],
            "in_progress": in_progress or [],
            "next_steps": next_steps or [],
            "blockers": blockers or [],
            "key_decisions": key_decisions or [],
            "files_modified": files_modified or [],
            "device": device,
            "session_key": session_key,
            "source": source,
            "idempotency_key": idempotency_key,
        }
        snm = SnapshotManager(driver=driver)
        snapshot_id = snm.save_snapshot(session_id, snapshot_data)

        return {
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "group_id": group_id,
            "idempotent_hit": False,
        }
    finally:
        sm.close()


def get_latest_handoff(
    driver,
    *,
    group_id: str,
    max_age_hours: int = 72,
) -> Optional[dict[str, Any]]:
    """Return the most recent [HANDOFF] Episode for group_id within max_age_hours.

    Returns None if no handoff found (or all handoffs are older than the cutoff).
    """
    group_id = _validate_group_id(group_id)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

    with driver.session() as db:
        row = db.run(
            """
            MATCH (e:Episode {group_id: $gid})
            WHERE e.memory_type = 'handoff'
              AND e.created_at >= datetime($cutoff)
            RETURN e.uuid AS uuid,
                   e.content AS content,
                   e.created_at AS created_at,
                   e.session_id AS session_id,
                   e.device AS device,
                   e.source AS source,
                   e.session_key AS session_key
            ORDER BY e.created_at DESC LIMIT 1
            """,
            gid=group_id, cutoff=cutoff,
        ).single()
        if row is None:
            return None
        return {
            # episode_id is canonical; `uuid` is the legacy alias kept so
            # older clients (and REST callers that shipped before v1.1) don't
            # break. jarvis-start-guard + CLI both read episode_id.
            "episode_id": row["uuid"],
            "uuid": row["uuid"],
            "content": row["content"],
            "created_at": str(row["created_at"]),
            "session_id": row["session_id"],
            "device": row["device"],
            "source": row["source"],
            "session_key": row["session_key"],
            "group_id": group_id,
        }


def list_groups(driver) -> list[dict[str, Any]]:
    """Return [{group_id, episode_count, session_count, latest_episode_at}, ...]
    for every group_id with at least one Episode or Session.

    Useful for debugging "where did my memory go?" — one call shows the
    distribution across groups.
    """
    with driver.session() as db:
        rows = db.run(
            """
            CALL {
                MATCH (e:Episode)
                WHERE e.group_id IS NOT NULL
                RETURN e.group_id AS gid,
                       count(e) AS episode_count,
                       max(e.created_at) AS latest_episode_at
                UNION ALL
                MATCH (s:Session)
                WHERE s.group_id IS NOT NULL
                RETURN s.group_id AS gid,
                       0 AS episode_count,
                       null AS latest_episode_at
            }
            WITH gid,
                 sum(episode_count) AS episode_count,
                 max(latest_episode_at) AS latest_episode_at
            RETURN gid,
                   episode_count,
                   latest_episode_at
            ORDER BY episode_count DESC, gid ASC
            """
        ).data()

    # Add session counts in a second pass so the UNION stays cheap.
    with driver.session() as db:
        session_rows = db.run(
            """
            MATCH (s:Session)
            WHERE s.group_id IS NOT NULL
            RETURN s.group_id AS gid, count(s) AS session_count
            """
        ).data()
    sessions_by_gid = {r["gid"]: r["session_count"] for r in session_rows}

    return [
        {
            "group_id": r["gid"],
            "episode_count": r["episode_count"] or 0,
            "session_count": sessions_by_gid.get(r["gid"], 0),
            "latest_episode_at": str(r["latest_episode_at"]) if r["latest_episode_at"] else None,
        }
        for r in rows
    ]


# ── idempotency helpers ──────────────────────────────────────────────

def _find_existing_handoff(
    driver, *, group_id: str, idempotency_key: str, max_age_hours: int = 1,
) -> Optional[dict[str, Any]]:
    """Return {episode_id, snapshot_id, session_id} if a handoff with this
    idempotency_key exists for this group_id within max_age_hours. Else None.

    Scoped by group_id (not session_id) because each successful handoff ends
    its session with status='handoff', so a retry resolves to a new session
    before the idempotency check runs — session-scoped dedupe misses the hit.
    group_id is caller-stable: the retry sees the same key.

    Matches the Episode AND the Snapshot both by ``idempotency_key`` — the
    two are written atomically with the same key by ``save_handoff``, so
    if we see the Episode we always see the paired Snapshot.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    with driver.session() as db:
        row = db.run(
            """
            MATCH (e:Episode {group_id: $gid, idempotency_key: $ik})
            WHERE e.memory_type = 'handoff'
              AND e.created_at >= datetime($cutoff)
            OPTIONAL MATCH (snap:Snapshot {group_id: $gid, idempotency_key: $ik})
            WHERE snap.created_at >= datetime($cutoff)
            RETURN e.uuid AS episode_id,
                   e.session_id AS session_id,
                   snap.uuid AS snapshot_id
            ORDER BY e.created_at DESC LIMIT 1
            """,
            gid=group_id, ik=idempotency_key, cutoff=cutoff,
        ).single()
        if row is None:
            return None
        return {
            "episode_id": row["episode_id"],
            "snapshot_id": row["snapshot_id"],
            "session_id": row["session_id"],
        }


def _find_existing_snapshot(
    driver, *, session_id: str, idempotency_key: str, max_age_hours: int = 1,
) -> Optional[dict[str, Any]]:
    """Return {snapshot_id} if a snapshot with this idempotency_key exists
    for this session within max_age_hours. Else None.

    Queries by the ``idempotency_key`` top-level property (set atomically
    by save_handoff). Falls back to a JSON substring probe for snapshots
    written by ``save_state_snapshot`` via the older SnapshotManager path
    which doesn't expose the column directly — the parameterized probe
    is safely passed as a Cypher parameter so special characters in keys
    can't break the query shape.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
    with driver.session() as db:
        row = db.run(
            """
            MATCH (snap:Snapshot)
            WHERE snap.session_id = $sid
              AND snap.created_at >= datetime($cutoff)
              AND (snap.idempotency_key = $ik OR snap.data CONTAINS $probe)
            RETURN snap.uuid AS snapshot_id
            ORDER BY snap.created_at DESC LIMIT 1
            """,
            sid=session_id,
            cutoff=cutoff,
            ik=idempotency_key,
            probe=json.dumps({"idempotency_key": idempotency_key})[1:-1],
        ).single()
        if row is None:
            return None
        return {"snapshot_id": row["snapshot_id"]}
