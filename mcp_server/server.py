"""Jarvis Memory MCP Server — semantic search, typed graph, temporal facts.

27 tools total (see tests/test_mcp_parity.py for the locked name set).
Tool surface is frozen by parity tests; additions or renames require a PR
that bumps the count there.

Tool groups (at a glance):
  - Search / recall: scored_search, wake_up, search_rooms, fact_timeline
  - Writes:          save_episode, save_state, session_handoff
  - Sessions:        list_sessions, get_session, continue_session
  - Lifecycle:       lifecycle_status, lifecycle_transition,
                     classify_memory, supersede_memory, contradict_memory,
                     set_fact_validity, restore_memory, bulk_archive_stale
  - Compaction:      compact_session, compact_daily, compact_weekly,
                     compaction_status
  - Stats / utility: memory_stats

Run:  python -m mcp_server.server   (preferred)
      jarvis-mcp                     (pip console script — Run 3 fix)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Jarvis modules
from jarvis_memory.scoring import score_results, composite_score
from jarvis_memory.classifier import classify_memory as do_classify, MEMORY_TYPES
from jarvis_memory.lifecycle import MemoryLifecycle, LIFECYCLE_STATES, VALID_TRANSITIONS
from jarvis_memory.compaction import CompactionEngine
from jarvis_memory.conversation import SessionManager, EpisodeRecorder, SnapshotManager
from jarvis_memory.embeddings import EmbeddingStore
from jarvis_memory.rooms import detect_room, get_hall
from jarvis_memory.temporal import set_validity, get_timeline, filter_by_date
from jarvis_memory.wake_up import wake_up as do_wake_up
from jarvis_memory.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    HYBRID_ALPHA, DEFAULT_GROUP_ID, DEVICE_ID,
)

logger = logging.getLogger(__name__)

# ── Tool Definitions ───────────────────────────────────────────────────

JARVIS_TOOLS = [
    # === RUN 2 — ENTITY LAYER ===
    # 4 tools surfacing the Page + typed-edge knowledge graph:
    # find_orphans, doctor, get_page, list_pages.
    # See brain/projects/jarvis-memory/plans/runs/2026-04-20-entity-layer/spec.md.
    Tool(
        name="find_orphans",
        description=(
            "Return Pages with zero inbound typed edges (any of ATTENDED, WORKS_AT, "
            "INVESTED_IN, FOUNDED, ADVISES, DECIDED_ON, MENTIONS, REFERS_TO). "
            "EVIDENCED_BY does NOT count — timeline-only Pages are still orphaned. "
            "Results grouped by domain. Useful for spotting entities that haven't "
            "yet been connected to the rest of the graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Filter to a single domain (e.g., 'person', 'company').",
                },
            },
        },
    ),
    Tool(
        name="doctor",
        description=(
            "Run health checks against the entity layer. Returns PASS/WARN/FAIL for "
            "schema_v2_present (expected constraints + indexes exist), page_completeness "
            "(fraction of Pages with non-empty compiled_truth), edge_validity "
            "(no dangling EVIDENCED_BY edges), and orphan_count_reasonable "
            "(orphan Pages < 10% of total). Each check carries a fix_hint on failure."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "fast": {
                    "type": "boolean",
                    "description": "Skip the orphan_count_reasonable check (cheaper run).",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="get_page",
        description=(
            "Fetch a single Page by slug. Returns the Page node's slug, domain, "
            "compiled_truth, created_at, updated_at — or null if not found."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "Canonical Page identifier (e.g., 'foundry').",
                },
            },
            "required": ["slug"],
        },
    ),
    Tool(
        name="list_pages",
        description=(
            "List all Pages, optionally filtered by domain, newest-first by updated_at. "
            "Returns up to `limit` Page dicts (default 100)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Filter by domain (e.g., 'person', 'company', 'project').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 100).",
                    "default": 100,
                },
            },
        },
    ),
    # === END RUN 2 — ENTITY LAYER ===
    Tool(
        name="scored_search",
        description=(
            "Search memories with composite scoring (semantic similarity × recency decay "
            "× importance × access frequency). Uses ChromaDB vector embeddings for real "
            "semantic search. Supports room/hall/temporal filtering for precision. "
            "Use this instead of raw search_nodes/search_facts for better relevance."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "group_id": {"type": "string", "description": "Project group (wing) to search within"},
                "room": {"type": "string", "description": "Topic filter (e.g., 'auth', 'frontend', 'infrastructure')"},
                "hall": {"type": "string", "description": "Memory category filter: decisions, plans, milestones, problems, context"},
                "as_of": {"type": "string", "description": "ISO date — only return facts valid at this date (temporal filter)"},
                "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                "memory_type": {"type": "string", "description": "Filter by specific memory type (optional)"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="classify_memory",
        description=(
            "Classify a piece of text into one of 21 memory types: "
            + ", ".join(sorted(MEMORY_TYPES.keys()))
            + ". Uses fast keyword heuristic first, LLM fallback for ambiguous cases. "
            "Set detailed=true for confidence score and sentiment analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory content to classify"},
                "use_llm": {"type": "boolean", "description": "Use LLM for ambiguous cases (default false)", "default": False},
                "detailed": {"type": "boolean", "description": "Return confidence + sentiment (default false)", "default": False},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="lifecycle_status",
        description="Get the current lifecycle status of a memory (active, confirmed, outdated, archived, contradicted, merged, superseded, deleted).",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the memory"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="lifecycle_transition",
        description=(
            "Transition a memory to a new lifecycle state. Validates that the transition is allowed. "
            "Valid transitions: active→{confirmed,outdated,archived,contradicted,merged,superseded,deleted}, "
            "confirmed→{outdated,archived,superseded,deleted}, outdated→{archived,deleted,active}, "
            "archived→{active,deleted}, contradicted→{deleted,active}, merged→{deleted}, superseded→{deleted}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the memory"},
                "from_status": {"type": "string", "description": "Expected current status"},
                "to_status": {"type": "string", "description": "Target status"},
            },
            "required": ["memory_id", "from_status", "to_status"],
        },
    ),
    Tool(
        name="bulk_archive_stale",
        description="Find and optionally archive memories older than a threshold. Use dry_run=true to preview.",
        inputSchema={
            "type": "object",
            "properties": {
                "threshold_days": {"type": "integer", "description": "Days threshold (default 30)", "default": 30},
                "group_id": {"type": "string", "description": "Filter by project group (optional)"},
                "dry_run": {"type": "boolean", "description": "Preview only, don't archive (default true)", "default": True},
            },
        },
    ),
    Tool(
        name="compact_session",
        description="Deduplicate memories within a single session. Removes exact duplicates, keeps most recent.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier to compact"},
                "group_id": {"type": "string", "description": "Project group (optional)"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="compact_daily",
        description="Run daily digest compaction — merge similar memories from the past 24 hours. Uses semantic dedup when ChromaDB is available.",
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group (optional)"},
                "lookback_hours": {"type": "integer", "description": "Hours to look back (default 24)", "default": 24},
            },
        },
    ),
    Tool(
        name="compact_weekly",
        description="Run weekly merge — consolidate memories from the past 7 days with high dedup threshold.",
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group (optional)"},
            },
        },
    ),
    Tool(
        name="compaction_status",
        description="Get compaction health metrics: how many memories have been compacted at each tier.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="memory_stats",
        description="Get full system statistics: lifecycle counts, compaction status, ChromaDB embedding count, total memories by group.",
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Filter by project group (optional)"},
            },
        },
    ),
    Tool(
        name="supersede_memory",
        description="Mark an old memory as superseded by a newer one. Auto-sets valid_to on the old memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "old_id": {"type": "string", "description": "UUID of the memory being replaced"},
                "new_id": {"type": "string", "description": "UUID of the replacement memory"},
            },
            "required": ["old_id", "new_id"],
        },
    ),
    Tool(
        name="contradict_memory",
        description="Mark a memory as contradicted by another. Auto-sets valid_to and flags for review.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the contradicted memory"},
                "contradicting_id": {"type": "string", "description": "UUID of the contradicting memory"},
            },
            "required": ["memory_id", "contradicting_id"],
        },
    ),
    Tool(
        name="restore_memory",
        description="Restore an archived, contradicted, or outdated memory back to active status.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the memory to restore"},
            },
            "required": ["memory_id"],
        },
    ),
    # ── Conversation Persistence Tools ────────────────────────────────
    Tool(
        name="save_episode",
        description=(
            "Save a conversation episode to the current session. Call this when a decision "
            "is made, a plan is set, a task is completed, an approach is chosen or rejected, "
            "or key context is established. Dual-writes to Neo4j and ChromaDB for semantic search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The episode content (decision, plan, context, etc.)"},
                "group_id": {"type": "string", "description": "Project group_id (auto-creates session if needed)"},
                "session_id": {"type": "string", "description": "Session UUID (auto-created if omitted)"},
                "episode_type": {"type": "string", "description": "Memory type (auto-classified if omitted)"},
                "importance": {"type": "number", "description": "Importance 0-1 (default 0.8)", "default": 0.8},
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="save_state",
        description=(
            "Save a full session state snapshot on demand. Use when the user says 'save state', "
            "before a known break, or when switching tasks. This is the primary mechanism for "
            "cross-device handoff — the next session on any device will load this snapshot."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What the session was working on"},
                "status": {"type": "string", "description": "Current status: in_progress, completed, blocked", "default": "in_progress"},
                "completed": {"type": "array", "items": {"type": "string"}, "description": "List of completed items"},
                "in_progress": {"type": "array", "items": {"type": "string"}, "description": "List of in-progress items"},
                "next_steps": {"type": "array", "items": {"type": "string"}, "description": "List of next steps"},
                "key_decisions": {"type": "array", "items": {"type": "string"}, "description": "Key decisions made and their rationale"},
                "blockers": {"type": "array", "items": {"type": "string"}, "description": "Current blockers"},
                "files_modified": {"type": "array", "items": {"type": "string"}, "description": "Files modified in this session"},
                "session_id": {"type": "string", "description": "Session UUID (auto-detected if omitted)"},
            },
            "required": ["task"],
        },
    ),
    Tool(
        name="get_session",
        description="Get a session by ID including its episodes and snapshot. Use to review what happened in a specific session.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session UUID"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="list_sessions",
        description="List recent sessions for a project, ordered by most recent. Shows session metadata, device, and status.",
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group ID"},
                "limit": {"type": "integer", "description": "Max sessions to return (default 10)", "default": 10},
            },
            "required": ["group_id"],
        },
    ),
    Tool(
        name="continue_session",
        description=(
            "Load full context from the most recent session for a project to continue where it left off. "
            "Returns the session snapshot, episode chain, and session metadata. This is what enables "
            "cross-device continuity — call this when picking up work on a different machine."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group ID to continue"},
                "session_id": {"type": "string", "description": "Specific session UUID to continue from (optional, uses latest if omitted)"},
            },
            "required": ["group_id"],
        },
    ),
    Tool(
        name="session_handoff",
        description=(
            "Prepare a handoff package for cross-device continuation. Saves the current snapshot, "
            "marks the session as ready for pickup, and returns a summary of what the next session "
            "will receive. Use before switching from MacBook Pro to Mac Mini or vice versa."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What was being worked on"},
                "notes": {"type": "string", "description": "Any additional handoff notes for the next session"},
                "next_steps": {"type": "array", "items": {"type": "string"}, "description": "What the next session should do"},
                "session_id": {"type": "string", "description": "Session UUID (auto-detected if omitted)"},
            },
            "required": ["task"],
        },
    ),
    # ── v2 Tools ──────────────────────────────────────────────────────
    Tool(
        name="wake_up",
        description=(
            "Token-budgeted context loading. Returns Layer 0 (identity, ~100 tokens) + "
            "Layer 1 (essential story, ~500 tokens) as a pre-formatted context block. "
            "Call this at session start instead of scored_search for efficient context priming."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group ID"},
            },
            "required": ["group_id"],
        },
    ),
    Tool(
        name="set_fact_validity",
        description=(
            "Set temporal validity bounds on a memory. Use when facts change over time: "
            "'Max works on Orion' → valid_from 2025-06, valid_to 2026-02. "
            "If valid_to is set, the fact is treated as historical (no longer current)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "UUID of the memory"},
                "valid_from": {"type": "string", "description": "ISO datetime when the fact became true (defaults to created_at)"},
                "valid_to": {"type": "string", "description": "ISO datetime when the fact stopped being true (null = still current)"},
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="fact_timeline",
        description=(
            "Get chronological fact history for an entity or topic. Shows all facts "
            "mentioning the entity with their validity windows. Useful for understanding "
            "how knowledge about something has evolved over time."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name or topic to search for"},
                "group_id": {"type": "string", "description": "Project group filter (optional)"},
                "limit": {"type": "integer", "description": "Max results (default 50)", "default": 50},
            },
            "required": ["entity"],
        },
    ),
    Tool(
        name="search_rooms",
        description=(
            "List all rooms (topics) with memory counts for a project. "
            "Useful for discovering what topics have been discussed and "
            "which rooms to filter on in scored_search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "group_id": {"type": "string", "description": "Project group ID"},
            },
            "required": ["group_id"],
        },
    ),
]


# ── Server Implementation ─────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the Jarvis Memory MCP server."""
    server = Server("jarvis-memory")

    # Lazy-init shared resources
    _lifecycle: MemoryLifecycle | None = None
    _compactor: CompactionEngine | None = None
    _embed_store: EmbeddingStore | None = None
    _neo4j_driver = None

    def get_lifecycle() -> MemoryLifecycle:
        nonlocal _lifecycle
        if _lifecycle is None:
            _lifecycle = MemoryLifecycle()
        return _lifecycle

    def get_compactor() -> CompactionEngine:
        nonlocal _compactor
        if _compactor is None:
            _compactor = CompactionEngine(embedding_store=get_embed_store())
        return _compactor

    def get_embed_store() -> EmbeddingStore:
        nonlocal _embed_store
        if _embed_store is None:
            _embed_store = EmbeddingStore()
        return _embed_store

    def get_driver():
        nonlocal _neo4j_driver
        if _neo4j_driver is None:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        return _neo4j_driver

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return JARVIS_TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            result = await _dispatch(
                name, arguments, get_lifecycle, get_compactor,
                get_embed_store, get_driver,
            )
            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _get_current_session_id(expected_group_id: str | None = None) -> str | None:
    """Read the current session ID from the temp file written by session_start hook.

    If expected_group_id is provided, the cached session is only returned if its
    group_id matches. This prevents a previous group's session from silently
    capturing writes intended for a different project.
    """
    import pathlib
    session_file = pathlib.Path("/tmp/jarvis_current_session.json")
    try:
        if session_file.exists():
            data = json.loads(session_file.read_text())
            cached_gid = data.get("group_id")
            if expected_group_id is not None and cached_gid and cached_gid != expected_group_id:
                # Cache belongs to a different group — don't reuse it.
                return None
            return data.get("session_id")
    except Exception:
        pass
    return None


def _get_or_create_session_id(group_id: str = DEFAULT_GROUP_ID) -> str:
    """Get the current session ID for this group_id, or auto-create a new one.

    Per-group_id scoping: if the cached session is for a different group_id,
    create a new session for the requested group_id rather than silently
    returning the cached (wrong-group) session.
    """
    import pathlib

    session_id = _get_current_session_id(expected_group_id=group_id)
    if session_id:
        return session_id

    sm = SessionManager()
    result = sm.create_session(
        group_id=group_id,
        device=DEVICE_ID,
        task_summary="Auto-created session (no hook)",
    )
    sm.close()

    new_id = result.get("uuid")
    if not new_id:
        raise RuntimeError(f"Failed to auto-create session: {result}")

    session_file = pathlib.Path("/tmp/jarvis_current_session.json")
    try:
        session_file.write_text(json.dumps({
            "session_id": new_id,
            "group_id": group_id,
            "device": DEVICE_ID,
        }))
    except Exception:
        pass

    logger.info(f"Auto-created session {new_id} for group {group_id}")
    return new_id


def _chromadb_write(store: EmbeddingStore, memory_id: str, content: str,
                    group_id: str, memory_type: str) -> None:
    """Dual-write helper: embed content in ChromaDB with metadata."""
    if not store or not store.health_check():
        return
    try:
        room = detect_room(content, group_id)
        hall = get_hall(memory_type)
        metadata = {
            "wing": group_id,
            "room": room,
            "hall": hall,
            "memory_type": memory_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        store.embed_and_store(memory_id, content, metadata)
    except Exception as e:
        logger.warning(f"ChromaDB dual-write failed for {memory_id}: {e}")


async def _dispatch(
    name: str,
    args: dict[str, Any],
    get_lifecycle,
    get_compactor,
    get_embed_store,
    get_driver,
) -> dict[str, Any]:
    """Route tool calls to the appropriate handler."""

    # === RUN 2 — ENTITY LAYER ===
    # Dispatch for 4 Run 2 tools. Kept at the top of the if-chain so the
    # fenced block stays together. Run 4 adds its trust-boundary handlers
    # at the bottom of this function (no textual overlap with this block).
    if name == "find_orphans":
        from jarvis_memory.orphans import find_orphans as _find_orphans

        driver = get_driver()
        grouped = _find_orphans(domain=args.get("domain"), driver=driver)
        return {
            "count": sum(len(v) for v in grouped.values()),
            "by_domain": {d: [p.to_dict() for p in pages] for d, pages in grouped.items()},
        }

    if name == "doctor":
        from jarvis_memory.doctor import run_health_checks

        driver = get_driver()
        report = run_health_checks(driver=driver, fast=bool(args.get("fast", False)))
        return report

    if name == "get_page":
        from jarvis_memory.pages import get_page as _get_page

        driver = get_driver()
        page = _get_page(args["slug"], driver=driver)
        return {"page": page.to_dict() if page else None}

    if name == "list_pages":
        from jarvis_memory.pages import list_pages as _list_pages

        driver = get_driver()
        pages = _list_pages(
            domain=args.get("domain"),
            driver=driver,
            limit=int(args.get("limit", 100)),
        )
        return {"count": len(pages), "pages": [p.to_dict() for p in pages]}
    # === END RUN 2 — ENTITY LAYER ===

    # ── scored_search (v2: ChromaDB semantic + metadata filtering) ────

    if name == "scored_search":
        query = args.get("query", "")
        group_id = args.get("group_id")
        room = args.get("room")
        hall = args.get("hall")
        as_of = args.get("as_of")
        limit = args.get("limit", 10)
        memory_type = args.get("memory_type")

        store = get_embed_store()
        driver = get_driver()

        # Try ChromaDB semantic search first
        if store.health_check():
            try:
                # Build metadata filter
                where_filter = {}
                if group_id:
                    where_filter["wing"] = group_id
                if room:
                    where_filter["room"] = room
                if hall:
                    where_filter["hall"] = hall
                if memory_type:
                    where_filter["memory_type"] = memory_type

                # Oversample from ChromaDB for scoring
                chromadb_results = store.search(
                    query=query,
                    limit=limit * 3,
                    where_filter=where_filter if where_filter else None,
                )

                if chromadb_results:
                    # Fetch full nodes from Neo4j
                    uuids = [r["id"] for r in chromadb_results]
                    similarity_map = {r["id"]: r["similarity"] for r in chromadb_results}

                    results = []
                    with driver.session() as db:
                        records = db.run(
                            """
                            UNWIND $uuids AS uid
                            MATCH (n) WHERE n.uuid = uid
                            RETURN n, labels(n) AS labels
                            """,
                            uuids=uuids,
                        )
                        for record in records:
                            node = dict(record["n"])
                            for k, v in node.items():
                                if hasattr(v, 'isoformat'):
                                    node[k] = v.isoformat()
                            node["_labels"] = record["labels"]
                            results.append(node)

                    # Apply composite scoring with real similarity
                    scored = []
                    for r in results:
                        uid = r.get("uuid", "")
                        sim = similarity_map.get(uid, 0.5)
                        cs = composite_score(
                            semantic_similarity=sim,
                            created_at=r.get("created_at"),
                            importance=r.get("importance", 0.8),
                            access_count=r.get("access_count", 0),
                            memory_type=r.get("memory_type", r.get("episode_type", "fact")),
                        )
                        r["composite_score"] = cs
                        r["semantic_similarity"] = sim
                        scored.append(r)

                    # Apply temporal filter if requested
                    if as_of:
                        scored = filter_by_date(scored, as_of)

                    scored.sort(key=lambda x: x["composite_score"], reverse=True)

                    return {
                        "results": scored[:limit],
                        "count": len(scored),
                        "query": query,
                        "group_id": group_id,
                        "search_mode": "semantic",
                        "filters": {"room": room, "hall": hall, "as_of": as_of},
                    }
            except Exception as e:
                logger.warning(f"ChromaDB search failed, falling back to text: {e}")

        # Fallback: original text-based search
        try:
            cypher_parts = []
            params: dict[str, Any] = {"limit": limit}

            for label in ["EntityNode", "EpisodicNode", "Episode", "Entity"]:
                where_clauses = []
                if group_id:
                    where_clauses.append("n.group_id = $group_id")
                    params["group_id"] = group_id
                if memory_type:
                    where_clauses.append("(n.memory_type = $memory_type OR n.episode_type = $memory_type)")
                    params["memory_type"] = memory_type

                text_match = (
                    "(toLower(n.name) CONTAINS toLower($search_text) "
                    "OR toLower(coalesce(n.content, '')) CONTAINS toLower($search_text) "
                    "OR toLower(coalesce(n.summary, '')) CONTAINS toLower($search_text) "
                    "OR toLower(coalesce(n.fact, '')) CONTAINS toLower($search_text))"
                )
                where_clauses.append(text_match)
                params["search_text"] = query

                where_str = " AND ".join(where_clauses)
                cypher_parts.append(
                    f"MATCH (n:{label}) WHERE {where_str} "
                    f"RETURN n, labels(n) AS labels"
                )

            cypher = " UNION ".join(cypher_parts) + " LIMIT $limit"

            results = []
            with driver.session() as db:
                records = db.run(cypher, parameters=params)
                for record in records:
                    node = dict(record["n"])
                    for k, v in node.items():
                        if hasattr(v, 'isoformat'):
                            node[k] = v.isoformat()
                    node["_labels"] = record["labels"]
                    results.append(node)

            scored = []
            for r in results:
                cs = composite_score(
                    semantic_similarity=0.7,
                    created_at=r.get("created_at"),
                    importance=r.get("importance", 0.8),
                    access_count=r.get("access_count", 0),
                    memory_type=r.get("memory_type", r.get("episode_type", "fact")),
                )
                r["composite_score"] = cs
                r["semantic_similarity"] = 0.7
                scored.append(r)

            if as_of:
                scored = filter_by_date(scored, as_of)

            scored.sort(key=lambda x: x["composite_score"], reverse=True)

            return {
                "results": scored[:limit],
                "count": len(scored),
                "query": query,
                "group_id": group_id,
                "search_mode": "text_fallback",
            }

        except Exception as e:
            logger.error(f"scored_search failed: {e}")
            return {"error": str(e), "query": query, "group_id": group_id}

    # ── classify_memory (v2: confidence + sentiment) ─────────────────

    elif name == "classify_memory":
        text = args["text"]
        use_llm = args.get("use_llm", False)
        detailed = args.get("detailed", False)

        result = do_classify(text, use_llm=use_llm, detailed=detailed)

        if detailed and isinstance(result, dict):
            result["description"] = MEMORY_TYPES.get(result.get("type", ""), "Unknown type")
            result["text_preview"] = text[:100]
            return result
        else:
            mem_type = result if isinstance(result, str) else result.get("type", "fact")
            return {
                "memory_type": mem_type,
                "description": MEMORY_TYPES.get(mem_type, "Unknown type"),
                "text_preview": text[:100],
            }

    elif name == "lifecycle_status":
        lc = get_lifecycle()
        status = lc.get_status(args["memory_id"])
        valid_next = list(VALID_TRANSITIONS.get(status, set()))
        return {
            "memory_id": args["memory_id"],
            "status": status,
            "valid_transitions": valid_next,
        }

    elif name == "lifecycle_transition":
        lc = get_lifecycle()
        success = lc.transition(
            args["memory_id"],
            args["from_status"],
            args["to_status"],
        )
        return {
            "success": success,
            "memory_id": args["memory_id"],
            "from": args["from_status"],
            "to": args["to_status"],
        }

    elif name == "bulk_archive_stale":
        lc = get_lifecycle()
        result = lc.bulk_archive_stale(
            threshold_days=args.get("threshold_days", 30),
            group_id=args.get("group_id"),
            dry_run=args.get("dry_run", True),
        )
        return result

    elif name == "compact_session":
        ce = get_compactor()
        return ce.compact_session(
            session_id=args["session_id"],
            group_id=args.get("group_id"),
        )

    elif name == "compact_daily":
        ce = get_compactor()
        return ce.daily_digest(
            group_id=args.get("group_id"),
            lookback_hours=args.get("lookback_hours", 24),
        )

    elif name == "compact_weekly":
        ce = get_compactor()
        return ce.weekly_merge(group_id=args.get("group_id"))

    elif name == "compaction_status":
        ce = get_compactor()
        return ce.get_compaction_status()

    elif name == "memory_stats":
        lc = get_lifecycle()
        ce = get_compactor()
        store = get_embed_store()
        group_id = args.get("group_id")
        return {
            "lifecycle": lc.get_lifecycle_stats(group_id=group_id),
            "compaction": ce.get_compaction_status(),
            "chromadb": {
                "available": store.health_check(),
                "embedding_count": store.count(),
            },
        }

    # ── supersede_memory (v2: auto-set valid_to) ─────────────────────

    elif name == "supersede_memory":
        lc = get_lifecycle()
        driver = get_driver()
        success = lc.supersede(args["old_id"], args["new_id"])
        # Auto-set valid_to on the superseded memory
        if success:
            now = datetime.now(timezone.utc).isoformat()
            set_validity(driver, args["old_id"], valid_to=now)
        return {"success": success, "old_id": args["old_id"], "new_id": args["new_id"]}

    # ── contradict_memory (v2: auto-set valid_to) ────────────────────

    elif name == "contradict_memory":
        lc = get_lifecycle()
        driver = get_driver()
        success = lc.contradict(args["memory_id"], args["contradicting_id"])
        # Auto-set valid_to on the contradicted memory
        if success:
            now = datetime.now(timezone.utc).isoformat()
            set_validity(driver, args["memory_id"], valid_to=now)
        return {"success": success, **args}

    elif name == "restore_memory":
        lc = get_lifecycle()
        success = lc.restore(args["memory_id"])
        return {"success": success, "memory_id": args["memory_id"]}

    # ── Conversation Persistence Tools ────────────────────────────────

    elif name == "save_episode":
        group_id = args.get("group_id", DEFAULT_GROUP_ID)
        session_id = args.get("session_id") or _get_or_create_session_id(group_id)

        sm = SessionManager()
        er = EpisodeRecorder(driver=sm._driver)
        episode_type = args.get("episode_type")
        content = args["content"]

        episode_id = er.record_episode(
            session_id=session_id,
            content=content,
            episode_type=episode_type,
            group_id=group_id,  # explicit — don't rely on session inheritance
            importance=args.get("importance", 0.8),
        )
        sm.close()

        if episode_id is None:
            return {"saved": False, "reason": "Episode filtered (not significant enough or too short)"}

        # v2: Dual-write to ChromaDB
        store = get_embed_store()
        if not episode_type:
            episode_type = do_classify(content) if isinstance(do_classify(content), str) else "fact"
        _chromadb_write(store, episode_id, content, group_id, episode_type)

        return {"saved": True, "episode_id": episode_id, "session_id": session_id}

    elif name == "save_state":
        group_id = DEFAULT_GROUP_ID
        session_id = args.get("session_id") or _get_or_create_session_id(group_id)

        snapshot_data = {
            "type": "session_snapshot",
            "task": args["task"],
            "status": args.get("status", "in_progress"),
            "completed": args.get("completed", []),
            "in_progress": args.get("in_progress", []),
            "next_steps": args.get("next_steps", []),
            "key_decisions": args.get("key_decisions", []),
            "blockers": args.get("blockers", []),
            "files_modified": args.get("files_modified", []),
        }

        sm = SessionManager()
        snm = SnapshotManager(driver=sm._driver)
        snapshot_id = snm.save_snapshot(session_id, snapshot_data)
        sm.close()

        if snapshot_id is None:
            return {"saved": False, "error": "Failed to save snapshot"}
        return {"saved": True, "snapshot_id": snapshot_id, "session_id": session_id}

    elif name == "get_session":
        sm = SessionManager()
        er = EpisodeRecorder(driver=sm._driver)

        session_id = args["session_id"]

        with sm._driver.session() as db:
            result = db.run(
                "MATCH (s:Session {uuid: $uuid}) RETURN s",
                uuid=session_id,
            )
            record = result.single()
            if record is None:
                sm.close()
                return {"error": f"Session {session_id} not found"}
            session_data = {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(record["s"]).items()}

        episodes = er.get_session_episodes(session_id)

        with sm._driver.session() as db:
            result = db.run(
                """
                MATCH (s:Session {uuid: $uuid})-[:HAS_SNAPSHOT]->(snap:Snapshot)
                RETURN snap ORDER BY snap.created_at DESC LIMIT 1
                """,
                uuid=session_id,
            )
            snap_record = result.single()
            snapshot = json.loads(snap_record["snap"]["data"]) if snap_record else None

        sm.close()
        return {
            "session": session_data,
            "episodes": episodes,
            "snapshot": snapshot,
            "episode_count": len(episodes),
        }

    elif name == "list_sessions":
        sm = SessionManager()
        sessions = sm.list_sessions(
            group_id=args["group_id"],
            limit=args.get("limit", 10),
        )
        sm.close()
        return {"sessions": sessions, "count": len(sessions)}

    elif name == "continue_session":
        group_id = args["group_id"]
        target_session_id = args.get("session_id")

        sm = SessionManager()
        er = EpisodeRecorder(driver=sm._driver)
        snm = SnapshotManager(driver=sm._driver)

        if target_session_id:
            with sm._driver.session() as db:
                result = db.run(
                    "MATCH (s:Session {uuid: $uuid}) RETURN s",
                    uuid=target_session_id,
                )
                record = result.single()
                prev_session = {k: str(v) if hasattr(v, 'isoformat') else v for k, v in dict(record["s"]).items()} if record else None
        else:
            prev_session = sm.get_latest_session(group_id)

        if not prev_session:
            sm.close()
            return {"error": f"No previous session found for group '{group_id}'"}

        prev_id = prev_session.get("uuid")
        snapshot = snm.get_latest_snapshot(group_id)
        episodes = er.get_session_episodes(prev_id)
        chain = sm.get_session_chain(prev_id)
        sm.close()

        return {
            "previous_session": prev_session,
            "snapshot": snapshot,
            "episodes": episodes,
            "session_chain": chain,
            "context_summary": SnapshotManager.format_snapshot_for_injection(snapshot) if snapshot else "No snapshot available",
        }

    elif name == "session_handoff":
        group_id = DEFAULT_GROUP_ID
        session_id = args.get("session_id") or _get_or_create_session_id(group_id)

        sm = SessionManager()
        snm = SnapshotManager(driver=sm._driver)

        snapshot_data = {
            "type": "handoff_snapshot",
            "task": args["task"],
            "status": "handoff",
            "next_steps": args.get("next_steps", []),
            "notes": args.get("notes", ""),
            "device": DEVICE_ID,
        }
        snapshot_id = snm.save_snapshot(session_id, snapshot_data)
        sm.end_session(session_id, status="handoff")
        sm.close()

        return {
            "handoff_ready": True,
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "task": args["task"],
            "next_steps": args.get("next_steps", []),
            "notes": args.get("notes", ""),
            "instruction": "Next session on any device: call continue_session with the group_id to pick up where this left off.",
        }

    # ── v2 Tools ──────────────────────────────────────────────────────

    elif name == "wake_up":
        store = get_embed_store()
        driver = get_driver()
        return do_wake_up(store, driver, args["group_id"])

    elif name == "set_fact_validity":
        driver = get_driver()
        return set_validity(
            driver,
            args["memory_id"],
            valid_from=args.get("valid_from"),
            valid_to=args.get("valid_to"),
        )

    elif name == "fact_timeline":
        driver = get_driver()
        timeline = get_timeline(
            driver,
            entity=args["entity"],
            group_id=args.get("group_id"),
            limit=args.get("limit", 50),
        )
        return {"entity": args["entity"], "timeline": timeline, "count": len(timeline)}

    elif name == "search_rooms":
        driver = get_driver()
        group_id = args["group_id"]
        try:
            with driver.session() as db:
                result = db.run(
                    """
                    MATCH (n)
                    WHERE n.group_id = $gid
                      AND n.room IS NOT NULL
                      AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                    RETURN n.room AS room, count(n) AS count
                    ORDER BY count DESC
                    """,
                    gid=group_id,
                )
                rooms = [{"room": r["room"], "count": r["count"]} for r in result]
            return {"group_id": group_id, "rooms": rooms, "total_rooms": len(rooms)}
        except Exception as e:
            return {"error": str(e)}

    else:
        return {"error": f"Unknown tool: {name}"}


async def main():
    """Run the MCP server over stdio."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_sync() -> None:
    """Synchronous entrypoint for the ``jarvis-mcp`` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()


# === TRUST BOUNDARY — RUN 4 ===
#
# Every MCP tool invocation sets a ``contextvars.ContextVar`` to an
# ``OperationContext(remote=True, source="mcp", caller=...)`` marker.
# Downstream code that accepts an optional ``ctx`` kwarg (e.g.
# ``EpisodeRecorder.record_episode``) can read this to apply the
# trust-boundary audit defined in ``jarvis_memory.conversation``'s own
# trust-boundary block.
#
# This block is placed at EOF so Run 2's tool-list additions land earlier
# in the file without conflicting with our additions.
#
# Logged-only in this run — no refusals, no new error paths.

import contextvars as _ctxvars_run4  # noqa: E402 — deferred import at EOF
import functools as _functools_run4  # noqa: E402

from jarvis_memory.operation_context import OperationContext as _OperationContext_run4  # noqa: E402

# ContextVar holding the current MCP request's OperationContext.
_MCP_CTX: _ctxvars_run4.ContextVar[_OperationContext_run4 | None] = _ctxvars_run4.ContextVar(
    "jarvis_memory_mcp_ctx", default=None,
)


def current_mcp_context() -> _OperationContext_run4 | None:
    """Read the current MCP request's OperationContext (None if no MCP call is in flight)."""
    return _MCP_CTX.get()


def _extract_caller(arguments: dict[str, Any]) -> str:
    """Best-effort caller-identity extraction from MCP tool arguments.

    Agents typically don't identify themselves explicitly — fall back to
    a generic "mcp-agent" label. Real identification is a future concern.
    """
    return str(arguments.get("_mcp_caller") or arguments.get("caller") or "mcp-agent")


_ORIGINAL_DISPATCH_RUN4 = _dispatch


async def _dispatch_with_ctx(name, arguments, *args, **kwargs):  # type: ignore[no-redef]
    """Wrap ``_dispatch`` so every MCP tool call runs under an MCP OperationContext.

    Tools that inject ``ctx`` into internal calls (like ``save_episode`` →
    ``record_episode``) can read the context via ``current_mcp_context()``.
    Tools that don't care are unaffected.
    """
    ctx = _OperationContext_run4.for_mcp(_extract_caller(arguments or {}))
    token = _MCP_CTX.set(ctx)
    try:
        return await _ORIGINAL_DISPATCH_RUN4(name, arguments, *args, **kwargs)
    finally:
        _MCP_CTX.reset(token)


# Monkey-patch: redirect the module-global name ``_dispatch`` to the wrapped
# version. ``create_server`` above references ``_dispatch`` via a closure in
# ``call_tool``, so this replacement is picked up for every new server.
_dispatch = _dispatch_with_ctx  # type: ignore[assignment]


__all_run4__ = [
    "current_mcp_context",
    "_MCP_CTX",
]

# === END TRUST BOUNDARY — RUN 4 ===
