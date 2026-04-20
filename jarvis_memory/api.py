"""jarvis-memory REST API — standalone FastAPI server.

Replaces the memclawz shim. Only serves v2 endpoints backed by
Neo4j + ChromaDB. No Mem0, no Qdrant, no legacy v1 routes.

Run:
    python -m jarvis_memory.api
    # or: uvicorn jarvis_memory.api:app --host 0.0.0.0 --port 3500

Env vars:
    NEO4J_URI          bolt://localhost:7687
    NEO4J_USER         neo4j
    NEO4J_PASSWORD     neo4j
    JARVIS_API_HOST    0.0.0.0
    JARVIS_API_PORT    3500
    JARVIS_DEVICE_ID   mac-mini
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import (
    API_HOST,
    API_PORT,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    DEVICE_ID,
)

logger = logging.getLogger("jarvis_memory.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# ── Neo4j + ChromaDB singletons ────────────────────────────────────────

_neo4j_driver = None
_embed_store = None


def _get_driver():
    """Lazy Neo4j driver with reconnect on failure."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        try:
            _neo4j_driver.verify_connectivity()
            return _neo4j_driver
        except Exception:
            logger.warning("Neo4j driver stale, reconnecting...")
            try:
                _neo4j_driver.close()
            except Exception:
                pass
            _neo4j_driver = None

    try:
        from neo4j import GraphDatabase

        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        _neo4j_driver.verify_connectivity()
        logger.info(f"Neo4j connected: {NEO4J_URI}")
        return _neo4j_driver
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")


def _get_embed_store():
    """Lazy ChromaDB store — returns None if unavailable (graceful degradation)."""
    global _embed_store
    if _embed_store is None:
        try:
            from .embeddings import EmbeddingStore

            _embed_store = EmbeddingStore()
            if not _embed_store.health_check():
                logger.warning("ChromaDB health check failed")
        except Exception as e:
            logger.warning(f"ChromaDB init failed (will use fallback): {e}")
    return _embed_store


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Neo4j. Shutdown: close driver."""
    try:
        _get_driver()
        _get_embed_store()
        logger.info("jarvis-memory API ready")
    except Exception as e:
        logger.warning(f"Startup check failed (will retry on first request): {e}")
    yield
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None
        logger.info("Neo4j driver closed")


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="jarvis-memory API",
    version="2.0.0",
    description="Standalone REST API for jarvis-memory (Neo4j + ChromaDB). No legacy Mem0/Qdrant.",
    lifespan=lifespan,
)


# ── Request Models ──────────────────────────────────────────────────────

class WakeUpRequest(BaseModel):
    group_id: str


class ScoredSearchRequest(BaseModel):
    query: str
    group_id: Optional[str] = None
    room: Optional[str] = None
    hall: Optional[str] = None
    as_of: Optional[str] = None
    limit: int = 10
    memory_type: Optional[str] = None


class SaveEpisodeRequest(BaseModel):
    content: str
    group_id: str
    episode_type: Optional[str] = None
    importance: float = 0.8
    source: str = "atlas"


class FactValidityRequest(BaseModel):
    memory_id: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class FactTimelineRequest(BaseModel):
    entity: str
    group_id: Optional[str] = None
    limit: int = 50


class SessionHandoffRequest(BaseModel):
    task: str
    group_id: str
    next_steps: list[str] = []
    notes: str = ""


class SaveStateRequest(BaseModel):
    task: str
    group_id: str
    status: str = "in_progress"
    completed: list[str] = []
    in_progress: list[str] = []
    next_steps: list[str] = []
    blockers: list[str] = []
    key_decisions: list[str] = []
    files_modified: list[str] = []


# === RUN 2 — ENTITY LAYER ===
# 4 REST endpoints surfacing the Page + typed-edge knowledge graph:
#   GET /api/v2/orphans     — find_orphans
#   GET /api/v2/doctor      — run_health_checks
#   GET /api/v2/page/{slug} — get_page
#   GET /api/v2/pages       — list_pages
# Response shapes mirror the MCP tool returns. All 503 if Neo4j is
# unreachable (same pattern as the rest of /api/v2).

@app.get("/api/v2/orphans")
async def run2_orphans(domain: Optional[str] = None):
    """Find Pages with zero inbound typed edges, grouped by domain."""
    try:
        from .orphans import find_orphans

        driver = _get_driver()
        grouped = find_orphans(domain=domain, driver=driver)
        return {
            "count": sum(len(v) for v in grouped.values()),
            "by_domain": {d: [p.to_dict() for p in pages] for d, pages in grouped.items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"orphans failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/doctor")
async def run2_doctor(fast: bool = False):
    """Run entity-layer health checks; returns report dict."""
    try:
        from .doctor import run_health_checks

        driver = _get_driver()
        return run_health_checks(driver=driver, fast=fast)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"doctor failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/page/{slug}")
async def run2_get_page(slug: str):
    """Fetch a single Page by slug."""
    try:
        from .pages import get_page

        driver = _get_driver()
        page = get_page(slug, driver=driver)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{slug}' not found")
        return {"page": page.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_page failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/pages")
async def run2_list_pages(
    domain: Optional[str] = None,
    limit: int = 100,
):
    """List Pages, optionally filtered by domain, newest-first."""
    try:
        from .pages import list_pages

        driver = _get_driver()
        pages = list_pages(domain=domain, driver=driver, limit=limit)
        return {"count": len(pages), "pages": [p.to_dict() for p in pages]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"list_pages failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === END RUN 2 — ENTITY LAYER ===


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — Neo4j + ChromaDB status."""
    result = {
        "status": "ok",
        "version": "2.0.0",
        "device": DEVICE_ID,
        "neo4j": "unknown",
        "chromadb": "unknown",
    }
    try:
        driver = _get_driver()
        with driver.session() as db:
            r = db.run("MATCH (n) RETURN count(n) AS cnt").single()
            result["neo4j"] = "ok"
            result["neo4j_nodes"] = r["cnt"]
    except Exception as e:
        result["neo4j"] = f"error: {e}"
        result["status"] = "degraded"

    try:
        store = _get_embed_store()
        if store and store.health_check():
            result["chromadb"] = "ok"
            result["chromadb_count"] = store.count()
        else:
            result["chromadb"] = "unavailable"
    except Exception as e:
        result["chromadb"] = f"error: {e}"

    return result


# ── v1 compatibility shim (for OpenClaw hooks during migration) ─────────

class V1AddRequest(BaseModel):
    """Accepts the old memclawz /api/v1/add payload format."""
    content: str
    user_id: str = "yoni"
    agent_id: str = "main"
    memory_type: str = "fact"
    metadata: Optional[dict] = None


@app.post("/api/v1/add")
async def v1_add_compat(req: V1AddRequest):
    """Thin v1 compatibility layer — routes to save_episode.

    OpenClaw's mem0-extractor hook POSTs to /api/v1/add. This shim
    catches those requests and routes them through the v2 pipeline
    so nothing breaks during migration.
    """
    group_id = "system"
    if req.metadata and "group_id" in req.metadata:
        group_id = req.metadata["group_id"]

    return await save_episode(SaveEpisodeRequest(
        content=req.content,
        group_id=group_id,
        episode_type=req.memory_type if req.memory_type != "fact" else None,
        source="openclaw-hook-compat",
    ))


@app.get("/api/v1/search")
async def v1_search_compat(
    q: str,
    user_id: str = "yoni",
    limit: int = 20,
    agent_id: Optional[str] = None,
    memory_type: Optional[str] = None,
):
    """v1 compat: basic search — routes to scored_search.

    Used by: jarvis-start-guard hook (handler.ts).
    Returns: {results: [...], count: N} in v1 shape.
    """
    v2_result = await scored_search(ScoredSearchRequest(
        query=q,
        limit=limit,
    ))
    # Reshape v2 results to v1 format (hooks expect 'memory' field, 'id' field)
    v1_results = []
    for r in v2_result.get("results", []):
        v1_results.append({
            "id": r.get("uuid", r.get("id", "")),
            "memory": r.get("content", r.get("name", "")),
            "created_at": r.get("created_at", ""),
            "agent_id": r.get("source", "unknown"),
            "group_id": r.get("group_id", ""),
            "metadata": {
                "memory": r.get("content", r.get("name", "")),
                "group_id": r.get("group_id", ""),
            },
        })
    return {"results": v1_results, "count": len(v1_results)}


@app.get("/api/v1/hybrid-search")
async def v1_hybrid_search_compat(
    q: str,
    group_id: Optional[str] = None,
    limit: int = 5,
):
    """v1 compat: hybrid search — routes to scored_search with group filter.

    Used by: session_start.py hook.
    Returns: {results: [...]} — hook reads the results array directly.
    """
    v2_result = await scored_search(ScoredSearchRequest(
        query=q,
        group_id=group_id,
        limit=limit,
    ))
    return {"results": v2_result.get("results", [])}


class V1CompactSessionRequest(BaseModel):
    session_id: str
    group_id: Optional[str] = None


@app.post("/api/v1/compact/session")
async def v1_compact_session_compat(req: V1CompactSessionRequest):
    """v1 compat: session compaction trigger — fire-and-forget.

    Used by: session_stop.py hook. Just logs and returns OK.
    The actual compaction runs via the jarvis-memory compaction module.
    """
    try:
        from .compaction import CompactionEngine

        engine = CompactionEngine(driver=_get_driver(), embedding_store=_get_embed_store())
        result = engine.compact_session(req.session_id)
        engine.close()
        return {"status": "ok", "result": result}
    except ImportError:
        logger.warning("compaction module not available, skipping session compact")
        return {"status": "skipped", "reason": "compaction module unavailable"}
    except Exception as e:
        logger.warning(f"session compaction failed (non-fatal): {e}")
        return {"status": "error", "error": str(e)}


# ── Core v2 Endpoints ──────────────────────────────────────────────────

@app.post("/api/v2/wake_up")
async def wake_up(req: WakeUpRequest):
    """Token-budgeted context loading.

    Returns Layer 0 (identity) + Layer 1 (essential story) as a
    pre-formatted context block. Call at session start.
    """
    try:
        from .wake_up import wake_up as do_wake_up

        store = _get_embed_store()
        driver = _get_driver()
        return do_wake_up(store, driver, req.group_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"wake_up failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/scored_search")
async def scored_search(req: ScoredSearchRequest):
    """Semantic search with composite scoring and room/hall/temporal filters.

    Uses ChromaDB for real semantic similarity when available,
    falls back to Neo4j text matching.
    """
    try:
        from .scoring import score_results
        from .rooms import detect_room, get_hall
        from .temporal import filter_by_date

        store = _get_embed_store()
        driver = _get_driver()
        results = []

        # Try ChromaDB semantic search first
        if store and store.health_check():
            try:
                where_filter = {}
                if req.group_id:
                    where_filter["wing"] = req.group_id
                if req.room:
                    where_filter["room"] = req.room
                if req.hall:
                    where_filter["hall"] = req.hall
                if req.memory_type:
                    where_filter["memory_type"] = req.memory_type

                chromadb_results = store.search(
                    query=req.query,
                    n_results=min(req.limit * 3, 60),
                    where=where_filter if where_filter else None,
                )

                # Enrich with Neo4j metadata
                for cr in chromadb_results:
                    mem_id = cr["id"]
                    try:
                        with driver.session() as db:
                            node = db.run(
                                "MATCH (n) WHERE n.uuid = $uid RETURN n",
                                uid=mem_id,
                            ).single()
                            if node:
                                props = dict(node["n"])
                                props["similarity"] = cr.get("similarity", 0.7)
                                results.append(props)
                    except Exception:
                        results.append({
                            "uuid": mem_id,
                            "content": cr.get("document", ""),
                            "similarity": cr.get("similarity", 0.7),
                            **cr.get("metadata", {}),
                        })
            except Exception as e:
                logger.warning(f"ChromaDB search failed, falling back to Neo4j: {e}")

        # Fallback: Neo4j text search
        if not results:
            try:
                with driver.session() as db:
                    cypher = """
                        MATCH (n)
                        WHERE (n.content CONTAINS $q OR n.name CONTAINS $q
                               OR n.summary CONTAINS $q)
                    """
                    params = {"q": req.query, "lim": req.limit * 2}
                    if req.group_id:
                        cypher += " AND n.group_id = $gid"
                        params["gid"] = req.group_id
                    if req.room:
                        cypher += " AND n.room = $room"
                        params["room"] = req.room
                    if req.hall:
                        cypher += " AND n.hall = $hall"
                        params["hall"] = req.hall
                    cypher += """
                        AND coalesce(n.lifecycle_status, 'active') IN ['active', 'confirmed']
                        RETURN n ORDER BY n.created_at DESC LIMIT $lim
                    """
                    rows = db.run(cypher, **params)
                    results = [dict(r["n"]) for r in rows]
            except Exception as e:
                logger.warning(f"Neo4j fallback search failed: {e}")

        # Temporal filter
        if req.as_of:
            results = filter_by_date(results, as_of=req.as_of)

        # Composite scoring
        scored = score_results(results)

        return {
            "results": scored[: req.limit],
            "count": min(len(scored), req.limit),
            "query": req.query,
            "group_id": req.group_id,
            "search_mode": "semantic" if store and store.health_check() else "neo4j_text",
            "filters": {
                "room": req.room,
                "hall": req.hall,
                "as_of": req.as_of,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"scored_search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/save_episode")
async def save_episode(req: SaveEpisodeRequest):
    """Save a memory episode with auto-tagging (room, hall, ChromaDB dual-write).

    This is the primary write path. Routes through the Neo4j session
    system with full metadata.
    """
    try:
        from .conversation import SessionManager, EpisodeRecorder
        from .classifier import classify_memory as do_classify
        from .rooms import detect_room, get_hall

        driver = _get_driver()
        store = _get_embed_store()

        sm = SessionManager()
        er = EpisodeRecorder(driver=sm._driver)

        # Auto-classify if not provided
        episode_type = req.episode_type
        if not episode_type:
            result = do_classify(req.content)
            episode_type = result if isinstance(result, str) else "fact"

        # Get or create session
        latest = sm.get_latest_session(req.group_id)
        if latest:
            session_id = latest["uuid"]
        else:
            session_result = sm.create_session(
                group_id=req.group_id,
                device=req.source or DEVICE_ID,
                task_summary=req.content[:200],
            )
            session_id = session_result["uuid"]

        # Record episode
        episode_id = er.record_episode(
            session_id=session_id,
            content=req.content,
            episode_type=episode_type,
            importance=req.importance,
        )
        sm.close()

        if episode_id is None:
            return {
                "saved": False,
                "reason": "Episode filtered (not significant enough or too short)",
            }

        # Set room/hall on Neo4j node
        room = detect_room(req.content, req.group_id)
        hall = get_hall(episode_type)
        try:
            with _get_driver().session() as db:
                db.run(
                    """
                    MATCH (n) WHERE n.uuid = $uid
                    SET n.room = $room, n.hall = $hall, n.group_id = $gid
                    """,
                    uid=episode_id,
                    room=room,
                    hall=hall,
                    gid=req.group_id,
                )
        except Exception as e:
            logger.warning(f"Failed to set room/hall on {episode_id}: {e}")

        # Dual-write to ChromaDB
        if store and store.health_check():
            try:
                metadata = {
                    "wing": req.group_id,
                    "room": room,
                    "hall": hall,
                    "memory_type": episode_type,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source": req.source,
                }
                store.embed_and_store(episode_id, req.content, metadata)
            except Exception as e:
                logger.warning(f"ChromaDB dual-write failed: {e}")

        return {
            "saved": True,
            "episode_id": episode_id,
            "session_id": session_id,
            "room": room,
            "hall": hall,
            "episode_type": episode_type,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"save_episode failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/fact/validity")
async def set_fact_validity(req: FactValidityRequest):
    """Set temporal validity bounds on a memory."""
    try:
        from .temporal import set_validity

        driver = _get_driver()
        return set_validity(
            driver,
            req.memory_id,
            valid_from=req.valid_from,
            valid_to=req.valid_to,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"set_fact_validity failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/fact/timeline")
async def fact_timeline(req: FactTimelineRequest):
    """Chronological fact history for an entity."""
    try:
        from .temporal import get_timeline

        driver = _get_driver()
        timeline = get_timeline(
            driver,
            entity=req.entity,
            group_id=req.group_id,
            limit=req.limit,
        )
        return {"entity": req.entity, "timeline": timeline, "count": len(timeline)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"fact_timeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/rooms/{group_id}")
async def search_rooms(group_id: str):
    """List all rooms (topics) with memory counts for a project."""
    try:
        driver = _get_driver()
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"search_rooms failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/stats")
async def memory_stats(group_id: Optional[str] = None):
    """System statistics: lifecycle counts, ChromaDB status, per-group counts."""
    try:
        driver = _get_driver()
        store = _get_embed_store()

        stats = {"lifecycle": {}, "chromadb": {}, "compaction": {}}

        # Lifecycle counts
        with driver.session() as db:
            cypher = """
                MATCH (n)
                WHERE true
            """
            params = {}
            if group_id:
                cypher += " AND n.group_id = $gid"
                params["gid"] = group_id
            cypher += """
                RETURN coalesce(n.lifecycle_status, 'active') AS status, count(n) AS cnt
            """
            rows = db.run(cypher, **params)
            for r in rows:
                stats["lifecycle"][r["status"]] = r["cnt"]

        # ChromaDB
        if store and store.health_check():
            stats["chromadb"] = {
                "available": True,
                "embedding_count": store.count(),
            }
        else:
            stats["chromadb"] = {"available": False}

        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"memory_stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session endpoints ───────────────────────────────────────────────────

@app.get("/api/v2/sessions/{group_id}")
async def list_sessions(group_id: str, limit: int = 10):
    """List recent sessions for a project."""
    try:
        from .conversation import SessionManager

        sm = SessionManager()
        sessions = sm.list_sessions(group_id, limit=limit)
        sm.close()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"list_sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/session/handoff")
async def session_handoff(req: SessionHandoffRequest):
    """Prepare a handoff package for cross-device continuation."""
    try:
        from .conversation import SessionManager

        sm = SessionManager()
        # Save state first
        latest = sm.get_latest_session(req.group_id)
        if not latest:
            sm.close()
            raise HTTPException(status_code=404, detail=f"No active session for {req.group_id}")

        session_id = latest["uuid"]
        sm.save_state(
            session_id=session_id,
            task=req.task,
            status="handoff",
            next_steps=req.next_steps,
        )
        sm.close()

        return {
            "status": "handoff_ready",
            "session_id": session_id,
            "group_id": req.group_id,
            "task": req.task,
            "next_steps": req.next_steps,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"session_handoff failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/session/save_state")
async def save_state(req: SaveStateRequest):
    """Save a full session state snapshot."""
    try:
        from .conversation import SessionManager

        sm = SessionManager()
        latest = sm.get_latest_session(req.group_id)
        if not latest:
            sm.close()
            raise HTTPException(status_code=404, detail=f"No active session for {req.group_id}")

        session_id = latest["uuid"]
        sm.save_state(
            session_id=session_id,
            task=req.task,
            status=req.status,
            completed=req.completed,
            in_progress=req.in_progress,
            next_steps=req.next_steps,
            blockers=req.blockers,
            key_decisions=req.key_decisions,
            files_modified=req.files_modified,
        )
        sm.close()

        return {"status": "saved", "session_id": session_id, "group_id": req.group_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"save_state failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    import uvicorn

    logger.info(f"Starting jarvis-memory API on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
