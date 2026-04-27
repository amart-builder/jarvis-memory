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


def _load_env_file() -> None:
    """Load ``.env`` into ``os.environ`` before :mod:`.config` reads it.

    ``jarvis_memory.api`` may be launched via ``launchd`` (with
    ``--noprofile --norc``) or by a bare ``python -m jarvis_memory.api``
    command in a shell that never sourced ``.env``. In both cases
    ``os.getenv()`` would see the code defaults (``NEO4J_PASSWORD="neo4j"``)
    and silently fail Neo4j auth on any machine with a real password.

    Uses ``setdefault`` so env vars already exported by the parent shell
    win; this stays compatible with the venv-launcher pattern that does
    ``source .env``. Parser matches :mod:`scripts.run_compaction`.
    """
    import os
    from pathlib import Path

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), val)


_load_env_file()


import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

# ── Bearer-auth helpers (middleware registered below, after `app`) ─────
#
# Fixes Astack VPS ship-gate bug B8: the REST API previously enforced no
# auth on non-health routes, so a leaked port = a writable memory store.

_AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _bearer_token() -> Optional[str]:
    tok = os.environ.get("JARVIS_API_BEARER_TOKEN")
    return tok.strip() if tok and tok.strip() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Neo4j. Shutdown: close driver."""
    try:
        _get_driver()
        _get_embed_store()
        logger.info("jarvis-memory API ready")
    except Exception as e:
        logger.warning(f"Startup check failed (will retry on first request): {e}")

    # Non-loopback + no-bearer: shout loudly (or refuse if strict).
    if not _is_loopback_host(API_HOST) and _bearer_token() is None:
        strict = os.environ.get("JARVIS_REQUIRE_AUTH", "0") == "1"
        msg = (
            f"API bound to non-loopback host {API_HOST!r} with no "
            "JARVIS_API_BEARER_TOKEN set. Any process that reaches this "
            "port can read/write memory. Set JARVIS_API_BEARER_TOKEN or "
            "bind to 127.0.0.1 to close this hole."
        )
        if strict:
            raise RuntimeError(msg + " (JARVIS_REQUIRE_AUTH=1)")
        logger.warning("SECURITY: %s", msg)

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


# ── Bearer-auth middleware (registered on `app`) ───────────────────────
#
# If JARVIS_API_BEARER_TOKEN is set, every non-exempt request must present
# "Authorization: Bearer <token>". No token + loopback bind = allow-all
# (local-dev default). No token + non-loopback bind = allow-all with a
# startup warning (see lifespan); JARVIS_REQUIRE_AUTH=1 flips that to a
# startup refusal for strict deployments.

@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    token = _bearer_token()
    if token is None:
        return await call_next(request)

    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        return JSONResponse(
            {"detail": "Missing bearer token. Send Authorization: Bearer <token>."},
            status_code=401,
        )
    supplied = header[len("Bearer "):].strip()
    import hmac as _hmac
    if not _hmac.compare_digest(supplied, token):
        return JSONResponse({"detail": "Invalid bearer token."}, status_code=401)
    return await call_next(request)


# ── Request Models ──────────────────────────────────────────────────────

class WakeUpRequest(BaseModel):
    group_id: str


class ScoredSearchRequest(BaseModel):
    query: str
    group_id: Optional[str] = None
    room: Optional[str] = None
    hall: Optional[str] = None
    # ``as_of`` = event-time anchor ("what was true on date X?").
    # ``seen_as_of`` = ingestion-time anchor ("what did we believe on date X?").
    # Independent; pass either, both, or neither.
    as_of: Optional[str] = None
    seen_as_of: Optional[str] = None
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
    # v1.1 contract additions (all optional for backward compat):
    device: Optional[str] = None
    session_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    session_key: Optional[str] = None
    source: Optional[str] = None


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
    # v1.1 contract additions:
    device: Optional[str] = None
    session_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    session_key: Optional[str] = None
    source: Optional[str] = None


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
    """Accepts the old memclawz /api/v1/add payload format.

    v1.1 contract: ``group_id`` at top level is preferred. If missing, we fall
    back to ``metadata.group_id`` (older OpenClaw hook shape). If both missing,
    we route to ``system`` with a WARNING log so the pollution is visible.
    """
    content: str
    user_id: str = "yoni"
    agent_id: str = "main"
    memory_type: str = "fact"
    metadata: Optional[dict] = None
    # v1.1: top-level group_id. Optional for backward compat — hooks that
    # haven't been updated still land via metadata.group_id.
    group_id: Optional[str] = None


@app.post("/api/v1/add")
async def v1_add_compat(req: V1AddRequest):
    """Thin v1 compatibility layer — routes to save_episode.

    OpenClaw's mem0-extractor hook POSTs to /api/v1/add. This shim catches
    those requests and routes them through the v2 pipeline.

    group_id precedence: top-level > metadata.group_id > 'system' (with warn).
    """
    group_id: Optional[str] = None
    if req.group_id and req.group_id.strip():
        group_id = req.group_id.strip()
    elif req.metadata and isinstance(req.metadata.get("group_id"), str) and req.metadata["group_id"].strip():
        group_id = req.metadata["group_id"].strip()

    if group_id is None:
        # Don't fail hard (legacy hooks depend on this path), but surface the
        # drift loudly so it can be cleaned up. v2.0 will flip this to a 400.
        logger.warning(
            "v1_add: no group_id on /api/v1/add — falling back to 'system'. "
            "Update the caller (user_id=%s, agent_id=%s) to pass group_id at top level.",
            req.user_id, req.agent_id,
        )
        group_id = "system"

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
    """Hybrid RRF scored search with intent routing + Haiku expansion.

    Implementation notes (Run 3):
        The heavy lifting lives in :func:`jarvis_memory.scoring.scored_search`
        — ChromaDB vector + Neo4j full-text are fused via RRF, then
        boosted by Page.compiled_truth and typed-edge backlinks. The
        response envelope here is FROZEN (spec Run 3 §"Must-not-break
        flows") — any change must be a separate contract change.

    Legacy fallback:
        Setting ``JARVIS_SEARCH_LEGACY=1`` forces the Run 1 composite
        scoring path inside ``jarvis_memory.scoring.scored_search``.
    """
    try:
        import os as _os

        from .scoring import scored_search as core_scored_search

        store = _get_embed_store()
        driver = _get_driver()

        results = core_scored_search(
            req.query,
            group_id=req.group_id,
            room=req.room,
            hall=req.hall,
            memory_type=req.memory_type,
            as_of=req.as_of,
            seen_as_of=req.seen_as_of,
            limit=req.limit,
            driver=driver,
            embedding_store=store,
        )

        if _os.environ.get("JARVIS_SEARCH_LEGACY", "").strip() == "1":
            search_mode = "legacy_composite"
        elif store and store.health_check():
            search_mode = "hybrid_rrf"
        else:
            search_mode = "neo4j_text"

        return {
            "results": results[: req.limit],
            "count": min(len(results), req.limit),
            "query": req.query,
            "group_id": req.group_id,
            "search_mode": search_mode,
            "filters": {
                "room": req.room,
                "hall": req.hall,
                "as_of": req.as_of,
                "seen_as_of": req.seen_as_of,
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
    """Write a session handoff: snapshot + retrievable [HANDOFF] Episode.

    Contract:
      - ``group_id`` required (Pydantic-enforced).
      - If no session exists for the group, one is created on the fly.
      - If ``idempotency_key`` is set and a handoff with the same key
        exists in the last hour for this session, the write is a no-op
        and the existing IDs are returned.

    Returns: ``{handoff_ready, group_id, session_id, snapshot_id, episode_id,
              idempotent_hit, next_steps, task}``.
    """
    try:
        from . import handoff as handoff_module

        driver = _get_driver()
        result = handoff_module.save_handoff(
            driver,
            group_id=req.group_id,
            task=req.task,
            next_steps=req.next_steps,
            notes=req.notes,
            device=req.device or DEVICE_ID,
            session_id=req.session_id,
            idempotency_key=req.idempotency_key,
            session_key=req.session_key,
            source=req.source or "rest",
        )
        return {
            "handoff_ready": True,
            "group_id": result.group_id,
            "session_id": result.session_id,
            "snapshot_id": result.snapshot_id,
            "episode_id": result.episode_id,
            "idempotent_hit": result.idempotent_hit,
            "task": req.task,
            "next_steps": req.next_steps,
        }
    except handoff_module.GroupIDRequired as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("session_handoff failed")
        raise HTTPException(status_code=500, detail="Internal error writing handoff.")


@app.post("/api/v2/session/save_state")
async def save_state(req: SaveStateRequest):
    """Save a session state snapshot (non-terminal; no Episode written).

    Contract: ``group_id`` required. Creates a session if none exists for
    this group. Honors ``idempotency_key`` (last hour).

    Use ``/api/v2/session/handoff`` when you want a retrievable handoff
    Episode; use this when you want to checkpoint progress mid-session.
    """
    try:
        from . import handoff as handoff_module

        driver = _get_driver()
        result = handoff_module.save_state_snapshot(
            driver,
            group_id=req.group_id,
            task=req.task,
            status=req.status,
            completed=req.completed,
            in_progress=req.in_progress,
            next_steps=req.next_steps,
            blockers=req.blockers,
            key_decisions=req.key_decisions,
            files_modified=req.files_modified,
            device=req.device or DEVICE_ID,
            session_id=req.session_id,
            idempotency_key=req.idempotency_key,
            session_key=req.session_key,
            source=req.source or "rest",
        )
        return {
            "status": "saved",
            "group_id": result["group_id"],
            "session_id": result["session_id"],
            "snapshot_id": result["snapshot_id"],
            "idempotent_hit": result["idempotent_hit"],
        }
    except handoff_module.GroupIDRequired as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("save_state failed")
        raise HTTPException(status_code=500, detail="Internal error saving session state.")


@app.get("/api/v2/handoff/latest")
async def latest_handoff(group_id: str, max_age_hours: int = 72):
    """Return the most recent [HANDOFF] Episode for ``group_id``.

    Returns ``404`` if no handoff found within ``max_age_hours``.
    This is the read path the PreCompact → SessionStart flow depends on.
    """
    try:
        from . import handoff as handoff_module

        driver = _get_driver()
        result = handoff_module.get_latest_handoff(
            driver, group_id=group_id, max_age_hours=max_age_hours,
        )
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No handoff within the last {max_age_hours}h for group_id={group_id!r}",
            )
        return result
    except handoff_module.GroupIDRequired as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("latest_handoff failed")
        raise HTTPException(status_code=500, detail="Internal error reading handoff.")


@app.get("/api/v2/groups")
async def list_groups():
    """Return every known ``group_id`` with episode + session counts.

    Use this to debug "where did my memory go?" or to spot-check that
    a recent write landed in the right group.
    """
    try:
        from . import handoff as handoff_module

        driver = _get_driver()
        groups = handoff_module.list_groups(driver)
        return {"groups": groups, "count": len(groups)}
    except Exception as e:
        logger.exception("list_groups failed")
        raise HTTPException(status_code=500, detail="Internal error listing groups.")


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    import uvicorn

    logger.info(f"Starting jarvis-memory API on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()


# === TRUST BOUNDARY — RUN 4 ===
#
# FastAPI middleware that wraps every request in an ``OperationContext``
# marker ``(remote=True, source="rest", caller=<client_ip>)``. Downstream
# functions that accept an optional ``ctx`` kwarg read this via a
# contextvars-backed getter so the audit heuristic in
# ``jarvis_memory.conversation``'s own trust-boundary block can fire.
#
# Logged-only — no refusals. No new error paths.

import contextvars as _ctxvars_run4  # noqa: E402 — deferred import at EOF

from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware_run4  # noqa: E402
from starlette.requests import Request as _Request_run4  # noqa: E402

from jarvis_memory.operation_context import OperationContext as _OperationContext_run4  # noqa: E402


_REST_CTX: _ctxvars_run4.ContextVar[_OperationContext_run4 | None] = _ctxvars_run4.ContextVar(
    "jarvis_memory_rest_ctx", default=None,
)


def current_rest_context() -> _OperationContext_run4 | None:
    """Read the current REST request's OperationContext (None outside a request)."""
    return _REST_CTX.get()


class _RestTrustBoundaryMiddleware(_BaseHTTPMiddleware_run4):
    """Sets the REST OperationContext contextvar for the duration of each request."""

    async def dispatch(self, request: _Request_run4, call_next):
        client = getattr(request, "client", None)
        caller = (client.host if client and client.host else None) or "rest-unknown"
        ctx = _OperationContext_run4.for_rest(caller)
        token = _REST_CTX.set(ctx)
        try:
            return await call_next(request)
        finally:
            _REST_CTX.reset(token)


# Install the middleware on the existing ``app``. Starlette preserves insertion
# order; adding at EOF means Run 4's middleware runs OUTERMOST, so contextvars
# are set before any other middleware (and any endpoint handler) runs.
try:
    app.add_middleware(_RestTrustBoundaryMiddleware)
except Exception as _mw_exc:  # noqa: BLE001 — defensive (shouldn't happen)
    logger.warning("Run 4: failed to install trust-boundary middleware: %s", _mw_exc)


__all_run4__ = [
    "current_rest_context",
    "_REST_CTX",
]

# === END TRUST BOUNDARY — RUN 4 ===
