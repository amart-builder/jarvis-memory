"""REST parity — /api/v2 response shapes must stay unchanged across Run 1.

Run 1 adds a detect_layer warning on the write path but MUST NOT change
the JSON shape of any v2 endpoint. This test spins the FastAPI app up
in-process via TestClient — no :3500 listener needed. If Neo4j isn't
reachable we skip with a clear reason (this test cannot usefully run
without a live backend).
"""
from __future__ import annotations

import os

import pytest

# FastAPI TestClient requires httpx. Skip if missing rather than failing.
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from jarvis_memory.api import app  # noqa: E402


def _neo4j_reachable() -> bool:
    """Quick TCP probe to the configured NEO4J_URI host."""
    import socket
    from jarvis_memory.config import NEO4J_URI

    # bolt://host:port or neo4j://host:port
    try:
        # strip scheme + optional trailing path
        rest = NEO4J_URI.split("://", 1)[1]
        host, _, port_str = rest.partition(":")
        port = int(port_str.split("/")[0]) if port_str else 7687
    except Exception:
        return False
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except OSError:
        return False


NEO4J_LIVE = _neo4j_reachable()

neo4j_required = pytest.mark.skipif(
    not NEO4J_LIVE,
    reason="Neo4j is not reachable at NEO4J_URI — REST endpoints cannot return 2xx without it.",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ── save_episode shape ────────────────────────────────────────────────


@neo4j_required
def test_save_episode_response_shape(client: TestClient):
    """POST /api/v2/save_episode returns the canonical keys + 200."""
    body = {
        "content": (
            "[FACT] Parity check — jarvis-memory REST /api/v2/save_episode. "
            "Test runner just asserts the response schema, not the persistence."
        ),
        "group_id": "system",
        "episode_type": "fact",
        "importance": 0.5,
        "source": "pytest-parity",
    }
    resp = client.post("/api/v2/save_episode", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert isinstance(payload, dict), type(payload)
    # Canonical v2 save_episode fields.
    assert "saved" in payload
    if payload.get("saved"):
        for key in ("episode_id", "session_id", "room", "hall", "episode_type"):
            assert key in payload, f"missing {key} in {payload!r}"
    else:
        # The only legitimate non-saved path: content filtered as insignificant.
        assert "reason" in payload, payload


# ── scored_search shape ───────────────────────────────────────────────


@neo4j_required
def test_scored_search_response_shape(client: TestClient):
    body = {"query": "jarvis-memory parity test", "limit": 3}
    resp = client.post("/api/v2/scored_search", json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # Required keys.
    assert set(["results", "count", "query", "search_mode", "filters"]).issubset(payload.keys()), payload
    assert isinstance(payload["results"], list)
    assert isinstance(payload["count"], int)


# ── wake_up shape ─────────────────────────────────────────────────────


@neo4j_required
def test_wake_up_response_shape(client: TestClient):
    resp = client.post("/api/v2/wake_up", json={"group_id": "system"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # wake_up returns a dict (formatted identity + context). Shape lives in
    # jarvis_memory.wake_up; here we just assert it hasn't regressed to
    # something non-dict or empty.
    assert isinstance(payload, dict)
    assert payload, "wake_up should return non-empty payload"


# ── no-network parity: app wires up without side effects ──────────────


def test_app_imports_without_contacting_neo4j():
    """Importing jarvis_memory.api must not crash when Neo4j is absent.

    The lifespan context catches startup failures and keeps the app alive.
    This guards against Run 1 accidentally making startup strict.
    """
    from importlib import reload
    import jarvis_memory.api as api_mod

    reload(api_mod)
    # Instantiating TestClient triggers the lifespan 'startup' phase.
    with TestClient(api_mod.app) as _:
        assert api_mod.app is not None


def test_tool_surface_available_via_client():
    """/health endpoint returns a dict with status + version keys."""
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data


# ── Run 2: 4 new /api/v2 endpoints — shape contracts ────────────────


@neo4j_required
def test_orphans_response_shape(client: TestClient):
    """GET /api/v2/orphans returns {count, by_domain}."""
    resp = client.get("/api/v2/orphans")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "count" in payload, payload
    assert "by_domain" in payload, payload
    assert isinstance(payload["count"], int)
    assert isinstance(payload["by_domain"], dict)


@neo4j_required
def test_doctor_response_shape(client: TestClient):
    """GET /api/v2/doctor returns {overall, checks, summary}."""
    resp = client.get("/api/v2/doctor?fast=true")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    for key in ("overall", "checks", "summary"):
        assert key in payload, payload
    assert payload["overall"] in {"PASS", "WARN", "FAIL"}
    assert isinstance(payload["checks"], dict)


@neo4j_required
def test_list_pages_response_shape(client: TestClient):
    """GET /api/v2/pages returns {count, pages}."""
    resp = client.get("/api/v2/pages?limit=5")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "count" in payload
    assert "pages" in payload
    assert isinstance(payload["pages"], list)


@neo4j_required
def test_get_page_404_when_missing(client: TestClient):
    """GET /api/v2/page/{slug} returns 404 for non-existent slug."""
    resp = client.get("/api/v2/page/run-2-nonexistent-canary-slug-abc123")
    # 404 is the contract for missing
    assert resp.status_code == 404, resp.text
