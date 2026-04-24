"""Contract tests for the handoff module + API endpoints.

Two layers of coverage:
    1. Unit: pure logic (validation, no DB) runs always.
    2. Integration: full round-trip through Neo4j, gated on Neo4j being
       reachable. Skipped gracefully otherwise (matches the existing
       test_api_parity pattern).
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from jarvis_memory import handoff as handoff_module
from jarvis_memory.api import app


# ── Neo4j-gated helper (copied from test_api_parity's pattern) ─────────

def _neo4j_reachable() -> bool:
    try:
        from neo4j import GraphDatabase
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4j")
        drv = GraphDatabase.driver(uri, auth=(user, password))
        try:
            drv.verify_connectivity()
        finally:
            drv.close()
        return True
    except Exception:
        return False


NEO4J_LIVE = _neo4j_reachable()
neo4j_required = pytest.mark.skipif(
    not NEO4J_LIVE, reason="Neo4j not reachable — handoff integration tests skipped."
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def unique_group_id() -> str:
    """A group_id unique to this test run, so parallel / repeated runs
    don't cross-pollute each other."""
    return f"test-contract-{uuid.uuid4().hex[:8]}"


# ═══════════════════════════════════════════════════════════════════════
# Unit: validation (no DB)
# ═══════════════════════════════════════════════════════════════════════


class TestGroupIDValidation:
    """group_id is required on every write path."""

    def test_empty_string_raises(self):
        with pytest.raises(handoff_module.GroupIDRequired):
            handoff_module._validate_group_id("")

    def test_none_raises(self):
        with pytest.raises(handoff_module.GroupIDRequired):
            handoff_module._validate_group_id(None)

    def test_whitespace_only_raises(self):
        with pytest.raises(handoff_module.GroupIDRequired):
            handoff_module._validate_group_id("   ")

    def test_valid_group_id_round_trips(self):
        assert handoff_module._validate_group_id("navi") == "navi"

    def test_whitespace_is_stripped(self):
        assert handoff_module._validate_group_id("  navi  ") == "navi"


class TestRequestModelValidation:
    """Pydantic rejects missing group_id at the REST boundary."""

    def test_session_handoff_requires_group_id(self, client: TestClient):
        resp = client.post(
            "/api/v2/session/handoff",
            json={"task": "test", "next_steps": ["one"]},  # no group_id
        )
        assert resp.status_code == 422  # Pydantic validation error
        # Error mentions the missing field.
        assert "group_id" in resp.text.lower()

    def test_save_state_requires_group_id(self, client: TestClient):
        resp = client.post(
            "/api/v2/session/save_state",
            json={"task": "test"},  # no group_id
        )
        assert resp.status_code == 422
        assert "group_id" in resp.text.lower()

    def test_save_episode_requires_group_id(self, client: TestClient):
        resp = client.post(
            "/api/v2/save_episode",
            json={"content": "test content"},  # no group_id
        )
        assert resp.status_code == 422
        assert "group_id" in resp.text.lower()


class TestLatestHandoffNoGroupID:
    def test_missing_group_id_query_param_errors(self, client: TestClient):
        resp = client.get("/api/v2/handoff/latest")  # missing ?group_id=
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════
# Integration: Neo4j round-trip
# ═══════════════════════════════════════════════════════════════════════


@neo4j_required
class TestSaveHandoffIntegration:
    """save_handoff writes both snapshot + Episode; latest_handoff reads it."""

    def test_handoff_writes_both_snapshot_and_episode(
        self, client: TestClient, unique_group_id: str
    ):
        resp = client.post(
            "/api/v2/session/handoff",
            json={
                "task": "contract-test handoff",
                "group_id": unique_group_id,
                "next_steps": ["verify", "ship"],
                "notes": "contract test writing handoff",
                "source": "pytest",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["handoff_ready"] is True
        assert data["group_id"] == unique_group_id
        assert data["snapshot_id"] is not None
        assert data["episode_id"] is not None
        assert data["idempotent_hit"] is False

    def test_latest_handoff_reads_what_handoff_wrote(
        self, client: TestClient, unique_group_id: str
    ):
        # Write.
        client.post(
            "/api/v2/session/handoff",
            json={
                "task": "read-back test",
                "group_id": unique_group_id,
                "next_steps": ["read", "verify"],
                "source": "pytest",
            },
        )
        # Read.
        resp = client.get(f"/api/v2/handoff/latest?group_id={unique_group_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["group_id"] == unique_group_id
        assert "read-back test" in data["content"]
        assert "read" in data["content"] and "verify" in data["content"]

    def test_latest_handoff_404_when_none(
        self, client: TestClient, unique_group_id: str
    ):
        resp = client.get(f"/api/v2/handoff/latest?group_id={unique_group_id}-empty")
        assert resp.status_code == 404


@neo4j_required
class TestIdempotency:
    """idempotency_key prevents duplicate handoff writes."""

    def test_duplicate_key_is_noop(
        self, client: TestClient, unique_group_id: str
    ):
        key = f"idk-{uuid.uuid4().hex[:8]}"
        body = {
            "task": "idempotency test",
            "group_id": unique_group_id,
            "next_steps": ["one"],
            "idempotency_key": key,
            "source": "pytest",
        }

        first = client.post("/api/v2/session/handoff", json=body).json()
        assert first["idempotent_hit"] is False
        first_episode_id = first["episode_id"]

        second = client.post("/api/v2/session/handoff", json=body).json()
        assert second["idempotent_hit"] is True
        assert second["episode_id"] == first_episode_id


@neo4j_required
class TestListGroups:
    def test_endpoint_returns_groups_with_counts(
        self, client: TestClient, unique_group_id: str
    ):
        # Write one episode so this group_id shows up.
        client.post(
            "/api/v2/save_episode",
            json={
                "content": "listgroups probe",
                "group_id": unique_group_id,
                "source": "pytest",
            },
        )
        resp = client.get("/api/v2/groups")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "groups" in data
        assert "count" in data
        names = {g["group_id"] for g in data["groups"]}
        assert unique_group_id in names


@neo4j_required
class TestV1AddGroupID:
    """v1_add should prefer top-level group_id, fall back to metadata, then warn."""

    def test_top_level_group_id_wins(
        self, client: TestClient, unique_group_id: str
    ):
        resp = client.post(
            "/api/v1/add",
            json={
                "content": "v1 top-level group test",
                "group_id": unique_group_id,
                "metadata": {"group_id": "WRONG"},
            },
        )
        assert resp.status_code == 200, resp.text
        # Round-trip: list_groups should now include unique_group_id.
        groups = client.get("/api/v2/groups").json()["groups"]
        names = {g["group_id"] for g in groups}
        assert unique_group_id in names
        assert "WRONG" not in names  # metadata override shouldn't have won

    def test_metadata_group_id_fallback(
        self, client: TestClient, unique_group_id: str
    ):
        resp = client.post(
            "/api/v1/add",
            json={
                "content": "v1 metadata-only group test",
                "metadata": {"group_id": unique_group_id},
            },
        )
        assert resp.status_code == 200, resp.text
        groups = client.get("/api/v2/groups").json()["groups"]
        names = {g["group_id"] for g in groups}
        assert unique_group_id in names


# ═══════════════════════════════════════════════════════════════════════
# Bearer auth
# ═══════════════════════════════════════════════════════════════════════


class TestBearerAuth:
    """Bearer auth is off by default (no token) and on when token is set."""

    def test_no_token_allows_request(self, client: TestClient, monkeypatch):
        monkeypatch.delenv("JARVIS_API_BEARER_TOKEN", raising=False)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_token_set_but_missing_header_returns_401(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_API_BEARER_TOKEN", "secret-abc")
        # /api/v2/groups is not in the auth-exempt list.
        resp = client.get("/api/v2/groups")
        assert resp.status_code == 401
        assert "bearer" in resp.text.lower()

    def test_token_set_wrong_header_returns_401(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_API_BEARER_TOKEN", "secret-abc")
        resp = client.get(
            "/api/v2/groups", headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

    def test_health_always_exempt_even_with_token(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_API_BEARER_TOKEN", "secret-abc")
        resp = client.get("/health")
        assert resp.status_code == 200  # health is exempt

    @neo4j_required
    def test_token_set_correct_header_returns_200(
        self, client: TestClient, monkeypatch
    ):
        monkeypatch.setenv("JARVIS_API_BEARER_TOKEN", "secret-abc")
        resp = client.get(
            "/api/v2/groups", headers={"Authorization": "Bearer secret-abc"},
        )
        assert resp.status_code == 200
