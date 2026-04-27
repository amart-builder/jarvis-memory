"""``agent_id`` scoping (B2).

Validates the writer-identity dimension added to Episode nodes:
  * ``record_episode`` accepts ``agent_id`` and derives a per-surface
    default from a passed ``OperationContext``.
  * ``scored_search`` exposes an ``agent_id`` filter that matches
    strict equality.
  * Pre-B2 episodes (no ``agent_id`` property) don't match a specific
    filter — they're invisible to ``agent_id="claude-code"`` queries
    but reachable when no filter is passed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from jarvis_memory.scoring import _apply_filters, scored_search


def _make_record(uuid: str, **kwargs: Any) -> dict:
    return {"uuid": uuid, **kwargs}


# ── _apply_filters honors agent_id ──────────────────────────────────────


def test_apply_filters_passes_agent_id_through():
    records = [
        _make_record("a", agent_id="claude-code"),
        _make_record("b", agent_id="openclaw"),
        _make_record("c", agent_id="cron"),
    ]
    out = _apply_filters(
        records,
        group_id=None,
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
        agent_id="claude-code",
    )
    assert [r["uuid"] for r in out] == ["a"]


def test_apply_filters_excludes_legacy_records_without_agent_id():
    """Pre-B2 episodes have no agent_id. A specific filter must NOT
    accidentally include them — that would violate "show me writes
    from system X" semantics."""
    records = [
        _make_record("legacy"),  # no agent_id at all
        _make_record("modern", agent_id="claude-code"),
    ]
    out = _apply_filters(
        records,
        group_id=None,
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
        agent_id="claude-code",
    )
    assert [r["uuid"] for r in out] == ["modern"]


def test_apply_filters_no_agent_id_filter_returns_all():
    """When the caller doesn't pass agent_id, every record is included
    regardless of writer (or its absence)."""
    records = [
        _make_record("legacy"),
        _make_record("claude", agent_id="claude-code"),
        _make_record("openclaw", agent_id="openclaw"),
    ]
    out = _apply_filters(
        records,
        group_id=None,
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
    )
    assert {r["uuid"] for r in out} == {"legacy", "claude", "openclaw"}


def test_apply_filters_compose_agent_id_with_other_filters():
    records = [
        _make_record("a", agent_id="claude-code", group_id="navi"),
        _make_record("b", agent_id="claude-code", group_id="catalyst"),
        _make_record("c", agent_id="openclaw", group_id="navi"),
    ]
    out = _apply_filters(
        records,
        group_id="navi",
        room=None,
        hall=None,
        memory_type=None,
        as_of=None,
        agent_id="claude-code",
    )
    assert [r["uuid"] for r in out] == ["a"]


# ── End-to-end through scored_search ───────────────────────────────────


def test_scored_search_accepts_agent_id_kwarg_without_raising():
    """Passing the new kwarg must not break callers — empty rankings
    fall through cleanly via the legacy path."""
    out = scored_search(
        "test",
        agent_id="claude-code",
        vector_search_fn=lambda q, n: [],
    )
    assert isinstance(out, list)


def test_scored_search_legacy_path_filters_by_agent_id(monkeypatch):
    """``JARVIS_SEARCH_LEGACY=1`` path must honor the same filter."""
    monkeypatch.setenv("JARVIS_SEARCH_LEGACY", "1")

    fake_hits = [
        {
            "id": "claude-1",
            "uuid": "claude-1",
            "similarity": 0.9,
            "metadata": {
                "agent_id": "claude-code",
                "group_id": "navi",
                "memory_type": "decision",
            },
        },
        {
            "id": "openclaw-1",
            "uuid": "openclaw-1",
            "similarity": 0.85,
            "metadata": {
                "agent_id": "openclaw",
                "group_id": "navi",
                "memory_type": "decision",
            },
        },
    ]

    out = scored_search(
        "decision",
        agent_id="claude-code",
        vector_search_fn=lambda q, n: fake_hits,
    )
    uuids = [r.get("uuid") for r in out]
    assert "claude-1" in uuids
    assert "openclaw-1" not in uuids


# ── record_episode default-derivation from ctx ─────────────────────────


def test_record_episode_default_agent_id_is_unknown_without_ctx():
    """No agent_id, no ctx → "unknown" stamped on the node.

    We don't run a real Neo4j call — patch the session layer and
    capture the params to record_episode's CREATE statement."""
    from jarvis_memory.conversation import EpisodeRecorder

    captured: dict[str, Any] = {}

    class _FakeRunResult:
        def single(self):
            return {"gid": "navi"}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            # First call: group_id lookup. Subsequent: episode + edge writes.
            if "RETURN s.group_id" in query:
                return _FakeRunResult()
            if "CREATE (e:Episode" in query:
                captured["episode_params"] = params
                return MagicMock()
            return MagicMock()

    driver = MagicMock()
    driver.session.return_value = _FakeSession()

    # Patch out page-maintenance to keep this test focused on the write.
    with patch.object(EpisodeRecorder, "_maintain_pages_for_episode", return_value=None):
        er = EpisodeRecorder(driver=driver)
        er.record_episode(
            session_id="sess-1",
            content="[DECISION] testing default agent_id resolution with sufficient length to pass the significance filter",
            episode_type="decision",
            group_id="navi",
        )

    assert captured.get("episode_params"), "Episode CREATE was not executed"
    assert captured["episode_params"]["agent_id"] == "unknown"


def test_record_episode_default_agent_id_derived_from_ctx_source():
    """ctx.source='mcp' → agent_id='claude-code' (default MCP client)."""
    from jarvis_memory.conversation import EpisodeRecorder
    from jarvis_memory.operation_context import OperationContext

    captured: dict[str, Any] = {}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            if "CREATE (e:Episode" in query:
                captured["episode_params"] = params
            return MagicMock(single=lambda: None)

    driver = MagicMock()
    driver.session.return_value = _FakeSession()

    with patch.object(EpisodeRecorder, "_maintain_pages_for_episode", return_value=None):
        er = EpisodeRecorder(driver=driver)
        er.record_episode(
            session_id="sess-1",
            content="[DECISION] mcp default test — this content needs to be long enough to pass should_record threshold",
            episode_type="decision",
            group_id="navi",
            ctx=OperationContext.for_mcp(),
        )

    assert captured["episode_params"]["agent_id"] == "claude-code"


def test_record_episode_explicit_agent_id_wins_over_ctx():
    """An explicit ``agent_id`` argument overrides any ctx-derived default."""
    from jarvis_memory.conversation import EpisodeRecorder
    from jarvis_memory.operation_context import OperationContext

    captured: dict[str, Any] = {}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            if "CREATE (e:Episode" in query:
                captured["episode_params"] = params
            return MagicMock(single=lambda: None)

    driver = MagicMock()
    driver.session.return_value = _FakeSession()

    with patch.object(EpisodeRecorder, "_maintain_pages_for_episode", return_value=None):
        er = EpisodeRecorder(driver=driver)
        er.record_episode(
            session_id="sess-1",
            content="[DECISION] explicit override — testing that explicit agent_id arg wins over the ctx-derived default",
            episode_type="decision",
            group_id="navi",
            agent_id="cron",
            ctx=OperationContext.for_mcp(),  # would default to claude-code
        )

    assert captured["episode_params"]["agent_id"] == "cron"


def test_record_episode_writes_t_created_alongside_created_at():
    """B2 also closes a small bi-temporal gap: new Episodes get
    ``t_created`` initialized at write time, so post-A3.1 writes
    don't need the migration to run after each insert."""
    from jarvis_memory.conversation import EpisodeRecorder

    captured: dict[str, Any] = {}

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def run(self, query, **params):
            if "CREATE (e:Episode" in query:
                captured["query"] = query
                captured["params"] = params
            return MagicMock(single=lambda: None)

    driver = MagicMock()
    driver.session.return_value = _FakeSession()

    with patch.object(EpisodeRecorder, "_maintain_pages_for_episode", return_value=None):
        er = EpisodeRecorder(driver=driver)
        er.record_episode(
            session_id="sess-1",
            content="[FACT] t_created init test — verify new Episode writes carry the bi-temporal init from B2",
            episode_type="fact",
            group_id="navi",
        )

    query = captured["query"]
    assert "t_created" in query, "new Episode writes should set t_created"
    assert "created_at" in query
