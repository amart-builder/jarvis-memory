"""MCP surface parity — the frozen tool surface.

Run 1 baseline: 23 tools. Run 2 (entity layer) adds 4 more: find_orphans,
doctor, get_page, list_pages — bringing the total to 27. This test
prevents accidental drift; update EXPECTED_TOOL_NAMES only through a
spec change.
"""
from __future__ import annotations

from mcp_server.server import JARVIS_TOOLS


# The 27 MCP tools — 23 from Run 1 + 4 new in Run 2 (entity layer).
EXPECTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Run 1 baseline (23)
        "scored_search",
        "classify_memory",
        "lifecycle_status",
        "lifecycle_transition",
        "bulk_archive_stale",
        "compact_session",
        "compact_daily",
        "compact_weekly",
        "compaction_status",
        "memory_stats",
        "supersede_memory",
        "contradict_memory",
        "restore_memory",
        "save_episode",
        "save_state",
        "get_session",
        "list_sessions",
        "continue_session",
        "session_handoff",
        "wake_up",
        "set_fact_validity",
        "fact_timeline",
        "search_rooms",
        # Run 2 additions (4)
        "find_orphans",
        "doctor",
        "get_page",
        "list_pages",
    }
)


def test_tool_count_is_27():
    assert len(JARVIS_TOOLS) == 27, (
        f"expected exactly 27 MCP tools (23 Run 1 + 4 Run 2), "
        f"found {len(JARVIS_TOOLS)}: {[t.name for t in JARVIS_TOOLS]}"
    )


def test_tool_names_exact_match():
    actual = {t.name for t in JARVIS_TOOLS}
    assert actual == EXPECTED_TOOL_NAMES, (
        f"MCP tool surface drift detected.\n"
        f"  added: {sorted(actual - EXPECTED_TOOL_NAMES)}\n"
        f"  removed: {sorted(EXPECTED_TOOL_NAMES - actual)}"
    )


def test_tool_names_unique():
    names = [t.name for t in JARVIS_TOOLS]
    assert len(names) == len(set(names)), (
        f"duplicate MCP tool names detected: "
        f"{[n for n in names if names.count(n) > 1]}"
    )


def test_every_tool_has_description_and_schema():
    """Every Tool object must carry a non-empty description + input schema."""
    for tool in JARVIS_TOOLS:
        assert tool.description, f"{tool.name}: missing description"
        assert tool.inputSchema, f"{tool.name}: missing inputSchema"
        assert isinstance(tool.inputSchema, dict), (
            f"{tool.name}: inputSchema must be a dict, got {type(tool.inputSchema)}"
        )
