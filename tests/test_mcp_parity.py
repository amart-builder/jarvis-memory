"""MCP surface parity — the 23 tools must remain the 23 tools.

Spec §10. Run 1 must NOT add, remove, rename, or reshape any MCP tool.
We import the canonical list that mcp_server/server.py exposes and
compare it against the frozen set captured here at Run 1 freeze time.
"""
from __future__ import annotations

from mcp_server.server import JARVIS_TOOLS


# The 23 MCP tools — frozen as of the start of Run 1 (2026-04-20).
# If Sentinel / Verdict ever see this set drift, investigate before
# merging. Do NOT update this constant without a spec change.
EXPECTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
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
    }
)


def test_tool_count_is_23():
    assert len(JARVIS_TOOLS) == 23, (
        f"expected exactly 23 MCP tools, found {len(JARVIS_TOOLS)}: "
        f"{[t.name for t in JARVIS_TOOLS]}"
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
