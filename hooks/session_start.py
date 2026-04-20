#!/usr/bin/env python3
"""SessionStart hook — cross-device session continuity.

Upgraded for shared brain architecture. This hook fires at the start of
every Claude session and:

1. Reads group_id from CLAUDE.md
2. Queries Neo4j (on Mac Mini) for the most recent session in this project
3. Pulls the session snapshot (task, completed, next steps, key decisions)
4. Pulls the episode chain (detailed rationale and context)
5. Runs scored_search for broader project memories
6. Creates a new Session node linked via CONTINUES_FROM
7. Injects everything as a structured context block

This ensures zero-loss continuity when switching between MacBook Pro and Mac Mini.
"""
import json
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Add parent to path so we can import jarvis_memory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MEMORY_API_URL = os.getenv("JARVIS_MEMORY_API", "http://localhost:3500")
MAX_MEMORIES = int(os.getenv("JARVIS_HOOK_MAX_MEMORIES", "5"))
MAX_EPISODES = int(os.getenv("JARVIS_HOOK_MAX_EPISODES", "10"))
TIMEOUT_SECONDS = int(os.getenv("JARVIS_HOOK_TIMEOUT", "5"))


def get_group_id() -> str:
    """Determine the current project's group_id from CLAUDE.md or env."""
    group_id = os.getenv("JARVIS_GROUP_ID")
    if group_id:
        return group_id

    claude_md = Path.cwd() / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text()
        for line in content.splitlines():
            if line.strip().startswith("group_id:"):
                return line.split(":", 1)[1].strip()

    return Path.cwd().name


def get_device_id() -> str:
    return os.getenv("JARVIS_DEVICE_ID", "unknown")


def load_previous_session(group_id: str) -> dict:
    """Load the most recent session, its snapshot, and episodes from Neo4j.

    Returns a dict with 'session', 'snapshot', 'episodes' keys.
    Falls back gracefully if Neo4j is unreachable.
    """
    try:
        from jarvis_memory.conversation import SessionManager, EpisodeRecorder, SnapshotManager

        sm = SessionManager()
        er = EpisodeRecorder(driver=sm._driver)
        snm = SnapshotManager(driver=sm._driver)

        # Get latest session for this project
        latest = sm.get_latest_session(group_id)
        if latest is None:
            sm.close()
            return {}

        session_id = latest.get("uuid")

        # Get snapshot
        snapshot = snm.get_latest_snapshot(group_id)

        # Get episodes from the last session
        episodes = er.get_session_episodes(session_id, limit=MAX_EPISODES) if session_id else []

        sm.close()
        return {
            "session": latest,
            "snapshot": snapshot,
            "episodes": episodes,
        }

    except ImportError:
        logger.warning("jarvis_memory not installed, skipping session load")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load previous session (graceful): {e}")
        return {}


def create_new_session(group_id: str, continues_from: str = None, task_summary: str = "") -> str:
    """Create a new session node in Neo4j, linked to the previous one.

    Returns the new session UUID.
    """
    try:
        from jarvis_memory.conversation import SessionManager

        sm = SessionManager()
        result = sm.create_session(
            group_id=group_id,
            device=get_device_id(),
            task_summary=task_summary,
            continues_from=continues_from,
        )
        sm.close()
        return result.get("uuid", "")

    except Exception as e:
        logger.warning(f"Failed to create session (graceful): {e}")
        return ""


def search_memories(query: str, group_id: str) -> list:
    """Search jarvis-memory for broader project context."""
    import urllib.request
    import urllib.parse
    import urllib.error

    params = urllib.parse.urlencode({
        "q": query,
        "group_id": group_id,
        "limit": MAX_MEMORIES,
    })
    url = f"{MEMORY_API_URL}/api/v1/hybrid-search?{params}"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode())
            return data.get("results", data) if isinstance(data, dict) else data
    except Exception as e:
        logger.debug(f"Memory search failed (non-critical): {e}")
        return []


def format_context_block(prev_data: dict, memories: list, new_session_id: str) -> str:
    """Format the full context injection block."""
    lines = []

    # Session snapshot (highest priority)
    snapshot = prev_data.get("snapshot")
    if snapshot:
        from jarvis_memory.conversation import SnapshotManager
        lines.append(SnapshotManager.format_snapshot_for_injection(snapshot))

    # Episode chain from previous session
    episodes = prev_data.get("episodes", [])
    if episodes:
        lines.append("## Recent Context (from previous session)")
        lines.append("")
        for i, ep in enumerate(episodes, 1):
            ep_type = ep.get("episode_type", "fact")
            content = ep.get("content", "")
            lines.append(f"{i}. [{ep_type}] {content}")
        lines.append("")

    # Broader project memories
    if memories:
        lines.append("## Project Memories")
        lines.append("")
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", mem.get("memory", mem.get("name", "")))
            mem_type = mem.get("memory_type", mem.get("type", "fact"))
            lines.append(f"{i}. [{mem_type}] {content}")
        lines.append("")

    # Session metadata
    prev_session = prev_data.get("session")
    if prev_session or new_session_id:
        lines.append("---")
        if prev_session:
            lines.append(f"_Previous session: {prev_session.get('uuid', '?')[:8]} on {prev_session.get('device', '?')}_")
        if new_session_id:
            lines.append(f"_Current session: {new_session_id[:8]} on {get_device_id()}_")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main hook entry point."""
    group_id = get_group_id()
    device = get_device_id()
    logger.info(f"SessionStart hook firing: group={group_id}, device={device}")

    # Read initial context from stdin
    session_context = ""
    if not sys.stdin.isatty():
        try:
            session_context = sys.stdin.read().strip()
        except Exception:
            pass

    # Load previous session from Neo4j
    prev_data = load_previous_session(group_id)

    # Create new session, linked to the previous one
    prev_session_id = None
    if prev_data.get("session"):
        prev_session_id = prev_data["session"].get("uuid")

    new_session_id = create_new_session(
        group_id=group_id,
        continues_from=prev_session_id,
        task_summary=session_context[:200] if session_context else "",
    )

    # Store session ID in env for other hooks to use
    if new_session_id:
        # Write to a temp file that other hooks can read
        session_file = Path("/tmp/jarvis_current_session.json")
        try:
            session_file.write_text(json.dumps({
                "session_id": new_session_id,
                "group_id": group_id,
                "device": device,
            }))
        except Exception:
            pass

    # Search for broader project memories
    search_query = session_context[:200] if session_context else f"project context for {group_id}"
    memories = search_memories(search_query, group_id)

    # Format and inject context
    context_block = format_context_block(prev_data, memories, new_session_id)

    if context_block.strip():
        logger.info(
            f"Injecting context: snapshot={'yes' if prev_data.get('snapshot') else 'no'}, "
            f"episodes={len(prev_data.get('episodes', []))}, memories={len(memories)}"
        )
        print(context_block)
    else:
        logger.info("No context to inject — first session for this project")


if __name__ == "__main__":
    main()
