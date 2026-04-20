#!/usr/bin/env python3
"""Stop hook — save session state for cross-device continuity.

Upgraded for shared brain architecture. When a session ends:

1. Reads the current session ID from temp file (written by session_start hook)
2. Generates a session snapshot from the conversation summary
3. Saves the snapshot to Neo4j with HAS_SNAPSHOT edge
4. Marks the session as completed
5. Updates STATUS.md as a file-level fallback
6. Triggers session compaction
"""
import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUTO_COMPACT = os.getenv("JARVIS_AUTO_COMPACT_SESSION", "true").lower() == "true"


def get_current_session() -> dict:
    """Read the current session info written by session_start hook."""
    session_file = Path("/tmp/jarvis_current_session.json")
    try:
        if session_file.exists():
            return json.loads(session_file.read_text())
    except Exception:
        pass
    return {}


def parse_summary_to_snapshot(summary: str) -> dict:
    """Parse a conversation summary into a structured snapshot."""
    snapshot = {
        "type": "session_snapshot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "",
        "status": "completed",
        "completed": [],
        "in_progress": [],
        "next_steps": [],
        "key_decisions": [],
        "blockers": [],
        "files_modified": [],
        "raw_summary": summary[:2000],
    }

    lines = summary.split("\n")
    for line in lines:
        line_stripped = line.strip().lower()
        if any(kw in line_stripped for kw in ["completed", "finished", "done", "built", "created"]):
            snapshot["completed"].append(line.strip())
        elif any(kw in line_stripped for kw in ["in progress", "working on", "implementing", "halfway"]):
            snapshot["in_progress"].append(line.strip())
        elif any(kw in line_stripped for kw in ["next", "todo", "need to", "should", "will"]):
            snapshot["next_steps"].append(line.strip())
        elif any(kw in line_stripped for kw in ["decided", "chose", "using", "approach", "because"]):
            snapshot["key_decisions"].append(line.strip())
        elif any(kw in line_stripped for kw in ["blocked", "issue", "problem", "can't"]):
            snapshot["blockers"].append(line.strip())

    for line in lines:
        if len(line.strip()) > 10:
            snapshot["task"] = line.strip()[:200]
            break

    return snapshot


def save_session_state(session_info: dict, summary: str):
    """Save snapshot and end the session in Neo4j."""
    try:
        from jarvis_memory.conversation import SessionManager, SnapshotManager

        session_id = session_info.get("session_id")
        if not session_id:
            logger.warning("No session_id found, skipping state save")
            return

        sm = SessionManager()
        snm = SnapshotManager(driver=sm._driver)

        snapshot = parse_summary_to_snapshot(summary)
        snm.save_snapshot(session_id, snapshot)
        sm.end_session(session_id, status="completed")
        sm.close()
        logger.info(f"Session {session_id[:8]} state saved and ended")

    except ImportError:
        logger.warning("jarvis_memory not installed, skipping state save")
    except Exception as e:
        logger.warning(f"Failed to save session state (graceful): {e}")


def update_status_md(session_info: dict, summary: str):
    """Write STATUS.md as a file-level fallback."""
    group_id = session_info.get("group_id", "unknown")
    device = session_info.get("device", "unknown")
    session_id = session_info.get("session_id", "unknown")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    status_content = f"""# Status — {group_id}

_Auto-saved by session stop hook at {timestamp}_
_Session: {session_id[:8] if len(session_id) > 8 else session_id} on {device}_

## Summary

{summary[:2000]}

## Instructions for Next Session

This file is a fallback. The primary session state is in Neo4j.
Use `continue_session` MCP tool or check Jarvis Memory for full context.
"""
    try:
        status_path = Path.cwd() / "STATUS.md"
        status_path.write_text(status_content)
        logger.info(f"STATUS.md updated at {status_path}")
    except Exception as e:
        logger.warning(f"Failed to update STATUS.md: {e}")


def trigger_compaction(session_info: dict):
    """Fire-and-forget session compaction."""
    import urllib.request

    session_id = session_info.get("session_id", "")
    group_id = session_info.get("group_id", "")
    api_url = os.getenv("JARVIS_MEMORY_API", "http://localhost:3500")

    try:
        payload = json.dumps({"session_id": session_id, "group_id": group_id}).encode()
        req = urllib.request.Request(
            f"{api_url}/api/v1/compact/session",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            logger.info("Session compaction triggered")
    except Exception:
        logger.debug("Compaction trigger failed (non-critical)")


def main():
    """Main hook entry point."""
    session_info = get_current_session()
    logger.info(f"Stop hook firing: session={session_info.get('session_id', 'unknown')[:8]}")

    summary = ""
    if not sys.stdin.isatty():
        try:
            summary = sys.stdin.read().strip()
        except Exception:
            pass

    if not summary:
        logger.info("No summary provided, marking session as interrupted")
        if session_info.get("session_id"):
            try:
                from jarvis_memory.conversation import SessionManager
                sm = SessionManager()
                sm.end_session(session_info["session_id"], status="interrupted")
                sm.close()
            except Exception:
                pass
        return

    save_session_state(session_info, summary)
    update_status_md(session_info, summary)

    if AUTO_COMPACT and session_info.get("session_id"):
        trigger_compaction(session_info)

    try:
        Path("/tmp/jarvis_current_session.json").unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
