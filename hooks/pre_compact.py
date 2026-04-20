#!/usr/bin/env python3
"""PreCompact hook — save critical context before context window compaction.

This hook fires when Claude is about to compact its context window (i.e.,
when the conversation is getting too long and Claude needs to summarize/trim).

This is critical because context compaction can lose important details.
The hook captures the current working context and saves it to jarvis-memory
so it can be retrieved later.

Usage in .claude/hooks.json:
{
  "hooks": {
    "PreCompact": [{
      "type": "command",
      "command": "python3 /path/to/hooks/pre_compact.py"
    }]
  }
}

Note: PreCompact hooks are not yet available in Claude's hook system as of
April 2026. This is built in anticipation of the feature. In the meantime,
the Stop hook captures end-of-session context, and STATUS.md files provide
manual continuity.
"""
import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

MEMORY_API_URL = os.getenv("JARVIS_MEMORY_API", "http://localhost:3500")
TIMEOUT_SECONDS = int(os.getenv("JARVIS_HOOK_TIMEOUT", "5"))


def get_group_id() -> str:
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


def save_context_snapshot(content: str, group_id: str) -> bool:
    """Save a context snapshot to jarvis-memory."""
    import urllib.request
    import urllib.error

    session_id = os.getenv("CLAUDE_SESSION_ID", "unknown")

    payload = json.dumps({
        "content": content,
        "agent_id": "claude-session",
        "memory_type": "meta",
        "group_id": group_id,
        "metadata": {
            "source": "pre_compact_hook",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance": 0.9,  # high importance — this is rescue data
        },
    }).encode()

    try:
        req = urllib.request.Request(
            f"{MEMORY_API_URL}/api/v1/add",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            logger.info(f"Context snapshot saved before compaction")
            return True
    except Exception as e:
        logger.warning(f"Failed to save context snapshot: {e}")
        return False


def update_status_md(context: str, group_id: str) -> None:
    """Also write to STATUS.md as a fallback persistence mechanism.

    This ensures continuity even if jarvis-memory is down.
    """
    status_path = Path.cwd() / "STATUS.md"
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        status_content = f"""# Status — {group_id}

_Auto-saved by PreCompact hook at {timestamp}_

## Context Snapshot

{context[:2000]}

## Instructions for Next Session

Read this file first. The previous session was compacted mid-conversation.
The above context snapshot captures the working state at the point of compaction.
Check jarvis-memory for additional context with: `scored_search` using group_id="{group_id}"
"""
        status_path.write_text(status_content)
        logger.info(f"STATUS.md updated at {status_path}")
    except Exception as e:
        logger.warning(f"Failed to update STATUS.md: {e}")


def main():
    group_id = get_group_id()
    logger.info(f"PreCompact hook firing for group: {group_id}")

    # Read context from stdin
    context = ""
    if not sys.stdin.isatty():
        try:
            context = sys.stdin.read().strip()
        except Exception:
            pass

    if not context:
        logger.info("No context provided for pre-compact save")
        return

    # Save to both jarvis-memory and STATUS.md
    save_context_snapshot(context, group_id)
    update_status_md(context, group_id)


if __name__ == "__main__":
    main()
