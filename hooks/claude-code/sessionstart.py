#!/usr/bin/env python3
"""SessionStart hook for Claude Code — inject project context at session start.

Fires on `SessionStart` event (matcher=startup or resume). Detects the project
group_id from cwd, pulls the last session's state + recent episodes from
Jarvis, and injects a concise context block into the new session.

## Registered in ~/.claude/settings.json

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "<REPO_ROOT>/.venv/bin/python <REPO_ROOT>/hooks/claude-code/sessionstart.py",
            "timeout": 10
          }
        ]
      },
      {
        "matcher": "resume",
        "hooks": [
          {
            "type": "command",
            "command": "<REPO_ROOT>/.venv/bin/python <REPO_ROOT>/hooks/claude-code/sessionstart.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

## Input (stdin, JSON from Claude Code)

```
{
  "session_id": "...",
  "transcript_path": "/path/to/transcript.jsonl",
  "cwd": "/current/working/directory",
  "hook_event_name": "SessionStart",
  "source": "startup" | "resume" | "clear" | "compact"
}
```

## Output (stdout, injected into Claude's session as additional context)

Markdown block with:
- Detected group_id + device
- Whether a recent PreCompact handoff exists (< 72h)
- Up to 5 most recent decisions/plans from this group_id
- Guidance for the agent

## Behavior

- Always exits 0 — never blocks session start
- Keeps injected context under ~2000 chars
- Silent on no-activity projects (first session for group_id)
- All errors logged to ``${JARVIS_LOG_DIR}/sessionstart-hook.log``
  (defaults to ``~/.jarvis-memory/logs/``)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PACKAGE_ROOT))

env_file = PACKAGE_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), val)

LOG_FILE = Path.home() / "Atlas" / "brain" / "logs" / "sessionstart-hook.log"
ATLAS_ROOT = Path.home() / "Atlas"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] sessionstart: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)


def detect_group_id(cwd: str) -> str:
    """Derive group_id from cwd. Same logic as PreCompact hook."""
    if not cwd:
        return "system"
    try:
        cwd_path = Path(cwd).resolve()
    except Exception:
        return "system"

    # Inside a project folder
    try:
        rel = cwd_path.relative_to(ATLAS_ROOT / "brain" / "projects")
        slug = rel.parts[0] if rel.parts else None
        if slug and slug not in ("archive", "_TEMPLATE_STATUS.md"):
            return slug
    except ValueError:
        pass

    # Worktree of brain itself
    try:
        cwd_path.relative_to(ATLAS_ROOT / ".claude" / "worktrees")
        return "system"
    except ValueError:
        pass

    # CLAUDE.md with group_id line
    cmd = cwd_path / "CLAUDE.md"
    if cmd.exists():
        try:
            for line in cmd.read_text().splitlines()[:50]:
                if line.strip().startswith("group_id:"):
                    g = line.split(":", 1)[1].strip()
                    if g and g != "auto-detect":
                        return g
        except Exception:
            pass

    return "system"


def fetch_recent_context(group_id: str) -> dict:
    """Query Neo4j for recent session, handoff, and top episodes in this group_id.

    Returns dict with keys: latest_session, latest_handoff, recent_episodes.
    All graceful — returns {} on failure.
    """
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not (uri and user and password):
            return {}

        driver = GraphDatabase.driver(uri, auth=(user, password))
        result: dict = {}
        try:
            with driver.session() as s:
                # Latest session
                row = s.run(
                    """
                    MATCH (sess:Session {group_id: $gid})
                    RETURN sess.uuid AS uuid, sess.device AS device,
                           sess.created_at AS created, sess.task_summary AS task
                    ORDER BY sess.created_at DESC LIMIT 1
                    """,
                    gid=group_id,
                ).single()
                if row:
                    result["latest_session"] = dict(row)

                # Most recent HANDOFF episode (<72h)
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
                row = s.run(
                    """
                    MATCH (e:Episode {group_id: $gid})
                    WHERE e.memory_type = 'handoff'
                      AND e.created_at >= datetime($cutoff)
                    RETURN e.uuid AS uuid, e.content AS content, e.created_at AS created
                    ORDER BY e.created_at DESC LIMIT 1
                    """,
                    gid=group_id, cutoff=cutoff,
                ).single()
                if row:
                    result["latest_handoff"] = dict(row)

                # Top 5 most recent non-handoff episodes
                rows = s.run(
                    """
                    MATCH (e:Episode {group_id: $gid})
                    WHERE coalesce(e.memory_type, '') <> 'handoff'
                    RETURN e.uuid AS uuid, e.content AS content,
                           coalesce(e.episode_type, 'fact') AS type,
                           e.created_at AS created
                    ORDER BY e.created_at DESC LIMIT 5
                    """,
                    gid=group_id,
                ).data()
                result["recent_episodes"] = rows
        finally:
            driver.close()
        return result

    except Exception as e:
        log.warning("Neo4j query failed: %s", e)
        log.debug(traceback.format_exc())
        return {}


def format_context_block(group_id: str, source: str, ctx: dict) -> str:
    """Format a concise markdown block for Claude to ingest."""
    lines: list[str] = []
    lines.append(f"## Jarvis Context — group_id: `{group_id}`")

    if not ctx:
        lines.append("")
        lines.append(f"_No recent activity for this group_id. Fresh session (source: {source})._")
        lines.append("")
        lines.append(f"If this session produces meaningful decisions, save them via "
                     f"`save_episode(group_id=\"{group_id}\", ...)` so future sessions can pick up.")
        return "\n".join(lines)

    # Recent handoff (highest priority)
    handoff = ctx.get("latest_handoff")
    if handoff:
        lines.append("")
        lines.append("### 🔴 Recent PreCompact Handoff (< 72h ago)")
        lines.append("_A previous session compacted — pick up from this handoff:_")
        lines.append("")
        preview = (handoff.get("content") or "")[:800]
        lines.append(preview)

    # Latest session info
    sess = ctx.get("latest_session")
    if sess:
        lines.append("")
        created = str(sess.get("created", ""))[:19]
        device = sess.get("device") or "?"
        task = sess.get("task") or "(no task summary)"
        lines.append(f"_Last session: {sess.get('uuid', '?')[:8]} on **{device}** at {created} — {task}_")

    # Recent episodes (decisions/plans/facts)
    eps = ctx.get("recent_episodes", [])
    if eps:
        lines.append("")
        lines.append(f"### Recent memories ({len(eps)} most recent)")
        for i, e in enumerate(eps, 1):
            et = e.get("type", "fact")
            content = (e.get("content") or "").strip().split("\n")[0][:180]
            lines.append(f"{i}. **[{et}]** {content}")

    lines.append("")
    lines.append(
        f"_Full context: `scored_search(group_id=\"{group_id}\", query=...)`. "
        f"Handoffs: `memory_type=handoff`. Write new decisions: "
        f"`save_episode(group_id=\"{group_id}\", ...)`._"
    )
    return "\n".join(lines)


def main() -> int:
    try:
        envelope = json.loads(sys.stdin.read() or "{}")
    except Exception:
        envelope = {}

    cwd = envelope.get("cwd", os.getcwd())
    source = envelope.get("source", "unknown")
    session_id = envelope.get("session_id", "unknown")

    group_id = detect_group_id(cwd)
    log.info("SessionStart firing: source=%s group_id=%s cwd=%s session=%s",
             source, group_id, cwd, session_id[:12])

    ctx = fetch_recent_context(group_id)
    block = format_context_block(group_id, source, ctx)

    # Emit context block via JSON for structured injection
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": block,
        }
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        log.error("Hook crashed: %s", e)
        log.debug(traceback.format_exc())
        rc = 0
    sys.exit(rc)
