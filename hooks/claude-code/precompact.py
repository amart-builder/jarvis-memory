#!/usr/bin/env python3
"""PreCompact hook for Claude Code — auto-handoff before /compact.

Fires BEFORE Claude Code compacts a conversation (manual /compact or auto on
context limit). Writes a structured [HANDOFF] episode to Jarvis so the next
session can pick up where this one left off via continue_session().

## Registered in ~/.claude/settings.json

```json
{
  "hooks": {
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "<REPO_ROOT>/.venv/bin/python <REPO_ROOT>/hooks/claude-code/precompact.py",
            "timeout": 15
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
  "hook_event_name": "PreCompact",
  "compaction_trigger": "manual" | "auto"
}
```

## Behavior

- Detects `group_id` from cwd (brain/projects/{name}/ → {name}), CLAUDE.md, or fallback to 'system'
- Parses the last ~10 user messages from the transcript for a concise summary
- Writes [HANDOFF] episode via direct Neo4j (reliable from both MBP and Mini)
- On any failure, writes a fallback JSON line to ``${JARVIS_LOG_DIR}/precompact-fallback.log``
  (defaults to ``~/.jarvis-memory/logs/``)
- ALWAYS exits 0 — never blocks compaction
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add jarvis-memory package path
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent  # jarvis-memory/
sys.path.insert(0, str(PACKAGE_ROOT))

# Load .env
env_file = PACKAGE_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key.strip(), val)

LOG_FILE = Path.home() / "Atlas" / "brain" / "logs" / "precompact-hook.log"
FALLBACK_LOG = Path.home() / "Atlas" / "brain" / "logs" / "precompact-fallback.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] precompact: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger(__name__)

ATLAS_ROOT = Path.home() / "Atlas"


def detect_group_id(cwd: str) -> str:
    """Derive group_id from cwd, falling back intelligently."""
    if not cwd:
        return "system"

    try:
        cwd_path = Path(cwd).resolve()
    except Exception:
        return "system"

    # Case 1: inside a project folder — brain/projects/{name}/...
    try:
        rel = cwd_path.relative_to(ATLAS_ROOT / "brain" / "projects")
        slug = rel.parts[0] if rel.parts else None
        if slug and slug not in ("archive", "_TEMPLATE_STATUS.md"):
            return slug
    except ValueError:
        pass

    # Case 2: inside a worktree — .claude/worktrees/{name}/... (brain worktree)
    try:
        rel = cwd_path.relative_to(ATLAS_ROOT / ".claude" / "worktrees")
        # Worktree of brain itself — this is typically meta work
        return "system"
    except ValueError:
        pass

    # Case 3: CLAUDE.md in cwd with a group_id line
    claude_md = cwd_path / "CLAUDE.md"
    if claude_md.exists():
        try:
            for line in claude_md.read_text().splitlines()[:50]:
                if line.strip().startswith("group_id:"):
                    gid = line.split(":", 1)[1].strip()
                    if gid and gid != "auto-detect":
                        return gid
        except Exception:
            pass

    # Case 4: cwd is Atlas root itself
    if cwd_path == ATLAS_ROOT:
        return "system"

    # Case 5: last resort — use the deepest dir name
    return "system"


def extract_transcript_summary(transcript_path: str, max_user_messages: int = 10) -> tuple[str, int]:
    """Pull the last N user messages from the transcript JSONL.

    Returns (summary_text, total_message_count).
    Each transcript line is a JSON object with at least {role, content}.
    """
    if not transcript_path:
        return "(no transcript_path provided)", 0

    p = Path(transcript_path)
    if not p.exists():
        return f"(transcript file missing: {transcript_path})", 0

    try:
        lines = p.read_text().splitlines()
    except Exception as e:
        return f"(could not read transcript: {e})", 0

    user_messages: list[str] = []
    total = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Support both flat {role, content} and nested {message: {role, content}} shapes.
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        role = msg.get("role") or msg.get("type")
        if role != "user":
            continue

        content = msg.get("content")
        text = None
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Content blocks — pull the text parts
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if t:
                        parts.append(t)
            text = "\n".join(parts) if parts else None

        if text:
            # Truncate each to ~250 chars so a long prompt doesn't dominate
            text = text.strip().replace("\n", " ")
            if len(text) > 250:
                text = text[:247] + "..."
            user_messages.append(text)

    # Keep last N
    tail = user_messages[-max_user_messages:]
    if not tail:
        return "(no user messages found in transcript)", total

    summary = "\n".join(f"  - {m}" for m in tail)
    return summary, total


def save_handoff_neo4j(content: str, group_id: str, session_id: str) -> str | None:
    """Write a [HANDOFF] episode directly to Neo4j. Returns UUID on success."""
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USER")
        password = os.environ.get("NEO4J_PASSWORD")
        if not (uri and user and password):
            log.warning("Neo4j credentials missing from env")
            return None

        node_uuid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as s:
                s.run(
                    """
                    CREATE (e:Episode {
                        uuid: $uuid,
                        group_id: $group_id,
                        content: $content,
                        name: $name,
                        memory_type: 'handoff',
                        episode_type: 'outcome',
                        importance: 0.9,
                        access_count: 0,
                        created_at: datetime($created_at),
                        source: 'claude_code_precompact_hook',
                        session_id: $session_id
                    })
                    """,
                    uuid=node_uuid,
                    group_id=group_id,
                    content=content,
                    name=f"precompact-handoff-{now[:19]}",
                    created_at=now,
                    session_id=session_id,
                )
            return node_uuid
        finally:
            driver.close()
    except Exception as e:
        log.warning("Neo4j write failed: %s", e)
        log.debug(traceback.format_exc())
        return None


def save_fallback(payload: dict) -> None:
    """Last-resort fallback: append the payload to a local JSONL file."""
    try:
        FALLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FALLBACK_LOG.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception as e:
        log.error("Fallback log write failed: %s", e)


def main() -> int:
    # Read Claude Code's JSON envelope from stdin
    try:
        stdin_text = sys.stdin.read()
        envelope = json.loads(stdin_text) if stdin_text.strip() else {}
    except Exception as e:
        log.warning("Failed to parse stdin envelope: %s", e)
        envelope = {}

    session_id = envelope.get("session_id", "unknown")
    transcript_path = envelope.get("transcript_path", "")
    cwd = envelope.get("cwd", os.getcwd())
    trigger = envelope.get("compaction_trigger", "unknown")

    log.info(
        "PreCompact firing: session=%s trigger=%s cwd=%s",
        session_id[:12], trigger, cwd,
    )

    group_id = detect_group_id(cwd)
    summary, msg_count = extract_transcript_summary(transcript_path)
    now = datetime.now(timezone.utc).isoformat()

    content = (
        f"[HANDOFF] Claude Code session compacted ({trigger})\n"
        f"WHEN: {now}\n"
        f"SESSION: {session_id}\n"
        f"TRIGGER: {trigger}\n"
        f"CWD: {cwd}\n"
        f"TRANSCRIPT: {transcript_path} ({msg_count} total messages)\n"
        f"GROUP_ID: {group_id}\n"
        f"\n"
        f"LAST USER MESSAGES:\n"
        f"{summary}\n"
        f"\n"
        f"NEXT SESSION: call continue_session(group_id=\"{group_id}\") to pick up. "
        f"Also read the transcript at TRANSCRIPT path for full context if needed."
    )

    uuid_written = save_handoff_neo4j(content, group_id, session_id)
    if uuid_written:
        log.info("Handoff written: uuid=%s group_id=%s", uuid_written[:8], group_id)
    else:
        log.warning("Neo4j write failed — saving fallback")
        save_fallback({
            "timestamp": now,
            "session_id": session_id,
            "cwd": cwd,
            "trigger": trigger,
            "group_id": group_id,
            "content": content,
        })

    # ALWAYS exit 0. Never block compaction.
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        log.error("Hook crashed: %s", e)
        log.debug(traceback.format_exc())
        rc = 0  # don't block on crash
    sys.exit(rc)
