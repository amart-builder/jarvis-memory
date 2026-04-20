# Jarvis Memory

Shared, persistent memory for Alex's AI systems (OpenClaw on Mac Mini, Claude Code on MacBook Pro, crons, plugins). Both systems read and write to the same Neo4j + ChromaDB backend, so decisions made in one surface are immediately visible to the other.

## What problem it solves

- Claude forgets between sessions. Compaction erases context.
- OpenClaw and Claude need to see the same memories, or they diverge.
- Useful memories (decisions, plans, corrections) need structure so they're retrievable weeks later.

Jarvis is a two-layer store with auto-tagging, semantic search, temporal facts, and compaction — designed to be the canonical shared backend for cross-system continuity.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Writers                                 │
│                                                              │
│  OpenClaw (Mini)           Claude Code (MBP)    Crons / Hooks│
│       │                           │                    │     │
│       ▼                           ▼                    ▼     │
│  REST API                    MCP server         Python import│
│  localhost:3500              23 tools           direct       │
└──────────┬─────────────────────┬─────────────────────┬───────┘
           │                     │                     │
           ▼                     ▼                     ▼
       ┌──────────────────────────────────────────────────┐
       │  jarvis_memory package (Python)                   │
       │  SessionManager · EpisodeRecorder · Compaction   │
       │  EmbeddingStore · LifecycleMgr · Classifier      │
       └────────┬───────────────────────────────┬─────────┘
                │                               │
                ▼                               ▼
          ┌──────────┐                    ┌──────────┐
          │  Neo4j   │   source of        │ ChromaDB │
          │  graph   │   truth            │ vector   │
          │          │                    │ index    │
          └──────────┘                    └──────────┘
       (Mac Mini over Tailscale)       (local sidecar)
```

- **Neo4j** is authoritative. Every Episode, Entity, Session, Snapshot lives here.
- **ChromaDB** is a read-optimized embedding sidecar for semantic search. Can be rebuilt from Neo4j at any time (`EmbeddingStore.rebuild_from_neo4j()`).
- **REST API** on `localhost:3500` (Mini only) exposes v1 + v2 endpoints for OpenClaw hooks.
- **MCP server** exposes 23 tools to Claude Code. Connects to Neo4j over Tailscale.
- **Python package** is importable for direct access from scripts, crons, and hooks.

## Memory model

Every episode has:

| Field | Example | Purpose |
|---|---|---|
| `uuid` | `bb86abbf-af7c...` | Unique id |
| `content` | `[DECISION] Chose Clerk over Auth0...` | Text |
| `group_id` | `navi`, `foundry`, `system` | Project scope (a.k.a. "wing") |
| `episode_type` | `decision`, `plan`, `fact`, `correction`, `completion`, `handoff` | Shape of memory |
| `importance` | `0.5` – `1.0` | Retrieval weight |
| `created_at` | timestamp | Recency ranking |
| `session_id` | UUID | Parent session |

Auto-tagged on write:
- `wing` = `group_id` (project scope)
- `room` = topic (auth / frontend / database / legal / infrastructure / ...) via 70+ keyword list
- `hall` = memory category (decisions / plans / milestones / problems / context)

## Canonical group_ids

See [../brain/MEMORY_PROTOCOL.md](../brain/MEMORY_PROTOCOL.md) §1 for the full table. Summary:

- Money projects: `navi`, `catalyst`, `foundry`, `forge`
- Leverage: `atlas-system`, `supernova`, `combinator`
- Side bets: `hello-world`, `library`, `openclaw-dreaming`, `atlas-web`
- Bots: `gravity`, `sentinel`
- Meta: `jarvis`, `system`
- Historical (read-only): `legacy-memclawz`

**Always** pass `group_id` on writes. Never omit. If a task doesn't map to a project, use `system`.

## Common operations

### Save a decision
```python
# Via Python (from a script)
from jarvis_memory.conversation import SessionManager, EpisodeRecorder
sm = SessionManager()
er = EpisodeRecorder(driver=sm._driver)
er.record_episode(
    session_id=sm.create_session(group_id="navi", device="macbook-pro")["uuid"],
    content="[DECISION] Moved Navi from Claude to Codex. WHY: ... IMPACT: ...",
    episode_type="decision",
    group_id="navi",  # IMPORTANT: pass explicitly
    importance=0.9,
)
sm.close()
```

From Claude Code (MCP):
```
save_episode(content="[DECISION] ...", group_id="navi", episode_type="decision")
```

From OpenClaw (REST):
```bash
curl -X POST http://localhost:3500/api/v2/save_episode \
  -H "Content-Type: application/json" \
  -d '{"content":"[DECISION] ...","group_id":"navi","episode_type":"decision"}'
```

### Load context at session start
From Claude Code (MCP):
```
wake_up(group_id="navi")           # ~600 tokens of identity + essentials
list_sessions(group_id="navi")     # recent sessions across both systems
continue_session(group_id="navi")  # handoff from latest session if < 2h old
```

### Search project memories
```
scored_search(query="auth setup", group_id="navi", room="auth")
```

Filters: `group_id`, `room`, `hall`, `memory_type`, `as_of` (temporal).

### Compaction
Scheduled via LaunchAgents (see `launchagents/`):
- `compact-daily` runs at 3 AM daily (dedup near-duplicates from last 24h)
- `compact-weekly` runs Sundays at 4 AM (consolidate daily digests)

Manual trigger:
```bash
.venv/bin/python scripts/run_compaction.py --tier daily
```

## Hooks

| Hook | Fires on | What it does |
|---|---|---|
| `claude_code_precompact.py` | `/compact` in Claude Code (manual or auto) | Writes `[HANDOFF]` episode + current user-message tail |
| `claude_code_sessionstart.py` | New/resumed Claude Code session | Injects group_id's latest handoff + 5 recent episodes |
| `session_start.py` | OpenClaw session start (REST-based) | Same idea but uses REST endpoints |
| `session_stop.py` | OpenClaw session end | Triggers session compaction |
| `pre_compact.py` | (inactive) older PreCompact variant for when Claude Code didn't support the event |

Register in `~/.claude/settings.json`. See hook file docstrings for the exact config.

## Config

Edit `.env` (mode 600). Defaults sensible for local dev.

| Var | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `bolt://100.102.6.81:7687` | Neo4j on Mini via Tailscale |
| `NEO4J_USER` / `NEO4J_PASSWORD` | — | Auth |
| `ANTHROPIC_API_KEY` | — | LLM classifier for ambiguous episodes (optional) |
| `JARVIS_W_SEMANTIC` / `W_RECENCY` / `W_IMPORTANCE` | 0.5 / 0.3 / 0.2 | Scored search weights |
| `JARVIS_HALF_LIFE_DAYS` | 90 | Recency decay |
| `JARVIS_DEDUP_DAILY` / `DEDUP_WEEKLY` | 0.88 / 0.92 | Compaction similarity thresholds |
| `JARVIS_API_HOST` / `PORT` | `0.0.0.0` / `3500` | REST server (Mini) |
| `JARVIS_GROUP_ID` | empty | Auto-detected from cwd if empty |
| `JARVIS_DEVICE_ID` | hostname | Tagged on session nodes |

## File layout

```
jarvis-memory/
├── README.md                  ← you are here
├── .env                       ← secrets (mode 600, gitignored)
├── .env.example               ← template
├── pyproject.toml
├── setup.sh / setup_venv.sh   ← bootstrap
├── jarvis_memory/             ← core package
│   ├── api.py                 ← FastAPI REST server
│   ├── config.py              ← env loading
│   ├── conversation.py        ← SessionManager, EpisodeRecorder, SnapshotManager
│   ├── embeddings.py          ← ChromaDB wrapper
│   ├── compaction.py          ← 3-tier dedup
│   ├── lifecycle.py           ← state machine (active/archived/superseded/...)
│   ├── rooms.py               ← 70+ keyword room detection + hall mapping
│   ├── scoring.py             ← composite semantic × recency × importance
│   ├── temporal.py            ← valid_from/valid_to fact management
│   ├── classifier.py          ← 21-type memory classifier
│   ├── wake_up.py             ← token-budgeted session-start context
│   └── backfill_v2.py         ← migration helper
├── mcp_server/
│   └── server.py              ← MCP tool surface (23 tools)
├── hooks/                     ← event hooks
│   ├── claude_code_precompact.py
│   ├── claude_code_sessionstart.py
│   ├── pre_compact.py
│   ├── session_start.py
│   └── session_stop.py
├── launchagents/              ← macOS launchd plists
│   ├── com.atlas.jarvis-compact-daily.plist
│   ├── com.atlas.jarvis-compact-weekly.plist
│   └── INSTALL_COMPACTION_CRON.md
├── scripts/
│   ├── run_compaction.py      ← cron runner
│   └── backfill_v2.py
├── chromadb/                  ← local vector store (not in git)
└── tests/
```

## Troubleshooting

**MCP `save_episode` puts everything in `system`**
Cached session in `/tmp/jarvis_current_session.json` is stuck on an old group_id. Fix: `rm /tmp/jarvis_current_session.json` and/or restart the MCP server after the 2026-04-17 code fix in `mcp_server/server.py`.

**ChromaDB out of sync with Neo4j**
Run `.venv/bin/python -c "from jarvis_memory.embeddings import EmbeddingStore; from neo4j import GraphDatabase; import os; d=GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])); print(EmbeddingStore().rebuild_from_neo4j(d))"`

**Compaction counts are 0 forever**
Check if the LaunchAgents are loaded: `launchctl list | grep jarvis-compact`. If not, re-run `launchagents/INSTALL_COMPACTION_CRON.md`.

**Hook seems to do nothing**
Tail the log: `tail -f ~/Atlas/brain/logs/{precompact,sessionstart}-hook.log`. Hooks exit 0 even on error to avoid blocking Claude Code.

**Neo4j unreachable**
Test Tailscale: `nc -zv 100.102.6.81 7687`. If down, Mac Mini is offline or Tailscale is disconnected.

## Key files to know

| If you want to... | Edit this |
|---|---|
| Add a new memory type | `classifier.py` + `rooms.py` HALL_MAP |
| Add a new room / keyword | `rooms.py` ROOM_KEYWORDS |
| Change search scoring | `scoring.py` + `.env` weights |
| Add a new MCP tool | `mcp_server/server.py` |
| Add a new REST endpoint | `api.py` |
| Add a new hook | `hooks/` + register in `~/.claude/settings.json` |

## Recent changes

- **2026-04-17**: Rename `yoniclaw` → `legacy-memclawz` (137 nodes). MCP save_episode group_id bug fix in `mcp_server/server.py`. PreCompact + SessionStart hooks for Claude Code. Compaction LaunchAgents. Embeddings backfilled. Room-detection fallback logging.
- **2026-04-07**: Cutover from MemClawz shim to standalone `jarvis_memory.api`. v1 compat layer preserved for OpenClaw hooks.
- **v2 endpoints live** on `/api/v2/*` with room/hall auto-tagging and temporal fact management.
