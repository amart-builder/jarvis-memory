# Jarvis Memory

Shared, persistent memory for an agent fleet (OpenClaw on Mac Mini, Claude Code / Desktop, crons, plugins). Multiple writers share one Neo4j + ChromaDB backend, so a decision recorded from one surface is immediately visible in every other.

## What problem it solves

- Claude forgets between sessions. Compaction erases context.
- Multiple AI systems need to see the same memories, or they diverge.
- Useful memories (decisions, plans, corrections) need structure so they're retrievable weeks later.

Jarvis is a typed-graph store with hybrid RRF search, compiled-truth entity pages, temporal facts, and a dream-cycle compactor — designed to be the canonical shared backend for cross-system continuity.

## Quick start (for users installing this as a service)

**→ See [CLIENT_INSTALL.md](CLIENT_INSTALL.md) for the full walkthrough.** TL;DR:

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
bash scripts/client-install.sh
```

The installer handles venv, `.env`, schema migration, model pre-cache, scheduled compaction, and (optionally) MCP registration with Claude Code + Codex, Claude Code hooks, and the Minions background worker. Re-runnable and idempotent.

## Developer quick start

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                            # dev deps include pytest
cp .env.example .env                               # fill in NEO4J_* and ANTHROPIC_API_KEY
python scripts/migrate_to_v2.py                    # apply the entity-layer schema
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"  # pre-cache embedding model
bash scripts/verify_install.sh                     # 30-second sanity check
python -m jarvis_memory.api                        # REST on :3500
```

`scripts/verify_install.sh` is the authoritative "is this install healthy?" check — 7 gates covering imports, entrypoints, Neo4j connectivity, schema state, ChromaDB, core flows, and the MCP tool surface.

**Note on `setup.sh`:** deprecated — hardcodes a two-machine MBP/Mini assumption that no longer applies. Fresh installs use `scripts/client-install.sh` or the developer steps above. `setup.sh` refuses to run without `JARVIS_LEGACY_SETUP=1`.

## Optional integrations

Each is a single command — install whichever ones fit your setup.

| Integration | One-liner | What it does |
|---|---|---|
| **Claude Code MCP** | `python scripts/register_mcp.py --client claude-code` | Adds Jarvis as an MCP server in `~/.claude/settings.json` so Claude Code can call all 27 tools directly. |
| **Codex CLI MCP** | `python scripts/register_mcp.py --client codex` | Same, for Codex: writes a `[mcp_servers.jarvis-memory]` block into `~/.codex/config.toml`. |
| **Claude Code hooks** | `python install_hooks.py` | SessionStart injects recent project context at session open; PreCompact writes a [HANDOFF] episode before Claude Code auto-compacts. |
| **Minions background worker** | `scripts/generate_launchagents.sh --with-minion-worker` (Mac) or `scripts/generate_systemd_units.sh --with-minion-worker` (Linux) | Starts a durable SQLite-backed job queue (ported from [garrytan/gbrain](https://github.com/garrytan/gbrain)) for deterministic scheduled work. Most installs don't need this on day one. |

Each comes with a matching `--uninstall` flag for clean removal.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Writers                                 │
│                                                              │
│  OpenClaw (Mini)           Claude Code (MBP)    Crons / Hooks│
│       │                           │                    │     │
│       ▼                           ▼                    ▼     │
│  REST API                    MCP server         Python import│
│  localhost:3500              27 tools           direct       │
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

- **Neo4j** is authoritative. Episodes, Entities, Pages, Sessions, Snapshots, typed edges.
- **ChromaDB** is a read-optimized embedding sidecar for semantic search. Can be rebuilt from Neo4j at any time (`EmbeddingStore.rebuild_from_neo4j()`).
- **REST API** on `localhost:3500` exposes v1 + v2 endpoints (used by OpenClaw + generic HTTP clients).
- **MCP server** exposes 27 tools to any MCP client (Claude Code, Claude Desktop). Parity-locked by `tests/test_mcp_parity.py`.
- **Python package** is importable for direct access from scripts, crons, and hooks.

### What's inside (gbrain-import features, merged 2026-04-20)

- **Typed-edge graph + `:Page` entities** (`jarvis_memory.pages`, `graph`, `schema_v2`) — compiled-truth snapshots, typed relationships (`WORKS_AT`, `FOUNDED`, `DECIDED_ON`, …), orphan detection, doctor command.
- **RRF hybrid search + intent classifier + Haiku multi-query expansion** (`jarvis_memory.search.*`) — `scored_search` now fuses Chroma vector results with Neo4j fulltext, routes by intent (`entity/temporal/event/general`), and fans out via Claude Haiku 4-5 (fails-open if `ANTHROPIC_API_KEY` is unset). `JARVIS_SEARCH_LEGACY=1` reverts to the pre-RRF composite scorer for A/B or rollback.
- **Dream-cycle compaction** — daily cron now also runs `fix_citations`, `report_orphans`, `reconcile_stale_edges` subphases. Read-only hygiene; logs findings without mutating data.
- **Minions** (`jarvis_memory.minions.*`, `data/minions.sqlite`) — SQLite-backed deterministic job queue. Shell handler is gated behind `GBRAIN_ALLOW_SHELL_JOBS=1`; default off.
- **OperationContext trust boundary** (`jarvis_memory.operation_context`) — every write tagged `source=mcp|rest|cli`, `remote=True|False` for audit.
- **Retrieval eval harness** (`jarvis_memory.eval`, `tests/eval_data/synthetic_*.jsonl`) — run `python -m jarvis_memory.eval ...` to measure R@k / MRR / nDCG. Baseline floor: R@5 ≥ 0.640, MRR ≥ 0.756, nDCG@10 ≥ 0.683.

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
├── jarvis_memory/                ← core package
│   ├── api.py                    ← FastAPI REST server (v1 compat + /api/v2/*)
│   ├── config.py                 ← env loading
│   ├── conversation.py           ← SessionManager, EpisodeRecorder, SnapshotManager
│   ├── embeddings.py             ← ChromaDB wrapper
│   ├── compaction.py             ← 3-tier dedup + dream-cycle hygiene phases
│   ├── lifecycle.py              ← state machine (active/archived/superseded/...)
│   ├── rooms.py                  ← 70+ keyword room detection + hall mapping
│   ├── scoring.py                ← scored_search (RRF + intent + expansion; legacy composite behind env flag)
│   ├── temporal.py               ← valid_from/valid_to fact management
│   ├── classifier.py             ← 21-type memory classifier
│   ├── wake_up.py                ← token-budgeted session-start context
│   ├── schema_v2.py              ← Run 2 Neo4j schema (:Page, typed edges)
│   ├── pages.py / graph.py       ← Page CRUD + typed-edge extraction
│   ├── orphans.py / doctor.py    ← orphan detection + health reporter
│   ├── operation_context.py      ← OperationContext trust boundary (Run 4)
│   ├── search/                   ← RRF hybrid search (Run 3)
│   │   ├── rrf.py                ← reciprocal-rank fusion
│   │   ├── keyword.py            ← Neo4j fulltext search
│   │   ├── boosts.py             ← compiled-truth + backlink boosts
│   │   ├── intent.py             ← rule-based query intent classifier
│   │   └── expansion.py          ← Haiku-backed multi-query expansion (fails-open)
│   ├── minions/                  ← SQLite job queue (Run 4)
│   │   ├── queue.py              ← MinionQueue with BEGIN IMMEDIATE locking
│   │   ├── worker.py             ← MinionWorker (stdio + launchd)
│   │   └── handlers/             ← built-in handlers (compact, shell-gated)
│   ├── eval.py                   ← retrieval eval harness (R@k / MRR / nDCG)
│   └── backfill_v2.py            ← back-populate legacy data into Run 2 schema
├── mcp_server/
│   └── server.py                 ← MCP tool surface (27 tools, parity-locked)
├── hooks/                        ← event hooks (Claude Code + OpenClaw)
├── launchagents/                 ← macOS launchd plists
│   ├── com.atlas.jarvis-compact-daily.plist
│   ├── com.atlas.jarvis-compact-weekly.plist
│   ├── com.atlas.minion-worker.plist  ← Run 4 worker (not loaded by default)
│   └── INSTALL_COMPACTION_CRON.md
├── scripts/
│   ├── run_compaction.py         ← cron runner (daily + weekly tiers, dream-cycle)
│   ├── migrate_to_v2.py          ← idempotent Run 2 schema migration
│   ├── gen_eval_corpus.py        ← synthetic corpus generator (Opus-4.7 backed)
│   ├── verify_install.sh         ← 30-second post-install sanity check
│   └── backfill_v2.py
├── docs/eval/                    ← before/after eval reports per run
├── tests/                        ← 514 tests
│   ├── search/                   ← Run 3 (RRF, keyword, boosts, intent, expansion)
│   ├── minions/                  ← Run 4 (queue, worker, handlers, shell, audit)
│   └── eval_data/                ← synthetic_corpus_v1 / queries_v1 / qrels_v1
├── data/                         ← SQLite (minions.sqlite), audit logs (gitignored)
├── chromadb/                     ← local vector store (gitignored)
└── audit/                        ← weekly-rotated shell-job audit trail (gitignored)
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

- **Unreleased — client-install path (for v1.0.0)**: `CLIENT_INSTALL.md` walkthrough + `scripts/client-install.sh` one-command installer, portable default paths (`CHROMADB_PATH` → `~/.jarvis-memory/chromadb`, hook logs via `JARVIS_LOG_DIR`), `scripts/generate_launchagents.sh` + `scripts/generate_systemd_units.sh` templated scheduling, `scripts/register_mcp.py` for Claude Code + Codex MCP registration, `scripts/upgrade.sh` for tag-based updates. `install_hooks.py` rewritten to be path-aware.
- **2026-04-20 — feature port from [garrytan/gbrain](https://github.com/garrytan/gbrain) (4 runs, 514 tests, +36 from pre-port)**:
    - *Run 1* — retrieval eval harness (P@k / R@k / MRR / nDCG), brain/memory/session routing rule, parity-lock tests.
    - *Run 2* — `:Page` entity layer with `compiled_truth`/`timeline`, 8 typed edges, `orphans` + `doctor` commands, MCP surface 23 → 27 tools.
    - *Run 3* — RRF hybrid search in `scored_search` (Chroma + Neo4j fulltext + compiled-truth/backlink boosts), rule-based intent classifier, Haiku-4-5 multi-query expansion with prompt-injection defense, dream-cycle compaction (`fix_citations`, `report_orphans`, `reconcile_stale_edges`). Eval delta: R@5 0.640 → 0.843, MRR 0.756 → 0.899, nDCG@10 0.683 → 0.841.
    - *Run 4* — `MinionQueue` (SQLite-backed job queue with `BEGIN IMMEDIATE` claim locking), shell handler gated behind `GBRAIN_ALLOW_SHELL_JOBS`, `OperationContext` trust boundary tagging every write `source=mcp|rest|cli`.
- **2026-04-17**: Rename `yoniclaw` → `legacy-memclawz` (137 nodes). MCP save_episode group_id bug fix. PreCompact + SessionStart hooks for Claude Code. Compaction LaunchAgents. Embeddings backfilled. Room-detection fallback logging.
- **2026-04-07**: Cutover from MemClawz shim to standalone `jarvis_memory.api`. v1 compat layer preserved for OpenClaw hooks.
- **v2 endpoints live** on `/api/v2/*` with room/hall auto-tagging and temporal fact management.
