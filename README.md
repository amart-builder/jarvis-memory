# Jarvis Memory

Persistent memory for AI agents.

Jarvis Memory gives Claude Code, Codex, OpenClaw, crons, and custom agents one shared memory backend: Neo4j for the authoritative graph, ChromaDB for semantic recall, REST for app/runtime integrations, and MCP for coding-agent access.

This is the memory system Alex Martin uses in his own agent stack. On April 30, 2026, Jarvis Memory scored **488 / 500 = 97.60%** raw correct on LongMemEval. LongMemEval is not a perfect test of true agent memory, but it is a useful signal: Jarvis is built for the harder real-world version, where agents need to remember decisions, handoffs, people, projects, corrections, and what changed over time.

Proof: [reports/longmemeval-proof-20260430.md](reports/longmemeval-proof-20260430.md)

## What It Does

- Saves durable memories as typed episodes: decisions, facts, plans, corrections, outcomes, preferences, handoffs.
- Scopes memory by `group_id`, so one project or client does not leak into another.
- Builds an entity graph in Neo4j with `:Page` nodes, typed edges, and evidence timelines.
- Retrieves with hybrid search: vector similarity, Neo4j full-text, query expansion, graph boosts, and optional cross-encoder reranking.
- Handles time: event-time validity (`as_of`) and ingestion-time validity (`seen_as_of`).
- Keeps agent sessions continuous with `wake_up`, `latest_handoff`, `continue_session`, and `session_handoff`.
- Exposes the same core system through REST and a 29-tool MCP server.

## Mental Model

```
Agents and apps
  Claude Code / Codex / OpenClaw / scripts / crons
        |
        |  MCP tools or REST calls
        v
Jarvis Memory
  session manager
  episode recorder
  classifier
  graph builder
  retrieval pipeline
        |
        +--> Neo4j: source of truth, sessions, episodes, Pages, typed edges
        |
        +--> ChromaDB: rebuildable vector index for semantic search
```

Neo4j is the memory. ChromaDB is a fast recall sidecar. If ChromaDB disappears, the graph still holds the source of truth.

## The Context Graph

Jarvis Memory does use a context graph:

- `Episode` nodes store the raw memory text.
- `Session` nodes group episodes and chain work across devices.
- `Page` nodes represent entities, projects, people, companies, systems, and concepts.
- `EVIDENCED_BY` edges connect Pages back to the episodes that support them.
- Typed edges such as `WORKS_AT`, `FOUNDED`, `ADVISES`, `DECIDED_ON`, `MENTIONS`, and `REFERS_TO` connect Pages to each other.
- Multi-hop search can run Personalized PageRank over that graph for relationship-style queries.

The honest caveat: graph extraction is conservative, and not every Page has a rich `compiled_truth` summary yet. The graph exists and is used, but the system is intentionally evidence-first instead of pretending every entity summary is perfect.

## Quick Start

For a local single-machine install:

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
bash scripts/client-install.sh
```

For an agent-led install, give your agent this:

```text
Install Jarvis Memory for my agent system. First read AGENT_RUNBOOK.md.
Audit my OS, agent surfaces, MCP configs, Neo4j situation, security posture,
and where I want memory to run. Then propose the smallest safe topology,
install it, verify it, and show me the exact smoke-test output.
```

Agent runbook: [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md)

Full client install guide: [CLIENT_INSTALL.md](CLIENT_INSTALL.md)

## Requirements

- Python 3.10+
- Neo4j 5.x
- Git and curl
- Optional but recommended: `ANTHROPIC_API_KEY` for query expansion and ambiguous classification
- Optional: Claude Code, Codex, OpenClaw, or any MCP-speaking agent

## Start The REST API

```bash
source .venv/bin/activate
python -m jarvis_memory.api
```

Default REST address: `http://localhost:3500`

If you bind beyond loopback, set `JARVIS_API_BEARER_TOKEN` and preferably `JARVIS_REQUIRE_AUTH=1`. A writable memory server should not be exposed without auth.

## Smoke Test

```bash
curl -X POST http://localhost:3500/api/v2/save_episode \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "demo",
    "content": "[FACT] Jarvis Memory smoke test works. WHY: install verification.",
    "episode_type": "fact"
  }'

curl -X POST http://localhost:3500/api/v2/scored_search \
  -H "Content-Type: application/json" \
  -d '{"group_id":"demo","query":"smoke test","limit":5}'
```

Run the local install verifier:

```bash
bash scripts/verify_install.sh
```

## MCP Setup

Jarvis ships a 29-tool MCP server.

Claude Code:

```bash
python scripts/register_mcp.py --client claude-code
```

Codex:

```bash
python scripts/register_mcp.py --client codex
```

Then restart the client so it reloads MCP config.

Important tools:

- `save_episode`: write durable memory.
- `scored_search`: search by meaning, recency, importance, graph hints, and filters.
- `wake_up`: load a token-budgeted project context block at session start.
- `session_handoff`: save a handoff before switching sessions or compacting context.
- `latest_handoff` / `continue_session`: resume from another agent or device.
- `get_page` / `list_pages`: inspect entity graph Pages.
- `doctor`: check graph health.

## Choosing `group_id`

`group_id` is the memory boundary. Use one stable ID per project, client, product, or durable workstream.

Good examples:

- `acme-client`
- `personal-chief-of-staff`
- `my-saas-app`
- `research`
- `system`

Rules:

- Always pass `group_id` on writes.
- Use short lowercase slugs.
- Do not put unrelated clients or projects in the same `group_id`.
- Use `system` only for admin/infrastructure memory.

## What To Save

Save durable information the agent should remember weeks later:

- Decisions and why they were made.
- User preferences and constraints.
- Project status, blockers, and next steps.
- Corrections: "that old fact is wrong now."
- Handoffs before context compaction or switching agents.
- Stable relationships between people, companies, projects, and systems.

Do not save:

- Scratch reasoning.
- Temporary "in this chat" context.
- Secrets.
- Huge raw logs.
- Anything the user would not want resurfacing later.

Recommended memory shape:

```text
[DECISION] Chose Clerk over Auth0 for signup.
WHY: Faster implementation and better current docs.
IMPACT: Future auth work should assume Clerk unless this is superseded.
```

## Search And Time

Basic recall:

```text
scored_search(query="what did we decide about auth?", group_id="my-saas-app")
```

Topic-filtered recall:

```text
scored_search(query="signup flow", group_id="my-saas-app", room="auth")
```

Temporal recall:

```text
scored_search(query="who owned billing?", group_id="my-saas-app", as_of="2026-04-01")
```

"What did we believe then?" recall:

```text
scored_search(query="deployment status", group_id="my-saas-app", seen_as_of="2026-04-01")
```

## Production Notes

- Keep Neo4j backed up. It is the source of truth.
- Treat ChromaDB as rebuildable cache.
- Keep the REST API on loopback unless you set bearer auth.
- Set `JARVIS_RERANK_DEVICE=cpu` on Apple Silicon if MPS is unstable.
- Set `JARVIS_SEMANTIC_DEDUP=0` or lower `JARVIS_SEMANTIC_DEDUP_TIMEOUT` on small machines if compaction is too expensive.
- The stop hook will not write canonical `STATUS.md` files unless you deliberately configure `JARVIS_STOP_STATUS_FALLBACK_PATH`, and it refuses `STATUS.md` by name.

## File Layout

```text
jarvis-memory/
|-- jarvis_memory/             core package
|   |-- api.py                 FastAPI REST server
|   |-- conversation.py        sessions, episodes, snapshots
|   |-- graph.py               typed-edge extraction
|   |-- pages.py               entity Page CRUD + timelines
|   |-- scoring.py             hybrid retrieval entry point
|   |-- search/                RRF, keyword, expansion, PPR, rerank
|   |-- temporal.py            validity windows
|   `-- wake_up.py             session-start context loader
|-- mcp_server/server.py       29-tool MCP server
|-- hooks/                     Claude Code and OpenClaw hooks
|-- scripts/                   install, migration, compaction, MCP registration
|-- launchagents/              macOS scheduled jobs
|-- systemd/                   Linux scheduled jobs
|-- tests/                     unit and contract tests
|-- reports/                   benchmark proof reports
|-- CLIENT_INSTALL.md          human install guide
`-- AGENT_RUNBOOK.md           agent-led install and ops guide
```

## Benchmark

Jarvis Memory scored **97.60%** on LongMemEval, measured as raw correct out of 500:

- Correct: `488`
- Total: `500`
- Answerer: GPT-4.1
- Judge: GPT-4o
- Run ID: `phase12_full500_chk_20260430-0419`

Proof pack:

- [reports/longmemeval-proof-20260430.md](reports/longmemeval-proof-20260430.md)
- [reports/proof/phase12_full500_chk_20260430-0419.hashes.txt](reports/proof/phase12_full500_chk_20260430-0419.hashes.txt)
- [reports/proof/phase12_full500_chk_20260430-0419_merged500.summary.json](reports/proof/phase12_full500_chk_20260430-0419_merged500.summary.json)

LongMemEval is a useful signal, not the whole product. The real goal is not leaderboard chasing; it is making agents act like they have a durable working memory.

## License

MIT.
