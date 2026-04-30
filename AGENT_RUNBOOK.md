# Agent Runbook: Install Jarvis Memory For A User

This guide is written for Claude Code, Codex, OpenClaw, or another agent installing Jarvis Memory on behalf of a human.

Your job is not to copy Alex's setup. Your job is to understand the user's system, choose the smallest safe topology, install Jarvis Memory, verify it, and leave the user with a working memory loop.

## Operating Rule

Audit first, then install.

Do not assume:

- The user is on macOS.
- Neo4j is already installed.
- Claude Code, Codex, and OpenClaw are all present.
- The user wants a Mac Mini server.
- The user wants the REST API exposed beyond localhost.
- The user's project names or memory scopes match this repo's examples.

## 1. Audit The User's System

Before running install commands, inspect:

- OS and architecture: macOS, Linux, Apple Silicon, x86, VPS, local laptop.
- Python version: needs Python 3.10+.
- Neo4j status: installed, running, credentials known, local or remote.
- Agent surfaces: Claude Code, Codex, OpenClaw, Claude Desktop, custom scripts.
- MCP config paths for the user's actual clients.
- Whether the machine should host the REST API or only connect to one.
- Whether this is single-machine or multi-machine.
- Security posture: is anything bound to `0.0.0.0`, Tailscale, LAN, or public internet?
- Desired memory scopes: projects, clients, personal assistant, research, system.

Then summarize your proposed topology in plain language.

Example:

```text
Recommended topology:
- Run Neo4j and the Jarvis REST API on this Mac Mini.
- Register MCP on Claude Code and Codex on this laptop.
- Keep REST bound to localhost or Tailscale only.
- Use group_ids: personal-chief-of-staff, startup-a, system.
```

## 2. Choose A Topology

Common options:

### Single Machine

Best for a first install.

- Neo4j runs locally.
- Jarvis REST API runs locally.
- MCP clients launch `jarvis-mcp` from the same repo.
- ChromaDB lives under `~/.jarvis-memory/chromadb`.

### Server + Laptop

Best when the user has an always-on machine.

- Neo4j and REST API run on the server.
- Coding agents on laptops connect through MCP to the same Neo4j backend.
- Use Tailscale or equivalent private networking.
- Set `JARVIS_API_BEARER_TOKEN` if REST is reachable by anything beyond trusted localhost.

### MCP Only

Best when the user only wants Claude Code or Codex memory and does not need HTTP integrations.

- Install the Python package and Neo4j.
- Register MCP.
- REST API can be skipped until needed.

## 3. Install

The normal path:

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
bash scripts/client-install.sh
```

If you need to do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python scripts/migrate_to_v2.py
bash scripts/verify_install.sh
```

Never print secrets. If the user needs to provide a password or API key, have them paste it into `.env` or an interactive prompt.

## 4. Configure Memory Scopes

Pick stable `group_id` values with the user.

Good pattern:

```text
personal-chief-of-staff
company-name
client-name
research-topic
system
```

Avoid:

- Random one-off IDs.
- Uppercase names.
- Spaces.
- Putting unrelated clients into one scope.

Tell the user's agents:

```text
Always include group_id on every Jarvis write. Use system only for admin/infrastructure memory.
```

## 5. Register MCP

Claude Code:

```bash
python scripts/register_mcp.py --client claude-code
```

Codex:

```bash
python scripts/register_mcp.py --client codex
```

Restart the client after registration.

Then verify the client can see tools such as:

- `save_episode`
- `scored_search`
- `wake_up`
- `session_handoff`
- `latest_handoff`
- `doctor`

## 6. Configure Hooks Deliberately

Claude Code hooks are useful but optional:

```bash
python install_hooks.py
```

They do two things:

- `SessionStart`: inject recent memory when a session starts.
- `PreCompact`: write a handoff before compaction.

Important: the stop hook does not write canonical `STATUS.md` files by default. If the user wants a file fallback, configure `JARVIS_STOP_STATUS_FALLBACK_PATH` to a non-`STATUS.md` path.

## 7. REST API

Start it:

```bash
source .venv/bin/activate
python -m jarvis_memory.api
```

Health:

```bash
curl http://localhost:3500/health
```

If binding beyond loopback:

```env
JARVIS_API_BEARER_TOKEN=generate-a-long-random-token
JARVIS_REQUIRE_AUTH=1
```

Then send:

```bash
curl -H "Authorization: Bearer $JARVIS_API_BEARER_TOKEN" http://host:3500/health
```

Do not expose a writable memory API to the public internet without auth.

## 8. Smoke Test

Save:

```bash
curl -X POST http://localhost:3500/api/v2/save_episode \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "demo",
    "content": "[FACT] Jarvis Memory smoke test works. WHY: install verification.",
    "episode_type": "fact"
  }'
```

Search:

```bash
curl -X POST http://localhost:3500/api/v2/scored_search \
  -H "Content-Type: application/json" \
  -d '{"group_id":"demo","query":"smoke test","limit":5}'
```

Graph health:

```bash
curl http://localhost:3500/api/v2/doctor
```

Install verification:

```bash
bash scripts/verify_install.sh
```

Report exact outputs to the user. Do not say "it works" until these pass.

## 9. Teach The User's Agents The Memory Protocol

Add this to the user's agent instructions:

```text
Use Jarvis Memory for durable cross-session memory.

At session start:
1. Call wake_up(group_id="<current-scope>").
2. Call latest_handoff(group_id="<current-scope>") or list_sessions/continue_session if resuming work.

During work:
- Save decisions, corrections, commitments, plans, blockers, outcomes, and durable user preferences.
- Use structured memory:
  [DECISION] What changed
  WHY: why it changed
  IMPACT: what future agents should do differently
- Do not save scratchpad reasoning, temporary chat context, secrets, or huge raw logs.

At session end or before compaction:
- Call session_handoff with task, completed state, blockers, and next steps.
```

## 10. Production Checklist

Before calling the install done:

- `scripts/verify_install.sh` passes or any failure is explained.
- `/health` shows Neo4j reachable.
- `save_episode` and `scored_search` smoke test pass.
- MCP client can see Jarvis tools.
- `group_id` scopes are documented for the user's system.
- REST binding and bearer-token posture are safe.
- The user knows where `.env`, logs, ChromaDB, and Neo4j data live.
- Scheduled compaction is installed only if the user wants it.
- Minions are enabled only if the user explicitly wants a durable job queue.

## 11. What To Avoid

- Do not hardcode Alex's paths.
- Do not assume the user's machine names.
- Do not register every optional integration by default.
- Do not expose REST write endpoints without auth.
- Do not silently wipe Neo4j or ChromaDB.
- Do not save secrets into Jarvis Memory.
- Do not create dozens of `group_id`s before the user has a clear scope model.
