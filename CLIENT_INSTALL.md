# Client Install Guide

> **Who this is for:** a user (or an agent acting on behalf of a user) setting up a fresh Jarvis-Memory install on a single machine — Mac Mini, MacBook, or Linux VPS. Everything is opt-in; you can bolt on integrations (Claude Code, Codex, Minions) à la carte.
>
> **How long this takes:** 5–10 minutes on a fast connection. Most of the wall-clock is a one-time 90 MB download of the embedding model.

---

## The short version

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
bash scripts/client-install.sh
```

That's it. The script asks a few questions (Neo4j credentials, Anthropic key, whether to register with Claude Code, etc.) and does the rest. Skip to **[Troubleshooting](#troubleshooting)** if something breaks.

---

## Prerequisites

You need these on the target machine *before* running the installer:

| Requirement | macOS (one-liner) | Linux (Debian/Ubuntu) |
|---|---|---|
| **Python 3.10+** | `brew install python@3.12` | `sudo apt install python3.12 python3.12-venv` |
| **git** | usually preinstalled | `sudo apt install git` |
| **curl** | preinstalled | `sudo apt install curl` |
| **Neo4j 5.x** | `brew install neo4j && brew services start neo4j` | See below |
| **Anthropic API key** | grab from [console.anthropic.com](https://console.anthropic.com) | same |

### Installing Neo4j on a fresh Linux VPS

```bash
sudo apt update
sudo apt install -y openjdk-17-jre-headless
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neotechnology.gpg
echo 'deb [signed-by=/usr/share/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' \
    | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install -y neo4j
sudo systemctl enable --now neo4j
# Set a password (default user is 'neo4j', default password is 'neo4j' until you change it):
cypher-shell -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD 'your-new-password';"
```

After Neo4j is running, verify with:

```bash
curl -s http://localhost:7474 | head -1   # should return HTML from the Neo4j browser
```

---

## Step-by-step install

### 1. Clone the repo

```bash
git clone https://github.com/amart-builder/jarvis-memory.git
cd jarvis-memory
```

### 2. Run the installer

```bash
bash scripts/client-install.sh
```

The script walks through **10 steps**, printing a clear checkpoint at each. If any step fails, it exits non-zero and points you at the relevant log file.

The steps:

| # | What | Interactive? |
|---|---|---|
| 1 | Check prerequisites (python, git, curl) | no |
| 2 | Create Python virtualenv + install deps | no |
| 3 | Create `.env` — asks for Neo4j URI/user/password + Anthropic key | YES |
| 4 | Pre-cache the sentence-transformer embedding model (~90 MB, one-time) | no |
| 5 | Apply Neo4j schema migration (idempotent) | no |
| 6 | Run 7-gate verification | no |
| 7 | Install scheduled daily/weekly compaction (launchd on Mac / systemd on Linux) | no |
| 8 | **Optional**: install Minions background worker | YES (default N) |
| 9 | **Optional**: register MCP server with Claude Code and/or Codex CLI | YES (default N for each) |
| 10 | **Optional**: register Claude Code hooks (SessionStart + PreCompact) | YES (default N) |

If you run it with `--yes`, it answers "default" to every prompt (which means: yes for required steps, N for every optional one).

### 3. Verify it's running

```bash
source .venv/bin/activate
python -m jarvis_memory.api &      # REST API on localhost:3500
curl -X POST localhost:3500/api/v2/save_episode \
  -H 'Content-Type: application/json' \
  -d '{"group_id":"smoke","content":"it works","type":"fact"}'
curl 'localhost:3500/api/v2/scored_search?group_id=smoke&query=works'
```

If the second curl returns your `"it works"` episode, you're done.

---

## Optional integrations — explained

Each of these is a single command you can run now or later. They don't depend on each other.

### Claude Code — MCP server

Lets Claude Code call the 27 Jarvis-Memory tools directly from any session. Adds an entry to `~/.claude/settings.json`.

```bash
python scripts/register_mcp.py --client claude-code        # install
python scripts/register_mcp.py --client claude-code --uninstall  # remove
```

After install, **restart Claude Code** for it to pick up the new server.

### Codex CLI — MCP server

Same server, different config file. Adds a `[mcp_servers.jarvis-memory]` block to `~/.codex/config.toml` (preserving everything else in the file).

```bash
python scripts/register_mcp.py --client codex
python scripts/register_mcp.py --client codex --uninstall
```

### Claude Code — hooks (SessionStart + PreCompact)

Two hooks that make Claude Code sessions continuous:
- **SessionStart** — on session open, auto-inject recent context for this project from Jarvis-Memory (recent decisions, open questions, last handoff).
- **PreCompact** — right before Claude Code auto-compacts, write a [HANDOFF] episode so the next session picks up where this one left off.

```bash
python install_hooks.py                 # install
python install_hooks.py --uninstall     # remove
```

Both hooks read `JARVIS_LOG_DIR` (default `~/.jarvis-memory/logs/`) for their log files.

### Minions — durable background job queue

Ported from [garrytan/gbrain](https://github.com/garrytan/gbrain). Lets Jarvis-Memory (or any other script you wire up) run deterministic scheduled work in a SQLite-backed queue — without burning LLM tokens and without relying on OpenClaw sub-agents that can stall mid-flight.

**You probably don't need this on first install.** Scheduled compaction runs fine via standard launchd/systemd without Minions. Enable it if:
- Your OpenClaw cron jobs have been dropping
- You want to wire up your own deterministic background tasks (ingest a feed nightly, sync a dataset, etc.)
- You want the "durable, tokens-free" execution guarantee

Install:

```bash
# macOS
bash scripts/generate_launchagents.sh --with-minion-worker
launchctl load ~/Library/LaunchAgents/com.atlas.minion-worker.plist

# Linux
bash scripts/generate_systemd_units.sh --user --with-minion-worker
systemctl --user enable --now jarvis-minion-worker.service
```

The shell-execution handler (which lets Minions run arbitrary commands) is **disabled by default**. Leave it off unless you have a specific reason to enable it — enabling it means any code that can submit a job can also run shell commands as your user. If you do need it, set `GBRAIN_ALLOW_SHELL_JOBS=1` in `.env`.

---

## Keeping up to date

The repo is tagged with release versions (`v1.0.0`, `v1.0.1`, …). To upgrade to the latest release:

```bash
cd jarvis-memory
bash scripts/upgrade.sh                 # upgrade to latest tag
bash scripts/upgrade.sh --rolling       # use origin/main (development, may be unstable)
bash scripts/upgrade.sh --to v1.2.3     # pin to a specific version
```

The upgrade script fetches tags, checks out the target ref, re-runs the installer in non-interactive mode (preserves your `.env` and data), and re-runs verification. If anything breaks, it exits non-zero.

Your **data** — `.env`, ChromaDB, the Minions SQLite queue, logs, and your Neo4j database — is never touched by the upgrade script.

### Stay within one major version

Releases follow semver:

- **Major** (v1.x → v2.x): breaking changes, read the changelog first.
- **Minor** (v1.1 → v1.2): new features, safe to auto-upgrade.
- **Patch** (v1.0.0 → v1.0.1): bug fixes, always safe.

`scripts/upgrade.sh` without flags picks the latest tag regardless of major version, so pin with `--to` if you want to stay on a specific major.

---

## Troubleshooting

### `client-install.sh` dies at step 4 (embedding model download)

The sentence-transformer model is ~90 MB and downloads from huggingface.co. On a slow or restricted network this can time out. Retry with:

```bash
source .venv/bin/activate
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### `client-install.sh` dies at step 5 (schema migration)

Almost always Neo4j connectivity or credentials. Check:

```bash
cypher-shell -u <your-user> -p <your-password> -a <your-uri> "RETURN 1"
```

If that fails, fix `.env` and re-run `scripts/client-install.sh` (it's idempotent).

### `verify_install.sh` says ChromaDB is missing

The default ChromaDB path is `~/.jarvis-memory/chromadb/`. If you overrode `JARVIS_CHROMADB_PATH`, make sure the directory exists and is writable:

```bash
mkdir -p "${JARVIS_CHROMADB_PATH:-$HOME/.jarvis-memory/chromadb}"
```

### Claude Code isn't seeing the MCP server after `register_mcp.py`

Restart Claude Code. MCP server configs are only read at startup.

### I want a fresh install

```bash
bash scripts/client-install.sh --uninstall   # NOT IMPLEMENTED yet; see below
```

For now, manually:

```bash
# Remove scheduled jobs (macOS)
bash scripts/generate_launchagents.sh --uninstall

# Remove scheduled jobs (Linux)
bash scripts/generate_systemd_units.sh --uninstall

# Remove MCP registrations
python scripts/register_mcp.py --client claude-code --uninstall
python scripts/register_mcp.py --client codex --uninstall

# Remove Claude Code hooks
python install_hooks.py --uninstall

# Delete data (DESTRUCTIVE — kills your memory graph)
cypher-shell "MATCH (n) DETACH DELETE n"   # wipes Neo4j
rm -rf ~/.jarvis-memory/                    # wipes ChromaDB + logs
```

---

## For agents installing this for a user

If you're an OpenClaw / Claude Code / Codex agent driving this install on behalf of a user:

1. The user should be watching you at step 3 (they need to paste credentials).
2. Default every optional integration to **N** unless the user explicitly asks for it. "Do you want Minions?" is an opt-in decision the user should make deliberately.
3. Always end by running `scripts/verify_install.sh` and reporting its output verbatim.
4. If any step fails, surface the exact log file path (`$JARVIS_LOG_DIR/client-install.log` or `/tmp/jarvis-verify.log`) — don't paraphrase.
5. After success, offer two smoke-test curls (see "Verify it's running" above) and wait for the user to confirm they see output.

---

## What's next after install

- Read [`README.md`](README.md) for the architecture + API reference.
- Your agent can now use MCP tools like `save_episode`, `scored_search`, `wake_up`, `session_handoff`. Full list: `python -m mcp_server.server --list-tools` or see the `## MCP tools` section in the README.
- Integrate with your own scripts via the REST API at `localhost:3500/api/v2/*`.
