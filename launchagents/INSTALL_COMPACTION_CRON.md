# Install jarvis-memory compaction LaunchAgents (macOS)

Schedules the daily + weekly compaction runs that consolidate near-duplicate memories.

> **Use `scripts/generate_launchagents.sh` — do NOT copy the raw plists.**
> The plists in this directory contain `{{JARVIS_ROOT}}` + `{{LOG_DIR}}` placeholders that must be substituted with paths appropriate to your install. The generator handles that.

## What you're installing

Two user-level LaunchAgents:

| Agent | When | What it does |
|---|---|---|
| `com.atlas.jarvis-compact-daily` | Every day at 3:00 AM local | Calls `CompactionEngine.daily_digest()` — merges near-duplicate memories from the last 24h across all `group_id`s |
| `com.atlas.jarvis-compact-weekly` | Sundays at 4:00 AM local | Calls `CompactionEngine.weekly_merge()` — consolidates daily digests into long-term memories |

Both exit 0 on success, non-zero on failure. Stdout / stderr are captured to `${JARVIS_LOG_DIR}/jarvis-compact-{daily,weekly}.{out,err}` (defaults to `~/.jarvis-memory/logs/`).

## Prerequisites

1. `.venv` exists in the repo root (created by `scripts/client-install.sh`).
2. Dependencies installed (`pip install -e .`).
3. `.env` populated with `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `ANTHROPIC_API_KEY`.
4. Neo4j is reachable at the configured `NEO4J_URI`.

## Install

From the repo root:

```bash
bash scripts/generate_launchagents.sh
```

The generator:
1. Reads `JARVIS_ROOT` (defaults to the repo's top-level directory) and `JARVIS_LOG_DIR` (defaults to `~/.jarvis-memory/logs/`).
2. Substitutes both into the plist templates.
3. Writes the resolved plists to `~/Library/LaunchAgents/com.atlas.jarvis-compact-{daily,weekly}.plist`.
4. Loads both agents via `launchctl`.

Override either location via env before running:

```bash
JARVIS_ROOT=/custom/path/to/jarvis-memory \
JARVIS_LOG_DIR=/var/log/jarvis \
bash scripts/generate_launchagents.sh
```

## Verify

```bash
launchctl list | grep jarvis-compact
# Expected: two lines (one per agent), PID "-" means "scheduled, not currently running"
```

## Smoke test

Run one compaction manually (idempotent — won't double-compact):

```bash
source .venv/bin/activate
python scripts/run_compaction.py --tier daily
```

A healthy run looks like:

```json
{"tier":"daily","group_id":null,"result":{"merged":N,"scanned":M,"run_id":"daily-YYYYMMDD-xxxxxxxx"}}
```

`merged=0` on a fresh install with few episodes is expected.

Check the log file for detail:

```bash
tail "$JARVIS_LOG_DIR/jarvis-compact-daily.out"
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist
launchctl unload ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist
rm ~/Library/LaunchAgents/com.atlas.jarvis-compact-{daily,weekly}.plist
```

## Linux

See `systemd/` + `scripts/generate_systemd_units.sh` for the equivalent systemd timer/service setup.
