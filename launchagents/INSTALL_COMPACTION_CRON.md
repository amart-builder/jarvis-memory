# Install Jarvis compaction LaunchAgents

**Target machine:** Mac Mini (where `jarvis_memory.api` runs). Not the MacBook Pro.
**Who runs this:** OpenClaw, invoked by Alex via Discord.

> ⚠️ **Mini users: use [INSTALL_MINI.md](INSTALL_MINI.md) instead.** This doc has MBP paths hardcoded (`/Users/alexanderjmartin/Atlas/...`). The Mini uses a different username AND `~/Desktop/Atlas/` as the Atlas root — the plists here won't resolve correctly. Use the Mini-specific install doc.

## What you're installing

Two user-level LaunchAgents that run Jarvis memory compaction on a schedule:

| Agent | When | What it does |
|---|---|---|
| `com.atlas.jarvis-compact-daily` | Every day at **3:00 AM local** | Calls `CompactionEngine.daily_digest()` — merges near-duplicate memories from the last 24h across all `group_id`s |
| `com.atlas.jarvis-compact-weekly` | **Sundays at 4:00 AM local** | Calls `CompactionEngine.weekly_merge()` — consolidates daily digests into long-term memories across all `group_id`s |

Both exit 0 on success, non-zero on failure. Stdout and stderr are captured to `~/Atlas/brain/logs/jarvis-compact-{tier}.{out,err}`.

## Prerequisites

1. Mac Mini has the synced `~/Atlas/` folder (via Syncthing) — confirmed by Alex's setup
2. Jarvis venv exists at `~/Atlas/jarvis-memory/.venv/` with `neo4j`, `chromadb`, etc. installed — should be there per the cutover on 2026-04-07
3. `~/Atlas/brain/logs/` exists and is writable
4. Neo4j is reachable via `bolt://100.102.6.81:7687` (i.e., the Mini itself)

## Install steps

Run as Alex's user (not sudo):

```bash
# 1) Verify the venv and runner exist
ls -la ~/Atlas/jarvis-memory/.venv/bin/python ~/Atlas/jarvis-memory/scripts/run_compaction.py

# 2) Ensure logs directory exists
mkdir -p ~/Atlas/brain/logs

# 3) Symlink (not copy) the plists into ~/Library/LaunchAgents/
#    Symlinks let edits to the source files propagate automatically via Syncthing.
ln -sf ~/Atlas/jarvis-memory/launchagents/com.atlas.jarvis-compact-daily.plist \
       ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist
ln -sf ~/Atlas/jarvis-memory/launchagents/com.atlas.jarvis-compact-weekly.plist \
       ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist

# 4) Load both agents
launchctl load ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist
launchctl load ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist

# 5) Verify they're loaded
launchctl list | grep jarvis-compact
# Expected: two lines, one per agent, with PID "-" (not running right now, just scheduled)
```

## Smoke test (do ONE of these now so we know it works)

Pick the first run tomorrow at 3 AM OR trigger a manual test run:

```bash
# Manual test run (safe — respects idempotency, won't double-compact):
~/Atlas/jarvis-memory/.venv/bin/python \
  ~/Atlas/jarvis-memory/scripts/run_compaction.py --tier daily

# Expected: single-line JSON result on stdout, structured logs on stderr.
# Check the logs:
cat ~/Atlas/brain/logs/jarvis-compact-daily.out
```

A healthy first run looks like:
```json
{"tier":"daily","group_id":null,"result":{"merged":0,"scanned":N,"run_id":"daily-YYYYMMDD-xxxxxxxx"}}
```

With only ~63 episodes currently, expect `merged=0` (nothing duplicated yet). That's correct. As memory grows, merged counts will grow.

## Uninstall (if ever needed)

```bash
launchctl unload ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist
launchctl unload ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist
rm ~/Library/LaunchAgents/com.atlas.jarvis-compact-{daily,weekly}.plist
```

## Reporting back

After installing, reply to Alex on Discord with:
1. ✓ or ✗ for each step
2. Output of `launchctl list | grep jarvis-compact`
3. Output of the smoke test (the JSON line)
4. Any error messages from `~/Atlas/brain/logs/jarvis-compact-*.err`

If anything fails, don't improvise. Show Alex the error and stop.
