# Install Jarvis compaction cron — Mini-specific

**Why this exists**: the synced `INSTALL_COMPACTION_CRON.md` uses MBP paths (`/Users/alexanderjmartin/Atlas/...`). The Mac Mini uses a different username (`alexandermartin`, no "j") AND a different Atlas root (`~/Desktop/Atlas/`). So the plists must be written with Mini-specific paths directly to `~/Library/LaunchAgents/`. Do not `ln -s` from the synced `launchagents/` folder — those files have the wrong paths for the Mini.

## Run these on the Mini (as user `alexandermartin`)

```bash
# 0) Quick reality checks (fail fast if any of these is wrong)
ls -la /Users/alexandermartin/Desktop/Atlas/jarvis-memory/.venv/bin/python
ls -la /Users/alexandermartin/Desktop/Atlas/jarvis-memory/scripts/run_compaction.py
mkdir -p /Users/alexandermartin/Desktop/Atlas/brain/logs

# 1) Write the daily plist directly to LaunchAgents (Mini paths hardcoded)
cat > ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.atlas.jarvis-compact-daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory/.venv/bin/python</string>
        <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory/scripts/run_compaction.py</string>
        <string>--tier</string>
        <string>daily</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>3</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/Users/alexandermartin/Desktop/Atlas/brain/logs/jarvis-compact-daily.out</string>
    <key>StandardErrorPath</key>
    <string>/Users/alexandermartin/Desktop/Atlas/brain/logs/jarvis-compact-daily.err</string>
    <key>WorkingDirectory</key>
    <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

# 2) Write the weekly plist (Sundays at 4 AM)
cat > ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.atlas.jarvis-compact-weekly</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory/.venv/bin/python</string>
        <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory/scripts/run_compaction.py</string>
        <string>--tier</string>
        <string>weekly</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key><integer>0</integer>
        <key>Hour</key><integer>4</integer>
        <key>Minute</key><integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/Users/alexandermartin/Desktop/Atlas/brain/logs/jarvis-compact-weekly.out</string>
    <key>StandardErrorPath</key>
    <string>/Users/alexandermartin/Desktop/Atlas/brain/logs/jarvis-compact-weekly.err</string>
    <key>WorkingDirectory</key>
    <string>/Users/alexandermartin/Desktop/Atlas/jarvis-memory</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

# 3) Load both agents
launchctl load ~/Library/LaunchAgents/com.atlas.jarvis-compact-daily.plist
launchctl load ~/Library/LaunchAgents/com.atlas.jarvis-compact-weekly.plist

# 4) Verify loaded (should show 2 lines)
launchctl list | grep jarvis-compact

# 5) Manual smoke test — run the daily tier once now
/Users/alexandermartin/Desktop/Atlas/jarvis-memory/.venv/bin/python \
    /Users/alexandermartin/Desktop/Atlas/jarvis-memory/scripts/run_compaction.py --tier daily
```

## Expected smoke-test output

Single-line JSON on stdout, structured logs on stderr. Something like:

```json
{"tier":"daily","group_id":null,"result":{"merged":0,"scanned":N,"run_id":"daily-20260417-xxxxxxxx"}}
```

With ~60 Episodes currently in Neo4j, expect `merged=0` (nothing duplicated yet). That's correct — dedup pressure grows as memory accumulates.

## Report back to Alex

1. Output of `launchctl list | grep jarvis-compact` (should show 2 entries with PID `-`)
2. The exact JSON result line from step 5
3. Any stderr from `~/Desktop/Atlas/brain/logs/jarvis-compact-daily.err` (if populated)
4. Any errors, don't improvise — show Alex and stop

## Uninstall (if needed)

```bash
launchctl unload ~/Library/LaunchAgents/com.atlas.jarvis-compact-{daily,weekly}.plist
rm ~/Library/LaunchAgents/com.atlas.jarvis-compact-{daily,weekly}.plist
```
