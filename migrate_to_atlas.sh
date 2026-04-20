#!/bin/bash
# Migration script: Move jarvis-memory venv from ~/Jarvis/ to ~/Atlas/
# Run from Terminal: bash ~/Atlas/jarvis-memory/migrate_to_atlas.sh
set -e

ATLAS_DIR="$HOME/Atlas/jarvis-memory"
JARVIS_DIR="$HOME/Jarvis/jarvis-memory"

echo "=== Jarvis-Memory Migration: ~/Jarvis/ → ~/Atlas/ ==="
echo ""

# Step 1: Find MCP config
echo "--- Step 1: Finding MCP config ---"
MCP_CONFIG=""
CANDIDATES=(
    "$HOME/.claude/mcp.json"
    "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
    "$HOME/.config/claude/mcp.json"
)
for f in "${CANDIDATES[@]}"; do
    if [ -f "$f" ]; then
        echo "Found: $f"
        MCP_CONFIG="$f"
        break
    fi
done
if [ -z "$MCP_CONFIG" ]; then
    echo "WARNING: Could not find MCP config. You'll need to update it manually."
    echo "Checked: ${CANDIDATES[*]}"
fi

# Step 1b: Show current MCP config for jarvis-memory
if [ -n "$MCP_CONFIG" ]; then
    echo ""
    echo "Current jarvis-memory config:"
    python3 -c "
import json, sys
with open('$MCP_CONFIG') as f:
    cfg = json.load(f)
servers = cfg.get('mcpServers', {})
for name, val in servers.items():
    s = json.dumps(val)
    if 'jarvis' in name.lower() or 'jarvis' in s.lower():
        print(f'  Server: {name}')
        print(f'  Config: {json.dumps(val, indent=4)}')
" 2>/dev/null || echo "  (could not parse config)"
fi

# Step 2: Kill existing MCP server
echo ""
echo "--- Step 2: Killing existing MCP server ---"
pkill -f "mcp_server.server" 2>/dev/null && echo "Killed old process" || echo "No running process found"

# Step 3: Remove old Atlas venv (it's a copy of the Jarvis one with wrong paths)
echo ""
echo "--- Step 3: Removing old venv at $ATLAS_DIR/.venv ---"
if [ -d "$ATLAS_DIR/.venv" ]; then
    rm -rf "$ATLAS_DIR/.venv"
    echo "Removed old venv"
else
    echo "No existing venv to remove"
fi

# Step 4: Create fresh venv at Atlas path
echo ""
echo "--- Step 4: Creating fresh venv at $ATLAS_DIR/.venv ---"
python3.14 -m venv "$ATLAS_DIR/.venv" 2>/dev/null || python3 -m venv "$ATLAS_DIR/.venv"
echo "Venv created"

# Step 5: Install dependencies
echo ""
echo "--- Step 5: Installing dependencies ---"
"$ATLAS_DIR/.venv/bin/pip" install --upgrade pip
"$ATLAS_DIR/.venv/bin/pip" install -e "$ATLAS_DIR"
echo "Dependencies installed"

# Step 6: Verify the install
echo ""
echo "--- Step 6: Verifying installation ---"
"$ATLAS_DIR/.venv/bin/python" -c "
from mcp_server.server import main
from jarvis_memory.scoring import composite_score
from jarvis_memory.classifier import classify_memory
from jarvis_memory.lifecycle import MemoryLifecycle
from jarvis_memory.conversation import SessionManager
print('All imports successful')
"
echo "Verification passed"

# Step 7: Update MCP config
if [ -n "$MCP_CONFIG" ]; then
    echo ""
    echo "--- Step 7: Updating MCP config ---"
    # Backup
    cp "$MCP_CONFIG" "${MCP_CONFIG}.bak.$(date +%s)"
    echo "Backup saved to ${MCP_CONFIG}.bak.*"

    python3 -c "
import json, re, sys

config_path = '$MCP_CONFIG'
atlas_venv = '$ATLAS_DIR/.venv'
jarvis_dir = '$JARVIS_DIR'
atlas_dir = '$ATLAS_DIR'

with open(config_path) as f:
    raw = f.read()

# Replace all references to Jarvis path with Atlas path
updated = raw.replace(jarvis_dir, atlas_dir)
# Also handle the venv python path specifically
updated = updated.replace('/Jarvis/jarvis-memory/.venv', '/Atlas/jarvis-memory/.venv')
updated = updated.replace('/Jarvis/jarvis-memory', '/Atlas/jarvis-memory')

with open(config_path, 'w') as f:
    f.write(updated)

# Show what changed
cfg = json.loads(updated)
servers = cfg.get('mcpServers', {})
for name, val in servers.items():
    s = json.dumps(val)
    if 'jarvis' in name.lower() or 'jarvis' in s.lower():
        print(f'  Updated server: {name}')
        print(f'  New config: {json.dumps(val, indent=4)}')
"
    echo "MCP config updated"
else
    echo ""
    echo "--- Step 7: MANUAL ACTION NEEDED ---"
    echo "Update your MCP config to use this Python path:"
    echo "  $ATLAS_DIR/.venv/bin/python"
    echo "And this working directory:"
    echo "  $ATLAS_DIR"
fi

echo ""
echo "=== Migration complete ==="
echo ""
echo "Next steps:"
echo "  1. Restart Claude Cowork (close and reopen)"
echo "  2. Test with a scored_search call"
echo ""
echo "The ~/Jarvis/jarvis-memory/ directory can be removed once everything is confirmed working."
