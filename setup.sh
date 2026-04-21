#!/bin/bash
# ──────────────────────────────────────────────────────────
# Jarvis Memory — LEGACY Setup Script (DEPRECATED 2026-04-20)
#
# This script hardcodes ~/Desktop/Jarvis/jarvis-memory and assumes a
# two-machine MBP/Mini "brain / client" architecture that does NOT
# apply to fresh installs.
#
# For a new install, use one of these instead:
#   (1) Astack MEGA-PROMPT (autonomous OpenClaw install)  ← recommended
#   (2) Manual install per README.md (Quick-start section)
#   (3) bash setup_venv.sh  (just the Python venv + pip install)
#
# Re-run with JARVIS_LEGACY_SETUP=1 if you intentionally want Alex's
# old MBP/Mini-pair behavior.
# ──────────────────────────────────────────────────────────

cat >&2 <<'WARN'

  ⚠  setup.sh is DEPRECATED. Use one of:
       bash setup_venv.sh                (Python venv + deps)
       cp .env.example .env              (fill in Neo4j creds)
       python scripts/migrate_to_v2.py   (apply Run 2 schema)
       python -m jarvis_memory.api       (start REST on :3500)

     Override with JARVIS_LEGACY_SETUP=1 if you really mean it.

WARN

if [[ "${JARVIS_LEGACY_SETUP:-0}" != "1" ]]; then
    exit 2
fi

set -e

JARVIS_DIR="$HOME/Desktop/Jarvis/jarvis-memory"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "═══════════════════════════════════════════"
echo "  Jarvis Memory — Shared Brain Setup"
echo "═══════════════════════════════════════════"
echo ""

# ── 0. Detect machine role ──────────────────────────────
echo -e "${YELLOW}[0/7] Detecting machine role...${NC}"

HOSTNAME=$(hostname -s 2>/dev/null || hostname)
echo -e "  Hostname: ${CYAN}$HOSTNAME${NC}"

# Auto-detect or ask
if [[ "$HOSTNAME" == *"mini"* ]] || [[ "$HOSTNAME" == *"Mini"* ]]; then
    DEVICE_ID="mac-mini"
    echo -e "  ${GREEN}Detected: Mac Mini (brain/server)${NC}"
elif [[ "$HOSTNAME" == *"MacBook"* ]] || [[ "$HOSTNAME" == *"macbook"* ]]; then
    DEVICE_ID="macbook-pro"
    echo -e "  ${GREEN}Detected: MacBook Pro (client)${NC}"
else
    echo ""
    echo "  Which machine is this?"
    echo "  1) Mac Mini   (runs Neo4j, always-on brain)"
    echo "  2) MacBook Pro (connects to Mac Mini remotely)"
    read -p "  Select (1/2): " -n 1 -r
    echo ""
    if [[ $REPLY == "1" ]]; then
        DEVICE_ID="mac-mini"
    else
        DEVICE_ID="macbook-pro"
    fi
fi

echo ""

# ── 1. Check Python ──────────────────────────────────────
echo -e "${YELLOW}[1/7] Checking Python...${NC}"
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}Found: $PY_VERSION${NC}"
else
    echo -e "  ${RED}Python 3 not found. Install it first: brew install python${NC}"
    exit 1
fi

# ── 2. Check / Install Neo4j ─────────────────────────────
echo -e "${YELLOW}[2/7] Checking Neo4j...${NC}"

if [[ "$DEVICE_ID" == "mac-mini" ]]; then
    # Mac Mini: Neo4j runs locally
    NEO4J_HOST="localhost"
    NEO4J_RUNNING=false

    if curl -s http://localhost:7474 &>/dev/null; then
        echo -e "  ${GREEN}Neo4j is already running on port 7474${NC}"
        NEO4J_RUNNING=true
    elif command -v neo4j &>/dev/null; then
        echo -e "  Neo4j is installed but not running."
        echo -e "  Start it with: ${YELLOW}neo4j start${NC}"
    elif brew list neo4j &>/dev/null 2>&1; then
        echo -e "  Neo4j installed via Homebrew but not running."
        echo -e "  Start it with: ${YELLOW}brew services start neo4j${NC}"
    else
        echo -e "  ${YELLOW}Neo4j not found. Installing via Homebrew...${NC}"
        echo ""
        echo "  Neo4j is the graph database that powers Jarvis Memory."
        echo "  It will run on this Mac Mini as the shared brain."
        echo ""
        read -p "  Install Neo4j now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            brew install neo4j
            echo -e "  ${GREEN}Neo4j installed.${NC}"
            echo -e "  Starting Neo4j..."
            brew services start neo4j
            sleep 5
            NEO4J_RUNNING=true
        else
            echo -e "  ${RED}Skipping Neo4j install. You'll need it before Jarvis Memory can run.${NC}"
        fi
    fi

    # Configure Neo4j for remote access
    echo ""
    echo -e "  ${YELLOW}Configuring Neo4j for remote access (MacBook Pro → Mac Mini)...${NC}"
    NEO4J_CONF=""
    # Check common Neo4j config locations
    for conf_path in \
        "/opt/homebrew/etc/neo4j/neo4j.conf" \
        "/usr/local/etc/neo4j/neo4j.conf" \
        "$HOME/.neo4j/neo4j.conf"; do
        if [ -f "$conf_path" ]; then
            NEO4J_CONF="$conf_path"
            break
        fi
    done

    if [ -n "$NEO4J_CONF" ]; then
        echo -e "  Found config: $NEO4J_CONF"
        # Check if remote access is already enabled
        if grep -q "^server.default_listen_address=0.0.0.0" "$NEO4J_CONF" 2>/dev/null; then
            echo -e "  ${GREEN}Remote access already enabled${NC}"
        else
            echo "  Neo4j needs to listen on 0.0.0.0 for MacBook Pro to connect."
            read -p "  Enable remote access? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Uncomment or add the listen address
                if grep -q "#server.default_listen_address=0.0.0.0" "$NEO4J_CONF"; then
                    sed -i '' 's/#server.default_listen_address=0.0.0.0/server.default_listen_address=0.0.0.0/' "$NEO4J_CONF"
                else
                    echo "server.default_listen_address=0.0.0.0" >> "$NEO4J_CONF"
                fi
                echo -e "  ${GREEN}Remote access enabled. Restart Neo4j to apply.${NC}"
                echo -e "  ${YELLOW}Run: brew services restart neo4j${NC}"
            fi
        fi
    else
        echo -e "  ${YELLOW}Neo4j config not found at expected paths.${NC}"
        echo "  Ensure server.default_listen_address=0.0.0.0 is set in neo4j.conf"
    fi

else
    # MacBook Pro: connects to Mac Mini's Neo4j
    echo "  MacBook Pro connects to Neo4j on the Mac Mini."
    echo ""
    read -p "  Enter Mac Mini hostname or IP (default: macmini.local): " NEO4J_HOST
    NEO4J_HOST=${NEO4J_HOST:-macmini.local}

    echo -e "  ${YELLOW}Testing connection to $NEO4J_HOST:7687...${NC}"
    if nc -z -w3 "$NEO4J_HOST" 7687 2>/dev/null; then
        echo -e "  ${GREEN}Connection successful!${NC}"
    else
        echo -e "  ${RED}Cannot reach $NEO4J_HOST:7687${NC}"
        echo "  Make sure:"
        echo "    1. Neo4j is running on the Mac Mini"
        echo "    2. Neo4j is configured to accept remote connections"
        echo "    3. Both machines are on the same network"
        echo ""
        echo "  Continuing setup anyway — you can fix connectivity later."
    fi
fi

# ── 3. Install jarvis-memory package ─────────────────────
echo -e "${YELLOW}[3/7] Installing jarvis-memory Python package...${NC}"

# Adjust path for MacBook Pro (synced to Atlas Copy)
if [[ "$DEVICE_ID" == "macbook-pro" ]]; then
    MBP_DIR="$HOME/Desktop/Atlas Copy/jarvis-memory"
    if [ -d "$MBP_DIR" ]; then
        JARVIS_DIR="$MBP_DIR"
        echo -e "  ${CYAN}Using Syncthing path: $JARVIS_DIR${NC}"
    fi
fi

cd "$JARVIS_DIR"
pip3 install -e ".[dev]" 2>&1 | tail -5
echo -e "  ${GREEN}jarvis-memory installed${NC}"

# ── 4. Create .env file ──────────────────────────────────
echo -e "${YELLOW}[4/7] Setting up environment...${NC}"
if [ -f "$JARVIS_DIR/.env" ]; then
    echo -e "  ${GREEN}.env already exists, skipping${NC}"
else
    cp "$JARVIS_DIR/.env.example" "$JARVIS_DIR/.env"
    echo ""
    echo "  Created .env from template."

    # Set device ID
    sed -i '' "s|JARVIS_DEVICE_ID=macbook-pro|JARVIS_DEVICE_ID=$DEVICE_ID|" "$JARVIS_DIR/.env"
    echo -e "  Device ID set to: ${CYAN}$DEVICE_ID${NC}"

    # Set Neo4j URI based on role
    if [[ "$DEVICE_ID" == "mac-mini" ]]; then
        NEO4J_URI="bolt://localhost:7687"
    else
        NEO4J_URI="bolt://${NEO4J_HOST}:7687"
    fi
    sed -i '' "s|NEO4J_URI=bolt://localhost:7687|NEO4J_URI=$NEO4J_URI|" "$JARVIS_DIR/.env"
    echo -e "  Neo4j URI: ${CYAN}$NEO4J_URI${NC}"

    # Ask for Anthropic API key (required — multi-query expansion in
    # jarvis_memory/search/expansion.py fails open to identity rewrite
    # without it, which cuts recall 3-8x. Opt out explicitly if you
    # really want that.)
    while :; do
        read -p "  Enter your Anthropic API key (sk-ant-api03-...) or SKIP-NO-RECALL to opt out: " API_KEY
        if [[ "$API_KEY" == "SKIP-NO-RECALL" ]]; then
            echo -e "  ${RED}⚠  WARNING: proceeding WITHOUT ANTHROPIC_API_KEY.${NC}"
            echo -e "  ${RED}   Multi-query expansion (Run 3) will be DISABLED.${NC}"
            echo -e "  ${RED}   Expected recall drop: 3-8x on the eval harness.${NC}"
            echo -e "  ${RED}   Add later via: echo 'ANTHROPIC_API_KEY=sk-ant-...' >> $JARVIS_DIR/.env${NC}"
            break
        fi
        # Key must start with the Anthropic v3 prefix AND be plausibly long.
        # Real keys are ~100 chars; reject anything under 40 as a typo.
        if [[ "$API_KEY" == sk-ant-api03-* && ${#API_KEY} -ge 40 ]]; then
            # Escape any '|' in the key so sed doesn't mis-parse (extremely unlikely
            # since keys are base64url-ish, but defense-in-depth).
            ESCAPED_KEY="${API_KEY//|/\\|}"
            sed -i '' "s|ANTHROPIC_API_KEY=sk-ant-...|ANTHROPIC_API_KEY=$ESCAPED_KEY|" "$JARVIS_DIR/.env"
            echo -e "  ${GREEN}API key saved${NC}"
            break
        fi
        echo -e "  ${RED}✗ API key required. Get one at https://console.anthropic.com/settings/keys${NC}"
        echo -e "  ${RED}  Must start with 'sk-ant-api03-' and be ≥40 chars.${NC}"
        echo -e "  ${RED}  Or type SKIP-NO-RECALL to opt out (degraded recall).${NC}"
    done

    # Ask for Neo4j password
    read -p "  Enter your Neo4j password (or press Enter for 'password'): " NEO_PASS
    if [ -n "$NEO_PASS" ]; then
        sed -i '' "s|NEO4J_PASSWORD=your-neo4j-password|NEO4J_PASSWORD=$NEO_PASS|" "$JARVIS_DIR/.env"
    fi

    echo -e "  ${GREEN}.env configured${NC}"
fi

# ── 5. Configure Claude Desktop MCP ──────────────────────
echo -e "${YELLOW}[5/7] Configuring Claude Desktop MCP server...${NC}"

CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

# Load .env values
source "$JARVIS_DIR/.env" 2>/dev/null || true
NEO_PASS=${NEO4J_PASSWORD:-password}
API_KEY_VAL=${ANTHROPIC_API_KEY:-sk-ant-...}
NEO4J_URI_VAL=${NEO4J_URI:-bolt://localhost:7687}
DEVICE_ID_VAL=${JARVIS_DEVICE_ID:-$DEVICE_ID}

if [ -f "$CLAUDE_CONFIG" ]; then
    echo "  Existing Claude Desktop config found."
    read -p "  Add/update jarvis-memory MCP server? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -c "
import json

config_path = '$CLAUDE_CONFIG'
jarvis_dir = '$JARVIS_DIR'
neo_pass = '$NEO_PASS'
api_key = '$API_KEY_VAL'
neo4j_uri = '$NEO4J_URI_VAL'
device_id = '$DEVICE_ID_VAL'

try:
    with open(config_path) as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if 'mcpServers' not in config:
    config['mcpServers'] = {}

config['mcpServers']['jarvis-memory'] = {
    'command': 'python3',
    'args': ['-m', 'mcp_server.server'],
    'cwd': jarvis_dir,
    'env': {
        'NEO4J_URI': neo4j_uri,
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': neo_pass,
        'ANTHROPIC_API_KEY': api_key,
        'JARVIS_DEVICE_ID': device_id,
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('  MCP config updated successfully')
"
        echo -e "  ${GREEN}Claude Desktop MCP configured${NC}"
        echo -e "  ${YELLOW}Restart Claude Desktop to pick up the changes.${NC}"
    fi
else
    echo -e "  ${YELLOW}Claude Desktop config not found at expected path.${NC}"
    echo "  You may need to configure it manually."
fi

# ── 6. Configure hooks ───────────────────────────────────
echo -e "${YELLOW}[6/7] Setting up Claude hooks...${NC}"

HOOKS_DIR="$HOME/.claude"
HOOKS_FILE="$HOOKS_DIR/hooks.json"

if [ -f "$HOOKS_FILE" ]; then
    echo "  Existing hooks.json found."
    echo "  You may want to merge jarvis-memory hooks manually."
    echo "  See: $JARVIS_DIR/hooks/"
else
    mkdir -p "$HOOKS_DIR"
    cat > "$HOOKS_FILE" << HOOKEOF
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "python3 $JARVIS_DIR/hooks/session_start.py",
        "timeout": 5000
      }
    ],
    "Stop": [
      {
        "type": "command",
        "command": "python3 $JARVIS_DIR/hooks/session_stop.py",
        "timeout": 5000
      }
    ]
  }
}
HOOKEOF
    echo -e "  ${GREEN}Hooks configured at $HOOKS_FILE${NC}"
fi

# ── 7. Verify connectivity ───────────────────────────────
echo -e "${YELLOW}[7/7] Verifying setup...${NC}"

# Quick Python import check
python3 -c "from jarvis_memory.conversation import SessionManager, EpisodeRecorder, SnapshotManager; print('  Conversation persistence: OK')" 2>/dev/null || echo -e "  ${RED}Import check failed — run: pip3 install -e '.[dev]'${NC}"
python3 -c "from jarvis_memory.scoring import composite_score; print('  Scoring engine: OK')" 2>/dev/null || echo -e "  ${RED}Scoring import failed${NC}"
python3 -c "from jarvis_memory.lifecycle import MemoryLifecycle; print('  Lifecycle engine: OK')" 2>/dev/null || echo -e "  ${RED}Lifecycle import failed${NC}"

# Neo4j connectivity check
echo ""
echo -e "  ${YELLOW}Testing Neo4j connection...${NC}"
python3 -c "
from neo4j import GraphDatabase
import os

uri = os.getenv('NEO4J_URI', '$NEO4J_URI_VAL')
user = os.getenv('NEO4J_USER', 'neo4j')
pwd = os.getenv('NEO4J_PASSWORD', '$NEO_PASS')

try:
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session() as s:
        result = s.run('RETURN 1 AS n')
        result.single()
    driver.close()
    print(f'  Neo4j connection to {uri}: OK')
except Exception as e:
    print(f'  Neo4j connection to {uri}: FAILED ({e})')
    print('  Make sure Neo4j is running and credentials are correct.')
" 2>/dev/null || echo -e "  ${RED}Neo4j driver not installed. Run: pip3 install neo4j${NC}"

# ── Done ──────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════"
echo -e "  ${GREEN}Jarvis Memory setup complete!${NC}"
echo -e "  ${CYAN}Device: $DEVICE_ID${NC}"
echo "═══════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Make sure Neo4j is running on the Mac Mini"
if [[ "$DEVICE_ID" == "macbook-pro" ]]; then
echo "  2. Verify you can reach the Mac Mini: nc -z ${NEO4J_HOST:-macmini.local} 7687"
fi
echo "  3. Restart Claude Desktop to load the MCP server"
echo "  4. Test: open Claude and ask 'What memory tools do you have?'"
echo ""
echo "  Files:"
echo "  - Package: $JARVIS_DIR"
echo "  - MCP config: $CLAUDE_CONFIG"
echo "  - Hooks: $HOOKS_FILE"
echo "  - Env: $JARVIS_DIR/.env"
echo ""
echo "  Run tests: cd $JARVIS_DIR && python3 -m pytest tests/ -v"
echo ""
