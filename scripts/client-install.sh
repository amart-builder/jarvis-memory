#!/usr/bin/env bash
#
# client-install.sh — one-command Jarvis-Memory install for clients.
#
# Guides a non-technical user (or an agent) through: prereqs check, venv,
# .env setup, model pre-cache, schema migration, verification, and scheduled
# compaction. Minions (optional background job queue) is opt-in via prompt.
#
# Usage:
#   scripts/client-install.sh                 # interactive
#   scripts/client-install.sh --yes           # assume yes on every prompt (default N for minions)
#   scripts/client-install.sh --no-schedule   # skip launchd/systemd install
#   scripts/client-install.sh --no-hooks      # skip Claude Code hook registration
#
# Re-run anytime. Idempotent.

set -euo pipefail

# --- Resolve paths ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${JARVIS_LOG_DIR:-$HOME/.jarvis-memory/logs}"
mkdir -p "$LOG_DIR"

# --- Args ---
ASSUME_YES=0
SKIP_SCHEDULE=0
SKIP_HOOKS=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)       ASSUME_YES=1 ;;
        --no-schedule)  SKIP_SCHEDULE=1 ;;
        --no-hooks)     SKIP_HOOKS=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $arg (--help for usage)" >&2; exit 2 ;;
    esac
done

# --- Tiny helpers ---
say()  { printf '\n\033[1;34m▶\033[0m %s\n' "$1"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$1"; }
err()  { printf '  \033[1;31m✗\033[0m %s\n' "$1" >&2; }

# ask <prompt> <default-Y-or-N> → echo "y" or "n"
ask() {
    local prompt="$1" default="${2:-n}" reply
    if [[ "$ASSUME_YES" == "1" ]]; then
        echo "$default"; return
    fi
    local hint="[y/N]"
    [[ "$default" == "y" ]] && hint="[Y/n]"
    read -rp "  ${prompt} ${hint} " reply < /dev/tty || true
    reply="${reply:-$default}"
    case "${reply,,}" in y|yes) echo "y" ;; *) echo "n" ;; esac
}

ask_value() {
    # ask_value <prompt> <default>
    local prompt="$1" default="$2" reply
    if [[ "$ASSUME_YES" == "1" ]]; then
        echo "$default"; return
    fi
    read -rp "  ${prompt} [${default}]: " reply < /dev/tty || true
    echo "${reply:-$default}"
}

# --- 1. Prereqs ---
say "Checking prerequisites"

if ! command -v python3 &>/dev/null; then
    err "python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3,10) else 0)')"
if [[ "$PY_OK" != "1" ]]; then
    err "Python 3.10+ required; found $PY_VER."
    exit 1
fi
ok "python $PY_VER"

for bin in git curl; do
    if ! command -v "$bin" &>/dev/null; then
        err "$bin not found. Install it and re-run."
        exit 1
    fi
    ok "$bin"
done

# --- 2. Virtualenv ---
say "Setting up Python virtualenv"
if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    bash "$REPO_ROOT/setup_venv.sh"
else
    ok ".venv already exists — reusing"
fi
# shellcheck source=/dev/null
. "$REPO_ROOT/.venv/bin/activate"
ok "venv activated: $(python -c 'import sys; print(sys.executable)')"

# --- 3. .env ---
say "Configuring .env"
if [[ ! -f "$REPO_ROOT/.env" ]]; then
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    chmod 600 "$REPO_ROOT/.env"
    ok ".env created from .env.example (chmod 600)"

    if [[ "$ASSUME_YES" == "0" ]]; then
        echo "  Fill in Neo4j + Anthropic credentials now? (You can also edit .env by hand later.)"
        if [[ "$(ask "Edit .env interactively?" y)" == "y" ]]; then
            neo4j_uri="$(ask_value "NEO4J_URI" "bolt://localhost:7687")"
            neo4j_user="$(ask_value "NEO4J_USER" "neo4j")"
            read -rp "  NEO4J_PASSWORD: " neo4j_pass < /dev/tty
            anth_key="$(ask_value "ANTHROPIC_API_KEY (blank to skip; search falls back to keyword-only)" "")"
            device_id="$(ask_value "JARVIS_DEVICE_ID (a short name for this machine)" "$(hostname -s 2>/dev/null || echo client)")"

            # Inline sed — BSD compat via the sentinel-extension trick.
            sed -i.bak \
                -e "s|^NEO4J_URI=.*|NEO4J_URI=${neo4j_uri}|" \
                -e "s|^NEO4J_USER=.*|NEO4J_USER=${neo4j_user}|" \
                -e "s|^NEO4J_PASSWORD=.*|NEO4J_PASSWORD=${neo4j_pass}|" \
                -e "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anth_key}|" \
                "$REPO_ROOT/.env"
            # Append JARVIS_DEVICE_ID if not present, else update.
            if grep -q '^JARVIS_DEVICE_ID=' "$REPO_ROOT/.env"; then
                sed -i.bak -e "s|^JARVIS_DEVICE_ID=.*|JARVIS_DEVICE_ID=${device_id}|" "$REPO_ROOT/.env"
            else
                echo "JARVIS_DEVICE_ID=${device_id}" >> "$REPO_ROOT/.env"
            fi
            rm -f "$REPO_ROOT/.env.bak"
            chmod 600 "$REPO_ROOT/.env"
            ok ".env populated (chmod 600)"
        else
            warn "You MUST edit .env before the next step. File: $REPO_ROOT/.env"
        fi
    fi
else
    # Enforce 600 even on existing files — reruns shouldn't leave secrets world-readable.
    chmod 600 "$REPO_ROOT/.env"
    ok ".env already exists — chmod 600 enforced"
fi

# --- 4. Pre-cache embedding model (~90 MB download on first run) ---
say "Pre-caching the sentence-transformer embedding model (~90 MB, one-time)"
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    >>"$LOG_DIR/client-install.log" 2>&1 && ok "embedding model cached" || {
    err "embedding model download failed — see $LOG_DIR/client-install.log"
    exit 1
}

# --- 5. Neo4j schema migration (idempotent) ---
say "Applying Neo4j schema migration"
python "$REPO_ROOT/scripts/migrate_to_v2.py" >>"$LOG_DIR/client-install.log" 2>&1 && \
    ok "schema migration applied" || {
    err "schema migration failed — see $LOG_DIR/client-install.log"
    err "Most common cause: Neo4j not reachable at \$NEO4J_URI or wrong credentials."
    exit 1
}

# --- 6. Verify install ---
say "Running 7-gate install verification"
if bash "$REPO_ROOT/scripts/verify_install.sh"; then
    ok "verify_install.sh passed"
else
    err "verify_install.sh failed — see /tmp/jarvis-verify.log + $LOG_DIR/client-install.log"
    exit 1
fi

# --- 7. Schedule compaction ---
if [[ "$SKIP_SCHEDULE" == "1" ]]; then
    warn "Skipping schedule install (--no-schedule)"
else
    say "Scheduling daily + weekly compaction"
    case "$(uname)" in
        Darwin)
            bash "$REPO_ROOT/scripts/generate_launchagents.sh"
            for p in com.atlas.jarvis-compact-daily com.atlas.jarvis-compact-weekly; do
                launchctl unload "$HOME/Library/LaunchAgents/${p}.plist" 2>/dev/null || true
                launchctl load "$HOME/Library/LaunchAgents/${p}.plist"
                ok "loaded ${p}"
            done
            ;;
        Linux)
            # User scope by default (no sudo prompt). Users who want system-scoped
            # timers can re-run `scripts/generate_systemd_units.sh` without --user.
            bash "$REPO_ROOT/scripts/generate_systemd_units.sh" --user
            systemctl --user enable --now jarvis-compact-daily.timer
            systemctl --user enable --now jarvis-compact-weekly.timer
            ok "timers enabled (user scope)"
            ;;
        *)  warn "Unsupported OS: $(uname). Skip scheduling; run manually." ;;
    esac
fi

# --- 8. Minions (optional) ---
say "Minions — optional background job queue"
cat <<'EOF'
  Minions is a durable SQLite-backed job queue, inspired by garrytan/gbrain.
  It handles deterministic scheduled work (cleanup, ingestion, periodic
  reports) without spending LLM tokens or risking a spawned sub-agent stalling.

  You can skip this for now and add it later — compaction already runs on the
  standard schedule without it. Most installs should skip on first setup.
EOF
if [[ "$(ask "Install the minion worker daemon now?" n)" == "y" ]]; then
    case "$(uname)" in
        Darwin)
            bash "$REPO_ROOT/scripts/generate_launchagents.sh" --with-minion-worker
            launchctl unload "$HOME/Library/LaunchAgents/com.atlas.minion-worker.plist" 2>/dev/null || true
            launchctl load "$HOME/Library/LaunchAgents/com.atlas.minion-worker.plist"
            ok "minion worker loaded"
            ;;
        Linux)
            bash "$REPO_ROOT/scripts/generate_systemd_units.sh" --user --with-minion-worker
            systemctl --user enable --now jarvis-minion-worker.service
            ok "minion worker enabled (user scope)"
            ;;
    esac
else
    ok "skipped — install later with scripts/generate_launchagents.sh --with-minion-worker"
fi

# --- 9. MCP integrations (optional — Claude Code + Codex) ---
say "MCP integrations — optional, per-client"
cat <<'EOF'
  Jarvis-Memory ships an MCP server exposing 27 tools. Any MCP-speaking
  client (Claude Code, Codex CLI, Claude Desktop) can consume them.
  This step writes a registration into each client's config file; skip
  anything you don't use.
EOF
if [[ "$(ask "Register MCP server with Claude Code (~/.claude/settings.json)?" n)" == "y" ]]; then
    python "$REPO_ROOT/scripts/register_mcp.py" --client claude-code
else
    ok "Claude Code MCP skipped — install later: python scripts/register_mcp.py --client claude-code"
fi
if [[ "$(ask "Register MCP server with Codex CLI (~/.codex/config.toml)?" n)" == "y" ]]; then
    python "$REPO_ROOT/scripts/register_mcp.py" --client codex
else
    ok "Codex MCP skipped — install later: python scripts/register_mcp.py --client codex"
fi

# --- 10. Claude Code hooks (optional) ---
if [[ "$SKIP_HOOKS" == "1" ]]; then
    warn "Skipping Claude Code hook registration (--no-hooks)"
elif [[ "$(ask "Register Claude Code hooks (SessionStart + PreCompact)?" n)" == "y" ]]; then
    say "Registering Claude Code hooks"
    python "$REPO_ROOT/install_hooks.py"
else
    ok "Hooks skipped — install later: python install_hooks.py"
fi

# --- 11. Done ---
say "Install complete"
cat <<EOF
  Repo:          $REPO_ROOT
  Logs:          $LOG_DIR
  ChromaDB:      \${JARVIS_CHROMADB_PATH:-\$HOME/.jarvis-memory/chromadb}

  Start the REST API + MCP server:
    cd $REPO_ROOT
    source .venv/bin/activate
    python -m jarvis_memory.api        # REST on localhost:3500
    jarvis-mcp                         # MCP (for Claude Code / Desktop)

  Write + search a test episode:
    curl -X POST localhost:3500/api/v2/save_episode \\
      -H 'Content-Type: application/json' \\
      -d '{"group_id":"smoke","content":"it works","type":"fact"}'
    curl 'localhost:3500/api/v2/scored_search?group_id=smoke&query=works'

  Update to the latest release:
    scripts/upgrade.sh
EOF
