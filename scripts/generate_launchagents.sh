#!/usr/bin/env bash
#
# generate_launchagents.sh — emit per-user macOS LaunchAgent plists.
#
# The templates in ../launchagents/ use {{JARVIS_ROOT}} and {{LOG_DIR}} as
# placeholders. This script substitutes them with real paths and writes the
# result to ~/Library/LaunchAgents/, then prints the next commands you need
# to run to load them.
#
# Usage:
#   scripts/generate_launchagents.sh                          # install compaction plists
#   scripts/generate_launchagents.sh --with-minion-worker     # also install the minion worker
#   scripts/generate_launchagents.sh --uninstall              # unload + remove all jarvis plists
#
# Idempotent — safe to re-run. Will overwrite existing files with the same name.

set -euo pipefail

# Resolve repo root from script location (script is at scripts/generate_launchagents.sh).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/launchagents"
OUT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="${JARVIS_LOG_DIR:-$HOME/.jarvis-memory/logs}"

# Which plist files to install. Compaction is default-on (safe, boring).
# Minion worker is opt-in (--with-minion-worker flag) so non-minions users
# don't get a background worker they didn't ask for.
COMPACTION_PLISTS=(
    "com.atlas.jarvis-compact-daily.plist"
    "com.atlas.jarvis-compact-weekly.plist"
)
MINION_WORKER_PLIST="com.atlas.minion-worker.plist"

# --- Arg parsing ---
WITH_MINION_WORKER=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --with-minion-worker) WITH_MINION_WORKER=1 ;;
        --uninstall)          UNINSTALL=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# --- Preflight ---
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: generate_launchagents.sh is for macOS only." >&2
    echo "On Linux, use the systemd templates at systemd/ instead." >&2
    exit 1
fi

# --- Uninstall path ---
if [[ "$UNINSTALL" == "1" ]]; then
    echo "Unloading + removing jarvis-memory LaunchAgents..."
    for plist in "${COMPACTION_PLISTS[@]}" "$MINION_WORKER_PLIST"; do
        target="$OUT_DIR/$plist"
        if [[ -f "$target" ]]; then
            launchctl unload "$target" 2>/dev/null || true
            rm "$target"
            echo "  removed $plist"
        fi
    done
    echo "Done."
    exit 0
fi

# --- Install path ---
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Build the list of plists to install.
plists_to_install=("${COMPACTION_PLISTS[@]}")
if [[ "$WITH_MINION_WORKER" == "1" ]]; then
    plists_to_install+=("$MINION_WORKER_PLIST")
fi

echo "Generating LaunchAgent plists:"
echo "  JARVIS_ROOT = $REPO_ROOT"
echo "  LOG_DIR     = $LOG_DIR"
echo "  OUT_DIR     = $OUT_DIR"
echo

for plist in "${plists_to_install[@]}"; do
    template="$TEMPLATE_DIR/$plist"
    target="$OUT_DIR/$plist"

    if [[ ! -f "$template" ]]; then
        echo "ERROR: template missing: $template" >&2
        exit 1
    fi

    # sed -i '' for BSD (macOS); we write to target first, then substitute in place.
    cp "$template" "$target"
    sed -i '' \
        -e "s|{{JARVIS_ROOT}}|$REPO_ROOT|g" \
        -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
        "$target"

    echo "  ✓ wrote $target"
done

echo
echo "Next step — load the plists with launchctl:"
for plist in "${plists_to_install[@]}"; do
    echo "  launchctl load $OUT_DIR/$plist"
done
echo
echo "To unload later: scripts/generate_launchagents.sh --uninstall"
