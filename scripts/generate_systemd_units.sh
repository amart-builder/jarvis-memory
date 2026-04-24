#!/usr/bin/env bash
#
# generate_systemd_units.sh — emit per-user systemd unit files.
#
# The templates in ../systemd/ use {{USER}}, {{JARVIS_ROOT}}, and {{LOG_DIR}}
# as placeholders. This script substitutes them with real values and writes
# the result to /etc/systemd/system/ (system-wide) or
# ~/.config/systemd/user/ (user-scoped). System-wide requires sudo.
#
# Usage:
#   scripts/generate_systemd_units.sh                          # compaction only, system-wide
#   scripts/generate_systemd_units.sh --user                   # user-scoped instead of system
#   scripts/generate_systemd_units.sh --with-minion-worker     # also install minion worker
#   scripts/generate_systemd_units.sh --uninstall              # remove all jarvis units
#
# Idempotent — safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="$REPO_ROOT/systemd"
LOG_DIR="${JARVIS_LOG_DIR:-$HOME/.jarvis-memory/logs}"

# Default to system-wide; --user switches to the user scope.
USER_SCOPE=0
WITH_MINION_WORKER=0
UNINSTALL=0

for arg in "$@"; do
    case "$arg" in
        --user)               USER_SCOPE=1 ;;
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

if [[ "$(uname)" != "Linux" ]]; then
    echo "ERROR: generate_systemd_units.sh is for Linux only." >&2
    echo "On macOS, use scripts/generate_launchagents.sh instead." >&2
    exit 1
fi

# Install location + systemctl prefix depend on scope.
if [[ "$USER_SCOPE" == "1" ]]; then
    OUT_DIR="$HOME/.config/systemd/user"
    SYSTEMCTL="systemctl --user"
    SUDO=""
else
    OUT_DIR="/etc/systemd/system"
    SYSTEMCTL="systemctl"
    SUDO="sudo"
fi

COMPACTION_UNITS=(
    "jarvis-compact-daily.service"
    "jarvis-compact-daily.timer"
    "jarvis-compact-weekly.service"
    "jarvis-compact-weekly.timer"
)
MINION_WORKER_UNIT="jarvis-minion-worker.service"

# --- Uninstall path ---
if [[ "$UNINSTALL" == "1" ]]; then
    echo "Disabling + removing jarvis-memory systemd units..."
    for unit in "${COMPACTION_UNITS[@]}" "$MINION_WORKER_UNIT"; do
        $SYSTEMCTL disable --now "$unit" 2>/dev/null || true
        if [[ -f "$OUT_DIR/$unit" ]]; then
            $SUDO rm "$OUT_DIR/$unit"
            echo "  removed $unit"
        fi
    done
    $SUDO $SYSTEMCTL daemon-reload
    echo "Done."
    exit 0
fi

# --- Install path ---
mkdir -p "$LOG_DIR"
$SUDO mkdir -p "$OUT_DIR"

units_to_install=("${COMPACTION_UNITS[@]}")
if [[ "$WITH_MINION_WORKER" == "1" ]]; then
    units_to_install+=("$MINION_WORKER_UNIT")
fi

echo "Generating systemd units:"
echo "  JARVIS_ROOT = $REPO_ROOT"
echo "  LOG_DIR     = $LOG_DIR"
echo "  USER        = $USER"
echo "  OUT_DIR     = $OUT_DIR"
[[ "$USER_SCOPE" == "1" ]] && echo "  scope       = user" || echo "  scope       = system"
echo

for unit in "${units_to_install[@]}"; do
    template="$TEMPLATE_DIR/$unit"
    target="$OUT_DIR/$unit"

    if [[ ! -f "$template" ]]; then
        echo "ERROR: template missing: $template" >&2
        exit 1
    fi

    tmp="$(mktemp)"
    sed \
        -e "s|{{JARVIS_ROOT}}|$REPO_ROOT|g" \
        -e "s|{{LOG_DIR}}|$LOG_DIR|g" \
        -e "s|{{USER}}|$USER|g" \
        "$template" > "$tmp"
    $SUDO mv "$tmp" "$target"
    $SUDO chmod 644 "$target"
    echo "  ✓ wrote $target"
done

$SUDO $SYSTEMCTL daemon-reload

echo
echo "Next — enable + start the timers:"
echo "  $SUDO $SYSTEMCTL enable --now jarvis-compact-daily.timer"
echo "  $SUDO $SYSTEMCTL enable --now jarvis-compact-weekly.timer"
if [[ "$WITH_MINION_WORKER" == "1" ]]; then
    echo "  $SUDO $SYSTEMCTL enable --now jarvis-minion-worker.service"
fi
echo
echo "To remove later: scripts/generate_systemd_units.sh --uninstall"
