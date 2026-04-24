#!/usr/bin/env bash
#
# upgrade.sh — pull the latest Jarvis-Memory release + re-verify.
#
# Default behavior: fetch tags, check out the latest v*.*.* tag, re-run
# the install script in --yes mode (skipping interactive prompts), then
# run verify_install.sh. If anything fails, exits non-zero so the caller
# (e.g. a cron or an agent) can react.
#
# Usage:
#   scripts/upgrade.sh                  # check out latest stable tag
#   scripts/upgrade.sh --rolling        # check out origin/main instead of a tag
#   scripts/upgrade.sh --to vX.Y.Z      # check out a specific tag
#
# The .env file and any local data (ChromaDB, SQLite queue, logs) are
# never touched.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROLLING=0
PIN_TAG=""
for arg in "$@"; do
    case "$arg" in
        --rolling) ROLLING=1 ;;
        --to)      shift; PIN_TAG="${1:?--to requires a tag argument}" ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown arg: $arg (--help for usage)" >&2; exit 2 ;;
    esac
done

echo "▶ Fetching from origin"
git fetch origin --tags --prune

# Decide what to check out.
if [[ -n "$PIN_TAG" ]]; then
    TARGET="$PIN_TAG"
elif [[ "$ROLLING" == "1" ]]; then
    TARGET="origin/main"
else
    TARGET="$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n 1)"
    if [[ -z "$TARGET" ]]; then
        echo "No release tags found. Use --rolling or --to <tag>." >&2
        exit 1
    fi
fi

# Check for local uncommitted changes — don't overwrite without asking.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree has uncommitted changes. Commit or stash first." >&2
    git status --short >&2
    exit 1
fi

CURRENT="$(git rev-parse HEAD)"
RESOLVED="$(git rev-parse "$TARGET")"

if [[ "$CURRENT" == "$RESOLVED" ]]; then
    echo "✓ Already at $TARGET ($(git describe --tags --always)) — nothing to do."
    exit 0
fi

echo "▶ Checking out $TARGET ($(git log -1 --format='%h %s' "$RESOLVED"))"
git checkout --detach "$TARGET"

echo "▶ Re-running install in non-interactive mode"
bash "$REPO_ROOT/scripts/client-install.sh" --yes --no-schedule --no-hooks

echo "▶ Verifying"
bash "$REPO_ROOT/scripts/verify_install.sh"

echo "✓ Upgrade complete — now at $(git describe --tags --always)"
