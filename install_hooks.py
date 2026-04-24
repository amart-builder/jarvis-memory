"""Install Claude Code hooks for jarvis-memory.

Registers two hooks in ``~/.claude/settings.json``:

  - ``SessionStart`` (startup + resume) → ``claude_code_sessionstart.py``
    Injects recent context from jarvis-memory at the start of each
    Claude Code session, so the agent picks up where the last run left off.

  - ``PreCompact`` → ``claude_code_precompact.py``
    Writes a [HANDOFF] episode before Claude Code auto-compacts, preserving
    the conversation's key state for the next session.

Paths are resolved dynamically from this script's location, so the
registration works no matter where the jarvis-memory repo lives on disk.

Usage:
    python3 install_hooks.py                # install / update
    python3 install_hooks.py --uninstall    # remove jarvis-memory hooks

The script is idempotent: safe to run multiple times.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve paths based on this file's location
REPO_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SESSIONSTART_HOOK = REPO_ROOT / "hooks" / "claude_code_sessionstart.py"
PRECOMPACT_HOOK = REPO_ROOT / "hooks" / "claude_code_precompact.py"

CLAUDE_DIR = Path.home() / ".claude"
SETTINGS_PATH = CLAUDE_DIR / "settings.json"

# Marker that identifies hooks installed by this script — lets us find and
# remove them on uninstall without disturbing user-added hooks.
HOOK_MARKER = "__jarvis_memory__"


def build_hook_entries() -> dict:
    """Build the hook entries for ~/.claude/settings.json."""
    # Use the venv python if available, otherwise fall back to system python3.
    python_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python3"

    sessionstart_cmd = f"{python_bin} {SESSIONSTART_HOOK}"
    precompact_cmd = f"{python_bin} {PRECOMPACT_HOOK}"

    return {
        "SessionStart": [
            {
                "matcher": "startup",
                HOOK_MARKER: True,
                "hooks": [
                    {"type": "command", "command": sessionstart_cmd, "timeout": 10}
                ],
            },
            {
                "matcher": "resume",
                HOOK_MARKER: True,
                "hooks": [
                    {"type": "command", "command": sessionstart_cmd, "timeout": 10}
                ],
            },
        ],
        "PreCompact": [
            {
                HOOK_MARKER: True,
                "hooks": [
                    {"type": "command", "command": precompact_cmd, "timeout": 15}
                ],
            },
        ],
    }


def load_settings() -> dict:
    """Load existing Claude Code settings, or return an empty skeleton."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: {SETTINGS_PATH} is not valid JSON: {e}", file=sys.stderr)
        print("Fix the file by hand before re-running this installer.", file=sys.stderr)
        sys.exit(2)


def save_settings(settings: dict) -> None:
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def strip_our_hooks(settings: dict) -> dict:
    """Remove any hook entries we previously installed (marked with HOOK_MARKER)."""
    hooks = settings.get("hooks", {})
    for event_name in list(hooks.keys()):
        hooks[event_name] = [h for h in hooks[event_name] if not h.get(HOOK_MARKER)]
        if not hooks[event_name]:
            del hooks[event_name]
    if not hooks:
        settings.pop("hooks", None)
    return settings


def install() -> None:
    if not SESSIONSTART_HOOK.exists() or not PRECOMPACT_HOOK.exists():
        print(
            f"ERROR: hook scripts missing under {REPO_ROOT / 'hooks'}\n"
            f"Expected:\n  {SESSIONSTART_HOOK}\n  {PRECOMPACT_HOOK}",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = load_settings()
    settings = strip_our_hooks(settings)  # idempotent

    new_hooks = build_hook_entries()
    existing_hooks = settings.setdefault("hooks", {})
    for event_name, entries in new_hooks.items():
        existing_hooks.setdefault(event_name, []).extend(entries)

    save_settings(settings)

    python_status = "venv" if VENV_PYTHON.exists() else "system python3"
    print(f"✓ Hooks installed in {SETTINGS_PATH}")
    print(f"  SessionStart → {SESSIONSTART_HOOK.name} ({python_status})")
    print(f"  PreCompact   → {PRECOMPACT_HOOK.name} ({python_status})")
    print()
    print("Start a new Claude Code session to activate the hooks.")


def uninstall() -> None:
    settings = load_settings()
    if not settings.get("hooks"):
        print("No hooks found in settings.json — nothing to uninstall.")
        return

    settings = strip_our_hooks(settings)
    save_settings(settings)
    print(f"✓ jarvis-memory hooks removed from {SETTINGS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove jarvis-memory hooks from ~/.claude/settings.json",
    )
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
    else:
        install()


if __name__ == "__main__":
    main()
