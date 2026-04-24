"""Register (or unregister) Jarvis-Memory as an MCP server in Claude Code and Codex CLI.

Both tools speak standard MCP over stdio, so one server (``jarvis-mcp``) works for
both — only the config-file format differs. This script handles both cleanly,
merging into existing user config rather than overwriting it.

Usage:
    python scripts/register_mcp.py --client claude-code
    python scripts/register_mcp.py --client codex
    python scripts/register_mcp.py --client claude-code --uninstall

Paths that get edited:
    Claude Code : ~/.claude/settings.json         (JSON, merged under mcpServers.jarvis-memory)
    Codex       : ~/.codex/config.toml            (TOML, appended as [mcp_servers.jarvis-memory])

Idempotent — re-running is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_BIN_MCP = REPO_ROOT / ".venv" / "bin" / "jarvis-mcp"

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
CODEX_CONFIG = Path.home() / ".codex" / "config.toml"

SERVER_NAME = "jarvis-memory"
# Marker placed in the Codex TOML comment block so we can uninstall later
# without a full TOML parser.
CODEX_BEGIN = f"# BEGIN jarvis-memory mcp registration"
CODEX_END = f"# END jarvis-memory mcp registration"


def mcp_command() -> str:
    """Return the command to launch jarvis-mcp.

    Prefer the console script inside the repo's .venv (absolute path);
    fall back to ``jarvis-mcp`` on PATH if the venv binary doesn't exist
    (e.g. development install).
    """
    if VENV_BIN_MCP.exists():
        return str(VENV_BIN_MCP)
    return "jarvis-mcp"


# ───────────────────── Claude Code (JSON) ─────────────────────

def claude_install() -> None:
    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    if CLAUDE_SETTINGS.exists():
        try:
            settings = json.loads(CLAUDE_SETTINGS.read_text())
        except json.JSONDecodeError as e:
            print(f"ERROR: {CLAUDE_SETTINGS} is not valid JSON: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        settings = {}

    settings.setdefault("mcpServers", {})
    settings["mcpServers"][SERVER_NAME] = {
        "command": mcp_command(),
        "args": [],
        # cwd lets the server find its .env file.
        "cwd": str(REPO_ROOT),
    }
    CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2))
    print(f"✓ Registered '{SERVER_NAME}' in {CLAUDE_SETTINGS}")
    print(f"  command: {mcp_command()}")
    print(f"  cwd:     {REPO_ROOT}")
    print(f"  Restart Claude Code to pick up the new server.")


def claude_uninstall() -> None:
    if not CLAUDE_SETTINGS.exists():
        print(f"Nothing to remove — {CLAUDE_SETTINGS} does not exist.")
        return
    try:
        settings = json.loads(CLAUDE_SETTINGS.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: {CLAUDE_SETTINGS} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)
    servers = settings.get("mcpServers", {})
    if SERVER_NAME in servers:
        del servers[SERVER_NAME]
        if not servers:
            settings.pop("mcpServers", None)
        CLAUDE_SETTINGS.write_text(json.dumps(settings, indent=2))
        print(f"✓ Removed '{SERVER_NAME}' from {CLAUDE_SETTINGS}")
    else:
        print(f"'{SERVER_NAME}' not found in {CLAUDE_SETTINGS} — nothing to remove.")


# ───────────────────── Codex CLI (TOML) ─────────────────────
#
# We don't pull in a TOML-writer lib (would add a dep); we surround our block
# with BEGIN/END comment markers and edit the file as text. This is fragile
# only if the user manually edits inside our markers, which we don't expect.

def codex_block() -> str:
    cmd = mcp_command()
    return f"""\
{CODEX_BEGIN}
[mcp_servers.{SERVER_NAME}]
command = "{cmd}"
args = []
# Working directory for the server (so it finds its .env file).
cwd = "{REPO_ROOT}"
{CODEX_END}
"""


def codex_install() -> None:
    CODEX_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    existing = CODEX_CONFIG.read_text() if CODEX_CONFIG.exists() else ""

    # Replace existing block if present, else append.
    pattern = re.compile(
        re.escape(CODEX_BEGIN) + r".*?" + re.escape(CODEX_END) + r"\n?",
        flags=re.DOTALL,
    )
    block = codex_block()
    if pattern.search(existing):
        new = pattern.sub(block, existing)
    else:
        sep = "" if not existing or existing.endswith("\n") else "\n"
        new = existing + sep + "\n" + block

    CODEX_CONFIG.write_text(new)
    print(f"✓ Registered '{SERVER_NAME}' in {CODEX_CONFIG}")
    print(f"  command: {mcp_command()}")
    print(f"  cwd:     {REPO_ROOT}")
    print(f"  Restart Codex to pick up the new server.")


def codex_uninstall() -> None:
    if not CODEX_CONFIG.exists():
        print(f"Nothing to remove — {CODEX_CONFIG} does not exist.")
        return
    existing = CODEX_CONFIG.read_text()
    pattern = re.compile(
        r"\n?" + re.escape(CODEX_BEGIN) + r".*?" + re.escape(CODEX_END) + r"\n?",
        flags=re.DOTALL,
    )
    new = pattern.sub("", existing)
    if new == existing:
        print(f"'{SERVER_NAME}' block not found in {CODEX_CONFIG} — nothing to remove.")
        return
    CODEX_CONFIG.write_text(new)
    print(f"✓ Removed '{SERVER_NAME}' block from {CODEX_CONFIG}")


# ───────────────────── CLI ─────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--client",
        required=True,
        choices=["claude-code", "codex"],
        help="Which client's config to edit.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the registration instead of adding it.",
    )
    args = parser.parse_args()

    if args.client == "claude-code":
        (claude_uninstall if args.uninstall else claude_install)()
    else:
        (codex_uninstall if args.uninstall else codex_install)()


if __name__ == "__main__":
    main()
