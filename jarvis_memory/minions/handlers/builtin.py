"""Built-in Minion handlers.

Three handlers register themselves on import:
  - ``echo``          — returns its params dict verbatim. Used by tests
                        and as a sentinel for "is the worker up?".
  - ``compact_daily`` — subprocess-spawns ``scripts/run_compaction.py
                        --tier daily`` and captures exit code + stdout/
                        stderr tail.
  - ``compact_weekly`` — same, ``--tier weekly``.

The compaction handlers parallel-operate alongside the existing launchd
compaction plists. Cutover is a later run once Minions has been running
stably for at least a week.

These handlers are UNTRUSTED by default — they run ``python`` as a
subprocess from a known path (the project venv + ``scripts/run_compaction.py``).
They do NOT execute arbitrary user-provided commands. The real shell
handler (``shell.py``) is gated behind ``GBRAIN_ALLOW_SHELL_JOBS=1``.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import register_handler

logger = logging.getLogger(__name__)


# Repo root is two levels up from this file:
#   jarvis_memory/minions/handlers/builtin.py
#     → jarvis_memory/minions/handlers/
#     → jarvis_memory/minions/
#     → jarvis_memory/
#     → <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPACTION_SCRIPT = PROJECT_ROOT / "scripts" / "run_compaction.py"

# Cap captured stdout/stderr to a few KB each so a runaway script doesn't
# bloat the result JSON.
_TAIL_BYTES = 4096


def echo(params: dict[str, Any]) -> dict[str, Any]:
    """Return ``params`` verbatim. Wrapped in a handler signature so workers
    have something to exercise without external side effects."""
    if not isinstance(params, dict):
        raise TypeError(f"echo expects dict params, got {type(params).__name__}")
    return {"echoed": params}


def _tail(text: str, limit: int = _TAIL_BYTES) -> str:
    """Return the last ``limit`` bytes of ``text``, safely UTF-8 boundary-aligned."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    # Drop from start; decode-align by stripping the leading possibly-broken char.
    snippet = text[-limit:]
    # If we've landed mid-multibyte-sequence, Python's str can't represent that
    # (it's already decoded). So snippet is safe. Prefix with marker.
    return "...[truncated]..." + snippet


def _run_compaction(tier: str, params: dict[str, Any]) -> dict[str, Any]:
    """Invoke ``scripts/run_compaction.py --tier <tier>`` and return structured result.

    Parameters (all optional):
      - ``group_id``: restrict compaction to one project.
      - ``python_path``: override the interpreter (default: sys.executable).
      - ``timeout_seconds``: subprocess wait bound (default 1800 = 30 min).

    Return shape:
      {
        "tier": "daily|weekly",
        "exit_code": int,
        "stdout_tail": str,
        "stderr_tail": str,
        "cmd": [...]
      }
    """
    if tier not in {"daily", "weekly"}:
        raise ValueError(f"tier must be 'daily' or 'weekly', got {tier!r}")
    if not COMPACTION_SCRIPT.exists():
        raise FileNotFoundError(f"compaction script not found: {COMPACTION_SCRIPT}")

    python_path = params.get("python_path") or sys.executable
    timeout_seconds = int(params.get("timeout_seconds", 1800))
    cmd: list[str] = [python_path, str(COMPACTION_SCRIPT), "--tier", tier]
    group_id = params.get("group_id")
    if group_id:
        cmd.extend(["--group-id", str(group_id)])

    logger.info("compact_%s invoking %s", tier, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "tier": tier,
            "exit_code": -1,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
            "cmd": cmd,
            "error": f"timeout after {timeout_seconds}s",
        }

    result: dict[str, Any] = {
        "tier": tier,
        "exit_code": proc.returncode,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
        "cmd": cmd,
    }

    # Best-effort parse of the last JSON line on stdout (run_compaction.py
    # prints a single-line JSON summary as its final output).
    tail_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if tail_line.startswith("{"):
        try:
            result["parsed"] = json.loads(tail_line)
        except json.JSONDecodeError:
            pass
    return result


def compact_daily(params: dict[str, Any]) -> dict[str, Any]:
    """Run the daily compaction tier via ``scripts/run_compaction.py``."""
    return _run_compaction("daily", params or {})


def compact_weekly(params: dict[str, Any]) -> dict[str, Any]:
    """Run the weekly compaction tier via ``scripts/run_compaction.py``."""
    return _run_compaction("weekly", params or {})


# ── Auto-registration on import ─────────────────────────────────────────
#
# Registering at import time means ``from jarvis_memory.minions.handlers
# import builtin`` is sufficient to make all three names available to the
# worker. The registry module's duplicate-name check keeps re-imports
# safe (overwrite=True is passed so tests that clear + re-import work).

register_handler("echo", echo, overwrite=True)
register_handler("compact_daily", compact_daily, overwrite=True)
register_handler("compact_weekly", compact_weekly, overwrite=True)


__all__ = [
    "echo",
    "compact_daily",
    "compact_weekly",
]
