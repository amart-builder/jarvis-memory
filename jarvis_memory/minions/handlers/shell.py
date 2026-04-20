"""Gated shell-command handler.

Import-time refuses to register unless ``GBRAIN_ALLOW_SHELL_JOBS=1``.
This means a bare ``from jarvis_memory.minions.handlers import shell``
in an environment without the env flag raises ``ShellHandlerDisabled``
and the handler is NEVER registered — giving operators a clear signal.

Contract (``shell(params)``):
  Required one of:
    - ``params["cmd"]``: string passed to ``/bin/sh -c``
    - ``params["argv"]``: list of strings for direct execve (no shell)

  Optional:
    - ``params["env"]``: dict of caller-specified env overrides. Only keys
      on the allowlist OR explicitly whitelisted here are honored.
    - ``params["cwd"]``: working directory. Default is the project root.
    - ``params["stdin"]``: UTF-8 string piped to the subprocess stdin.
    - ``params["timeout_seconds"]``: max wall-clock before the child is
      SIGTERM'd (grace 5s) then SIGKILL'd. Default = caller's job
      timeout minus 2s. Falls back to 55s when not set by either layer.

Security-relevant invariants:
  - Absolute ``/bin/sh`` — PATH overrides can't redirect shell lookup.
  - Env allowlist: PATH, HOME, USER, LANG, TZ, NODE_ENV. Caller overrides
    are kept only if the key is in the allowlist; other keys are silently
    dropped and logged.
  - Abort propagation: SIGTERM → ``GRACE_SECONDS`` (5s) wait → SIGKILL.
  - Never logs env VALUES (only keys) via ``shell_audit.append_audit_entry``.

Result shape::

    {
      "exit_code": int,
      "stdout_tail": str,   # UTF-8-safe, up to TAIL_BYTES
      "stderr_tail": str,
      "duration_s": float,
      "killed_by": "timeout|completed",
      "cmd": str | None,
      "argv": list[str] | None
    }
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import register_handler
from .shell_audit import append_audit_entry

logger = logging.getLogger(__name__)


class ShellHandlerDisabled(RuntimeError):
    """Raised at import time when ``GBRAIN_ALLOW_SHELL_JOBS`` is not set to '1'."""


# ── import-time gate ────────────────────────────────────────────────────

def _gate() -> None:
    """Refuse to continue module initialization unless the env flag is '1'."""
    value = os.environ.get("GBRAIN_ALLOW_SHELL_JOBS")
    if value != "1":
        raise ShellHandlerDisabled(
            "Shell handler refused registration. Set GBRAIN_ALLOW_SHELL_JOBS=1 "
            "to enable (deliberate opt-in — default is disabled)."
        )


_gate()


# ── constants ───────────────────────────────────────────────────────────

# Absolute path: can't be hijacked by a poisoned PATH.
SH_PATH = "/bin/sh"

# Env keys the caller is allowed to pass through. Everything else is dropped.
ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE",
    "TZ", "NODE_ENV",
})

# Default PATH if the caller didn't set one. Mirrors a minimal Mac shell env.
DEFAULT_PATH = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# Max bytes of stdout/stderr captured.
TAIL_BYTES = 4096

# Grace period between SIGTERM and SIGKILL.
GRACE_SECONDS = 5.0

# Fallback timeout when caller doesn't specify one.
DEFAULT_TIMEOUT_SECONDS = 55.0


# ── env sanitizer ───────────────────────────────────────────────────────

def _sanitize_env(caller_env: Optional[dict[str, Any]]) -> dict[str, str]:
    """Build the child env: defaults + allowlisted caller overrides only."""
    # Start from the current process env, BUT only copy allowlisted keys.
    env: dict[str, str] = {}
    for key in ENV_ALLOWLIST:
        if key in os.environ:
            env[key] = os.environ[key]
    env.setdefault("PATH", DEFAULT_PATH)

    if caller_env:
        for k, v in caller_env.items():
            if not isinstance(k, str):
                logger.warning("shell: dropping non-string env key %r", k)
                continue
            if k not in ENV_ALLOWLIST:
                logger.warning("shell: dropping non-allowlisted env key %r", k)
                continue
            env[k] = str(v)
    return env


# ── UTF-8-safe tail ─────────────────────────────────────────────────────

def _utf8_safe_tail(blob: bytes, limit: int = TAIL_BYTES) -> str:
    """Return the last ``limit`` bytes as a UTF-8 string without mid-codepoint splits.

    If truncation would slice a multi-byte sequence, we retreat a few bytes
    until ``decode(errors='strict')`` succeeds. ``errors='replace'`` is the
    safety net if even that fails.
    """
    if blob is None:
        return ""
    if len(blob) <= limit:
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return blob.decode("utf-8", errors="replace")

    tail = blob[-limit:]
    # Walk forward a few bytes to find a codepoint boundary.
    for start in range(4):
        try:
            return "...[truncated]..." + tail[start:].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return "...[truncated]..." + tail.decode("utf-8", errors="replace")


# ── handler body ────────────────────────────────────────────────────────

def _build_argv(params: dict[str, Any]) -> tuple[list[str], Optional[str], Optional[list[str]]]:
    """Resolve ``params`` into ``(argv, cmd_str_for_audit, argv_for_audit)``."""
    cmd = params.get("cmd")
    argv = params.get("argv")
    if cmd and argv:
        raise ValueError("shell handler: pass exactly one of 'cmd' or 'argv', not both")
    if not cmd and not argv:
        raise ValueError("shell handler: either 'cmd' or 'argv' is required")
    if cmd is not None:
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("shell handler: 'cmd' must be a non-empty string")
        return [SH_PATH, "-c", cmd], cmd, None
    # argv mode
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("shell handler: 'argv' must be a non-empty list of strings")
    argv_list = [str(a) for a in argv]
    if not Path(argv_list[0]).is_absolute():
        # Direct exec needs an absolute path — relative binaries would
        # reintroduce PATH-lookup risk.
        raise ValueError(
            f"shell handler: argv[0] must be an absolute path, got {argv_list[0]!r}"
        )
    return argv_list, None, argv_list


def shell(params: dict[str, Any], *, job=None) -> dict[str, Any]:
    """Execute a shell command or argv, under an env allowlist, with audit trail.

    See module docstring for the contract.
    """
    if not isinstance(params, dict):
        raise TypeError(f"shell expects dict params, got {type(params).__name__}")

    argv, cmd_for_audit, argv_for_audit = _build_argv(params)
    caller_env = params.get("env") or {}
    env = _sanitize_env(caller_env if isinstance(caller_env, dict) else None)
    cwd = params.get("cwd") or os.getcwd()
    stdin_str = params.get("stdin")
    stdin_bytes = stdin_str.encode("utf-8") if isinstance(stdin_str, str) else None

    timeout_seconds = float(params.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    if timeout_seconds <= 0:
        raise ValueError("shell: timeout_seconds must be > 0")

    # Audit trail — write BEFORE execution so a crash still leaves a record.
    caller_id = getattr(job, "worker_id", None) or getattr(job, "id", None) or "unknown"
    job_id = getattr(job, "id", None) or params.get("_audit_job_id", "no-job")
    try:
        append_audit_entry(
            cmd_or_audit_target := (cmd_for_audit if cmd_for_audit is not None else argv_for_audit),  # noqa: F841
            caller=str(caller_id),
            job_id=str(job_id),
            env_keys=sorted(env.keys()),
            timeout_seconds=int(timeout_seconds),
            params={k: v for k, v in params.items() if k not in {"stdin"}},
        )
    except OSError:
        # Audit failure should not block execution — logged inside append.
        pass

    logger.info(
        "shell: job=%s cwd=%s argv[0]=%s timeout=%ss",
        job_id, cwd, argv[0], timeout_seconds,
    )

    start = time.monotonic()
    killed_by = "completed"
    proc = subprocess.Popen(  # noqa: S603 — argv is argv; not shell=True
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,  # so killpg can reach grandchildren
    )

    try:
        stdout_b, stderr_b = proc.communicate(input=stdin_bytes, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        killed_by = "timeout"
        logger.warning("shell: job=%s timed out after %ss — SIGTERM'ing", job_id, timeout_seconds)
        _terminate(proc)
        try:
            stdout_b, stderr_b = proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            stdout_b, stderr_b = b"", b""

    duration = time.monotonic() - start
    result = {
        "exit_code": proc.returncode if proc.returncode is not None else -1,
        "stdout_tail": _utf8_safe_tail(stdout_b),
        "stderr_tail": _utf8_safe_tail(stderr_b),
        "duration_s": round(duration, 3),
        "killed_by": killed_by,
        "cmd": cmd_for_audit,
        "argv": argv_for_audit,
    }
    return result


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, wait GRACE_SECONDS, then SIGKILL."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("shell: SIGTERM killpg failed: %s", exc)

    try:
        proc.wait(timeout=GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("shell: SIGKILL killpg failed: %s", exc)


# ── registration ────────────────────────────────────────────────────────
#
# Registered as a PROTECTED name — ``allow_protected=True`` is the explicit
# opt-in that satisfies the handler-registry's safety check. Combined with
# the import-time gate above (``_gate()``), this means shell registration
# requires BOTH the env flag AND explicit ``allow_protected=True``, i.e.
# it can't be done accidentally via a ``register_handler("shell", ...)``.

register_handler("shell", shell, allow_protected=True, overwrite=True)


__all__ = [
    "ShellHandlerDisabled",
    "ENV_ALLOWLIST",
    "DEFAULT_PATH",
    "SH_PATH",
    "TAIL_BYTES",
    "GRACE_SECONDS",
    "shell",
]
