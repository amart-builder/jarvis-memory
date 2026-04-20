"""Weekly-rotated JSONL audit trail for shell-job submissions.

One line per submission. Rotates on ISO-week boundary (filename encodes
``YYYY-Www``). Does NOT log env values — only env KEYS and whether they
came from the caller-supplied overrides.

Filesystem layout::

    ~/Atlas/jarvis-memory/audit/
        shell-jobs-2026-W17.jsonl       # current week
        shell-jobs-2026-W16.jsonl       # last week
        ...

Override the base directory via ``GBRAIN_AUDIT_DIR`` — tests set this to a
tmp path so the real home dir is untouched.

Entry shape::

    {
      "ts":        "2026-04-20T00:00:00+00:00",
      "job_id":    "uuid",
      "caller":    "worker-ab12cd34",
      "mode":      "shell|argv",        # shell=/bin/sh -c, argv=direct spawn
      "cmd":       "echo hello"[:80],   # present iff mode=shell
      "argv":      ["/bin/echo","hi"]   # present iff mode=argv (trimmed to 20)
      "env_keys":  ["PATH","HOME",...], # just the keys, never values
      "timeout_s": 60,
      "params_hash": "sha256..."        # for later correlation
    }

Why JSONL: cheap to tail-append, trivial to parse with ``jq``, and each line
is self-contained so a mid-write crash doesn't corrupt the file (only the
partial last line is lost).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_DIR = Path.home() / "Atlas" / "jarvis-memory" / "audit"
CMD_CHAR_LIMIT = 80
ARGV_ELEMENT_LIMIT = 20
ARGV_PER_ELEMENT_LIMIT = 80


def _audit_dir() -> Path:
    """Resolve the audit base directory.

    Tests monkeypatch ``GBRAIN_AUDIT_DIR``. Falls back to
    ``~/Atlas/jarvis-memory/audit/``.
    """
    override = os.environ.get("GBRAIN_AUDIT_DIR")
    return Path(override) if override else DEFAULT_AUDIT_DIR


def _current_iso_week(ts: Optional[datetime] = None) -> str:
    """Return ``YYYY-Www`` for the given timestamp (default: now UTC).

    Uses ISO week (week 1 is the week containing the first Thursday).
    ``date.isocalendar()`` is stable and matches what ``jq`` / `strftime`
    ``%G-W%V`` would produce.
    """
    ts = ts or datetime.now(timezone.utc)
    iso_year, iso_week, _ = ts.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _audit_path(ts: Optional[datetime] = None) -> Path:
    return _audit_dir() / f"shell-jobs-{_current_iso_week(ts)}.jsonl"


def _truncate_cmd(cmd: str) -> str:
    if len(cmd) <= CMD_CHAR_LIMIT:
        return cmd
    return cmd[: CMD_CHAR_LIMIT - 3] + "..."


def _truncate_argv(argv: Sequence[str]) -> list[str]:
    trimmed = list(argv[:ARGV_ELEMENT_LIMIT])
    if len(argv) > ARGV_ELEMENT_LIMIT:
        trimmed.append(f"...[+{len(argv) - ARGV_ELEMENT_LIMIT} more]")
    return [
        a[:ARGV_PER_ELEMENT_LIMIT] if isinstance(a, str) else str(a)[:ARGV_PER_ELEMENT_LIMIT]
        for a in trimmed
    ]


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_audit_entry(
    cmd_or_argv: Union[str, Sequence[str]],
    *,
    caller: str,
    job_id: str,
    env_keys: Optional[Sequence[str]] = None,
    timeout_seconds: int,
    params: Optional[dict[str, Any]] = None,
    ts: Optional[datetime] = None,
) -> Path:
    """Append a single JSONL entry for a shell-job submission.

    Returns the path of the file that was appended to (useful for tests).
    Creates the audit directory if it doesn't exist.
    """
    ts = ts or datetime.now(timezone.utc)

    if isinstance(cmd_or_argv, str):
        mode = "shell"
        cmd_field: dict[str, Any] = {"cmd": _truncate_cmd(cmd_or_argv)}
    else:
        mode = "argv"
        cmd_field = {"argv": _truncate_argv(cmd_or_argv)}

    entry: dict[str, Any] = {
        "ts": ts.isoformat(),
        "job_id": job_id,
        "caller": caller,
        "mode": mode,
        **cmd_field,
        "env_keys": sorted(list(env_keys or [])),
        "timeout_s": int(timeout_seconds),
        "params_hash": _hash_params(params or {}),
    }

    path = _audit_path(ts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError as exc:
        # Don't block a shell job on an audit-write failure, but DO log it
        # loudly so a supervisor can tell. (Callers typically don't check
        # the return value — they pass in just to get the side effect.)
        logger.error("shell audit write failed to %s: %s", path, exc)
        raise
    return path


def iter_audit_entries(week: Optional[str] = None) -> list[dict[str, Any]]:
    """Read + parse all entries for a given week (or current week).

    ``week`` is a ``YYYY-Www`` string. Returns an empty list if the file
    doesn't exist.
    """
    if week is None:
        week = _current_iso_week()
    path = _audit_dir() / f"shell-jobs-{week}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed audit line in %s", path)
    return out


__all__ = [
    "CMD_CHAR_LIMIT",
    "ARGV_ELEMENT_LIMIT",
    "DEFAULT_AUDIT_DIR",
    "append_audit_entry",
    "iter_audit_entries",
]
