"""Protected-name constants for the Minions handler registry.

Pure module. Imports nothing from other handler modules. This is deliberate:
if a malicious handler module could trigger registration at import time, we
don't want it to be able to sneak past the protected-name check by
importing and mutating state. ``protected_names`` is a constant
dependency-free source of truth.

The protected set covers names that would let an attacker escalate privilege
if registered without ``allow_protected=True``:

  - ``shell``, ``shell-exec`` — shell-command execution.
  - ``system``              — OS-level catch-all.
  - ``eval``                — arbitrary-code evaluation.
  - ``exec``                — arbitrary-code execution.
  - ``subprocess``          — direct subprocess spawn.
  - ``bash``, ``sh``, ``zsh`` — shell aliases.
  - ``powershell``          — Windows shell.
  - ``python``, ``python3`` — interpreter invocation.
  - ``command``             — generic command.
  - ``script``              — script runner.

Match policy (enforced by ``is_protected_job_name``):
  1. ``name.strip()`` — leading/trailing whitespace and control chars are ignored.
  2. Case-folded comparison — ``"SHELL"`` matches ``"shell"``.
  3. Underscores/hyphens/spaces normalized to a single hyphen — ``"shell_exec"``,
     ``"shell exec"``, and ``"shell--exec"`` all match ``"shell-exec"``.

This makes whitespace-bypass and case-variation attacks impossible via the
registration path.
"""
from __future__ import annotations

import re
from typing import Final


PROTECTED_JOB_NAMES: Final[frozenset[str]] = frozenset({
    "shell",
    "shell-exec",
    "system",
    "eval",
    "exec",
    "subprocess",
    "bash",
    "sh",
    "zsh",
    "powershell",
    "python",
    "python3",
    "command",
    "script",
})


# Single expression to collapse runs of underscores, hyphens, whitespace.
_NORMALIZE_RE = re.compile(r"[\s_\-]+")


def _normalize(name: str) -> str:
    """Lowercase + strip + collapse separators to a single hyphen.

    Kept as a private helper so callers don't accidentally bypass
    normalization by comparing raw strings.
    """
    if not isinstance(name, str):
        return ""
    stripped = name.strip()
    lowered = stripped.casefold()
    return _NORMALIZE_RE.sub("-", lowered)


# Pre-computed normalized set for O(1) membership.
_NORMALIZED_PROTECTED: Final[frozenset[str]] = frozenset(
    _normalize(n) for n in PROTECTED_JOB_NAMES
)


def is_protected_job_name(name: str) -> bool:
    """Return True if ``name`` normalizes to a protected job name.

    Whitespace-bypass safe, case-bypass safe, separator-normalized.
    """
    if not name:
        return False
    return _normalize(name) in _NORMALIZED_PROTECTED


__all__ = [
    "PROTECTED_JOB_NAMES",
    "is_protected_job_name",
]
