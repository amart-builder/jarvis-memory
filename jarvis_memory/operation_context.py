"""OperationContext — trust-boundary marker for jarvis-memory writes.

Every mutation path (REST, MCP, CLI, internal) can accept an optional
``ctx: OperationContext`` arg. The context carries:

  - ``remote``: True if this came in over a remote-accessible surface
    (MCP or REST), False for local CLI / direct Python / tests.
  - ``source``: one of ``"cli" | "mcp" | "rest"``.
  - ``caller``: a string identifier for logging (the MCP agent id, the
    REST request IP, or ``$USER`` for CLI). Best-effort; never
    authoritative for authz in this run.

This run is LOGGED-ONLY: downstream code that receives a remote=True
context with suspicious content emits a WARNING but NEVER refuses. A
future run will add refusal mode once we have log data about what
``abuse'' actually looks like in practice.

Backward compatibility: every caller that doesn't know about
``OperationContext`` can keep calling functions without a ``ctx`` arg —
the receiver treats ``None`` as "unknown / assume-local" and skips the
audit checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SourceLiteral = Literal["cli", "mcp", "rest"]


@dataclass(frozen=True)
class OperationContext:
    """Marker for where a request originated.

    Use the classmethod constructors (``for_cli()``, ``for_mcp()``,
    ``for_rest()``) rather than constructing directly — they set
    ``remote`` correctly for each surface.
    """

    remote: bool = False
    caller: str = "local"
    source: SourceLiteral = "cli"

    # ── Constructors for each surface ──────────────────────────────────

    @classmethod
    def for_cli(cls, caller: Optional[str] = None) -> "OperationContext":
        """Local CLI (including direct Python). ``remote=False``."""
        import os

        return cls(
            remote=False,
            caller=caller or os.environ.get("USER", "local"),
            source="cli",
        )

    @classmethod
    def for_mcp(cls, caller: Optional[str] = None) -> "OperationContext":
        """Agent-facing MCP tool invocation. ``remote=True``."""
        return cls(
            remote=True,
            caller=caller or "mcp-agent",
            source="mcp",
        )

    @classmethod
    def for_rest(cls, caller: Optional[str] = None) -> "OperationContext":
        """Remote REST API call. ``remote=True``. ``caller`` is typically the
        request's client IP (set by the endpoint)."""
        return cls(
            remote=True,
            caller=caller or "rest-unknown",
            source="rest",
        )

    # ── Predicates ─────────────────────────────────────────────────────

    def is_trusted(self) -> bool:
        """Convenience: ``not remote`` — local CLI and internal calls are trusted."""
        return not self.remote


__all__ = ["OperationContext", "SourceLiteral"]
