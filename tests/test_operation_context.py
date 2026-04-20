"""OperationContext tests — constructors, predicates, immutability."""
from __future__ import annotations

import dataclasses
import os

import pytest

from jarvis_memory.operation_context import OperationContext


class TestDefaults:
    def test_default_is_local_cli(self):
        ctx = OperationContext()
        assert ctx.remote is False
        assert ctx.source == "cli"
        assert ctx.caller == "local"

    def test_is_trusted_when_not_remote(self):
        assert OperationContext(remote=False).is_trusted() is True

    def test_not_trusted_when_remote(self):
        assert OperationContext(remote=True).is_trusted() is False


class TestConstructors:
    def test_for_cli_respects_user_env(self, monkeypatch):
        monkeypatch.setenv("USER", "alex")
        ctx = OperationContext.for_cli()
        assert ctx.remote is False
        assert ctx.source == "cli"
        assert ctx.caller == "alex"

    def test_for_cli_accepts_explicit_caller(self):
        ctx = OperationContext.for_cli("test-runner")
        assert ctx.caller == "test-runner"
        assert ctx.remote is False

    def test_for_mcp_sets_remote_true(self):
        ctx = OperationContext.for_mcp("agent-007")
        assert ctx.remote is True
        assert ctx.source == "mcp"
        assert ctx.caller == "agent-007"

    def test_for_mcp_default_caller(self):
        ctx = OperationContext.for_mcp()
        assert ctx.caller == "mcp-agent"

    def test_for_rest_sets_remote_true(self):
        ctx = OperationContext.for_rest("127.0.0.1")
        assert ctx.remote is True
        assert ctx.source == "rest"
        assert ctx.caller == "127.0.0.1"

    def test_for_rest_default_caller(self):
        ctx = OperationContext.for_rest()
        assert ctx.caller == "rest-unknown"


class TestImmutability:
    def test_frozen(self):
        ctx = OperationContext()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.remote = True  # type: ignore[misc]


class TestSourceLiteral:
    def test_all_surface_values_supported(self):
        # Sanity check — the three constructors cover all literals.
        assert OperationContext.for_cli().source == "cli"
        assert OperationContext.for_mcp().source == "mcp"
        assert OperationContext.for_rest().source == "rest"


class TestMCPIntegration:
    """Verify the MCP trust-boundary block at EOF of mcp_server.server."""

    def test_current_mcp_context_default_none(self):
        from mcp_server.server import current_mcp_context

        assert current_mcp_context() is None

    def test_mcp_dispatch_is_patched(self):
        from mcp_server import server as mcp_srv

        # The wrapper's __name__ should surface — it's _dispatch_with_ctx.
        assert mcp_srv._dispatch.__name__ == "_dispatch_with_ctx"


class TestRESTIntegration:
    """Verify the REST trust-boundary block at EOF of jarvis_memory.api."""

    def test_current_rest_context_default_none(self):
        from jarvis_memory.api import current_rest_context

        assert current_rest_context() is None

    def test_middleware_installed(self):
        from jarvis_memory.api import app, _RestTrustBoundaryMiddleware

        # FastAPI stores middleware as user_middleware list; Run 4's class must
        # be there.
        installed = [m.cls for m in app.user_middleware]
        assert _RestTrustBoundaryMiddleware in installed
