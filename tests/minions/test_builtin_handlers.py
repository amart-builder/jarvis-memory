"""Built-in handler tests (echo, compact_daily, compact_weekly).

The compact_* handlers spawn subprocesses. We mock the actual invocation
and assert the argv shape so these tests don't depend on a live Neo4j
connection.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis_memory.minions.handlers import (
    get_handler,
    list_handlers,
)


@pytest.fixture(autouse=True)
def _register_builtins():
    """The builtin module registers on import. Re-import fresh per test."""
    import importlib

    from jarvis_memory.minions.handlers import builtin as _builtin

    importlib.reload(_builtin)
    yield


class TestEcho:
    def test_echo_roundtrips_dict(self):
        echo = get_handler("echo")
        assert echo({"a": 1, "b": "hi"}) == {"echoed": {"a": 1, "b": "hi"}}

    def test_echo_rejects_non_dict(self):
        echo = get_handler("echo")
        with pytest.raises(TypeError):
            echo("not a dict")  # type: ignore[arg-type]


class TestCompaction:
    def test_compact_daily_registered(self):
        assert "compact_daily" in list_handlers()

    def test_compact_weekly_registered(self):
        assert "compact_weekly" in list_handlers()

    def test_compact_daily_invokes_subprocess_with_tier(self):
        fake = SimpleNamespace(returncode=0, stdout='{"ok":true}\n', stderr="")

        with patch("jarvis_memory.minions.handlers.builtin.subprocess.run", return_value=fake) as mocked:
            with patch("jarvis_memory.minions.handlers.builtin.COMPACTION_SCRIPT") as mocked_path:
                mocked_path.exists.return_value = True
                mocked_path.__str__ = lambda _self=None: "/tmp/run_compaction.py"  # type: ignore[assignment]
                compact_daily = get_handler("compact_daily")
                result = compact_daily({})

        assert result["tier"] == "daily"
        assert result["exit_code"] == 0
        # Verify --tier daily made it to the subprocess argv.
        called_cmd = mocked.call_args.args[0]
        assert "--tier" in called_cmd
        assert "daily" in called_cmd

    def test_compact_daily_rejects_bad_tier(self):
        # Calling _run_compaction directly — the handler API only exposes
        # daily/weekly so it's a regression guard on the underlying helper.
        from jarvis_memory.minions.handlers.builtin import _run_compaction

        with patch("jarvis_memory.minions.handlers.builtin.COMPACTION_SCRIPT") as mp:
            mp.exists.return_value = True
            with pytest.raises(ValueError):
                _run_compaction("hourly", {})

    def test_compact_daily_subprocess_timeout_captured(self):
        import subprocess as sp

        with patch(
            "jarvis_memory.minions.handlers.builtin.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd=["python"], timeout=5, output="partial", stderr=""),
        ):
            with patch("jarvis_memory.minions.handlers.builtin.COMPACTION_SCRIPT") as mp:
                mp.exists.return_value = True
                mp.__str__ = lambda _self=None: "/tmp/run_compaction.py"  # type: ignore[assignment]
                compact_daily = get_handler("compact_daily")
                result = compact_daily({"timeout_seconds": 5})

        assert result["exit_code"] == -1
        assert "timeout" in result["error"]
