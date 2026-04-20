"""Shell handler security + behavior tests.

Only exercises trivial commands (echo, true, false, sleep 0.1) — never
destructive. The ``shell_enabled`` / ``shell_disabled`` fixtures from
conftest control ``GBRAIN_ALLOW_SHELL_JOBS``.

The shell module import-time gate means we must re-import it after the
env flag changes, so each test-class that needs the handler uses a
helper to trigger a fresh import.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jarvis_memory.minions.handlers import (
    _clear_registry_for_tests,
    get_handler,
)


# Helper: re-import the shell module to re-run the import-time gate.
SHELL_MODULE = "jarvis_memory.minions.handlers.shell"


def _fresh_shell_module():
    if SHELL_MODULE in sys.modules:
        del sys.modules[SHELL_MODULE]
    return importlib.import_module(SHELL_MODULE)


class TestGate:
    def test_refuses_to_register_without_env(self, shell_disabled):
        _clear_registry_for_tests()
        # Drop any cached copy so the import-time gate re-runs.
        if SHELL_MODULE in sys.modules:
            del sys.modules[SHELL_MODULE]
        # Import must raise — the gate fires at module load time.
        with pytest.raises(Exception) as exc_info:
            importlib.import_module(SHELL_MODULE)
        err_msg = str(exc_info.value)
        assert "GBRAIN_ALLOW_SHELL_JOBS" in err_msg or "Shell handler" in err_msg
        # And 'shell' must NOT be in the registry.
        from jarvis_memory.minions.handlers import list_handlers
        assert "shell" not in list_handlers()
        # Cleanup: drop the failed module from sys.modules so subsequent
        # tests can re-import fresh.
        sys.modules.pop(SHELL_MODULE, None)

    def test_registers_with_env(self, shell_enabled):
        _clear_registry_for_tests()
        _fresh_shell_module()
        fn = get_handler("shell")
        assert fn is not None


class TestBasicExecution:
    def test_echo_via_cmd(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        result = shell({"cmd": "echo hello_world"})
        assert result["exit_code"] == 0
        assert "hello_world" in result["stdout_tail"]
        assert result["killed_by"] == "completed"

    def test_false_returns_nonzero(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        result = shell({"cmd": "false"})
        assert result["exit_code"] != 0


class TestEnvAllowlist:
    def test_disallowed_env_key_dropped(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        # ZZZ_NOT_ALLOWED is not on the allowlist — the subprocess must not
        # see it.
        result = shell({
            "cmd": "echo $ZZZ_NOT_ALLOWED",
            "env": {"ZZZ_NOT_ALLOWED": "secret"},
        })
        assert result["exit_code"] == 0
        assert "secret" not in result["stdout_tail"]

    def test_allowlisted_env_passes(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        result = shell({
            "cmd": "echo $LANG",
            "env": {"LANG": "en_US.UTF-8"},
        })
        assert "en_US.UTF-8" in result["stdout_tail"]

    def test_absolute_sh_path_not_affected_by_path_override(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        # Even if PATH is bogus, /bin/sh is absolute so the shell still runs.
        result = shell({
            "cmd": "echo resilient",
            "env": {"PATH": "/this/path/does/not/exist"},
        })
        assert result["exit_code"] == 0
        assert "resilient" in result["stdout_tail"]


class TestTimeoutAbort:
    def test_timeout_triggers_kill(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        result = shell({"cmd": "sleep 5", "timeout_seconds": 0.2})
        assert result["killed_by"] == "timeout"
        assert result["exit_code"] != 0


class TestAuditWrite:
    def test_audit_file_created_on_submission(self, shell_enabled, tmp_path, monkeypatch):
        audit_dir = tmp_path / "audit"
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(audit_dir))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        shell({"cmd": "echo auditme"})
        # An audit file should exist for the current ISO week.
        files = sorted(audit_dir.glob("shell-jobs-*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text().strip()
        assert "auditme" in content  # cmd text is logged

    def test_audit_has_no_env_values(self, shell_enabled, tmp_path, monkeypatch):
        audit_dir = tmp_path / "audit"
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(audit_dir))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        shell({
            "cmd": "echo hi",
            "env": {"LANG": "REVEALED_SECRET"},
        })
        raw = next(audit_dir.glob("shell-jobs-*.jsonl")).read_text()
        assert "REVEALED_SECRET" not in raw
        assert "LANG" in raw  # key ok


class TestUtf8Safety:
    def test_multibyte_output_decoded_cleanly(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        # Use printf with a multibyte character.
        result = shell({"cmd": "printf '%s' 'café'"})
        assert result["exit_code"] == 0
        assert "café" in result["stdout_tail"]

    def test_utf8_safe_tail_handles_truncation(self):
        """Direct unit test on the safe-tail helper."""
        from jarvis_memory.minions.handlers.shell import _utf8_safe_tail, TAIL_BYTES

        # Create a blob longer than TAIL_BYTES with multibyte characters sprinkled in.
        raw = ("a" * TAIL_BYTES + "é" * 500).encode("utf-8")
        out = _utf8_safe_tail(raw)
        assert out.startswith("...[truncated]...")
        # Must be valid unicode end-to-end.
        out.encode("utf-8")  # would raise if broken


class TestInputValidation:
    def test_both_cmd_and_argv_rejected(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        with pytest.raises(ValueError, match="exactly one"):
            shell({"cmd": "x", "argv": ["/bin/echo"]})

    def test_neither_cmd_nor_argv_rejected(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        with pytest.raises(ValueError, match="required"):
            shell({})

    def test_relative_argv_rejected(self, shell_enabled, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        _clear_registry_for_tests()
        _fresh_shell_module()
        shell = get_handler("shell")
        with pytest.raises(ValueError, match="absolute"):
            shell({"argv": ["echo", "hi"]})
