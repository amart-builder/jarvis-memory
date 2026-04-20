"""Shell-audit JSONL tests.

The conftest autouse fixture monkeypatches ``GBRAIN_AUDIT_DIR`` to a per-test
tmp directory, so these tests never write to the real home folder.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jarvis_memory.minions.handlers.shell_audit import (
    ARGV_ELEMENT_LIMIT,
    CMD_CHAR_LIMIT,
    _audit_dir,
    _audit_path,
    _current_iso_week,
    append_audit_entry,
    iter_audit_entries,
)


class TestPathResolution:
    def test_audit_dir_respects_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "x"))
        assert _audit_dir() == tmp_path / "x"

    def test_week_encoding_iso(self):
        ts = datetime(2026, 4, 20, tzinfo=timezone.utc)
        # 2026-04-20 is a Monday; ISO week 17.
        assert _current_iso_week(ts) == "2026-W17"

    def test_week_early_january_uses_prior_year(self):
        # 2026-01-01 (Thursday) → ISO week 1 of 2026.
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _current_iso_week(ts) == "2026-W01"


class TestAppendEntry:
    def test_file_created_with_single_line(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "audit"))
        path = append_audit_entry(
            "echo hello",
            caller="worker-test",
            job_id="job-1",
            env_keys=["PATH", "HOME"],
            timeout_seconds=30,
            params={"cmd": "echo hello"},
        )
        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["job_id"] == "job-1"
        assert entry["mode"] == "shell"
        assert entry["cmd"] == "echo hello"
        assert entry["env_keys"] == ["HOME", "PATH"]
        assert entry["timeout_s"] == 30
        assert "params_hash" in entry

    def test_append_adds_line(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path))
        append_audit_entry("echo 1", caller="w", job_id="a", timeout_seconds=10)
        append_audit_entry("echo 2", caller="w", job_id="b", timeout_seconds=10)
        entries = iter_audit_entries()
        assert len(entries) == 2

    def test_cmd_truncated_at_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path))
        long_cmd = "echo " + ("x" * 200)
        append_audit_entry(long_cmd, caller="w", job_id="t", timeout_seconds=10)
        entries = iter_audit_entries()
        assert len(entries[0]["cmd"]) <= CMD_CHAR_LIMIT
        assert entries[0]["cmd"].endswith("...")

    def test_argv_truncated_at_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path))
        argv = ["/bin/echo"] + [f"arg{i}" for i in range(50)]
        append_audit_entry(argv, caller="w", job_id="t", timeout_seconds=10)
        entries = iter_audit_entries()
        assert entries[0]["mode"] == "argv"
        # argv stored, with last marker indicating truncation.
        assert len(entries[0]["argv"]) <= ARGV_ELEMENT_LIMIT + 1
        assert "[+" in entries[0]["argv"][-1]

    def test_env_values_never_logged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path))
        append_audit_entry(
            "echo hi",
            caller="w",
            job_id="secret-test",
            env_keys=["PATH", "SECRET_TOKEN"],
            timeout_seconds=10,
            params={"cmd": "echo hi", "env": {"SECRET_TOKEN": "supersecret"}},
        )
        raw = (tmp_path / f"shell-jobs-{_current_iso_week()}.jsonl").read_text()
        # env_keys holds names only; the value must not appear anywhere.
        assert "supersecret" not in raw
        assert "SECRET_TOKEN" in raw  # key is OK; the value is not


class TestRotation:
    def test_different_weeks_different_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path))
        ts1 = datetime(2026, 4, 20, tzinfo=timezone.utc)  # W17
        ts2 = datetime(2026, 4, 27, tzinfo=timezone.utc)  # W18
        p1 = append_audit_entry("echo w17", caller="w", job_id="a", timeout_seconds=1, ts=ts1)
        p2 = append_audit_entry("echo w18", caller="w", job_id="b", timeout_seconds=1, ts=ts2)
        assert p1 != p2
        assert p1.name == "shell-jobs-2026-W17.jsonl"
        assert p2.name == "shell-jobs-2026-W18.jsonl"


class TestReadBack:
    def test_iter_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GBRAIN_AUDIT_DIR", str(tmp_path / "nothing"))
        assert iter_audit_entries() == []
