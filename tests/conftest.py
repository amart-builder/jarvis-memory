"""Pytest bootstrap.

Loads ``.env`` from the repo root into ``os.environ`` *before* any test
module imports ``jarvis_memory.api`` (or anything else that reads config
at import time). Without this, tests hit the config defaults
(``bolt://localhost:7687``, ``neo4j``/``neo4j``) even when the contributor
has populated a valid ``.env`` with remote credentials — the same class
of bug that used to hit ``scripts/migrate_to_v2.py`` when invoked as a
subprocess whose shell never sourced ``.env``.

Uses ``setdefault`` so env vars already exported by the parent shell
always win (e.g. CI that sets them via GitHub Actions secrets).
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        _val = _val.split("#", 1)[0].strip()
        os.environ.setdefault(_key.strip(), _val)
