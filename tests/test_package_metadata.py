from __future__ import annotations

import tomllib
from pathlib import Path

import jarvis_memory


def test_package_version_matches_project_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]

    assert jarvis_memory.__version__ == project["version"]
