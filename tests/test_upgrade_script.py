from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_upgrade_to_accepts_tag_argument(tmp_path):
    """`scripts/upgrade.sh --to <tag>` should not treat the tag as an extra arg."""
    source = Path(__file__).resolve().parents[1] / "scripts" / "upgrade.sh"
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(source, scripts / "upgrade.sh")

    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", "scripts/upgrade.sh")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "tag", "v1.0.0")
    _git(repo, "remote", "add", "origin", repo.as_uri())

    result = subprocess.run(
        ["bash", "scripts/upgrade.sh", "--to", "v1.0.0"],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Already at v1.0.0" in result.stdout
