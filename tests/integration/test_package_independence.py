from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_installed_package_imports_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import lingxu, json; print(json.dumps({'version': lingxu.__version__}))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"version": "0.1.0"}


@pytest.mark.integration
def test_git_voice_dependency_is_pinned_to_a_full_commit() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert "discord-ext-voice-recv.git@ee160c0f36516927b6214bc9d6babe524016770f" in pyproject
    assert ".git@main" not in pyproject
    assert ".git@master" not in pyproject
