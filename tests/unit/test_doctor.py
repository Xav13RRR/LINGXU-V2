from pathlib import Path

from lingxu.cli.doctor import run_checks
from lingxu.config import Settings


def test_doctor_passes_required_local_checks_without_credentials(tmp_path: Path) -> None:
    checks = run_checks(Settings(_env_file=None, runtime_tmp=tmp_path / "runtime"))
    by_name = {check.name: check for check in checks}

    assert not [check for check in checks if check.status == "FAILED"]
    assert by_name["independent_discord_credentials"].status == "WARNING"
    assert by_name["runtime_independence"].status == "PASSED"


def test_strict_doctor_requires_discord_credentials(tmp_path: Path) -> None:
    checks = run_checks(
        Settings(_env_file=None, runtime_tmp=tmp_path / "runtime"), strict_discord=True
    )
    by_name = {check.name: check for check in checks}

    assert by_name["independent_discord_credentials"].status == "FAILED"
