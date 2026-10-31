from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
_SETUP_PYTHON_SHA = "5fda3b95a4ea91299a34e894583c3862153e4b97"
_LINUX_PYTHON_VERSION = "3.12.11"
_WINDOWS_PYTHON_VERSION = "3.12.10"


def _workflow() -> dict[str, object]:
    return yaml.safe_load((_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    return job["steps"]  # type: ignore[return-value]


def test_ci_cancels_superseded_runs_and_parallelises_expensive_linux_work() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]

    assert workflow["concurrency"] == {
        "group": "ci-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": True,
    }
    assert {"static-quality", "tests", "container", "windows-quality"} <= set(jobs)

    commands = {
        job_name: "\n".join(str(step.get("run", "")) for step in _steps(job))
        for job_name, job in jobs.items()
    }
    assert "uv run pytest" in commands["tests"]
    assert "uv run pytest" not in commands["static-quality"]
    assert "docker build" in commands["container"]
    assert "docker build" not in commands["static-quality"]


def test_ci_actions_and_python_runtime_are_exactly_pinned() -> None:
    jobs = _workflow()["jobs"]

    assert (_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12.11"

    for job in jobs.values():
        checkout_steps = [
            step
            for step in _steps(job)
            if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        assert [step["uses"] for step in checkout_steps] == [f"actions/checkout@{_CHECKOUT_SHA}"]

    expected_versions = {
        "static-quality": _LINUX_PYTHON_VERSION,
        "tests": _LINUX_PYTHON_VERSION,
        "windows-quality": _WINDOWS_PYTHON_VERSION,
    }
    for job_name, expected_version in expected_versions.items():
        setup_steps = [
            step
            for step in _steps(jobs[job_name])
            if str(step.get("uses", "")).startswith("actions/setup-python@")
        ]
        assert [step["uses"] for step in setup_steps] == [
            f"actions/setup-python@{_SETUP_PYTHON_SHA}"
        ]
        assert setup_steps[0]["with"]["python-version"] == expected_version


def test_ci_smokes_packaged_resource_from_an_isolated_wheel_install() -> None:
    static_steps = _steps(_workflow()["jobs"]["static-quality"])
    smoke_step = next(
        step for step in static_steps if step.get("name") == "Smoke packaged wheel resource"
    )
    command = str(smoke_step["run"])

    assert "python -m venv" in command
    assert "pip install --no-deps dist/*.whl" in command
    assert "from lingxu.resources import deepseek_full_transcript_minutes_template" in command
    assert "deepseek_full_transcript_minutes.md" not in command
