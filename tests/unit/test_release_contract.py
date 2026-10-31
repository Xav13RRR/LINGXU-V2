import ast
import re
import tomllib
from pathlib import Path

import lingxu
from lingxu.resources import deepseek_full_transcript_minutes_template

_ROOT = Path(__file__).resolve().parents[2]


def test_package_and_application_image_share_traceable_release_metadata() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert lingxu.__version__ == project["version"]
    assert f'ARG BUILD_VERSION="{project["version"]}"' in dockerfile
    assert 'org.opencontainers.image.version="${BUILD_VERSION}"' in dockerfile
    assert 'org.opencontainers.image.revision="${BUILD_REVISION}"' in dockerfile
    assert 'org.opencontainers.image.source="${BUILD_SOURCE}"' in dockerfile


def test_revision_metadata_is_applied_after_stable_runtime_layers() -> None:
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_base = dockerfile.split("FROM ${PYTHON_BASE} AS runtime-base", maxsplit=1)[1].split(
        "FROM runtime-base AS tools", maxsplit=1
    )[0]
    tools, runtime = dockerfile.split("FROM runtime-base AS tools", maxsplit=1)[1].split(
        "FROM runtime-base AS runtime", maxsplit=1
    )

    assert "BUILD_REVISION" not in runtime_base
    assert "org.opencontainers.image.revision" not in runtime_base
    for target in (tools, runtime):
        assert target.index("COPY scripts/") < target.index("ARG BUILD_REVISION")
        assert target.index("ARG BUILD_REVISION") < target.index("LABEL org.opencontainers.image")


def test_application_runtime_image_is_pinned_and_excludes_build_only_git() -> None:
    dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    digest = re.search(
        r"^ARG PYTHON_BASE=python:3\.12\.11-slim-bookworm@(sha256:[0-9a-f]{64})$",
        dockerfile,
        flags=re.MULTILINE,
    )

    assert digest is not None
    builder, runtime = dockerfile.split(" AS runtime", maxsplit=1)
    assert " AS builder" in builder
    assert re.search(r"^\s+git\s*\\?$", builder, flags=re.MULTILINE) is not None
    assert re.search(r"^\s+git\s*\\?$", runtime, flags=re.MULTILINE) is None
    assert "COPY --from=builder /app/.venv /app/.venv" in runtime
    assert "GLOSSARY_ROOT=/data/glossaries" in runtime
    assert "TTS_CACHE_DIR=/data/tts_cache" in runtime


def test_final_minutes_template_is_an_explicit_wheel_resource() -> None:
    resource = deepseek_full_transcript_minutes_template()
    payload = resource.read_bytes()
    wheel = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["hatch"][
        "build"
    ]["targets"]["wheel"]

    assert payload.startswith("# 聆序会议纪要".encode())
    assert b"{{" in payload
    assert len(payload) <= 64_000
    assert wheel["force-include"] == {
        "src/lingxu/resources/deepseek_full_transcript_minutes.md": (
            "lingxu/resources/deepseek_full_transcript_minutes.md"
        )
    }
    assert not (_ROOT / "templates" / "deepseek_full_transcript_minutes.md").exists()
    assert "COPY templates ./templates" not in (_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_tts_image_binds_its_source_fingerprint_after_runtime_files() -> None:
    dockerfile = (_ROOT / "Dockerfile.cosyvoice").read_text(encoding="utf-8")

    assert dockerfile.index("COPY --chown=10002:10002 services/cosyvoice/") < dockerfile.index(
        "ARG TTS_BUILD_FINGERPRINT"
    )
    assert dockerfile.index("ARG TTS_BUILD_FINGERPRINT") < dockerfile.index(
        'io.lingxu.tts.build-fingerprint="${TTS_BUILD_FINGERPRINT}"'
    )


def test_integration_tests_do_not_import_other_collected_test_modules() -> None:
    violations: list[str] = []
    for path in sorted((_ROOT / "tests" / "integration").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "tests.integration.test_"
            ):
                violations.append(f"{path.name}:{node.lineno}:{node.module}")

    assert violations == []


def test_outcome_partition_planning_delegates_recursive_event_search() -> None:
    source = (_ROOT / "src" / "lingxu" / "worker" / "outcome_partitions.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    planner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "plan_outcome_partitions"
    )

    assert any(
        isinstance(node, ast.ClassDef) and node.name == "_OutcomeEventPlanner" for node in tree.body
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node is not planner for node in ast.walk(planner)
    )


def test_outcome_extraction_stage_delegates_durable_orchestration_steps() -> None:
    source = (_ROOT / "src" / "lingxu" / "worker" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    pipeline = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PostMeetingPipeline"
    )
    methods = {
        node.name: node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    stage = methods["_stage_extract_topic_outcomes"]

    assert any(
        isinstance(node, ast.ClassDef) and node.name == "_OutcomeExtractionContext"
        for node in tree.body
    )
    assert {
        "_prepare_outcome_extraction",
        "_try_reuse_outcome_artifact",
        "_run_missing_outcome_tasks",
        "_finalize_outcome_extraction",
    } <= methods.keys()
    assert not any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node is not stage
        for node in ast.walk(stage)
    )
