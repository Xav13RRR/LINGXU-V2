import json
import subprocess
from pathlib import Path

import pytest

from scripts.check_public_files import check, inspect_public_file, private_path


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        "nested/.env.example",
        "private/report.md",
        "audio.wav",
        "local.sqlite3",
        "database.db-wal",
        "exports/meeting.json",
        "backup.tar.gz",
        "minutes.md",
        "debug.log.1",
        "credential.dpapi",
        "tests/private_fixtures/report.md",
    ],
)
def test_private_paths_are_denied_even_if_forced_into_git(name: str) -> None:
    assert private_path(name)
    assert inspect_public_file(name, b"safe")


@pytest.mark.parametrize(
    "name",
    [
        ".env.example",
        "src/lingxu/storage/models.py",
        "src/lingxu/worker/minutes.py",
        "src/lingxu/resources/deepseek_full_transcript_minutes.md",
        "docs/CONFIGURATION.en.md",
        "tests/unit/test_minutes_models.py",
    ],
)
def test_public_source_and_template_paths_are_allowed(name: str) -> None:
    assert not private_path(name)


def test_findings_do_not_include_document_contents_or_secret_values() -> None:
    identity = "987654321" + "012345678"
    secret = "sk-" + "a" * 32
    report = inspect_public_file("docs/example.md", f"Guild {identity}\n{secret}".encode())
    assert {item.rule for item in report} == {"discord_identity_in_document", "openai_style_key"}
    rendered = json.dumps([item.__dict__ for item in report])
    assert identity not in rendered
    assert secret not in rendered


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_index_scan_checks_blob_hidden_by_safe_worktree_edit(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    identity = "987654321" + "012345678"
    doc = tmp_path / "README.md"
    doc.write_text(f"Guild {identity}\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    doc.write_text("Use <GUILD_ID>\n", encoding="utf-8")
    assert check(tmp_path)[1] == ()
    count, findings = check(tmp_path, index=True)
    assert count == 1
    assert findings[0].rule == "discord_identity_in_document"
    doc.unlink()
    assert check(tmp_path, index=True)[1] == findings


def test_forced_private_files_fail_worktree_and_index(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    (tmp_path / ".env").write_text("IGNORED=placeholder\n", encoding="utf-8")
    assert not check(tmp_path)[1]
    _git(tmp_path, "add", "-f", ".env")
    assert check(tmp_path)[1][0].rule == "private_file_path"
    assert check(tmp_path, index=True)[1][0].rule == "private_file_path"


def test_template_contains_only_blank_credentials() -> None:
    root = Path(__file__).resolve().parents[2]
    assert inspect_public_file(".env.example", (root / ".env.example").read_bytes()) == ()


def test_ci_checks_public_files_before_dependencies_or_builds() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Repository publication boundary" in workflow
    assert "python -m scripts.check_public_files" in workflow


@pytest.mark.parametrize(
    "name",
    [
        ".env.local",
        ".env.production",
        ".env.backup",
        "local.sqlite3",
        "audio.wav",
        "minutes.md",
        "private/review.md",
        "exports/example.json",
        "credential.dpapi",
    ],
)
def test_repository_ignore_rules_cover_common_accidental_exports(name: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "-q", name],
        capture_output=True,
    )
    assert result.returncode == 0


def test_bilingual_configuration_lists_identical_settings_and_safe_entrypoint() -> None:
    import re

    root = Path(__file__).resolve().parents[2]
    english = (root / "docs/CONFIGURATION.en.md").read_text(encoding="utf-8")
    chinese = (root / "docs/CONFIGURATION.zh-CN.md").read_text(encoding="utf-8")

    def extract(text: str) -> set[str]:
        return set(re.findall(r"`([A-Z][A-Z_]+)`", text))

    assert extract(english) == extract(chinese)
    for text in (english, chinese):
        assert "python -m scripts.check_public_files" in text
        assert "set_deepseek_api_key.ps1" in text
        assert "Attach Files" in text
        assert "SUMMARY_LLM_API_KEY" in text


def test_nonstandard_git_entries_are_not_followed(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    blob = (
        subprocess.run(
            ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
            input=b"outside-target",
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"120000,{blob},linked.md")
    assert check(tmp_path, index=True)[1][0].rule == "unsupported_git_mode"


def test_snapshot_cannot_silently_scan_an_ancestor_repository(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    snapshot = tmp_path / "private/candidate/source"
    snapshot.mkdir(parents=True)
    with pytest.raises(ValueError, match="repository_root"):
        check(snapshot)
    with pytest.raises(ValueError, match="repository_root"):
        check(snapshot, index=True)


def test_private_fixture_dependency_is_rejected_without_reading_fixture() -> None:
    synthetic_source = (
        b"from pathlib import Path\n"
        b'_FIXTURE_DIR = Path(__file__).parents[1] / "private_fixtures" / "example"\n'
    )
    findings = inspect_public_file("tests/test_example.py", synthetic_source)
    assert {item.rule for item in findings} == {"private_fixture_test_dependency"}


def test_pinned_meeting_artifact_manifest_is_rejected_without_echoing_digest() -> None:
    synthetic_digest = "a" * 64
    synthetic_source = ("EXPECTED = {'transcript.json': '" + synthetic_digest + "'}\n").encode()
    findings = inspect_public_file("tests/test_example.py", synthetic_source)
    assert {item.rule for item in findings} == {"pinned_meeting_artifact_fingerprint"}
    assert synthetic_digest not in json.dumps([item.__dict__ for item in findings])


def test_public_model_pins_and_computed_synthetic_hashes_remain_allowed() -> None:
    source = (
        "import hashlib\n"
        "EXPECTED = {'transcript.json': hashlib.sha256(b'synthetic text').hexdigest()}\n"
        "MODEL = {'model.pt': '" + "b" * 64 + "'}\n"
    ).encode()
    assert not inspect_public_file("tests/test_example.py", source)


def test_real_meeting_statistics_are_not_public_documentation() -> None:
    # Deliberately fabricated source; never use an actual retained measurement in this regression.
    payload = b"The private Meeting contains 3 exported transcript entries.\n"
    findings = inspect_public_file("docs/example.md", payload)
    assert {item.rule for item in findings} == {"real_meeting_measurement_in_document"}


def test_private_fixture_source_hidden_by_worktree_edit_is_rejected_in_index(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    test_path = tmp_path / "tests/test_example.py"
    test_path.parent.mkdir()
    test_path.write_text("EXPECTED = {'minutes.json': '" + "c" * 64 + "'}\n", encoding="utf-8")
    _git(tmp_path, "add", "tests/test_example.py")
    test_path.write_text("# synthetic-only test placeholder\n", encoding="utf-8")
    assert not check(tmp_path)[1]
    assert check(tmp_path, index=True)[1][0].rule == "pinned_meeting_artifact_fingerprint"
