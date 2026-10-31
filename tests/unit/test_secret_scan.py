import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "scan_secrets.py"
_SPEC = importlib.util.spec_from_file_location("lingxu_secret_scan", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
scan = _MODULE.scan


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _init_git(root: Path) -> None:
    _git(root, "init", "--quiet")


def test_secret_scan_reports_location_without_secret_value(tmp_path: Path) -> None:
    source = tmp_path / "settings.txt"
    source.write_text(
        "DISCORD_TOKEN=" + "A" * 24 + "." + "B" * 6 + "." + "C" * 27,
        encoding="utf-8",
    )

    report = scan(tmp_path)

    assert report.findings
    assert all(finding.path == Path("settings.txt") for finding in report.findings)
    assert all(finding.line == 1 for finding in report.findings)
    assert all("A" * 24 not in repr(finding) for finding in report.findings)
    assert not report.passed


def test_secret_scan_ignores_local_env_and_blank_example(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DISCORD_TOKEN=local-secret", encoding="utf-8")
    (tmp_path / ".env.example").write_text("DISCORD_TOKEN=\n", encoding="utf-8")

    report = scan(tmp_path)

    assert report.passed
    assert report.findings == ()
    assert report.candidate_files == 1


def test_secret_scan_does_not_exempt_nonblank_example_assignments(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("DISCORD_TOKEN=not-a-placeholder\n", encoding="utf-8")

    report = scan(tmp_path)

    assert tuple(finding.rule for finding in report.findings) == ("configured_secret_assignment",)


@pytest.mark.parametrize(
    "key",
    (
        "DISCORD_TOKEN",
        "FILE_ENCRYPTION_KEY",
        "LLM_API_KEY",
        "SUMMARY_LLM_API_KEY",
        "MEETING_ASSISTANT_LLM_API_KEY",
        "AGENDA_IMPORT_LLM_API_KEY",
        "HOST_CONSOLE_TOKEN",
    ),
)
def test_secret_scan_flags_every_configured_secret_assignment(
    tmp_path: Path,
    key: str,
) -> None:
    source = tmp_path / "settings.txt"
    source.write_text(f"{key}=private-value\n", encoding="utf-8")

    report = scan(tmp_path)

    assert tuple(finding.rule for finding in report.findings) == ("configured_secret_assignment",)


def test_secret_scan_stats_large_file_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * (_MODULE._MAX_TEXT_BYTES + 1))

    def fail_if_read(_path: Path) -> bytes:
        raise AssertionError("large candidates must be rejected from stat metadata")

    monkeypatch.setattr(_MODULE, "_read_candidate_bytes", fail_if_read)

    report = scan(tmp_path, candidate_paths=(Path("large.txt"),))

    assert report.files_scanned == 0
    assert report.errors == (_MODULE.ScanIssue(Path("large.txt"), "file_too_large"),)
    assert not report.passed


def test_secret_scan_ignores_local_backup_tree_in_filesystem_mode(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / "settings.txt").write_text("DISCORD_TOKEN=private-value\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("safe\n", encoding="utf-8")

    report = scan(tmp_path)

    assert report.passed
    assert report.candidate_files == 1
    assert report.files_scanned == 1


def test_secret_scan_includes_nonignored_untracked_git_candidates(tmp_path: Path) -> None:
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("safe\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("DISCORD_TOKEN=private-value\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("DISCORD_TOKEN=ignored-value\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "tracked.txt")

    report = scan(tmp_path)

    assert tuple(finding.path for finding in report.findings) == (Path("untracked.txt"),)
    assert all(issue.path != Path("ignored.txt") for issue in (*report.skipped, *report.errors))
    assert not report.passed


def test_secret_scan_excludes_worktree_deleted_paths_but_scans_the_staged_blob(
    tmp_path: Path,
) -> None:
    _init_git(tmp_path)
    source = tmp_path / "tracked.txt"
    source.write_text("safe\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    source.unlink()

    report = scan(tmp_path)

    assert report.passed
    assert report.candidate_files == 1
    assert report.files_scanned == 1
    assert all(issue.reason != "candidate_missing" for issue in report.errors)


def test_secret_scan_scans_index_blob_for_unstaged_worktree_deletion(tmp_path: Path) -> None:
    _init_git(tmp_path)
    source = tmp_path / "tracked.txt"
    secret = "deleted-worktree-secret-must-not-appear"
    source.write_text(f"DISCORD_TOKEN={secret}\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Secret Scan Test",
        "-c",
        "user.email=secret-scan@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )
    source.unlink()

    report = scan(tmp_path)
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert tuple(finding.path for finding in report.findings) == (Path("tracked.txt"),)
    assert not report.passed
    assert secret not in rendered


def test_secret_scan_scans_index_blob_for_unstaged_safe_worktree_edit(tmp_path: Path) -> None:
    _init_git(tmp_path)
    source = tmp_path / "tracked.txt"
    secret = "index-secret-hidden-by-safe-worktree-edit"
    source.write_text(f"DISCORD_TOKEN={secret}\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Secret Scan Test",
        "-c",
        "user.email=secret-scan@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )
    source.write_text("DISCORD_TOKEN=\n", encoding="utf-8")

    report = scan(tmp_path)
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert tuple(finding.path for finding in report.findings) == (Path("tracked.txt"),)
    assert not report.passed
    assert secret not in rendered


def test_secret_scan_finds_staged_secret_when_worktree_copy_is_safe(tmp_path: Path) -> None:
    _init_git(tmp_path)
    source = tmp_path / "settings.txt"
    staged_secret = "staged-secret-must-not-appear"
    source.write_text(f"DISCORD_TOKEN={staged_secret}\n", encoding="utf-8")
    _git(tmp_path, "add", "settings.txt")
    source.write_text("DISCORD_TOKEN=\n", encoding="utf-8")

    report = scan(tmp_path)
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert tuple(finding.rule for finding in report.findings) == ("configured_secret_assignment",)
    assert not report.passed
    assert staged_secret not in rendered


def test_secret_scan_fails_closed_when_only_index_state_is_unmerged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()

    def fake_run_git(
        _root: Path,
        arguments: tuple[str, ...],
        *,
        failure_reason: str,
    ) -> tuple[bytes | None, None]:
        del failure_reason
        if arguments[:2] == ("ls-files", "--unmerged"):
            return b"100644 0000000000000000000000000000000000000000 1\tsettings.txt\0", None
        return b"", None

    monkeypatch.setattr(_MODULE, "_run_git", fake_run_git)

    report = scan(tmp_path)

    assert report.errors == (_MODULE.ScanIssue(Path("."), "git_index_unmerged"),)
    assert not report.passed


def test_secret_scan_sizes_staged_blob_before_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git(tmp_path)
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * (_MODULE._MAX_TEXT_BYTES + 1))
    _git(tmp_path, "add", "large.txt")
    source.write_text("safe\n", encoding="utf-8")

    def fail_if_read(_root: Path, _blob: object) -> tuple[bytes, None]:
        raise AssertionError("oversized staged blobs must not be read")

    monkeypatch.setattr(_MODULE, "_read_git_blob", fail_if_read)

    report = scan(tmp_path)

    assert _MODULE.ScanIssue(Path("large.txt"), "git_staged_file_too_large") in report.errors
    assert not report.passed


def test_secret_scan_reports_read_errors_without_exception_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "settings.txt"
    source.write_text("safe\n", encoding="utf-8")
    secret_in_exception = "exception-secret-must-not-appear"

    def fail_read(_path: Path) -> bytes:
        raise OSError(secret_in_exception)

    monkeypatch.setattr(_MODULE, "_read_candidate_bytes", fail_read)

    report = scan(tmp_path, candidate_paths=(Path("settings.txt"),))
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert report.errors == (_MODULE.ScanIssue(Path("settings.txt"), "file_read_error"),)
    assert not report.passed
    assert secret_in_exception not in rendered


def test_secret_scan_reports_utf8_decode_errors(tmp_path: Path) -> None:
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe")

    report = scan(tmp_path, candidate_paths=(Path("invalid.txt"),))

    assert report.errors == (_MODULE.ScanIssue(Path("invalid.txt"), "utf8_decode_error"),)
    assert not report.passed


def test_secret_scan_explicitly_reports_known_binary_skip(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"not decoded")

    report = scan(tmp_path, candidate_paths=(Path("image.png"),))

    assert report.skipped == (_MODULE.ScanIssue(Path("image.png"), "known_binary_extension"),)
    assert report.passed
    assert report.files_scanned == 0


@pytest.mark.parametrize(
    ("name", "prefix", "expected_skip"),
    (
        ("disguised.png", b"", "known_binary_extension"),
        ("nul-payload.dat", b"\0\n", "binary_content"),
    ),
)
def test_secret_scan_checks_ascii_credentials_before_binary_skip(
    tmp_path: Path,
    name: str,
    prefix: bytes,
    expected_skip: str,
) -> None:
    secret = b"binary-secret-must-not-appear"
    (tmp_path / name).write_bytes(prefix + b"DISCORD_TOKEN=" + secret + b"\n")

    report = scan(tmp_path, candidate_paths=(Path(name),))
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert tuple(finding.rule for finding in report.findings) == ("configured_secret_assignment",)
    assert report.skipped == (_MODULE.ScanIssue(Path(name), expected_skip),)
    assert not report.passed
    assert secret.decode() not in rendered


@pytest.mark.parametrize("encoding", ("utf-16-le", "utf-16-be"))
def test_secret_scan_checks_utf16_ascii_credentials_before_binary_skip(
    tmp_path: Path,
    encoding: str,
) -> None:
    secret = "utf16-secret-must-not-appear"
    (tmp_path / "settings.dat").write_bytes(f"DISCORD_TOKEN={secret}\n".encode(encoding))

    report = scan(tmp_path, candidate_paths=(Path("settings.dat"),))
    rendered = json.dumps(_MODULE._report_payload(report), sort_keys=True)

    assert tuple(finding.rule for finding in report.findings) == ("configured_secret_assignment",)
    assert report.skipped == (_MODULE.ScanIssue(Path("settings.dat"), "binary_content"),)
    assert not report.passed
    assert secret not in rendered


def test_secret_scan_rejects_linked_ancestor_before_reading_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "settings.txt").write_text(
        "DISCORD_TOKEN=outside-secret-must-not-appear\n",
        encoding="utf-8",
    )
    linked = root / "linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {type(exc).__name__}")

    report = scan(root, candidate_paths=(Path("linked/settings.txt"),))

    assert report.findings == ()
    assert report.errors == (
        _MODULE.ScanIssue(Path("linked/settings.txt"), "candidate_ancestor_link"),
    )
    assert not report.passed
