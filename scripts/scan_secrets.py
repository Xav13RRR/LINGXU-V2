"""Fail closed on credential-shaped values without printing matched secrets."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_TEXT_BYTES = 3 * 1_024 * 1_024
_LOCAL_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    ".venv",
    ".worktrees",
    "backups",
    "build",
    "data",
    "dist",
    "htmlcov",
    "models",
}
_LOCAL_SKIP_FILES = {".env"}
_KNOWN_BINARY_SUFFIXES = {
    ".aac",
    ".bin",
    ".flac",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".ogg",
    ".opus",
    ".pdf",
    ".png",
    ".wav",
    ".webp",
    ".zip",
}
_CONFIGURED_SECRET_KEYS = (
    "AGENDA_IMPORT_LLM_API_KEY",
    "DISCORD_TOKEN",
    "FILE_ENCRYPTION_KEY",
    "HOST_CONSOLE_TOKEN",
    "LLM_API_KEY",
    "MEETING_ASSISTANT_LLM_API_KEY",
    "SUMMARY_LLM_API_KEY",
)
_CONFIGURED_SECRET_ASSIGNMENT = re.compile(
    rf"(?im)^(?:{'|'.join(map(re.escape, _CONFIGURED_SECRET_KEYS))})"
    r"[ \t]*=[ \t]*(?!$|<)\S+"
)


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    rule: str


@dataclass(frozen=True, slots=True)
class ScanIssue:
    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class ScanReport:
    candidate_files: int
    files_scanned: int
    findings: tuple[Finding, ...]
    skipped: tuple[ScanIssue, ...]
    errors: tuple[ScanIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.findings and not self.errors


@dataclass(frozen=True, slots=True)
class _IndexBlob:
    path: Path
    object_id: str
    mode: str


_RULES = (
    (
        "discord_bot_token",
        re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{25,}"),
    ),
    (
        "openai_style_key",
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "configured_secret_assignment",
        _CONFIGURED_SECRET_ASSIGNMENT,
    ),
)


def _repository_uses_git(root: Path) -> tuple[bool, ScanIssue | None]:
    try:
        (root / ".git").stat(follow_symlinks=False)
    except FileNotFoundError:
        return False, None
    except OSError:
        return False, ScanIssue(path=Path("."), reason="git_marker_stat_error")
    return True, None


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    failure_reason: str,
) -> tuple[bytes | None, ScanIssue | None]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ScanIssue(path=Path("."), reason=failure_reason)
    if completed.returncode != 0:
        return None, ScanIssue(path=Path("."), reason=failure_reason)
    return completed.stdout, None


def _decode_nul_paths(
    payload: bytes,
    *,
    failure_reason: str,
) -> tuple[tuple[Path, ...], ScanIssue | None]:
    try:
        names = payload.decode("utf-8").split("\0")
    except UnicodeDecodeError:
        return (), ScanIssue(path=Path("."), reason=failure_reason)
    return tuple(Path(name) for name in names if name), None


def _git_worktree_candidates(root: Path) -> tuple[tuple[Path, ...], tuple[ScanIssue, ...]]:
    listed_payload, listed_error = _run_git(
        root,
        ("ls-files", "-z", "--cached", "--others", "--exclude-standard"),
        failure_reason="git_candidate_enumeration_error",
    )
    if listed_error is not None or listed_payload is None:
        return (), (listed_error or ScanIssue(Path("."), "git_candidate_enumeration_error"),)
    deleted_payload, deleted_error = _run_git(
        root,
        ("ls-files", "-z", "--deleted"),
        failure_reason="git_deleted_enumeration_error",
    )
    if deleted_error is not None or deleted_payload is None:
        return (), (deleted_error or ScanIssue(Path("."), "git_deleted_enumeration_error"),)
    listed, listed_decode_error = _decode_nul_paths(
        listed_payload,
        failure_reason="git_candidate_decode_error",
    )
    deleted, deleted_decode_error = _decode_nul_paths(
        deleted_payload,
        failure_reason="git_deleted_decode_error",
    )
    decode_errors = tuple(
        issue for issue in (listed_decode_error, deleted_decode_error) if issue is not None
    )
    if decode_errors:
        return (), decode_errors
    deleted_set = set(deleted)
    return tuple(path for path in listed if path not in deleted_set), ()


def _parse_index_entries(payload: bytes) -> tuple[dict[Path, _IndexBlob], tuple[ScanIssue, ...]]:
    try:
        records = payload.decode("utf-8").split("\0")
    except UnicodeDecodeError:
        return {}, (ScanIssue(Path("."), "git_index_decode_error"),)
    entries: dict[Path, _IndexBlob] = {}
    errors: list[ScanIssue] = []
    for record in records:
        if not record:
            continue
        try:
            header, name = record.split("\t", 1)
            mode, object_id, stage = header.split()
        except ValueError:
            errors.append(ScanIssue(Path("."), "git_index_record_invalid"))
            continue
        path = Path(name)
        if stage != "0":
            errors.append(ScanIssue(path, "git_index_unmerged"))
            continue
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id):
            errors.append(ScanIssue(path, "git_index_object_invalid"))
            continue
        entries[path] = _IndexBlob(path=path, object_id=object_id, mode=mode)
    return entries, tuple(errors)


def _git_staged_candidates(root: Path) -> tuple[tuple[_IndexBlob, ...], tuple[ScanIssue, ...]]:
    unmerged_payload, unmerged_error = _run_git(
        root,
        ("ls-files", "--unmerged", "-z"),
        failure_reason="git_unmerged_enumeration_error",
    )
    if unmerged_error is not None or unmerged_payload is None:
        return (), (unmerged_error or ScanIssue(Path("."), "git_unmerged_enumeration_error"),)
    if unmerged_payload:
        return (), (ScanIssue(Path("."), "git_index_unmerged"),)
    changed_payload, changed_error = _run_git(
        root,
        (
            "diff",
            "--cached",
            "--name-only",
            "--no-ext-diff",
            "--no-renames",
            "--diff-filter=ACMT",
            "-z",
            "--",
        ),
        failure_reason="git_staged_enumeration_error",
    )
    if changed_error is not None or changed_payload is None:
        return (), (changed_error or ScanIssue(Path("."), "git_staged_enumeration_error"),)
    changed, decode_error = _decode_nul_paths(
        changed_payload,
        failure_reason="git_staged_decode_error",
    )
    if decode_error is not None:
        return (), (decode_error,)
    deleted_payload, deleted_error = _run_git(
        root,
        ("ls-files", "--deleted", "-z"),
        failure_reason="git_deleted_index_enumeration_error",
    )
    if deleted_error is not None or deleted_payload is None:
        return (), (deleted_error or ScanIssue(Path("."), "git_deleted_index_enumeration_error"),)
    deleted, deleted_decode_error = _decode_nul_paths(
        deleted_payload,
        failure_reason="git_deleted_index_decode_error",
    )
    if deleted_decode_error is not None:
        return (), (deleted_decode_error,)
    modified_payload, modified_error = _run_git(
        root,
        (
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--no-renames",
            "--diff-filter=MT",
            "-z",
            "--",
        ),
        failure_reason="git_modified_index_enumeration_error",
    )
    if modified_error is not None or modified_payload is None:
        return (), (modified_error or ScanIssue(Path("."), "git_modified_index_enumeration_error"),)
    modified, modified_decode_error = _decode_nul_paths(
        modified_payload,
        failure_reason="git_modified_index_decode_error",
    )
    if modified_decode_error is not None:
        return (), (modified_decode_error,)
    selected_paths = tuple(dict.fromkeys((*changed, *deleted, *modified)))
    if not selected_paths:
        return (), ()
    index_payload, index_error = _run_git(
        root,
        ("ls-files", "--stage", "-z"),
        failure_reason="git_index_enumeration_error",
    )
    if index_error is not None or index_payload is None:
        return (), (index_error or ScanIssue(Path("."), "git_index_enumeration_error"),)
    entries, index_errors = _parse_index_entries(index_payload)
    errors = list(index_errors)
    blobs: list[_IndexBlob] = []
    for path in selected_paths:
        blob = entries.get(path)
        if blob is None:
            errors.append(ScanIssue(path, "git_staged_candidate_missing"))
        else:
            blobs.append(blob)
    return tuple(blobs), tuple(errors)


def _git_blob_size(root: Path, blob: _IndexBlob) -> tuple[int | None, ScanIssue | None]:
    payload, issue = _run_git(
        root,
        ("cat-file", "-s", blob.object_id),
        failure_reason="git_staged_size_error",
    )
    if issue is not None or payload is None:
        return None, ScanIssue(blob.path, "git_staged_size_error")
    try:
        size = int(payload.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        return None, ScanIssue(blob.path, "git_staged_size_error")
    if size < 0:
        return None, ScanIssue(blob.path, "git_staged_size_error")
    return size, None


def _read_git_blob(root: Path, blob: _IndexBlob) -> tuple[bytes | None, ScanIssue | None]:
    payload, issue = _run_git(
        root,
        ("cat-file", "blob", blob.object_id),
        failure_reason="git_staged_read_error",
    )
    if issue is not None or payload is None:
        return None, ScanIssue(blob.path, "git_staged_read_error")
    return payload, None


def _is_local_backup_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith((".bak", ".backup", "~")) or lowered.startswith(".#")


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _filesystem_candidates(root: Path) -> tuple[tuple[Path, ...], tuple[ScanIssue, ...]]:
    files: list[Path] = []
    errors: list[ScanIssue] = []

    def record_walk_error(_error: OSError) -> None:
        errors.append(ScanIssue(path=Path("."), reason="filesystem_candidate_enumeration_error"))

    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=record_walk_error,
    ):
        base = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            if name in _LOCAL_SKIP_DIRS:
                continue
            child = base / name
            relative = child.relative_to(root)
            try:
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode) or _is_junction(child):
                    errors.append(ScanIssue(relative, "candidate_ancestor_link"))
                    continue
            except OSError:
                errors.append(ScanIssue(relative, "candidate_ancestor_stat_error"))
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            if name in _LOCAL_SKIP_FILES or _is_local_backup_file(name):
                continue
            files.append((base / name).relative_to(root))
    return tuple(files), tuple(errors)


def _normalise_candidate(root: Path, candidate: Path) -> tuple[Path, Path] | None:
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            return None
    else:
        relative = candidate
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return root.joinpath(relative), relative


def _validate_candidate_path(
    root: Path,
    candidate: Path,
) -> tuple[Path | None, Path, os.stat_result | None, ScanIssue | None]:
    normalised = _normalise_candidate(root, candidate)
    if normalised is None:
        return None, Path("."), None, ScanIssue(Path("."), "candidate_path_invalid")
    path, relative = normalised
    current = root
    for part in relative.parts[:-1]:
        current /= part
        try:
            metadata = current.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or _is_junction(current):
                return None, relative, None, ScanIssue(relative, "candidate_ancestor_link")
            if not stat.S_ISDIR(metadata.st_mode):
                return (
                    None,
                    relative,
                    None,
                    ScanIssue(
                        relative,
                        "candidate_ancestor_not_directory",
                    ),
                )
        except FileNotFoundError:
            return None, relative, None, ScanIssue(relative, "candidate_missing")
        except OSError:
            return None, relative, None, ScanIssue(relative, "candidate_ancestor_stat_error")
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None, relative, None, ScanIssue(relative, "candidate_missing")
    except OSError:
        return None, relative, None, ScanIssue(relative, "candidate_stat_error")
    try:
        if stat.S_ISLNK(metadata.st_mode) or _is_junction(path):
            return None, relative, None, ScanIssue(relative, "candidate_link")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except ValueError:
        return None, relative, None, ScanIssue(relative, "candidate_outside_root")
    except (OSError, RuntimeError):
        return None, relative, None, ScanIssue(relative, "candidate_resolve_error")
    if not stat.S_ISREG(metadata.st_mode):
        return None, relative, None, ScanIssue(relative, "candidate_not_regular_file")
    return path, relative, metadata, None


def _read_candidate_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(_MAX_TEXT_BYTES + 1)


def _findings_in_text(text: str, path: Path) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for rule, pattern in _RULES:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    path=path,
                    line=text.count("\n", 0, match.start()) + 1,
                    rule=rule,
                )
            )
    return tuple(findings)


def _inspect_payload(
    payload: bytes,
    path: Path,
) -> tuple[tuple[Finding, ...], bool, ScanIssue | None, ScanIssue | None]:
    ascii_views = [payload.decode("ascii", errors="ignore")]
    if b"\0" in payload:
        ascii_views.append(payload.replace(b"\0", b"").decode("ascii", errors="ignore"))
    ascii_findings = tuple(
        dict.fromkeys(finding for view in ascii_views for finding in _findings_in_text(view, path))
    )
    if path.suffix.lower() in _KNOWN_BINARY_SUFFIXES:
        return (
            ascii_findings,
            False,
            ScanIssue(path, "known_binary_extension"),
            None,
        )
    if b"\0" in payload:
        return ascii_findings, False, ScanIssue(path, "binary_content"), None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ascii_findings, False, None, ScanIssue(path, "utf8_decode_error")
    return _findings_in_text(text, path), True, None, None


def scan(root: Path, *, candidate_paths: Iterable[Path] | None = None) -> ScanReport:
    root = root.resolve()
    staged_candidates: tuple[_IndexBlob, ...] = ()
    if candidate_paths is None:
        uses_git, marker_error = _repository_uses_git(root)
        if marker_error is not None:
            candidates: tuple[Path, ...] = ()
            enumeration_errors: tuple[ScanIssue, ...] = (marker_error,)
        elif uses_git:
            candidates, enumeration_errors = _git_worktree_candidates(root)
            staged_candidates, staged_errors = _git_staged_candidates(root)
            enumeration_errors = (*enumeration_errors, *staged_errors)
        else:
            candidates, enumeration_errors = _filesystem_candidates(root)
    else:
        candidates = tuple(candidate_paths)
        enumeration_errors = ()

    findings: list[Finding] = []
    skipped: list[ScanIssue] = []
    errors = list(enumeration_errors)
    files_scanned = 0
    seen: set[Path] = set()

    for candidate in candidates:
        path, relative, metadata, validation_error = _validate_candidate_path(root, candidate)
        if validation_error is not None or path is None or metadata is None:
            errors.append(validation_error or ScanIssue(Path("."), "candidate_path_invalid"))
            continue
        if relative in seen:
            continue
        seen.add(relative)
        if metadata.st_size > _MAX_TEXT_BYTES:
            errors.append(ScanIssue(path=relative, reason="file_too_large"))
            continue
        try:
            payload = _read_candidate_bytes(path)
        except OSError:
            errors.append(ScanIssue(path=relative, reason="file_read_error"))
            continue
        if len(payload) > _MAX_TEXT_BYTES:
            errors.append(ScanIssue(path=relative, reason="file_too_large"))
            continue
        payload_findings, scanned_as_text, skip_issue, payload_error = _inspect_payload(
            payload,
            relative,
        )
        findings.extend(payload_findings)
        files_scanned += int(scanned_as_text)
        if skip_issue is not None:
            skipped.append(skip_issue)
        if payload_error is not None:
            errors.append(payload_error)

    for blob in staged_candidates:
        normalised = _normalise_candidate(root, blob.path)
        if normalised is None:
            errors.append(ScanIssue(Path("."), "git_staged_path_invalid"))
            continue
        _, relative = normalised
        if not blob.mode.startswith("100"):
            errors.append(ScanIssue(relative, "git_staged_not_regular_file"))
            continue
        size, size_error = _git_blob_size(root, blob)
        if size_error is not None or size is None:
            errors.append(size_error or ScanIssue(relative, "git_staged_size_error"))
            continue
        if size > _MAX_TEXT_BYTES:
            errors.append(ScanIssue(relative, "git_staged_file_too_large"))
            continue
        staged_payload, read_error = _read_git_blob(root, blob)
        if read_error is not None or staged_payload is None:
            errors.append(read_error or ScanIssue(relative, "git_staged_read_error"))
            continue
        if len(staged_payload) != size or len(staged_payload) > _MAX_TEXT_BYTES:
            errors.append(ScanIssue(relative, "git_staged_size_changed"))
            continue
        payload_findings, scanned_as_text, skip_issue, payload_error = _inspect_payload(
            staged_payload,
            relative,
        )
        findings.extend(payload_findings)
        files_scanned += int(scanned_as_text)
        if skip_issue is not None:
            skipped.append(skip_issue)
        if payload_error is not None:
            errors.append(payload_error)

    return ScanReport(
        candidate_files=len(candidates) + len(staged_candidates),
        files_scanned=files_scanned,
        findings=tuple(dict.fromkeys(findings)),
        skipped=tuple(dict.fromkeys(skipped)),
        errors=tuple(dict.fromkeys(errors)),
    )


def _report_payload(report: ScanReport) -> dict[str, Any]:
    return {
        "candidate_files": report.candidate_files,
        "files_scanned": report.files_scanned,
        "findings": [
            {
                "path": finding.path.as_posix(),
                "line": finding.line,
                "rule": finding.rule,
            }
            for finding in report.findings
        ],
        "skipped": [
            {"path": issue.path.as_posix(), "reason": issue.reason} for issue in report.skipped
        ],
        "errors": [
            {"path": issue.path.as_posix(), "reason": issue.reason} for issue in report.errors
        ],
        "passed": report.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = scan(args.root)
    print(json.dumps(_report_payload(report), sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
