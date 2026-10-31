"""Reject private file paths and identity-shaped public documentation; never print content.

Default: inspect tracked and non-ignored worktree candidates, including forced tracked files.
--index: inspect the complete staged snapshot, including blobs hidden by worktree edits/deletion.
This is not a history scanner or a semantic detector of every possible meeting record.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from scripts.scan_secrets import (
    _MAX_TEXT_BYTES,
    _git_worktree_candidates,
    _inspect_payload,
    _parse_index_entries,
    _validate_candidate_path,
)

_PRIVATE_PARTS = {
    ".runtime",
    "backups",
    "data",
    "exports",
    "logs",
    "meeting-exports",
    "models",
    "private",
    "private_fixtures",
    "secrets",
}
_PRIVATE_SUFFIXES = (
    ".db",
    ".db-wal",
    ".db-shm",
    ".sqlite",
    ".sqlite3",
    ".sqlite-wal",
    ".sqlite-shm",
    ".dpapi",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".wav",
    ".flac",
    ".pcm",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".bak",
    ".backup",
)
_PRIVATE_NAMES = {
    "minutes.md",
    "minutes.json",
    "transcript.txt",
    "transcript.json",
    "transcript.jsonl",
    "attempts.jsonl",
    "lingxu.db",
}
_IDENTITIES = (
    ("discord_identity_in_document", re.compile(r"(?<![\w])\d{17,20}(?![\w])")),
    ("session_identity_in_document", re.compile(r"\b01[0-9A-HJKMNP-TV-Z]{24}\b")),
    ("discord_message_link", re.compile(r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\d")),
    (
        "dated_private_evidence_path",
        re.compile(r"(?:\.runtime|backups|private_fixtures)/[^\s`]*20\d{2}[-_]?\d{2}"),
    ),
    (
        "real_meeting_measurement_in_document",
        re.compile(
            r"(?i)\b(?:private|real|retained|production|historical)\s+"
            r"(?:meeting|session|transcript|fixture)\b[^\n]{0,100}?"
            r"\b\d[\d,]*\s+(?:exported\s+)?(?:transcript\s+)?"
            r"(?:rows|entries|sources|utterances|segments|participants|tokens)\b"
        ),
    ),
)
_MEETING_ARTIFACT_NAMES = {
    "events.json",
    "evidence.json",
    "minutes.json",
    "minutes.md",
    "session-manifest.json",
    "transcript.json",
    "transcript.txt",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str


def private_path(path: str) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/").lower())
    name = normalized.name
    if normalized.as_posix() == ".env.example":
        return False
    return (
        bool(_PRIVATE_PARTS.intersection(normalized.parts))
        or name == ".env"
        or name.startswith(".env.")
        or name in _PRIVATE_NAMES
        or name.endswith(_PRIVATE_SUFFIXES)
        or ".log" in normalized.suffixes
    )


def inspect_public_file(path: str, payload: bytes) -> tuple[Finding, ...]:
    if private_path(path):
        return (Finding(path, 0, "private_file_path"),)
    findings: list[Finding] = []
    secrets, _, _, error = _inspect_payload(payload, Path(path))
    findings.extend(Finding(path, item.line, item.rule) for item in secrets)
    if error:
        findings.append(Finding(path, 0, error.reason))
    if Path(path).suffix.lower() == ".py":
        findings.extend(_inspect_python_privacy(path, payload))
    if Path(path).suffix.lower() in {".md", ".rst", ".txt"}:
        text = payload.decode("utf-8", errors="replace")
        for rule, pattern in _IDENTITIES:
            findings.extend(
                Finding(path, text.count("\n", 0, match.start()) + 1, rule)
                for match in pattern.finditer(text)
            )
    return tuple(findings)


def _inspect_python_privacy(path: str, payload: bytes) -> tuple[Finding, ...]:
    """Reject pinned meeting manifests and test dependencies on private evidence.

    Do not embed a private digest/identifier deny list here: the list itself would disclose data.
    Tests use synthetic source strings; public model pins and code-schema hashes remain valid.
    """
    try:
        tree = ast.parse(payload.decode("utf-8-sig"))
    except (SyntaxError, UnicodeError):
        return (Finding(path, 0, "python_parse_failed"),)
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in _MEETING_ARTIFACT_NAMES
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", value.value)
                ):
                    findings.append(
                        Finding(path, node.lineno, "pinned_meeting_artifact_fingerprint")
                    )
        if path.replace("\\", "/").startswith("tests/") and isinstance(
            node, (ast.Assign, ast.AnnAssign)
        ):
            expression = node.value
            if expression is None:
                continue
            nodes = tuple(ast.walk(expression))
            private_directory = any(
                isinstance(item, ast.Constant) and item.value == "private_fixtures"
                for item in nodes
            )
            file_anchored = any(
                isinstance(item, ast.Name) and item.id == "__file__" for item in nodes
            )
            if private_directory and file_anchored:
                findings.append(Finding(path, node.lineno, "private_fixture_test_dependency"))
    return tuple(dict.fromkeys(findings))


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), stderr=subprocess.DEVNULL, timeout=30
    )


def check(root: Path, *, index: bool = False) -> tuple[int, tuple[Finding, ...]]:
    root = root.resolve()
    git_root = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if git_root != root:
        raise ValueError("scan_requires_repository_root_not_an_ancestor_repository")
    findings: list[Finding] = []
    count = 0
    if index:
        entries, errors = _parse_index_entries(_git(root, "ls-files", "--stage", "-z"))
        findings.extend(Finding(str(item.path), 0, item.reason) for item in errors)
        for entry in entries.values():
            name = entry.path.as_posix()
            count += 1
            if private_path(name):
                findings.append(Finding(name, 0, "private_file_path"))
                continue
            if entry.mode not in {"100644", "100755"}:
                findings.append(Finding(name, 0, "unsupported_git_mode"))
                continue
            if int(_git(root, "cat-file", "-s", entry.object_id)) > _MAX_TEXT_BYTES:
                findings.append(Finding(name, 0, "file_too_large"))
                continue
            findings.extend(
                inspect_public_file(name, _git(root, "cat-file", "blob", entry.object_id))
            )
    else:
        candidates, errors = _git_worktree_candidates(root)
        findings.extend(Finding(str(item.path), 0, item.reason) for item in errors)
        for candidate in dict.fromkeys(candidates):
            count += 1
            name = candidate.as_posix()
            if private_path(name):
                findings.append(Finding(name, 0, "private_file_path"))
                continue
            path, _, metadata, error = _validate_candidate_path(root, candidate)
            if error or path is None or metadata is None:
                findings.append(Finding(name, 0, error.reason if error else "file_invalid"))
                continue
            if metadata.st_size > _MAX_TEXT_BYTES:
                findings.append(Finding(name, 0, "file_too_large"))
                continue
            with path.open("rb") as handle:
                payload = handle.read(_MAX_TEXT_BYTES + 1)
            if len(payload) > _MAX_TEXT_BYTES:
                findings.append(Finding(name, 0, "file_too_large"))
                continue
            findings.extend(inspect_public_file(name, payload))
    return count, tuple(dict.fromkeys(findings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    try:
        count, findings = check(args.root, index=args.index)
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"passed": False, "error": "public_file_scan_failed"}))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "passed": not findings,
                "scope": "index" if args.index else "worktree",
                "files_checked": count,
                "findings": [asdict(item) for item in findings],
                "history_checked": False,
            },
            sort_keys=True,
        )
    )
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
