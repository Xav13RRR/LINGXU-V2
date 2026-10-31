"""Build a local, allowlisted candidate without Git history or publication actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from scripts.check_public_files import inspect_public_file, private_path
from scripts.scan_secrets import _MAX_TEXT_BYTES, _validate_candidate_path

_MANIFEST = ".publication-files.txt"
_BANNER = "docs/assets/lingxu-banner.png"
_BANNER_SHA256 = "ccce62a2e35cb210a9458a00a8c0ca667b518b6092204a9af27aeba251526a52"
_PUBLIC_REPOSITORY_URL = "https://github.com/Xav13RRR/LINGXU-V2"
_REQUIRED = {
    _MANIFEST,
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "PUBLICATION_STATUS.json",
    "docs/RIGHTS_CLEARANCE.md",
    "src/lingxu/__init__.py",
}
_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".worktrees",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "build",
    "htmlcov",
}


def candidate_path_allowed(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and "\\" not in name
        and ":" not in name
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and not _EXCLUDED_PARTS.intersection(part.lower() for part in path.parts)
        and not private_path(name)
        and not name.lower().startswith("docs/history/")
        and (not name.lower().startswith("docs/assets/") or name == _BANNER)
        and not name.endswith((".pyc", ".part"))
    )


def transform(name: str, payload: bytes) -> bytes:
    """Only change OCI source metadata, never application code or configuration values."""
    if name == "Dockerfile":
        text, count = re.subn(
            r'(?m)^ARG BUILD_SOURCE="[^"]*"',
            f'ARG BUILD_SOURCE="{_PUBLIC_REPOSITORY_URL}"',
            payload.decode("utf-8"),
        )
        if count != 2:
            raise ValueError("unexpected_docker_source_metadata")
        return text.encode("utf-8")
    if name == "docker-compose.yml":
        text, count = re.subn(
            r"BUILD_SOURCE: \$\{LINGXU_BUILD_SOURCE:-[^}]+\}",
            f"BUILD_SOURCE: ${{LINGXU_BUILD_SOURCE:-{_PUBLIC_REPOSITORY_URL}}}",
            payload.decode("utf-8"),
        )
        if count != 1:
            raise ValueError("unexpected_compose_source_metadata")
        return text.encode("utf-8")
    return payload


def load_inputs(root: Path) -> dict[str, bytes]:
    manifest, _, _, issue = _validate_candidate_path(root, Path(_MANIFEST))
    if issue or manifest is None:
        raise ValueError("invalid_file_allowlist")
    names = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if names != sorted(set(names)) or not _REQUIRED.issubset(names):
        raise ValueError("invalid_or_incomplete_file_allowlist")
    materials: dict[str, bytes] = {}
    for name in names:
        if not candidate_path_allowed(name):
            raise ValueError("forbidden_allowlist_path")
        source, _, metadata, issue = _validate_candidate_path(root, Path(name))
        if issue or source is None or metadata is None or metadata.st_size > _MAX_TEXT_BYTES:
            raise ValueError("unsafe_source_file")
        with source.open("rb") as handle:
            original = handle.read(_MAX_TEXT_BYTES + 1)
        if len(original) > _MAX_TEXT_BYTES or inspect_public_file(name, original):
            raise ValueError("source_failed_public_file_check")
        payload = transform(name, original)
        if inspect_public_file(name, payload):
            raise ValueError("transformed_file_failed_public_check")
        materials[name] = payload
    status = json.loads(materials["PUBLICATION_STATUS.json"])
    if (
        not isinstance(status, dict)
        or status.get("publication_authorized") is not False
        or status.get("license") != "Apache-2.0"
    ):
        raise ValueError("candidate_must_remain_on_release_hold")
    if _BANNER in materials:
        if (
            status.get("banner") != "owner_confirmed"
            or status.get("banner_license") != "Apache-2.0"
        ):
            raise ValueError("banner_authorization_missing")
        if hashlib.sha256(materials[_BANNER]).hexdigest() != _BANNER_SHA256:
            raise ValueError("banner_identity_mismatch")
    elif status.get("banner") == "owner_confirmed":
        raise ValueError("confirmed_banner_missing_from_candidate")
    return materials


def _output_path(root: Path, output: Path) -> Path:
    target = output if output.is_absolute() else root / output
    # Check every existing ancestor before resolve; junctions and symlinks are not export targets.
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink() or ancestor.is_junction():
            raise ValueError("linked_output_path")
    target = target.resolve()
    private_root = root / "private"
    if target == private_root or not target.is_relative_to(private_root):
        raise ValueError("output_must_be_new_directory_under_private")
    if target.exists():
        raise ValueError("refusing_existing_output")
    return target


def prepare(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    target = _output_path(root, output)
    materials = load_inputs(root)
    # A failure leaves a partial local directory for inspection, never deletes or overwrites data.
    target.mkdir(parents=True, exist_ok=False)
    source_root = target / "source"
    inventory = []
    for name, payload in materials.items():
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
        if name == ".githooks/pre-commit":
            destination.chmod(0o755)
        actual = destination.read_bytes()
        if actual != payload:
            raise ValueError("snapshot_copy_mismatch")
        inventory.append(
            {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    actual_paths = {
        p.relative_to(source_root).as_posix() for p in source_root.rglob("*") if p.is_file()
    }
    if actual_paths != set(materials) or (source_root / ".git").exists():
        raise ValueError("unexpected_snapshot_members")
    status = json.loads(materials["PUBLICATION_STATUS.json"])
    blockers = []
    if status.get("code_rights") != "owner_confirmed":
        blockers.append("migrated_code_rights_and_attribution")
    if status.get("public_repository_url") != _PUBLIC_REPOSITORY_URL:
        blockers.append("public_repository_identity")
    blockers.append("final_owner_publication_authorization")
    report: dict[str, object] = {
        "schema_version": 1,
        "preparation_passed": True,
        "ready_to_publish": False,
        "license": "Apache-2.0",
        "history_included": False,
        "banner_included": _BANNER in materials,
        "code_rights": status.get("code_rights"),
        "copyright_holder": status.get("copyright_holder"),
        "public_repository_url": status.get("public_repository_url"),
        "public_repository_created": False,
        "blockers": blockers,
        "files": inventory,
    }
    with (target / "candidate-review.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.root, args.output)
    except (OSError, ValueError, RuntimeError):
        print(
            json.dumps(
                {
                    "preparation_passed": False,
                    "ready_to_publish": False,
                    "error": "candidate_preparation_failed",
                }
            )
        )
        raise SystemExit(1) from None
    print(
        json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
