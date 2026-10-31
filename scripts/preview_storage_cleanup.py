"""Produce a content-safe, read-only local storage cleanup preview.

The report deliberately has no mutation mode. A ``review-candidate`` is only an
operator review hint; it is never automatic approval to remove an image or backup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_GZIP_MAGIC = b"\x1f\x8b"


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item != "<none>:<none>"]


def _repository(reference: str) -> str:
    without_digest = reference.split("@", maxsplit=1)[0]
    slash = without_digest.rfind("/")
    colon = without_digest.rfind(":")
    return without_digest[:colon] if colon > slash else without_digest


def _is_lingxu_image(image: dict[str, Any]) -> bool:
    tags = _string_list(image.get("RepoTags"))
    if any(_repository(tag).split("/")[-1].lower().startswith("lingxu") for tag in tags):
        return True
    config = image.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        return False
    source = str(labels.get("org.opencontainers.image.source", "")).rstrip("/").lower()
    return source.endswith("/lingxu")


def plan_images(
    images: Sequence[dict[str, Any]],
    *,
    container_image_ids: set[str],
    now: datetime,
    minimum_age_days: int,
    keep_per_repository: int,
) -> dict[str, Any]:
    """Classify only Lingxu images without mutating Docker state."""
    if minimum_age_days < 0 or keep_per_repository < 0:
        raise ValueError("retention values must not be negative")
    now = now.astimezone(UTC)
    scoped = [image for image in images if _is_lingxu_image(image)]
    retained_ids: set[str] = set()
    by_repository: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in scoped:
        for repository in {_repository(tag) for tag in _string_list(image.get("RepoTags"))}:
            by_repository[repository].append(image)
    for repository_images in by_repository.values():
        newest = sorted(
            repository_images,
            key=lambda item: _parse_datetime(str(item["Created"])),
            reverse=True,
        )[:keep_per_repository]
        retained_ids.update(str(image["Id"]) for image in newest)

    cutoff = now - timedelta(days=minimum_age_days)
    items: list[dict[str, Any]] = []
    candidate_virtual_bytes = 0
    for image in sorted(scoped, key=lambda item: str(item["Id"])):
        image_id = str(image["Id"])
        created = _parse_datetime(str(image["Created"]))
        tags = sorted(_string_list(image.get("RepoTags")))
        reasons: list[str] = []
        if image_id in container_image_ids:
            reasons.append("referenced-by-container")
        if any(tag.endswith(":latest") for tag in tags):
            reasons.append("latest-tag")
        if any(tag.endswith(":rollback") for tag in tags):
            reasons.append("rollback-tag")
        if image_id in retained_ids:
            reasons.append("newest-retention")
        if created > cutoff:
            reasons.append("minimum-age")
        size = int(image.get("Size", 0))
        disposition = "protected" if reasons else "review-candidate"
        if disposition == "review-candidate":
            candidate_virtual_bytes += size
        items.append(
            {
                "id": image_id,
                "tags": tags,
                "created_at": created.isoformat(),
                "virtual_bytes": size,
                "disposition": disposition,
                "reasons": reasons,
            }
        )
    return {
        "scope": "lingxu-images-only",
        "items": items,
        "candidate_count": sum(item["disposition"] == "review-candidate" for item in items),
        "candidate_virtual_bytes": candidate_virtual_bytes,
        "size_warning": (
            "Virtual image sizes can share layers and are not additive reclaimable bytes."
        ),
    }


def plan_backups(
    backup_root: Path,
    *,
    now: datetime,
    minimum_age_days: int,
    keep_count: int,
) -> dict[str, Any]:
    """Classify archive names and metadata without reading archive content."""
    if minimum_age_days < 0 or keep_count < 0:
        raise ValueError("retention values must not be negative")
    now = now.astimezone(UTC)
    paths = sorted(backup_root.glob("lingxu-data-*.tar.gz")) if backup_root.is_dir() else []
    metadata: list[tuple[Path, datetime, int, bool]] = []
    for path in paths:
        stat = path.stat()
        modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        with path.open("rb") as stream:
            has_gzip_header = stream.read(2) == _GZIP_MAGIC
        metadata.append((path, modified, stat.st_size, has_gzip_header))
    valid_newest = sorted(
        (item for item in metadata if item[3]),
        key=lambda item: item[1],
        reverse=True,
    )
    newest = {path.name for path, _modified, _size, _valid in valid_newest[:keep_count]}
    cutoff = now - timedelta(days=minimum_age_days)
    items: list[dict[str, Any]] = []
    candidate_bytes = 0
    for path, modified, size, has_gzip_header in metadata:
        reasons: list[str] = []
        if not has_gzip_header:
            reasons.append("unverified-gzip-header")
        if path.name in newest:
            reasons.append("newest-retention")
        if modified > cutoff:
            reasons.append("minimum-age")
        disposition = "protected" if reasons else "review-candidate"
        if disposition == "review-candidate":
            candidate_bytes += size
        items.append(
            {
                "name": path.name,
                "modified_at": modified.isoformat(),
                "bytes": size,
                "disposition": disposition,
                "reasons": reasons,
            }
        )
    return {
        "root": str(backup_root.resolve()),
        "items": items,
        "candidate_count": sum(item["disposition"] == "review-candidate" for item in items),
        "candidate_bytes": candidate_bytes,
        "validation_warning": (
            "A gzip header is not archive-integrity evidence; validate every "
            "candidate before review."
        ),
    }


def _docker_json(arguments: Sequence[str]) -> Any:
    completed = subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def _docker_lines(arguments: Sequence[str]) -> list[str]:
    completed = subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def inspect_docker() -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    image_ids = sorted(set(_docker_lines(["image", "ls", "--no-trunc", "--quiet"])))
    images = _docker_json(["image", "inspect", *image_ids]) if image_ids else []
    container_ids = _docker_lines(["container", "ls", "--all", "--no-trunc", "--quiet"])
    containers = _docker_json(["container", "inspect", *container_ids]) if container_ids else []
    referenced = {str(container["Image"]) for container in containers}
    disk_rows = [
        json.loads(line) for line in _docker_lines(["system", "df", "--format", "{{json .}}"])
    ]
    return images, referenced, disk_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=_ROOT / "backups")
    parser.add_argument("--minimum-age-days", type=int, default=14)
    parser.add_argument("--keep-backups", type=int, default=3)
    parser.add_argument("--keep-images-per-repository", type=int, default=2)
    parser.add_argument("--skip-docker", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    now = datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "preview-only",
        "generated_at": now.isoformat(),
        "warning": "Review candidates are not approved for removal; this command changes nothing.",
        "backups": plan_backups(
            arguments.backup_root,
            now=now,
            minimum_age_days=arguments.minimum_age_days,
            keep_count=arguments.keep_backups,
        ),
    }
    if not arguments.skip_docker:
        images, referenced, disk_rows = inspect_docker()
        report["docker"] = plan_images(
            images,
            container_image_ids=referenced,
            now=now,
            minimum_age_days=arguments.minimum_age_days,
            keep_per_repository=arguments.keep_images_per_repository,
        )
        report["docker"]["system_disk_usage"] = disk_rows
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
