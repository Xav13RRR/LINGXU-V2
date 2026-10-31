from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from scripts.preview_storage_cleanup import plan_backups, plan_images


def _set_modified(path: Path, value: datetime) -> None:
    timestamp = value.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_backup_preview_protects_recent_and_unverified_archives(tmp_path: Path) -> None:
    old_valid = tmp_path / "lingxu-data-pre-old.tar.gz"
    recent_valid = tmp_path / "lingxu-data-pre-recent.tar.gz"
    malformed = tmp_path / "lingxu-data-pre-malformed.tar.gz"
    old_valid.write_bytes(b"\x1f\x8bsynthetic-old")
    recent_valid.write_bytes(b"\x1f\x8bsynthetic-recent")
    malformed.write_bytes(b"\xff\xfesynthetic-invalid")
    _set_modified(old_valid, datetime(2026, 7, 1, tzinfo=UTC))
    _set_modified(recent_valid, datetime(2026, 8, 25, tzinfo=UTC))
    _set_modified(malformed, datetime(2026, 7, 2, tzinfo=UTC))

    plan = plan_backups(
        tmp_path,
        now=datetime(2026, 8, 26, tzinfo=UTC),
        minimum_age_days=14,
        keep_count=1,
    )
    by_name = {item["name"]: item for item in plan["items"]}

    assert by_name[old_valid.name]["disposition"] == "review-candidate"
    assert by_name[recent_valid.name]["disposition"] == "protected"
    assert "newest-retention" in by_name[recent_valid.name]["reasons"]
    assert by_name[malformed.name]["disposition"] == "protected"
    assert "unverified-gzip-header" in by_name[malformed.name]["reasons"]
    assert plan["candidate_bytes"] == old_valid.stat().st_size


def test_image_preview_is_lingxu_scoped_and_protects_rollback_and_container_images() -> None:
    images = [
        {
            "Id": "sha256:current",
            "Created": "2026-06-01T00:00:00Z",
            "Size": 100,
            "RepoTags": ["lingxu-app:latest"],
            "Config": {"Labels": {}},
        },
        {
            "Id": "sha256:rollback",
            "Created": "2026-06-02T00:00:00Z",
            "Size": 200,
            "RepoTags": ["lingxu-app:rollback"],
            "Config": {"Labels": {}},
        },
        {
            "Id": "sha256:container",
            "Created": "2026-06-03T00:00:00Z",
            "Size": 300,
            "RepoTags": ["lingxu-check:old-container"],
            "Config": {"Labels": {}},
        },
        {
            "Id": "sha256:candidate",
            "Created": "2026-06-04T00:00:00Z",
            "Size": 400,
            "RepoTags": ["lingxu-check:old"],
            "Config": {"Labels": {}},
        },
        {
            "Id": "sha256:foreign",
            "Created": "2026-06-01T00:00:00Z",
            "Size": 500,
            "RepoTags": ["another-project:old"],
            "Config": {"Labels": {}},
        },
    ]

    plan = plan_images(
        images,
        container_image_ids={"sha256:container"},
        now=datetime(2026, 8, 26, tzinfo=UTC),
        minimum_age_days=14,
        keep_per_repository=0,
    )
    by_id = {item["id"]: item for item in plan["items"]}

    assert "sha256:foreign" not in by_id
    assert by_id["sha256:current"]["disposition"] == "protected"
    assert "latest-tag" in by_id["sha256:current"]["reasons"]
    assert by_id["sha256:rollback"]["disposition"] == "protected"
    assert "rollback-tag" in by_id["sha256:rollback"]["reasons"]
    assert by_id["sha256:container"]["disposition"] == "protected"
    assert "referenced-by-container" in by_id["sha256:container"]["reasons"]
    assert by_id["sha256:candidate"]["disposition"] == "review-candidate"
    assert plan["candidate_virtual_bytes"] == 400


def test_cleanup_preview_script_has_no_apply_or_delete_mode() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "preview_storage_cleanup.py"
    ).read_text(encoding="utf-8")

    assert "--apply" not in source
    assert "image rm" not in source
    assert "system prune" not in source
    assert ".unlink(" not in source
