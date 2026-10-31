import hashlib
import json
from pathlib import Path

import pytest

import scripts.prepare_publication as publication
from scripts.prepare_publication import _REQUIRED, candidate_path_allowed, prepare, transform


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/absolute",
        "dir\\windows",
        "./normalized",
        ".git/config",
        ".git",
        ".venv/key",
        "private/record.md",
        ".env",
        ".env.production",
        "data/transcript.json",
        "docs/assets/unreviewed.png",
        "Docs/Assets/unreviewed.png",
        "docs/history/old.md",
        "dump.db",
    ],
)
def test_export_rejects_private_and_ambiguous_paths(name: str) -> None:
    assert not candidate_path_allowed(name)


@pytest.mark.parametrize("name", ["src/lingxu/config.py", ".env.example", "LICENSE", "NOTICE"])
def test_export_keeps_public_source_paths(name: str) -> None:
    assert candidate_path_allowed(name)


def _fixture(root: Path) -> None:
    for name in _REQUIRED - {".publication-files.txt"}:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "# Synthetic public source\n" if path.suffix == ".py" else "Synthetic public source\n"
        )
        path.write_text(content, encoding="utf-8")
    (root / "PUBLICATION_STATUS.json").write_text(
        json.dumps({"publication_authorized": False, "license": "Apache-2.0"}), encoding="utf-8"
    )
    (root / ".publication-files.txt").write_text(
        "\n".join(sorted(_REQUIRED)) + "\n", encoding="utf-8"
    )


def test_snapshot_has_exact_inventory_no_history_and_no_unlisted_files(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / ".env").write_text("unexported placeholder", encoding="utf-8")
    (tmp_path / "unlisted.md").write_text("Do not export", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/config").write_text("private repository", encoding="utf-8")
    target = tmp_path / "private/candidate"
    report = prepare(tmp_path, target)
    persisted = json.loads((target / "candidate-review.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert not persisted["ready_to_publish"]
    assert not (target / "source/.git").exists()
    assert not (target / "source/.env").exists()
    assert not (target / "source/unlisted.md").exists()
    for entry in persisted["files"]:
        payload = (target / "source" / entry["path"]).read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    with pytest.raises(ValueError, match="existing_output"):
        prepare(tmp_path, target)


def test_invalid_input_fails_before_creating_export(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "README.md").write_text("sk-" + "a" * 32, encoding="utf-8")
    target = tmp_path / "private/candidate"
    with pytest.raises(ValueError, match="public_file_check"):
        prepare(tmp_path, target)
    assert not target.exists()


def test_output_cannot_target_source_or_outside_workspace(tmp_path: Path) -> None:
    _fixture(tmp_path)
    for target in (tmp_path, tmp_path / "private", tmp_path.parent / "candidate-outside"):
        with pytest.raises(ValueError):
            prepare(tmp_path, target)


def test_only_candidate_build_source_metadata_is_transformed() -> None:
    docker = b'ARG BUILD_SOURCE="https://example.invalid/archive"\n' * 2
    url = publication._PUBLIC_REPOSITORY_URL
    assert transform("Dockerfile", docker) == f'ARG BUILD_SOURCE="{url}"\n'.encode() * 2
    compose = b"    BUILD_SOURCE: ${LINGXU_BUILD_SOURCE:-https://example.invalid/archive}\n"
    assert transform("docker-compose.yml", compose).endswith(f"SOURCE:-{url}}}\n".encode())
    application = b"# synthetic application\n"
    assert transform("src/lingxu/config.py", application) is application
    with pytest.raises(ValueError):
        transform("Dockerfile", b"changed metadata shape")


def test_apache_metadata_and_owner_confirmed_banner_are_explicit() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert set(project["license-files"]) == {"LICENSE", "NOTICE"}
    assert "![" in (root / "README.md").read_text(encoding="utf-8")
    assert "OWNER_CONFIRMED_APACHE_2_0" in (root / "docs/ASSET_PROVENANCE.md").read_text(
        encoding="utf-8"
    )
    assert project["urls"]["Repository"] == publication._PUBLIC_REPOSITORY_URL
    assert "Copyright 2026 Xav13RRR." in (root / "NOTICE").read_text(encoding="utf-8")


def _add_banner(root: Path, *, authorized: bool) -> bytes:
    payload = b"synthetic banner bytes"
    path = root / publication._BANNER
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    (root / ".publication-files.txt").write_text(
        "\n".join(sorted(_REQUIRED | {publication._BANNER})) + "\n", encoding="utf-8"
    )
    (root / "PUBLICATION_STATUS.json").write_text(
        json.dumps(
            {
                "publication_authorized": False,
                "license": "Apache-2.0",
                "banner": "owner_confirmed" if authorized else "pending_owner_confirmation",
                "banner_license": "Apache-2.0",
                "code_rights": "owner_confirmed",
                "copyright_holder": "Xav13RRR",
                "public_repository_url": publication._PUBLIC_REPOSITORY_URL,
            }
        ),
        encoding="utf-8",
    )
    return payload


def test_banner_requires_owner_confirmation_before_export(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _add_banner(tmp_path, authorized=False)
    with pytest.raises(ValueError, match="banner_authorization_missing"):
        prepare(tmp_path, tmp_path / "private/candidate")
    assert not (tmp_path / "private/candidate").exists()


def test_banner_replacement_fails_closed(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _add_banner(tmp_path, authorized=True)
    with pytest.raises(ValueError, match="banner_identity_mismatch"):
        prepare(tmp_path, tmp_path / "private/candidate")
    assert not (tmp_path / "private/candidate").exists()


def test_cleared_rights_do_not_authorize_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)
    payload = _add_banner(tmp_path, authorized=True)
    monkeypatch.setattr(publication, "_BANNER_SHA256", hashlib.sha256(payload).hexdigest())
    report = prepare(tmp_path, tmp_path / "private/candidate")
    assert report["banner_included"] is True
    assert report["blockers"] == ["final_owner_publication_authorization"]
    assert report["ready_to_publish"] is False
    assert report["public_repository_created"] is False


def test_exporter_refuses_to_turn_into_a_release_tool(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "PUBLICATION_STATUS.json").write_text(
        json.dumps({"publication_authorized": True, "license": "Apache-2.0"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="release_hold"):
        prepare(tmp_path, tmp_path / "private/candidate")
