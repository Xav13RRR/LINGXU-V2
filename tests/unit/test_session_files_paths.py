from pathlib import Path

import pytest

from lingxu.storage.session_files import SessionFiles


def test_session_file_resolve_accepts_relative_path(tmp_path: Path) -> None:
    files = SessionFiles(tmp_path / "sessions")
    session_dir = files.root / "session-1"
    session_dir.mkdir()

    resolved = files.resolve("session-1", Path("audio/1/000001.flac.enc"))

    assert resolved == session_dir / "audio" / "1" / "000001.flac.enc"


def test_session_file_resolve_rejects_escape_and_absolute_path(tmp_path: Path) -> None:
    files = SessionFiles(tmp_path / "sessions")
    (files.root / "session-1").mkdir()

    with pytest.raises(ValueError, match="escapes"):
        files.resolve("session-1", Path("../other/file"))
    with pytest.raises(ValueError, match="must be relative"):
        files.resolve("session-1", (tmp_path / "absolute.enc").resolve())
