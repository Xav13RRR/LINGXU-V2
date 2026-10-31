import json
from pathlib import Path

from lingxu.domain.enums import MeetingState
from lingxu.storage.session_files import SessionFiles, SessionManifest
from lingxu.utils import new_ulid


def test_session_files_are_atomic_and_cleanup_parts(tmp_path: Path) -> None:
    files = SessionFiles(tmp_path / "sessions")
    session_id = new_ulid()
    manifest = SessionManifest.create(
        session_id=session_id,
        guild_id=1,
        creator_user_id=2,
    )
    session_dir = files.create(manifest)

    digest = files.atomic_write(
        session_id,
        Path("meeting/agenda.json"),
        b'{"title":"test"}\n',
    )
    assert len(digest) == 64
    assert not list(session_dir.rglob("*.part"))

    files.append_jsonl(session_id, Path("meeting/events.jsonl"), {"sequence": 1})
    files.append_jsonl(session_id, Path("meeting/events.jsonl"), {"sequence": 2})
    values = [
        json.loads(line)
        for line in (session_dir / "meeting/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert values == [{"sequence": 1}, {"sequence": 2}]

    (session_dir / "audio/orphan.flac.part").write_bytes(b"incomplete")
    assert files.cleanup_parts() == 1
    assert files.update_state(session_id, MeetingState.AWAITING_PARTICIPANTS).state == (
        MeetingState.AWAITING_PARTICIPANTS.value
    )
