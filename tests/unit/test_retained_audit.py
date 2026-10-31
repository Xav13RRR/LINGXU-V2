from __future__ import annotations

import gzip
import hashlib
import io
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from lingxu.cli.retained_audit import RetainedAuditError, audit_retained_meetings

_FILE_ARTIFACTS = (
    "CANONICAL_TRANSCRIPT_JSON",
    "CONSENT_TIMELINE_JSON",
    "EVIDENCE_JSON",
    "LANGUAGE_DISTRIBUTION_JSON",
    "MEETING_EVENTS_JSON",
    "MINUTES_JSON",
    "MINUTES_MARKDOWN",
    "OUTCOME_BUDGET_JOURNAL_JSON",
    "OUTCOME_PARTITION_MANIFEST_JSON",
    "OUTCOME_PARTITION_RESULTS_JSON",
    "PRIVACY_EXCLUSIONS_JSON",
    "TOPIC_SEGMENTS_JSON",
    "TRANSCRIPT_JSON",
    "TRANSCRIPT_TEXT",
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _create_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sessions = tmp_path / "sessions"
    meeting = sessions / "session-private" / "meeting"
    audio = sessions / "session-private" / "audio"
    meeting.mkdir(parents=True)
    audio.mkdir()
    manifest = sessions / "session-private" / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1}), encoding="utf-8")
    database = tmp_path / "lingxu.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE meeting_sessions (
          id TEXT PRIMARY KEY, next_sequence INTEGER NOT NULL,
          manifest_path TEXT NOT NULL, deleted_at TEXT
        );
        CREATE TABLE meeting_events (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence_number INTEGER NOT NULL
        );
        CREATE TABLE meeting_participants (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id INTEGER NOT NULL,
          consent_recording INTEGER NOT NULL, consent_minutes INTEGER NOT NULL,
          consent_realtime_asr INTEGER NOT NULL
        );
        CREATE TABLE audio_segments (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id INTEGER NOT NULL,
          segment_number INTEGER NOT NULL, encrypted_path TEXT NOT NULL,
          sha256 TEXT NOT NULL, started_ms INTEGER NOT NULL, ended_ms INTEGER NOT NULL,
          bytes_encrypted INTEGER NOT NULL
        );
        CREATE TABLE transcript_entries (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id INTEGER
        );
        CREATE TABLE meeting_realtime_utterances (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, user_id INTEGER NOT NULL
        );
        CREATE TABLE artifacts (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, artifact_type TEXT NOT NULL,
          path TEXT NOT NULL, sha256 TEXT NOT NULL, metadata_json TEXT
        );
        CREATE TABLE processing_jobs (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, status TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO meeting_sessions VALUES (?, ?, ?, NULL)",
        ("session-private", 2, "/data/sessions/session-private/manifest.json"),
    )
    connection.executemany(
        "INSERT INTO meeting_events VALUES (?, ?, ?)",
        (("event-a", "session-private", 1), ("event-b", "session-private", 2)),
    )
    connection.executemany(
        "INSERT INTO meeting_participants VALUES (?, ?, ?, 1, 1, 1)",
        (
            ("participant-a", "session-private", 101),
            ("participant-b", "session-private", 202),
        ),
    )
    for index, user_id in enumerate((101, 202), start=1):
        payload = f"encrypted-{index}".encode()
        path = audio / f"track-{index}.bin"
        path.write_bytes(payload)
        connection.execute(
            "INSERT INTO audio_segments VALUES (?, ?, ?, 0, ?, ?, 0, 600001, ?)",
            (
                f"audio-{index}",
                "session-private",
                user_id,
                f"audio/track-{index}.bin",
                _digest(payload),
                len(payload),
            ),
        )
        connection.execute(
            "INSERT INTO transcript_entries VALUES (?, ?, ?)",
            (f"transcript-{index}", "session-private", user_id),
        )
        connection.execute(
            "INSERT INTO meeting_realtime_utterances VALUES (?, ?, ?)",
            (f"utterance-{index}", "session-private", user_id),
        )
    for index, artifact_type in enumerate(_FILE_ARTIFACTS):
        payload = f"artifact-{index}".encode()
        filename = f"artifact-{index}.json"
        (meeting / filename).write_bytes(payload)
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"artifact-{index}",
                "session-private",
                artifact_type,
                f"/data/sessions/session-private/meeting/{filename}",
                _digest(payload),
                json.dumps({"size_bytes": len(payload)}),
            ),
        )
    for artifact_type in ("DISCORD_PUBLICATION", "DISCORD_PUBLICATION_INTENT"):
        connection.execute(
            "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, '{}')",
            (
                artifact_type,
                "session-private",
                artifact_type,
                "discord://redacted",
                "0" * 64,
            ),
        )
    connection.execute(
        "INSERT INTO processing_jobs VALUES ('job-private', 'session-private', 'COMPLETED')"
    )
    connection.commit()
    connection.close()

    backups = tmp_path / "backups"
    backups.mkdir()
    with tarfile.open(backups / "valid.tar.gz", "w:gz") as bundle:
        payload = b"{}"
        info = tarfile.TarInfo("sessions/redacted/manifest.json")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    probes = tmp_path / "duplex-probes"
    probes.mkdir()
    return database, sessions, backups, probes


def test_complete_retained_fixture_passes_without_leaking_identifiers(tmp_path: Path) -> None:
    database, sessions, backups, probes = _create_fixture(tmp_path)

    result = audit_retained_meetings(
        database,
        sessions,
        backup_root=backups,
        probe_root=probes,
    )

    assert result["status"] == "PASSED"
    assert result["database"]["profiles"] == {
        "sessions": 1,
        "multi_participant_sessions": 1,
        "multi_user_audio_sessions": 1,
        "multi_user_transcript_sessions": 1,
        "multi_user_audio_span_at_least_10_minutes": 1,
        "complete_multi_user_published_sessions": 1,
        "current_pipeline_artifact_complete_sessions": 1,
    }
    assert result["duplex_probes"] == {"status": "EVIDENCE_MISSING", "metrics_files": 0}
    serialized = json.dumps(result)
    assert "session-private" not in serialized
    assert "participant-a" not in serialized
    assert str(tmp_path) not in serialized


def test_tamper_and_invalid_backup_fail_closed(tmp_path: Path) -> None:
    database, sessions, backups, probes = _create_fixture(tmp_path)
    (sessions / "session-private" / "audio" / "track-1.bin").write_bytes(b"tampered")
    with gzip.open(backups / "truncated.gz", "wb") as handle:
        handle.write(b"not-a-tar")

    result = audit_retained_meetings(
        database,
        sessions,
        backup_root=backups,
        probe_root=probes,
    )

    assert result["status"] == "FAILED"
    assert result["files"]["encrypted_audio"]["failures"] == 1
    assert result["backups"]["invalid_archives"] == 1


def test_nonempty_wal_is_rejected_before_immutable_read(tmp_path: Path) -> None:
    database, sessions, backups, probes = _create_fixture(tmp_path)
    database.with_name(f"{database.name}-wal").write_bytes(b"uncheckpointed")

    with pytest.raises(RetainedAuditError, match="database_wal_not_checkpointed"):
        audit_retained_meetings(
            database,
            sessions,
            backup_root=backups,
            probe_root=probes,
        )
