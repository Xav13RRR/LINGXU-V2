"""Content-safe, read-only audit of retained Meeting records and backups."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

_EXTERNAL_ARTIFACT_TYPES = frozenset(
    {
        "DISCORD_PUBLICATION",
        "DISCORD_PUBLICATION_INTENT",
    }
)
_COMPLETE_RECORD_ARTIFACT_TYPES = frozenset(
    {
        "DISCORD_PUBLICATION",
        "EVIDENCE_JSON",
        "MEETING_EVENTS_JSON",
        "MINUTES_MARKDOWN",
        "TRANSCRIPT_JSON",
        "TRANSCRIPT_TEXT",
    }
)
_CURRENT_PIPELINE_ARTIFACT_TYPES = frozenset(
    {
        "CANONICAL_TRANSCRIPT_JSON",
        "CONSENT_TIMELINE_JSON",
        "DISCORD_PUBLICATION",
        "DISCORD_PUBLICATION_INTENT",
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
    }
)


class RetainedAuditError(RuntimeError):
    """A closed, content-free audit setup failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_read_only_database(database_path: Path) -> sqlite3.Connection:
    if database_path.is_symlink() or not database_path.is_file():
        raise RetainedAuditError("database_not_regular")
    wal_path = database_path.with_name(f"{database_path.name}-wal")
    if wal_path.exists() and (
        wal_path.is_symlink() or not wal_path.is_file() or wal_path.stat().st_size
    ):
        raise RetainedAuditError("database_wal_not_checkpointed")
    uri = f"{database_path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise RetainedAuditError("database_open_failed") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _resolve_session_file(
    sessions_root: Path,
    session_id: str,
    stored_path: str,
    *,
    relative_to_session: bool,
) -> Path | None:
    """Resolve a stored path without permitting escape or any symlink component."""

    root = sessions_root.resolve()
    raw = PurePosixPath(stored_path.replace("\\", "/"))
    if raw.is_absolute():
        try:
            relative = raw.relative_to(PurePosixPath("/data/sessions"))
        except ValueError:
            return None
    elif relative_to_session:
        relative = PurePosixPath(session_id) / raw
    else:
        return None
    if not relative.parts or ".." in relative.parts:
        return None
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    return resolved


def _audit_database(connection: sqlite3.Connection) -> dict[str, object]:
    quick_rows = tuple(row[0] for row in connection.execute("PRAGMA quick_check"))
    foreign_key_failures = sum(1 for _ in connection.execute("PRAGMA foreign_key_check"))
    sessions = tuple(
        connection.execute(
            "SELECT id, next_sequence FROM meeting_sessions WHERE deleted_at IS NULL"
        )
    )
    invalid_event_streams = 0
    for session in sessions:
        sequences = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sequence_number FROM meeting_events "
                "WHERE session_id = ? ORDER BY sequence_number",
                (session["id"],),
            )
        )
        if sequences != tuple(range(1, len(sequences) + 1)) or session["next_sequence"] != len(
            sequences
        ):
            invalid_event_streams += 1

    relation_queries = {
        "audio_without_participant": (
            "SELECT count(*) FROM audio_segments a "
            "LEFT JOIN meeting_participants p ON p.session_id = a.session_id "
            "AND p.user_id = a.user_id WHERE p.id IS NULL"
        ),
        "audio_without_current_recording_consent": (
            "SELECT count(*) FROM audio_segments a "
            "JOIN meeting_participants p ON p.session_id = a.session_id "
            "AND p.user_id = a.user_id WHERE NOT p.consent_recording"
        ),
        "transcript_without_participant": (
            "SELECT count(*) FROM transcript_entries t "
            "LEFT JOIN meeting_participants p ON p.session_id = t.session_id "
            "AND p.user_id = t.user_id WHERE t.user_id IS NOT NULL AND p.id IS NULL"
        ),
        "transcript_without_current_minutes_consent": (
            "SELECT count(*) FROM transcript_entries t "
            "JOIN meeting_participants p ON p.session_id = t.session_id "
            "AND p.user_id = t.user_id WHERE NOT p.consent_minutes"
        ),
        "utterance_without_participant": (
            "SELECT count(*) FROM meeting_realtime_utterances u "
            "LEFT JOIN meeting_participants p ON p.session_id = u.session_id "
            "AND p.user_id = u.user_id WHERE p.id IS NULL"
        ),
        "utterance_without_current_asr_consent": (
            "SELECT count(*) FROM meeting_realtime_utterances u "
            "JOIN meeting_participants p ON p.session_id = u.session_id "
            "AND p.user_id = u.user_id WHERE NOT p.consent_realtime_asr"
        ),
    }
    relation_failures = {
        name: int(connection.execute(query).fetchone()[0])
        for name, query in relation_queries.items()
    }

    participants: dict[str, set[int]] = defaultdict(set)
    audio_users: dict[str, set[int]] = defaultdict(set)
    transcript_users: dict[str, set[int]] = defaultdict(set)
    audio_bounds: dict[str, tuple[int, int]] = {}
    artifact_types: dict[str, set[str]] = defaultdict(set)
    completed_jobs: set[str] = set()
    for row in connection.execute("SELECT session_id, user_id FROM meeting_participants"):
        participants[row["session_id"]].add(row["user_id"])
    for row in connection.execute(
        "SELECT session_id, user_id, started_ms, ended_ms FROM audio_segments"
    ):
        session_id = row["session_id"]
        audio_users[session_id].add(row["user_id"])
        bounds = audio_bounds.get(session_id)
        audio_bounds[session_id] = (
            min(bounds[0], row["started_ms"]) if bounds else row["started_ms"],
            max(bounds[1], row["ended_ms"]) if bounds else row["ended_ms"],
        )
    for row in connection.execute(
        "SELECT session_id, user_id FROM transcript_entries WHERE user_id IS NOT NULL"
    ):
        transcript_users[row["session_id"]].add(row["user_id"])
    for row in connection.execute("SELECT session_id, artifact_type FROM artifacts"):
        artifact_types[row["session_id"]].add(row["artifact_type"])
    for row in connection.execute(
        "SELECT DISTINCT session_id FROM processing_jobs WHERE status = 'COMPLETED'"
    ):
        completed_jobs.add(row["session_id"])

    session_ids = {str(row["id"]) for row in sessions}
    multi_participant = {
        session_id for session_id in session_ids if len(participants[session_id]) >= 2
    }
    multi_audio = {session_id for session_id in session_ids if len(audio_users[session_id]) >= 2}
    multi_transcript = {
        session_id for session_id in session_ids if len(transcript_users[session_id]) >= 2
    }
    ten_minute_multi_audio = {
        session_id
        for session_id in multi_audio
        if session_id in audio_bounds
        and audio_bounds[session_id][1] - audio_bounds[session_id][0] >= 600_000
    }
    complete_published = {
        session_id
        for session_id in session_ids
        if session_id in multi_participant
        and session_id in multi_audio
        and session_id in multi_transcript
        and session_id in completed_jobs
        and _COMPLETE_RECORD_ARTIFACT_TYPES <= artifact_types[session_id]
    }
    current_pipeline_complete = {
        session_id
        for session_id in session_ids
        if _CURRENT_PIPELINE_ARTIFACT_TYPES <= artifact_types[session_id]
    }

    passed = (
        quick_rows == ("ok",)
        and foreign_key_failures == 0
        and invalid_event_streams == 0
        and not any(relation_failures.values())
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "quick_check": "PASSED" if quick_rows == ("ok",) else "FAILED",
        "foreign_key_failures": foreign_key_failures,
        "invalid_event_streams": invalid_event_streams,
        "relation_failures": relation_failures,
        "profiles": {
            "sessions": len(session_ids),
            "multi_participant_sessions": len(multi_participant),
            "multi_user_audio_sessions": len(multi_audio),
            "multi_user_transcript_sessions": len(multi_transcript),
            "multi_user_audio_span_at_least_10_minutes": len(ten_minute_multi_audio),
            "complete_multi_user_published_sessions": len(complete_published),
            "current_pipeline_artifact_complete_sessions": len(current_pipeline_complete),
        },
    }


def _audit_files(connection: sqlite3.Connection, sessions_root: Path) -> dict[str, object]:
    if sessions_root.is_symlink() or not sessions_root.is_dir():
        raise RetainedAuditError("sessions_root_not_directory")
    manifest_failures = 0
    manifest_files = 0
    for row in connection.execute("SELECT id, manifest_path FROM meeting_sessions"):
        path = _resolve_session_file(
            sessions_root,
            row["id"],
            row["manifest_path"],
            relative_to_session=False,
        )
        if path is None:
            manifest_failures += 1
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            manifest_failures += 1
            continue
        if not isinstance(payload, dict):
            manifest_failures += 1
            continue
        manifest_files += 1

    audio_failures = 0
    audio_files = 0
    audio_bytes = 0
    duplicate_audio_paths = int(
        connection.execute(
            "SELECT count(*) FROM (SELECT session_id, encrypted_path FROM audio_segments "
            "GROUP BY session_id, encrypted_path HAVING count(*) > 1)"
        ).fetchone()[0]
    )
    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in connection.execute(
        "SELECT session_id, user_id, segment_number, encrypted_path, sha256, bytes_encrypted "
        "FROM audio_segments"
    ):
        tracks[(row["session_id"], row["user_id"])].append(row["segment_number"])
        path = _resolve_session_file(
            sessions_root,
            row["session_id"],
            row["encrypted_path"],
            relative_to_session=True,
        )
        if (
            path is None
            or path.stat().st_size != row["bytes_encrypted"]
            or _sha256(path) != row["sha256"]
        ):
            audio_failures += 1
            continue
        audio_files += 1
        audio_bytes += row["bytes_encrypted"]
    non_contiguous_tracks = 0
    for numbers in tracks.values():
        ordered = sorted(numbers)
        if ordered and ordered != list(range(ordered[0], ordered[-1] + 1)):
            non_contiguous_tracks += 1

    artifact_failures = 0
    artifact_files = 0
    external_artifacts = 0
    artifact_bytes = 0
    for row in connection.execute(
        "SELECT session_id, artifact_type, path, sha256, metadata_json FROM artifacts"
    ):
        if row["artifact_type"] in _EXTERNAL_ARTIFACT_TYPES:
            external_artifacts += 1
            continue
        path = _resolve_session_file(
            sessions_root,
            row["session_id"],
            row["path"],
            relative_to_session=False,
        )
        if path is None or _sha256(path) != row["sha256"]:
            artifact_failures += 1
            continue
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            artifact_failures += 1
            continue
        size = path.stat().st_size
        if isinstance(metadata, dict) and isinstance(metadata.get("size_bytes"), int):
            if metadata["size_bytes"] != size:
                artifact_failures += 1
                continue
        artifact_files += 1
        artifact_bytes += size

    passed = (
        manifest_failures == 0
        and audio_failures == 0
        and artifact_failures == 0
        and duplicate_audio_paths == 0
        and non_contiguous_tracks == 0
    )
    return {
        "status": "PASSED" if passed else "FAILED",
        "manifests": {"verified": manifest_files, "failures": manifest_failures},
        "encrypted_audio": {
            "verified": audio_files,
            "failures": audio_failures,
            "bytes_verified": audio_bytes,
            "distinct_user_tracks": len(tracks),
            "duplicate_paths": duplicate_audio_paths,
            "non_contiguous_tracks": non_contiguous_tracks,
        },
        "registered_file_artifacts": {
            "verified": artifact_files,
            "failures": artifact_failures,
            "bytes_verified": artifact_bytes,
        },
        "external_artifact_records": external_artifacts,
    }


def _audit_backups(backup_root: Path | None) -> dict[str, object]:
    if backup_root is None:
        return {"status": "NOT_RUN"}
    if backup_root.is_symlink() or not backup_root.is_dir():
        raise RetainedAuditError("backup_root_not_directory")
    archives = tuple(sorted(path for path in backup_root.glob("*.gz") if path.is_file()))
    valid_archives = 0
    invalid_archives = 0
    unsafe_member_archives = 0
    session_manifests = 0
    duplex_probe_metrics_members = 0
    for archive in archives:
        try:
            with archive.open("rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    invalid_archives += 1
                    continue
            with tarfile.open(archive, mode="r:gz") as bundle:
                members = bundle.getmembers()
        except (OSError, EOFError, tarfile.TarError):
            invalid_archives += 1
            continue
        unsafe = False
        for member in members:
            name = PurePosixPath(member.name.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                unsafe = True
            if len(name.parts) >= 2 and name.parts[-1] == "manifest.json":
                session_manifests += 1
            if member.isfile() and "duplex-probes" in name.parts and name.name == "metrics.json":
                duplex_probe_metrics_members += 1
        if unsafe:
            unsafe_member_archives += 1
        else:
            valid_archives += 1
    passed = bool(archives) and invalid_archives == 0 and unsafe_member_archives == 0
    return {
        "status": "PASSED" if passed else "FAILED",
        "archives": len(archives),
        "valid_tar_gzip_archives": valid_archives,
        "invalid_archives": invalid_archives,
        "unsafe_member_archives": unsafe_member_archives,
        "session_manifest_members": session_manifests,
        "duplex_probe_metrics_members": duplex_probe_metrics_members,
    }


def _audit_duplex_probes(probe_root: Path | None) -> dict[str, object]:
    if probe_root is None:
        return {"status": "NOT_RUN"}
    if probe_root.is_symlink() or not probe_root.is_dir():
        raise RetainedAuditError("probe_root_not_directory")
    metrics_files = sum(1 for path in probe_root.rglob("metrics.json") if path.is_file())
    return {
        "status": "EVIDENCE_FOUND" if metrics_files else "EVIDENCE_MISSING",
        "metrics_files": metrics_files,
    }


def audit_retained_meetings(
    database_path: Path,
    sessions_root: Path,
    *,
    backup_root: Path | None = None,
    probe_root: Path | None = None,
) -> dict[str, object]:
    """Audit retained records without returning content, identifiers, or local paths."""

    connection = _open_read_only_database(database_path)
    try:
        database = _audit_database(connection)
        files = _audit_files(connection, sessions_root)
    except sqlite3.Error as exc:
        raise RetainedAuditError("database_query_failed") from exc
    finally:
        connection.close()
    backups = _audit_backups(backup_root)
    probes = _audit_duplex_probes(probe_root)
    sections = (database, files, backups)
    failed = any(section.get("status") == "FAILED" for section in sections)
    return {
        "status": "FAILED" if failed else "PASSED",
        "content_safe": True,
        "database": database,
        "files": files,
        "backups": backups,
        "duplex_probes": probes,
    }
