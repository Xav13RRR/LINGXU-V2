import struct
from pathlib import Path

import pytest
import soundfile

from lingxu.domain.enums import ConsentKind
from lingxu.storage.database import Database
from lingxu.storage.encryption import FileCipher
from lingxu.storage.recording_service import RecordingCoordinator
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingSessionRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.voice.meeting_consent import MeetingConsentRegistry
from lingxu.voice.segment_recorder import (
    EncryptedSegmentRecorder,
    FormalRecordingPipeline,
    MeetingRecordingQueue,
    RecordingMetrics,
)

_FRAME_NS = 20_000_000
_BASE_NS = 1_000_000_000
_VOICE_FRAME = struct.pack("<hh", 2_000, -2_000) * 960
_SILENT_FRAME = b"\0" * len(_VOICE_FRAME)


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def _coordinator(
    *,
    session_id: str,
    session_files: SessionFiles,
    runtime_tmp: Path,
    cipher: FileCipher,
    segments: AudioSegmentRepository,
    max_segment_seconds: float = 0.04,
) -> RecordingCoordinator:
    consents = MeetingConsentRegistry()
    recording_queue = MeetingRecordingQueue(max_frames=20)
    metrics = RecordingMetrics(session_id)
    recorder = EncryptedSegmentRecorder(
        session_id=session_id,
        session_started_monotonic_ns=_BASE_NS,
        recording_queue=recording_queue,
        session_files=session_files,
        runtime_tmp=runtime_tmp,
        cipher=cipher,
        metrics=metrics,
        silence_split_ms=40,
        silence_threshold=100,
        max_segment_seconds=max_segment_seconds,
    )
    pipeline = FormalRecordingPipeline(
        session_id=session_id,
        consents=consents,
        recording_queue=recording_queue,
        recorder=recorder,
        metrics=metrics,
    )
    return RecordingCoordinator(
        session_id=session_id,
        pipeline=pipeline,
        segments=segments,
        files=session_files,
    )


@pytest.mark.asyncio
async def test_consent_segment_restart_encryption_and_personal_deletion(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "recording.db"))
    await database.create_schema()
    session_files = SessionFiles(tmp_path / "sessions")
    session_repository = MeetingSessionRepository(database.sessions)
    session = await MeetingSessionService(session_repository, session_files).create(
        guild_id=100,
        creator_user_id=200,
    )
    segments = AudioSegmentRepository(database.sessions)
    cipher = FileCipher.from_secret(FileCipher.generate_key())
    runtime_tmp = tmp_path / "runtime"
    coordinator = _coordinator(
        session_id=session.id,
        session_files=session_files,
        runtime_tmp=runtime_tmp,
        cipher=cipher,
        segments=segments,
    )
    coordinator.start()
    pipeline = coordinator.pipeline

    assert (
        pipeline.receive(
            user_id=9,
            received_monotonic_ns=_BASE_NS,
            pcm_s16le_48k_stereo=_VOICE_FRAME,
        )
        is False
    )
    pipeline.grant(1, ConsentKind.RECORDING)
    pipeline.grant(1, ConsentKind.MINUTES)
    for index in range(3):
        assert pipeline.receive(
            user_id=1,
            received_monotonic_ns=_BASE_NS + index * _FRAME_NS,
            pcm_s16le_48k_stereo=_VOICE_FRAME,
        )
    await coordinator.withdraw_recording(1)
    assert (
        pipeline.receive(
            user_id=1,
            received_monotonic_ns=_BASE_NS + 4 * _FRAME_NS,
            pcm_s16le_48k_stereo=_VOICE_FRAME,
        )
        is False
    )

    pipeline.grant(2, ConsentKind.RECORDING)
    for index, payload in enumerate((_VOICE_FRAME, _SILENT_FRAME, _SILENT_FRAME), start=5):
        assert pipeline.receive(
            user_id=2,
            received_monotonic_ns=_BASE_NS + index * _FRAME_NS,
            pcm_s16le_48k_stereo=payload,
        )
    await coordinator.stop()

    user_one = await segments.list_user(session_id=session.id, user_id=1)
    user_two = await segments.list_user(session_id=session.id, user_id=2)
    assert [segment.segment_number for segment in user_one] == [1, 2]
    assert [segment.segment_number for segment in user_two] == [1]
    assert not (session_files.directory(session.id) / "audio/9").exists()
    assert (
        pipeline.metrics.snapshot()["per_user"]["9"]["frames_rejected_without_recording_consent"]
        == 1
    )

    encrypted_paths = [
        session_files.directory(session.id) / segment.encrypted_path
        for segment in (*user_one, *user_two)
    ]
    original_hashes = {path: path.read_bytes() for path in encrypted_paths}
    for index, encrypted in enumerate(encrypted_paths):
        assert not encrypted.read_bytes().startswith(b"fLaC")
        restored = tmp_path / f"restored-{index}.flac"
        cipher.decrypt_file(encrypted, restored)
        info = soundfile.info(str(restored))
        assert info.samplerate == 48_000
        assert info.channels == 2
    assert not list(runtime_tmp.rglob("*.flac*"))
    assert not list(session_files.directory(session.id).rglob("*.part"))

    restarted = _coordinator(
        session_id=session.id,
        session_files=session_files,
        runtime_tmp=runtime_tmp,
        cipher=cipher,
        segments=segments,
        max_segment_seconds=1.0,
    )
    restarted.start()
    restarted.pipeline.grant(1, ConsentKind.RECORDING)
    assert restarted.pipeline.receive(
        user_id=1,
        received_monotonic_ns=_BASE_NS + 20 * _FRAME_NS,
        pcm_s16le_48k_stereo=_VOICE_FRAME,
    )
    await restarted.withdraw_recording(1)

    user_one_after_restart = await segments.list_user(session_id=session.id, user_id=1)
    assert [segment.segment_number for segment in user_one_after_restart] == [1, 2, 3]
    for path, payload in original_hashes.items():
        assert path.read_bytes() == payload
    assert not list(runtime_tmp.rglob("*.flac*"))

    await restarted.delete_user_data(1)
    await restarted.stop()
    assert not (session_files.directory(session.id) / "audio/1").exists()
    assert await segments.list_user(session_id=session.id, user_id=1) == ()
    assert (session_files.directory(session.id) / "audio/2/000001.flac.enc").is_file()
    assert len(await segments.list_user(session_id=session.id, user_id=2)) == 1
    await database.close()
