import threading
from pathlib import Path
from typing import Any, cast

import pytest

from lingxu.domain.enums import ConsentKind
from lingxu.domain.errors import RecordingPipelineError
from lingxu.storage.recording_service import RecordingCoordinator
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.meeting_consent import MeetingConsentRegistry
from lingxu.voice.segment_recorder import (
    FormalRecordingPipeline,
    MeetingRecordingQueue,
    RecordingMetrics,
    cleanup_stale_recording_plaintext,
)


def _frame(user_id: int, index: int) -> SessionAudioFrame:
    return SessionAudioFrame(
        session_id="session",
        user_id=user_id,
        received_monotonic_ns=index,
        pcm_s16le_48k_stereo=b"\x01\x00\x01\x00",
    )


def test_recording_queue_drops_oldest_frame_but_preserves_close_order() -> None:
    recording_queue = MeetingRecordingQueue(max_frames=2)

    assert recording_queue.put(_frame(1, 1)) is False
    closed = recording_queue.close_user(1)
    assert recording_queue.put(_frame(2, 2)) is False
    assert recording_queue.put(_frame(3, 3)) is True

    first = recording_queue.get()
    second = recording_queue.get()
    third = recording_queue.get()

    assert first.user_id == 1
    assert second.user_id == 2
    assert third.user_id == 3
    assert closed.is_set() is False


def test_formal_pipeline_counts_bounded_queue_drops() -> None:
    recording_queue = MeetingRecordingQueue(max_frames=1)
    metrics = RecordingMetrics("session")
    pipeline = FormalRecordingPipeline(
        session_id="session",
        consents=MeetingConsentRegistry(),
        recording_queue=recording_queue,
        recorder=cast(Any, object()),
        metrics=metrics,
    )
    pipeline.grant(1, ConsentKind.RECORDING)

    assert pipeline.receive(
        user_id=1,
        received_monotonic_ns=1,
        pcm_s16le_48k_stereo=b"\x01\x00\x01\x00",
    )
    assert pipeline.receive(
        user_id=1,
        received_monotonic_ns=2,
        pcm_s16le_48k_stereo=b"\x02\x00\x02\x00",
    )

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["frames_dropped_queue_full"] == 1
    assert snapshot["per_user"]["1"]["frames_dropped_queue_full"] == 1


def test_startup_cleanup_removes_only_recording_plaintext(tmp_path: Path) -> None:
    stale = tmp_path / "recording/session/user/000001.flac.part"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"plaintext")
    unrelated = tmp_path / "worker.ready"
    unrelated.write_text("ready", encoding="utf-8")

    assert cleanup_stale_recording_plaintext(tmp_path) == 1
    assert not (tmp_path / "recording").exists()
    assert unrelated.is_file()


@pytest.mark.asyncio
async def test_cutoff_withdrawal_surfaces_recorder_cleanup_failure() -> None:
    completed = threading.Event()
    completed.set()

    class FailedRecorder:
        failure = OSError("injected cleanup failure")

        @staticmethod
        def drain_completed() -> tuple[()]:
            return ()

    class FailedPipeline:
        recorder = FailedRecorder()

        @staticmethod
        def withdraw_at(
            *,
            user_id: int,
            effective_monotonic_ns: int,
        ) -> threading.Event:
            del user_id, effective_monotonic_ns
            return completed

    coordinator = RecordingCoordinator(
        session_id="session",
        pipeline=cast(Any, FailedPipeline()),
        segments=cast(Any, object()),
        files=cast(Any, object()),
    )

    with pytest.raises(RecordingPipelineError, match="failed during withdrawal"):
        await coordinator.withdraw_recording_at(1, 1_000)
