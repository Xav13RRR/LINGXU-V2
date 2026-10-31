import threading
from typing import Any, cast

from lingxu.domain.enums import ConsentKind
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.meeting_consent import MeetingConsentRegistry
from lingxu.voice.segment_recorder import (
    FormalRecordingPipeline,
    MeetingRecordingQueue,
    RecordingMetrics,
)


def test_three_consent_purposes_are_independent_and_snapshot_is_immutable() -> None:
    consents = MeetingConsentRegistry()

    assert consents.grant(10, ConsentKind.REALTIME_ASR) is True
    assert consents.grant(10, ConsentKind.MINUTES) is True
    snapshot = consents.snapshot()

    assert snapshot.allows(10, ConsentKind.REALTIME_ASR)
    assert snapshot.allows(10, ConsentKind.MINUTES)
    assert not snapshot.allows(10, ConsentKind.RECORDING)

    assert consents.grant(10, ConsentKind.RECORDING) is True
    assert not snapshot.allows(10, ConsentKind.RECORDING)
    assert consents.snapshot().allows(10, ConsentKind.RECORDING)

    assert consents.withdraw(10, ConsentKind.REALTIME_ASR) is True
    assert consents.snapshot().allows(10, ConsentKind.MINUTES)
    assert consents.snapshot().allows(10, ConsentKind.RECORDING)


def test_consent_action_is_linearised_with_withdrawal() -> None:
    consents = MeetingConsentRegistry()
    actions: list[str] = []

    accepted, _ = consents.perform_if_consented(
        20,
        ConsentKind.RECORDING,
        1,
        lambda: actions.append("before"),
    )
    assert accepted is False

    consents.grant(20, ConsentKind.RECORDING)
    accepted_negative, _ = consents.perform_if_consented(
        20,
        ConsentKind.RECORDING,
        -1,
        lambda: actions.append("invalid"),
    )
    accepted, _ = consents.perform_if_consented(
        20,
        ConsentKind.RECORDING,
        2,
        lambda: actions.append("accepted"),
    )
    consents.withdraw(20, ConsentKind.RECORDING)

    assert accepted_negative is False
    assert accepted is True
    assert actions == ["accepted"]


def test_withdrawal_waits_for_an_inflight_consented_action() -> None:
    consents = MeetingConsentRegistry()
    consents.grant(30, ConsentKind.RECORDING)
    entered = threading.Event()
    release = threading.Event()
    withdrawn = threading.Event()

    def action() -> None:
        entered.set()
        assert release.wait(2)

    receiver = threading.Thread(
        target=lambda: consents.perform_if_consented(
            30,
            ConsentKind.RECORDING,
            1,
            action,
        )
    )
    withdrawal = threading.Thread(
        target=lambda: (
            consents.withdraw(30, ConsentKind.RECORDING),
            withdrawn.set(),
        )
    )
    receiver.start()
    assert entered.wait(2)
    withdrawal.start()
    assert not withdrawn.wait(0.05)
    release.set()
    receiver.join(2)
    withdrawal.join(2)

    assert withdrawn.is_set()
    assert not consents.snapshot().allows(30, ConsentKind.RECORDING)


def test_recording_withdrawal_rejects_cutoff_and_discards_queued_post_cutoff_frame() -> None:
    user_id = 40
    start_ns = 8_000_000_000
    cutoff_ns = start_ns + 1_000_000_000
    recording_queue = MeetingRecordingQueue(max_frames=10)

    class QueueBackedRecorder:
        def request_close_user(
            self,
            selected_user_id: int,
            effective_monotonic_ns: int | None = None,
        ) -> threading.Event:
            return recording_queue.close_user(
                selected_user_id,
                effective_monotonic_ns=effective_monotonic_ns,
            )

    pipeline = FormalRecordingPipeline(
        session_id="session",
        consents=MeetingConsentRegistry(),
        recording_queue=recording_queue,
        recorder=cast(Any, QueueBackedRecorder()),
        metrics=RecordingMetrics("session"),
    )
    pcm = b"\x01\x00\x01\x00"
    pipeline.grant(user_id, ConsentKind.RECORDING)

    assert pipeline.receive(
        user_id=user_id,
        received_monotonic_ns=start_ns + 999_000_000,
        pcm_s16le_48k_stereo=pcm,
    )
    # This can enter the bounded queue while the durable event transaction is completing.
    assert pipeline.receive(
        user_id=user_id,
        received_monotonic_ns=cutoff_ns,
        pcm_s16le_48k_stereo=pcm,
    )

    pipeline.withdraw_at(
        user_id=user_id,
        effective_monotonic_ns=cutoff_ns,
    )

    assert recording_queue.frame_count == 1
    accepted = recording_queue.get()
    assert isinstance(accepted, SessionAudioFrame)
    assert accepted.received_monotonic_ns == start_ns + 999_000_000
    assert not pipeline.receive(
        user_id=user_id,
        received_monotonic_ns=cutoff_ns,
        pcm_s16le_48k_stereo=pcm,
    )
