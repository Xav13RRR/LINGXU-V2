import struct

from lingxu.asr.utterance import UtteranceBuilder
from lingxu.voice.frames import SessionAudioFrame

_FRAME_NS = 20_000_000
_VOICE_ONE = struct.pack("<hh", 2_000, -2_000) * 960
_VOICE_TWO = struct.pack("<hh", 3_000, -3_000) * 960
_SILENCE = b"\0" * len(_VOICE_ONE)


def _frame(user_id: int, index: int, payload: bytes) -> SessionAudioFrame:
    return SessionAudioFrame(
        session_id="session",
        user_id=user_id,
        received_monotonic_ns=index * _FRAME_NS,
        pcm_s16le_48k_stereo=payload,
    )


def test_vad_builds_sentence_and_discards_leading_silence() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=40,
        min_utterance_ms=40,
    )

    assert builder.accept(_frame(1, 0, _SILENCE)) == ()
    assert builder.accept(_frame(1, 1, _VOICE_ONE)) == ()
    assert builder.accept(_frame(1, 2, _VOICE_ONE)) == ()
    assert builder.accept(_frame(1, 3, _VOICE_ONE)) == ()
    assert builder.accept(_frame(1, 4, _SILENCE)) == ()
    completed = builder.accept(_frame(1, 5, _SILENCE))

    assert len(completed) == 1
    assert completed[0].user_id == 1
    assert completed[0].started_monotonic_ns == _FRAME_NS
    assert completed[0].duration_ms == 100
    assert completed[0].pcm_s16le_48k_stereo.startswith(_VOICE_ONE)


def test_overlapping_users_never_mix_pcm_or_timeline() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=40,
        min_utterance_ms=40,
    )
    completed = []
    for index in range(5):
        first = _VOICE_ONE if index < 3 else _SILENCE
        second = _VOICE_TWO if index < 3 else _SILENCE
        completed.extend(builder.accept(_frame(10, index, first)))
        completed.extend(builder.accept(_frame(20, index, second)))

    assert [item.user_id for item in completed] == [10, 20]
    assert _VOICE_TWO not in completed[0].pcm_s16le_48k_stereo
    assert _VOICE_ONE not in completed[1].pcm_s16le_48k_stereo


def test_withdrawal_discards_partial_utterance() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=40,
        min_utterance_ms=40,
    )
    builder.accept(_frame(1, 0, _VOICE_ONE))
    builder.accept(_frame(1, 1, _VOICE_ONE))

    builder.discard_user(1)

    assert builder.flush_user(1) is None


def test_withdrawal_cutoff_discards_partial_utterance_touching_boundary() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=800,
        min_utterance_ms=40,
    )
    builder.accept(_frame(1, 48, _VOICE_ONE))
    builder.accept(_frame(1, 49, _VOICE_ONE))

    assert builder.discard_user_at_monotonic_ns(1, 1_000_000_000)
    assert builder.flush_user(1) is None


def test_wall_clock_finalises_when_discord_sends_no_silence_frames() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=800,
        min_utterance_ms=40,
    )
    builder.accept(_frame(1, 0, _VOICE_ONE))
    builder.accept(_frame(1, 1, _VOICE_ONE))

    assert builder.expire_inactive(839_000_000) == ()
    completed = builder.expire_inactive(840_000_000)

    assert len(completed) == 1
    assert completed[0].started_monotonic_ns == 0
    assert completed[0].ended_monotonic_ns == 40_000_000
    assert completed[0].duration_ms == 40


def test_new_frame_before_idle_deadline_postpones_wall_clock_endpoint() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=800,
        min_utterance_ms=40,
    )
    builder.accept(_frame(1, 0, _VOICE_ONE))
    builder.accept(_frame(1, 1, _VOICE_ONE))
    assert builder.expire_inactive(800_000_000) == ()

    builder.accept(_frame(1, 40, _VOICE_ONE))

    assert builder.expire_inactive(1_619_000_000) == ()
    completed = builder.expire_inactive(1_620_000_000)
    assert len(completed) == 1
    assert completed[0].ended_monotonic_ns == 820_000_000


def test_wall_clock_enforces_maximum_without_waiting_for_another_frame() -> None:
    builder = UtteranceBuilder(
        session_id="session",
        vad_threshold=100,
        vad_start_ms=40,
        vad_end_ms=1_000,
        min_utterance_ms=40,
        max_utterance_seconds=0.1,
    )
    builder.accept(_frame(1, 0, _VOICE_ONE))
    builder.accept(_frame(1, 1, _VOICE_ONE))

    assert builder.expire_inactive(99_000_000) == ()
    completed = builder.expire_inactive(100_000_000)

    assert len(completed) == 1
    assert completed[0].ended_monotonic_ns == 40_000_000
