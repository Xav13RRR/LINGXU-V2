import struct

import pytest

from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.voice.frames import SessionAudioFrame


def _frame(user_id: int, sequence: int) -> SessionAudioFrame:
    return SessionAudioFrame(
        session_id="session",
        user_id=user_id,
        received_monotonic_ns=sequence,
        pcm_s16le_48k_stereo=struct.pack("<hh", sequence, -sequence),
    )


def test_queue_is_bounded_per_user_and_round_robin_fair() -> None:
    queue = PerUserRealtimeQueue(max_frames_per_user=2)

    assert queue.put(_frame(1, 1)) is False
    assert queue.put(_frame(1, 2)) is False
    assert queue.put(_frame(1, 3)) is True
    assert queue.put(_frame(2, 4)) is False

    assert queue.size_for_user(1) == 2
    assert queue.get() == _frame(1, 2)
    assert queue.get() == _frame(2, 4)
    assert queue.get() == _frame(1, 3)

    queue.close()
    assert queue.get() is None


def test_clear_user_preserves_other_users() -> None:
    queue = PerUserRealtimeQueue(max_frames_per_user=3)
    queue.put(_frame(1, 1))
    queue.put(_frame(2, 2))
    queue.put(_frame(1, 3))

    assert queue.clear_user(1) == 2
    assert queue.size_for_user(1) == 0
    assert queue.get() == _frame(2, 2)

    queue.close()
    assert queue.get() is None


def test_bounded_batch_drain_preserves_round_robin_and_close_semantics() -> None:
    queue = PerUserRealtimeQueue(max_frames_per_user=4)
    for frame in (
        _frame(1, 1),
        _frame(1, 2),
        _frame(1, 3),
        _frame(2, 4),
        _frame(2, 5),
    ):
        assert queue.put(frame) is False

    assert queue.get_many(4) == (
        _frame(1, 1),
        _frame(2, 4),
        _frame(1, 2),
        _frame(2, 5),
    )
    queue.close()
    assert queue.get_many(4) == (_frame(1, 3),)
    assert queue.get_many(4) is None


def test_batch_drain_rejects_nonpositive_capacity() -> None:
    queue = PerUserRealtimeQueue(max_frames_per_user=1)

    with pytest.raises(ValueError):
        queue.get_many(0)
