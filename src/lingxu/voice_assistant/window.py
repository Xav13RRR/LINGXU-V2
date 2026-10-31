"""Bounded per-user rolling PCM windows for streaming keyword spotting."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

from lingxu.asr.models import UtteranceAudio
from lingxu.voice.frames import SessionAudioFrame

_SAMPLE_RATE = 48_000
_BYTES_PER_SAMPLE_FRAME = 4


def _frame_duration_ns(payload: bytes) -> int:
    return max(1, len(payload) * 1_000_000_000 // (_SAMPLE_RATE * _BYTES_PER_SAMPLE_FRAME))


def _rms(payload: bytes) -> float:
    samples = memoryview(payload).cast("h")
    if not samples:
        return 0.0
    return math.sqrt(sum(int(sample) * int(sample) for sample in samples) / len(samples))


@dataclass(slots=True)
class _WindowState:
    frames: deque[SessionAudioFrame] = field(default_factory=deque)
    active: bool = False
    last_emit_ns: int | None = None


class WakeWindowBuilder:
    """Emit overlapping speech windows without waiting for a full VAD utterance."""

    def __init__(
        self,
        *,
        session_id: str,
        activity_threshold: int = 500,
        window_ms: int = 4_000,
        hop_ms: int = 500,
        min_window_ms: int = 1_200,
        preroll_ms: int = 300,
    ) -> None:
        if not 0 <= activity_threshold <= 32_767:
            raise ValueError("activity_threshold must be an int16 magnitude")
        if min(window_ms, hop_ms, min_window_ms, preroll_ms) < 20:
            raise ValueError("wake window durations must be at least 20 ms")
        if max(min_window_ms, hop_ms, preroll_ms) > window_ms:
            raise ValueError("wake window minimum, hop, and preroll must not exceed the window")
        self._session_id = session_id
        self._activity_threshold = activity_threshold
        self._window_ns = window_ms * 1_000_000
        self._hop_ns = hop_ms * 1_000_000
        self._min_window_ns = min_window_ms * 1_000_000
        self._preroll_ns = preroll_ms * 1_000_000
        self._states: dict[int, _WindowState] = {}
        self._lock = threading.RLock()

    def accept(self, frame: SessionAudioFrame) -> UtteranceAudio | None:
        """Return the next rolling window, or ``None`` before the next hop."""

        if frame.session_id != self._session_id or not frame.pcm_s16le_48k_stereo:
            return None
        if len(frame.pcm_s16le_48k_stereo) % _BYTES_PER_SAMPLE_FRAME:
            return None
        with self._lock:
            state = self._states.setdefault(frame.user_id, _WindowState())
            state.frames.append(frame)
            current_end_ns = frame.received_monotonic_ns + _frame_duration_ns(
                frame.pcm_s16le_48k_stereo
            )
            if not state.active:
                preroll_cutoff_ns = current_end_ns - self._preroll_ns
                self._trim(state, cutoff_ns=preroll_cutoff_ns)
                if _rms(frame.pcm_s16le_48k_stereo) < self._activity_threshold:
                    return None
                state.active = True
            cutoff_ns = current_end_ns - self._window_ns
            self._trim(state, cutoff_ns=cutoff_ns)
            if not state.frames:
                return None
            first_ns = state.frames[0].received_monotonic_ns
            if current_end_ns - first_ns < self._min_window_ns:
                return None
            if (
                state.last_emit_ns is not None
                and current_end_ns - state.last_emit_ns < self._hop_ns
            ):
                return None
            state.last_emit_ns = current_end_ns
            return UtteranceAudio(
                session_id=self._session_id,
                user_id=frame.user_id,
                started_monotonic_ns=first_ns,
                ended_monotonic_ns=current_end_ns,
                pcm_s16le_48k_stereo=b"".join(
                    selected.pcm_s16le_48k_stereo for selected in state.frames
                ),
            )

    @staticmethod
    def _trim(state: _WindowState, *, cutoff_ns: int) -> None:
        while state.frames:
            first = state.frames[0]
            first_end_ns = first.received_monotonic_ns + _frame_duration_ns(
                first.pcm_s16le_48k_stereo
            )
            if first_end_ns > cutoff_ns:
                return
            state.frames.popleft()

    def discard_user(self, user_id: int) -> None:
        with self._lock:
            self._states.pop(user_id, None)

    def discard_all(self) -> None:
        with self._lock:
            self._states.clear()
