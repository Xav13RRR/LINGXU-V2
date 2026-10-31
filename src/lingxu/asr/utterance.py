"""Per-user energy VAD and deterministic sentence construction."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field

from lingxu.asr.models import UtteranceAudio
from lingxu.voice.frames import SessionAudioFrame

_SAMPLE_RATE = 48_000
_BYTES_PER_SAMPLE_FRAME = 4


def _frame_duration_ms(payload: bytes) -> int:
    return max(1, len(payload) * 1_000 // (_SAMPLE_RATE * _BYTES_PER_SAMPLE_FRAME))


def _rms(payload: bytes) -> float:
    samples = memoryview(payload).cast("h")
    if not samples:
        return 0.0
    return math.sqrt(sum(int(sample) * int(sample) for sample in samples) / len(samples))


@dataclass(slots=True)
class _UserState:
    pending: deque[SessionAudioFrame] = field(default_factory=deque)
    pending_speech_ms: int = 0
    active: list[SessionAudioFrame] = field(default_factory=list)
    active_voiced_ms: int = 0
    trailing_silence_ms: int = 0


class UtteranceBuilder:
    """Build independently timed utterances without touching the receive thread."""

    def __init__(
        self,
        *,
        session_id: str,
        vad_threshold: int = 500,
        vad_start_ms: int = 60,
        vad_end_ms: int = 800,
        min_utterance_ms: int = 300,
        max_utterance_seconds: float = 30.0,
    ) -> None:
        if not 0 <= vad_threshold <= 32_767:
            raise ValueError("vad_threshold must be an int16 magnitude")
        if min(vad_start_ms, vad_end_ms, min_utterance_ms) < 20:
            raise ValueError("VAD durations must be at least 20 ms")
        if max_utterance_seconds <= 0:
            raise ValueError("max_utterance_seconds must be positive")
        self._session_id = session_id
        self._threshold = vad_threshold
        self._start_ms = vad_start_ms
        self._end_ms = vad_end_ms
        self._min_ms = min_utterance_ms
        self._max_ms = int(max_utterance_seconds * 1_000)
        self._states: dict[int, _UserState] = {}
        self._lock = threading.RLock()

    def accept(self, frame: SessionAudioFrame) -> tuple[UtteranceAudio, ...]:
        """Consume one frame and return zero or more completed utterances."""

        if frame.session_id != self._session_id or not frame.pcm_s16le_48k_stereo:
            return ()
        if len(frame.pcm_s16le_48k_stereo) % _BYTES_PER_SAMPLE_FRAME:
            return ()
        with self._lock:
            state = self._states.setdefault(frame.user_id, _UserState())
            voiced = _rms(frame.pcm_s16le_48k_stereo) >= self._threshold
            duration_ms = _frame_duration_ms(frame.pcm_s16le_48k_stereo)
            completed: list[UtteranceAudio] = []

            if not state.active:
                if voiced:
                    state.pending.append(frame)
                    state.pending_speech_ms += duration_ms
                    if state.pending_speech_ms >= self._start_ms:
                        state.active.extend(state.pending)
                        state.active_voiced_ms = state.pending_speech_ms
                        state.pending.clear()
                        state.pending_speech_ms = 0
                else:
                    state.pending.clear()
                    state.pending_speech_ms = 0
                return ()

            state.active.append(frame)
            if voiced:
                state.active_voiced_ms += duration_ms
                state.trailing_silence_ms = 0
            else:
                state.trailing_silence_ms += duration_ms

            elapsed_ms = self._elapsed_ms(state.active)
            if state.trailing_silence_ms >= self._end_ms or elapsed_ms >= self._max_ms:
                utterance = self._finalise(frame.user_id, state)
                if utterance is not None:
                    completed.append(utterance)
            return tuple(completed)

    def flush_user(self, user_id: int) -> UtteranceAudio | None:
        """Finalise one accepted user's partial utterance during graceful stop."""

        with self._lock:
            state = self._states.pop(user_id, None)
            if state is None:
                return None
            return self._build(user_id, state)

    def flush_all(self) -> tuple[UtteranceAudio, ...]:
        with self._lock:
            completed = [
                utterance
                for user_id, state in tuple(self._states.items())
                if (utterance := self._build(user_id, state)) is not None
            ]
            self._states.clear()
            return tuple(completed)

    def expire_inactive(self, now_monotonic_ns: int) -> tuple[UtteranceAudio, ...]:
        """Finalise speech when Discord stops delivering packets during silence.

        Discord commonly sends no PCM at all after a member stops speaking. Energy VAD therefore
        cannot observe its configured trailing-silence frames. This wall-clock path preserves the
        last real frame as the utterance endpoint and never fabricates audio.
        """

        if now_monotonic_ns < 0:
            raise ValueError("now_monotonic_ns must be non-negative")
        with self._lock:
            completed: list[UtteranceAudio] = []
            for user_id, state in tuple(self._states.items()):
                frames = state.active or list(state.pending)
                if not frames:
                    self._states.pop(user_id, None)
                    continue
                last = frames[-1]
                last_end_ns = (
                    last.received_monotonic_ns
                    + _frame_duration_ms(last.pcm_s16le_48k_stereo) * 1_000_000
                )
                idle_due_ns = last_end_ns + self._end_ms * 1_000_000
                maximum_due_ns = (
                    state.active[0].received_monotonic_ns + self._max_ms * 1_000_000
                    if state.active
                    else idle_due_ns
                )
                if now_monotonic_ns < min(idle_due_ns, maximum_due_ns):
                    continue
                self._states.pop(user_id, None)
                utterance = self._build(user_id, state)
                if utterance is not None:
                    completed.append(utterance)
            return tuple(completed)

    def discard_user(self, user_id: int) -> None:
        """Forget all unrecognised PCM immediately after consent withdrawal."""

        with self._lock:
            self._states.pop(user_id, None)

    def discard_user_at_monotonic_ns(
        self,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> bool:
        """Discard a partial utterance and report whether it touches the cutoff."""

        if effective_monotonic_ns < 0:
            raise ValueError("effective_monotonic_ns must be non-negative")
        with self._lock:
            state = self._states.pop(user_id, None)
            if state is None:
                return False
            frames = state.active or list(state.pending)
            if not frames:
                return False
            last = frames[-1]
            ended_ns = (
                last.received_monotonic_ns
                + _frame_duration_ms(last.pcm_s16le_48k_stereo) * 1_000_000
            )
            return ended_ns >= effective_monotonic_ns

    def discard_all(self) -> None:
        """Forget all partial utterances without producing transcript content."""

        with self._lock:
            self._states.clear()

    @staticmethod
    def _elapsed_ms(frames: list[SessionAudioFrame]) -> int:
        first = frames[0]
        last = frames[-1]
        return max(
            0,
            (last.received_monotonic_ns - first.received_monotonic_ns) // 1_000_000
            + _frame_duration_ms(last.pcm_s16le_48k_stereo),
        )

    def _finalise(self, user_id: int, state: _UserState) -> UtteranceAudio | None:
        self._states.pop(user_id, None)
        return self._build(user_id, state)

    def _build(self, user_id: int, state: _UserState) -> UtteranceAudio | None:
        if not state.active or state.active_voiced_ms < self._min_ms:
            return None
        first = state.active[0]
        last = state.active[-1]
        generation = first.consent_generation
        if any(frame.consent_generation != generation for frame in state.active):
            return None
        return UtteranceAudio(
            session_id=self._session_id,
            user_id=user_id,
            started_monotonic_ns=first.received_monotonic_ns,
            ended_monotonic_ns=(
                last.received_monotonic_ns
                + _frame_duration_ms(last.pcm_s16le_48k_stereo) * 1_000_000
            ),
            pcm_s16le_48k_stereo=b"".join(frame.pcm_s16le_48k_stereo for frame in state.active),
            consent_generation=generation,
        )
