"""Consent-gated, per-user encrypted Meeting Segment recording."""

from __future__ import annotations

import os
import queue
import shutil
import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile
import structlog

from lingxu.domain.enums import ConsentKind
from lingxu.domain.errors import RecordingPipelineError
from lingxu.domain.models import AudioSegmentRecord
from lingxu.storage.encryption import FileCipher
from lingxu.storage.session_files import SessionFiles
from lingxu.utils import new_ulid
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.meeting_consent import MeetingConsentRegistry

_BYTES_PER_SAMPLE_FRAME = 4
_SAMPLE_RATE = 48_000


def cleanup_stale_recording_plaintext(runtime_tmp: Path) -> int:
    """Remove only the formal recorder's stale tmpfs subtree."""

    runtime_root = runtime_tmp.expanduser().resolve()
    recording_root = (runtime_root / "recording").resolve()
    if not recording_root.is_relative_to(runtime_root):
        raise ValueError("recording cleanup path escapes RUNTIME_TMP")
    if not recording_root.exists():
        return 0
    files_removed = sum(1 for path in recording_root.rglob("*") if path.is_file())
    shutil.rmtree(recording_root)
    return files_removed


@dataclass(frozen=True, slots=True)
class _CloseUser:
    user_id: int
    completed: threading.Event
    effective_monotonic_ns: int | None = None


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


_RecordingQueueItem = SessionAudioFrame | _CloseUser | _Stop


class MeetingRecordingQueue:
    """Bound queued audio while preserving ordered close/stop controls."""

    def __init__(self, max_frames: int) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._max_frames = max_frames
        self._items: deque[_RecordingQueueItem] = deque()
        self._frame_count = 0
        self._closed = False
        self._condition = threading.Condition()

    @property
    def max_frames(self) -> int:
        return self._max_frames

    @property
    def frame_count(self) -> int:
        with self._condition:
            return self._frame_count

    def put(self, frame: SessionAudioFrame) -> bool:
        """Insert without blocking; drop only the oldest audio frame when full."""

        with self._condition:
            if self._closed:
                return True
            dropped = False
            if self._frame_count >= self._max_frames:
                for index, item in enumerate(self._items):
                    if isinstance(item, SessionAudioFrame):
                        del self._items[index]
                        self._frame_count -= 1
                        dropped = True
                        break
            self._items.append(frame)
            self._frame_count += 1
            self._condition.notify()
            return dropped

    def close_user(
        self,
        user_id: int,
        *,
        effective_monotonic_ns: int | None = None,
    ) -> threading.Event:
        """Queue an ordered per-user finalisation control."""

        if effective_monotonic_ns is not None and effective_monotonic_ns < 0:
            raise ValueError("effective_monotonic_ns must be non-negative")
        completed = threading.Event()
        with self._condition:
            if self._closed:
                completed.set()
                return completed
            if effective_monotonic_ns is not None:
                retained: deque[_RecordingQueueItem] = deque()
                for item in self._items:
                    if (
                        isinstance(item, SessionAudioFrame)
                        and item.user_id == user_id
                        and item.received_monotonic_ns >= effective_monotonic_ns
                    ):
                        self._frame_count -= 1
                        continue
                    retained.append(item)
                self._items = retained
            self._items.append(
                _CloseUser(
                    user_id=user_id,
                    completed=completed,
                    effective_monotonic_ns=effective_monotonic_ns,
                )
            )
            self._condition.notify()
        return completed

    def close(self) -> None:
        """Preserve accepted frames, then stop the consumer."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._items.append(_Stop())
            self._condition.notify()

    def get(self) -> _RecordingQueueItem:
        """Wait for the next audio or control item."""

        with self._condition:
            while not self._items:
                self._condition.wait()
            item = self._items.popleft()
            if isinstance(item, SessionAudioFrame):
                self._frame_count -= 1
            return item


class RecordingMetrics:
    """Thread-safe counters that retain no audio or Meeting content."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._counters: Counter[str] = Counter()
        self._per_user: dict[int, Counter[str]] = defaultdict(Counter)
        self._error_types: Counter[str] = Counter()
        self._lock = threading.RLock()

    def increment(self, name: str, amount: int = 1, *, user_id: int | None = None) -> None:
        with self._lock:
            self._counters[name] += amount
            if user_id is not None:
                self._per_user[user_id][name] += amount

    def record_error(self, error: BaseException) -> None:
        with self._lock:
            self._error_types[type(error).__name__] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self._session_id,
                "counters": dict(sorted(self._counters.items())),
                "per_user": {
                    str(user_id): dict(sorted(values.items()))
                    for user_id, values in sorted(self._per_user.items())
                },
                "error_types": dict(sorted(self._error_types.items())),
            }


@dataclass(slots=True)
class _OpenSegment:
    user_id: int
    segment_number: int
    writer: Any
    temporary_path: Path
    started_ns: int
    ended_ns: int
    last_frame_duration_ms: int
    pcm_bytes: int = 0
    frames: int = 0
    trailing_silence_ms: int = 0


class EncryptedSegmentRecorder:
    """Write plaintext only under runtime tmpfs, then persist encrypted FLAC."""

    def __init__(
        self,
        *,
        session_id: str,
        session_started_monotonic_ns: int,
        recording_queue: MeetingRecordingQueue,
        session_files: SessionFiles,
        runtime_tmp: Path,
        cipher: FileCipher,
        metrics: RecordingMetrics,
        silence_split_ms: int = 800,
        silence_threshold: int = 256,
        max_segment_seconds: float = 300.0,
    ) -> None:
        if silence_split_ms < 20:
            raise ValueError("silence_split_ms must be at least one 20 ms frame")
        if not 0 <= silence_threshold <= 32_767:
            raise ValueError("silence_threshold must be an int16 magnitude")
        if max_segment_seconds <= 0:
            raise ValueError("max_segment_seconds must be positive")
        self._session_id = session_id
        self._session_started_ns = session_started_monotonic_ns
        self._queue = recording_queue
        self._files = session_files
        self._runtime_root = runtime_tmp.expanduser().resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._temporary_root = (self._runtime_root / "recording" / session_id).resolve()
        if not self._temporary_root.is_relative_to(self._runtime_root):
            raise ValueError("recording tmp path escapes RUNTIME_TMP")
        self._cipher = cipher
        self._metrics = metrics
        self._silence_split_ms = silence_split_ms
        self._silence_threshold = silence_threshold
        self._max_pcm_bytes = int(max_segment_seconds * _SAMPLE_RATE * _BYTES_PER_SAMPLE_FRAME)
        self._open: dict[int, _OpenSegment] = {}
        self._next_numbers: dict[int, int] = {}
        self._completed: queue.SimpleQueue[AudioSegmentRecord] = queue.SimpleQueue()
        self._failure: BaseException | None = None
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"lingxu-segment-recorder-{session_id}",
            daemon=False,
        )
        self._log = structlog.get_logger(__name__).bind(session_id=session_id)

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def start(self) -> None:
        """Remove stale plaintext for this Session and start the recorder."""

        if self._started:
            raise RuntimeError("segment recorder already started")
        self._files.read_manifest(self._session_id)
        if self._temporary_root.exists():
            shutil.rmtree(self._temporary_root)
            self._metrics.increment("stale_plaintext_roots_removed")
        self._temporary_root.mkdir(parents=True)
        self._started = True
        self._thread.start()

    def request_close_user(
        self,
        user_id: int,
        effective_monotonic_ns: int | None = None,
    ) -> threading.Event:
        """Finalise a user's current Segment after all earlier accepted frames."""

        if not self._started:
            completed = threading.Event()
            completed.set()
            return completed
        return self._queue.close_user(
            user_id,
            effective_monotonic_ns=effective_monotonic_ns,
        )

    def stop(self, timeout: float = 30.0) -> tuple[AudioSegmentRecord, ...]:
        """Drain, finalise, encrypt, and remove every plaintext temporary."""

        if not self._started:
            return ()
        self._queue.close()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RecordingPipelineError("segment recorder did not stop before timeout")
        if self._failure is not None:
            raise RecordingPipelineError("segment recorder failed") from self._failure
        return self.drain_completed()

    def drain_completed(self) -> tuple[AudioSegmentRecord, ...]:
        """Return newly completed encrypted Segment metadata."""

        values: list[AudioSegmentRecord] = []
        while True:
            try:
                values.append(self._completed.get_nowait())
            except queue.Empty:
                break
        return tuple(values)

    def timeline_ms(self, monotonic_ns: int) -> int:
        """Convert one monotonic timestamp to this recorder's Meeting-relative time."""

        if monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        return max(0, (monotonic_ns - self._session_started_ns) // 1_000_000)

    def delete_user_plaintext(self, user_id: int) -> None:
        """Remove one withdrawn user's exact temporary subtree after close acknowledgement."""

        user_tmp = (self._temporary_root / str(user_id)).resolve()
        if user_tmp.parent != self._temporary_root:
            raise ValueError("user tmp path escapes the Session recording root")
        if user_tmp.exists():
            shutil.rmtree(user_tmp)

    def _next_segment_number(self, user_id: int) -> int:
        selected = self._next_numbers.get(user_id)
        if selected is not None:
            self._next_numbers[user_id] = selected + 1
            return selected
        user_dir = self._files.directory(self._session_id) / "audio" / str(user_id)
        existing = [
            int(path.name.split(".", maxsplit=1)[0])
            for path in user_dir.glob("*.flac.enc")
            if path.name.split(".", maxsplit=1)[0].isdigit()
        ]
        selected = max(existing, default=0) + 1
        self._next_numbers[user_id] = selected + 1
        return selected

    def _open_segment(self, frame: SessionAudioFrame) -> _OpenSegment:
        number = self._next_segment_number(frame.user_id)
        user_tmp = self._temporary_root / str(frame.user_id)
        user_tmp.mkdir(parents=True, exist_ok=True)
        temporary = user_tmp / f"{number:06}.flac.part"
        writer = soundfile.SoundFile(
            str(temporary),
            mode="w",
            samplerate=_SAMPLE_RATE,
            channels=2,
            subtype="PCM_16",
            format="FLAC",
        )
        state = _OpenSegment(
            user_id=frame.user_id,
            segment_number=number,
            writer=writer,
            temporary_path=temporary,
            started_ns=frame.received_monotonic_ns,
            ended_ns=frame.received_monotonic_ns,
            last_frame_duration_ms=0,
        )
        self._open[frame.user_id] = state
        return state

    @staticmethod
    def _frame_duration_ms(payload: bytes) -> int:
        sample_frames = len(payload) // _BYTES_PER_SAMPLE_FRAME
        return max(1, sample_frames * 1_000 // _SAMPLE_RATE)

    def _is_silent(self, payload: bytes) -> bool:
        samples = memoryview(payload).cast("h")
        return not samples or max(abs(sample) for sample in samples) <= self._silence_threshold

    def _consume(self, frame: SessionAudioFrame) -> None:
        if frame.session_id != self._session_id:
            self._metrics.increment("foreign_session_frames", user_id=frame.user_id)
            return
        payload = frame.pcm_s16le_48k_stereo
        if not payload:
            self._metrics.increment("empty_pcm_frames", user_id=frame.user_id)
            return
        if len(payload) % _BYTES_PER_SAMPLE_FRAME:
            self._metrics.increment("invalid_pcm_frames", user_id=frame.user_id)
            return

        state = self._open.get(frame.user_id)
        if state is not None:
            gap_ms = max(0, (frame.received_monotonic_ns - state.ended_ns) // 1_000_000)
            if gap_ms >= self._silence_split_ms or (
                state.pcm_bytes + len(payload) > self._max_pcm_bytes
            ):
                self._finalise_user(frame.user_id)
                state = None

        silent = self._is_silent(payload)
        if state is None and silent:
            self._metrics.increment("leading_silence_frames_discarded", user_id=frame.user_id)
            return
        if state is None:
            state = self._open_segment(frame)

        frame_duration_ms = self._frame_duration_ms(payload)
        state.writer.buffer_write(payload, dtype="int16")
        state.pcm_bytes += len(payload)
        state.frames += 1
        state.ended_ns = frame.received_monotonic_ns
        state.last_frame_duration_ms = frame_duration_ms
        state.trailing_silence_ms = state.trailing_silence_ms + frame_duration_ms if silent else 0
        self._metrics.increment("frames_written", user_id=frame.user_id)
        self._metrics.increment("pcm_bytes_written", len(payload), user_id=frame.user_id)
        if state.trailing_silence_ms >= self._silence_split_ms:
            self._finalise_user(frame.user_id)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _finalise_user(self, user_id: int) -> None:
        state = self._open.pop(user_id, None)
        if state is None:
            return
        relative_path = Path("audio") / str(user_id) / (f"{state.segment_number:06}.flac.enc")
        encrypted_path = self._files.directory(self._session_id) / relative_path
        try:
            state.writer.flush()
            state.writer.close()
            self._fsync_file(state.temporary_path)
            sha256 = self._cipher.encrypt_file(state.temporary_path, encrypted_path)
            started_ms = max(0, (state.started_ns - self._session_started_ns) // 1_000_000)
            ended_ms = max(
                started_ms,
                (state.ended_ns - self._session_started_ns) // 1_000_000
                + state.last_frame_duration_ms,
            )
            segment = AudioSegmentRecord(
                id=new_ulid(),
                session_id=self._session_id,
                user_id=user_id,
                segment_number=state.segment_number,
                encrypted_path=relative_path,
                sha256=sha256,
                started_ms=started_ms,
                ended_ms=ended_ms,
                bytes_encrypted=encrypted_path.stat().st_size,
            )
            self._files.append_jsonl(
                self._session_id,
                Path("audio") / str(user_id) / "segments.jsonl",
                {
                    "id": segment.id,
                    "segment_number": segment.segment_number,
                    "path": segment.encrypted_path.as_posix(),
                    "sha256": segment.sha256,
                    "started_ms": segment.started_ms,
                    "ended_ms": segment.ended_ms,
                    "bytes_encrypted": segment.bytes_encrypted,
                },
            )
            self._completed.put(segment)
            self._metrics.increment("segments_completed", user_id=user_id)
        finally:
            try:
                state.writer.close()
            except Exception:
                pass
            state.temporary_path.unlink(missing_ok=True)

    def _abort_open_segments(self) -> None:
        for state in self._open.values():
            try:
                state.writer.close()
            except Exception:
                pass
            state.temporary_path.unlink(missing_ok=True)
        self._open.clear()

    def _abort_user(self, user_id: int) -> None:
        state = self._open.pop(user_id, None)
        if state is None:
            return
        try:
            state.writer.close()
        finally:
            state.temporary_path.unlink(missing_ok=True)
        self._metrics.increment("segments_discarded_at_withdrawal_cutoff", user_id=user_id)

    def _discard_completed_at_or_after_cutoff(
        self,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> None:
        """Delete completed-but-unpersisted Segments that overlap a withdrawal cutoff."""

        cutoff_ms = max(
            0,
            (effective_monotonic_ns - self._session_started_ns) // 1_000_000,
        )
        retained: list[AudioSegmentRecord] = []
        while True:
            try:
                segment = self._completed.get_nowait()
            except queue.Empty:
                break
            if segment.user_id != user_id or segment.ended_ms <= cutoff_ms:
                retained.append(segment)
                continue
            self._files.resolve(self._session_id, segment.encrypted_path).unlink(missing_ok=True)
            self._metrics.increment(
                "completed_segments_discarded_at_withdrawal_cutoff",
                user_id=user_id,
            )
        for segment in retained:
            self._completed.put(segment)

    def _cleanup_temporary_root(self) -> None:
        if self._temporary_root.exists():
            shutil.rmtree(self._temporary_root)
        recording_root = self._temporary_root.parent
        if recording_root.exists() and not any(recording_root.iterdir()):
            recording_root.rmdir()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if isinstance(item, SessionAudioFrame):
                    self._consume(item)
                    continue
                if isinstance(item, _CloseUser):
                    try:
                        state = self._open.get(item.user_id)
                        crosses_cutoff = (
                            state is not None
                            and item.effective_monotonic_ns is not None
                            and state.ended_ns + state.last_frame_duration_ms * 1_000_000
                            > item.effective_monotonic_ns
                        )
                        if crosses_cutoff:
                            self._abort_user(item.user_id)
                        else:
                            self._finalise_user(item.user_id)
                        if item.effective_monotonic_ns is not None:
                            self._discard_completed_at_or_after_cutoff(
                                item.user_id,
                                item.effective_monotonic_ns,
                            )
                    except BaseException as exc:
                        # Publish failure before acknowledging the async cleanup boundary.
                        self._failure = exc
                        raise
                    finally:
                        item.completed.set()
                    continue
                for user_id in tuple(self._open):
                    self._finalise_user(user_id)
                break
        except BaseException as exc:
            self._failure = exc
            self._metrics.record_error(exc)
            self._abort_open_segments()
            self._log.error("segment_recorder_failed", error_type=type(exc).__name__)
        finally:
            try:
                self._cleanup_temporary_root()
            except BaseException as exc:
                if self._failure is None:
                    self._failure = exc
                self._metrics.record_error(exc)
                self._log.error(
                    "recording_plaintext_cleanup_failed",
                    error_type=type(exc).__name__,
                )


class FormalRecordingPipeline:
    """Receive-path facade that enforces recording consent before queuing."""

    def __init__(
        self,
        *,
        session_id: str,
        consents: MeetingConsentRegistry,
        recording_queue: MeetingRecordingQueue,
        recorder: EncryptedSegmentRecorder,
        metrics: RecordingMetrics,
    ) -> None:
        self._session_id = session_id
        self.consents = consents
        self.queue = recording_queue
        self.recorder = recorder
        self.metrics = metrics
        self._control_lock = threading.RLock()
        self._stopping = False

    def start(self) -> None:
        self.recorder.start()

    def grant(self, user_id: int, kind: ConsentKind) -> bool:
        with self._control_lock:
            if self._stopping:
                return False
            return self.consents.grant(user_id, kind)

    def withdraw(self, user_id: int, kind: ConsentKind) -> threading.Event | None:
        with self._control_lock:
            self.consents.withdraw(user_id, kind)
            if kind is ConsentKind.RECORDING:
                return self.recorder.request_close_user(user_id)
            return None

    def withdraw_all(self, user_id: int) -> threading.Event | None:
        with self._control_lock:
            changed = self.consents.withdraw_all(user_id)
            if ConsentKind.RECORDING in changed:
                return self.recorder.request_close_user(user_id)
            return None

    def withdraw_at(
        self,
        *,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> threading.Event | None:
        """Apply one ordered half-open cutoff and close the user's active Segment."""

        with self._control_lock:
            self.consents.withdraw_all_at_monotonic_ns(
                user_id,
                effective_monotonic_ns,
            )
            return self.recorder.request_close_user(user_id, effective_monotonic_ns)

    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool:
        """Copy and enqueue one frame only while recording consent is held."""

        frame = SessionAudioFrame(
            session_id=self._session_id,
            user_id=user_id,
            received_monotonic_ns=received_monotonic_ns,
            pcm_s16le_48k_stereo=bytes(pcm_s16le_48k_stereo),
        )
        with self._control_lock:
            if self._stopping:
                self.metrics.increment("frames_rejected_recording_stopping", user_id=user_id)
                return False
            accepted, dropped = self.consents.perform_if_consented(
                user_id,
                ConsentKind.RECORDING,
                received_monotonic_ns,
                lambda: self.queue.put(frame),
            )
        if not accepted:
            self.metrics.increment("frames_rejected_without_recording_consent", user_id=user_id)
            return False
        self.metrics.increment("frames_accepted", user_id=user_id)
        if dropped:
            self.metrics.increment("frames_dropped_queue_full", user_id=user_id)
        return True

    def stop(self, timeout: float = 30.0) -> tuple[AudioSegmentRecord, ...]:
        with self._control_lock:
            self._stopping = True
            self.consents.clear()
        return self.recorder.stop(timeout)
