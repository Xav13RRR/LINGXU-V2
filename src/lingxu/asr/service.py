"""Consent-gated realtime ASR orchestration with bounded failure domains."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from lingxu.asr.models import UtteranceAudio
from lingxu.asr.provider import PreparableRealtimeASRProvider, RealtimeASRProvider
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.domain.enums import ConsentKind
from lingxu.domain.models import RealtimeUtteranceRecord
from lingxu.utils import new_ulid
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.meeting_consent import MeetingConsentRegistry


class RealtimeUtteranceStore(Protocol):
    async def add(self, utterance: RealtimeUtteranceRecord) -> RealtimeUtteranceRecord:
        """Persist one immutable provider result."""

    async def delete_at_or_after(
        self,
        *,
        session_id: str,
        user_id: int,
        cutoff_ms: int,
    ) -> int:
        """Delete exact-user results touching or following a closed transcript cutoff."""


@dataclass(frozen=True, slots=True)
class _PendingUtterance:
    audio: UtteranceAudio
    consent_generation: int


class RealtimeASRMetrics:
    """Content-free counters and stop-to-final latency distribution."""

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._counters: Counter[str] = Counter()
        self._per_user: dict[int, Counter[str]] = defaultdict(Counter)
        self._error_types: Counter[str] = Counter()
        self._latencies_ms: list[int] = []
        self._lock = threading.RLock()

    def increment(self, name: str, amount: int = 1, *, user_id: int | None = None) -> None:
        with self._lock:
            self._counters[name] += amount
            if user_id is not None:
                self._per_user[user_id][name] += amount

    def record_error(self, error: BaseException) -> None:
        with self._lock:
            self._error_types[type(error).__name__] += 1

    def record_latency(self, latency_ms: int) -> None:
        with self._lock:
            self._latencies_ms.append(max(0, latency_ms))

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies_ms)
            return {
                "session_id": self._session_id,
                "counters": dict(sorted(self._counters.items())),
                "per_user": {
                    str(user_id): dict(sorted(values.items()))
                    for user_id, values in sorted(self._per_user.items())
                },
                "error_types": dict(sorted(self._error_types.items())),
                "latency_ms": {
                    "count": len(latencies),
                    "p50": self._percentile(latencies, 0.50),
                    "p95": self._percentile(latencies, 0.95),
                    "max": max(latencies) if latencies else None,
                },
            }


class RealtimeASRService:
    """Keep receive, VAD, inference, and persistence on separate bounded paths."""

    def __init__(
        self,
        *,
        session_id: str,
        session_started_monotonic_ns: int,
        consents: MeetingConsentRegistry,
        frame_queue: PerUserRealtimeQueue,
        utterance_builder: UtteranceBuilder,
        provider: RealtimeASRProvider,
        utterances: RealtimeUtteranceStore,
        metrics: RealtimeASRMetrics,
        language: str = "auto",
        hotwords: tuple[str, ...] = (),
        timeout_seconds: float = 8.0,
        max_concurrency: int = 1,
        max_pending_utterances: int = 32,
        endpoint_poll_seconds: float = 0.05,
        clock_ns: Any = time.monotonic_ns,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_concurrency < 1 or max_pending_utterances < 1:
            raise ValueError("ASR concurrency and capacity must be positive")
        if endpoint_poll_seconds <= 0:
            raise ValueError("endpoint_poll_seconds must be positive")
        self._session_id = session_id
        self._session_started_ns = session_started_monotonic_ns
        self._consents = consents
        self._frames = frame_queue
        self._builder = utterance_builder
        self._provider = provider
        self._utterances = utterances
        self.metrics = metrics
        self._language = language
        self._hotwords = hotwords
        self._timeout = timeout_seconds
        self._worker_count = max_concurrency
        self._max_pending = max_pending_utterances
        self._endpoint_poll_seconds = endpoint_poll_seconds
        self._clock_ns = clock_ns
        self._control_lock = threading.RLock()
        self._generations: defaultdict[int, int] = defaultdict(int)
        self._persistence_locks: dict[int, asyncio.Lock] = {}
        self._started = False
        self._stopping = False
        self._consumer: asyncio.Task[None] | None = None
        self._endpoint_watchdog: asyncio.Task[None] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._pending: asyncio.Queue[_PendingUtterance | None] | None = None
        self._log = structlog.get_logger(__name__).bind(session_id=session_id)

    async def prepare(self) -> None:
        """Load and verify the selected provider before joining Discord voice."""

        if isinstance(self._provider, PreparableRealtimeASRProvider):
            await self._provider.prepare()

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("realtime ASR service already started")
        self._started = True
        self._pending = asyncio.Queue(maxsize=self._max_pending)
        self._consumer = asyncio.create_task(
            self._consume_frames(),
            name=f"lingxu-asr-vad-{self._session_id}",
        )
        self._endpoint_watchdog = asyncio.create_task(
            self._expire_inactive_utterances(),
            name=f"lingxu-asr-endpoint-{self._session_id}",
        )
        self._workers = [
            asyncio.create_task(
                self._recognition_worker(index),
                name=f"lingxu-asr-{self._session_id}-{index}",
            )
            for index in range(self._worker_count)
        ]

    def grant(self, user_id: int) -> bool:
        with self._control_lock:
            if self._stopping:
                return False
            changed = self._consents.grant(user_id, ConsentKind.REALTIME_ASR)
            if changed:
                self._generations[user_id] += 1
            return changed

    def withdraw(self, user_id: int) -> bool:
        """Immediately block new PCM and invalidate queued/in-flight output."""

        with self._control_lock:
            changed = self._consents.withdraw(user_id, ConsentKind.REALTIME_ASR)
            self._generations[user_id] += 1
            removed = self._frames.clear_user(user_id)
            self._builder.discard_user(user_id)
        self.metrics.increment("frames_discarded_on_withdrawal", removed, user_id=user_id)
        self.metrics.increment("consent_withdrawals", user_id=user_id)
        return changed

    async def withdraw_at(
        self,
        *,
        user_id: int,
        effective_monotonic_ns: int,
    ) -> bool:
        """Invalidate one user's live ASR at a durable half-open cutoff."""

        persistence_lock = self._persistence_locks.setdefault(user_id, asyncio.Lock())
        async with persistence_lock:
            with self._control_lock:
                changed = self._consents.withdraw_all_at_monotonic_ns(
                    user_id,
                    effective_monotonic_ns,
                )
                self._generations[user_id] += 1
                removed = self._frames.clear_user(user_id)
                crossed_cutoff = self._builder.discard_user_at_monotonic_ns(
                    user_id,
                    effective_monotonic_ns,
                )
            cutoff_ms = max(
                0,
                (effective_monotonic_ns - self._session_started_ns) // 1_000_000,
            )
            removed_persisted = await self._utterances.delete_at_or_after(
                session_id=self._session_id,
                user_id=user_id,
                cutoff_ms=cutoff_ms,
            )
        self.metrics.increment("frames_discarded_on_withdrawal", removed, user_id=user_id)
        if crossed_cutoff:
            self.metrics.increment(
                "partial_utterances_discarded_at_cutoff",
                user_id=user_id,
            )
        self.metrics.increment("consent_withdrawals", user_id=user_id)
        self.metrics.increment(
            "persisted_utterances_discarded_at_cutoff",
            removed_persisted,
            user_id=user_id,
        )
        return ConsentKind.REALTIME_ASR in changed

    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool:
        """Copy and enqueue PCM in constant bounded work; never run VAD or ASR here."""

        with self._control_lock:
            if not self._started or self._stopping:
                self.metrics.increment("frames_rejected_stopping", user_id=user_id)
                return False
            frame = SessionAudioFrame(
                session_id=self._session_id,
                user_id=user_id,
                received_monotonic_ns=received_monotonic_ns,
                pcm_s16le_48k_stereo=bytes(pcm_s16le_48k_stereo),
                consent_generation=self._generations[user_id],
            )
            accepted, dropped = self._consents.perform_if_consented(
                user_id,
                ConsentKind.REALTIME_ASR,
                received_monotonic_ns,
                lambda: self._frames.put(frame),
            )
        if not accepted:
            self.metrics.increment("frames_rejected_without_consent", user_id=user_id)
            return False
        self.metrics.increment("frames_accepted", user_id=user_id)
        self.metrics.increment(
            "pcm_bytes_accepted", len(frame.pcm_s16le_48k_stereo), user_id=user_id
        )
        if dropped:
            self.metrics.increment("frames_dropped_queue_full", user_id=user_id)
        return True

    def _valid_generation(self, user_id: int, generation: int) -> bool:
        with self._control_lock:
            return generation == self._generations[user_id] and self._consents.snapshot().allows(
                user_id, ConsentKind.REALTIME_ASR
            )

    def _enqueue_utterance(self, audio: UtteranceAudio) -> None:
        pending = self._pending
        assert pending is not None
        try:
            pending.put_nowait(
                _PendingUtterance(
                    audio=audio,
                    consent_generation=audio.consent_generation,
                )
            )
            self.metrics.increment("utterances_queued", user_id=audio.user_id)
        except asyncio.QueueFull:
            self.metrics.increment("utterances_dropped_queue_full", user_id=audio.user_id)

    async def _consume_frames(self) -> None:
        while True:
            frames = await asyncio.to_thread(self._frames.get_many, 64)
            if frames is None:
                break
            for frame in frames:
                with self._control_lock:
                    valid_before_vad = frame.consent_generation == self._generations[
                        frame.user_id
                    ] and self._consents.allows_at_monotonic_ns(
                        frame.user_id,
                        ConsentKind.REALTIME_ASR,
                        frame.received_monotonic_ns,
                    )
                if not valid_before_vad:
                    self.metrics.increment(
                        "dequeued_frames_discarded_after_consent_change",
                        user_id=frame.user_id,
                    )
                    continue
                completed = self._builder.accept(frame)
                with self._control_lock:
                    valid_after_vad = frame.consent_generation == self._generations[
                        frame.user_id
                    ] and self._consents.allows_at_monotonic_ns(
                        frame.user_id,
                        ConsentKind.REALTIME_ASR,
                        frame.received_monotonic_ns,
                    )
                    if not valid_after_vad:
                        self._builder.discard_user(frame.user_id)
                        self.metrics.increment(
                            "dequeued_frames_discarded_after_consent_change",
                            user_id=frame.user_id,
                        )
                        continue
                for utterance in completed:
                    self._enqueue_utterance(utterance)
            await asyncio.sleep(0)
        for utterance in self._builder.flush_all():
            self._enqueue_utterance(utterance)

    async def _expire_inactive_utterances(self) -> None:
        """Advance VAD from the monotonic clock when Discord sends no silence packets."""

        while True:
            await asyncio.sleep(self._endpoint_poll_seconds)
            with self._control_lock:
                if self._stopping:
                    return
            completed = self._builder.expire_inactive(self._clock_ns())
            for utterance in completed:
                if not self._valid_generation(
                    utterance.user_id,
                    utterance.consent_generation,
                ):
                    self.metrics.increment(
                        "utterances_discarded_without_consent",
                        user_id=utterance.user_id,
                    )
                    continue
                self.metrics.increment(
                    "utterances_finalized_by_clock",
                    user_id=utterance.user_id,
                )
                self._enqueue_utterance(utterance)

    async def _recognition_worker(self, worker_index: int) -> None:
        pending = self._pending
        assert pending is not None
        while True:
            item = await pending.get()
            try:
                if item is None:
                    return
                await self._recognise(item)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self.metrics.record_error(exc)
                self._log.error(
                    "realtime_asr_worker_item_failed",
                    worker_index=worker_index,
                    error_type=type(exc).__name__,
                )
            finally:
                pending.task_done()

    async def _recognise(self, item: _PendingUtterance) -> None:
        audio = item.audio
        if not self._valid_generation(audio.user_id, item.consent_generation):
            self.metrics.increment("utterances_discarded_without_consent", user_id=audio.user_id)
            return
        try:
            result = await asyncio.wait_for(
                self._provider.transcribe(
                    audio,
                    language=self._language,
                    hotwords=self._hotwords,
                ),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            self.metrics.increment("provider_timeouts", user_id=audio.user_id)
            self.metrics.record_error(exc)
            return
        except BaseException as exc:
            self.metrics.increment("provider_failures", user_id=audio.user_id)
            self.metrics.record_error(exc)
            self._log.warning(
                "realtime_asr_provider_failed",
                user_id=audio.user_id,
                error_type=type(exc).__name__,
            )
            return
        if not result.normalised_text:
            self.metrics.increment("empty_provider_results", user_id=audio.user_id)
            return
        record = RealtimeUtteranceRecord(
            id=new_ulid(),
            session_id=self._session_id,
            user_id=audio.user_id,
            started_ms=max(
                0,
                (audio.started_monotonic_ns - self._session_started_ns) // 1_000_000,
            ),
            ended_ms=max(
                0,
                (audio.ended_monotonic_ns - self._session_started_ns) // 1_000_000,
            ),
            raw_text=result.raw_text,
            normalised_text=result.normalised_text,
        )
        persistence_lock = self._persistence_locks.setdefault(audio.user_id, asyncio.Lock())
        async with persistence_lock:
            if not self._valid_generation(audio.user_id, item.consent_generation):
                self.metrics.increment(
                    "provider_results_discarded_after_withdrawal",
                    user_id=audio.user_id,
                )
                return
            try:
                await self._utterances.add(record)
            except BaseException as exc:
                self.metrics.increment("persistence_failures", user_id=audio.user_id)
                self.metrics.record_error(exc)
                self._log.error(
                    "realtime_utterance_persistence_failed",
                    user_id=audio.user_id,
                    error_type=type(exc).__name__,
                )
                return
        latency_ms = max(0, (self._clock_ns() - audio.ended_monotonic_ns) // 1_000_000)
        self.metrics.record_latency(latency_ms)
        self.metrics.increment("utterances_final", user_id=audio.user_id)

    async def stop(self) -> None:
        """Drain accepted work, preserving each user's consent boundary."""

        with self._control_lock:
            if not self._started or self._stopping:
                return
            self._stopping = True
            consented_users = self._consents.snapshot().realtime_asr_user_ids
            self._frames.close()
        endpoint_watchdog = self._endpoint_watchdog
        if endpoint_watchdog is not None:
            endpoint_watchdog.cancel()
            await asyncio.gather(endpoint_watchdog, return_exceptions=True)
        consumer = self._consumer
        if consumer is not None:
            await consumer
        pending = self._pending
        assert pending is not None
        for _ in self._workers:
            await pending.put(None)
        await asyncio.gather(*self._workers)
        with self._control_lock:
            for user_id in consented_users:
                self._consents.withdraw(user_id, ConsentKind.REALTIME_ASR)
