"""Bounded, consent-independent, ephemeral Meeting voice assistant turn loop."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

from lingxu.asr.models import UtteranceAudio
from lingxu.asr.provider import PreparableRealtimeASRProvider, RealtimeASRProvider
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.domain.wake_phrase import normalise_wake_phrase
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice_assistant.fuzzy import split_wake_query
from lingxu.voice_assistant.response import ShortAnswerProvider
from lingxu.voice_assistant.wakeword import KeywordSpotter
from lingxu.voice_assistant.window import WakeWindowBuilder

ContextProvider = Callable[[], Awaitable[dict[str, Any]]]
SpeakCallback = Callable[[str], Awaitable[None]]


class AssistantTurnState(StrEnum):
    IDLE = "IDLE"
    CAPTURING_QUERY = "CAPTURING_QUERY"
    PROCESSING = "PROCESSING"
    AWAITING_QUERY = "AWAITING_QUERY"
    SPEAKING = "SPEAKING"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class _WakeWindowEvent:
    audio: UtteranceAudio
    generation: int


@dataclass(frozen=True, slots=True)
class _CompletedUtteranceEvent:
    audio: UtteranceAudio
    generation: int


type _AssistantEvent = _WakeWindowEvent | _CompletedUtteranceEvent


class _WakeEnqueueResult(StrEnum):
    QUEUED = "QUEUED"
    COALESCED = "COALESCED"
    FULL = "FULL"
    CLOSED = "CLOSED"


class _AssistantEventBuffer:
    """Prioritise completed speech and fairly coalesce rolling wake work per user.

    All methods run on the owning asyncio event loop. A user can retain at most one queued wake
    window plus one latest window while that user's current window is being inspected. This keeps
    overlapping 500 ms windows from crowding out other users or completed utterances.
    """

    def __init__(self, *, max_pending: int) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._completed: deque[_CompletedUtteranceEvent] = deque()
        self._wake_pending: dict[int, _WakeWindowEvent] = {}
        self._wake_order: deque[int] = deque()
        self._wake_inflight: set[int] = set()
        self._wake_deferred: dict[int, _WakeWindowEvent] = {}
        self._work_ready = asyncio.Event()
        self._completed_space = asyncio.Event()
        self._completed_space.set()
        self._closed = False

    def put_wake(self, event: _WakeWindowEvent) -> _WakeEnqueueResult:
        if self._closed:
            return _WakeEnqueueResult.CLOSED
        user_id = event.audio.user_id
        if user_id in self._wake_pending:
            self._wake_pending[user_id] = event
            return _WakeEnqueueResult.COALESCED
        if user_id in self._wake_inflight:
            self._wake_deferred[user_id] = event
            return _WakeEnqueueResult.COALESCED
        if len(self._wake_pending) + len(self._wake_inflight) >= self._max_pending:
            return _WakeEnqueueResult.FULL
        self._wake_pending[user_id] = event
        self._wake_order.append(user_id)
        self._work_ready.set()
        return _WakeEnqueueResult.QUEUED

    async def put_completed(self, event: _CompletedUtteranceEvent) -> bool:
        while not self._closed and len(self._completed) >= self._max_pending:
            self._completed_space.clear()
            if len(self._completed) < self._max_pending:
                continue
            await self._completed_space.wait()
        if self._closed:
            return False
        self._completed.append(event)
        if len(self._completed) >= self._max_pending:
            self._completed_space.clear()
        self._work_ready.set()
        return True

    async def get(self) -> _AssistantEvent | None:
        while True:
            if self._completed:
                completed_event = self._completed.popleft()
                self._completed_space.set()
                return completed_event
            while self._wake_order:
                user_id = self._wake_order.popleft()
                wake_event = self._wake_pending.pop(user_id, None)
                if wake_event is None:
                    continue
                self._wake_inflight.add(user_id)
                return wake_event
            if self._closed:
                return None
            self._work_ready.clear()
            if self._completed or self._wake_order or self._closed:
                continue
            await self._work_ready.wait()

    def wake_done(self, user_id: int) -> None:
        self._wake_inflight.discard(user_id)
        deferred = self._wake_deferred.pop(user_id, None)
        if deferred is None:
            return
        self._wake_pending[user_id] = deferred
        self._wake_order.append(user_id)
        self._work_ready.set()

    def close(self) -> None:
        self._closed = True
        self._work_ready.set()
        self._completed_space.set()


class MeetingVoiceAssistant:
    """Allow one non-queued wake turn while keeping all caller content ephemeral."""

    def __init__(
        self,
        *,
        session_id: str,
        wake_phrase: str,
        frame_queue: PerUserRealtimeQueue,
        utterance_builder: UtteranceBuilder,
        wake_window_builder: WakeWindowBuilder,
        keyword_spotter: KeywordSpotter,
        asr_provider: RealtimeASRProvider,
        answer_provider: ShortAnswerProvider,
        context_provider: ContextProvider,
        speak: SpeakCallback,
        language: str = "auto",
        asr_timeout_seconds: float = 8.0,
        followup_seconds: float = 4.0,
        quarantine_ms: int = 600,
        max_pending_utterances: int = 8,
        endpoint_poll_seconds: float = 0.05,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if endpoint_poll_seconds <= 0:
            raise ValueError("endpoint_poll_seconds must be positive")
        self._session_id = session_id
        self._wake_phrase = normalise_wake_phrase(wake_phrase)
        self._frames = frame_queue
        self._builder = utterance_builder
        self._wake_windows = wake_window_builder
        self._keyword_spotter = keyword_spotter
        self._asr = asr_provider
        self._answers = answer_provider
        self._context_provider = context_provider
        self._speak = speak
        self._language = language
        self._asr_timeout = asr_timeout_seconds
        self._followup_seconds = followup_seconds
        self._quarantine_ns = quarantine_ms * 1_000_000
        self._max_pending = max_pending_utterances
        self._endpoint_poll_seconds = endpoint_poll_seconds
        self._clock_ns = clock_ns
        self._control = threading.RLock()
        self._state = AssistantTurnState.STOPPED
        self._active_user_id: int | None = None
        self._active_wake_phrase: str | None = None
        self._external_output_depth = 0
        self._quarantine_until_ns = 0
        self._stopping = False
        self._input_generation = 0
        self._pending: _AssistantEventBuffer | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._endpoint_watchdog: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._followup_task: asyncio.Task[None] | None = None
        self._counters: Counter[str] = Counter()
        self._prepare_lock = asyncio.Lock()
        self._wake_phrase_lock = asyncio.Lock()
        self._prepared = False
        self._log = structlog.get_logger(__name__).bind(session_id=session_id)

    @property
    def state(self) -> AssistantTurnState:
        with self._control:
            return self._state

    @property
    def wake_phrase(self) -> str:
        with self._control:
            return self._wake_phrase

    def metrics_snapshot(self) -> dict[str, int]:
        with self._control:
            return dict(sorted(self._counters.items()))

    def _increment(self, name: str) -> None:
        with self._control:
            self._counters[name] += 1

    async def prepare(self) -> None:
        """Load every assistant dependency before the Discord voice connection."""

        async with self._prepare_lock:
            if self._prepared:
                return
            preparations: list[Awaitable[None]] = [
                self._keyword_spotter.prepare(),
                self._answers.prepare(),
            ]
            if isinstance(self._asr, PreparableRealtimeASRProvider):
                preparations.append(self._asr.prepare())
            await asyncio.gather(*preparations)
            self._prepared = True

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        """Prepare a new acoustic path and apply it to subsequent turns atomically."""

        selected = normalise_wake_phrase(wake_phrase)
        async with self._wake_phrase_lock:
            with self._control:
                if selected == self._wake_phrase:
                    return
            operation = asyncio.create_task(
                self._apply_wake_phrase(selected),
                name=f"meeting-assistant-wake-update-{self._session_id}",
            )
            cancellation: asyncio.CancelledError | None = None
            while True:
                try:
                    await asyncio.shield(operation)
                    break
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                    if operation.done():
                        break
            operation.result()
            if cancellation is not None:
                raise cancellation

    async def _apply_wake_phrase(self, selected: str) -> None:
        await self._keyword_spotter.update_wake_phrase(selected)
        with self._control:
            self._wake_phrase = selected
            self._input_generation += 1
            self._counters["wake_phrase_updates"] += 1
            if self._state in {
                AssistantTurnState.CAPTURING_QUERY,
                AssistantTurnState.AWAITING_QUERY,
            }:
                self._state = AssistantTurnState.IDLE
                self._active_user_id = None
                self._active_wake_phrase = None
                followup = self._followup_task
                self._followup_task = None
                self._counters["turns_cancelled_by_wake_phrase_update"] += 1
            else:
                followup = None
        if followup is not None:
            followup.cancel()
        self._frames.clear_all()
        self._builder.discard_all()
        self._wake_windows.discard_all()

    async def start(self) -> None:
        with self._control:
            if self._state is not AssistantTurnState.STOPPED or self._stopping:
                raise RuntimeError("Meeting voice assistant already started")
        await self.prepare()
        with self._control:
            self._state = AssistantTurnState.IDLE
            self._stopping = False
            self._pending = _AssistantEventBuffer(max_pending=self._max_pending)
        self._consumer = asyncio.create_task(
            self._consume_frames(),
            name=f"meeting-assistant-vad-{self._session_id}",
        )
        self._endpoint_watchdog = asyncio.create_task(
            self._expire_inactive_utterances(),
            name=f"meeting-assistant-endpoint-{self._session_id}",
        )
        self._worker = asyncio.create_task(
            self._process_events(),
            name=f"meeting-assistant-turn-{self._session_id}",
        )

    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool:
        """Perform bounded checks/copy/queue insertion on the Discord receive thread."""

        with self._control:
            now_ns = self._clock_ns()
            allowed = (
                not self._stopping
                and self._external_output_depth == 0
                and now_ns >= self._quarantine_until_ns
                and (
                    self._state is AssistantTurnState.IDLE
                    or (
                        self._state
                        in {
                            AssistantTurnState.CAPTURING_QUERY,
                            AssistantTurnState.AWAITING_QUERY,
                        }
                        and self._active_user_id == user_id
                    )
                )
            )
            if not allowed:
                self._counters["frames_dropped_busy"] += 1
                return False
            frame = SessionAudioFrame(
                session_id=self._session_id,
                user_id=user_id,
                received_monotonic_ns=received_monotonic_ns,
                pcm_s16le_48k_stereo=bytes(pcm_s16le_48k_stereo),
            )
            dropped = self._frames.put(frame)
            self._counters["frames_accepted"] += 1
            if dropped:
                self._counters["frames_dropped_queue_full"] += 1
            return True

    def begin_bot_output(self) -> None:
        """Block new invocations before any Bot synthesis or playback starts."""

        with self._control:
            self._external_output_depth += 1
            self._input_generation += 1
            self._counters["bot_output_started"] += 1
            if self._state in {
                AssistantTurnState.CAPTURING_QUERY,
                AssistantTurnState.AWAITING_QUERY,
            }:
                self._state = AssistantTurnState.IDLE
                self._active_user_id = None
                self._active_wake_phrase = None
                self._counters["turns_cancelled_by_bot_output"] += 1
                followup = self._followup_task
                self._followup_task = None
            else:
                followup = None
        if followup is not None:
            followup.cancel()
        self._frames.clear_all()
        self._builder.discard_all()
        self._wake_windows.discard_all()

    def end_bot_output(self) -> None:
        """Keep a short post-playback quarantine so echo cannot become a wake turn."""

        with self._control:
            self._external_output_depth = max(0, self._external_output_depth - 1)
            self._quarantine_until_ns = max(
                self._quarantine_until_ns,
                self._clock_ns() + self._quarantine_ns,
            )
            self._counters["bot_output_finished"] += 1
        self._frames.clear_all()
        self._builder.discard_all()
        self._wake_windows.discard_all()

    def _generation(self) -> int:
        with self._control:
            return self._input_generation

    def _event_is_current(self, generation: int) -> bool:
        with self._control:
            return not self._stopping and generation == self._input_generation

    def _enqueue_window(self, event: _WakeWindowEvent) -> None:
        pending = self._pending
        assert pending is not None
        result = pending.put_wake(event)
        if result is _WakeEnqueueResult.QUEUED:
            self._increment("wake_windows_queued")
        elif result is _WakeEnqueueResult.COALESCED:
            self._increment("wake_windows_queued")
            self._increment("wake_windows_coalesced")
        elif result is _WakeEnqueueResult.FULL:
            self._increment("wake_windows_dropped_queue_full")
        else:
            self._increment("wake_windows_dropped_stopping")

    async def _enqueue_completed(self, event: _CompletedUtteranceEvent) -> None:
        """Preserve query completion while keeping the receive callback non-blocking."""

        pending = self._pending
        assert pending is not None
        if await pending.put_completed(event):
            self._increment("utterances_queued")
        else:
            self._increment("utterances_dropped_stopping")

    async def _consume_frames(self) -> None:
        while True:
            frames = await asyncio.to_thread(self._frames.get_many, 64)
            if frames is None:
                return
            for frame in frames:
                generation = self._generation()
                with self._control:
                    state = self._state
                if state is AssistantTurnState.IDLE:
                    window = self._wake_windows.accept(frame)
                    if window is not None:
                        self._enqueue_window(_WakeWindowEvent(window, generation))
                else:
                    self._wake_windows.discard_user(frame.user_id)
                for utterance in self._builder.accept(frame):
                    self._wake_windows.discard_user(utterance.user_id)
                    await self._enqueue_completed(_CompletedUtteranceEvent(utterance, generation))
            await asyncio.sleep(0)

    async def _expire_inactive_utterances(self) -> None:
        """Finalise a caller after wall-clock silence even when Discord sends no PCM."""

        while True:
            await asyncio.sleep(self._endpoint_poll_seconds)
            with self._control:
                if self._stopping:
                    return
                generation = self._input_generation
            for utterance in self._builder.expire_inactive(self._clock_ns()):
                self._wake_windows.discard_user(utterance.user_id)
                self._increment("utterances_finalized_by_clock")
                await self._enqueue_completed(_CompletedUtteranceEvent(utterance, generation))

    def _claim_followup(self, user_id: int) -> bool:
        with self._control:
            if (
                self._stopping
                or self._state is not AssistantTurnState.AWAITING_QUERY
                or self._active_user_id != user_id
            ):
                return False
            self._state = AssistantTurnState.PROCESSING
            return True

    def _claim_wake(self, user_id: int, wake_phrase: str) -> bool:
        with self._control:
            if (
                self._stopping
                or self._state is not AssistantTurnState.IDLE
                or self._external_output_depth
                or self._clock_ns() < self._quarantine_until_ns
            ):
                return False
            self._state = AssistantTurnState.CAPTURING_QUERY
            self._active_user_id = user_id
            self._active_wake_phrase = wake_phrase
            self._counters["wake_turns_claimed"] += 1
        self._wake_windows.discard_all()
        return True

    def _claim_captured_query(self, user_id: int) -> bool:
        with self._control:
            if (
                self._stopping
                or self._state is not AssistantTurnState.CAPTURING_QUERY
                or self._active_user_id != user_id
            ):
                return False
            self._state = AssistantTurnState.PROCESSING
            return True

    async def _process_events(self) -> None:
        pending = self._pending
        assert pending is not None
        while True:
            event = await pending.get()
            try:
                if event is None:
                    return
                if not self._event_is_current(event.generation):
                    self._increment("events_dropped_stale")
                    continue
                if isinstance(event, _WakeWindowEvent):
                    await self._process_wake_window(event.audio, generation=event.generation)
                else:
                    await self._process_completed_utterance(
                        event.audio,
                        generation=event.generation,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._increment("turn_failures")
                self._log.error(
                    "meeting_assistant_turn_failed",
                    error_type=type(exc).__name__,
                )
                self._release_turn()
            finally:
                if isinstance(event, _WakeWindowEvent):
                    pending.wake_done(event.audio.user_id)

    async def _detect_wake(self, audio: UtteranceAudio) -> tuple[bool, str]:
        async with self._wake_phrase_lock:
            with self._control:
                wake_phrase = self._wake_phrase
            return await self._keyword_spotter.detect(audio), wake_phrase

    async def _process_wake_window(
        self,
        window: UtteranceAudio,
        *,
        generation: int,
    ) -> None:
        with self._control:
            if self._state is not AssistantTurnState.IDLE:
                self._counters["wake_windows_dropped_busy"] += 1
                return
        try:
            detected, wake_phrase = await self._detect_wake(window)
        except Exception as exc:
            self._increment("wakeword_failures")
            self._log.warning(
                "meeting_assistant_wakeword_failed",
                error_type=type(exc).__name__,
            )
            return
        if not detected:
            self._increment("wake_windows_without_wake")
            return
        if not self._event_is_current(generation):
            self._increment("events_dropped_stale")
            return
        self._increment("wake_windows_detected")
        if not self._claim_wake(window.user_id, wake_phrase):
            self._increment("wake_candidates_dropped_busy")

    async def _process_completed_utterance(
        self,
        utterance: UtteranceAudio,
        *,
        generation: int,
    ) -> None:
        with self._control:
            state = self._state
            active_user_id = self._active_user_id
        if state is AssistantTurnState.CAPTURING_QUERY and active_user_id == utterance.user_id:
            if self._claim_captured_query(utterance.user_id):
                await self._handle_claimed(utterance, require_wake_occurrence=True)
            return
        if state is AssistantTurnState.AWAITING_QUERY and active_user_id == utterance.user_id:
            if self._claim_followup(utterance.user_id):
                await self._handle_claimed(utterance, require_wake_occurrence=False)
            return
        if state is not AssistantTurnState.IDLE:
            self._increment("utterances_dropped_busy")
            return
        self._increment("utterance_wake_fallbacks_started")
        try:
            detected, wake_phrase = await self._detect_wake(utterance)
        except Exception as exc:
            self._increment("wakeword_failures")
            self._log.warning(
                "meeting_assistant_wakeword_failed",
                error_type=type(exc).__name__,
            )
            return
        if not detected:
            self._increment("utterance_wake_fallbacks_without_wake")
            self._increment("utterances_completed_without_claim")
            return
        if not self._event_is_current(generation):
            self._increment("events_dropped_stale")
            return
        self._increment("utterance_wake_fallbacks_detected")
        if not self._claim_wake(utterance.user_id, wake_phrase):
            self._increment("wake_candidates_dropped_busy")
            return
        if self._claim_captured_query(utterance.user_id):
            await self._handle_claimed(utterance, require_wake_occurrence=True)

    async def _transcribe(self, utterance: UtteranceAudio, *, wake_phrase: str) -> str:
        result = await asyncio.wait_for(
            self._asr.transcribe(
                utterance,
                language=self._language,
                hotwords=(wake_phrase,),
            ),
            timeout=self._asr_timeout,
        )
        return result.normalised_text.strip()

    async def _handle_claimed(
        self,
        utterance: UtteranceAudio,
        *,
        require_wake_occurrence: bool,
    ) -> None:
        with self._control:
            wake_phrase = self._active_wake_phrase or self._wake_phrase
        try:
            text = await self._transcribe(utterance, wake_phrase=wake_phrase)
        except Exception as exc:
            self._increment("asr_failures")
            self._log.warning(
                "meeting_assistant_asr_failed",
                error_type=type(exc).__name__,
            )
            self._release_turn()
            return
        if not text:
            self._increment("empty_asr_results")
            self._release_turn()
            return
        query = split_wake_query(text, wake_phrase=wake_phrase) if require_wake_occurrence else text
        if query is None:
            self._increment("wake_occurrence_rejected")
            self._release_turn()
            return
        if not query.strip():
            self._arm_followup()
            return
        await self._answer_and_speak(query.strip())

    def _arm_followup(self) -> None:
        with self._control:
            if self._stopping or self._state is not AssistantTurnState.PROCESSING:
                return
            self._state = AssistantTurnState.AWAITING_QUERY
            self._counters["followups_armed"] += 1
        if self._followup_task is not None:
            self._followup_task.cancel()
        self._followup_task = asyncio.create_task(
            self._expire_followup(),
            name=f"meeting-assistant-followup-{self._session_id}",
        )

    async def _expire_followup(self) -> None:
        try:
            await asyncio.sleep(self._followup_seconds)
        except asyncio.CancelledError:
            return
        with self._control:
            if self._state is not AssistantTurnState.AWAITING_QUERY:
                return
            self._counters["followups_expired"] += 1
        self._release_turn()

    async def _answer_and_speak(self, question: str) -> None:
        followup = self._followup_task
        self._followup_task = None
        if followup is not None and followup is not asyncio.current_task():
            followup.cancel()
        try:
            context = await self._context_provider()
            answer = await self._answers.answer(question=question, context=context)
        except Exception as exc:
            self._increment("answer_failures")
            self._log.warning(
                "meeting_assistant_answer_failed",
                error_type=type(exc).__name__,
            )
            answer = "抱歉，语音助手暂时无法回答。"
        with self._control:
            if self._stopping:
                self._release_turn()
                return
            self._state = AssistantTurnState.SPEAKING
        try:
            await self._speak(answer)
            self._increment("answers_completed")
        except Exception as exc:
            self._increment("playback_failures")
            self._log.warning(
                "meeting_assistant_playback_failed",
                error_type=type(exc).__name__,
            )
        finally:
            self._release_turn()

    def _release_turn(self) -> None:
        with self._control:
            if self._stopping:
                self._state = AssistantTurnState.STOPPED
            else:
                self._state = AssistantTurnState.IDLE
                self._quarantine_until_ns = max(
                    self._quarantine_until_ns,
                    self._clock_ns() + self._quarantine_ns,
                )
            self._active_user_id = None
            self._active_wake_phrase = None
            self._input_generation += 1
        self._frames.clear_all()
        self._builder.discard_all()
        self._wake_windows.discard_all()

    async def stop(self) -> None:
        with self._control:
            if self._stopping or self._state is AssistantTurnState.STOPPED:
                return
            self._stopping = True
            self._state = AssistantTurnState.STOPPED
            self._active_user_id = None
            self._active_wake_phrase = None
            self._frames.close()
        endpoint_watchdog = self._endpoint_watchdog
        if endpoint_watchdog is not None:
            endpoint_watchdog.cancel()
            await asyncio.gather(endpoint_watchdog, return_exceptions=True)
        followup = self._followup_task
        self._followup_task = None
        if followup is not None:
            followup.cancel()
        consumer = self._consumer
        if consumer is not None:
            await consumer
        pending = self._pending
        worker = self._worker
        if pending is not None and worker is not None:
            pending.close()
            await worker
        self._builder.discard_all()
        self._wake_windows.discard_all()
        self._log.info(
            "meeting_assistant_stopped",
            metrics=self.metrics_snapshot(),
        )
