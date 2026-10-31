"""Priority playback scheduling, idempotency, fallback, and echo suppression."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Any, Protocol

import structlog

from lingxu.tts.models import InterruptPolicy, PlaybackRequest, TTSAudio
from lingxu.tts.provider import TTSProvider, normalise_tts_text


class PlaybackTarget(Protocol):
    async def play(self, audio: TTSAudio) -> None:
        """Play one complete artifact or raise."""

    def stop(self) -> None:
        """Interrupt current playback if active."""


class TextFallback(Protocol):
    async def send(self, text: str) -> None:
        """Send the original prompt to the meeting text channel."""


class PlaybackEventSink(Protocol):
    async def emit(
        self,
        event_type: str,
        *,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Receive content-free, idempotent scheduler events."""


class EchoGuard:
    """Identify ASR text similar to recent Bot speech during a short debounce."""

    def __init__(self, debounce_ms: int = 350, *, clock_ns: Any = time.monotonic_ns) -> None:
        if not 200 <= debounce_ms <= 500:
            raise ValueError("echo debounce must be between 200 and 500 ms")
        self._debounce_ns = debounce_ms * 1_000_000
        self._clock_ns = clock_ns
        self._active: str | None = None
        self._recent: deque[tuple[str, int]] = deque()
        self._lock = threading.RLock()

    @staticmethod
    def _canonical(text: str) -> str:
        return "".join(
            character.casefold() for character in normalise_tts_text(text) if character.isalnum()
        )

    def begin(self, text: str) -> None:
        with self._lock:
            self._active = self._canonical(text)

    def finish(self) -> None:
        with self._lock:
            if self._active:
                self._recent.append((self._active, self._clock_ns() + self._debounce_ns))
            self._active = None
            self._purge()

    def _purge(self) -> None:
        now = self._clock_ns()
        while self._recent and self._recent[0][1] < now:
            self._recent.popleft()

    @staticmethod
    def _similar(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if len(left) >= 2 and len(right) >= 2 and (left in right or right in left):
            return True
        return SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.82

    def is_echo(self, transcript: str) -> bool:
        candidate = self._canonical(transcript)
        with self._lock:
            self._purge()
            references = ([self._active] if self._active else []) + [
                text for text, _ in self._recent
            ]
            return any(self._similar(candidate, reference) for reference in references)


class PlaybackScheduler:
    """Serialise Discord voice send with priority and failure-safe completion."""

    def __init__(
        self,
        *,
        session_id: str,
        provider: TTSProvider,
        target: PlaybackTarget,
        fallback: TextFallback,
        events: PlaybackEventSink,
        echo_guard: EchoGuard,
    ) -> None:
        self._session_id = session_id
        self._provider = provider
        self._target = target
        self._fallback = fallback
        self._events = events
        self.echo_guard = echo_guard
        self._queue: asyncio.PriorityQueue[tuple[int, int, PlaybackRequest | None]] = (
            asyncio.PriorityQueue()
        )
        self._sequence = 0
        self._known_request_ids: set[str] = set()
        self._known_dedupe_keys: set[str] = set()
        self._finished_ids: set[str] = set()
        self._interrupted_ids: set[str] = set()
        self._current: PlaybackRequest | None = None
        self._worker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._stopping = False
        self._log = structlog.get_logger(__name__).bind(session_id=session_id)

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("playback scheduler already started")
        self._worker = asyncio.create_task(
            self._run(),
            name=f"lingxu-playback-{self._session_id}",
        )

    async def enqueue(self, request: PlaybackRequest) -> bool:
        """Deduplicate and queue, optionally interrupting lower-priority audio."""

        async with self._lock:
            if self._stopping:
                return False
            if (
                request.request_id in self._known_request_ids
                or request.dedupe_key in self._known_dedupe_keys
            ):
                return False
            self._known_request_ids.add(request.request_id)
            self._known_dedupe_keys.add(request.dedupe_key)
            self._sequence += 1
            self._queue.put_nowait((-int(request.priority), self._sequence, request))
            current = self._current
            should_interrupt = (
                current is not None
                and request.interrupt_policy is not InterruptPolicy.NEVER
                and (
                    request.interrupt_policy is InterruptPolicy.ALWAYS
                    or request.priority > current.priority
                )
            )
            if should_interrupt:
                assert current is not None
                self._interrupted_ids.add(current.request_id)
                self._target.stop()
            return True

    async def _consume_interrupted(self, request_id: str) -> bool:
        async with self._lock:
            if request_id not in self._interrupted_ids:
                return False
            self._interrupted_ids.remove(request_id)
            return True

    async def finish_once(
        self,
        request: PlaybackRequest,
        *,
        status: str,
    ) -> bool:
        """Emit exactly one `TTS_FINISHED` for a request ID."""

        async with self._lock:
            if request.request_id in self._finished_ids:
                return False
            self._finished_ids.add(request.request_id)
        try:
            await self._events.emit(
                "TTS_FINISHED",
                request_id=request.request_id,
                payload={"status": status, "priority": int(request.priority)},
            )
        except BaseException as exc:
            self._log.error(
                "tts_finished_event_failed",
                request_id=request.request_id,
                error_type=type(exc).__name__,
            )
        return True

    async def _fallback_once(self, request: PlaybackRequest, error: BaseException) -> None:
        try:
            await self._fallback.send(request.text)
        except BaseException as fallback_error:
            self._log.error(
                "tts_text_fallback_failed",
                request_id=request.request_id,
                error_type=type(fallback_error).__name__,
            )
        self._log.warning(
            "tts_playback_failed",
            request_id=request.request_id,
            error_type=type(error).__name__,
        )

    async def _run(self) -> None:
        while True:
            _, _, request = await self._queue.get()
            try:
                if request is None:
                    return
                async with self._lock:
                    self._current = request
                status = "COMPLETED"
                self.echo_guard.begin(request.text)
                try:
                    audio = await self._provider.synthesize(request.text)
                    await self._target.play(audio)
                except BaseException as exc:
                    if await self._consume_interrupted(request.request_id):
                        status = "INTERRUPTED"
                    else:
                        status = "FALLBACK"
                        await self._fallback_once(request, exc)
                else:
                    if await self._consume_interrupted(request.request_id):
                        status = "INTERRUPTED"
                finally:
                    self.echo_guard.finish()
                    async with self._lock:
                        if self._current is request:
                            self._current = None
                await self.finish_once(request, status=status)
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        """Drain accepted requests and stop without leaving a waiter."""

        async with self._lock:
            if self._stopping:
                worker = self._worker
            else:
                self._stopping = True
                self._sequence += 1
                self._queue.put_nowait((1_000_000, self._sequence, None))
                worker = self._worker
        if worker is not None:
            await worker
