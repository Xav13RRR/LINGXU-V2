import asyncio
import inspect
import struct
import threading
from collections.abc import Callable
from dataclasses import replace

import pytest

from lingxu.asr.models import ASRResult, UtteranceAudio
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.service import RealtimeASRMetrics, RealtimeASRService
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.domain.models import RealtimeUtteranceRecord
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice.meeting_consent import MeetingConsentRegistry

_BASE_NS = 1_000_000_000
_FRAME_NS = 20_000_000
_VOICE = struct.pack("<hh", 2_000, -2_000) * 960
_SILENCE = b"\0" * len(_VOICE)


class MemoryStore:
    def __init__(self) -> None:
        self.values: list[RealtimeUtteranceRecord] = []
        self.added = asyncio.Event()

    async def add(self, utterance: RealtimeUtteranceRecord) -> RealtimeUtteranceRecord:
        persisted = replace(utterance)
        self.values.append(persisted)
        self.added.set()
        return persisted

    async def delete_at_or_after(
        self,
        *,
        session_id: str,
        user_id: int,
        cutoff_ms: int,
    ) -> int:
        retained = [
            value
            for value in self.values
            if not (
                value.session_id == session_id
                and value.user_id == user_id
                and value.ended_ms >= cutoff_ms
            )
        ]
        removed = len(self.values) - len(retained)
        self.values = retained
        return removed


class BlockingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.add_started = asyncio.Event()
        self.release_add = asyncio.Event()

    async def add(self, utterance: RealtimeUtteranceRecord) -> RealtimeUtteranceRecord:
        self.add_started.set()
        await self.release_add.wait()
        return await super().add(utterance)


class BlockingDequeueQueue(PerUserRealtimeQueue):
    def __init__(self) -> None:
        super().__init__(max_frames_per_user=20)
        self.dequeued = threading.Event()
        self.release = threading.Event()

    def get_many(self, max_items: int) -> tuple[SessionAudioFrame, ...] | None:
        selected = super().get_many(max_items)
        if selected:
            self.dequeued.set()
            assert self.release.wait(2)
        return selected


class FixtureProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, tuple[str, ...]]] = []

    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        self.calls.append((utterance.user_id, language, hotwords))
        if utterance.user_id == 4:
            raise RuntimeError("isolated fixture failure")
        text = {
            1: ("<|zh|>聆序开始会议", "聆序开始会议", "zh"),
            2: ("<|en|>Lingxu starts the meeting.", "Lingxu starts the meeting.", "en"),
            3: ("<|zh|>聆序 start meeting", "聆序 start meeting", "zh"),
        }.get(utterance.user_id, ("<|en|>recovered", "recovered", "en"))
        return ASRResult(
            raw_text=text[0],
            normalised_text=text[1],
            language=text[2],
            processing_ms=10,
        )


class BlockingProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(
        self,
        utterance: UtteranceAudio,
        *,
        language: str,
        hotwords: tuple[str, ...],
    ) -> ASRResult:
        self.started.set()
        await self.release.wait()
        return ASRResult("raw", "normalised", "en", 1)


def _service(
    provider: object,
    store: MemoryStore,
    *,
    frame_queue: PerUserRealtimeQueue | None = None,
    timeout_seconds: float = 1.0,
    endpoint_poll_seconds: float = 0.05,
    clock_ns: Callable[[], int] = lambda: _BASE_NS + 1_000_000_000,
) -> RealtimeASRService:
    session_id = "session"
    return RealtimeASRService(
        session_id=session_id,
        session_started_monotonic_ns=_BASE_NS,
        consents=MeetingConsentRegistry(),
        frame_queue=frame_queue or PerUserRealtimeQueue(max_frames_per_user=20),
        utterance_builder=UtteranceBuilder(
            session_id=session_id,
            vad_threshold=100,
            vad_start_ms=40,
            vad_end_ms=40,
            min_utterance_ms=40,
        ),
        provider=provider,  # type: ignore[arg-type]
        utterances=store,
        metrics=RealtimeASRMetrics(session_id),
        language="auto",
        hotwords=("聆序", "GRS Studio"),
        timeout_seconds=timeout_seconds,
        max_concurrency=1,
        max_pending_utterances=10,
        endpoint_poll_seconds=endpoint_poll_seconds,
        clock_ns=clock_ns,
    )


def _send_sentence(service: RealtimeASRService, user_id: int, start_index: int = 0) -> None:
    for offset, payload in enumerate((_VOICE, _VOICE, _VOICE, _SILENCE, _SILENCE)):
        assert service.receive(
            user_id=user_id,
            received_monotonic_ns=_BASE_NS + (start_index + offset) * _FRAME_NS,
            pcm_s16le_48k_stereo=payload,
        )


@pytest.mark.asyncio
async def test_zh_en_mixed_hotwords_failure_isolation_and_latency() -> None:
    provider = FixtureProvider()
    store = MemoryStore()
    service = _service(provider, store)
    await service.start()

    assert (
        service.receive(
            user_id=99,
            received_monotonic_ns=_BASE_NS,
            pcm_s16le_48k_stereo=_VOICE,
        )
        is False
    )
    for user_id in (1, 2, 3, 4, 5):
        assert service.grant(user_id)
        _send_sentence(service, user_id, start_index=user_id * 10)

    await service.stop()

    assert [(row.user_id, row.normalised_text) for row in store.values] == [
        (1, "聆序开始会议"),
        (2, "Lingxu starts the meeting."),
        (3, "聆序 start meeting"),
        (5, "recovered"),
    ]
    assert all(call[1:] == ("auto", ("聆序", "GRS Studio")) for call in provider.calls)
    snapshot = service.metrics.snapshot()
    assert snapshot["per_user"]["99"]["frames_rejected_without_consent"] == 1
    assert snapshot["per_user"]["4"]["provider_failures"] == 1
    assert snapshot["counters"]["utterances_final"] == 4
    assert snapshot["latency_ms"]["p95"] is not None
    assert snapshot["latency_ms"]["p95"] <= 4_000


@pytest.mark.asyncio
async def test_withdrawal_invalidates_inflight_provider_output() -> None:
    provider = BlockingProvider()
    store = MemoryStore()
    service = _service(provider, store)
    await service.start()
    service.grant(1)
    _send_sentence(service, 1)
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    assert service.withdraw(1)
    provider.release.set()
    await service.stop()

    assert store.values == []
    snapshot = service.metrics.snapshot()
    assert snapshot["per_user"]["1"]["provider_results_discarded_after_withdrawal"] == 1


@pytest.mark.asyncio
async def test_withdrawal_waits_for_final_persistence_window_and_removes_cutoff_result() -> None:
    store = BlockingStore()
    service = _service(FixtureProvider(), store)
    await service.start()
    service.grant(1)
    _send_sentence(service, 1)
    await asyncio.wait_for(store.add_started.wait(), timeout=1)

    withdrawal_result = service.withdraw_at(
        user_id=1,
        effective_monotonic_ns=_BASE_NS + 5 * _FRAME_NS,
    )
    withdrawal_task = (
        asyncio.create_task(withdrawal_result) if inspect.isawaitable(withdrawal_result) else None
    )
    if withdrawal_task is not None:
        await asyncio.sleep(0)
        assert not withdrawal_task.done()
    store.release_add.set()
    if withdrawal_task is not None:
        await withdrawal_task
    await service.stop()

    assert withdrawal_task is not None
    assert store.values == []


@pytest.mark.asyncio
async def test_dequeued_frames_cannot_cross_withdraw_and_regrant_generation() -> None:
    frame_queue = BlockingDequeueQueue()
    provider = FixtureProvider()
    store = MemoryStore()
    service = _service(provider, store, frame_queue=frame_queue)
    await service.start()
    service.grant(1)
    _send_sentence(service, 1)
    assert await asyncio.to_thread(frame_queue.dequeued.wait, 1)

    assert service.withdraw(1)
    assert service.grant(1)
    frame_queue.release.set()
    await service.stop()

    assert store.values == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_provider_timeout_does_not_deadlock_stop() -> None:
    provider = BlockingProvider()
    store = MemoryStore()
    service = _service(provider, store, timeout_seconds=0.01)
    await service.start()
    service.grant(1)
    _send_sentence(service, 1)

    await service.stop()

    assert store.values == []
    assert service.metrics.snapshot()["per_user"]["1"]["provider_timeouts"] == 1


@pytest.mark.asyncio
async def test_missing_discord_silence_is_finalised_from_wall_clock() -> None:
    provider = FixtureProvider()
    store = MemoryStore()
    now_ns = [_BASE_NS]
    service = _service(
        provider,
        store,
        endpoint_poll_seconds=0.005,
        clock_ns=lambda: now_ns[0],
    )
    await service.start()
    service.grant(1)
    for index in range(2):
        assert service.receive(
            user_id=1,
            received_monotonic_ns=_BASE_NS + index * _FRAME_NS,
            pcm_s16le_48k_stereo=_VOICE,
        )
    await asyncio.sleep(0.02)

    now_ns[0] = _BASE_NS + 840_000_000
    await asyncio.wait_for(store.added.wait(), timeout=1)
    await service.stop()

    assert len(store.values) == 1
    assert store.values[0].started_ms == 0
    assert store.values[0].ended_ms == 40
    assert service.metrics.snapshot()["counters"]["utterances_finalized_by_clock"] == 1
