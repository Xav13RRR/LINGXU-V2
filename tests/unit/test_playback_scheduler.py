import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lingxu.tts.models import (
    InterruptPolicy,
    PlaybackPriority,
    PlaybackRequest,
    TTSAudio,
)
from lingxu.tts.scheduler import EchoGuard, PlaybackScheduler


class FakeProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> TTSAudio:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("fixture synthesis failure")
        return TTSAudio("key", Path("audio.wav"), "0" * 64, 100, 22_050, 1, False)


class FakeTarget:
    def __init__(self) -> None:
        self.played: list[TTSAudio] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stop_calls = 0

    async def play(self, audio: TTSAudio) -> None:
        self.played.append(audio)
        self.started.set()
        await self.release.wait()

    def stop(self) -> None:
        self.stop_calls += 1
        self.release.set()


class FakeFallback:
    def __init__(self) -> None:
        self.values: list[str] = []

    async def send(self, text: str) -> None:
        self.values.append(text)


@dataclass
class Emitted:
    event_type: str
    request_id: str
    payload: dict[str, Any]


class FakeEvents:
    def __init__(self) -> None:
        self.values: list[Emitted] = []

    async def emit(
        self,
        event_type: str,
        *,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.values.append(Emitted(event_type, request_id, payload))


def _request(
    request_id: str,
    *,
    priority: PlaybackPriority = PlaybackPriority.NORMAL_NOTICE,
    policy: InterruptPolicy = InterruptPolicy.LOWER_PRIORITY,
) -> PlaybackRequest:
    return PlaybackRequest(
        request_id, f"dedupe-{request_id}", f"提示 {request_id}", priority, policy
    )


@pytest.mark.asyncio
async def test_priority_interrupt_dedupe_and_finished_idempotency() -> None:
    provider = FakeProvider()
    target = FakeTarget()
    fallback = FakeFallback()
    events = FakeEvents()
    scheduler = PlaybackScheduler(
        session_id="session",
        provider=provider,
        target=target,
        fallback=fallback,
        events=events,
        echo_guard=EchoGuard(),
    )
    await scheduler.start()
    normal = _request("normal")
    emergency = _request("emergency", priority=PlaybackPriority.EMERGENCY)
    assert await scheduler.enqueue(normal)
    assert await scheduler.enqueue(normal) is False
    await asyncio.wait_for(target.started.wait(), timeout=1)

    assert await scheduler.enqueue(emergency)
    assert target.stop_calls == 1
    target.started.clear()
    await asyncio.wait_for(target.started.wait(), timeout=1)
    target.release.set()
    await scheduler.stop()

    assert provider.calls == ["提示 normal", "提示 emergency"]
    assert [value.request_id for value in events.values] == ["normal", "emergency"]
    assert events.values[0].payload["status"] == "INTERRUPTED"
    assert fallback.values == []
    assert await scheduler.finish_once(normal, status="COMPLETED") is False


@pytest.mark.asyncio
async def test_failure_falls_back_and_does_not_deadlock() -> None:
    fallback = FakeFallback()
    events = FakeEvents()
    scheduler = PlaybackScheduler(
        session_id="session",
        provider=FakeProvider(fail=True),
        target=FakeTarget(),
        fallback=fallback,
        events=events,
        echo_guard=EchoGuard(),
    )
    await scheduler.start()
    request = _request("failure")
    assert await scheduler.enqueue(request)

    await asyncio.wait_for(scheduler.stop(), timeout=1)

    assert fallback.values == ["提示 failure"]
    assert events.values[0].payload["status"] == "FALLBACK"


def test_echo_guard_active_debounce_and_expiry() -> None:
    now = [1_000_000_000]
    guard = EchoGuard(debounce_ms=300, clock_ns=lambda: now[0])
    guard.begin("会议暂停。")

    assert guard.is_echo("会议暂停")
    assert guard.is_echo("暂停")
    assert not guard.is_echo("继续会议")
    guard.finish()
    now[0] += 250_000_000
    assert guard.is_echo("会议暂停")
    now[0] += 100_000_000
    assert not guard.is_echo("会议暂停")
