from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from lingxu.bot.client import LingxuBot
from lingxu.bot.meeting_voice import MeetingVoiceCoordinator
from lingxu.bot.voice_modes import GuildVoiceMode
from lingxu.domain.enums import MeetingState, ServiceCapacityMode


class FakeLog:
    def __init__(self) -> None:
        self.events: list[str] = []

    def error(self, event: str, **_fields: Any) -> None:
        self.events.append(event)

    def warning(self, event: str, **_fields: Any) -> None:
        self.events.append(event)


class FakeVoiceChannel:
    def __init__(self, guild_id: int, channel_id: int) -> None:
        self.id = channel_id
        self.guild = SimpleNamespace(id=guild_id)


class FakeMeetingRecovery:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.pending = True
        self.connected = False
        self.connect_count = 0
        self.completed: list[tuple[int, str]] = []

    async def retry_failed_wake_phrase_recovery(self, guild_id: int) -> Any | None:
        if not self.pending or guild_id != self.session.guild_id:
            return None
        if not self.connected:
            self.connected = True
            self.connect_count += 1
        return self.session

    def complete_failed_wake_phrase_recovery(
        self,
        *,
        guild_id: int,
        session_id: str,
    ) -> None:
        self.completed.append((guild_id, session_id))
        self.pending = False


class FailOnceFlowController:
    def __init__(self) -> None:
        self.calls: list[set[str]] = []

    async def recover_automatic_flows(self, *, session_ids: set[str]) -> None:
        self.calls.append(session_ids)
        if len(self.calls) == 1:
            raise RuntimeError("synthetic flow recovery failure")


class FakePanels:
    def __init__(self) -> None:
        self.refreshed: list[str] = []

    async def refresh(self, session_id: str) -> None:
        self.refreshed.append(session_id)


class FakeResidentRecovery:
    def __init__(self) -> None:
        self.guild_ids: list[int] = []

    async def retry_failed_wake_phrase_recovery(self, guild_id: int) -> bool:
        self.guild_ids.append(guild_id)
        return True


@pytest.mark.asyncio
async def test_meeting_retry_retains_pending_until_flow_recovers_without_reconnect() -> None:
    session = SimpleNamespace(id="meeting-recovery", guild_id=100)
    meeting = FakeMeetingRecovery(session)
    flows = FailOnceFlowController()
    panels = FakePanels()
    resident = FakeResidentRecovery()
    bot = SimpleNamespace(
        _meeting_voice=meeting,
        meetings=SimpleNamespace(controller=flows),
        _meeting_panels=panels,
        _resident_assistant=resident,
        _log=FakeLog(),
        _wake_recovery_locks={},
    )

    await LingxuBot._retry_failed_wake_recoveries(cast(Any, bot), 100)

    assert meeting.pending
    assert meeting.connect_count == 1
    assert meeting.completed == []
    assert panels.refreshed == []
    assert resident.guild_ids == [100]

    await LingxuBot._retry_failed_wake_recoveries(cast(Any, bot), 100)

    assert not meeting.pending
    assert meeting.connect_count == 1
    assert meeting.completed == [(100, "meeting-recovery")]
    assert flows.calls == [{"meeting-recovery"}, {"meeting-recovery"}]
    assert panels.refreshed == ["meeting-recovery"]
    assert resident.guild_ids == [100, 100]


@pytest.mark.asyncio
async def test_concurrent_full_meeting_retries_run_flow_only_once() -> None:
    session = SimpleNamespace(id="meeting-concurrent", guild_id=110)
    meeting = FakeMeetingRecovery(session)
    panels = FakePanels()
    resident = FakeResidentRecovery()

    class CountingFlowController:
        def __init__(self) -> None:
            self.calls: list[set[str]] = []

        async def recover_automatic_flows(self, *, session_ids: set[str]) -> None:
            self.calls.append(session_ids)
            await asyncio.sleep(0)

    flows = CountingFlowController()
    bot = SimpleNamespace(
        _meeting_voice=meeting,
        meetings=SimpleNamespace(controller=flows),
        _meeting_panels=panels,
        _resident_assistant=resident,
        _log=FakeLog(),
        _wake_recovery_locks={},
    )

    await asyncio.gather(
        LingxuBot._retry_failed_wake_recoveries(cast(Any, bot), 110),
        LingxuBot._retry_failed_wake_recoveries(cast(Any, bot), 110),
    )

    assert meeting.connect_count == 1
    assert flows.calls == [{"meeting-concurrent"}]
    assert panels.refreshed == ["meeting-concurrent"]


@pytest.mark.asyncio
async def test_meeting_voice_marker_requires_explicit_full_recovery_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 200
    channel = FakeVoiceChannel(guild_id, 201)
    session = SimpleNamespace(
        id="meeting-marker",
        guild_id=guild_id,
        state=MeetingState.OPENING,
    )
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    coordinator._bot = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[attr-defined]
    coordinator._application = SimpleNamespace(  # type: ignore[attr-defined]
        sessions=SimpleNamespace(active_for_guild=AsyncMock(return_value=session))
    )
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._failed_wake_recoveries = {}  # type: ignore[attr-defined]
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._voice_modes = SimpleNamespace(current=lambda _guild_id: None)  # type: ignore[attr-defined]
    coordinator._capacity = None  # type: ignore[attr-defined]
    coordinator._log = FakeLog()  # type: ignore[attr-defined]
    connect_count = 0

    async def start_locked(
        *,
        session: Any,
        channel: FakeVoiceChannel,
        protected_owner_id: str,
    ) -> bool:
        nonlocal connect_count
        assert protected_owner_id == session.id
        if guild_id in coordinator._sessions:  # type: ignore[attr-defined]
            return False
        connect_count += 1
        coordinator._sessions[guild_id] = SimpleNamespace(  # type: ignore[attr-defined]
            session=session,
            channel=channel,
            client=SimpleNamespace(is_connected=lambda: True),
        )
        return True

    coordinator._start_locked = cast(Any, start_locked)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.discord.VoiceChannel",
        FakeVoiceChannel,
    )
    coordinator.mark_failed_wake_phrase_recovery(
        session=cast(Any, session),
        channel_id=channel.id,
    )

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is session
    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is session
    assert connect_count == 1

    coordinator.complete_failed_wake_phrase_recovery(
        guild_id=guild_id,
        session_id="different-session",
    )
    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is session
    coordinator.complete_failed_wake_phrase_recovery(
        guild_id=guild_id,
        session_id=session.id,
    )
    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is None


@pytest.mark.asyncio
async def test_meeting_voice_retry_discards_replaced_marker_without_touching_live_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 210
    channel = FakeVoiceChannel(guild_id, 211)
    stale = SimpleNamespace(
        id="meeting-stale",
        guild_id=guild_id,
        state=MeetingState.OPENING,
    )
    durable = SimpleNamespace(
        id="meeting-live",
        guild_id=guild_id,
        state=MeetingState.DISCUSSION_READY,
    )
    live_runtime = SimpleNamespace(session=durable, channel=channel)
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    active_for_guild = AsyncMock(return_value=durable)
    coordinator._application = SimpleNamespace(  # type: ignore[attr-defined]
        sessions=SimpleNamespace(active_for_guild=active_for_guild)
    )
    coordinator._bot = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[attr-defined]
    coordinator._sessions = {guild_id: live_runtime}  # type: ignore[attr-defined]
    coordinator._failed_wake_recoveries = {  # type: ignore[attr-defined]
        guild_id: (stale, channel.id)
    }
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._voice_modes = SimpleNamespace(current=lambda _guild_id: None)  # type: ignore[attr-defined]
    coordinator._capacity = None  # type: ignore[attr-defined]
    coordinator._log = FakeLog()  # type: ignore[attr-defined]
    start_locked = AsyncMock(return_value=True)
    coordinator._start_locked = start_locked  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.discord.VoiceChannel",
        FakeVoiceChannel,
    )

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is None

    active_for_guild.assert_awaited_once_with(guild_id)
    start_locked.assert_not_awaited()
    assert coordinator._sessions[guild_id] is live_runtime  # type: ignore[attr-defined]
    assert guild_id not in coordinator._failed_wake_recoveries  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_meeting_voice_retry_retains_marker_during_temporary_voice_owner_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 215
    channel = FakeVoiceChannel(guild_id, 216)
    session = SimpleNamespace(
        id="meeting-voice-deferred",
        guild_id=guild_id,
        state=MeetingState.OPENING,
    )
    failed = (session, channel.id)
    owner: list[Any | None] = [
        SimpleNamespace(
            mode=GuildVoiceMode.RESIDENT_ASSISTANT,
            owner_id="temporary-resident-owner",
        )
    ]
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    coordinator._application = SimpleNamespace(  # type: ignore[attr-defined]
        sessions=SimpleNamespace(active_for_guild=AsyncMock(return_value=session))
    )
    coordinator._bot = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[attr-defined]
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._failed_wake_recoveries = {guild_id: failed}  # type: ignore[attr-defined]
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._voice_modes = SimpleNamespace(  # type: ignore[attr-defined]
        current=lambda _guild_id: owner[0]
    )
    coordinator._capacity = None  # type: ignore[attr-defined]
    coordinator._log = FakeLog()  # type: ignore[attr-defined]
    start_locked = AsyncMock(return_value=True)
    coordinator._start_locked = start_locked  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.discord.VoiceChannel",
        FakeVoiceChannel,
    )

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is None
    assert coordinator._failed_wake_recoveries[guild_id] == failed  # type: ignore[attr-defined]
    start_locked.assert_not_awaited()

    owner[0] = None
    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is session
    start_locked.assert_awaited_once_with(
        session=session,
        channel=channel,
        protected_owner_id=session.id,
    )


@pytest.mark.asyncio
async def test_meeting_voice_retry_retains_marker_during_temporary_capacity_owner_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 217
    channel = FakeVoiceChannel(guild_id, 218)
    session = SimpleNamespace(
        id="meeting-capacity-deferred",
        guild_id=guild_id,
        state=MeetingState.OPENING,
    )
    failed = (session, channel.id)
    capacity_current = AsyncMock(
        side_effect=(
            SimpleNamespace(
                guild_id=999,
                mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
                owner_id="temporary-capacity-owner",
            ),
            None,
        )
    )
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    coordinator._application = SimpleNamespace(  # type: ignore[attr-defined]
        sessions=SimpleNamespace(active_for_guild=AsyncMock(return_value=session))
    )
    coordinator._bot = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[attr-defined]
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._failed_wake_recoveries = {guild_id: failed}  # type: ignore[attr-defined]
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._voice_modes = SimpleNamespace(current=lambda _guild_id: None)  # type: ignore[attr-defined]
    coordinator._capacity = SimpleNamespace(current=capacity_current)  # type: ignore[attr-defined]
    coordinator._log = FakeLog()  # type: ignore[attr-defined]
    start_locked = AsyncMock(return_value=True)
    coordinator._start_locked = start_locked  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.discord.VoiceChannel",
        FakeVoiceChannel,
    )

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is None
    assert coordinator._failed_wake_recoveries[guild_id] == failed  # type: ignore[attr-defined]
    start_locked.assert_not_awaited()

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is session
    start_locked.assert_awaited_once_with(
        session=session,
        channel=channel,
        protected_owner_id=session.id,
    )


@pytest.mark.parametrize(
    "durable",
    [
        None,
        SimpleNamespace(
            id="meeting-terminal",
            guild_id=220,
            state=MeetingState.STOPPED,
        ),
    ],
    ids=("missing", "terminal"),
)
@pytest.mark.asyncio
async def test_meeting_voice_retry_discards_missing_or_terminal_marker(
    durable: Any | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 220
    channel = FakeVoiceChannel(guild_id, 221)
    marker = SimpleNamespace(
        id="meeting-terminal",
        guild_id=guild_id,
        state=MeetingState.OPENING,
    )
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    active_for_guild = AsyncMock(return_value=durable)
    coordinator._application = SimpleNamespace(  # type: ignore[attr-defined]
        sessions=SimpleNamespace(active_for_guild=active_for_guild)
    )
    coordinator._bot = SimpleNamespace(get_channel=lambda channel_id: channel)  # type: ignore[attr-defined]
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._failed_wake_recoveries = {  # type: ignore[attr-defined]
        guild_id: (marker, channel.id)
    }
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._voice_modes = SimpleNamespace(current=lambda _guild_id: None)  # type: ignore[attr-defined]
    coordinator._capacity = None  # type: ignore[attr-defined]
    coordinator._log = FakeLog()  # type: ignore[attr-defined]
    start_locked = AsyncMock(return_value=True)
    coordinator._start_locked = start_locked  # type: ignore[method-assign]
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.discord.VoiceChannel",
        FakeVoiceChannel,
    )

    assert await coordinator.retry_failed_wake_phrase_recovery(guild_id) is None

    active_for_guild.assert_awaited_once_with(guild_id)
    start_locked.assert_not_awaited()
    assert guild_id not in coordinator._failed_wake_recoveries  # type: ignore[attr-defined]
