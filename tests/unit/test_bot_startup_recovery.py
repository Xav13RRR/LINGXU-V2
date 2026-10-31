from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from lingxu.bot.client import LingxuBot
from lingxu.domain.enums import MeetingState


class FakeMeetingVoice:
    def __init__(self) -> None:
        self.marked: list[tuple[str, int]] = []
        self.starts = 0

    async def reconcile_mutes(self) -> None:
        return None

    def mark_failed_wake_phrase_recovery(self, *, session: Any, channel_id: int) -> None:
        self.marked.append((session.id, channel_id))

    async def start(self, *, session: Any, channel: Any) -> bool:
        self.starts += 1
        return True


class FakeFlowController:
    def __init__(self) -> None:
        self.recovered: list[set[str]] = []
        self.handoffs: list[set[str]] = []

    async def reconcile_timer_handoffs(self, *, session_ids: set[str]) -> tuple[Any, ...]:
        self.handoffs.append(session_ids)
        return ()

    async def recover_automatic_flows(self, *, session_ids: set[str]) -> tuple[Any, ...]:
        self.recovered.append(session_ids)
        return ()


@pytest.mark.asyncio
async def test_startup_retains_recovery_when_voice_channel_is_temporarily_unavailable(
    tmp_path: Path,
) -> None:
    session = SimpleNamespace(
        id="pending-meeting",
        guild_id=100,
        publication_channel_id=200,
        state=MeetingState.OPENING,
        resume_state=None,
    )
    voice = FakeMeetingVoice()
    controller = FakeFlowController()

    async def refresh_recoverable() -> None:
        return None

    panels = SimpleNamespace(refresh_recoverable=refresh_recoverable)
    bot = SimpleNamespace(
        _commands_synced=True,
        _capacity_reconciled=True,
        _flows_recovered=False,
        _imports_recovered=True,
        _resident_recovered=True,
        _resident_assistant=None,
        _settings=SimpleNamespace(runtime_tmp=tmp_path),
        _meeting_voice=voice,
        _meeting_panels=panels,
        meetings=SimpleNamespace(
            sessions=SimpleNamespace(list_recoverable=lambda: None),
            controller=controller,
        ),
        get_channel=lambda _channel_id: None,
        user=None,
        _log=SimpleNamespace(error=lambda *_args, **_kwargs: None),
    )

    async def list_recoverable() -> tuple[Any, ...]:
        return (session,)

    bot.meetings.sessions.list_recoverable = list_recoverable

    await LingxuBot.on_ready(cast(Any, bot))

    assert voice.starts == 0
    assert voice.marked == [(session.id, session.publication_channel_id)]
    assert controller.recovered == [set()]
    assert controller.handoffs == []
    assert bot._flows_recovered is True


@pytest.mark.asyncio
async def test_startup_recovers_voice_for_a_paused_meeting_with_voice_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        id="paused-meeting",
        guild_id=101,
        publication_channel_id=201,
        state=MeetingState.PAUSED,
        resume_state=MeetingState.OPENING,
    )
    voice = FakeMeetingVoice()
    controller = FakeFlowController()

    class FakeVoiceChannel:
        pass

    channel = FakeVoiceChannel()
    monkeypatch.setattr("lingxu.bot.client.discord.VoiceChannel", FakeVoiceChannel)

    async def refresh_recoverable() -> None:
        return None

    async def list_recoverable() -> tuple[Any, ...]:
        return (session,)

    bot = SimpleNamespace(
        _commands_synced=True,
        _capacity_reconciled=True,
        _flows_recovered=False,
        _imports_recovered=True,
        _resident_recovered=True,
        _resident_assistant=None,
        _settings=SimpleNamespace(runtime_tmp=tmp_path),
        _meeting_voice=voice,
        _meeting_panels=SimpleNamespace(refresh_recoverable=refresh_recoverable),
        meetings=SimpleNamespace(
            sessions=SimpleNamespace(list_recoverable=list_recoverable),
            controller=controller,
        ),
        get_channel=lambda _channel_id: channel,
        user=None,
        _log=SimpleNamespace(error=lambda *_args, **_kwargs: None),
    )

    await LingxuBot.on_ready(cast(Any, bot))

    assert voice.starts == 1
    assert voice.marked == []
    assert controller.handoffs == [{session.id}]
    assert controller.recovered == [{session.id}]
    assert bot._flows_recovered is True
