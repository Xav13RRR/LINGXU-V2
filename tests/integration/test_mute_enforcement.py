from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import discord
import pytest

from lingxu.bot.mute_enforcement import MeetingMuteCoordinator
from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    AgendaStatus,
    MeetingEventType,
    MeetingState,
    MuteLeaseOwnership,
    SpeakingFlowStage,
    TimerStatus,
)
from lingxu.domain.models import AgendaItemRecord, TimerRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.meeting.components import SpeakingQueueState, TimerService
from lingxu.meeting.controller import MeetingStatus
from lingxu.meeting.factory import MeetingApplication
from lingxu.meeting.mute_enforcement import MuteEnforcementService, MuteEnforcementState
from lingxu.meeting.speaking_flow import SpeakingFlowState, SpeakingIdentity
from lingxu.storage.component_repositories import TimerRepository
from lingxu.storage.database import Database
from lingxu.storage.mute_lease_repository import MuteLeaseRepository
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


class FakeMember:
    def __init__(self, user_id: int, *, muted: bool = False) -> None:
        self.id = user_id
        self.bot = False
        self.voice = SimpleNamespace(mute=muted)
        self.edits: list[bool] = []

    async def edit(self, *, mute: bool, reason: str) -> None:
        assert "Lingxu mute lease" in reason
        self.edits.append(mute)
        self.voice.mute = mute


async def _services(
    tmp_path: Path,
) -> tuple[
    Database,
    MeetingSessionRepository,
    MeetingEventStore,
    TimerService,
    MuteEnforcementService,
    MuteLeaseRepository,
]:
    database = Database.open(_database_url(tmp_path / "mute.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    events = MeetingEventStore(database.sessions)
    timers = TimerService(timers=TimerRepository(database.sessions), events=events)
    return (
        database,
        sessions,
        events,
        timers,
        MuteEnforcementService(events=events, timers=timers),
        MuteLeaseRepository(database.sessions),
    )


@pytest.mark.asyncio
async def test_mute_lease_preserves_first_observation_and_releases_idempotently(
    tmp_path: Path,
) -> None:
    database, sessions, events, _, _, leases = await _services(tmp_path)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=1,
        creator_user_id=100,
    )

    lease, created = await leases.acquire(
        session_id=session.id,
        agenda_item_id="agenda-1",
        user_id=200,
        original_server_mute=True,
        ownership=MuteLeaseOwnership.OBSERVED,
        reason="TEST_OBSERVED",
    )
    duplicate, duplicate_created = await leases.acquire(
        session_id=session.id,
        agenda_item_id="agenda-other",
        user_id=200,
        original_server_mute=False,
        ownership=MuteLeaseOwnership.APPLYING,
        reason="TEST_DUPLICATE",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate == lease
    assert duplicate.original_server_mute is True
    released = await leases.release(lease.id, reason="TEST_RELEASE")
    assert released is not None
    assert released.ownership is MuteLeaseOwnership.RELEASED
    assert await leases.release(lease.id, reason="TEST_REPEAT") is None
    assert await leases.list_active(session_id=session.id) == ()
    lease_events = [
        event
        for event in await events.list(session.id)
        if event.event_type.startswith("MUTE_LEASE_")
    ]
    assert [event.event_type for event in lease_events] == [
        MeetingEventType.MUTE_LEASE_ACQUIRED.value,
        MeetingEventType.MUTE_LEASE_RELEASED.value,
    ]
    assert all(event.payload["lease_id"] == lease.id for event in lease_events)
    assert lease_events[0].payload["ownership"] == MuteLeaseOwnership.OBSERVED.value
    await database.close()


@pytest.mark.asyncio
async def test_enforcement_pause_and_intervention_own_only_their_timer_pause(
    tmp_path: Path,
) -> None:
    database, sessions, events, timers, enforcement, _ = await _services(tmp_path)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=2,
        creator_user_id=100,
    )
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(SpeakerTurnPlan(user_id=200, planned_seconds=60),),
        mute_enforcement_requested=True,
    )
    item = AgendaItemRecord(
        id="agenda-1",
        session_id=session.id,
        position=1,
        title="strict",
        status=AgendaStatus.ACTIVE,
        duration_minutes=None,
        requires_vote=False,
        speaking_policy=policy,
    )
    timer_type = "FLOW:SPEAKING_TURN:agenda-1:0"
    await timers.start(session_id=session.id, timer_type=timer_type, duration_seconds=60)
    await enforcement.ensure_started(session_id=session.id, item=item)

    assert await enforcement.pause(
        session_id=session.id,
        agenda_item_id=item.id,
        reason="PERMISSION_LOST",
        timer_type=timer_type,
        actor_user_id=None,
        source="DISCORD",
    )
    state = await enforcement.state(session.id, item.id)
    assert state is not None and state.paused and state.pause_timer_owned
    assert (await timers.active(session.id))[0].status is TimerStatus.PAUSED
    assert await enforcement.resume(
        session_id=session.id,
        agenda_item_id=item.id,
        actor_user_id=100,
        source="BUTTON",
        resume_timer=True,
    )
    assert (await timers.active(session.id))[0].status is TimerStatus.RUNNING

    assert await enforcement.start_intervention(
        session_id=session.id,
        agenda_item_id=item.id,
        timer_type=timer_type,
        actor_user_id=100,
        source="BUTTON",
    )
    state = await enforcement.state(session.id, item.id)
    assert state is not None and state.host_intervention_active
    assert state.intervention_timer_owned
    await timers.resume(session.id)
    assert (await timers.active(session.id))[0].status is TimerStatus.RUNNING
    await enforcement.maintain_owned_timer_pauses(session_id=session.id, state=state)
    assert (await timers.active(session.id))[0].status is TimerStatus.PAUSED
    assert await enforcement.end_intervention(
        session_id=session.id,
        agenda_item_id=item.id,
        actor_user_id=100,
        source="BUTTON",
        resume_timer=True,
    )
    assert (await timers.active(session.id))[0].status is TimerStatus.RUNNING
    assert await timers.pause_type(
        session_id=session.id,
        timer_type=timer_type,
        reason="HOST_INTERVENTION",
    )
    recovered_crash_window = await enforcement.state(session.id, item.id)
    assert recovered_crash_window is not None
    assert recovered_crash_window.host_intervention_active
    assert await enforcement.end_intervention(
        session_id=session.id,
        agenda_item_id=item.id,
        actor_user_id=100,
        source="RECOVERY",
        resume_timer=True,
    )
    event_types = [event.event_type for event in await events.list(session.id)]
    assert MeetingEventType.MUTE_ENFORCEMENT_PAUSED.value in event_types
    assert MeetingEventType.HOST_INTERVENTION_STARTED.value in event_types
    assert MeetingEventType.HOST_INTERVENTION_ENDED.value in event_types
    await database.close()


def _status(
    *,
    session: Any,
    item: AgendaItemRecord,
    current_user_id: int,
    enforcement: MuteEnforcementState,
) -> MeetingStatus:
    flow = SpeakingFlowState(
        agenda_item_id=item.id,
        mode=AgendaSpeakingMode.ORDERED,
        stage=SpeakingFlowStage.ACTIVE,
        turns=item.speaking_policy.turns,
        presenter_user_ids=(),
        identities=(
            SpeakingIdentity(100, "Host"),
            SpeakingIdentity(200, "Speaker"),
            SpeakingIdentity(300, "Moderator muted"),
        ),
        followup_mode=AgendaFollowupMode.NONE,
        prep_seconds=10,
        presentation_seconds=None,
        current_index=0,
        current_user_ids=(current_user_id,),
        next_user_ids=(),
        planned_seconds=60,
        extension_seconds=0,
        stage_started_at=session.created_at,
    )
    return MeetingStatus(
        session=replace(session, state=MeetingState.DISCUSSION_READY),
        agenda=(item,),
        speaking=SpeakingQueueState((), None),
        vote=None,
        floor=flow,
        timers=(
            TimerRecord(
                id="timer-1",
                session_id=session.id,
                timer_type=flow.timer_type or "",
                deadline_at=session.created_at,
                remaining_ms=None,
                status=TimerStatus.RUNNING,
            ),
        ),
        enforcement=enforcement,
    )


@pytest.mark.asyncio
async def test_discord_coordinator_reapplies_owned_mute_and_never_unmutes_moderator_mute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, sessions, events, _, enforcement, leases = await _services(tmp_path)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=3,
        creator_user_id=100,
    )
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(
            SpeakerTurnPlan(user_id=200, planned_seconds=60),
            SpeakerTurnPlan(user_id=100, planned_seconds=60),
        ),
        mute_enforcement_requested=True,
    )
    item = AgendaItemRecord(
        id="agenda-1",
        session_id=session.id,
        position=1,
        title="strict",
        status=AgendaStatus.ACTIVE,
        duration_minutes=None,
        requires_vote=False,
        speaking_policy=policy,
    )
    state = await enforcement.ensure_started(session_id=session.id, item=item)
    assert state is not None
    host = FakeMember(100)
    speaker = FakeMember(200)
    moderator_muted = FakeMember(300, muted=True)
    channel = SimpleNamespace(
        members=[host, speaker, moderator_muted],
        guild=SimpleNamespace(me=object()),
        permissions_for=lambda _member: SimpleNamespace(mute_members=True),
    )
    controller = SimpleNamespace(
        status_for_session=AsyncMock(
            return_value=_status(
                session=session,
                item=item,
                current_user_id=200,
                enforcement=state,
            )
        )
    )
    application = cast(
        MeetingApplication,
        SimpleNamespace(
            controller=controller,
            sessions=sessions,
            mute_leases=leases,
            mute_enforcement=enforcement,
        ),
    )
    coordinator = MeetingMuteCoordinator(
        bot=cast(discord.Client, SimpleNamespace(get_channel=lambda _channel_id: None)),
        application=application,
    )
    monkeypatch.setattr(coordinator, "_voice_channel", lambda _session: channel)

    first = await coordinator.sync_session(session)
    assert first.muted == 1
    assert host.voice.mute is True
    assert speaker.voice.mute is False
    assert moderator_muted.voice.mute is True
    assert (await leases.active_for_member(session_id=session.id, user_id=300)).ownership is (
        MuteLeaseOwnership.OBSERVED
    )

    host.voice.mute = False
    moderator_muted.voice.mute = False
    second = await coordinator.sync_session(session)
    assert second.reapplied == 2
    assert host.voice.mute is True
    assert moderator_muted.voice.mute is True

    intervention = replace(state, host_intervention_active=True)
    controller.status_for_session.return_value = _status(
        session=session,
        item=item,
        current_user_id=200,
        enforcement=intervention,
    )
    await coordinator.sync_session(session)
    assert host.voice.mute is False
    assert speaker.voice.mute is True

    controller.status_for_session.return_value = _status(
        session=session,
        item=item,
        current_user_id=200,
        enforcement=state,
    )
    await coordinator.sync_session(session)
    assert host.voice.mute is True
    assert speaker.voice.mute is False

    controller.status_for_session.return_value = _status(
        session=session,
        item=item,
        current_user_id=100,
        enforcement=state,
    )
    third = await coordinator.sync_session(session)
    assert third.unmuted == 1
    assert host.voice.mute is False
    assert speaker.voice.mute is True

    permission_paused = replace(state, paused=True, pause_reason="PERMISSION_LOST")
    controller.status_for_session.return_value = _status(
        session=session,
        item=item,
        current_user_id=100,
        enforcement=permission_paused,
    )
    permission_result = await coordinator.sync_session(session)
    assert permission_result.paused is True
    assert speaker.voice.mute is True

    paused = replace(state, paused=True, pause_reason="HOST_PAUSED")
    controller.status_for_session.return_value = _status(
        session=session,
        item=item,
        current_user_id=100,
        enforcement=paused,
    )
    await coordinator.sync_session(session)
    assert speaker.voice.mute is False
    assert moderator_muted.voice.mute is True
    assert moderator_muted.edits == [True]
    assert await leases.list_active(session_id=session.id) == ()
    event_types = [event.event_type for event in await events.list(session.id)]
    assert MeetingEventType.MUTE_REAPPLIED.value in event_types
    assert event_types.count(MeetingEventType.MUTE_LEASE_RELEASED.value) >= 3
    await database.close()


@pytest.mark.asyncio
async def test_applying_crash_window_is_paused_and_never_guessed_as_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, sessions, _, _, enforcement, leases = await _services(tmp_path)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=4,
        creator_user_id=100,
    )
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(SpeakerTurnPlan(user_id=100, planned_seconds=60),),
        mute_enforcement_requested=True,
    )
    item = AgendaItemRecord(
        id="agenda-1",
        session_id=session.id,
        position=1,
        title="strict",
        status=AgendaStatus.ACTIVE,
        duration_minutes=None,
        requires_vote=False,
        speaking_policy=policy,
    )
    state = await enforcement.ensure_started(session_id=session.id, item=item)
    assert state is not None
    lease, _ = await leases.acquire(
        session_id=session.id,
        agenda_item_id=item.id,
        user_id=100,
        original_server_mute=False,
        ownership=MuteLeaseOwnership.APPLYING,
        reason="TEST_CRASH_WINDOW",
    )
    host = FakeMember(100, muted=True)
    channel = SimpleNamespace(
        members=[host],
        guild=SimpleNamespace(me=object()),
        permissions_for=lambda _member: SimpleNamespace(mute_members=True),
    )
    status = _status(
        session=session,
        item=item,
        current_user_id=100,
        enforcement=state,
    )
    application = cast(
        MeetingApplication,
        SimpleNamespace(
            controller=SimpleNamespace(status_for_session=AsyncMock(return_value=status)),
            sessions=sessions,
            mute_leases=leases,
            mute_enforcement=enforcement,
        ),
    )
    coordinator = MeetingMuteCoordinator(
        bot=cast(discord.Client, SimpleNamespace(get_channel=lambda _channel_id: None)),
        application=application,
    )
    monkeypatch.setattr(coordinator, "_voice_channel", lambda _session: channel)

    result = await coordinator.sync_session(session)

    assert result.paused is True
    assert host.edits == []
    recovered = await leases.active_for_member(session_id=session.id, user_id=100)
    assert recovered is not None
    assert recovered.id == lease.id
    assert recovered.ownership is MuteLeaseOwnership.UNCERTAIN
    projected = await enforcement.state(session.id, item.id)
    assert projected is not None and projected.paused
    assert projected.pause_reason == "SIDE_EFFECT_UNCERTAIN"
    await database.close()


@pytest.mark.asyncio
async def test_startup_reconciliation_releases_terminal_owned_mute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, sessions, _, _, enforcement, leases = await _services(tmp_path)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=5,
        creator_user_id=100,
    )
    lease, _ = await leases.acquire(
        session_id=session.id,
        agenda_item_id="agenda-1",
        user_id=200,
        original_server_mute=False,
        ownership=MuteLeaseOwnership.APPLYING,
        reason="TEST_APPLY",
    )
    await leases.set_ownership(
        lease_id=lease.id,
        ownership=MuteLeaseOwnership.OWNED,
        reason="TEST_OWNED",
    )
    member = FakeMember(200, muted=True)
    channel = SimpleNamespace(
        members=[member],
        guild=SimpleNamespace(me=object()),
        permissions_for=lambda _member: SimpleNamespace(mute_members=True),
    )
    terminal_status = MeetingStatus(
        session=replace(session, state=MeetingState.STOPPED),
        agenda=(),
        speaking=SpeakingQueueState((), None),
        vote=None,
    )
    application = cast(
        MeetingApplication,
        SimpleNamespace(
            controller=SimpleNamespace(status_for_session=AsyncMock(return_value=terminal_status)),
            sessions=sessions,
            mute_leases=leases,
            mute_enforcement=enforcement,
        ),
    )
    coordinator = MeetingMuteCoordinator(
        bot=cast(discord.Client, SimpleNamespace(get_channel=lambda _channel_id: None)),
        application=application,
    )
    monkeypatch.setattr(coordinator, "_voice_channel", lambda _session: None)
    incomplete = await coordinator.release_session(session, reason="TEST_DISCONNECTED")
    assert incomplete.paused is True
    assert await leases.active_for_member(session_id=session.id, user_id=200) is not None
    monkeypatch.setattr(coordinator, "_voice_channel", lambda _session: channel)

    await coordinator.reconcile_all()

    assert member.edits == [False]
    assert await leases.list_active(session_id=session.id) == ()
    await database.close()
