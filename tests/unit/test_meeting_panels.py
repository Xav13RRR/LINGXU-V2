from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import discord
import pytest

from lingxu.bot.meeting_panels import MeetingPanelCoordinator, render_meeting_panel
from lingxu.bot.meeting_ui import HostControlView, HostPanelRuntime
from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    AgendaStatus,
    MeetingState,
    SpeakingFlowStage,
    TimerStatus,
)
from lingxu.domain.models import AgendaItemRecord, MeetingSessionRecord, TimerRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.meeting.components import SpeakingQueueState
from lingxu.meeting.controller import MeetingController, MeetingStatus
from lingxu.meeting.factory import MeetingApplication
from lingxu.meeting.mute_enforcement import MuteEnforcementState
from lingxu.meeting.speaking_flow import SpeakingFlowState, SpeakingIdentity


def _status(now: datetime) -> MeetingStatus:
    turns = (
        SpeakerTurnPlan(user_id=100, planned_seconds=60),
        SpeakerTurnPlan(user_id=200, planned_seconds=90),
    )
    policy = AgendaSpeakingPolicy.ordered_explicit(turns=turns)
    session = MeetingSessionRecord(
        id="session-panel",
        guild_id=1,
        creator_user_id=100,
        state=MeetingState.DISCUSSION_READY,
        resume_state=None,
        next_sequence=1,
        manifest_path=Path("session-panel/manifest.json"),
        created_at=now,
        updated_at=now,
    )
    agenda = (
        AgendaItemRecord(
            id="agenda-1",
            session_id=session.id,
            position=1,
            title="主持人也参与的汇报",
            status=AgendaStatus.ACTIVE,
            duration_minutes=None,
            requires_vote=False,
            speaking_policy=policy,
        ),
        AgendaItemRecord(
            id="agenda-2",
            session_id=session.id,
            position=2,
            title="下一项",
            status=AgendaStatus.PENDING,
            duration_minutes=5,
            requires_vote=False,
        ),
    )
    floor = SpeakingFlowState(
        agenda_item_id="agenda-1",
        mode=AgendaSpeakingMode.ORDERED,
        stage=SpeakingFlowStage.ACTIVE,
        turns=turns,
        presenter_user_ids=(),
        identities=(
            SpeakingIdentity(user_id=100, display_name="Host"),
            SpeakingIdentity(user_id=200, display_name="Alice"),
        ),
        followup_mode=AgendaFollowupMode.NONE,
        prep_seconds=10,
        presentation_seconds=None,
        current_index=0,
        current_user_ids=(100,),
        next_user_ids=(200,),
        planned_seconds=60,
        extension_seconds=0,
        stage_started_at=now,
    )
    timer = TimerRecord(
        id="timer-1",
        session_id=session.id,
        timer_type="FLOW:SPEAKING_TURN:agenda-1:0",
        deadline_at=now + timedelta(seconds=42),
        remaining_ms=None,
        status=TimerStatus.RUNNING,
    )
    return MeetingStatus(
        session=session,
        agenda=agenda,
        speaking=SpeakingQueueState(queued_user_ids=(), current_user_id=None),
        vote=None,
        floor=floor,
        timers=(timer,),
    )


def test_public_panel_is_read_only_and_shows_whole_meeting_progress() -> None:
    now = datetime(2026, 8, 26, tzinfo=UTC)
    content = render_meeting_panel(_status(now), private=False, now=now)

    assert "议程进度: 0/2" in content
    assert "当前议题: 主持人也参与的汇报" in content
    assert "当前: Host" in content
    assert "下一位: Alice" in content
    assert f"剩余: <t:{int((now + timedelta(seconds=42)).timestamp())}:R>" in content
    assert "完整顺序: Host → Alice" in content
    assert "1. 主持人也参与的汇报" in content
    assert "2. 下一项" in content
    assert "下方按钮" not in content


def test_private_host_panel_has_contextual_controls() -> None:
    now = datetime(2026, 8, 26, tzinfo=UTC)
    status = _status(now)
    private_content = render_meeting_panel(status, private=True, now=now)
    controller = cast(MeetingController, object())
    panels = cast(HostPanelRuntime, object())
    view = HostControlView(controller, panels, status=status)
    disabled = {cast(Any, item).custom_id: cast(Any, item).disabled for item in view.children}

    assert "仅在本私信中供主持人使用" in private_content
    assert disabled["lingxu:host:resume:v1"] is True
    assert disabled["lingxu:host:next:v1"] is False
    assert disabled["lingxu:host:extend:v1"] is False
    assert disabled["lingxu:host:end-turn:v1"] is False
    assert disabled["lingxu:host:end-followup:v1"] is True
    assert disabled["lingxu:host:abandon:v1"] is False

    terminal = replace(status, session=replace(status.session, state=MeetingState.STOPPED))
    terminal_view = HostControlView(controller, panels, status=terminal)
    assert all(cast(Any, item).disabled for item in terminal_view.children)


def test_private_panel_exposes_strict_controls_without_giving_host_floor_bypass() -> None:
    now = datetime(2026, 8, 26, tzinfo=UTC)
    status = _status(now)
    strict_policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=status.agenda[0].speaking_policy.turns,
        mute_enforcement_requested=True,
    )
    status = replace(
        status,
        agenda=(replace(status.agenda[0], speaking_policy=strict_policy), *status.agenda[1:]),
        enforcement=MuteEnforcementState(agenda_item_id="agenda-1"),
    )
    view = HostControlView(
        cast(MeetingController, object()),
        cast(HostPanelRuntime, object()),
        status=status,
    )
    disabled = {cast(Any, item).custom_id: cast(Any, item).disabled for item in view.children}

    assert "静音执行: 强制静音运行中" in render_meeting_panel(
        status,
        private=True,
        now=now,
    )
    assert disabled["lingxu:host:intervene-start:v1"] is False
    assert disabled["lingxu:host:intervene-end:v1"] is True
    assert disabled["lingxu:host:enforcement-pause:v1"] is False
    assert disabled["lingxu:host:enforcement-resume:v1"] is True
    assert disabled["lingxu:host:exception-grant:v1"] is False


@pytest.mark.asyncio
async def test_guild_refresh_uses_latest_session_after_stop() -> None:
    session = _status(datetime(2026, 8, 26, tzinfo=UTC)).session
    sessions = SimpleNamespace(
        active_for_guild=AsyncMock(return_value=None),
        latest_for_guild=AsyncMock(return_value=session),
    )
    application = cast(MeetingApplication, SimpleNamespace(sessions=sessions))
    coordinator = MeetingPanelCoordinator(
        bot=cast(discord.Client, object()),
        application=application,
    )
    refresh = AsyncMock()
    coordinator.refresh = refresh

    await coordinator.refresh_guild(session.guild_id)

    sessions.latest_for_guild.assert_awaited_once_with(session.guild_id)
    refresh.assert_awaited_once_with(session.id)
