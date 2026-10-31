from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxu.domain.enums import MeetingEventType, SpeakingFlowStage
from lingxu.domain.models import AgendaItemRecord, MeetingSessionRecord, TimerRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.meeting.components import AgendaService, TimerService
from lingxu.meeting.speaking_flow import (
    ControlledSpeakingFlowService,
    SpeakingFlowNoticeKind,
    SpeakingFlowState,
    SpeakingFlowUpdate,
)
from lingxu.storage.component_repositories import (
    AgendaRepository,
    ParticipantRepository,
    TimerRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@dataclass
class SpeakingTimerStack:
    database: Database
    events: MeetingEventStore
    timer_repository: TimerRepository
    timers: TimerService
    flow: ControlledSpeakingFlowService
    session: MeetingSessionRecord
    item: AgendaItemRecord


async def _stack(tmp_path: Path, *, two_turns: bool = False) -> SpeakingTimerStack:
    database = Database.open(_database_url(tmp_path / "speaking-timer-recovery.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session = await MeetingSessionService(
        sessions,
        SessionFiles(tmp_path / "sessions"),
    ).create(guild_id=1, creator_user_id=100)
    events = MeetingEventStore(database.sessions)
    participants = ParticipantRepository(database.sessions)
    await participants.join(
        session_id=session.id,
        user_id=200,
        display_name="First speaker",
    )
    turns = [SpeakerTurnPlan(user_id=200, planned_seconds=30)]
    if two_turns:
        await participants.join(
            session_id=session.id,
            user_id=300,
            display_name="Second speaker",
        )
        turns.append(SpeakerTurnPlan(user_id=300, planned_seconds=45))
    item = await AgendaService(AgendaRepository(database.sessions), events).add(
        session_id=session.id,
        title="Durable speaking timer",
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=tuple(turns),
            prep_seconds=10,
        ),
        actor_user_id=100,
        source="TEST",
    )
    timer_repository = TimerRepository(database.sessions)
    timers = TimerService(timers=timer_repository, events=events)
    flow = ControlledSpeakingFlowService(
        participants=participants,
        events=events,
        timers=timers,
    )
    return SpeakingTimerStack(
        database=database,
        events=events,
        timer_repository=timer_repository,
        timers=timers,
        flow=flow,
        session=session,
        item=item,
    )


async def _expiry_for(stack: SpeakingTimerStack, timer: TimerRecord):
    notices = await stack.timers.tick(now=timer.deadline_at + timedelta(seconds=1))
    return next(notice for notice in notices if notice.timer_id == timer.id)


async def _active_timer(stack: SpeakingTimerStack, timer_type: str) -> TimerRecord:
    return next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session.id)
        if timer.timer_type == timer_type
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reconcile_restores_initial_preparation_timer_without_repeating_stage_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path)
    original_start = stack.timers.start_after_handoff_barrier
    failed = False

    async def fail_first_preparation_start(
        *,
        session_id: str,
        timer_type: str,
        duration_seconds: float,
        now: datetime | None = None,
    ) -> TimerRecord | None:
        nonlocal failed
        if timer_type.startswith("FLOW:SPEAKING_PREP:") and not failed:
            failed = True
            raise RuntimeError("synthetic initial preparation timer crash")
        return await original_start(
            session_id=session_id,
            timer_type=timer_type,
            duration_seconds=duration_seconds,
            now=now,
        )

    monkeypatch.setattr(
        stack.timers,
        "start_after_handoff_barrier",
        fail_first_preparation_start,
    )
    try:
        with pytest.raises(RuntimeError, match="synthetic initial preparation timer crash"):
            await stack.flow.activate(session=stack.session, item=stack.item)
        projected = await stack.flow.state(stack.session.id, stack.item.id)
        assert projected is not None and projected.stage is SpeakingFlowStage.PREPARING
        assert await stack.timer_repository.list_active(stack.session.id) == ()
        stage_events_before = tuple(
            event
            for event in await stack.events.list(stack.session.id)
            if event.event_type
            in {
                MeetingEventType.SPEAKING_FLOW_STARTED.value,
                MeetingEventType.SPEAKING_PREPARATION_STARTED.value,
            }
        )

        monkeypatch.setattr(stack.timers, "start_after_handoff_barrier", original_start)
        reconciled = await stack.flow.reconcile_current_timer(stack.session.id, stack.item.id)
        assert reconciled.state == projected
        assert not reconciled.changed
        repaired = await _active_timer(stack, projected.timer_type or "")
        reconciled_again = await stack.flow.reconcile_current_timer(
            stack.session.id,
            stack.item.id,
        )
        assert reconciled_again.state == projected
        assert not reconciled_again.changed
        assert (await _active_timer(stack, projected.timer_type or "")).id == repaired.id
        stage_events_after = tuple(
            event
            for event in await stack.events.list(stack.session.id)
            if event.event_type
            in {
                MeetingEventType.SPEAKING_FLOW_STARTED.value,
                MeetingEventType.SPEAKING_PREPARATION_STARTED.value,
            }
        )
        assert stage_events_after == stage_events_before
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_preparation_expiry_replay_repairs_missing_turn_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path)
    try:
        preparing = await stack.flow.activate(session=stack.session, item=stack.item)
        assert preparing.state is not None
        prep_timer = await _active_timer(stack, preparing.state.timer_type or "")
        prep_expiry = await _expiry_for(stack, prep_timer)
        original_start = stack.timers.start_after_handoff_barrier
        failed = False

        async def fail_first_turn_start(
            *,
            session_id: str,
            timer_type: str,
            duration_seconds: float,
            now: datetime | None = None,
        ) -> TimerRecord | None:
            nonlocal failed
            if timer_type.startswith("FLOW:SPEAKING_TURN:") and not failed:
                failed = True
                raise RuntimeError("synthetic active turn timer crash")
            return await original_start(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=duration_seconds,
                now=now,
            )

        monkeypatch.setattr(
            stack.timers,
            "start_after_handoff_barrier",
            fail_first_turn_start,
        )
        with pytest.raises(RuntimeError, match="synthetic active turn timer crash"):
            await stack.flow.handle_timer_notice(prep_expiry)
        projected = await stack.flow.state(stack.session.id, stack.item.id)
        assert projected is not None and projected.stage is SpeakingFlowStage.ACTIVE
        assert await stack.timer_repository.list_active(stack.session.id) == ()

        monkeypatch.setattr(stack.timers, "start_after_handoff_barrier", original_start)
        replayed = await stack.flow.handle_timer_notice(prep_expiry)
        assert replayed.state == projected
        assert not replayed.changed
        repaired = await _active_timer(stack, projected.timer_type or "")
        reconciled = await stack.flow.reconcile_current_timer(stack.session.id, stack.item.id)
        assert reconciled.state == projected
        assert not reconciled.changed
        assert (await _active_timer(stack, projected.timer_type or "")).id == repaired.id
        turn_events = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_TURN_STARTED.value,),
        )
        assert len(turn_events) == 1
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_expiry_replay_repairs_missing_next_preparation_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, two_turns=True)
    try:
        preparing = await stack.flow.activate(session=stack.session, item=stack.item)
        assert preparing.state is not None
        prep_timer = await _active_timer(stack, preparing.state.timer_type or "")
        prep_expiry = await _expiry_for(stack, prep_timer)
        active = await stack.flow.handle_timer_notice(prep_expiry)
        assert active.state is not None and active.state.stage is SpeakingFlowStage.ACTIVE
        await stack.timers.complete_handoff(prep_expiry)
        turn_timer = await _active_timer(stack, active.state.timer_type or "")
        turn_expiry = await _expiry_for(stack, turn_timer)
        original_start = stack.timers.start_after_handoff_barrier
        failed = False

        async def fail_next_preparation_start(
            *,
            session_id: str,
            timer_type: str,
            duration_seconds: float,
            now: datetime | None = None,
        ) -> TimerRecord | None:
            nonlocal failed
            if timer_type.endswith(":1") and timer_type.startswith("FLOW:SPEAKING_PREP:"):
                if not failed:
                    failed = True
                    raise RuntimeError("synthetic next preparation timer crash")
            return await original_start(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=duration_seconds,
                now=now,
            )

        monkeypatch.setattr(
            stack.timers,
            "start_after_handoff_barrier",
            fail_next_preparation_start,
        )
        with pytest.raises(RuntimeError, match="synthetic next preparation timer crash"):
            await stack.flow.handle_timer_notice(turn_expiry)
        projected = await stack.flow.state(stack.session.id, stack.item.id)
        assert projected is not None and projected.stage is SpeakingFlowStage.PREPARING
        assert projected.current_index == 1
        assert await stack.timer_repository.list_active(stack.session.id) == ()

        monkeypatch.setattr(stack.timers, "start_after_handoff_barrier", original_start)
        replayed = await stack.flow.handle_timer_notice(turn_expiry)
        assert replayed.state == projected
        assert not replayed.changed
        repaired = await _active_timer(stack, projected.timer_type or "")
        reconciled = await stack.flow.reconcile_current_timer(stack.session.id, stack.item.id)
        assert reconciled.state == projected
        assert not reconciled.changed
        assert (await _active_timer(stack, projected.timer_type or "")).id == repaired.id
        preparation_events = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_PREPARATION_STARTED.value,),
        )
        assert len(preparation_events) == 2
        finished_events = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_TURN_FINISHED.value,),
        )
        assert len(finished_events) == 1
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reconcile_continues_flow_started_before_initial_preparation_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path)
    original_prepare = stack.flow._prepare_next_locked
    failed = False

    async def fail_before_initial_preparation(
        session_id: str,
        state: SpeakingFlowState,
    ) -> SpeakingFlowUpdate:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("synthetic crash before initial preparation event")
        return await original_prepare(session_id, state)

    monkeypatch.setattr(stack.flow, "_prepare_next_locked", fail_before_initial_preparation)
    try:
        with pytest.raises(RuntimeError, match="synthetic crash before initial preparation event"):
            await stack.flow.activate(session=stack.session, item=stack.item)
        starting = await stack.flow.state(stack.session.id, stack.item.id)
        assert starting is not None and starting.stage is SpeakingFlowStage.STARTING
        assert await stack.timer_repository.list_active(stack.session.id) == ()

        monkeypatch.setattr(stack.flow, "_prepare_next_locked", original_prepare)
        reconciled = await stack.flow.reconcile_current_timer(stack.session.id, stack.item.id)
        assert reconciled.changed
        assert reconciled.state is not None
        assert reconciled.state.stage is SpeakingFlowStage.PREPARING
        assert [notice.kind for notice in reconciled.notices] == [SpeakingFlowNoticeKind.PREPARING]
        repaired = await _active_timer(stack, reconciled.state.timer_type or "")

        reconciled_again = await stack.flow.reconcile_current_timer(
            stack.session.id,
            stack.item.id,
        )
        assert reconciled_again.state == reconciled.state
        assert not reconciled_again.changed
        assert (await _active_timer(stack, reconciled.state.timer_type or "")).id == repaired.id
        flow_started = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_FLOW_STARTED.value,),
        )
        preparations = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_PREPARATION_STARTED.value,),
        )
        assert len(flow_started) == 1
        assert len(preparations) == 1
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_expiry_replay_continues_finished_turn_before_next_preparation_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, two_turns=True)
    try:
        preparing = await stack.flow.activate(session=stack.session, item=stack.item)
        assert preparing.state is not None
        prep_timer = await _active_timer(stack, preparing.state.timer_type or "")
        prep_expiry = await _expiry_for(stack, prep_timer)
        active = await stack.flow.handle_timer_notice(prep_expiry)
        assert active.state is not None and active.state.stage is SpeakingFlowStage.ACTIVE
        await stack.timers.complete_handoff(prep_expiry)
        turn_timer = await _active_timer(stack, active.state.timer_type or "")
        turn_expiry = await _expiry_for(stack, turn_timer)
        original_prepare = stack.flow._prepare_next_locked
        failed = False

        async def fail_after_turn_finished(
            session_id: str,
            state: SpeakingFlowState,
        ) -> SpeakingFlowUpdate:
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("synthetic crash before next preparation event")
            return await original_prepare(session_id, state)

        monkeypatch.setattr(stack.flow, "_prepare_next_locked", fail_after_turn_finished)
        with pytest.raises(RuntimeError, match="synthetic crash before next preparation event"):
            await stack.flow.handle_timer_notice(turn_expiry)
        starting = await stack.flow.state(stack.session.id, stack.item.id)
        assert starting is not None and starting.stage is SpeakingFlowStage.STARTING
        assert starting.current_index == 0
        assert await stack.timer_repository.list_active(stack.session.id) == ()

        monkeypatch.setattr(stack.flow, "_prepare_next_locked", original_prepare)
        replayed = await stack.flow.handle_timer_notice(turn_expiry)
        assert replayed.changed
        assert replayed.state is not None
        assert replayed.state.stage is SpeakingFlowStage.PREPARING
        assert replayed.state.current_index == 1
        assert [notice.kind for notice in replayed.notices] == [SpeakingFlowNoticeKind.PREPARING]
        await _active_timer(stack, replayed.state.timer_type or "")
        finished_events = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_TURN_FINISHED.value,),
        )
        preparation_events = await stack.events.list_types(
            stack.session.id,
            (MeetingEventType.SPEAKING_PREPARATION_STARTED.value,),
        )
        assert len(finished_events) == 1
        assert len(preparation_events) == 2
    finally:
        await stack.database.close()
