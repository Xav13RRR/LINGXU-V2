import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    AgendaStatus,
    ConsentKind,
    HostParticipationMode,
    JobStatus,
    MeetingEventType,
    MeetingState,
    TimerStatus,
    VoteStatus,
)
from lingxu.domain.errors import MeetingPermissionError, ParticipantNotPresentError
from lingxu.domain.meeting_state import forward_state
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.meeting.components import (
    AgendaService,
    SpeakingQueueService,
    TimerService,
    VotingService,
)
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.runtime import MeetingRuntime, RuntimeResult
from lingxu.meeting.speaking_flow import ControlledSpeakingFlowService
from lingxu.storage.component_repositories import (
    AgendaRepository,
    ConsentChangeRepository,
    GuildConsentRepository,
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
    ProcessingJobRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import MeetingVoteEntryRow
from lingxu.worker.speaking_facts import project_authoritative_speaking_facts


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@dataclass
class ComponentStack:
    database: Database
    sessions: MeetingSessionRepository
    session_service: MeetingSessionService
    events: MeetingEventStore
    participants: ParticipantRepository
    agenda: AgendaService
    speaking: SpeakingQueueService
    timer_repository: TimerRepository
    voting: VotingService
    timers: TimerService
    speaking_flow: ControlledSpeakingFlowService
    runtime: MeetingRuntime
    controller: MeetingController
    jobs: ProcessingJobRepository
    session_id: str


async def _stack(tmp_path: Path, guild_id: int = 1) -> ComponentStack:
    database = Database.open(_database_url(tmp_path / f"phase7-{guild_id}.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session_service = MeetingSessionService(
        sessions,
        SessionFiles(tmp_path / f"sessions-{guild_id}"),
    )
    session = await session_service.create(guild_id=guild_id, creator_user_id=100)
    events = MeetingEventStore(database.sessions)
    participants = ParticipantRepository(database.sessions)
    agenda = AgendaService(AgendaRepository(database.sessions), events)
    speaking = SpeakingQueueService(participants=participants, events=events)
    timer_repository = TimerRepository(database.sessions)
    voting = VotingService(
        votes=VoteRepository(database.sessions),
        timers=timer_repository,
        participants=participants,
        events=events,
    )
    timers = TimerService(
        timers=timer_repository,
        events=events,
        vote_closer=voting,
    )
    speaking_flow = ControlledSpeakingFlowService(
        participants=participants,
        events=events,
        timers=timers,
    )
    runtime = MeetingRuntime(
        sessions=sessions,
        events=events,
        snapshots=MeetingSnapshotStore(database.sessions),
        snapshot_interval=50,
    )
    jobs = ProcessingJobRepository(database.sessions)
    controller = MeetingController(
        sessions=sessions,
        events=events,
        session_service=session_service,
        runtime=runtime,
        participants=participants,
        guild_consents=GuildConsentRepository(database.sessions),
        consent_changes=ConsentChangeRepository(database.sessions),
        agenda=agenda,
        speaking=speaking,
        timers=timers,
        voting=voting,
        speaking_flow=speaking_flow,
        jobs=jobs,
    )
    return ComponentStack(
        database=database,
        sessions=sessions,
        session_service=session_service,
        events=events,
        participants=participants,
        agenda=agenda,
        speaking=speaking,
        timer_repository=timer_repository,
        voting=voting,
        timers=timers,
        speaking_flow=speaking_flow,
        runtime=runtime,
        controller=controller,
        jobs=jobs,
        session_id=session.id,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_host_retry_authorizes_a_new_full_transcript_attempt_generation(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    job = await stack.jobs.create(
        session_id=stack.session_id,
        job_type="POST_MEETING",
        idempotency_key="manual-content-retry",
    )
    claimed = await stack.jobs.claim(
        worker_id="failed-worker",
        lease_seconds=30,
        job_types=("POST_MEETING",),
    )
    assert claimed is not None and claimed.id == job.id
    await stack.jobs.checkpoint(
        job_id=job.id,
        worker_id="failed-worker",
        stage="GENERATE_MINUTES",
        checkpoint={
            "schema_version": 4,
            "completed_stages": [],
            "full_transcript_minutes_attempt_generation": 2,
        },
        lease_seconds=30,
    )
    await stack.jobs.fail(
        job_id=job.id,
        worker_id="failed-worker",
        error_type="FullTranscriptPreviewError[preview_finish_reason_invalid]@GENERATE_MINUTES",
    )

    assert await stack.controller.retry(guild_id=1, actor_user_id=100)

    retried = await stack.jobs.get(job.id)
    assert retried is not None and retried.status is JobStatus.PENDING
    assert retried.checkpoint["full_transcript_minutes_attempt_generation"] == 3
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completed_manual_retry_converges_failed_session_to_published(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    session = await stack.sessions.get(stack.session_id)
    while session.state is not MeetingState.POST_PROCESSING:
        target = forward_state(session.state)
        assert target is not None
        session = await stack.sessions.transition(
            session_id=session.id,
            target=target,
            actor_user_id=None,
            source="SYSTEM",
            event_type=MeetingEventType.STATE_ADVANCED.value,
        )

    job = await stack.jobs.create(
        session_id=stack.session_id,
        job_type="POST_MEETING",
        idempotency_key="failed-then-completed-retry",
    )
    claimed = await stack.jobs.claim(
        worker_id="failed-worker",
        lease_seconds=30,
        job_types=("POST_MEETING",),
    )
    assert claimed is not None and claimed.id == job.id
    await stack.jobs.fail(
        job_id=job.id,
        worker_id="failed-worker",
        error_type="FullTranscriptPreviewError[final_minutes_speaker_identity_invalid]",
    )
    failed = await stack.runtime.post_processing_failed(stack.session_id)
    assert failed.session.state is MeetingState.FAILED_POST_PROCESSING
    assert await stack.controller.monitor_post_processing_jobs() == ()

    assert await stack.controller.retry(guild_id=1, actor_user_id=100)
    retried = await stack.jobs.claim(
        worker_id="successful-worker",
        lease_seconds=30,
        job_types=("POST_MEETING",),
    )
    assert retried is not None and retried.attempt_count == 2
    await stack.jobs.complete(job_id=retried.id, worker_id="successful-worker")

    monitored = await stack.controller.monitor_post_processing_jobs()

    assert monitored[-1].session.state is MeetingState.PUBLISHED
    events = await stack.events.list(stack.session_id)
    transition = next(
        event for event in reversed(events) if event.state_before is not event.state_after
    )
    assert transition.state_before is MeetingState.FAILED_POST_PROCESSING
    assert transition.state_after is MeetingState.PUBLISHED
    assert await stack.controller.monitor_post_processing_jobs() == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agenda_queue_dedupes_and_removes_offline_user_after_restart(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    for user_id in (100, 200, 300):
        await stack.participants.join(session_id=stack.session_id, user_id=user_id)

    first = await stack.agenda.add(
        session_id=stack.session_id,
        title="Budget",
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    second = await stack.agenda.add(
        session_id=stack.session_id,
        title="Roadmap",
        requires_vote=True,
        vote_content="Approve the roadmap?",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert (first.position, second.position) == (1, 2)
    assert (await stack.agenda.activate_next(session_id=stack.session_id)).id == first.id
    await stack.agenda.complete_active(session_id=stack.session_id)
    assert (await stack.agenda.activate_next(session_id=stack.session_id)).id == second.id

    assert await stack.speaking.request(
        session_id=stack.session_id,
        user_id=200,
        source="SLASH_COMMAND",
    )
    assert not await stack.speaking.request(
        session_id=stack.session_id,
        user_id=200,
        source="BUTTON",
    )
    assert await stack.speaking.request(
        session_id=stack.session_id,
        user_id=300,
        source="SLASH_COMMAND",
    )
    await stack.participants.leave(session_id=stack.session_id, user_id=200)

    recovered_queue = SpeakingQueueService(
        participants=ParticipantRepository(stack.database.sessions),
        events=MeetingEventStore(stack.database.sessions),
    )
    assert (
        await recovered_queue.next(
            session_id=stack.session_id,
            actor_user_id=100,
            source="SLASH_COMMAND",
        )
        == 300
    )
    state = await recovered_queue.state(stack.session_id)
    assert state.current_user_id == 300
    assert state.queued_user_ids == ()

    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agenda_speaking_policy_persists_and_is_copied_to_append_only_event(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=101)
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(
            SpeakerTurnPlan(user_id=100, planned_seconds=90),
            SpeakerTurnPlan(user_id=200, planned_seconds=120),
        ),
        followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
        mute_enforcement_requested=True,
    )

    with pytest.raises(MeetingPermissionError):
        await stack.controller.agenda_add(
            guild_id=101,
            actor_user_id=200,
            title="越权配置",
            duration_minutes=4,
            speaking_policy=policy,
        )
    item = await stack.controller.agenda_add(
        guild_id=101,
        actor_user_id=100,
        title="轮流汇报",
        duration_minutes=4,
        speaking_policy=policy,
    )
    recovered = await AgendaRepository(stack.database.sessions).list(stack.session_id)
    event = (await stack.events.list(stack.session_id))[-1]

    assert item.speaking_policy == policy
    assert recovered[0].speaking_policy == policy
    assert event.event_type == "AGENDA_ITEM_ADDED"
    assert event.payload["speaking_policy"] == policy.to_payload()

    legacy = await stack.agenda.add(
        session_id=stack.session_id,
        title="自由讨论",
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert legacy.speaking_policy.mode is AgendaSpeakingMode.FREE
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_presentation_requires_an_overall_agenda_duration(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=102)
    policy = AgendaSpeakingPolicy.presentation(presenter_user_ids=(200,))

    with pytest.raises(ValueError, match="duration"):
        await stack.agenda.add(
            session_id=stack.session_id,
            title="产品演讲",
            duration_minutes=None,
            speaking_policy=policy,
            actor_user_id=100,
            source="SLASH_COMMAND",
        )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_alphabetical_flow_snapshots_discord_names_and_can_exclude_host(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=105)
    for user_id, name in (
        (100, "Zulu"),
        (200, "\uff21lpha"),
        (300, "alpha"),
        (400, "0-missing"),
    ):
        await stack.participants.join(
            session_id=stack.session_id,
            user_id=user_id,
            display_name=name,
            is_host=user_id == 100,
        )
    await stack.participants.leave(session_id=stack.session_id, user_id=400)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="按姓名顺序",
        speaking_policy=AgendaSpeakingPolicy.ordered_alphabetical(
            default_turn_seconds=45,
            host_participation=HostParticipationMode.MODERATOR_ONLY,
            prep_seconds=0,
        ),
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    update = await stack.speaking_flow.activate(
        session=await stack.sessions.get(stack.session_id),
        item=item,
    )

    assert update.state is not None
    assert [turn.user_id for turn in update.state.turns] == [400, 200, 300]
    assert {turn.planned_seconds for turn in update.state.turns} == {45}
    assert update.state.current_user_ids == (200,)
    assert update.state.stage.value == "ACTIVE"
    assert [notice.kind.value for notice in update.notices] == ["FINISHED", "STARTED"]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ordered_flow_recovers_timers_extends_skips_absent_and_enters_followup(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=103)
    all_consents = frozenset(ConsentKind)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=all_consents,
        display_name="Host",
        is_host=True,
    )
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=200,
        inherited_consents=all_consents,
        display_name="Alice",
    )
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=300,
        inherited_consents=all_consents,
        display_name="Bob",
    )
    await stack.participants.leave(session_id=stack.session_id, user_id=300)
    item = await stack.controller.agenda_add(
        guild_id=103,
        actor_user_id=100,
        title="轮流汇报",
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=(
                SpeakerTurnPlan(user_id=200, planned_seconds=60),
                SpeakerTurnPlan(user_id=300, planned_seconds=90),
            ),
            followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
        ),
    )
    actions: list[str] = []
    notices: list[object] = []

    async def record_action(_session: object, action: str) -> None:
        actions.append(action)

    async def record_notice(_session: object, notice: object) -> None:
        notices.append(notice)

    stack.controller.set_runtime_action_handler(record_action)
    stack.controller.set_speaking_flow_handler(record_notice)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.controller.start(guild_id=103, actor_user_id=100)
    opening = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(opening[-1])

    preparing = await stack.speaking_flow.state(stack.session_id, item.id)
    assert preparing is not None
    assert preparing.stage.value == "PREPARING"
    assert preparing.current_user_ids == (200,)
    assert preparing.next_user_ids == ()
    active_before = await stack.timer_repository.list_active(stack.session_id)
    await stack.controller.recover_automatic_flows(session_ids={stack.session_id})
    active_after = await stack.timer_repository.list_active(stack.session_id)
    assert [timer.id for timer in active_after] == [timer.id for timer in active_before]

    prep_notices = await stack.timers.tick(now=baseline + timedelta(seconds=30))
    prep_expired = next(
        notice
        for notice in prep_notices
        if notice.timer_type.startswith("FLOW:SPEAKING_PREP:")
        and notice.event_type is MeetingEventType.TIMER_EXPIRED
    )
    await stack.controller.handle_timer_notice(prep_expired)
    active = await stack.speaking_flow.state(stack.session_id, item.id)
    assert active is not None and active.stage.value == "ACTIVE"

    before_extension = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type.startswith("FLOW:SPEAKING_TURN:")
    )
    assert await stack.controller.controlled_speaking_extend(
        guild_id=103,
        actor_user_id=100,
    )
    after_extension = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type.startswith("FLOW:SPEAKING_TURN:")
    )
    assert after_extension.deadline_at - before_extension.deadline_at == timedelta(seconds=30)

    with pytest.raises(MeetingPermissionError):
        await stack.controller.controlled_speaking_abandon(
            guild_id=103,
            actor_user_id=100,
        )
    assert await stack.controller.controlled_speaking_abandon(
        guild_id=103,
        actor_user_id=200,
    )
    followup = await stack.speaking_flow.state(stack.session_id, item.id)
    assert followup is not None and followup.stage.value == "FOLLOWUP"
    event_types = [event.event_type for event in await stack.events.list(stack.session_id)]
    assert MeetingEventType.SPEAKING_TURN_EXTENDED.value in event_types
    assert any(
        event.event_type == MeetingEventType.SPEAKING_TURN_FINISHED.value
        and event.payload.get("reason") == "ABSENT"
        for event in await stack.events.list(stack.session_id)
    )

    assert await stack.controller.controlled_followup_finish(
        guild_id=103,
        actor_user_id=100,
    )
    completed = await stack.speaking_flow.state(stack.session_id, item.id)
    assert completed is not None and completed.stage.value == "COMPLETED"
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.POST_PROCESSING
    assert notices
    assert "speak_closing" in actions
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_presentation_allows_host_as_presenter_but_keeps_controls_host_only(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=104)
    for user_id, name in ((100, "Host"), (200, "Co-presenter"), (300, "Audience")):
        await stack.participants.join(
            session_id=stack.session_id,
            user_id=user_id,
            inherited_consents=frozenset(ConsentKind),
            display_name=name,
            is_host=user_id == 100,
        )
    item = await stack.controller.agenda_add(
        guild_id=104,
        actor_user_id=100,
        title="产品演讲",
        duration_minutes=2,
        speaking_policy=AgendaSpeakingPolicy.presentation(
            presenter_user_ids=(100, 200),
        ),
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    flow_notices: list[object] = []

    async def record_notice(_session: object, notice: object) -> None:
        flow_notices.append(notice)

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_speaking_flow_handler(record_notice)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.controller.start(guild_id=104, actor_user_id=100)
    opening = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(opening[-1])
    prep = await stack.speaking_flow.state(stack.session_id, item.id)
    assert prep is not None
    assert prep.current_user_ids == (100, 200)
    assert 300 not in prep.current_user_ids

    with pytest.raises(MeetingPermissionError):
        await stack.controller.controlled_speaking_finish(
            guild_id=104,
            actor_user_id=200,
        )
    prep_notices = await stack.timers.tick(now=baseline + timedelta(seconds=30))
    await stack.controller.handle_timer_notice(
        next(
            notice
            for notice in prep_notices
            if notice.event_type is MeetingEventType.TIMER_EXPIRED
            and notice.timer_type.startswith("FLOW:SPEAKING_PREP:")
        )
    )
    active = await stack.speaking_flow.state(stack.session_id, item.id)
    assert active is not None and active.current_user_ids == (100, 200)
    assert active.planned_seconds == 120
    with pytest.raises(MeetingPermissionError):
        await stack.controller.controlled_speaking_abandon(
            guild_id=104,
            actor_user_id=200,
        )
    assert await stack.controller.controlled_speaking_finish(
        guild_id=104,
        actor_user_id=100,
    )
    completed = await stack.speaking_flow.state(stack.session_id, item.id)
    assert completed is not None and completed.stage.value == "COMPLETED"
    assert flow_notices
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stop_closes_active_controlled_floor_for_worker_facts(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=105)
    for user_id, name in ((100, "Host"), (200, "Speaker")):
        await stack.participants.join(
            session_id=stack.session_id,
            user_id=user_id,
            inherited_consents=frozenset(ConsentKind),
            display_name=name,
            is_host=user_id == 100,
        )
    item = await stack.controller.agenda_add(
        guild_id=105,
        actor_user_id=100,
        title="可中止轮次",
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=(SpeakerTurnPlan(user_id=200, planned_seconds=60),),
            prep_seconds=0,
        ),
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.controller.start(guild_id=105, actor_user_id=100)
    opening = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(opening[-1])
    active = await stack.speaking_flow.state(stack.session_id, item.id)
    assert active is not None and active.stage.value == "ACTIVE"

    stopped = await stack.controller.stop(guild_id=105, actor_user_id=100)

    assert stopped.session.state is MeetingState.STOPPED
    completed = await stack.speaking_flow.state(stack.session_id, item.id)
    assert completed is not None and completed.stage.value == "COMPLETED"
    events = await stack.events.list(stack.session_id)
    assert any(
        event.event_type == MeetingEventType.SPEAKING_FLOW_COMPLETED.value
        and event.payload.get("reason") == "MEETING_STOPPED"
        for event in events
    )
    facts = project_authoritative_speaking_facts(events, labels={100: "Host", 200: "Speaker"})
    assert len(facts.turns) == 1
    assert facts.turns[0].finish_reason == "HOST_ENDED_EARLY"
    assert facts.turns[0].started_at is not None
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_next_closes_preparation_and_back_skip_fail_closed(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=106)
    for user_id in (100, 200):
        await stack.participants.join(
            session_id=stack.session_id,
            user_id=user_id,
            inherited_consents=frozenset(ConsentKind),
            display_name=f"User {user_id}",
            is_host=user_id == 100,
        )
    item = await stack.controller.agenda_add(
        guild_id=106,
        actor_user_id=100,
        title="准备中跳转",
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=(SpeakerTurnPlan(user_id=200, planned_seconds=60),),
        ),
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.controller.start(guild_id=106, actor_user_id=100)
    opening = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(opening[-1])
    preparing = await stack.speaking_flow.state(stack.session_id, item.id)
    assert preparing is not None and preparing.stage.value == "PREPARING"

    with pytest.raises(ValueError, match="不能返回"):
        await stack.controller.back(guild_id=106, actor_user_id=100)
    with pytest.raises(ValueError, match="不能跳过固定状态"):
        await stack.controller.skip(guild_id=106, actor_user_id=100)

    await stack.controller.next(guild_id=106, actor_user_id=100)
    completed = await stack.speaking_flow.state(stack.session_id, item.id)
    assert completed is not None and completed.stage.value == "COMPLETED"
    events = await stack.events.list(stack.session_id)
    assert any(
        event.event_type == MeetingEventType.SPEAKING_TURN_FINISHED.value
        and event.payload.get("reason") == "HOST_SKIPPED"
        for event in events
    )
    assert any(
        event.event_type == MeetingEventType.SPEAKING_FLOW_COMPLETED.value
        and event.payload.get("reason") == "AGENDA_ADVANCED"
        for event in events
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timer_and_vote_recover_with_reminders_one_row_per_voter_and_exact_result(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=2)
    now = datetime(2026, 7, 27, 0, 0, tzinfo=UTC)
    for user_id in (100, 200, 300):
        await stack.participants.join(session_id=stack.session_id, user_id=user_id)

    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="Ship release",
        duration_seconds=180,
        actor_user_id=100,
        source="SLASH_COMMAND",
        now=now,
    )
    await stack.voting.cast(
        vote_id=vote.id,
        session_id=stack.session_id,
        user_id=100,
        option_id="yes",
        source="BUTTON",
        now=now,
    )
    await stack.voting.cast(
        vote_id=vote.id,
        session_id=stack.session_id,
        user_id=100,
        option_id="no",
        source="BUTTON",
        now=now + timedelta(seconds=1),
    )
    await stack.voting.cast(
        vote_id=vote.id,
        session_id=stack.session_id,
        user_id=200,
        option_id="yes",
        source="BUTTON",
        now=now,
    )
    await stack.voting.cast(
        vote_id=vote.id,
        session_id=stack.session_id,
        user_id=300,
        option_id="abstain",
        source="BUTTON",
        now=now,
    )
    async with stack.database.sessions() as db:
        entry_count = await db.scalar(
            select(func.count(MeetingVoteEntryRow.id)).where(MeetingVoteEntryRow.vote_id == vote.id)
        )
    assert entry_count == 3

    recovered_voting = VotingService(
        votes=VoteRepository(stack.database.sessions),
        timers=TimerRepository(stack.database.sessions),
        participants=ParticipantRepository(stack.database.sessions),
        events=MeetingEventStore(stack.database.sessions),
    )
    recovered_timers = TimerService(
        timers=TimerRepository(stack.database.sessions),
        events=MeetingEventStore(stack.database.sessions),
        vote_closer=recovered_voting,
    )
    assert await recovered_timers.tick(now=now + timedelta(seconds=119)) == ()
    reminder_60 = await recovered_timers.tick(now=now + timedelta(seconds=120))
    assert [notice.event_type.value for notice in reminder_60] == ["TIMER_REMINDER_60"]
    assert await recovered_timers.tick(now=now + timedelta(seconds=121)) == ()

    second_restart = TimerService(
        timers=TimerRepository(stack.database.sessions),
        events=MeetingEventStore(stack.database.sessions),
        vote_closer=recovered_voting,
    )
    assert await second_restart.tick(now=now + timedelta(seconds=170)) == ()
    expired = await second_restart.tick(now=now + timedelta(seconds=180))
    assert len(expired) == 1
    outcome = expired[0].vote_outcome
    assert outcome is not None
    assert outcome.counts == {"yes": 1, "no": 1, "abstain": 1}
    assert outcome.total == 3
    assert "共 3 票" in outcome.broadcast_text()
    assert await second_restart.tick(now=now + timedelta(seconds=181)) == ()

    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agenda_private_vote_snapshots_eligibility_and_counts_nonresponse_as_abstain(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=20)
    now = datetime(2026, 7, 28, 0, 0, tzinfo=UTC)
    for user_id in (100, 200, 300, 400):
        await stack.participants.join(session_id=stack.session_id, user_id=user_id)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="预算审批",
        requires_vote=True,
        vote_content="是否批准本季度预算?",
        vote_duration_seconds=20,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title=item.vote_content or item.title,
        duration_seconds=item.vote_duration_seconds or 20,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        eligible_user_ids=(100, 200, 300),
        now=now,
    )
    recipients = await stack.voting.recipients(vote.id)
    assert [recipient.user_id for recipient in recipients] == [100, 200, 300]
    await stack.voting.mark_delivery(
        vote_id=vote.id,
        user_id=100,
        delivery_status="DELIVERED",
        dm_message_id=9001,
    )
    assert (await stack.voting.vote_for_dm_message(message_id=9001, user_id=100)).id == vote.id

    with pytest.raises(ParticipantNotPresentError):
        await stack.voting.cast(
            vote_id=vote.id,
            session_id=stack.session_id,
            user_id=400,
            option_id="yes",
            source="SLASH_COMMAND",
            now=now,
        )
    await stack.voting.cast(
        vote_id=vote.id,
        session_id=stack.session_id,
        user_id=100,
        option_id="yes",
        source="DM_BUTTON",
        now=now,
    )
    expired = await stack.timers.tick(now=now + timedelta(seconds=20))
    outcome = expired[-1].vote_outcome
    assert outcome is not None
    assert outcome.counts == {"yes": 1, "no": 0, "abstain": 2}
    assert outcome.eligible_count == 3
    assert outcome.result == "是"
    assert "2人弃权" in outcome.broadcast_text()
    closed_event = [
        event
        for event in await stack.events.list(stack.session_id)
        if event.event_type == "VOTE_CLOSED"
    ][-1]
    assert closed_event.payload["agenda_item_id"] == item.id
    assert closed_event.payload["result"] == "是"
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_private_ballots_close_early_then_wait_five_seconds(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=21)
    for user_id in (100, 200):
        await stack.participants.join(
            session_id=stack.session_id,
            user_id=user_id,
            inherited_consents=frozenset(ConsentKind),
            is_host=user_id == 100,
        )
    await stack.agenda.add(
        session_id=stack.session_id,
        title="发布决策",
        duration_minutes=1,
        requires_vote=True,
        vote_content="是否立即发布?",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    outcomes = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def record_outcome(_session: object, outcome: object) -> None:
        outcomes.append(outcome)

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.controller.start(guild_id=21, actor_user_id=100)
    baseline = datetime.now(UTC)
    opening = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(opening[-1])
    discussion = await stack.timers.tick(now=baseline + timedelta(seconds=72))
    await stack.controller.handle_timer_notice(discussion[-1])

    status = await stack.controller.status(21)
    assert status.session.state is MeetingState.VOTE_OPEN
    assert status.vote is not None
    await stack.controller.mark_vote_delivery(
        vote_id=status.vote.id,
        user_id=100,
        delivery_status="DELIVERED",
        dm_message_id=9101,
    )
    await stack.controller.mark_vote_delivery(
        vote_id=status.vote.id,
        user_id=200,
        delivery_status="DELIVERED",
        dm_message_id=9102,
    )
    await stack.controller.vote_cast_private(message_id=9101, user_id=100, option_id="yes")
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    await stack.controller.vote_cast_private(message_id=9102, user_id=200, option_id="no")
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    assert len(outcomes) == 1
    assert outcomes[0].counts == {"yes": 1, "no": 1, "abstain": 0}
    assert outcomes[0].result == "平票"
    active_timers = await stack.timer_repository.list_active(stack.session_id)
    assert len(active_timers) == 1
    assert active_timers[0].timer_type.startswith("FLOW:VOTE_RESULT:")

    result_notices = await stack.timers.tick(now=datetime.now(UTC) + timedelta(seconds=6))
    await stack.controller.handle_timer_notice(result_notices[-1])
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.POST_PROCESSING
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timer_pause_resume_and_host_authorization_survive_component_flow(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=3)
    now = datetime(2026, 7, 27, 0, 0, tzinfo=UTC)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.timers.start(
        session_id=stack.session_id,
        timer_type="DISCUSSION",
        duration_seconds=120,
        now=now,
    )
    assert await stack.timers.pause(stack.session_id, now=now + timedelta(seconds=30)) == 1

    recovered_timers = TimerService(
        timers=TimerRepository(stack.database.sessions),
        events=MeetingEventStore(stack.database.sessions),
    )
    assert (
        await recovered_timers.resume(
            stack.session_id,
            now=now + timedelta(seconds=100),
        )
        == 1
    )
    active = await TimerRepository(stack.database.sessions).list_active(stack.session_id)
    assert len(active) == 1
    assert active[0].deadline_at == now + timedelta(seconds=190)
    assert await recovered_timers.tick(now=now + timedelta(seconds=130)) == ()

    with pytest.raises(MeetingPermissionError):
        await stack.controller.pause(guild_id=3, actor_user_id=999)

    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dropped_vote_expiry_and_failed_handler_are_reconciled_from_events(
    tmp_path: Path,
) -> None:
    """The event stream, not the transient TimerNotice, owns expiry recovery."""

    stack = await _stack(tmp_path, guild_id=220)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=frozenset(ConsentKind),
        is_host=True,
    )
    await stack.agenda.add(
        session_id=stack.session_id,
        title="需要表决的议题",
        duration_minutes=1,
        requires_vote=True,
        vote_content="是否通过?",
        vote_duration_seconds=10,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.controller.start(guild_id=220, actor_user_id=100)
    opening_timer = (await stack.timer_repository.list_active(stack.session_id))[0]
    opening = await stack.timers.tick(now=opening_timer.deadline_at + timedelta(seconds=1))
    await stack.controller.handle_timer_notice(opening[-1])

    agenda_timer = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type.startswith("FLOW:AGENDA:")
    )
    agenda_expiry = await stack.timers.tick(now=agenda_timer.deadline_at + timedelta(seconds=1))
    await stack.controller.handle_timer_notice(agenda_expiry[-1])
    status = await stack.controller.status(220)
    assert status.session.state is MeetingState.VOTE_OPEN
    assert status.vote is not None
    original_vote_id = status.vote.id

    vote_timer = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type == f"VOTE:{original_vote_id}"
    )
    dropped = await stack.timers.tick(now=vote_timer.deadline_at + timedelta(seconds=1))
    assert dropped[-1].event_type is MeetingEventType.TIMER_EXPIRED
    assert dropped[-1].vote_outcome is not None
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    assert await stack.voting.active_for_session(stack.session_id) is None

    failures_remaining = 1

    async def fail_result_action_once(_session: object, action: str) -> None:
        nonlocal failures_remaining
        if action == "close_vote" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("synthetic post-transition handler failure")

    stack.controller.set_runtime_action_handler(fail_result_action_once)
    with pytest.raises(RuntimeError, match="synthetic post-transition"):
        await stack.controller.recover_automatic_flows(session_ids={stack.session_id})
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    assert await stack.timer_repository.list_active(stack.session_id) == ()

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.VOTE_RESULT
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    latest_vote = await stack.voting.latest_for_session(stack.session_id)
    assert latest_vote is not None and latest_vote.id == original_vote_id
    events = await stack.events.list(stack.session_id)
    assert sum(event.event_type == MeetingEventType.VOTE_OPENED.value for event in events) == 1
    active = await stack.timer_repository.list_active(stack.session_id)
    assert len(active) == 1
    assert active[0].timer_type == f"FLOW:VOTE_RESULT:{original_vote_id}"
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expiry_committed_before_pause_is_consumed_before_resume_restarts_flow(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=226)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="暂停前已到期",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(4):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    baseline = datetime.now(UTC)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=f"FLOW:AGENDA:{item.id}",
        duration_seconds=60,
        now=baseline,
    )
    dropped = (await stack.timers.tick(now=baseline + timedelta(seconds=61)))[-1]
    assert dropped.timer_id == timer.id

    await stack.controller.pause(guild_id=226, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.PAUSED
    assert await stack.timers.durable_expiry_notices(stack.session_id) == (dropped,)

    resumed = await stack.controller.resume(guild_id=226, actor_user_id=100)

    assert resumed.session.state is not MeetingState.DISCUSSION_READY
    assert (await stack.sessions.get(stack.session_id)).state is not MeetingState.DISCUSSION_READY
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    assert await stack.timer_repository.list_active(stack.session_id) == ()
    handoffs = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
    )
    assert [event.payload["timer_id"] for event in handoffs] == [timer.id]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_cannot_replace_expiry_committed_after_its_first_handoff_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=234)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="恢复扫描竞态",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(4):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    baseline = datetime.now(UTC) - timedelta(seconds=120)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=f"FLOW:AGENDA:{item.id}",
        duration_seconds=60,
        now=baseline,
    )
    await stack.controller.pause(guild_id=234, actor_user_id=100)
    paused = await stack.timer_repository.list_active(stack.session_id)
    assert paused[0].status is TimerStatus.PAUSED
    assert paused[0].remaining_ms == 0

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    first_scan_finished = asyncio.Event()
    allow_first_scan_to_return = asyncio.Event()
    original_durable = stack.timers.durable_expiry_notices
    resume_scan_count = 0

    async def gate_first_resume_scan(session_id: str) -> object:
        nonlocal resume_scan_count
        notices = await original_durable(session_id)
        if asyncio.current_task() is resume_task and resume_scan_count == 0:
            resume_scan_count += 1
            assert notices == ()
            first_scan_finished.set()
            await allow_first_scan_to_return.wait()
        return notices

    monkeypatch.setattr(stack.timers, "durable_expiry_notices", gate_first_resume_scan)
    resume_task = asyncio.create_task(
        stack.controller.resume(guild_id=234, actor_user_id=100),
        name="resume-scan-race",
    )
    await asyncio.wait_for(first_scan_finished.wait(), timeout=1)
    dropped = await stack.timers.tick(now=datetime.now(UTC) + timedelta(seconds=1))
    assert dropped[-1].timer_id == timer.id
    allow_first_scan_to_return.set()

    resumed = await resume_task

    assert resumed.session.state is not MeetingState.DISCUSSION_READY
    assert (await stack.sessions.get(stack.session_id)).state is not MeetingState.DISCUSSION_READY
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    assert await stack.timer_repository.list_active(stack.session_id) == ()
    starts = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_STARTED.value,),
    )
    assert sum(event.payload.get("timer_type") == timer.timer_type for event in starts) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_vote_expiry_retries_broadcast_from_durable_vote_projection(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=227)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="公开投票恢复",
        duration_seconds=10,
        actor_user_id=100,
        source="TEST",
        now=baseline,
    )
    timer = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type == f"VOTE:{vote.id}"
    )
    dropped = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert dropped.vote_outcome is not None
    attempts = 0

    async def fail_once(_session: object, outcome: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic public vote broadcast failure")
        assert outcome is not None

    stack.controller.set_vote_outcome_handler(fail_once)
    with pytest.raises(RuntimeError, match="synthetic public vote broadcast failure"):
        await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})
    assert attempts == 1
    assert [
        notice.timer_id for notice in await stack.timers.durable_expiry_notices(stack.session_id)
    ] == [timer.id]

    pending = (await stack.timers.durable_expiry_notices(stack.session_id))[0]
    assert await asyncio.gather(
        stack.controller.handle_timer_notice(pending),
        stack.controller.handle_timer_notice(pending),
    ) == [None, None]
    assert attempts == 2
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    assert await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id}) == ()
    assert attempts == 2
    handoff_events = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
    )
    assert sum(event.payload.get("timer_id") == timer.id for event in handoff_events) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_vote_entry_is_blocked_and_legacy_expiry_is_deferred_until_resume(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=252)
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.controller.pause(guild_id=252, actor_user_id=100)

    with pytest.raises(ValueError, match="当前会议状态不能另开公开投票"):
        await stack.controller.vote_open(
            guild_id=252,
            actor_user_id=100,
            title="暂停态不得创建",
            duration_seconds=10,
        )
    assert await stack.timer_repository.list_active(stack.session_id) == ()

    baseline = datetime.now(UTC)
    legacy_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="旧版暂停态遗留票",
        duration_seconds=10,
        actor_user_id=100,
        source="LEGACY_PATH",
        now=baseline,
    )
    with pytest.raises(ValueError, match="Meeting is paused"):
        await stack.controller.vote_cast(
            guild_id=252,
            user_id=200,
            option_id="yes",
        )
    timer = next(
        candidate
        for candidate in await stack.timer_repository.list_active(stack.session_id)
        if candidate.timer_type == f"VOTE:{legacy_vote.id}"
    )
    dropped = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert dropped.timer_id == timer.id
    assert [
        notice.timer_id for notice in await stack.timers.durable_expiry_notices(stack.session_id)
    ] == [timer.id]

    delivered: list[str] = []

    async def record_outcome(_session: object, outcome: object) -> None:
        delivered.append(outcome.vote.id)  # type: ignore[attr-defined]

    stack.controller.set_vote_outcome_handler(record_outcome)
    assert await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id}) == ()
    assert delivered == []

    await stack.controller.resume(guild_id=252, actor_user_id=100)

    assert delivered == [legacy_vote.id]
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    handoffs = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
    )
    assert sum(event.payload.get("timer_id") == timer.id for event in handoffs) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_vote_blocks_navigation_before_flow_timers_are_cancelled(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=256)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="公开票与议程票隔离",
        duration_minutes=1,
        requires_vote=True,
        vote_content="议程内表决",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(4):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    public_vote = await stack.controller.vote_open(
        guild_id=256,
        actor_user_id=100,
        title="独立公开票",
        duration_seconds=60,
    )
    timers_before = await stack.timer_repository.list_active(stack.session_id)
    events_before = await stack.events.list(stack.session_id)

    with pytest.raises(ValueError, match="公开投票进行中不能切换会议流程"):
        await stack.controller.next(guild_id=256, actor_user_id=100)

    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.DISCUSSION_READY
    assert (await stack.voting.get(public_vote.id)).status is VoteStatus.OPEN
    assert await stack.timer_repository.list_active(stack.session_id) == timers_before
    assert await stack.events.list(stack.session_id) == events_before

    await stack.controller.vote_close(guild_id=256, actor_user_id=100)
    advanced = await stack.controller.next(guild_id=256, actor_user_id=100)

    assert advanced.session.state is MeetingState.VOTE_OPEN
    agenda_vote = await stack.voting.active_for_session(stack.session_id)
    assert agenda_vote is not None
    assert agenda_vote.id != public_vote.id
    assert agenda_vote.agenda_item_id == item.id
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_vote_expiry_recovery_binds_exact_vote_after_a_new_vote_opens(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=231)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    first_vote = await stack.controller.vote_open(
        guild_id=231,
        actor_user_id=100,
        title="第一票",
        duration_seconds=10,
    )
    first_timer = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type == f"VOTE:{first_vote.id}"
    )
    first_notice = (await stack.timers.tick(now=baseline + timedelta(seconds=30)))[-1]
    delivered_vote_ids: list[str] = []

    async def fail_first_delivery(_session: object, outcome: object) -> None:
        vote_id = outcome.vote.id  # type: ignore[attr-defined]
        delivered_vote_ids.append(vote_id)
        if len(delivered_vote_ids) == 1:
            raise RuntimeError("synthetic first vote delivery failure")

    stack.controller.set_vote_outcome_handler(fail_first_delivery)
    with pytest.raises(RuntimeError, match="synthetic first vote delivery failure"):
        await stack.controller.handle_timer_notice(first_notice)

    second_vote = await stack.controller.vote_open(
        guild_id=231,
        actor_user_id=100,
        title="第二票",
        duration_seconds=60,
    )
    assert second_vote.id != first_vote.id

    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert delivered_vote_ids == [first_vote.id, first_vote.id]
    assert (await stack.voting.active_for_session(stack.session_id)).id == second_vote.id  # type: ignore[union-attr]
    handoff_events = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
    )
    assert sum(event.payload.get("timer_id") == first_timer.id for event in handoff_events) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_closed_agenda_vote_projection_advances_instead_of_opening_a_second_vote(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=228)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="关闭后崩溃",
        duration_minutes=1,
        requires_vote=True,
        vote_content="是否通过",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="是否通过",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    await stack.voting.close(vote.id, actor_user_id=None, source="TEST_CRASH_WINDOW")
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.VOTE_RESULT
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    events = await stack.events.list(stack.session_id)
    assert sum(event.event_type == MeetingEventType.VOTE_OPENED.value for event in events) == 1
    assert (await stack.voting.latest_for_session(stack.session_id)).id == vote.id  # type: ignore[union-attr]
    active = await stack.timer_repository.list_active(stack.session_id)
    assert [timer.timer_type for timer in active] == [f"FLOW:VOTE_RESULT:{vote.id}"]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_closed_agenda_vote_recovery_ignores_a_later_closed_public_vote(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=232)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="历史错位恢复",
        duration_minutes=1,
        requires_vote=True,
        vote_content="议程票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    agenda_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="议程票",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    await stack.voting.close(
        agenda_vote.id,
        actor_user_id=None,
        source="LEGACY_CRASH_WINDOW",
    )
    public_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="后来公开票",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
    )
    await stack.voting.close(
        public_vote.id,
        actor_user_id=100,
        source="LEGACY_PATH",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.VOTE_RESULT
    events = await stack.events.list(stack.session_id)
    agenda_open_events = [
        event
        for event in events
        if event.event_type == MeetingEventType.VOTE_OPENED.value
        and event.payload.get("agenda_item_id") == item.id
    ]
    assert len(agenda_open_events) == 1
    active = await stack.timer_repository.list_active(stack.session_id)
    assert any(timer.timer_type == f"FLOW:VOTE_RESULT:{agenda_vote.id}" for timer in active)
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_result_recovery_remains_bound_to_agenda_vote_after_new_public_vote(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=233)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="精确结果绑定",
        duration_minutes=1,
        requires_vote=True,
        vote_content="原议程票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    agenda_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="原议程票",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    spoken_vote_ids: list[str] = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def fail_first_speech(_session: object, outcome: object) -> None:
        vote_id = outcome.vote.id  # type: ignore[attr-defined]
        spoken_vote_ids.append(vote_id)
        if len(spoken_vote_ids) == 1:
            raise RuntimeError("synthetic agenda result speech failure")

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(fail_first_speech)
    with pytest.raises(RuntimeError, match="synthetic agenda result speech failure"):
        await stack.controller.vote_close(guild_id=233, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    with pytest.raises(ValueError, match="不能另开公开投票"):
        await stack.controller.vote_open(
            guild_id=233,
            actor_user_id=100,
            title="被拒绝的公开票",
            duration_seconds=60,
        )

    public_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="模拟旧版产生的公开票",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
    )
    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert spoken_vote_ids == [agenda_vote.id, agenda_vote.id]
    assert (await stack.voting.active_for_session(stack.session_id)).id == public_vote.id  # type: ignore[union-attr]
    active = await stack.timer_repository.list_active(stack.session_id)
    assert any(timer.timer_type == f"FLOW:VOTE_RESULT:{agenda_vote.id}" for timer in active)
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["next", "skip"])
async def test_host_advance_from_vote_open_closes_the_current_epoch_vote(
    tmp_path: Path,
    operation: str,
) -> None:
    stack = await _stack(tmp_path, guild_id=235)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="主持人推进投票",
        duration_minutes=1,
        requires_vote=True,
        vote_content="本轮投票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="本轮投票",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    spoken_vote_ids: list[str] = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def record_outcome(_session: object, outcome: object) -> None:
        spoken_vote_ids.append(outcome.vote.id)  # type: ignore[attr-defined]

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)

    advanced = await getattr(stack.controller, operation)(
        guild_id=235,
        actor_user_id=100,
    )

    assert advanced.session.state is MeetingState.VOTE_RESULT
    closed = await stack.voting.get(vote.id)
    assert closed.status is VoteStatus.CLOSED
    assert spoken_vote_ids == [vote.id]
    active = await stack.timer_repository.list_active(stack.session_id)
    assert [timer.timer_type for timer in active] == [f"FLOW:VOTE_RESULT:{vote.id}"]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_back_to_vote_open_recovery_does_not_reuse_the_previous_epoch_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=236)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="返回重投",
        duration_minutes=1,
        requires_vote=True,
        vote_content="重新投票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=200,
    )
    first_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="第一次",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def ignore_outcome(_session: object, _outcome: object) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(ignore_outcome)
    await stack.controller.vote_close(guild_id=236, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT

    original_open = stack.voting.open
    failures_remaining = 1

    async def fail_first_reopen(**kwargs: object) -> object:
        nonlocal failures_remaining
        if kwargs.get("agenda_item_id") == item.id and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("synthetic reopen action crash")
        return await original_open(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(stack.voting, "open", fail_first_reopen)
    with pytest.raises(RuntimeError, match="synthetic reopen action crash"):
        await stack.controller.back(guild_id=236, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN

    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    current_vote = await stack.voting.active_for_session(stack.session_id)
    assert current_vote is not None
    assert current_vote.id != first_vote.id
    assert current_vote.agenda_item_id == item.id
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    opened = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.VOTE_OPENED.value,),
    )
    assert sum(event.payload.get("agenda_item_id") == item.id for event in opened) == 2
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_open_recovery_never_acks_a_conflicting_public_vote_as_agenda_vote(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=242)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="冲突票恢复",
        duration_minutes=1,
        requires_vote=True,
        vote_content="正确议程票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    vote_open_result: RuntimeResult | None = None
    for _ in range(7):
        vote_open_result = await stack.runtime.advance(stack.session_id)
    assert vote_open_result is not None and vote_open_result.event is not None
    assert vote_open_result.session.state is MeetingState.VOTE_OPEN
    await stack.agenda.activate_next(session_id=stack.session_id)
    public_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="旧版公开票冲突",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    with pytest.raises(RuntimeError, match="outside the current agenda epoch"):
        await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert not any(
        event.payload.get("transition_event_id") == vote_open_result.event.id
        and event.payload.get("action") in {"open_vote", "start_vote_timer"}
        for event in completions
    )

    closed = await stack.controller.vote_close(guild_id=242, actor_user_id=100)
    assert closed.vote.id == public_vote.id
    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    agenda_vote = await stack.voting.active_for_session(stack.session_id)
    assert agenda_vote is not None
    assert agenda_vote.id != public_vote.id
    assert agenda_vote.agenda_item_id == item.id
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert (
        sum(
            event.payload.get("transition_event_id") == vote_open_result.event.id
            and event.payload.get("action") == "open_vote"
            for event in completions
        )
        == 1
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_back_from_vote_open_is_rejected_without_mutating_vote_epoch(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=245)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="投票中禁止返回",
        duration_minutes=1,
        requires_vote=True,
        vote_content="是否保持当前轮次",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="是否保持当前轮次",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    timers_before = await stack.timer_repository.list_active(stack.session_id)
    events_before = await stack.events.list(stack.session_id)

    with pytest.raises(ValueError, match="投票进行中不能返回"):
        await stack.controller.back(guild_id=245, actor_user_id=100)

    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    assert (await stack.voting.get(vote.id)).status is VoteStatus.OPEN
    assert await stack.timer_repository.list_active(stack.session_id) == timers_before
    assert await stack.events.list(stack.session_id) == events_before
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_back_from_pending_next_agenda_item_cannot_complete_agenda_action(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=246)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="换项崩溃窗口",
        duration_minutes=1,
        requires_vote=True,
        vote_content="不会执行的投票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(9):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    events_before = await stack.events.list(stack.session_id)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.NEXT_AGENDA_ITEM

    with pytest.raises(ValueError, match="换项阶段"):
        await stack.controller.back(guild_id=246, actor_user_id=100)

    agenda = await stack.agenda.list(stack.session_id)
    assert (
        next(candidate for candidate in agenda if candidate.id == item.id).status
        is AgendaStatus.ACTIVE
    )
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.NEXT_AGENDA_ITEM
    assert await stack.events.list(stack.session_id) == events_before
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_open_recovery_closes_a_stale_agenda_epoch_before_reopening(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=247)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="旧时代投票自愈",
        duration_minutes=1,
        requires_vote=True,
        vote_content="应创建新轮次",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    stale_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="旧轮次",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )
    await stack.runtime.advance(stack.session_id)
    await stack.runtime.back(
        stack.session_id,
        actor_user_id=100,
        source="LEGACY_PATH",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert (await stack.voting.get(stale_vote.id)).status is VoteStatus.CLOSED
    current_vote = await stack.voting.active_for_session(stack.session_id)
    assert current_vote is not None
    assert current_vote.id != stale_vote.id
    assert current_vote.agenda_item_id == item.id
    active = await stack.timer_repository.list_active(stack.session_id)
    assert all(timer.timer_type != f"VOTE:{stale_vote.id}" for timer in active)
    assert any(timer.timer_type == f"VOTE:{current_vote.id}" for timer in active)
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_result_actions_ignore_same_agenda_vote_opened_after_transition(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=248)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="结果截止点绑定",
        duration_minutes=1,
        requires_vote=True,
        vote_content="只播报转换前投票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    epoch_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="转换前投票",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )
    await stack.voting.cast(
        vote_id=epoch_vote.id,
        session_id=stack.session_id,
        user_id=200,
        option_id="yes",
        source="TEST",
    )
    await stack.voting.close(epoch_vote.id, actor_user_id=None, source="TEST")
    transition = await stack.runtime.advance(stack.session_id)
    assert transition.session.state is MeetingState.VOTE_RESULT
    later_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="转换后同议程投票",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )
    await stack.voting.cast(
        vote_id=later_vote.id,
        session_id=stack.session_id,
        user_id=200,
        option_id="no",
        source="LEGACY_PATH",
    )
    spoken_vote_ids: list[str] = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def record_outcome(_session: object, outcome: object) -> None:
        spoken_vote_ids.append(outcome.vote.id)  # type: ignore[attr-defined]

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)
    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert spoken_vote_ids == [epoch_vote.id]
    active = await stack.timer_repository.list_active(stack.session_id)
    assert any(timer.timer_type == f"FLOW:VOTE_RESULT:{epoch_vote.id}" for timer in active)
    assert all(timer.timer_type != f"FLOW:VOTE_RESULT:{later_vote.id}" for timer in active)
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_result_resume_keeps_original_transition_cutoff(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=251)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="恢复仍绑定原结果",
        duration_minutes=1,
        requires_vote=True,
        vote_content="恢复不能改绑",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    epoch_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="原结果",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )
    spoken_vote_ids: list[str] = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def record_outcome(_session: object, outcome: object) -> None:
        spoken_vote_ids.append(outcome.vote.id)  # type: ignore[attr-defined]

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)
    await stack.controller.vote_close(guild_id=251, actor_user_id=100)
    later_vote = await stack.voting.open(
        session_id=stack.session_id,
        title="恢复前旧版错票",
        duration_seconds=60,
        actor_user_id=100,
        source="LEGACY_PATH",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )

    await stack.controller.pause(guild_id=251, actor_user_id=100)
    resumed = await stack.controller.resume(guild_id=251, actor_user_id=100)

    assert resumed.session.state is MeetingState.VOTE_RESULT
    assert spoken_vote_ids == [epoch_vote.id]
    active = await stack.timer_repository.list_active(stack.session_id)
    result_timer_types = {
        timer.timer_type for timer in active if timer.timer_type.startswith("FLOW:VOTE_RESULT:")
    }
    assert result_timer_types == {f"FLOW:VOTE_RESULT:{epoch_vote.id}"}
    assert f"FLOW:VOTE_RESULT:{later_vote.id}" not in result_timer_types
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cast_mode", ["public", "private"])
async def test_agenda_ballot_cannot_cross_skip_transition_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cast_mode: str,
) -> None:
    guild_id = 249 if cast_mode == "public" else 250
    stack = await _stack(tmp_path, guild_id=guild_id)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="并发截止线",
        duration_minutes=1,
        requires_vote=True,
        vote_content="转换后不可投票",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="转换后不可投票",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )
    if cast_mode == "private":
        await stack.controller.mark_vote_delivery(
            vote_id=vote.id,
            user_id=200,
            delivery_status="DELIVERED",
            dm_message_id=9250,
        )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    original_skip = stack.runtime.skip
    transition_committed = asyncio.Event()
    allow_skip_actions = asyncio.Event()

    async def gated_skip(*args: object, **kwargs: object) -> RuntimeResult:
        result = await original_skip(*args, **kwargs)  # type: ignore[arg-type]
        transition_committed.set()
        await allow_skip_actions.wait()
        return result

    monkeypatch.setattr(stack.runtime, "skip", gated_skip)
    skip_task = asyncio.create_task(
        stack.controller.skip(
            guild_id=guild_id,
            actor_user_id=100,
        )
    )
    await asyncio.wait_for(transition_committed.wait(), timeout=1)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    if cast_mode == "private":
        cast_task = asyncio.create_task(
            stack.controller.vote_cast_private(
                message_id=9250,
                user_id=200,
                option_id="yes",
            )
        )
    else:
        cast_task = asyncio.create_task(
            stack.controller.vote_cast(
                guild_id=guild_id,
                user_id=200,
                option_id="yes",
            )
        )
    await asyncio.sleep(0)
    assert not cast_task.done()
    allow_skip_actions.set()
    await skip_task
    with pytest.raises(ValueError, match=r"open vote|only while voting"):
        await cast_task

    outcome = await stack.voting.close(vote.id, actor_user_id=None, source="ASSERT")
    assert outcome.counts == {"yes": 0, "no": 0, "abstain": 1}
    cast_events = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.VOTE_CAST.value,),
    )
    assert cast_events == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recover_actions_replays_resume_timers_not_destination_state_actions(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=237)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="恢复精确动作",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(4):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=f"FLOW:AGENDA:{item.id}",
        duration_seconds=60,
    )
    discussion_entry = await stack.runtime.recover_actions(stack.session_id)
    assert discussion_entry.event is not None
    await stack.events.append(
        session_id=stack.session_id,
        event_type=MeetingEventType.RUNTIME_ACTION_COMPLETED.value,
        payload={
            "transition_event_id": discussion_entry.event.id,
            "action": "select_next_speaker",
        },
    )
    await stack.controller.pause(guild_id=237, actor_user_id=100)
    assert (await stack.timer_repository.list_active(stack.session_id))[0].status is (
        TimerStatus.PAUSED
    )
    resumed = await stack.runtime.resume(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert resumed.actions == ("resume_timers",)
    assert (await stack.timer_repository.list_active(stack.session_id))[0].status is (
        TimerStatus.PAUSED
    )
    replayed_actions: list[str] = []

    async def record_action(_session: object, action: str) -> None:
        replayed_actions.append(action)

    stack.controller.set_runtime_action_handler(record_action)
    await stack.controller.recover_automatic_flows(session_ids={stack.session_id})

    active = await stack.timer_repository.list_active(stack.session_id)
    restored = next(candidate for candidate in active if candidate.id == timer.id)
    assert restored.status is TimerStatus.RUNNING
    assert replayed_actions == ["resume_timers"]
    resumed_event = next(
        event
        for event in reversed(await stack.events.list(stack.session_id))
        if event.event_type == MeetingEventType.MEETING_RESUMED.value
    )
    assert resumed_event.payload["runtime_actions"] == ["resume_timers"]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_periodic_reconcile_retries_a_failed_resume_timers_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=241)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="恢复动作瞬时失败",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(4):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=f"FLOW:AGENDA:{item.id}",
        duration_seconds=60,
    )
    await stack.controller.pause(guild_id=241, actor_user_id=100)
    original_resume = stack.timers.resume
    failures_remaining = 1

    async def fail_resume_once(session_id: str, **kwargs: object) -> int:
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("synthetic timer resume failure")
        return await original_resume(session_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(stack.timers, "resume", fail_resume_once)
    with pytest.raises(RuntimeError, match="synthetic timer resume failure"):
        await stack.controller.resume(guild_id=241, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.DISCUSSION_READY
    assert (await stack.timer_repository.list_active(stack.session_id))[0].status is (
        TimerStatus.PAUSED
    )

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.DISCUSSION_READY
    active = await stack.timer_repository.list_active(stack.session_id)
    restored = next(candidate for candidate in active if candidate.id == timer.id)
    assert restored.status is TimerStatus.RUNNING
    completed = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    resumed_event = next(
        event
        for event in reversed(await stack.events.list(stack.session_id))
        if event.event_type == MeetingEventType.MEETING_RESUMED.value
    )
    assert any(
        event.payload
        == {
            "transition_event_id": resumed_event.id,
            "action": "resume_timers",
        }
        for event in completed
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pause_resume_replays_unfinished_actions_from_the_workflow_entry(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=243)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="暂停前未完成动作",
        duration_minutes=1,
        requires_vote=True,
        vote_content="必须恢复播报",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="必须恢复播报",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )
    attempts: list[str] = []

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def fail_first_speech(_session: object, outcome: object) -> None:
        attempts.append(outcome.vote.id)  # type: ignore[attr-defined]
        if len(attempts) == 1:
            raise RuntimeError("synthetic pre-pause speech failure")

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(fail_first_speech)
    with pytest.raises(RuntimeError, match="synthetic pre-pause speech failure"):
        await stack.controller.vote_close(guild_id=243, actor_user_id=100)
    await stack.controller.pause(guild_id=243, actor_user_id=100)

    resumed = await stack.controller.resume(guild_id=243, actor_user_id=100)

    assert resumed.session.state is MeetingState.VOTE_RESULT
    assert attempts == [vote.id, vote.id]
    active = await stack.timer_repository.list_active(stack.session_id)
    assert [timer.timer_type for timer in active] == [f"FLOW:VOTE_RESULT:{vote.id}"]
    transitions = [
        event
        for event in await stack.events.list(stack.session_id)
        if event.state_before is not event.state_after
        and event.state_after is MeetingState.VOTE_RESULT
        and event.state_before not in {MeetingState.PAUSED, MeetingState.INTERRUPTED}
    ]
    result_transition = transitions[-1]
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert (
        sum(
            event.payload.get("transition_event_id") == result_transition.id
            and event.payload.get("action") == "speak_vote_result"
            for event in completions
        )
        == 1
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_next_close_fact_recovers_when_transition_fails_after_timer_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=244)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="下一步故障恢复",
        duration_minutes=1,
        requires_vote=True,
        vote_content="关闭事实先持久化",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="关闭事实先持久化",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        eligible_user_ids=(200,),
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    async def fail_transition(*_args: object, **_kwargs: object) -> RuntimeResult:
        raise RuntimeError("synthetic next transition failure")

    stack.controller.set_runtime_action_handler(ignore_action)
    monkeypatch.setattr(stack.runtime, "next", fail_transition)
    with pytest.raises(RuntimeError, match="synthetic next transition failure"):
        await stack.controller.next(guild_id=244, actor_user_id=100)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_OPEN
    assert (await stack.voting.get(vote.id)).status is VoteStatus.CLOSED
    assert await stack.timer_repository.list_active(stack.session_id) == ()

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.VOTE_RESULT
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT
    active = await stack.timer_repository.list_active(stack.session_id)
    assert [timer.timer_type for timer in active] == [f"FLOW:VOTE_RESULT:{vote.id}"]
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_opening_resume_reuses_the_original_epoch_timer_identity(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=238)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    opening = await stack.runtime.advance(stack.session_id)
    assert opening.session.state is MeetingState.OPENING
    assert opening.event is not None
    timer_type = f"FLOW:OPENING:{opening.event.sequence_number}"
    original_timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=timer_type,
        duration_seconds=10,
    )
    await stack.controller.pause(guild_id=238, actor_user_id=100)

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    resumed = await stack.controller.resume(guild_id=238, actor_user_id=100)

    assert resumed.session.state is MeetingState.OPENING
    active = await stack.timer_repository.list_active(stack.session_id)
    assert len(active) == 1
    assert active[0].id == original_timer.id
    assert active[0].timer_type == timer_type
    assert active[0].status is TimerStatus.RUNNING
    starts = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_STARTED.value,),
    )
    assert sum(event.payload.get("timer_type") == timer_type for event in starts) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_agenda_vote_expiry_keeps_its_original_vote_open_epoch(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=239)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="暂停投票 epoch",
        duration_minutes=1,
        requires_vote=True,
        vote_content="暂停前到期",
        vote_duration_seconds=10,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    baseline = datetime.now(UTC)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="暂停前到期",
        duration_seconds=10,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
        now=baseline,
    )
    dropped = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert dropped.vote_outcome is not None
    await stack.controller.pause(guild_id=239, actor_user_id=100)

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    spoken: list[str] = []

    async def record_outcome(_session: object, outcome: object) -> None:
        spoken.append(outcome.vote.id)  # type: ignore[attr-defined]

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)
    resumed = await stack.controller.resume(guild_id=239, actor_user_id=100)

    assert resumed.session.state is MeetingState.VOTE_RESULT
    assert spoken == [vote.id]
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_manual_public_vote_close_cancels_its_timer(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=240)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    vote = await stack.controller.vote_open(
        guild_id=240,
        actor_user_id=100,
        title="手工关闭公开票",
        duration_seconds=10,
    )
    timer = next(
        candidate
        for candidate in await stack.timer_repository.list_active(stack.session_id)
        if candidate.timer_type == f"VOTE:{vote.id}"
    )

    await stack.controller.vote_close(guild_id=240, actor_user_id=100)

    assert await stack.timer_repository.list_active(stack.session_id) == ()
    assert await stack.timers.tick(now=timer.deadline_at + timedelta(seconds=1)) == ()
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completed_runtime_action_is_not_replayed_when_following_timer_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=229)
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="播报幂等",
        duration_minutes=1,
        requires_vote=True,
        vote_content="是否通过",
        vote_duration_seconds=60,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(7):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="是否通过",
        duration_seconds=60,
        actor_user_id=100,
        source="SYSTEM",
        agenda_item_id=item.id,
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    spoken = 0

    async def record_outcome(_session: object, _outcome: object) -> None:
        nonlocal spoken
        spoken += 1

    stack.controller.set_runtime_action_handler(ignore_action)
    stack.controller.set_vote_outcome_handler(record_outcome)
    original_start = stack.timers.start_after_handoff_barrier
    failures_remaining = 1

    async def fail_result_timer_once(**kwargs: object) -> object:
        nonlocal failures_remaining
        timer_type = kwargs.get("timer_type")
        if isinstance(timer_type, str) and timer_type.startswith("FLOW:VOTE_RESULT:"):
            if failures_remaining:
                failures_remaining -= 1
                raise RuntimeError("synthetic result timer failure")
        return await original_start(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        stack.timers,
        "start_after_handoff_barrier",
        fail_result_timer_once,
    )
    with pytest.raises(RuntimeError, match="synthetic result timer failure"):
        await stack.controller.vote_close(guild_id=229, actor_user_id=100)
    assert spoken == 1
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.VOTE_RESULT

    await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert spoken == 1
    active = await stack.timer_repository.list_active(stack.session_id)
    assert any(timer.timer_type == f"FLOW:VOTE_RESULT:{vote.id}" for timer in active)
    action_events = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert sum(event.payload.get("action") == "speak_vote_result" for event in action_events) == 1
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pause_serializes_runtime_transition_before_timer_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tick cannot expire a timer between MEETING_PAUSED and TIMER_PAUSED."""

    stack = await _stack(tmp_path, guild_id=221)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.timers.start(
        session_id=stack.session_id,
        timer_type="PAUSE-RACE",
        duration_seconds=120,
        now=baseline,
    )
    timer_lock_attempted = asyncio.Event()

    class ObservedTimerLock(asyncio.Lock):
        async def acquire(self) -> bool:
            task = asyncio.current_task()
            if task is not None and task.get_name() == "tick-during-pause":
                timer_lock_attempted.set()
            return await super().acquire()

    stack.timers._lock = ObservedTimerLock()
    original_runtime_pause = stack.runtime.pause
    transition_committed = asyncio.Event()
    allow_timer_freeze = asyncio.Event()

    async def gated_runtime_pause(
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        result = await original_runtime_pause(
            session_id,
            actor_user_id=actor_user_id,
            source=source,
        )
        transition_committed.set()
        await allow_timer_freeze.wait()
        return result

    monkeypatch.setattr(stack.runtime, "pause", gated_runtime_pause)
    pause_task = asyncio.create_task(
        stack.controller.pause(guild_id=221, actor_user_id=100),
        name="pause-before-freeze",
    )
    await asyncio.wait_for(transition_committed.wait(), timeout=1)
    tick_task = asyncio.create_task(
        stack.timers.tick(now=baseline + timedelta(seconds=180)),
        name="tick-during-pause",
    )
    await asyncio.wait_for(timer_lock_attempted.wait(), timeout=1)
    assert not tick_task.done()

    allow_timer_freeze.set()
    _, notices = await asyncio.gather(pause_task, tick_task)

    assert notices == ()
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.PAUSED
    active = await stack.timer_repository.list_active(stack.session_id)
    assert len(active) == 1
    assert active[0].status is TimerStatus.PAUSED
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skip_actions_complete_before_a_later_pause_freezes_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed skip cannot run stale timer actions after a later PAUSED state."""

    stack = await _stack(tmp_path, guild_id=225)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=frozenset(ConsentKind),
        is_host=True,
    )
    item = await stack.agenda.add(
        session_id=stack.session_id,
        title="锁顺序议题",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(3):
        await stack.runtime.advance(stack.session_id)
    await stack.agenda.activate_next(session_id=stack.session_id)

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    original_skip = stack.runtime.skip
    skip_committed = asyncio.Event()
    allow_skip_actions = asyncio.Event()

    async def gated_skip(
        session_id: str,
        *,
        actor_user_id: int,
        source: str,
    ) -> RuntimeResult:
        result = await original_skip(
            session_id,
            actor_user_id=actor_user_id,
            source=source,
        )
        skip_committed.set()
        await allow_skip_actions.wait()
        return result

    monkeypatch.setattr(stack.runtime, "skip", gated_skip)
    pause_lock_attempted = asyncio.Event()

    class ObservedGuildLock(asyncio.Lock):
        async def acquire(self) -> bool:
            task = asyncio.current_task()
            if task is not None and task.get_name() == "pause-after-skip":
                pause_lock_attempted.set()
            return await super().acquire()

    stack.controller._consent_locks[225] = ObservedGuildLock()
    skip_task = asyncio.create_task(
        stack.controller.skip(guild_id=225, actor_user_id=100),
        name="gated-skip",
    )
    await asyncio.wait_for(skip_committed.wait(), timeout=1)
    pause_task = asyncio.create_task(
        stack.controller.pause(guild_id=225, actor_user_id=100),
        name="pause-after-skip",
    )
    await asyncio.wait_for(pause_lock_attempted.wait(), timeout=1)

    allow_skip_actions.set()
    await asyncio.gather(skip_task, pause_task)

    session = await stack.sessions.get(stack.session_id)
    assert session.state is MeetingState.PAUSED
    active = await stack.timer_repository.list_active(stack.session_id)
    assert len(active) == 1
    assert active[0].timer_type == f"FLOW:AGENDA:{item.id}"
    assert active[0].status is TimerStatus.PAUSED
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recovery_freezes_timer_after_crash_between_pause_state_and_action(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=222)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    await stack.timers.start(
        session_id=stack.session_id,
        timer_type="PAUSE-CRASH",
        duration_seconds=120,
        now=baseline,
    )

    paused = await stack.runtime.pause(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert paused.session.state is MeetingState.PAUSED
    active_before = await stack.timer_repository.list_active(stack.session_id)
    assert active_before[0].status is TimerStatus.RUNNING

    recovered = await stack.controller.reconcile_timer_handoffs(session_ids={stack.session_id})

    assert recovered[-1].session.state is MeetingState.PAUSED
    active_after = await stack.timer_repository.list_active(stack.session_id)
    assert len(active_after) == 1
    assert active_after[0].status is TimerStatus.PAUSED
    assert await stack.timers.tick(now=baseline + timedelta(seconds=180)) == ()
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert (
        sum(
            event.payload
            == {
                "transition_event_id": paused.event.id,
                "action": "freeze_timers",
            }
            for event in completions
        )
        == 1
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_durable_replay_rejects_an_expired_timer_replaced_by_same_type(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=224)
    baseline = datetime.now(UTC)
    first = await stack.timers.start(
        session_id=stack.session_id,
        timer_type="FLOW:AGENDA:reentered-item",
        duration_seconds=10,
        now=baseline,
    )
    first_notice = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert first_notice.timer_id == first.id
    second = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=first.timer_type,
        duration_seconds=10,
        now=baseline + timedelta(seconds=20),
    )
    assert second.id != first.id

    assert not await stack.timers.is_latest_timer_identity(first_notice)
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()

    second_notice = (await stack.timers.tick(now=baseline + timedelta(seconds=31)))[-1]
    assert await stack.timers.is_latest_timer_identity(second_notice)
    assert await stack.timers.durable_expiry_notices(stack.session_id) == (second_notice,)
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_stop_terminal_actions_and_expiry_handoff_recover_exactly_once(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=253)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type="STOP-CRASH",
        duration_seconds=10,
        now=baseline,
    )
    await stack.runtime.pause(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    stopped = await stack.runtime.stop(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert stopped.event is not None
    dropped = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert dropped.timer_id == timer.id
    attempts: list[str] = []
    failed_once = False

    async def fail_finalise_once(_session: object, action: str) -> None:
        nonlocal failed_once
        attempts.append(action)
        if action == "finalise_capture" and not failed_once:
            failed_once = True
            raise RuntimeError("synthetic terminal finalise failure")

    stack.controller.set_runtime_action_handler(fail_finalise_once)
    with pytest.raises(RuntimeError, match="synthetic terminal finalise failure"):
        await stack.controller.recover_automatic_flows(session_ids=set())
    assert await stack.timers.durable_expiry_notices(stack.session_id) == (dropped,)
    assert (
        await stack.jobs.latest_for_session(
            stack.session_id,
            job_type="POST_MEETING",
        )
        is None
    )

    recovered = await stack.controller.recover_automatic_flows(session_ids=set())

    assert recovered[-1].session.state is MeetingState.STOPPED
    assert attempts == [
        "cancel_timers",
        "speak_closing",
        "finalise_capture",
        "finalise_capture",
        "trigger_post_meeting_job",
    ]
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    for action in (
        "cancel_timers",
        "speak_closing",
        "finalise_capture",
        "trigger_post_meeting_job",
    ):
        assert (
            sum(
                event.payload
                == {
                    "transition_event_id": stopped.event.id,
                    "action": action,
                }
                for event in completions
            )
            == 1
        )
    job = await stack.jobs.latest_for_session(
        stack.session_id,
        job_type="POST_MEETING",
    )
    assert job is not None
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    assert await stack.controller.recover_automatic_flows(session_ids=set()) == ()
    assert (await stack.jobs.latest_for_session(stack.session_id, job_type="POST_MEETING")) == job
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_published_notification_recovers_after_terminal_transition_failure(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=254)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(12):
        await stack.runtime.advance(stack.session_id)
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.POST_PROCESSING
    published = await stack.runtime.post_processing_completed(stack.session_id)
    assert published.event is not None
    attempts = 0

    async def fail_publish_once(_session: object, action: str) -> None:
        nonlocal attempts
        assert action == "publish_completion"
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic publish notification failure")

    stack.controller.set_runtime_action_handler(fail_publish_once)
    with pytest.raises(RuntimeError, match="synthetic publish notification failure"):
        await stack.controller.recover_automatic_flows(session_ids=set())

    recovered = await stack.controller.recover_automatic_flows(session_ids=set())

    assert recovered[-1].session.state is MeetingState.PUBLISHED
    assert attempts == 2
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert (
        sum(
            event.payload
            == {
                "transition_event_id": published.event.id,
                "action": "publish_completion",
            }
            for event in completions
        )
        == 1
    )
    assert await stack.controller.recover_automatic_flows(session_ids=set()) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy_payload",
    [
        {},
        {"runtime_version": 1, "runtime_actions": ["trigger_post_meeting_job"]},
        {"runtime_version": 2},
        {"runtime_version": 2, "runtime_actions": "not-an-action-list"},
        {"runtime_version": 2, "runtime_actions": []},
        {"runtime_version": 2, "runtime_actions": ["trigger_post_meeting_job"]},
        {"runtime_version": 2, "runtime_actions": ["unknown_terminal_action"]},
    ],
    ids=(
        "v1",
        "v1-with-actions",
        "v2-missing-actions",
        "v2-malformed-actions",
        "v2-empty-actions",
        "v2-subset-actions",
        "v2-unknown-actions",
    ),
)
async def test_terminal_recovery_does_not_infer_actions_without_a_v2_batch(
    tmp_path: Path,
    legacy_payload: dict[str, object],
) -> None:
    stack = await _stack(tmp_path, guild_id=260)
    baseline = datetime.now(UTC)
    timer = await stack.timers.start(
        session_id=stack.session_id,
        timer_type="LEGACY-TERMINAL",
        duration_seconds=10,
        now=baseline,
    )
    current = await stack.sessions.get(stack.session_id)
    stopped = await stack.sessions.transition_with_event(
        session_id=stack.session_id,
        expected_state=current.state,
        expected_resume_state=current.resume_state,
        expected_sequence=current.next_sequence,
        target=MeetingState.STOPPED,
        actor_user_id=100,
        source="LEGACY_RUNTIME",
        event_type=MeetingEventType.MEETING_STOPPED.value,
        payload=legacy_payload,
    )
    assert stopped.event is not None
    assert await stack.sessions.list_terminal_reconcilable() == ()
    notice = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert notice.timer_id == timer.id
    reconcilable = await stack.sessions.list_terminal_reconcilable()
    assert tuple(candidate.id for candidate in reconcilable) == (stack.session_id,)
    assert (await stack.runtime.recover_actions(stack.session_id)).actions == ()
    attempted_actions: list[str] = []

    async def record_action(_session: object, action: str) -> None:
        attempted_actions.append(action)

    stack.controller.set_runtime_action_handler(record_action)

    assert await stack.controller.recover_automatic_flows(session_ids=set()) == ()
    assert attempted_actions == []
    assert await stack.jobs.latest_for_session(stack.session_id, job_type="POST_MEETING") is None
    assert await stack.timers.uncompleted_expiry_notices(stack.session_id) == ()
    assert await stack.sessions.list_terminal_reconcilable() == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_published_transition_is_not_treated_as_a_pending_action_batch(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=261)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.runtime.start(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    for _ in range(12):
        await stack.runtime.advance(stack.session_id)
    current = await stack.sessions.get(stack.session_id)
    assert current.state is MeetingState.POST_PROCESSING
    published = await stack.sessions.transition_with_event(
        session_id=stack.session_id,
        expected_state=current.state,
        expected_resume_state=current.resume_state,
        expected_sequence=current.next_sequence,
        target=MeetingState.PUBLISHED,
        actor_user_id=None,
        source="LEGACY_RUNTIME",
        event_type=MeetingEventType.STATE_ADVANCED.value,
        payload={},
    )
    assert published.event is not None
    assert await stack.sessions.list_terminal_reconcilable() == ()
    assert (await stack.runtime.recover_actions(stack.session_id)).actions == ()
    assert await stack.controller.recover_automatic_flows(session_ids=set()) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_nonterminal_entry_and_resume_keep_their_v1_recovery_actions(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=262)
    current = await stack.sessions.get(stack.session_id)
    entered = await stack.sessions.transition_with_event(
        session_id=stack.session_id,
        expected_state=current.state,
        expected_resume_state=current.resume_state,
        expected_sequence=current.next_sequence,
        target=MeetingState.AWAITING_PARTICIPANTS,
        actor_user_id=100,
        source="LEGACY_RUNTIME",
        event_type=MeetingEventType.MEETING_INITIALISED.value,
        payload={"runtime_version": 1},
    )
    paused = await stack.sessions.transition_with_event(
        session_id=stack.session_id,
        expected_state=entered.session.state,
        expected_resume_state=entered.session.resume_state,
        expected_sequence=entered.session.next_sequence,
        target=MeetingState.PAUSED,
        actor_user_id=100,
        source="LEGACY_RUNTIME",
        event_type=MeetingEventType.MEETING_PAUSED.value,
        payload={"runtime_version": 1},
    )
    resumed = await stack.sessions.transition_with_event(
        session_id=stack.session_id,
        expected_state=paused.session.state,
        expected_resume_state=paused.session.resume_state,
        expected_sequence=paused.session.next_sequence,
        target=MeetingState.AWAITING_PARTICIPANTS,
        actor_user_id=100,
        source="LEGACY_RUNTIME",
        event_type=MeetingEventType.MEETING_RESUMED.value,
        payload={"runtime_version": 1},
    )
    assert resumed.event is not None

    batches = await stack.runtime.recover_action_batches(stack.session_id)

    assert tuple(batch.actions for batch in batches) == (
        ("publish_host_panel", "request_consent"),
        ("resume_timers",),
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terminal_cleanup_acknowledges_a_superseded_expiry_fact(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=255)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    baseline = datetime.now(UTC)
    first = await stack.timers.start(
        session_id=stack.session_id,
        timer_type="SUPERSEDED-AT-STOP",
        duration_seconds=10,
        now=baseline,
    )
    first_notice = (await stack.timers.tick(now=baseline + timedelta(seconds=11)))[-1]
    assert first_notice.timer_id == first.id
    second = await stack.timers.start(
        session_id=stack.session_id,
        timer_type=first.timer_type,
        duration_seconds=60,
        now=baseline + timedelta(seconds=20),
    )
    assert second.id != first.id
    assert await stack.timers.durable_expiry_notices(stack.session_id) == ()
    await stack.runtime.stop(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )

    await stack.controller.recover_automatic_flows(session_ids=set())

    handoffs = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
    )
    assert sum(event.payload.get("timer_id") == first.id for event in handoffs) == 1
    assert await stack.sessions.list_terminal_reconcilable() == ()
    assert await stack.controller.recover_automatic_flows(session_ids=set()) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("vote_kind", ["public", "agenda"])
async def test_terminal_action_recovery_closes_the_exact_active_vote(
    tmp_path: Path,
    vote_kind: str,
) -> None:
    guild_id = 257 if vote_kind == "public" else 258
    stack = await _stack(tmp_path, guild_id=guild_id)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    agenda_item_id: str | None = None
    if vote_kind == "agenda":
        item = await stack.agenda.add(
            session_id=stack.session_id,
            title="停止时关闭议程票",
            duration_minutes=1,
            requires_vote=True,
            vote_content="是否关闭",
            vote_duration_seconds=60,
            actor_user_id=100,
            source="SLASH_COMMAND",
        )
        await stack.runtime.start(
            stack.session_id,
            actor_user_id=100,
            source="SLASH_COMMAND",
        )
        for _ in range(7):
            await stack.runtime.advance(stack.session_id)
        await stack.agenda.activate_next(session_id=stack.session_id)
        agenda_item_id = item.id
    vote = await stack.voting.open(
        session_id=stack.session_id,
        title="停止时必须审计关闭",
        duration_seconds=60,
        actor_user_id=100,
        source="TEST",
        agenda_item_id=agenda_item_id,
    )
    stopped = await stack.runtime.stop(
        stack.session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert stopped.event is not None

    await stack.controller.recover_automatic_flows(session_ids=set())

    assert (await stack.voting.get(vote.id)).status is VoteStatus.CLOSED
    assert await stack.timer_repository.list_active(stack.session_id) == ()
    closed = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.VOTE_CLOSED.value,),
    )
    assert sum(event.payload.get("vote_id") == vote.id for event in closed) == 1
    completions = await stack.events.list_types(
        stack.session_id,
        (MeetingEventType.RUNTIME_ACTION_COMPLETED.value,),
    )
    assert (
        sum(
            event.payload
            == {
                "transition_event_id": stopped.event.id,
                "action": "cancel_timers",
            }
            for event in completions
        )
        == 1
    )
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_agenda_expiry_cannot_advance_while_next_owns_guild_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=223)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=frozenset(ConsentKind),
        is_host=True,
    )
    first = await stack.agenda.add(
        session_id=stack.session_id,
        title="第一项",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    second = await stack.agenda.add(
        session_id=stack.session_id,
        title="第二项",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.controller.start(guild_id=223, actor_user_id=100)
    opening_timer = (await stack.timer_repository.list_active(stack.session_id))[0]
    opening = await stack.timers.tick(now=opening_timer.deadline_at + timedelta(seconds=1))
    await stack.controller.handle_timer_notice(opening[-1])
    first_timer = next(
        timer
        for timer in await stack.timer_repository.list_active(stack.session_id)
        if timer.timer_type == f"FLOW:AGENDA:{first.id}"
    )
    stale_notice = (await stack.timers.tick(now=first_timer.deadline_at + timedelta(seconds=1)))[-1]

    next_owns_lock = asyncio.Event()
    allow_next = asyncio.Event()
    original_cancel = stack.timers.cancel

    async def gated_cancel(session_id: str) -> int:
        next_owns_lock.set()
        await allow_next.wait()
        return await original_cancel(session_id)

    monkeypatch.setattr(stack.timers, "cancel", gated_cancel)
    timer_lock_attempted = asyncio.Event()

    class ObservedGuildLock(asyncio.Lock):
        async def acquire(self) -> bool:
            task = asyncio.current_task()
            if task is not None and task.get_name() == "stale-timer-notice":
                timer_lock_attempted.set()
            return await super().acquire()

    stack.controller._consent_locks[223] = ObservedGuildLock()
    next_task = asyncio.create_task(
        stack.controller.next(guild_id=223, actor_user_id=100),
        name="next-holds-guild-lock",
    )
    await asyncio.wait_for(next_owns_lock.wait(), timeout=1)
    timer_task = asyncio.create_task(
        stack.controller.handle_timer_notice(stale_notice),
        name="stale-timer-notice",
    )
    await asyncio.wait_for(timer_lock_attempted.wait(), timeout=1)
    assert not timer_task.done()

    allow_next.set()
    await asyncio.gather(next_task, timer_task)

    session = await stack.sessions.get(stack.session_id)
    assert session.state is MeetingState.DISCUSSION_READY
    agenda = await stack.agenda.list(stack.session_id)
    assert next(item for item in agenda if item.id == first.id).status is AgendaStatus.COMPLETED
    assert next(item for item in agenda if item.id == second.id).status is AgendaStatus.ACTIVE
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timed_agendas_loop_automatically_and_publish_after_worker(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, guild_id=4)
    all_consents = frozenset(ConsentKind)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=all_consents,
        is_host=True,
    )
    first = await stack.agenda.add(
        session_id=stack.session_id,
        title="第一项",
        duration_minutes=1,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    second = await stack.agenda.add(
        session_id=stack.session_id,
        title="Second item",
        duration_minutes=1,
        language="en",
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    actions: list[str] = []

    async def record_action(_session: object, action: str) -> None:
        actions.append(action)

    stack.controller.set_runtime_action_handler(record_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    opening = await stack.controller.start(guild_id=4, actor_user_id=100)
    assert opening.session.state is MeetingState.OPENING
    assert "speak_welcome" in actions

    baseline = datetime.now(UTC)
    opening_notices = await stack.timers.tick(now=baseline + timedelta(seconds=11))
    assert len(opening_notices) == 1
    await stack.controller.handle_timer_notice(opening_notices[0])
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.DISCUSSION_READY
    assert (
        next(
            item
            for item in await stack.agenda.list(stack.session_id)
            if item.status is AgendaStatus.ACTIVE
        ).id
        == first.id
    )
    active_before_recovery = await stack.timer_repository.list_active(stack.session_id)
    recovered = await stack.controller.recover_automatic_flows(
        session_ids={stack.session_id},
    )
    active_after_recovery = await stack.timer_repository.list_active(stack.session_id)
    assert recovered[-1].session.state is MeetingState.DISCUSSION_READY
    assert [timer.id for timer in active_after_recovery] == [
        timer.id for timer in active_before_recovery
    ]

    first_notices = await stack.timers.tick(now=baseline + timedelta(seconds=70))
    await stack.controller.handle_timer_notice(first_notices[-1])
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.DISCUSSION_READY
    assert (
        next(
            item
            for item in await stack.agenda.list(stack.session_id)
            if item.status is AgendaStatus.ACTIVE
        ).id
        == second.id
    )

    second_notices = await stack.timers.tick(now=baseline + timedelta(seconds=140))
    await stack.controller.handle_timer_notice(second_notices[-1])
    assert (await stack.sessions.get(stack.session_id)).state is MeetingState.POST_PROCESSING
    assert all(
        item.status is AgendaStatus.COMPLETED for item in await stack.agenda.list(stack.session_id)
    )
    assert actions.count("speak_current_agenda_item") == 2
    assert "speak_closing" in actions

    job = await stack.jobs.claim(worker_id="test-worker", lease_seconds=60)
    assert job is not None
    await stack.jobs.complete(job_id=job.id, worker_id="test-worker")
    monitored = await stack.controller.monitor_post_processing_jobs()
    assert monitored[-1].session.state is MeetingState.PUBLISHED

    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_untimed_agenda_waits_for_host_before_closing(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=5)
    await stack.participants.join(
        session_id=stack.session_id,
        user_id=100,
        inherited_consents=frozenset(ConsentKind),
        is_host=True,
    )
    await stack.agenda.add(
        session_id=stack.session_id,
        title="开放讨论",
        duration_minutes=None,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )

    async def ignore_action(_session: object, _action: str) -> None:
        return None

    stack.controller.set_runtime_action_handler(ignore_action)
    await stack.runtime.initialise(stack.session_id, actor_user_id=100)
    await stack.controller.start(guild_id=5, actor_user_id=100)
    notices = await stack.timers.tick(now=datetime.now(UTC) + timedelta(seconds=11))
    await stack.controller.handle_timer_notice(notices[-1])

    waiting = await stack.sessions.get(stack.session_id)
    assert waiting.state is MeetingState.DISCUSSION_READY
    assert await stack.timer_repository.list_active(stack.session_id) == ()

    advanced = await stack.controller.next(guild_id=5, actor_user_id=100)
    assert advanced.session.state is MeetingState.POST_PROCESSING
    await stack.database.close()
