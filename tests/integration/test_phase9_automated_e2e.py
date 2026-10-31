from pathlib import Path

import pytest

from lingxu.domain.enums import JobStatus, MeetingState
from lingxu.domain.errors import SessionNotFoundError
from lingxu.domain.models import RealtimeUtteranceRecord
from lingxu.meeting.components import (
    AgendaService,
    SpeakingQueueService,
    TimerService,
    VotingService,
)
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.intents import RuleIntent, RuleIntentResolver
from lingxu.meeting.runtime import MeetingRuntime
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
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import ArtifactRepository, TranscriptRepository
from lingxu.utils import new_ulid
from lingxu.worker.pipeline import POST_MEETING_JOB, PostMeetingPipeline
from lingxu.worker.service import WorkerService


class NoEcho:
    def is_echo(self, transcript: str) -> bool:
        return False


class FakeDiscordPublisher:
    def __init__(self) -> None:
        self.calls = 0
        self.file_names: tuple[str, ...] = ()

    async def publish(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        content: str,
        files: tuple[Path, ...],
    ) -> tuple[int, str]:
        self.calls += 1
        self.file_names = tuple(path.name for path in files)
        assert guild_id == 90
        assert channel_id == 9000
        assert "未使用 LLM" in content
        return 9000, "1002"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_product_stage1_automated_e2e_without_external_discord_or_dave(
    tmp_path: Path,
) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'e2e.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    events = MeetingEventStore(database.sessions)
    participants = ParticipantRepository(database.sessions)
    files = SessionFiles(tmp_path / "sessions")
    session_service = MeetingSessionService(sessions, files)
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
    jobs = ProcessingJobRepository(database.sessions)
    runtime = MeetingRuntime(
        sessions=sessions,
        events=events,
        snapshots=MeetingSnapshotStore(database.sessions),
    )
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
        jobs=jobs,
    )

    session = await controller.create(
        guild_id=90,
        host_user_id=100,
        publication_channel_id=9000,
    )
    for user_id in (200, 300):
        await controller.participant_join(guild_id=90, user_id=user_id)
    for user_id in (100, 200, 300):
        await controller.consent_all(guild_id=90, user_id=user_id)
    await controller.agenda_add(
        guild_id=90,
        actor_user_id=100,
        title="First topic",
    )
    await controller.agenda_add(
        guild_id=90,
        actor_user_id=100,
        title="Second topic",
        requires_vote=True,
        vote_content="Approve the second topic?",
        vote_duration_seconds=60,
    )

    started = await controller.start(guild_id=90, actor_user_id=100)
    assert started.session.state is MeetingState.OPENING
    assert started.actions == ("speak_welcome",)
    paused = await controller.pause(guild_id=90, actor_user_id=100)
    assert paused.actions == ("freeze_timers",)
    resumed = await controller.resume(guild_id=90, actor_user_id=100)
    assert resumed.actions == ("resume_timers",)
    overview = await controller.next(guild_id=90, actor_user_id=100)
    assert overview.actions == ("speak_agenda",)
    await controller.next(guild_id=90, actor_user_id=100)
    backed = await controller.back(guild_id=90, actor_user_id=100)
    assert backed.session.state is MeetingState.AGENDA_OVERVIEW
    await controller.next(guild_id=90, actor_user_id=100)

    resolver = RuleIntentResolver(NoEcho())
    chinese = resolver.resolve("申请发言", actor_user_id=200, host_user_id=100)
    english = resolver.resolve("request to speak", actor_user_id=300, host_user_id=100)
    assert chinese is not None and chinese.intent is RuleIntent.REQUEST_SPEAK
    assert english is not None and english.intent is RuleIntent.REQUEST_SPEAK
    assert await controller.speak(guild_id=90, user_id=chinese.actor_user_id, source=chinese.source)
    assert await controller.speak(guild_id=90, user_id=english.actor_user_id, source=english.source)
    assert await controller.speaker_next(guild_id=90, actor_user_id=100) == 200
    await controller.speaker_finish(guild_id=90, actor_user_id=200)
    assert await controller.speaker_next(guild_id=90, actor_user_id=100) == 300

    utterances = RealtimeUtteranceRepository(database.sessions)
    for user_id, start, raw, normalised in (
        (200, 0, "<|zh|>中文发言", "中文发言"),
        (300, 1000, "<|en|>English speech", "English speech"),
        (200, 2000, "<|zh|>中英 mixed sentence", "中英 mixed sentence"),
    ):
        await utterances.add(
            RealtimeUtteranceRecord(
                id=new_ulid(),
                session_id=session.id,
                user_id=user_id,
                started_ms=start,
                ended_ms=start + 800,
                raw_text=raw,
                normalised_text=normalised,
            )
        )
    await timers.start(
        session_id=session.id,
        timer_type="SPEAKER",
        duration_seconds=180,
    )
    vote = await controller.vote_open(
        guild_id=90,
        actor_user_id=100,
        title="Approve",
        duration_seconds=60,
    )
    await controller.vote_cast(guild_id=90, user_id=200, option_id="yes")
    await controller.vote_cast(guild_id=90, user_id=300, option_id="no")
    outcome = await controller.vote_close(guild_id=90, actor_user_id=100)
    assert outcome.counts == {"yes": 1, "no": 1, "abstain": 0}
    assert outcome.vote.id == vote.id

    stopped = await controller.stop(guild_id=90, actor_user_id=100)
    assert stopped.session.state is MeetingState.STOPPED
    post_job = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="post-meeting-v1",
    )
    publisher = FakeDiscordPublisher()
    artifacts = ArtifactRepository(database.sessions)
    pipeline = PostMeetingPipeline(
        sessions=sessions,
        events=events,
        participants=participants,
        agenda=AgendaRepository(database.sessions),
        utterances=utterances,
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=files,
        transcriber=None,
        publisher=publisher,
        retention_hours=24,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="phase9-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(post_job.id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert publisher.calls == 1
    assert publisher.file_names == ("minutes.md",)
    meeting_dir = files.directory(session.id) / "meeting"
    for internal_name in (
        "minutes.json",
        "transcript.txt",
        "canonical_transcript.json",
        "evidence.json",
    ):
        assert (meeting_dir / internal_name).is_file()
    assert not await WorkerService(
        jobs=jobs,
        worker_id="phase9-retry",
        pipeline=pipeline,
    ).run_once()
    assert publisher.calls == 1
    assert (await runtime.recover(session.id)).session.state is MeetingState.STOPPED

    await controller.delete(guild_id=90, actor_user_id=100)
    with pytest.raises(SessionNotFoundError):
        await sessions.get(session.id)
    assert not files.directory(session.id).exists()
    await database.close()
