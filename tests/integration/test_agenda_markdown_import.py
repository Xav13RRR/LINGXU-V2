from dataclasses import dataclass
from pathlib import Path

import pytest

from lingxu.domain.enums import AgendaFieldOrigin, AgendaImportStatus, MeetingState
from lingxu.domain.errors import AgendaImportError
from lingxu.domain.models import AgendaImportItem
from lingxu.meeting.agenda_import import (
    AgendaImportChunk,
    AgendaImportService,
    AgendaSourceBlock,
)
from lingxu.meeting.components import (
    AgendaService,
    SpeakingQueueService,
    TimerService,
    VotingService,
)
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.runtime import MeetingRuntime
from lingxu.storage.agenda_import_repository import AgendaImportRepository
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
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import AgendaImportRow, MeetingAgendaItemRow


@dataclass
class FakeAgendaProvider:
    confidence: float = 0.94
    calls: int = 0

    async def extract(
        self,
        *,
        blocks: tuple[AgendaSourceBlock, ...],
        default_duration_minutes: int,
        default_vote_seconds: int,
    ) -> AgendaImportChunk:
        self.calls += 1
        assert blocks
        assert default_duration_minutes == 15
        assert default_vote_seconds == 60
        return AgendaImportChunk(
            (
                AgendaImportItem(
                    title="项目进展",
                    title_origin=AgendaFieldOrigin.INFERRED,
                    source_block_ids=(blocks[0].id,),
                    duration_minutes=15,
                    duration_origin=AgendaFieldOrigin.DEFAULTED,
                    requires_vote=False,
                    vote_origin=AgendaFieldOrigin.INFERRED,
                    vote_content=None,
                    vote_content_origin=None,
                    vote_duration_seconds=None,
                    vote_duration_origin=None,
                    language="zh",
                    language_origin=AgendaFieldOrigin.INFERRED,
                    confidence=self.confidence,
                ),
                AgendaImportItem(
                    title="发布审批",
                    title_origin=AgendaFieldOrigin.NORMALIZED,
                    source_block_ids=(blocks[-1].id,),
                    duration_minutes=10,
                    duration_origin=AgendaFieldOrigin.NORMALIZED,
                    requires_vote=True,
                    vote_origin=AgendaFieldOrigin.INFERRED,
                    vote_content="是否批准发布?",
                    vote_content_origin=AgendaFieldOrigin.INFERRED,
                    vote_duration_seconds=60,
                    vote_duration_origin=AgendaFieldOrigin.DEFAULTED,
                    language="zh",
                    language_origin=AgendaFieldOrigin.INFERRED,
                    confidence=self.confidence,
                ),
            )
        )


@dataclass
class ImportStack:
    database: Database
    sessions: MeetingSessionRepository
    events: MeetingEventStore
    agenda: AgendaRepository
    imports: AgendaImportRepository
    service: AgendaImportService
    controller: MeetingController
    session_id: str
    provider: FakeAgendaProvider


async def _stack(tmp_path: Path, *, confidence: float = 0.94) -> ImportStack:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'agenda-import.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session_service = MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions"))
    session = await session_service.create(guild_id=42, creator_user_id=100)
    session = await sessions.set_setup_channel(session_id=session.id, channel_id=500)
    events = MeetingEventStore(database.sessions)
    runtime = MeetingRuntime(
        sessions=sessions,
        events=events,
        snapshots=MeetingSnapshotStore(database.sessions),
        snapshot_interval=50,
    )
    await runtime.initialise(session.id, actor_user_id=100)
    imports = AgendaImportRepository(database.sessions)
    provider = FakeAgendaProvider(confidence=confidence)
    service = AgendaImportService(
        sessions=sessions,
        repository=imports,
        provider=provider,
        enabled=True,
        model_name="qwen3.5:9b",
        model_digest="a" * 64,
    )
    participants = ParticipantRepository(database.sessions)
    agenda = AgendaRepository(database.sessions)
    timer_repository = TimerRepository(database.sessions)
    voting = VotingService(
        votes=VoteRepository(database.sessions),
        timers=timer_repository,
        participants=participants,
        events=events,
    )
    controller = MeetingController(
        sessions=sessions,
        events=events,
        session_service=session_service,
        runtime=runtime,
        participants=participants,
        guild_consents=GuildConsentRepository(database.sessions),
        consent_changes=ConsentChangeRepository(database.sessions),
        agenda=AgendaService(agenda, events),
        speaking=SpeakingQueueService(participants=participants, events=events),
        timers=TimerService(
            timers=timer_repository,
            events=events,
            vote_closer=voting,
        ),
        voting=voting,
        agenda_import_guard=service,
    )
    return ImportStack(
        database=database,
        sessions=sessions,
        events=events,
        agenda=agenda,
        imports=imports,
        service=service,
        controller=controller,
        session_id=session.id,
        provider=provider,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_import_is_host_channel_scoped_idempotent_and_atomically_ready(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    markdown = "# 项目进展\n- 汇报近期状态\n- 发布审批, 讨论十分钟并进行表决"
    submission = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=700,
        attachment_id=800,
        filename="agenda.md",
        payload=markdown.encode(),
    )
    assert submission.created
    assert submission.record.status is AgendaImportStatus.PROCESSING
    assert await stack.service.has_processing(stack.session_id)
    with pytest.raises(ValueError, match="仍在处理中"):
        await stack.controller.start(guild_id=42, actor_user_id=100)

    result = await stack.service.process(submission.record.id)
    assert result.record.status is AgendaImportStatus.READY
    assert [item.title for item in result.agenda] == ["项目进展", "发布审批"]
    assert result.record.result == {
        "item_count": 2,
        "scheduled_seconds": 1560,
        "vote_count": 1,
        "defaulted_field_count": 2,
        "inferred_field_count": 5,
        "chunk_count": 1,
        "policy_version": "P2-v2",
    }
    assert "准备完毕" in result.ready_message
    assert not await stack.service.has_processing(stack.session_id)

    rows = await stack.agenda.list(stack.session_id)
    assert len(rows) == 2
    assert rows[0].duration_minutes == 15
    assert rows[1].requires_vote
    assert rows[1].vote_duration_seconds == 60
    async with stack.database.sessions() as db:
        import_row = await db.get(AgendaImportRow, submission.record.id)
        agenda_row = await db.get(MeetingAgendaItemRow, rows[0].id)
        assert import_row is not None
        assert import_row.source_text == markdown
        assert agenda_row is not None
        assert agenda_row.payload_json["import_provenance"]["title_origin"] == "INFERRED"

    duplicate = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=700,
        attachment_id=800,
        filename="agenda.md",
        payload=markdown.encode(),
    )
    assert not duplicate.created
    assert duplicate.record.status is AgendaImportStatus.READY
    assert len(await stack.agenda.list(stack.session_id)) == 2

    event_types = [event.event_type for event in await stack.events.list(stack.session_id)]
    assert event_types.count("AGENDA_IMPORT_ACCEPTED") == 1
    assert event_types.count("AGENDA_IMPORT_COMMITTED") == 1
    started = await stack.controller.start(guild_id=42, actor_user_id=100)
    assert started.session.state is MeetingState.CONSENT_CHECK
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_structured_markdown_compiles_templates_without_provider_or_name_inference(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    speaker_a = 711_201
    speaker_b = 711_202
    markdown = """# 可读说明
```lingxu-agenda
{"schema_version":1,"identity_map":{"正方":"<@711201>"},"items":[{"template":"ALTERNATING_SIDES","title":"正反陈述","duration_minutes":8,"turn_seconds":90,"members":["正方","<@711202>"],"followup":"SUPPLEMENT_TURNS","enforce_mute":true},{"template":"PRESENTATION_Q_AND_A","title":"产品演讲","duration_minutes":20,"presenters":["正方"],"prep_seconds":0}]}
```
"""
    submission = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=710,
        attachment_id=810,
        filename="structured.md",
        payload=markdown.encode(),
        validated_identity_user_ids=(speaker_a, speaker_b),
    )
    assert submission.record.model_name == "DETERMINISTIC_JSON"
    assert submission.record.validated_identity_user_ids == (speaker_a, speaker_b)

    result = await stack.service.process(submission.record.id)

    assert stack.provider.calls == 0
    assert result.record.result["import_mode"] == "DETERMINISTIC_JSON"
    assert result.record.result["inferred_field_count"] == 0
    assert [item.speaking_policy.mode.value for item in result.agenda] == [
        "ORDERED",
        "PRESENTATION",
    ]
    assert [turn.user_id for turn in result.agenda[0].speaking_policy.turns] == [
        speaker_a,
        speaker_b,
    ]
    assert result.agenda[0].speaking_policy.mute_enforcement_requested
    assert result.agenda[1].speaking_policy.presenter_user_ids == (speaker_a,)
    assert result.agenda[1].speaking_policy.prep_seconds == 0
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_structured_markdown_requires_host_identity_map_for_natural_names(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    markdown = """```lingxu-agenda
{"schema_version":1,"identity_map":{},"items":[{"template":"PRESENTATION","presenters":["Alice"]}]}
```"""
    submission = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=711,
        attachment_id=811,
        filename="unmapped.md",
        payload=markdown.encode(),
    )
    with pytest.raises(AgendaImportError, match="identity_map"):
        await stack.service.process(submission.record.id)
    assert await stack.agenda.list(stack.session_id) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_malformed_structured_marker_fails_closed_without_model_fallback(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path)
    with pytest.raises(AgendaImportError, match="未正确闭合"):
        await stack.service.submit(
            guild_id=42,
            actor_user_id=100,
            channel_id=500,
            message_id=712,
            attachment_id=812,
            filename="broken.md",
            payload=b'```lingxu-agenda\n{"schema_version":1}',
        )
    assert stack.provider.calls == 0
    assert await stack.agenda.list(stack.session_id) == ()
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_low_confidence_fails_without_partial_agenda_and_can_be_resubmitted(
    tmp_path: Path,
) -> None:
    stack = await _stack(tmp_path, confidence=0.2)
    submission = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=701,
        attachment_id=801,
        filename="messy.md",
        payload="可能讨论项目, 也可能审批发布".encode(),
    )

    with pytest.raises(AgendaImportError, match="无法可靠解析"):
        await stack.service.process(submission.record.id)
    failed = await stack.imports.get(submission.record.id)
    assert failed.status is AgendaImportStatus.FAILED
    assert await stack.agenda.list(stack.session_id) == ()
    assert not await stack.service.has_processing(stack.session_id)
    assert "AGENDA_IMPORT_FAILED" in {
        event.event_type for event in await stack.events.list(stack.session_id)
    }

    resubmission = await stack.service.submit(
        guild_id=42,
        actor_user_id=100,
        channel_id=500,
        message_id=702,
        attachment_id=802,
        filename="revised.md",
        payload="# 项目讨论\n- 发布审批".encode(),
    )
    assert resubmission.created
    await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_import_rejects_non_host_wrong_channel_and_existing_agenda(tmp_path: Path) -> None:
    stack = await _stack(tmp_path)
    common = {
        "guild_id": 42,
        "message_id": 703,
        "attachment_id": 803,
        "filename": "agenda.md",
        "payload": b"# Agenda",
    }
    with pytest.raises(AgendaImportError, match="主持人"):
        await stack.service.submit(actor_user_id=200, channel_id=500, **common)
    with pytest.raises(AgendaImportError, match="创建本次会议的频道"):
        await stack.service.submit(actor_user_id=100, channel_id=501, **common)

    await stack.agenda.add(
        session_id=stack.session_id,
        title="Manual",
        duration_minutes=10,
        requires_vote=False,
    )
    with pytest.raises(AgendaImportError, match="已有议程"):
        await stack.service.submit(actor_user_id=100, channel_id=500, **common)
    await stack.database.close()
