"""Shared synthetic Worker integration fixtures.

Keeping these helpers outside collected test modules prevents integration tests
from importing one another and makes their dependencies explicit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lingxu.config import Settings
from lingxu.domain.enums import ConsentKind
from lingxu.domain.errors import SummaryProviderError
from lingxu.domain.models import (
    AudioSegmentRecord,
    ProcessingJobRecord,
    RealtimeUtteranceRecord,
)
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import ArtifactRepository, TranscriptRepository
from lingxu.utils import new_ulid
from lingxu.worker.full_transcript_preview import (
    FullTranscriptMinutesProvider,
    PreviewTemplateSource,
)
from lingxu.worker.outcome_partitions import (
    LlmActionKind,
    LlmBudgetController,
    OutcomePartition,
    digest_task_hash,
    outcome_task_hash,
)
from lingxu.worker.outcomes import (
    OutcomeCandidate,
    OutcomeFailureCategory,
    OutcomeRequestKind,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)
from lingxu.worker.pipeline import (
    POST_MEETING_JOB,
    PostMeetingPipeline,
    SegmentTranscriber,
)
from lingxu.worker.summary import PostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TopicBoundaryCandidate,
    TopicBoundaryConfirmation,
    TopicBoundaryProviderPolicy,
)


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def _test_payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FakePublisher:
    def __init__(self, *, fail: bool = False, expected_content: str = "未使用 LLM") -> None:
        self.fail = fail
        self.expected_content = expected_content
        self.calls = 0
        self.contents: list[str] = []
        self.files: tuple[Path, ...] = ()

    async def publish(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        content: str,
        files: tuple[Path, ...],
    ) -> tuple[int, str]:
        self.calls += 1
        self.contents.append(content)
        self.files = files
        if self.fail:
            raise TimeoutError("simulated publication interruption")
        assert guild_id == 81
        assert channel_id == 8100
        assert self.expected_content in content
        assert all(path.is_file() for path in files)
        assert tuple(path.name for path in files) == ("minutes.md",)
        return 8100, "1001"


class FakeSummaryProvider:
    model = APPROVED_TOPIC_MODEL
    max_input_chars = 6_000

    def __init__(self, policy: TopicBoundaryProviderPolicy | None = None) -> None:
        self.received_text: list[str] = []
        self.topic_context_text: list[str] = []
        self.topic_calls = 0
        self.outcome_calls = 0
        self.outcome_event_batches: list[tuple[str, tuple[Any, ...]]] = []
        self.topic_unavailable = False
        self.outcome_unavailable = False
        self._topic_policy = policy or TopicBoundaryProviderPolicy(
            provider=APPROVED_TOPIC_PROVIDER,
            base_url=APPROVED_TOPIC_BASE_URL,
            model=APPROVED_TOPIC_MODEL,
            expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            allow_remote=True,
        )
        self._budget: LlmBudgetController | None = None

    def bind_llm_budget(self, controller: LlmBudgetController) -> None:
        self._budget = controller
        task_hash = digest_task_hash(controller.job_hash)
        if task_hash in controller.journal.expected_task_hashes:
            controller.reserve_task(task_hash)
            controller.reconcile_result(task_hash)

    def _record_outcome_action(
        self,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        *,
        action_kind: LlmActionKind = LlmActionKind.OUTCOME_INITIAL,
        failure_category: OutcomeFailureCategory | None = None,
    ) -> None:
        assert self._budget is not None
        task_hash = outcome_task_hash(self._budget.job_hash, partition, kind)
        self._budget.reserve_task(task_hash)
        action_hash, _ = self._budget.reserve_next_action(
            task_hash=task_hash,
            kind=action_kind,
            configured_timeout_seconds=1.0,
        )
        self._budget.complete_action(action_hash, failure_category=failure_category)

    @property
    def topic_boundary_policy(self) -> TopicBoundaryProviderPolicy:
        return self._topic_policy

    async def confirm_topic_boundary(
        self,
        *,
        candidate: TopicBoundaryCandidate,
        before: tuple[Any, ...],
        after: tuple[Any, ...],
    ) -> TopicBoundaryConfirmation:
        self.topic_calls += 1
        if self.topic_unavailable:
            raise SummaryProviderError("simulated topic outage")
        self.topic_context_text.extend(entry.text for entry in (*before, *after))
        return TopicBoundaryConfirmation(candidate.entry_id, True, 0.9)

    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        assert request_bytes == partition.request_bytes(kind)
        self._record_outcome_action(partition, kind)
        if kind is OutcomeRequestKind.DECISION_AGREEMENT:
            self.outcome_calls += 1
            self.outcome_event_batches.append((partition.topic_id, partition.core_events))
        if self.outcome_unavailable:
            raise SummaryProviderError("simulated outcome outage")
        self.received_text.extend(entry.text for entry in partition.core_entries)
        source_entry_ids = (partition.core_entries[0].entry_id,) if partition.core_entries else ()
        candidates = (
            (
                OutcomeCandidate(
                    id=f"{partition.partition_id}-risk-1",
                    type=OutcomeType.RISK,
                    text="已提炼风险",
                    owner_user_id=None,
                    owner_name=None,
                    deadline=None,
                    source_entry_ids=source_entry_ids,
                    source_event_ids=(),
                    confidence=0.9,
                    requires_confirmation=False,
                    verification_status=VerificationStatus.UNVERIFIED,
                ),
            )
            if source_entry_ids and kind is OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT
            else ()
        )
        return TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=candidates,
            extraction_method="local_llm",
            failure_types=(),
        )


class CrashAfterTopicWriteRepository(ProcessingJobRepository):
    """Inject one crash after topic bytes exist but before stage completion is durable."""

    def __init__(self, sessions: Any) -> None:
        super().__init__(sessions)
        self.crashed = False

    async def checkpoint(
        self,
        *,
        job_id: str,
        worker_id: str,
        stage: str,
        checkpoint: dict[str, Any],
        lease_seconds: float,
    ) -> ProcessingJobRecord:
        if stage == "SEGMENT_TOPICS" and "last_stage_details" in checkpoint and not self.crashed:
            self.crashed = True
            raise RuntimeError("injected post-topic-write crash")
        return await super().checkpoint(
            job_id=job_id,
            worker_id=worker_id,
            stage=stage,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
        )


async def _pipeline_fixture(
    tmp_path: Path,
    publisher: FakePublisher,
    summary_provider: PostMeetingInferenceProvider | None = None,
    transcriber: SegmentTranscriber | None = None,
    glossary_settings: Settings | None = None,
    full_transcript_minutes_provider: FullTranscriptMinutesProvider | None = None,
    full_transcript_minutes_template: PreviewTemplateSource | None = None,
) -> tuple[
    Database,
    ProcessingJobRepository,
    PostMeetingPipeline,
    ArtifactRepository,
    TranscriptRepository,
    str,
    str,
]:
    database = Database.open(_database_url(tmp_path / "worker.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    files = SessionFiles(tmp_path / "sessions")
    session = await MeetingSessionService(sessions, files).create(
        guild_id=81,
        creator_user_id=100,
    )
    await sessions.set_publication_channel(session_id=session.id, channel_id=8100)
    participants = ParticipantRepository(database.sessions)
    await participants.join(
        session_id=session.id,
        user_id=100,
        display_name="主持人甲",
        is_host=True,
    )
    await participants.join(session_id=session.id, user_id=200, display_name="参会者乙")
    await participants.set_consent(
        session_id=session.id,
        user_id=100,
        kind=ConsentKind.MINUTES,
        granted=True,
    )
    utterances = RealtimeUtteranceRepository(database.sessions)
    await utterances.add(
        RealtimeUtteranceRecord(
            id=new_ulid(),
            session_id=session.id,
            user_id=100,
            started_ms=0,
            ended_ms=1200,
            raw_text="<|zh|>会议开始",
            normalised_text="会议开始",
        )
    )
    agenda_repository = AgendaRepository(database.sessions)
    await agenda_repository.add(
        session_id=session.id,
        title="发布计划",
        duration_minutes=5,
        requires_vote=False,
    )
    covered_segment_path = Path("audio/100/000001.flac.enc")
    covered_segment_payload = b"encrypted-realtime-covered-fixture"
    files.atomic_write(session.id, covered_segment_path, covered_segment_payload)
    await AudioSegmentRepository(database.sessions).add(
        AudioSegmentRecord(
            id=new_ulid(),
            session_id=session.id,
            user_id=100,
            segment_number=1,
            encrypted_path=covered_segment_path,
            sha256=hashlib.sha256(covered_segment_payload).hexdigest(),
            started_ms=0,
            ended_ms=1200,
            bytes_encrypted=len(covered_segment_payload),
        )
    )
    relative_segment = Path("audio/200/000001.flac.enc")
    segment_payload = b"encrypted-non-consented-fixture"
    files.atomic_write(session.id, relative_segment, segment_payload)
    await AudioSegmentRepository(database.sessions).add(
        AudioSegmentRecord(
            id=new_ulid(),
            session_id=session.id,
            user_id=200,
            segment_number=1,
            encrypted_path=relative_segment,
            sha256=hashlib.sha256(segment_payload).hexdigest(),
            started_ms=1300,
            ended_ms=2400,
            bytes_encrypted=len(segment_payload),
        )
    )
    await utterances.add(
        RealtimeUtteranceRecord(
            id=new_ulid(),
            session_id=session.id,
            user_id=200,
            started_ms=1300,
            ended_ms=2400,
            raw_text="<|en|>private",
            normalised_text="private",
        )
    )
    events = MeetingEventStore(database.sessions)
    await events.append(
        session_id=session.id,
        event_type="CONSENT_GRANTED",
        actor_user_id=100,
        source="CONSENT",
        payload={
            "subject_user_id": 100,
            "consent_kinds": [kind.value for kind in ConsentKind],
            "effective_at_ms": 0,
        },
    )
    await events.append(
        session_id=session.id,
        event_type="VOTE_CLOSED",
        payload={
            "vote_id": "vote-1",
            "counts": {"yes": 2, "no": 0, "abstain": 0},
            "total": 2,
        },
    )
    transcripts = TranscriptRepository(database.sessions)
    await transcripts.upsert(
        session_id=session.id,
        source_key=f"BOT_TTS:{new_ulid()}",
        user_id=None,
        raw_text="现在进入议程: 发布计划。该流程预计持续5分钟。",
        normalised_text="现在进入议程: 发布计划。该流程预计持续5分钟。",
        started_ms=0,
        ended_ms=500,
    )
    await transcripts.upsert(
        session_id=session.id,
        source_key="segment:empty-speech",
        user_id=100,
        raw_text="",
        normalised_text="",
        started_ms=501,
        ended_ms=700,
    )
    artifacts = ArtifactRepository(database.sessions)
    pipeline = PostMeetingPipeline(
        sessions=sessions,
        events=events,
        participants=participants,
        agenda=agenda_repository,
        utterances=utterances,
        segments=AudioSegmentRepository(database.sessions),
        transcripts=transcripts,
        artifacts=artifacts,
        files=files,
        transcriber=transcriber,
        publisher=publisher,
        retention_hours=24,
        post_meeting_inference_provider=summary_provider,
        full_transcript_minutes_provider=full_transcript_minutes_provider,
        full_transcript_minutes_template=full_transcript_minutes_template,
        glossary_settings=glossary_settings,
    )
    jobs = ProcessingJobRepository(database.sessions)
    job = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="post-meeting-v1",
    )
    duplicate = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="post-meeting-v1",
    )
    assert duplicate.id == job.id
    return database, jobs, pipeline, artifacts, transcripts, session.id, job.id
