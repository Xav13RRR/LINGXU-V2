import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from sqlalchemy import delete, update

import lingxu.worker.pipeline as pipeline_module
from lingxu.config import Settings
from lingxu.domain.enums import (
    AgendaFollowupMode,
    ConsentKind,
    HostParticipationMode,
    JobStatus,
)
from lingxu.domain.errors import SummaryProviderError
from lingxu.domain.models import (
    AudioSegmentRecord,
    ProcessingJobRecord,
    RealtimeUtteranceRecord,
    TranscriptEntryRecord,
)
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.resources import deepseek_full_transcript_minutes_template
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
    utc_now,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import MeetingParticipantRow, ProcessingJobRow, TranscriptEntryRow
from lingxu.storage.worker_repositories import (
    ArtifactRepository,
    PublicationReconciliationRequired,
    TranscriptRepository,
)
from lingxu.utils import new_ulid
from lingxu.worker.consistency import MinutesConsistencyError
from lingxu.worker.full_transcript_preview import (
    AGENDA_TABLE_HEADER,
    AGENDA_TABLE_SEPARATOR,
    MUTE_ENFORCEMENT_TABLE_HEADER,
    MUTE_ENFORCEMENT_TABLE_SEPARATOR,
    SPEAKING_PLAN_TABLE_HEADER,
    SPEAKING_PLAN_TABLE_SEPARATOR,
    SPEAKING_TABLE_HEADER,
    SPEAKING_TABLE_SEPARATOR,
    VOTE_TABLE_HEADER,
    VOTE_TABLE_SEPARATOR,
    FullTranscriptFailureDiagnostics,
    FullTranscriptPreviewError,
    FullTranscriptPreviewResult,
)
from lingxu.worker.minutes_models import MinutesDocumentV2
from lingxu.worker.minutes_renderer import (
    MINUTES_RENDERER_REVISION,
    render_minutes,
    renderer_input_sha256,
)
from lingxu.worker.outcome_artifacts import (
    outcome_hash_input,
    read_partition_results,
    stored_outcome_policy_provenance,
    validated_outcome_hash_input,
)
from lingxu.worker.outcome_partitions import (
    LlmActionKind,
    LlmBudgetController,
    OutcomePartition,
    OutcomePartitionError,
    OutcomePartitionManifest,
    UnifiedLlmBudgetJournal,
    digest_task_hash,
    llm_job_hash,
    outcome_response_schema,
    outcome_task_hash,
    serialize_outcome_request,
)
from lingxu.worker.outcomes import (
    OutcomeCandidate,
    OutcomeCapacitySubtype,
    OutcomeFailureAttempt,
    OutcomeFailureCategory,
    OutcomeFailureRequestKind,
    OutcomeFormatSubtype,
    OutcomeProviderError,
    OutcomeProviderFailure,
    OutcomeRequestKind,
    OutcomeSizeBucket,
    OutcomeTaskDisposition,
    OutcomeType,
    OutcomeWireRowResolution,
    TopicOutcomeSet,
    VerificationStatus,
    outcome_failure_request_kind,
)
from lingxu.worker.pipeline import (
    _OUTCOME_PIPELINE_REVISION,
    POST_MEETING_JOB,
    POST_MEETING_STAGES,
    PostMeetingPipeline,
    SegmentTranscriber,
    _segment_is_covered_by_realtime,
)
from lingxu.worker.semantic_policy import SPEECH_ACT_POLICY_REVISION
from lingxu.worker.service import WorkerService
from lingxu.worker.summary import DeepSeekPostMeetingInferenceProvider
from lingxu.worker.topics import (
    APPROVED_TOPIC_BASE_URL,
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TopicBoundaryCandidate,
    TopicBoundaryConfirmation,
    TopicBoundaryProviderPolicy,
)
from tests.integration.worker_fixtures import (
    FakePublisher,
    FakeSummaryProvider,
    _database_url,
    _pipeline_fixture,
    _test_payload_hash,
)

_FULL_TRANSCRIPT_MINUTES_TEMPLATE = deepseek_full_transcript_minutes_template()


def _assert_summary_v1_shape(summary: object) -> None:
    assert isinstance(summary, dict)
    assert isinstance(summary.get("executive_summary"), str)
    for field in (
        "decisions",
        "action_items",
        "open_questions",
        "risks",
        "disagreements",
        "host_overrides",
        "source_entry_ids",
        "source_event_ids",
    ):
        assert isinstance(summary.get(field), list)


def _minutes_renderer_input(tmp_path: Path, session_id: str) -> str:
    payload = json.loads(
        (tmp_path / "sessions" / session_id / "meeting" / "minutes.json").read_text(
            encoding="utf-8"
        )
    )
    digest = payload["provenance"]["renderer_input_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    return digest


def _evidence_hash_input(payload: dict[str, object]) -> dict[str, object]:
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    return {
        **payload,
        "provenance": {
            key: value for key, value in provenance.items() if key != "evidence_payload_hash"
        },
    }


class PreparedObservingPublisher(FakePublisher):
    def __init__(self) -> None:
        super().__init__()
        self.artifacts: ArtifactRepository | None = None
        self.session_id: str | None = None
        self.observed_committed_prepared = False

    async def publish(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        content: str,
        files: tuple[Path, ...],
    ) -> tuple[int, str]:
        assert self.artifacts is not None
        assert self.session_id is not None
        intents = await self.artifacts.all_of_type(
            session_id=self.session_id,
            artifact_type="DISCORD_PUBLICATION_INTENT",
        )
        receipts = await self.artifacts.all_of_type(
            session_id=self.session_id,
            artifact_type="DISCORD_PUBLICATION",
        )
        assert len(intents) == 1
        assert intents[0].metadata["state"] == "PREPARED"
        assert not receipts
        self.observed_committed_prepared = True
        return await super().publish(
            guild_id=guild_id,
            channel_id=channel_id,
            content=content,
            files=files,
        )


class FakeFullTranscriptMinutesProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    @property
    def max_output_tokens(self) -> int:
        return 131_072

    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        self.calls += 1
        self.messages.append(messages)
        facts_text = (
            messages[1]["content"]
            .split("<authoritative_meeting_facts_json>\n", 1)[1]
            .split("\n</authoritative_meeting_facts_json>", 1)[0]
        )
        facts = json.loads(facts_text)
        agenda_rows = "\n".join(facts["agenda_markdown_rows"])
        vote_rows = "\n".join(facts["vote_markdown_rows"])
        speaking_plan_rows = "\n".join(facts["speaking_plan_markdown_rows"])
        mute_rows = "\n".join(facts["mute_enforcement_markdown_rows"])
        speaking_rows = "\n".join(facts["speaking_markdown_rows"])
        content = f"""# 聆序会议纪要

## 会议概览

- 参会者: [主持人甲]

## 执行摘要

测试摘要。

## 议程清单

{AGENDA_TABLE_HEADER}
{AGENDA_TABLE_SEPARATOR}
{agenda_rows}

## 权威流程配置

{SPEAKING_PLAN_TABLE_HEADER}
{SPEAKING_PLAN_TABLE_SEPARATOR}
{speaking_plan_rows}

## 权威静音执行

{MUTE_ENFORCEMENT_TABLE_HEADER}
{MUTE_ENFORCEMENT_TABLE_SEPARATOR}
{mute_rows}

## 权威发言流程

{SPEAKING_TABLE_HEADER}
{SPEAKING_TABLE_SEPARATOR}
{speaking_rows}

## 投票结果

{VOTE_TABLE_HEADER}
{VOTE_TABLE_SEPARATOR}
{vote_rows}

## 议题与讨论

测试议题。

## 已确认决策

- 无明确决策。

## 已确认共识

- 无明确共识。

## 行动事项

- 无。

## 开放问题

- 无。

## 提案

- 无。

## 风险

- 无。

## 分歧

- 无。

## 优化去重后的完整逐字稿

- [00:00:00-00:00:00] **[聆序]** <!-- source_ordinals:0 -->: 系统发言
- [00:00:00-00:00:01] **[主持人甲]** <!-- source_ordinals:1 -->: 真人发言
"""
        return FullTranscriptPreviewResult(
            content=content,
            finish_reason="stop",
            status="complete",
            response_content_bytes=len(content.encode("utf-8")),
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


class FailingFullTranscriptMinutesProvider(FakeFullTranscriptMinutesProvider):
    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        self.calls += 1
        self.messages.append(messages)
        raise FullTranscriptPreviewError(
            "preview_finish_reason_invalid",
            diagnostics=FullTranscriptFailureDiagnostics(
                phase="response_validation",
                elapsed_ms=321,
                finish_reason="insufficient_system_resource",
                content_state="missing",
                reasoning_state="null",
                reasoning_content_chars=0,
                response_body_bytes=456,
                usage_prompt_tokens=100,
                usage_completion_tokens=0,
                usage_total_tokens=100,
            ),
        )


class OutputLimitedFullTranscriptMinutesProvider(FakeFullTranscriptMinutesProvider):
    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        self.calls += 1
        self.messages.append(messages)
        content = "PRIVATE_TRUNCATED_MINUTES_MUST_NOT_BE_JOURNALED"
        return FullTranscriptPreviewResult(
            content=content,
            finish_reason="length",
            status="output_limit",
            response_content_bytes=len(content.encode("utf-8")),
            usage={"prompt_tokens": 1_000, "completion_tokens": 65_536, "total_tokens": 66_536},
            diagnostics=FullTranscriptFailureDiagnostics(
                phase="response_validation",
                elapsed_ms=401_000,
                http_status=200,
                finish_reason="length",
                content_state="present",
                reasoning_state="present",
                reasoning_content_chars=12_345,
                response_body_bytes=987_654,
                usage_prompt_tokens=1_000,
                usage_completion_tokens=65_536,
                usage_total_tokens=66_536,
            ),
        )


class CoverageDriftFullTranscriptMinutesProvider(FakeFullTranscriptMinutesProvider):
    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        result = await super().generate(messages)
        content = result.content.replace(
            "- [00:00:00-00:00:01] **[主持人甲]** <!-- source_ordinals:1 -->: 真人发言",
            "- [00:00:00-00:00:01] **[主持人甲]** <!-- source_ordinals:999 -->: 错误覆盖内容",
        )
        return replace(
            result,
            content=content,
            response_content_bytes=len(content.encode("utf-8")),
        )


class EvidenceDriftFullTranscriptMinutesProvider(FakeFullTranscriptMinutesProvider):
    async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
        result = await super().generate(messages)
        content = result.content.replace(
            "- 无明确决策。",
            "- 已确认测试。 <!-- evidence_ordinals:1,1 -->",
        ).replace(
            "## 行动事项\n\n- 无。",
            "\n".join(
                (
                    "## 行动事项",
                    "",
                    "| 行动事项 | 负责人 | 截止时间 | 依据 |",
                    "| --- | --- | --- | --- |",
                    "| 执行测试 | [错误负责人] | 未提及 | source_ordinals:1,1 |",
                )
            ),
        )
        return replace(
            result,
            content=content,
            response_content_bytes=len(content.encode("utf-8")),
        )


class ExactWireFoldSummaryProvider(FakeSummaryProvider):
    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        result = await super().extract_topic_outcome_kind(
            partition=partition,
            kind=kind,
            request_bytes=request_bytes,
        )
        if not result.candidates:
            return result
        candidate = result.candidates[0]
        return result.model_copy(
            update={
                "wire_input_row_count": 2,
                "wire_row_resolutions": (
                    OutcomeWireRowResolution(
                        output_candidate_id=candidate.id,
                        retained_ordinal=0,
                        contributor_ordinals=(0, 1),
                    ),
                ),
            }
        )


class SecondRepairSuccessSummaryProvider(FakeSummaryProvider):
    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        if self.outcome_unavailable:
            raise SummaryProviderError("simulated outcome outage")
        if kind is not OutcomeRequestKind.ACTION:
            return await super().extract_topic_outcome_kind(
                partition=partition,
                kind=kind,
                request_bytes=request_bytes,
            )
        assert request_bytes == partition.request_bytes(kind)
        self._record_outcome_action(partition, kind)
        for _ in range(2):
            self._record_outcome_action(
                partition,
                kind,
                action_kind=LlmActionKind.OUTCOME_REPAIR,
            )
        return TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=(
                OutcomeProviderFailure(
                    category=OutcomeFailureCategory.JSON,
                    request_kind=OutcomeFailureRequestKind.ACTION,
                    attempt=OutcomeFailureAttempt.INITIAL,
                    count=1,
                    size_bucket=OutcomeSizeBucket.LTE_1K,
                    capacity_subtype=None,
                    format_subtype=OutcomeFormatSubtype.JSON_SYNTAX_OTHER,
                ),
                OutcomeProviderFailure(
                    category=OutcomeFailureCategory.OUTPUT_CAPACITY,
                    request_kind=OutcomeFailureRequestKind.ACTION,
                    attempt=OutcomeFailureAttempt.REPAIR,
                    count=1,
                    size_bucket=OutcomeSizeBucket.LTE_1K,
                    capacity_subtype=OutcomeCapacitySubtype.COMPLETE_FALSE,
                ),
            ),
            task_disposition=OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS,
        )


class FailingSummaryProvider:
    model = APPROVED_TOPIC_MODEL
    max_input_chars = 6_000

    def bind_llm_budget(self, controller: LlmBudgetController) -> None:
        task_hash = digest_task_hash(controller.job_hash)
        if task_hash in controller.journal.expected_task_hashes:
            controller.reserve_task(task_hash)
            controller.reconcile_result(task_hash)

    @property
    def topic_boundary_policy(self) -> TopicBoundaryProviderPolicy:
        return TopicBoundaryProviderPolicy(
            provider=APPROVED_TOPIC_PROVIDER,
            base_url=APPROVED_TOPIC_BASE_URL,
            model=APPROVED_TOPIC_MODEL,
            expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
            allow_remote=True,
        )

    async def confirm_topic_boundary(
        self,
        *,
        candidate: TopicBoundaryCandidate,
        before: tuple[Any, ...],
        after: tuple[Any, ...],
    ) -> TopicBoundaryConfirmation:
        del candidate, before, after
        raise SummaryProviderError("simulated local outage")

    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        del partition, kind, request_bytes
        raise SummaryProviderError("simulated local outage")


class ClosedFailingOutcomeProvider(FakeSummaryProvider):
    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        del request_bytes
        self.outcome_calls += int(kind is OutcomeRequestKind.DECISION_AGREEMENT)
        self._record_outcome_action(partition, kind, failure_category=OutcomeFailureCategory.JSON)
        raise OutcomeProviderError(
            OutcomeProviderFailure(
                category=OutcomeFailureCategory.JSON,
                request_kind=outcome_failure_request_kind(kind),
                attempt=OutcomeFailureAttempt.INITIAL,
                count=1,
                size_bucket=OutcomeSizeBucket.LTE_1K,
                capacity_subtype=None,
                format_subtype=OutcomeFormatSubtype.JSON_DECODE,
            )
        )


class ClosedCapacityOutcomeProvider(FakeSummaryProvider):
    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        assert request_bytes == partition.request_bytes(kind)
        self._record_outcome_action(
            partition,
            kind,
            failure_category=None,
        )
        if kind is not OutcomeRequestKind.ACTION:
            return TopicOutcomeSet(
                topic_id=partition.topic_id,
                candidates=(),
                extraction_method="local_llm",
                failure_types=(),
            )
        for _ in range(2):
            self._record_outcome_action(
                partition,
                kind,
                action_kind=LlmActionKind.OUTCOME_REPAIR,
                failure_category=None,
            )
        failures = tuple(
            OutcomeProviderFailure(
                category=OutcomeFailureCategory.OUTPUT_CAPACITY,
                request_kind=OutcomeFailureRequestKind.ACTION,
                attempt=(
                    OutcomeFailureAttempt.INITIAL if index == 0 else OutcomeFailureAttempt.REPAIR
                ),
                count=1,
                size_bucket=OutcomeSizeBucket.LTE_1K,
                capacity_subtype=OutcomeCapacitySubtype.TEXT_CHARS,
            )
            for index in range(3)
        )
        return TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(),
            extraction_method="local_llm_partial",
            failure_types=(),
            provider_failures=failures,
        )


class UnsupportedOutcomeSummaryProvider(FakeSummaryProvider):
    async def extract_topic_outcome_kind(
        self,
        *,
        partition: OutcomePartition,
        kind: OutcomeRequestKind,
        request_bytes: bytes,
    ) -> TopicOutcomeSet:
        del request_bytes
        self.outcome_calls += int(kind is OutcomeRequestKind.DECISION_AGREEMENT)
        self._record_outcome_action(partition, kind)
        return TopicOutcomeSet(
            topic_id=partition.topic_id,
            candidates=(
                OutcomeCandidate(
                    id="unsupported-outcome",
                    type=OutcomeType.DECISION,
                    text="UNSUPPORTED_MODEL_FACT",
                    owner_user_id=None,
                    owner_name=None,
                    deadline=None,
                    source_entry_ids=("fabricated-entry",),
                    source_event_ids=(),
                    confidence=0.99,
                    requires_confirmation=False,
                    verification_status=VerificationStatus.EVIDENCE_LINKED,
                ),
            )
            if kind is OutcomeRequestKind.DECISION_AGREEMENT
            else (),
            extraction_method="local_llm",
            failure_types=(),
        )


class FakeSegmentTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    async def transcribe(
        self,
        *,
        session_id: str,
        user_id: int,
        encrypted_path: Path,
        started_ms: int,
        ended_ms: int,
    ) -> tuple[str, str]:
        del session_id, encrypted_path
        self.calls.append((user_id, started_ms, ended_ms))
        return "<|zh|>仅缺口补录", "仅缺口补录"


class CrashAfterNormalisationWriteRepository(ProcessingJobRepository):
    """Inject one crash at the durable checkpoint after normalisation writes its artifact."""

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
        if (
            stage == "NORMALISE_ENTITIES"
            and "last_stage_details" in checkpoint
            and not self.crashed
        ):
            self.crashed = True
            raise RuntimeError("injected post-normalisation-write crash")
        return await super().checkpoint(
            job_id=job_id,
            worker_id=worker_id,
            stage=stage,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
        )


class CrashAfterOutcomeWriteRepository(ProcessingJobRepository):
    """Inject one crash after outcome bytes exist but before stage completion is durable."""

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
        if (
            stage == "EXTRACT_TOPIC_OUTCOMES"
            and "last_stage_details" in checkpoint
            and not self.crashed
        ):
            self.crashed = True
            raise RuntimeError("injected post-outcome-write crash")
        return await super().checkpoint(
            job_id=job_id,
            worker_id=worker_id,
            stage=stage,
            checkpoint=checkpoint,
            lease_seconds=lease_seconds,
        )


async def _seed_schema_two_unfinished_job(
    jobs: ProcessingJobRepository,
    job_id: str,
) -> ProcessingJobRecord:
    claimed = await jobs.claim(
        worker_id="schema-two-worker",
        lease_seconds=30,
        job_types=(POST_MEETING_JOB,),
    )
    assert claimed is not None and claimed.id == job_id
    await jobs.checkpoint(
        job_id=job_id,
        worker_id="schema-two-worker",
        stage="PUBLISH_DISCORD",
        checkpoint={
            "schema_version": 2,
            "completed_stages": [
                "VERIFY_SESSION",
                "VERIFY_AUDIO_SEGMENTS",
                "FINALISE_REALTIME_UTTERANCES",
                "TRANSCRIBE_MISSING_SEGMENTS",
                "VALIDATE_LANGUAGES",
                "MERGE_TRANSCRIPT",
                "MERGE_MEETING_EVENTS",
                "CALCULATE_LANGUAGE_DISTRIBUTION",
                "GENERATE_EVIDENCE",
                "GENERATE_MINUTES",
                "RENDER_ARTIFACTS",
            ],
            "active_stage": "PUBLISH_DISCORD",
        },
        lease_seconds=30,
    )
    await jobs.fail(
        job_id=job_id,
        worker_id="schema-two-worker",
        error_type="InterruptedError@PUBLISH_DISCORD",
        retry_at=utc_now(),
    )
    retry = await jobs.get(job_id)
    assert retry is not None and retry.status is JobStatus.PENDING
    return retry


async def _seed_legacy_schema_three_job(
    jobs: ProcessingJobRepository,
    job_id: str,
    *,
    outcome_pipeline_revision: str | None = None,
) -> ProcessingJobRecord:
    claimed = await jobs.claim(
        worker_id="legacy-schema-three-worker",
        lease_seconds=30,
        job_types=(POST_MEETING_JOB,),
    )
    assert claimed is not None and claimed.id == job_id
    legacy_stages = [
        "VERIFY_SESSION",
        "BUILD_CONSENT_TIMELINE",
        "APPLY_PRIVACY_CUTOFFS",
        "VERIFY_AUDIO_SEGMENTS",
        "FINALISE_REALTIME_UTTERANCES",
        "TRANSCRIBE_MISSING_SEGMENTS",
        "VALIDATE_LANGUAGES",
        "MERGE_TRANSCRIPT",
        "MERGE_MEETING_EVENTS",
        "CALCULATE_LANGUAGE_DISTRIBUTION",
        "GENERATE_EVIDENCE",
        "GENERATE_MINUTES",
        "RENDER_ARTIFACTS",
        "PUBLISH_DISCORD",
    ]
    checkpoint_payload: dict[str, object] = {
        "schema_version": 3,
        "completed_stages": legacy_stages,
        "active_stage": "SCHEDULE_CLEANUP",
    }
    if outcome_pipeline_revision is not None:
        checkpoint_payload["outcome_pipeline_revision"] = outcome_pipeline_revision
    await jobs.checkpoint(
        job_id=job_id,
        worker_id="legacy-schema-three-worker",
        stage="PUBLISH_DISCORD",
        checkpoint=checkpoint_payload,
        lease_seconds=30,
    )
    await jobs.fail(
        job_id=job_id,
        worker_id="legacy-schema-three-worker",
        error_type="InterruptedError@SCHEDULE_CLEANUP",
        retry_at=utc_now(),
    )
    retry = await jobs.get(job_id)
    assert retry is not None and retry.status is JobStatus.PENDING
    return retry


async def _seed_prefingerprint_schema_three_job(
    jobs: ProcessingJobRepository,
    job_id: str,
    *,
    revision: str | None = None,
    speech_revision: str | None = None,
    minutes_revision: str | None = None,
    minutes_input_digest: str | None = None,
) -> ProcessingJobRecord:
    worker_id = "prefingerprint-schema-three-worker"
    claimed = await jobs.claim(
        worker_id=worker_id,
        lease_seconds=30,
        job_types=(POST_MEETING_JOB,),
    )
    assert claimed is not None and claimed.id == job_id
    completed = [
        stage
        for stage in POST_MEETING_STAGES[: POST_MEETING_STAGES.index("PUBLISH_DISCORD")]
        if stage != "VALIDATE_MINUTES_CONSISTENCY"
    ]
    checkpoint_payload: dict[str, object] = {
        "schema_version": 3,
        "completed_stages": completed,
        "active_stage": "PUBLISH_DISCORD",
    }
    if revision is not None:
        checkpoint_payload["outcome_pipeline_revision"] = revision
    if speech_revision is not None:
        checkpoint_payload["speech_act_policy_revision"] = speech_revision
    if minutes_revision is not None:
        checkpoint_payload["minutes_renderer_revision"] = minutes_revision
    if minutes_input_digest is not None:
        checkpoint_payload["renderer_input_sha256"] = minutes_input_digest
    await jobs.checkpoint(
        job_id=job_id,
        worker_id=worker_id,
        stage="PUBLISH_DISCORD",
        checkpoint=checkpoint_payload,
        lease_seconds=30,
    )
    await jobs.fail(
        job_id=job_id,
        worker_id=worker_id,
        error_type="InterruptedError@PUBLISH_DISCORD",
        retry_at=utc_now(),
    )
    retry = await jobs.get(job_id)
    assert retry is not None and retry.status is JobStatus.PENDING
    return retry


def test_realtime_coverage_requires_complete_same_user_segment_coverage() -> None:
    segment = AudioSegmentRecord(
        id="segment-1",
        session_id="session-1",
        user_id=100,
        segment_number=1,
        encrypted_path=Path("audio/100/000001.flac.enc"),
        sha256="0" * 64,
        started_ms=1_000,
        ended_ms=5_000,
        bytes_encrypted=1,
    )

    def entry(
        entry_id: str,
        *,
        user_id: int,
        started_ms: int,
        ended_ms: int,
        source_key: str | None = None,
    ) -> TranscriptEntryRecord:
        return TranscriptEntryRecord(
            id=entry_id,
            session_id="session-1",
            source_key=source_key or f"utterance:{entry_id}",
            user_id=user_id,
            raw_text="有效发言",
            normalised_text="有效发言",
            started_ms=started_ms,
            ended_ms=ended_ms,
        )

    fully_covered = (
        entry("one", user_id=100, started_ms=1_500, ended_ms=2_400),
        entry("two", user_id=100, started_ms=2_600, ended_ms=4_000),
    )
    partial = (entry("partial", user_id=100, started_ms=1_000, ended_ms=2_000),)
    material_gap = (
        entry("gap-one", user_id=100, started_ms=1_000, ended_ms=1_800),
        entry("gap-two", user_id=100, started_ms=3_000, ended_ms=5_000),
    )
    wrong_user = (entry("other", user_id=200, started_ms=1_000, ended_ms=5_000),)
    offline_source = (
        entry(
            "offline",
            user_id=100,
            started_ms=1_000,
            ended_ms=5_000,
            source_key="segment:offline",
        ),
    )

    assert _segment_is_covered_by_realtime(segment, fully_covered)
    assert not _segment_is_covered_by_realtime(segment, partial)
    assert not _segment_is_covered_by_realtime(segment, material_gap)
    assert not _segment_is_covered_by_realtime(segment, wrong_user)
    assert not _segment_is_covered_by_realtime(segment, offline_source)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_pipeline_renders_consent_safe_artifacts_and_publishes_once(
    tmp_path: Path,
) -> None:
    publisher = PreparedObservingPublisher()
    database, jobs, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path, publisher
    )
    publisher.artifacts = artifacts
    publisher.session_id = session_id

    assert await WorkerService(
        jobs=jobs,
        worker_id="worker-a",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None
    assert completed.status is JobStatus.COMPLETED
    assert completed.current_stage == "SCHEDULE_CLEANUP"
    assert completed.checkpoint["schema_version"] == 4
    assert len(completed.checkpoint["completed_stages"]) == 22
    assert completed.checkpoint["completed_stages"][7:13] == [
        "CANONICALISE_TRANSCRIPT",
        "NORMALISE_ENTITIES",
        "SEGMENT_TOPICS",
        "EXTRACT_TOPIC_OUTCOMES",
        "LINK_EVIDENCE",
        "VALIDATE_OUTCOMES",
    ]
    assert publisher.calls == 1
    assert publisher.observed_committed_prepared
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    assert publisher.files == (meeting_dir / "minutes.md",)

    entries = await transcripts.list(session_id)
    assert len(entries) == 2
    assert {entry.user_id for entry in entries} == {None, 100}
    rendered = await artifacts.list(session_id)
    assert {artifact.artifact_type for artifact in rendered} == {
        "TRANSCRIPT_JSON",
        "TRANSCRIPT_TEXT",
        "CANONICAL_TRANSCRIPT_JSON",
        "TOPIC_SEGMENTS_JSON",
        "OUTCOME_PARTITION_MANIFEST_JSON",
        "OUTCOME_BUDGET_JOURNAL_JSON",
        "OUTCOME_PARTITION_RESULTS_JSON",
        "CONSENT_TIMELINE_JSON",
        "MEETING_EVENTS_JSON",
        "LANGUAGE_DISTRIBUTION_JSON",
        "EVIDENCE_JSON",
        "PRIVACY_EXCLUSIONS_JSON",
        "MINUTES_MARKDOWN",
        "MINUTES_JSON",
        "DISCORD_PUBLICATION",
        "DISCORD_PUBLICATION_INTENT",
    }
    minutes = tmp_path / "sessions" / session_id / "meeting" / "minutes.md"
    assert "确定性模板 (未使用 LLM)" in minutes.read_text(encoding="utf-8")
    assert "主持人甲" in minutes.read_text(encoding="utf-8")
    assert "发布计划" in minutes.read_text(encoding="utf-8")
    assert "聆序机器人" in minutes.read_text(encoding="utf-8")
    assert "private" not in minutes.read_text(encoding="utf-8")
    minutes_json = json.loads(
        (tmp_path / "sessions" / session_id / "meeting" / "minutes.json").read_text(
            encoding="utf-8"
        )
    )
    assert minutes_json["schema_version"] == 2
    assert minutes_json["meeting_title"] is None
    assert minutes_json["provenance"]["minutes_renderer_revision"] == MINUTES_RENDERER_REVISION
    assert len(minutes_json["provenance"]["renderer_input_sha256"]) == 64
    assert minutes_json["executive_summary"]
    assert minutes_json["full_summary"]
    assert minutes_json["agendas"][0]["summary"]
    assert minutes_json["topics"][0]["summary"]
    assert minutes_json["summary_method"] == "deterministic"
    assert len(minutes_json["validated_outcomes"]) == 1
    assert minutes_json["validated_outcomes"][0]["candidates"] == []
    assert minutes_json["deterministic_counts"] == {
        "agenda_completed": 0,
        "agenda_speaking_plans": 1,
        "agenda_total": 1,
        "bot_tts_entries": 1,
        "closed_votes": 1,
        "controlled_speaking_followups": 0,
        "controlled_speaking_turns": 0,
        "mute_enforcement_summaries": 1,
        "participant_transcript_entries": 1,
        "topic_total": 1,
        "validated_outcomes": 0,
    }
    assert minutes_json["privacy_exclusions"] == {
        "measurement_available": True,
        "segment_files": 1,
        "segment_rows": 1,
        "transcript_entries": 0,
    }
    assert minutes_json["canonical_metrics"]["source_entry_count"] == 2
    assert minutes_json["evidence_coverage"]["evidence_required"] == 0
    assert (meeting_dir / "privacy_exclusions.json").is_file()
    assert "会议标题: 未设置" in minutes.read_text(encoding="utf-8")
    for private_name in (
        "topic_outcome_candidates.json",
        "linked_outcomes.json",
        "validated_outcomes.json",
    ):
        assert (meeting_dir / private_name).is_file()
        assert private_name not in {artifact.path.name for artifact in rendered}
    transcript = tmp_path / "sessions" / session_id / "meeting" / "transcript.txt"
    assert "主持人: 主持人甲" in transcript.read_text(encoding="utf-8")
    assert "聆序机器人" in transcript.read_text(encoding="utf-8")

    intents = await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    receipts = await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    assert len(intents) == len(receipts) == 1
    intent = intents[0]
    receipt = receipts[0]
    assert intent.metadata["state"] == receipt.metadata["state"] == "FINALIZED"
    assert intent.metadata["publication_contract_revision"] == "task11-one-file-v1"
    assert receipt.metadata["publication_contract_revision"] == "task11-one-file-v1"
    assert intent.metadata["message_id"] == receipt.metadata["message_id"] == "1001"
    assert intent.metadata["selected_channel_id"] == receipt.metadata["channel_id"] == 8100

    assert intent.metadata["intent_nonce"] == receipt.metadata["intent_nonce"]
    assert intent.metadata["receipt_payload_sha256"] == receipt.metadata["receipt_payload_sha256"]
    assert intent.metadata["attachments"] == receipt.metadata["attachments"]
    assert intent.metadata["pipeline_version"] == receipt.metadata["pipeline_version"] == 4
    assert [item["artifact_type"] for item in intent.metadata["attachments"]] == [
        "MINUTES_MARKDOWN",
    ]
    checkpoint = completed.checkpoint
    assert checkpoint["minutes_renderer_revision"] == MINUTES_RENDERER_REVISION
    assert (
        checkpoint["renderer_input_sha256"] == minutes_json["provenance"]["renderer_input_sha256"]
    )
    rendered_by_type = {artifact.artifact_type: artifact for artifact in rendered}
    for artifact_type in (
        "TRANSCRIPT_JSON",
        "TRANSCRIPT_TEXT",
        "CANONICAL_TRANSCRIPT_JSON",
        "TOPIC_SEGMENTS_JSON",
        "CONSENT_TIMELINE_JSON",
        "MEETING_EVENTS_JSON",
        "LANGUAGE_DISTRIBUTION_JSON",
        "EVIDENCE_JSON",
        "PRIVACY_EXCLUSIONS_JSON",
        "MINUTES_MARKDOWN",
        "MINUTES_JSON",
    ):
        assert rendered_by_type[artifact_type].metadata["pipeline_version"] == 4
    for artifact_type in ("MINUTES_JSON", "MINUTES_MARKDOWN"):
        assert (
            rendered_by_type[artifact_type].metadata["minutes_renderer_revision"]
            == MINUTES_RENDERER_REVISION
        )
        assert len(rendered_by_type[artifact_type].metadata["renderer_input_sha256"]) == 64

    assert not await WorkerService(
        jobs=jobs,
        worker_id="worker-b",
        pipeline=pipeline,
    ).run_once()
    assert publisher.calls == 1

    terminal_job = await jobs.create(
        session_id=session_id,
        job_type=POST_MEETING_JOB,
        idempotency_key="terminal-publication-recovery",
    )
    terminal_publisher = FakePublisher()
    terminal_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=terminal_publisher,
        retention_hours=24,
    )
    artifact_count = len(await artifacts.list(session_id))
    assert await WorkerService(
        jobs=jobs,
        worker_id="terminal-publication-worker",
        pipeline=terminal_pipeline,
    ).run_once()
    terminal = await jobs.get(terminal_job.id)
    assert terminal is not None and terminal.status is JobStatus.COMPLETED
    assert terminal.current_stage is None
    assert terminal.checkpoint == {}
    assert terminal_publisher.calls == 0
    assert len(await artifacts.list(session_id)) == artifact_count
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_uses_full_transcript_deepseek_markdown_as_final_artifact(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher(expected_content="逐字稿和已验证议题结果已生成")
    final_provider = FakeFullTranscriptMinutesProvider()
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )
    second_agenda = await AgendaRepository(database.sessions).add(
        session_id=session_id,
        title="风险复盘",
        duration_minutes=10,
        requires_vote=True,
        vote_content="是否接受风险?",
        vote_duration_seconds=60,
        speaking_policy=AgendaSpeakingPolicy.presentation(
            presenter_user_ids=(100, 200),
            followup_mode=AgendaFollowupMode.Q_AND_A,
            mute_enforcement_requested=True,
        ),
    )
    await AgendaRepository(database.sessions).add(
        session_id=session_id,
        title="显式顺序",
        duration_minutes=10,
        requires_vote=False,
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=(
                SpeakerTurnPlan(user_id=100, planned_seconds=120),
                SpeakerTurnPlan(user_id=200, planned_seconds=90),
            ),
            followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
            mute_enforcement_requested=True,
            prep_seconds=5,
        ),
    )
    await AgendaRepository(database.sessions).add(
        session_id=session_id,
        title="首字母顺序",
        duration_minutes=10,
        requires_vote=False,
        speaking_policy=AgendaSpeakingPolicy.ordered_alphabetical(
            default_turn_seconds=60,
            host_participation=HostParticipationMode.MODERATOR_ONLY,
            followup_mode=AgendaFollowupMode.Q_AND_A,
            prep_seconds=0,
        ),
    )
    event_store = MeetingEventStore(database.sessions)
    await event_store.append(
        session_id=session_id,
        event_type="VOTE_CLOSED",
        payload={
            "vote_id": "vote-2",
            "agenda_item_id": second_agenda.id,
            "vote_content": "是否接受风险?",
            "counts": {"yes": 1, "no": 0, "abstain": 1},
            "eligible_count": 2,
            "responded_count": 1,
            "result": "是",
        },
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_FLOW_STARTED",
        payload={
            "agenda_item_id": second_agenda.id,
            "mode": "PRESENTATION",
            "turns": [],
            "presenter_user_ids": [100, 200],
            "identities": [
                {"user_id": 100, "display_name": "主持人甲"},
                {"user_id": 200, "display_name": "参会者乙"},
            ],
            "followup_mode": "Q_AND_A",
            "prep_seconds": 10,
            "presentation_seconds": 600,
        },
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_TURN_STARTED",
        payload={
            "agenda_item_id": second_agenda.id,
            "turn_index": 0,
            "current_user_ids": [100],
            "next_user_ids": [],
            "planned_seconds": 600,
        },
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_TURN_EXTENDED",
        payload={
            "agenda_item_id": second_agenda.id,
            "turn_index": 0,
            "current_user_ids": [100],
            "additional_seconds": 30,
        },
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_TURN_FINISHED",
        payload={
            "agenda_item_id": second_agenda.id,
            "turn_index": 0,
            "current_user_ids": [100],
            "planned_seconds": 600,
            "extension_seconds": 30,
            "reason": "HOST_ENDED_EARLY",
        },
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_FOLLOWUP_STARTED",
        payload={"agenda_item_id": second_agenda.id, "followup_mode": "Q_AND_A"},
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKER_SELECTED",
        payload={"user_id": 200},
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKER_FINISHED",
        payload={"user_id": 200},
    )
    await event_store.append(
        session_id=session_id,
        event_type="SPEAKING_FOLLOWUP_FINISHED",
        payload={"agenda_item_id": second_agenda.id, "followup_mode": "Q_AND_A"},
    )
    for event_type, payload in (
        ("MUTE_ENFORCEMENT_STARTED", {"mode": "PRESENTATION"}),
        ("MUTE_PERMISSION_LOST", {"operation": "REAPPLY_MUTE"}),
        (
            "MUTE_ENFORCEMENT_PAUSED",
            {"reason": "PERMISSION_LOST", "timer_type": None, "timer_owned": False},
        ),
        ("MUTE_ENFORCEMENT_RESUMED", {}),
        ("MUTE_REAPPLIED", {"reason": "SERVER_UNMUTE_REAPPLIED"}),
        ("HOST_INTERVENTION_STARTED", {"timer_type": None, "timer_owned": False}),
        ("MUTE_ENFORCEMENT_ENDED", {}),
    ):
        await event_store.append(
            session_id=session_id,
            event_type=event_type,
            payload={"agenda_item_id": second_agenda.id, **payload},
        )

    assert await WorkerService(
        jobs=jobs,
        worker_id="full-transcript-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert final_provider.calls == 1
    assert publisher.contents == [f"聆序会议 `{session_id}` 的逐字稿和已验证议题结果已生成。"]
    system_message, user_message = final_provider.messages[0]
    assert '"speaker_name":"主持人甲"' in user_message["content"]
    assert '"speaker_label":"[主持人甲]"' in user_message["content"]
    assert '"speaker_name":"聆序"' in user_message["content"]
    assert '"speaker_label":"[聆序]"' in user_message["content"]
    facts_text = (
        user_message["content"]
        .split("<authoritative_meeting_facts_json>\n", 1)[1]
        .split("\n</authoritative_meeting_facts_json>", 1)[0]
    )
    facts = json.loads(facts_text)
    assert facts["agenda_count"] == 4
    assert facts["participant_count"] == 1
    assert facts["participant_labels"] == ["[主持人甲]"]
    assert [item["title"] for item in facts["agendas"]] == [
        "发布计划",
        "风险复盘",
        "显式顺序",
        "首字母顺序",
    ]
    assert facts["speaking_plan_count"] == 4
    assert facts["mute_enforcement_count"] == 4
    assert [item["mode"] for item in facts["speaking_plans"]] == [
        "FREE",
        "PRESENTATION",
        "ORDERED",
        "ORDERED",
    ]
    assert [item["order_strategy"] for item in facts["speaking_plans"]] == [
        "NONE",
        "NONE",
        "EXPLICIT",
        "DISCORD_ALPHABETICAL",
    ]
    assert facts["speaking_plans"][2]["followup_mode"] == "SUPPLEMENT_TURNS"
    assert facts["speaking_plans"][2]["mute_enforcement_requested"] is True
    assert facts["speaking_plans"][3]["host_participation"] == "MODERATOR_ONLY"
    assert facts["speaking_plans"][3]["prep_seconds"] == 0
    assert facts["mute_enforcement_summaries"][1] == {
        "agenda_item_id": second_agenda.id,
        "requested": True,
        "started": True,
        "pause_count": 1,
        "resume_count": 1,
        "permission_loss_count": 1,
        "reapplication_count": 1,
        "host_intervention_count": 1,
        "final_status": "ENDED",
        "agenda_title": "风险复盘",
        "markdown_row": facts["mute_enforcement_summaries"][1]["markdown_row"],
    }
    assert facts["vote_count"] == 2
    assert facts["speaking_row_count"] == 2
    assert [item["kind"] for item in facts["speaking_facts"]] == [
        "MAIN_TURN",
        "FOLLOWUP",
    ]
    assert facts["speaking_facts"][0]["planned_seconds"] == 600
    assert facts["speaking_facts"][0]["extension_seconds"] == 30
    assert facts["speaking_facts"][0]["finished_early"] is True
    assert facts["speaking_facts"][0]["absent_names"] == ["未获纪要授权成员"]
    speaking_payload = json.dumps(facts["speaking_facts"], ensure_ascii=False)
    assert "参会者乙" not in speaking_payload
    assert '"user_id": null' in speaking_payload
    assert "未获纪要授权成员" in speaking_payload
    assert all("user_id" not in item for item in facts["votes"])
    assert '"聆序"是本会议系统的系统名称' in system_message["content"]
    assert "投票只使用聚合票数" in system_message["content"]

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert "**[主持人甲]**" in markdown
    assert "**[聆序]**" in markdown
    assert "聆序机器人" not in markdown
    assert "evidence_ordinals" not in markdown
    assert "source_ordinals" not in markdown
    assert all(item["markdown_row"] in markdown for item in facts["agendas"])
    assert all(item["markdown_row"] in markdown for item in facts["votes"])
    assert all(row in markdown for row in facts["speaking_markdown_rows"])
    assert all(row in markdown for row in facts["speaking_plan_markdown_rows"])
    assert all(row in markdown for row in facts["mute_enforcement_markdown_rows"])
    manifest = json.loads(
        (meeting_dir / "full_transcript_minutes.json").read_text(encoding="utf-8")
    )
    assert manifest["model"] == APPROVED_TOPIC_MODEL
    assert manifest["thinking"] == "high"
    assert manifest["max_output_tokens"] == 131_072
    assert manifest["authoritative_agenda_count"] == 4
    assert manifest["authoritative_participant_count"] == 1
    assert manifest["authoritative_speaking_plan_count"] == 4
    assert manifest["authoritative_mute_enforcement_count"] == 4
    assert manifest["authoritative_vote_count"] == 2
    assert manifest["authoritative_speaking_row_count"] == 2
    assert len(manifest["authoritative_facts_sha256"]) == 64
    assert manifest["evidence_audit"]["transcript_binding_count"] == 2
    assert (
        manifest["evidence_audit"]["public_minutes_sha256"]
        == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    )
    assert len(manifest["evidence_audit_sha256"]) == 64
    assert (
        manifest["minutes_markdown_sha256"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    )
    assert await pipeline._full_transcript_minutes_are_current(session_id)
    reused = await pipeline._stage_generate_minutes(session_id)
    assert reused == {
        "final_output_method": "deepseek_full_transcript",
        "artifact_reused": True,
        "renderer_input_sha256": MinutesDocumentV2.model_validate_json(
            (meeting_dir / "minutes.json").read_text(encoding="utf-8")
        ).provenance.renderer_input_sha256,
    }
    assert final_provider.calls == 1

    manifest["template_sha256"] = "0" * 64
    (meeting_dir / "full_transcript_minutes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not await pipeline._full_transcript_minutes_are_current(session_id)

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_recovers_legacy_terminal_full_transcript_failure_with_manual_new_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_provider = FailingFullTranscriptMinutesProvider()
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="逐字稿和已验证议题结果已生成"),
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )
    service = WorkerService(
        jobs=jobs,
        worker_id="failed-full-transcript-worker",
        pipeline=pipeline,
    )

    # Reproduce the pre-ADR-0164 job state to retain coverage of historical manual recovery.
    monkeypatch.setattr(
        pipeline,
        "_generate_full_transcript_minutes",
        pipeline._generate_model_full_transcript_minutes,
    )

    assert await service.run_once()
    first_failure = await jobs.get(job_id)
    assert first_failure is not None and first_failure.status is JobStatus.FAILED
    assert first_failure.attempt_count == 1
    async with database.sessions() as db:
        stored_failure = await db.get(ProcessingJobRow, job_id)
        assert stored_failure is not None
        assert (
            stored_failure.last_error
            == "FullTranscriptPreviewError[preview_finish_reason_invalid]@GENERATE_MINUTES"
        )
    assert final_provider.calls == 1
    assert not await service.run_once()
    assert final_provider.calls == 1
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    first_attempt_path = meeting_dir / "full_transcript_minutes_attempt.json"
    first_attempt = json.loads(first_attempt_path.read_text(encoding="utf-8"))
    assert first_attempt["attempt_generation"] == 0
    assert first_attempt["status"] == "failed"
    assert first_attempt["failure_code"] == "preview_finish_reason_invalid"
    assert first_attempt["failure_diagnostics"] == {
        "phase": "response_validation",
        "elapsed_ms": 321,
        "finish_reason": "insufficient_system_resource",
        "content_state": "missing",
        "reasoning_state": "null",
        "reasoning_content_chars": 0,
        "response_body_bytes": 456,
        "usage_prompt_tokens": 100,
        "usage_completion_tokens": 0,
        "usage_total_tokens": 100,
    }
    assert not (meeting_dir / "full_transcript_minutes.json").exists()

    requeued = await jobs.retry_failed(
        session_id=session_id,
        job_type=POST_MEETING_JOB,
        authorize_content_generation_retry=True,
    )
    assert requeued is not None and requeued.status is JobStatus.PENDING
    assert requeued.checkpoint["full_transcript_minutes_attempt_generation"] == 1
    recovered_provider = FakeFullTranscriptMinutesProvider()
    pipeline._full_transcript_minutes_provider = recovered_provider

    assert await WorkerService(
        jobs=jobs,
        worker_id="manual-full-transcript-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert final_provider.calls == 1
    assert recovered_provider.calls == 1
    assert json.loads(first_attempt_path.read_text(encoding="utf-8")) == first_attempt
    manual_attempt = json.loads(
        (meeting_dir / "full_transcript_minutes_attempt.1.json").read_text(encoding="utf-8")
    )
    assert manual_attempt["attempt_generation"] == 1
    assert manual_attempt["status"] == "complete"
    assert manual_attempt["finish_reason"] == "stop"

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_reconciles_model_transcript_coverage_drift_without_losing_source_rows(
    tmp_path: Path,
) -> None:
    final_provider = CoverageDriftFullTranscriptMinutesProvider()
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="逐字稿和已验证议题结果已生成"),
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="coverage-reconciliation-worker",
        pipeline=pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert final_provider.calls == 1
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert "source_ordinals" not in markdown
    assert "evidence_ordinals" not in markdown
    assert "- [00:00:00.000-00:00:01.200] **[主持人甲]**:" in markdown
    assert "source_ordinals:999" not in markdown
    assert "错误覆盖内容" not in markdown
    manifest = json.loads(
        (meeting_dir / "full_transcript_minutes.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 8
    assert manifest["noise_removed"]["asr_control"] == 6
    assert manifest["transcript_reconciliation"] == {
        "accepted_model_group_count": 1,
        "accepted_model_source_count": 1,
        "fallback_source_count": 1,
        "model_line_count": 2,
        "rejected_model_line_count": 1,
        "source_row_count": 2,
    }
    assert manifest["evidence_audit"]["transcript_binding_count"] == 2
    assert [
        binding["source_ordinals"] for binding in manifest["evidence_audit"]["transcript_bindings"]
    ] == [[0], [1]]
    assert await pipeline._full_transcript_minutes_are_current(session_id)
    validated_minutes_sha256 = manifest["validated_minutes_sha256"]
    manifest["validated_minutes_sha256"] = "0" * 64
    (meeting_dir / "full_transcript_minutes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    manifest["validated_minutes_sha256"] = validated_minutes_sha256
    manifest["schema_version"] = 7
    (meeting_dir / "full_transcript_minutes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    manifest["schema_version"] = 8
    evidence_normalization = manifest["evidence_normalization"]
    manifest["evidence_normalization"] = {
        **evidence_normalization,
        "ordinal_lines_normalized": -1,
    }
    (meeting_dir / "full_transcript_minutes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    manifest["evidence_normalization"] = evidence_normalization
    manifest["transcript_reconciliation"]["fallback_source_count"] = 2
    (meeting_dir / "full_transcript_minutes.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_normalizes_evidence_without_inventing_action_owner(
    tmp_path: Path,
) -> None:
    final_provider = EvidenceDriftFullTranscriptMinutesProvider()
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="逐字稿和已验证议题结果已生成"),
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="evidence-normalization-worker",
        pipeline=pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert final_provider.calls == 1
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert "evidence_ordinals" not in markdown
    assert "source_ordinals" not in markdown
    assert "错误负责人" not in markdown
    assert "| 执行测试 | 待确认 | 未提及 |" in markdown
    manifest = json.loads(
        (meeting_dir / "full_transcript_minutes.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_normalization"] == {
        "action_owners_downgraded": 1,
        "ordinal_lines_normalized": 2,
    }
    assert len(manifest["evidence_normalization_revision"]) == 64
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_journals_safe_diagnostics_when_final_output_hits_limit(
    tmp_path: Path,
) -> None:
    final_provider = OutputLimitedFullTranscriptMinutesProvider()
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="已生成基础纪要"),
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="output-limited-worker",
        pipeline=pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert completed.last_error_type is None
    attempt = json.loads(
        (
            tmp_path / "sessions" / session_id / "meeting" / "full_transcript_minutes_attempt.json"
        ).read_text(encoding="utf-8")
    )
    assert attempt["failure_code"] == "final_minutes_generation_incomplete"
    assert attempt["failure_diagnostics"] == {
        "phase": "response_validation",
        "elapsed_ms": 401_000,
        "http_status": 200,
        "finish_reason": "length",
        "content_state": "present",
        "reasoning_state": "present",
        "reasoning_content_chars": 12_345,
        "response_body_bytes": 987_654,
        "usage_prompt_tokens": 1_000,
        "usage_completion_tokens": 65_536,
        "usage_total_tokens": 66_536,
    }
    assert "PRIVATE_TRUNCATED_MINUTES_MUST_NOT_BE_JOURNALED" not in json.dumps(
        attempt, ensure_ascii=False
    )
    assert final_provider.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_code",
    [
        "final_minutes_evidence_invalid",
        "final_minutes_consensus_evidence_invalid",
        "preview_transport_failed",
        "preview_finish_reason_invalid",
        "preview_request_too_large",
        "model_inventory_http_failed",
    ],
)
async def test_worker_publishes_reusable_basic_minutes_after_final_model_failure(
    tmp_path: Path,
    failure_code: str,
) -> None:
    class FailedProvider(FakeFullTranscriptMinutesProvider):
        async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
            self.calls += 1
            raise FullTranscriptPreviewError(failure_code)

    provider = FailedProvider()
    publisher = FakePublisher(expected_content="已生成基础纪要")
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=provider,
        full_transcript_minutes_template=_FULL_TRANSCRIPT_MINUTES_TEMPLATE,
    )
    service = WorkerService(jobs=jobs, worker_id="basic-minutes-worker", pipeline=pipeline)
    assert await service.run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert completed.attempt_count == 1
    assert provider.calls == 1
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    attempt_bytes = (meeting_dir / "full_transcript_minutes_attempt.json").read_bytes()
    assert json.loads(attempt_bytes)["failure_code"] == failure_code
    assert json.loads(attempt_bytes)["status"] == "failed"
    markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert "基础纪要" in markdown
    assert "原始发言摘录" in markdown
    assert "本节未完成智能整理, 请人工核对原始记录。" in markdown
    assert "source_ordinals" not in markdown
    assert "evidence_ordinals" not in markdown
    assert "## 投票结果" in markdown
    assert "## 权威发言流程" in markdown
    assert "**[主持人甲]**:" in markdown
    manifest_path = meeting_dir / "full_transcript_minutes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_method"] == "deterministic_basic"
    assert manifest["status"] == "fallback"
    assert manifest["fallback_reason"] == failure_code
    assert manifest["transcript_reconciliation"]["fallback_source_count"] == 2
    assert await pipeline._minutes_are_current(session_id)
    assert (await pipeline._stage_generate_minutes(session_id))["artifact_reused"]
    assert (await pipeline._stage_publish_discord(session_id))["deduplicated"]
    assert not await service.run_once()
    assert provider.calls == 1
    assert (meeting_dir / "full_transcript_minutes_attempt.json").read_bytes() == attempt_bytes

    # Even self-consistent public/audit hashes cannot authorize altered deterministic prose.
    (meeting_dir / "minutes.md").write_text(markdown + "\n伪造的额外结论。\n", encoding="utf-8")
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    (meeting_dir / "minutes.md").write_text(markdown, encoding="utf-8")
    manifest["basic_minutes_revision"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not await pipeline._full_transcript_minutes_are_current(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_never_publishes_rejected_model_prose_and_resumes_without_new_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidEvidenceProvider(FakeFullTranscriptMinutesProvider):
        async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
            result = await super().generate(messages)
            return replace(
                result,
                content=result.content.replace(
                    "- 无明确决策。", "- INVALID_MODEL_DECISION_WITHOUT_EVIDENCE"
                ),
            )

    provider = InvalidEvidenceProvider()
    publisher = FakePublisher(expected_content="已生成基础纪要")
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=provider,
        full_transcript_minutes_template=_FULL_TRANSCRIPT_MINUTES_TEMPLATE,
    )
    original_validate = pipeline._stage_validate_minutes_consistency

    async def interrupted_validation(session_id: str) -> dict[str, Any]:
        raise OSError("synthetic validation interruption")

    monkeypatch.setattr(pipeline, "_stage_validate_minutes_consistency", interrupted_validation)
    assert await WorkerService(jobs=jobs, worker_id="before-restart", pipeline=pipeline).run_once()
    pending = await jobs.get(job_id)
    assert pending is not None and pending.status is JobStatus.PENDING
    assert publisher.calls == 0
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    original_bytes = (meeting_dir / "minutes.md").read_bytes()
    assert b"INVALID_MODEL_DECISION_WITHOUT_EVIDENCE" not in original_bytes
    attempt_bytes = (meeting_dir / "full_transcript_minutes_attempt.json").read_bytes()
    assert json.loads(attempt_bytes)["failure_code"] == "final_minutes_evidence_invalid"
    monkeypatch.setattr(pipeline, "_stage_validate_minutes_consistency", original_validate)
    pipeline._minutes_renderer_inputs.clear()
    pipeline._full_transcript_attempt_generations.clear()
    assert await WorkerService(jobs=jobs, worker_id="after-restart", pipeline=pipeline).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert provider.calls == 1 and publisher.calls == 1
    assert (meeting_dir / "minutes.md").read_bytes() == original_bytes
    assert (meeting_dir / "full_transcript_minutes_attempt.json").read_bytes() == attempt_bytes
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["full_transcript_minutes_attempt_invalid", "transcript_input_invalid"]
)
async def test_basic_minutes_do_not_hide_integrity_failures(tmp_path: Path, code: str) -> None:
    class InvalidProvider(FakeFullTranscriptMinutesProvider):
        async def generate(self, messages: list[dict[str, str]]) -> FullTranscriptPreviewResult:
            self.calls += 1
            raise FullTranscriptPreviewError(code)

    provider = InvalidProvider()
    publisher = FakePublisher()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=provider,
        full_transcript_minutes_template=_FULL_TRANSCRIPT_MINUTES_TEMPLATE,
    )
    assert await WorkerService(
        jobs=jobs, worker_id="integrity-worker", pipeline=pipeline
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert publisher.calls == 0
    assert not (tmp_path / "sessions" / session_id / "meeting" / "minutes.md").exists()
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_records_partition_reason_and_does_not_retry_deterministic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, jobs, pipeline, _, _, _, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )

    async def fail_with_topic_capacity(
        job: ProcessingJobRecord,
        checkpoint: Any,
    ) -> None:
        await checkpoint(
            "SEGMENT_TOPICS",
            {
                "schema_version": 4,
                "completed_stages": ["NORMALISE_ENTITIES"],
                "active_stage": "SEGMENT_TOPICS",
            },
        )
        raise OutcomePartitionError("topic_boundary_call_cap")

    monkeypatch.setattr(pipeline, "run", fail_with_topic_capacity)
    service = WorkerService(
        jobs=jobs,
        worker_id="deterministic-capacity-worker",
        pipeline=pipeline,
        max_attempts=3,
    )

    assert await service.run_once()
    failed = await jobs.get(job_id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.attempt_count == 1
    assert failed.last_error_type == "OutcomePartitionError[topic_boundary_call_cap]@SEGMENT_TOPICS"
    assert not await service.run_once()
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_rebuilds_stale_event_export_and_reuses_completed_final_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_provider = FakeFullTranscriptMinutesProvider()
    publisher = FakePublisher(expected_content="逐字稿和已验证议题结果已生成")
    template_path = _FULL_TRANSCRIPT_MINUTES_TEMPLATE
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        full_transcript_minutes_provider=final_provider,
        full_transcript_minutes_template=template_path,
    )
    original_generate_minutes = pipeline._stage_generate_minutes

    async def generate_then_stale_events(selected_session_id: str) -> dict[str, Any]:
        details = await original_generate_minutes(selected_session_id)
        await pipeline._events.append(
            session_id=selected_session_id,
            event_type="EXPORT_RECONCILIATION_REQUIRED",
            source="TEST",
        )
        return details

    monkeypatch.setattr(pipeline, "_stage_generate_minutes", generate_then_stale_events)
    assert await WorkerService(
        jobs=jobs,
        worker_id="stale-events-first-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert final_provider.calls == 1
    assert publisher.calls == 0

    monkeypatch.setattr(pipeline, "_stage_generate_minutes", original_generate_minutes)
    requeued = await jobs.retry_failed(
        session_id=session_id,
        job_type=POST_MEETING_JOB,
        authorize_content_generation_retry=True,
    )
    assert requeued is not None and requeued.status is JobStatus.PENDING
    assert requeued.checkpoint["full_transcript_minutes_attempt_generation"] == 1

    assert await WorkerService(
        jobs=jobs,
        worker_id="stale-events-recovery-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert final_provider.calls == 1
    assert publisher.calls == 1
    assert await pipeline._meeting_events_are_current(session_id)
    assert (
        json.loads(
            (
                tmp_path
                / "sessions"
                / session_id
                / "meeting"
                / "full_transcript_minutes_attempt.json"
            ).read_text(encoding="utf-8")
        )["status"]
        == "complete"
    )

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_normalises_canonical_entities_and_excludes_control_context(
    tmp_path: Path,
) -> None:
    """Reviewed aliases reach summaries; backchannel/control entries never do."""

    summary_provider = FakeSummaryProvider()
    settings = Settings(
        _env_file=None,
        data_root=tmp_path / "sessions",
        glossary_root=tmp_path / "glossaries",
    )
    publisher = FakePublisher(expected_content="确定性证据验证")
    database, jobs, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        summary_provider,
        glossary_settings=settings,
    )
    meeting_dir = settings.data_root / session_id / "meeting"
    (meeting_dir / "glossary.yaml").write_text(
        "entities:\n  - canonical: CanonicalName\n    aliases: [alias-name]\n",
        encoding="utf-8",
    )
    alias_source_id = ""
    for source_key, text, started_ms in (
        ("legacy:alias", "alias-name", 3_000),
        ("legacy:backchannel", "嗯", 5_000),
        ("legacy:privacy-control", "停止录音", 7_000),
    ):
        record = await transcripts.upsert(
            session_id=session_id,
            source_key=source_key,
            user_id=100,
            raw_text=text,
            normalised_text=text,
            started_ms=started_ms,
            ended_ms=started_ms + 500,
        )
        if source_key == "legacy:alias":
            alias_source_id = record.id

    assert await WorkerService(
        jobs=jobs,
        worker_id="canonical-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    canonical_payload = json.loads(
        (meeting_dir / "canonical_transcript.json").read_text(encoding="utf-8")
    )
    by_type = {item["entry_type"]: item for item in canonical_payload["entries"]}
    alias_entry = next(
        item for item in canonical_payload["entries"] if item["text"] == "CanonicalName"
    )
    assert alias_entry["normalisation_changes"] == [
        {
            "source": "meeting_glossary",
            "original_text": "alias-name",
            "replacement_text": "CanonicalName",
            "started_offset": 0,
            "ended_offset": 10,
            "confidence": 1.0,
            "source_entry_ids": [alias_source_id],
        }
    ]
    assert by_type["backchannel"]["text"] == "嗯"
    assert by_type["privacy_control"]["text"] == "停止录音"
    assert "CanonicalName" in summary_provider.received_text
    assert "嗯" not in summary_provider.received_text
    assert "停止录音" not in summary_provider.received_text
    assert "嗯" not in summary_provider.topic_context_text
    assert "停止录音" not in summary_provider.topic_context_text
    assert any(
        item.artifact_type == "CANONICAL_TRANSCRIPT_JSON"
        for item in await artifacts.list(session_id)
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_outcome_artifact_reused_after_post_write_crash_without_second_model_call(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, _, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    crash_jobs = CrashAfterOutcomeWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="outcome-crash-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.PENDING
    outcome_path = tmp_path / "sessions" / session_id / "meeting" / "topic_outcome_candidates.json"
    first_bytes = outcome_path.read_bytes()
    assert provider.outcome_calls == 1

    provider.outcome_unavailable = True
    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="outcome-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await crash_jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert provider.outcome_calls == 1
    assert outcome_path.read_bytes() == first_bytes
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fix33_exact_wire_fold_terminal_authority_restarts_without_second_model_call(
    tmp_path: Path,
) -> None:
    provider = ExactWireFoldSummaryProvider()
    database, _, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    crash_jobs = CrashAfterOutcomeWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="wire-fold-crash-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.PENDING
    calls_before_retry = provider.outcome_calls
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    manifest = OutcomePartitionManifest.model_validate_json(
        (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
    )
    results = read_partition_results(
        meeting_dir / "outcome_partition_results.json",
        manifest,
        pipeline_revision=_OUTCOME_PIPELINE_REVISION,
    )
    assert sum(result.exact_duplicate_group_count for result in results.values()) == 1
    assert sum(result.dropped_wire_row_count for result in results.values()) == 1

    provider.outcome_unavailable = True
    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="wire-fold-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await crash_jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert provider.outcome_calls == calls_before_retry
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fix39_second_repair_success_restarts_from_terminal_cache_without_more_calls(
    tmp_path: Path,
) -> None:
    provider = SecondRepairSuccessSummaryProvider()
    database, _, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    crash_jobs = CrashAfterOutcomeWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="second-repair-crash-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.PENDING
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    journal_path = meeting_dir / "outcome_budget_journal.json"
    first_journal = journal_path.read_bytes()
    manifest = OutcomePartitionManifest.model_validate_json(
        (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
    )
    terminal_results = read_partition_results(
        meeting_dir / "outcome_partition_results.json",
        manifest,
        pipeline_revision=_OUTCOME_PIPELINE_REVISION,
    )
    action_results = tuple(
        result
        for (_, kind), result in terminal_results.items()
        if kind is OutcomeRequestKind.ACTION
    )
    assert len(action_results) == len(manifest.partitions)
    assert all(
        result.task_disposition is OutcomeTaskDisposition.STRICT_REPAIR_SUCCESS
        and [failure.attempt for failure in result.provider_failures]
        == [OutcomeFailureAttempt.INITIAL, OutcomeFailureAttempt.REPAIR]
        for result in action_results
    )

    provider.outcome_unavailable = True
    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="second-repair-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await crash_jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert journal_path.read_bytes() == first_journal
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normalisation_retry_rebuilds_from_immutable_canonical_input(
    tmp_path: Path,
) -> None:
    """A crash after artifact write cannot apply a second alias layer on retry."""

    settings = Settings(
        _env_file=None,
        data_root=tmp_path / "sessions",
        glossary_root=tmp_path / "glossaries",
    )
    database, _, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
        glossary_settings=settings,
    )
    meeting_dir = settings.data_root / session_id / "meeting"
    (meeting_dir / "glossary.yaml").write_text(
        "entities:\n"
        "  - canonical: beta\n"
        "    aliases: [alpha]\n"
        "  - canonical: gamma\n"
        "    aliases: [beta]\n",
        encoding="utf-8",
    )
    source = await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:restart-alias-chain",
        user_id=100,
        raw_text="alpha",
        normalised_text="alpha",
        started_ms=3_000,
        ended_ms=3_500,
    )
    crash_jobs = CrashAfterNormalisationWriteRepository(database.sessions)

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="normalise-crash-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await crash_jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.PENDING
    canonical_path = meeting_dir / "canonical_transcript.json"
    first_bytes = canonical_path.read_bytes()
    first_hash = hashlib.sha256(first_bytes).hexdigest()
    first_payload = json.loads(first_bytes)
    first_entry = next(item for item in first_payload["entries"] if item["text"] == "beta")
    assert first_entry["normalisation_changes"] == [
        {
            "source": "meeting_glossary",
            "original_text": "alpha",
            "replacement_text": "beta",
            "started_offset": 0,
            "ended_offset": 5,
            "confidence": 1.0,
            "source_entry_ids": [source.id],
        }
    ]

    assert await WorkerService(
        jobs=crash_jobs,
        worker_id="normalise-retry-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await crash_jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    second_bytes = canonical_path.read_bytes()
    assert second_bytes == first_bytes
    assert hashlib.sha256(second_bytes).hexdigest() == first_hash
    second_payload = json.loads(second_bytes)
    second_entry = next(item for item in second_payload["entries"] if item["text"] == "beta")
    assert second_entry["normalisation_changes"] == first_entry["normalisation_changes"]
    assert all(item["text"] != "gamma" for item in second_payload["entries"])
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_filters_by_event_consent_timeline_and_keeps_internal_revisions(
    tmp_path: Path,
) -> None:
    """Crossing/gap content is erased while prior revisions and public Bot TTS remain."""

    database = Database.open(_database_url(tmp_path / "worker-privacy.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    files = SessionFiles(tmp_path / "privacy-sessions")
    session = await MeetingSessionService(sessions, files).create(
        guild_id=81,
        creator_user_id=100,
    )
    participants = ParticipantRepository(database.sessions)
    await participants.join(
        session_id=session.id,
        user_id=100,
        display_name="主持人甲",
        is_host=True,
    )
    await participants.set_consent(
        session_id=session.id,
        user_id=100,
        kind=ConsentKind.MINUTES,
        granted=True,
    )

    events = MeetingEventStore(database.sessions)
    consent_kinds = [kind.value for kind in ConsentKind]
    for event_type, effective_at_ms in (
        ("CONSENT_GRANTED", 0),
        ("CONSENT_WITHDRAWN", 10_000),
        ("CONSENT_GRANTED", 20_000),
    ):
        await events.append(
            session_id=session.id,
            event_type=event_type,
            actor_user_id=100,
            source="CONSENT",
            payload={
                "subject_user_id": 100,
                "consent_kinds": consent_kinds,
                "effective_at_ms": effective_at_ms,
            },
        )

    utterances = RealtimeUtteranceRepository(database.sessions)
    utterance_intervals = (
        (1_000, 2_000, "eligible-before"),
        (9_000, 10_000, "ends-at-withdrawal"),
        (9_500, 10_500, "crosses-withdrawal"),
        (12_000, 13_000, "inside-gap"),
        (20_000, 21_000, "eligible-after-regrant"),
    )
    utterance_ids: list[str] = []
    for started_ms, ended_ms, marker in utterance_intervals:
        utterance_id = new_ulid()
        utterance_ids.append(utterance_id)
        await utterances.add(
            RealtimeUtteranceRecord(
                id=utterance_id,
                session_id=session.id,
                user_id=100,
                started_ms=started_ms,
                ended_ms=ended_ms,
                raw_text=marker,
                normalised_text=marker,
            )
        )

    transcripts = TranscriptRepository(database.sessions)
    await transcripts.upsert(
        session_id=session.id,
        source_key=f"utterance:{utterance_ids[0]}",
        user_id=100,
        raw_text="eligible-before",
        normalised_text="eligible-before",
        started_ms=1_000,
        ended_ms=2_000,
    )
    revision_history = [
        {
            "raw_text": "prior-revision",
            "normalised_text": "prior-revision",
            "reason": "human-correction",
        }
    ]
    async with database.sessions() as db, db.begin():
        await db.execute(
            update(TranscriptEntryRow)
            .where(
                TranscriptEntryRow.session_id == session.id,
                TranscriptEntryRow.source_key == f"utterance:{utterance_ids[0]}",
            )
            .values(revision_json=revision_history)
        )
    await transcripts.upsert(
        session_id=session.id,
        source_key="legacy:crossing",
        user_id=100,
        raw_text="legacy-crossing",
        normalised_text="legacy-crossing",
        started_ms=9_500,
        ended_ms=10_500,
    )
    await transcripts.upsert(
        session_id=session.id,
        source_key=f"BOT_TTS:{new_ulid()}",
        user_id=None,
        raw_text="public-bot-tts",
        normalised_text="public-bot-tts",
        started_ms=22_000,
        ended_ms=22_500,
    )

    segments = AudioSegmentRepository(database.sessions)
    segment_intervals = (
        (1_000, 2_000),
        (9_000, 10_000),  # raw Segment end is exclusive and remains eligible
        (9_500, 10_500),
        (12_000, 13_000),
        (20_000, 21_000),
    )
    segment_paths: dict[tuple[int, int], Path] = {}
    for segment_number, (started_ms, ended_ms) in enumerate(segment_intervals, start=1):
        relative_path = Path(f"audio/100/{segment_number:06d}.flac.enc")
        payload = f"encrypted-{segment_number}".encode()
        files.atomic_write(session.id, relative_path, payload)
        segment_paths[(started_ms, ended_ms)] = files.resolve(session.id, relative_path)
        await segments.add(
            AudioSegmentRecord(
                id=new_ulid(),
                session_id=session.id,
                user_id=100,
                segment_number=segment_number,
                encrypted_path=relative_path,
                sha256=hashlib.sha256(payload).hexdigest(),
                started_ms=started_ms,
                ended_ms=ended_ms,
                bytes_encrypted=len(payload),
            )
        )

    pipeline = PostMeetingPipeline(
        sessions=sessions,
        events=events,
        participants=participants,
        agenda=AgendaRepository(database.sessions),
        utterances=utterances,
        segments=segments,
        transcripts=transcripts,
        artifacts=ArtifactRepository(database.sessions),
        files=files,
        transcriber=None,
        publisher=None,
        retention_hours=24,
    )
    jobs = ProcessingJobRepository(database.sessions)
    job = await jobs.create(
        session_id=session.id,
        job_type=POST_MEETING_JOB,
        idempotency_key="privacy-filter-v1",
    )
    checkpoints: list[tuple[str, dict[str, object]]] = []

    async def checkpoint(stage: str, details: dict[str, object]) -> None:
        checkpoints.append((stage, details))

    await pipeline.run(job, checkpoint)

    entries = await transcripts.list(session.id)
    assert [(entry.started_ms, entry.ended_ms) for entry in entries] == [
        (1_000, 2_000),
        (20_000, 21_000),
        (22_000, 22_500),
    ]
    assert entries[0].revision_json == tuple(revision_history)
    meeting_dir = files.directory(session.id) / "meeting"
    raw_export = json.loads((meeting_dir / "transcript.json").read_text(encoding="utf-8"))
    assert raw_export[0]["revision_json"] == revision_history
    assert "revision_json" not in (meeting_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "prior-revision" not in (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    timeline_export = json.loads(
        (meeting_dir / "consent_timeline.json").read_text(encoding="utf-8")
    )
    assert timeline_export == {
        "schema_version": 1,
        "exclusion_journal_version": 1,
        "intervals": [
            {
                "user_id": 100,
                "consent_kind": kind.value,
                "started_ms": started_ms,
                "ended_ms": ended_ms,
            }
            for kind in sorted(ConsentKind, key=lambda value: value.value)
            for started_ms, ended_ms in ((0, 10_000), (20_000, None))
        ],
    }
    retained_segments = await segments.list_session(session.id)
    assert [(item.started_ms, item.ended_ms) for item in retained_segments] == [
        (1_000, 2_000),
        (9_000, 10_000),
        (20_000, 21_000),
    ]
    assert segment_paths[(9_500, 10_500)].exists() is False
    assert segment_paths[(12_000, 13_000)].exists() is False
    checkpoint_payload = json.dumps(checkpoints, ensure_ascii=False)
    for content_marker in (
        "eligible-before",
        "ends-at-withdrawal",
        "crosses-withdrawal",
        "inside-gap",
        "eligible-after-regrant",
        "public-bot-tts",
        "prior-revision",
    ):
        assert content_marker not in checkpoint_payload

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_transcribes_only_segment_gaps_not_realtime_covered_audio(
    tmp_path: Path,
) -> None:
    fake_transcriber = FakeSegmentTranscriber()
    publisher = FakePublisher()
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        transcriber=cast(SegmentTranscriber, fake_transcriber),
    )
    files = SessionFiles(tmp_path / "sessions")
    uncovered_path = Path("audio/100/000002.flac.enc")
    uncovered_payload = b"encrypted-uncovered-fixture"
    files.atomic_write(session_id, uncovered_path, uncovered_payload)
    await AudioSegmentRepository(database.sessions).add(
        AudioSegmentRecord(
            id=new_ulid(),
            session_id=session_id,
            user_id=100,
            segment_number=2,
            encrypted_path=uncovered_path,
            sha256=hashlib.sha256(uncovered_payload).hexdigest(),
            started_ms=5_000,
            ended_ms=6_000,
            bytes_encrypted=len(uncovered_payload),
        )
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="gap-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    assert fake_transcriber.calls == [(100, 5_000, 6_000)]

    entries = await transcripts.list(session_id)
    assert sum(entry.source_key.startswith("utterance:") for entry in entries) == 1
    assert sum(entry.source_key.startswith("segment:") for entry in entries) == 1
    assert sum(entry.source_key.startswith("BOT_TTS:") for entry in entries) == 1
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    assert (meeting_dir / "transcript.txt").read_text(encoding="utf-8").count("仅缺口补录") == 1
    assert (meeting_dir / "minutes.md").read_text(encoding="utf-8").count("仅缺口补录") == 1
    assert publisher.calls == 1

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_restart_resumes_checkpoint_and_retry_does_not_duplicate_artifacts(
    tmp_path: Path,
) -> None:
    failing = FakePublisher(fail=True)
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        failing,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="crashed-worker",
        pipeline=pipeline,
        max_attempts=3,
    ).run_once()
    retry = await jobs.get(job_id)
    assert retry is not None
    assert retry.status is JobStatus.PENDING
    assert retry.current_stage == "PUBLISH_DISCORD"
    assert retry.checkpoint["active_stage"] == "PUBLISH_DISCORD"
    records_after_timeout = await artifacts.list(session_id)
    assert len(records_after_timeout) == 15
    prepared = [
        record
        for record in records_after_timeout
        if record.artifact_type == "DISCORD_PUBLICATION_INTENT"
    ]
    assert len(prepared) == 1
    assert prepared[0].metadata["state"] == "PREPARED"
    assert not [
        record for record in records_after_timeout if record.artifact_type == "DISCORD_PUBLICATION"
    ]

    recovered_publisher = FakePublisher()
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="recovered-worker",
        pipeline=recovered_pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert recovered_publisher.calls == 0
    assert failing.calls == 1
    records_after_recovery = await artifacts.list(session_id)
    assert len(records_after_recovery) == 15
    assert [
        record.metadata["state"]
        for record in records_after_recovery
        if record.artifact_type == "DISCORD_PUBLICATION_INTENT"
    ] == ["PREPARED"]
    assert not [
        record for record in records_after_recovery if record.artifact_type == "DISCORD_PUBLICATION"
    ]

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema_two_unfinished_job_restarts_at_first_privacy_stage(
    tmp_path: Path,
) -> None:
    """Old rendering checkpoints cannot skip the newly inserted privacy stages."""

    publisher = FakePublisher()
    database, jobs, pipeline, _, _, _, job_id = await _pipeline_fixture(tmp_path, publisher)
    retry = await _seed_schema_two_unfinished_job(jobs, job_id)
    active_stages: list[str] = []

    async def checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await pipeline.run(retry, checkpoint)

    assert active_stages[:2] == ["BUILD_CONSENT_TIMELINE", "APPLY_PRIVACY_CUTOFFS"]
    assert "VERIFY_AUDIO_SEGMENTS" in active_stages[2:]
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_schema_three_checkpoint_reruns_canonical_dependants(
    tmp_path: Path,
) -> None:
    """A pre-canonical schema-3 checkpoint cannot preserve stale downstream completion."""

    publisher = FakePublisher()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path, publisher
    )
    retry = await _seed_legacy_schema_three_job(jobs, job_id)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    meeting_dir.mkdir(exist_ok=True)
    (meeting_dir / "consent_timeline.json").write_text(
        '{"schema_version":1,"intervals":[]}\n',
        encoding="utf-8",
    )
    active_stages: list[str] = []

    async def checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await pipeline.run(retry, checkpoint)

    assert active_stages[:3] == [
        "CANONICALISE_TRANSCRIPT",
        "NORMALISE_ENTITIES",
        "SEGMENT_TOPICS",
    ]
    assert active_stages[3:6] == [
        "EXTRACT_TOPIC_OUTCOMES",
        "LINK_EVIDENCE",
        "VALIDATE_OUTCOMES",
    ]
    assert active_stages[6] == "MERGE_TRANSCRIPT"
    assert "RENDER_ARTIFACTS" in active_stages
    minutes = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert minutes["privacy_exclusions"]["measurement_available"] is False
    journal = json.loads((meeting_dir / "privacy_exclusions.json").read_text(encoding="utf-8"))
    assert journal["state"] == "legacy_unavailable"
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_current_task9_checkpoint_migrates_journal_before_version_checkpoint(
    tmp_path: Path,
) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    retry = await _seed_legacy_schema_three_job(
        jobs,
        job_id,
        outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    meeting_dir.mkdir(exist_ok=True)
    (meeting_dir / "consent_timeline.json").write_text(
        '{"schema_version":1,"intervals":[]}\n',
        encoding="utf-8",
    )
    captured_checkpoint: dict[str, object] | None = None

    async def crash_after_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal captured_checkpoint
        assert (meeting_dir / "privacy_exclusions.json").is_file()
        captured_checkpoint = dict(payload)
        raise RuntimeError("content-safe migration checkpoint interruption")

    with pytest.raises(RuntimeError, match="migration checkpoint"):
        await pipeline.run(retry, crash_after_checkpoint)
    assert captured_checkpoint is not None
    assert captured_checkpoint["privacy_exclusion_journal_version"] == 1
    journal = json.loads((meeting_dir / "privacy_exclusions.json").read_text(encoding="utf-8"))
    assert journal["state"] == "legacy_unavailable"

    migrated_retry = replace(retry, checkpoint=captured_checkpoint)
    checkpoint_after_retry: dict[str, object] | None = None

    async def prove_post_checkpoint_retry(stage: str, payload: dict[str, object]) -> None:
        nonlocal checkpoint_after_retry
        checkpoint_after_retry = dict(payload)
        raise RuntimeError("content-safe post-checkpoint interruption")

    with pytest.raises(RuntimeError, match="post-checkpoint"):
        await pipeline.run(migrated_retry, prove_post_checkpoint_retry)
    assert checkpoint_after_retry is not None
    assert checkpoint_after_retry["privacy_exclusion_journal_version"] == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("crash_window", ("before_replace", "after_replace"))
async def test_legacy_privacy_journal_atomic_write_windows_retry_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_window: str,
) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    retry = await _seed_legacy_schema_three_job(
        jobs,
        job_id,
        outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    meeting_dir.mkdir(exist_ok=True)
    (meeting_dir / "consent_timeline.json").write_text(
        '{"schema_version":1,"intervals":[]}\n',
        encoding="utf-8",
    )
    journal_path = meeting_dir / "privacy_exclusions.json"
    original_atomic_write = pipeline_module.atomic_write_text
    interrupted = False

    def interrupt_migration_write(path: Path, payload: str) -> str:
        nonlocal interrupted
        if path == journal_path and not interrupted:
            interrupted = True
            if crash_window == "before_replace":
                raise RuntimeError("content-safe pre-replace migration interruption")
            original_atomic_write(path, payload)
            raise RuntimeError("content-safe post-replace migration interruption")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(pipeline_module, "atomic_write_text", interrupt_migration_write)

    async def checkpoint_must_not_run(stage: str, payload: dict[str, object]) -> None:
        raise AssertionError((stage, payload.get("schema_version")))

    with pytest.raises(RuntimeError, match="migration interruption"):
        await pipeline.run(retry, checkpoint_must_not_run)
    assert journal_path.is_file() is (crash_window == "after_replace")

    monkeypatch.setattr(pipeline_module, "atomic_write_text", original_atomic_write)
    bytes_before_retry = journal_path.read_bytes() if journal_path.is_file() else None

    async def crash_after_safe_retry(stage: str, payload: dict[str, object]) -> None:
        assert journal_path.is_file()
        assert payload["privacy_exclusion_measurement_state"] == "legacy_unavailable"
        raise RuntimeError("content-safe retry checkpoint interruption")

    with pytest.raises(RuntimeError, match="retry checkpoint"):
        await pipeline.run(retry, crash_after_safe_retry)
    if bytes_before_retry is not None:
        assert journal_path.read_bytes() == bytes_before_retry
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prefingerprint_schema_three_checkpoint_regenerates_outcomes_and_public_minutes(
    tmp_path: Path,
) -> None:
    marker = "LEGACY_PREFINGERPRINT_NARRATIVE"
    seed_publisher = FakePublisher(fail=True)
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        seed_publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, seed_checkpoint)

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    (meeting_dir / "validated_outcomes.json").write_text(
        json.dumps({"legacy": marker}),
        encoding="utf-8",
    )
    (meeting_dir / "evidence.json").write_text(
        json.dumps([{"legacy": marker}]),
        encoding="utf-8",
    )
    (meeting_dir / "minutes.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "meeting_summary": {"executive_summary": marker},
                "agenda_summaries": [{"summary": {"executive_summary": marker}}],
            }
        ),
        encoding="utf-8",
    )
    (meeting_dir / "minutes.md").write_text(marker, encoding="utf-8")
    (meeting_dir / "privacy_exclusions.json").unlink()
    timeline_path = meeting_dir / "consent_timeline.json"
    legacy_timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    legacy_timeline.pop("exclusion_journal_version", None)
    timeline_path.write_text(json.dumps(legacy_timeline), encoding="utf-8")
    for name, artifact_type in (
        ("evidence.json", "EVIDENCE_JSON"),
        ("minutes.json", "MINUTES_JSON"),
        ("minutes.md", "MINUTES_MARKDOWN"),
    ):
        await artifacts.add_file(
            session_id=session_id,
            artifact_type=artifact_type,
            path=meeting_dir / name,
            metadata={"pipeline_version": 3},
        )
    await _seed_prefingerprint_schema_three_job(jobs, job_id)
    recovered_publisher = FakePublisher(expected_content="确定性")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="prefingerprint-recovery-worker",
        pipeline=recovered_pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    revision = completed.checkpoint.get("outcome_pipeline_revision")
    assert isinstance(revision, str) and len(revision) == 64
    rendered = "\n".join(
        (meeting_dir / name).read_text(encoding="utf-8")
        for name in (
            "validated_outcomes.json",
            "evidence.json",
            "minutes.json",
            "minutes.md",
        )
    )
    assert marker not in rendered
    validated = json.loads((meeting_dir / "validated_outcomes.json").read_text(encoding="utf-8"))
    minutes = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert validated["provenance"]["outcome_pipeline_revision"] == revision
    assert minutes["provenance"]["outcome_pipeline_revision"] == revision
    assert minutes["privacy_exclusions"] == {
        "measurement_available": False,
        "segment_files": 0,
        "segment_rows": 0,
        "transcript_entries": 0,
    }
    for artifact in await artifacts.list(session_id):
        if artifact.artifact_type not in {"EVIDENCE_JSON", "MINUTES_JSON", "MINUTES_MARKDOWN"}:
            continue
        assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    assert recovered_publisher.calls == 1
    assert marker not in "\n".join(recovered_publisher.contents)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_deepseek_fake_http_uses_one_unified_durable_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    authorizations: list[str | None] = []

    async def models(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": APPROVED_TOPIC_MODEL,
                        "object": "model",
                        "owned_by": "deepseek",
                    }
                ],
            }
        )

    async def chat(request: web.Request) -> web.Response:
        authorizations.append(request.headers.get("Authorization"))
        payload = await request.json()
        requests.append(payload)
        context = json.loads(payload["messages"][1]["content"])
        if "candidate" in context:
            response = {"confirmed": True, "confidence": 0.9}
        else:
            source_ref = context["c"][0][0]
            kind = OutcomeRequestKind(context["k"])
            candidates: dict[OutcomeRequestKind, dict[str, object]] = {
                OutcomeRequestKind.DECISION_AGREEMENT: {
                    "type": "decision",
                    "text": "Synthetic decision",
                    "source_entry_refs": [source_ref],
                    "source_event_refs": [],
                },
                OutcomeRequestKind.ACTION: {
                    "text": "Synthetic action",
                    "source_entry_refs": [source_ref],
                    "owner_ref": None,
                    "deadline": None,
                },
                OutcomeRequestKind.QUESTION: {
                    "text": "Synthetic question",
                    "source_entry_refs": [source_ref],
                },
                OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: {
                    "type": "risk",
                    "text": "Synthetic risk",
                    "source_entry_refs": [source_ref],
                },
            }
            response = {"complete": True, "candidates": [candidates[kind]]}
        content = json.dumps(response)
        return web.json_response(
            {
                "model": APPROVED_TOPIC_MODEL,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/models", models)
    app.router.add_post("/chat/completions", chat)
    server = TestServer(app)
    await server.start_server()
    fake_base_url = str(server.make_url("/")).rstrip("/")
    monkeypatch.setattr("lingxu.worker.summary.APPROVED_TOPIC_BASE_URL", fake_base_url)
    monkeypatch.setattr("lingxu.worker.topics.APPROVED_TOPIC_BASE_URL", fake_base_url)
    provider = DeepSeekPostMeetingInferenceProvider(
        base_url=fake_base_url,
        model=APPROVED_TOPIC_MODEL,
        expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
        api_key="test-only-sentinel",
        allow_remote=True,
    )
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:unified-ledger-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def stop_after_outcomes(stage: str, checkpoint: dict[str, Any]) -> None:
        if stage == "EXTRACT_TOPIC_OUTCOMES" and "last_stage_details" in checkpoint:
            raise RuntimeError("intentional test boundary")

    try:
        with pytest.raises(RuntimeError, match="intentional test boundary"):
            await pipeline.run(job, stop_after_outcomes)
    finally:
        await server.close()

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    journal_payload = json.loads(
        (meeting_dir / "outcome_budget_journal.json").read_text(encoding="utf-8")
    )
    journal = UnifiedLlmBudgetJournal.model_validate_json(
        json.dumps(journal_payload["journal"], separators=(",", ":"))
    )
    manifest = OutcomePartitionManifest.model_validate_json(
        (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
    )
    assert journal.actual_http_requests == 10
    assert journal.actual_inference_calls == 9
    assert len(requests) == 9
    assert authorizations == ["Bearer test-only-sentinel"] * 10
    assert provider.actual_chat_requests == journal.actual_inference_calls
    assert [action.kind for action in journal.actions].count(LlmActionKind.DIGEST) == 1
    assert [action.kind for action in journal.actions].count(LlmActionKind.BOUNDARY_CHAT) == 1
    assert [action.kind for action in journal.actions].count(LlmActionKind.OUTCOME_INITIAL) == 8
    expected_contexts = {
        serialize_outcome_request(partition, kind).decode("utf-8")
        for partition in manifest.partitions
        for kind in OutcomeRequestKind
    }
    outcome_requests = [
        request for request in requests if '"k":' in request["messages"][1]["content"]
    ]
    assert {request["messages"][1]["content"] for request in outcome_requests} == expected_contexts
    expected_labels = {
        OutcomeRequestKind.DECISION_AGREEMENT: "本次只提取: 决议或明确一致意见",
        OutcomeRequestKind.ACTION: "本次只提取: 行动项",
        OutcomeRequestKind.QUESTION: "本次只提取: 未决问题",
        OutcomeRequestKind.PROPOSAL_RISK_DISAGREEMENT: "本次只提取: 提议、风险或分歧",
    }
    for request in outcome_requests:
        kind = OutcomeRequestKind(json.loads(request["messages"][1]["content"])["k"])
        assert expected_labels[kind] in request["messages"][0]["content"]
        matching_partition = next(
            partition
            for partition in manifest.partitions
            if partition.context_json(kind) == request["messages"][1]["content"]
        )
        expected_schema = json.dumps(
            outcome_response_schema(matching_partition, kind),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert expected_schema in request["messages"][0]["content"]
        assert request["thinking"] == {"type": "disabled"}
        assert request["response_format"] == {"type": "json_object"}
    outcome_payload = json.loads(
        (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
    )
    candidates = [
        candidate for topic in outcome_payload["topics"] for candidate in topic["candidates"]
    ]
    assert {candidate["type"] for candidate in candidates} == {
        "decision",
        "action_item",
        "open_question",
        "risk",
    }
    assert len({candidate["id"] for candidate in candidates}) == len(candidates)
    assert all(candidate["requires_confirmation"] is False for candidate in candidates)
    assert all(candidate["verification_status"] == "unverified" for candidate in candidates)
    assert all(candidate["confidence"] == 0.0 for candidate in candidates)
    assert all(not topic["provider_failures"] for topic in outcome_payload["topics"])
    journal.validate_final()
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "failure"),
    (
        ("result_only", "partition_result_incomplete"),
        ("journal_and_result", "budget_expected_mismatch"),
    ),
)
async def test_outcome_reuse_rejects_self_consistent_cartesian_deletion(
    tmp_path: Path,
    mutation: str,
    failure: str,
) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        FakeSummaryProvider(),
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def stop_after_outcomes(stage: str, checkpoint: dict[str, Any]) -> None:
        if stage == "EXTRACT_TOPIC_OUTCOMES" and "last_stage_details" in checkpoint:
            raise RuntimeError("intentional test boundary")

    with pytest.raises(RuntimeError, match="intentional test boundary"):
        await pipeline.run(job, stop_after_outcomes)

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    results_path = meeting_dir / "outcome_partition_results.json"
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    removed = results_payload["results"].pop()
    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if mutation == "journal_and_result":
        manifest = OutcomePartitionManifest.model_validate_json(
            (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
        )
        partition = next(
            item for item in manifest.partitions if item.partition_id == removed["partition_id"]
        )
        task_hash = outcome_task_hash(
            llm_job_hash(
                canonical_hash=manifest.canonical_hash,
                pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            ),
            partition,
            OutcomeRequestKind(removed["request_kind"]),
        )
        journal_path = meeting_dir / "outcome_budget_journal.json"
        journal_payload = json.loads(journal_path.read_text(encoding="utf-8"))
        for key in (
            "expected_task_hashes",
            "reserved_task_hashes",
            "completed_task_hashes",
            "result_task_hashes",
        ):
            journal_payload["journal"][key].remove(task_hash)
        journal_payload["journal"]["actions"] = [
            action
            for action in journal_payload["journal"]["actions"]
            if action["task_hash"] != task_hash
        ]
        journal_payload["journal"]["actual_http_requests"] = len(
            journal_payload["journal"]["actions"]
        )
        journal_payload["journal"]["actual_inference_calls"] = sum(
            action["kind"] != LlmActionKind.DIGEST.value
            for action in journal_payload["journal"]["actions"]
        )
        journal_path.write_text(
            json.dumps(journal_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(OutcomePartitionError, match=failure):
        await pipeline._stage_extract_topic_outcomes(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_schema_version", (10, 11))
async def test_fix26_restart_rejects_legacy_manifest_without_model_call(
    tmp_path: Path,
    legacy_schema_version: int,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def stop_after_outcomes(stage: str, checkpoint: dict[str, Any]) -> None:
        if stage == "EXTRACT_TOPIC_OUTCOMES" and "last_stage_details" in checkpoint:
            raise RuntimeError("intentional test boundary")

    with pytest.raises(RuntimeError, match="intentional test boundary"):
        await pipeline.run(job, stop_after_outcomes)

    manifest_path = (
        tmp_path / "sessions" / session_id / "meeting" / "outcome_partition_manifest.json"
    )
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = legacy_schema_version
    if legacy_schema_version == 10:
        legacy["context_target_chars"] = 4_000
        for field in (
            "target_ladder",
            "target_ladder_revision",
            "target_attempts",
            "planning_search_state_limit",
            "planning_search_states_used",
            "response_schema_hashes",
        ):
            legacy.pop(field)
    else:
        legacy.pop("response_schema_hashes")
    manifest_path.write_text(
        json.dumps(legacy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    calls_before_restart = provider.outcome_calls

    with pytest.raises(OutcomePartitionError, match="manifest_tamper"):
        await pipeline._stage_extract_topic_outcomes(session_id)

    assert provider.outcome_calls == calls_before_restart
    assert (
        json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"]
        == legacy_schema_version
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_current_exact_v1_disposition_metadata_migrates_without_additional_model_calls(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def stop_after_outcomes(stage: str, checkpoint: dict[str, Any]) -> None:
        if stage == "EXTRACT_TOPIC_OUTCOMES" and "last_stage_details" in checkpoint:
            raise RuntimeError("intentional test boundary")

    with pytest.raises(RuntimeError, match="intentional test boundary"):
        await pipeline.run(job, stop_after_outcomes)

    results_path = tmp_path / "sessions" / session_id / "meeting" / "outcome_partition_results.json"
    legacy = json.loads(results_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 1
    for result in legacy["results"]:
        for field in ("schema_version", "disposition_revision", "task_disposition"):
            result.pop(field)
        result["result_hash"] = _test_payload_hash(
            {key: value for key, value in result.items() if key != "result_hash"}
        )
    results_path.write_text(
        json.dumps(legacy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    calls_before_restart = provider.outcome_calls

    details = await pipeline._stage_extract_topic_outcomes(session_id)

    assert provider.outcome_calls == calls_before_restart
    migrated = json.loads(results_path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert all(result["schema_version"] == 2 for result in migrated["results"])
    assert details["recovered_initial_incidents"] == 0
    assert details["terminal_provider_failure_incidents"] == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_current_partial_v1_disposition_metadata_fails_closed_without_model_calls(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def stop_after_outcomes(stage: str, checkpoint: dict[str, Any]) -> None:
        if stage == "EXTRACT_TOPIC_OUTCOMES" and "last_stage_details" in checkpoint:
            raise RuntimeError("intentional test boundary")

    with pytest.raises(RuntimeError, match="intentional test boundary"):
        await pipeline.run(job, stop_after_outcomes)

    results_path = tmp_path / "sessions" / session_id / "meeting" / "outcome_partition_results.json"
    legacy = json.loads(results_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = 1
    legacy["results"].pop()
    for result in legacy["results"]:
        for field in ("schema_version", "disposition_revision", "task_disposition"):
            result.pop(field)
        result["result_hash"] = _test_payload_hash(
            {key: value for key, value in result.items() if key != "result_hash"}
        )
    results_path.write_text(
        json.dumps(legacy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    calls_before_restart = provider.outcome_calls

    with pytest.raises(OutcomePartitionError, match="outcome_disposition_migration_required"):
        await pipeline._stage_extract_topic_outcomes(session_id)

    assert provider.outcome_calls == calls_before_restart
    assert json.loads(results_path.read_text(encoding="utf-8"))["schema_version"] == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retired_fix9_checkpoint_fails_closed_before_pipeline_stage(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, _, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    stale = replace(
        job,
        checkpoint={
            "schema_version": 4,
            "outcome_pipeline_revision": pipeline_module._RETIRED_FIX9_OUTCOME_PIPELINE_REVISION,
            "completed_stages": ["VERIFY_SESSION", "SEGMENT_TOPICS"],
            "active_stage": "EXTRACT_TOPIC_OUTCOMES",
        },
    )

    async def checkpoint_must_not_run(stage: str, payload: dict[str, object]) -> None:
        raise AssertionError((stage, payload))

    with pytest.raises(
        OutcomePartitionError,
        match="retired_fix9_checkpoint_requires_matching_rollback",
    ):
        await pipeline.run(stale, checkpoint_must_not_run)

    assert provider.topic_calls == provider.outcome_calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_minutes_consistency_rejects_deleted_physical_partition_result(
    tmp_path: Path,
) -> None:
    database, jobs, pipeline, _, _, session_id, _ = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        FakeSummaryProvider(),
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="partition-cartesian-consistency-worker",
        pipeline=pipeline,
    ).run_once()

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    results_path = meeting_dir / "outcome_partition_results.json"
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    results_payload["results"].pop()
    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(OutcomePartitionError, match="partition_result_incomplete"):
        await pipeline._minutes_consistency_context(
            session_id,
            require_registered_artifacts=True,
        )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("corrupt_name", "expected_outcome_calls"),
    (("validated_outcomes.json", 1), ("minutes.json", 1)),
)
async def test_current_revision_validates_outcome_and_minutes_provenance_before_reuse(
    tmp_path: Path,
    corrupt_name: str,
    expected_outcome_calls: int,
) -> None:
    marker = "CURRENT_REVISION_HASH_MISMATCH"
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    seen_revision: str | None = None
    seen_speech_revision: str | None = None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal seen_revision, seen_speech_revision
        revision = payload.get("outcome_pipeline_revision")
        if isinstance(revision, str):
            seen_revision = revision
        speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(speech_revision, str):
            seen_speech_revision = speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, seed_checkpoint)
    assert seen_revision is not None
    assert seen_speech_revision is not None
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    renderer_input_digest = _minutes_renderer_input(tmp_path, session_id)
    (meeting_dir / corrupt_name).write_text(marker, encoding="utf-8")
    await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=seen_revision,
        speech_revision=seen_speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=renderer_input_digest,
    )
    recovered_publisher = FakePublisher(expected_content="确定性证据验证")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id=f"hash-recovery-{corrupt_name}",
        pipeline=recovered_pipeline,
    ).run_once()

    assert marker not in (meeting_dir / corrupt_name).read_text(encoding="utf-8")
    assert provider.outcome_calls == expected_outcome_calls
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper_kind",
    (
        "valid_as_rejected",
        "ambiguous_as_individual_rejection",
        "error_types_rewrite",
        "origin_diagnostics_rewrite",
        "linked_status_rewrite",
        "full_chain_candidate_delete",
        "full_chain_candidate_fabricate",
    ),
)
async def test_fix27_validated_currentness_rebuilds_evidence_authority_before_reuse(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publication(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("content-safe validation-currentness interruption")

    with pytest.raises(TimeoutError, match="validation-currentness"):
        await pipeline.run(job, interrupt_before_publication)
    model_calls_before = provider.outcome_calls
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    candidate_path = meeting_dir / "topic_outcome_candidates.json"
    linked_path = meeting_dir / "linked_outcomes.json"
    validated_path = meeting_dir / "validated_outcomes.json"
    candidate_artifact = json.loads(candidate_path.read_text(encoding="utf-8"))
    linked = json.loads(linked_path.read_text(encoding="utf-8"))
    validated = json.loads(validated_path.read_text(encoding="utf-8"))
    topic_index = next(
        index for index, topic in enumerate(validated["topics"]) if topic["candidates"]
    )
    topic_id = validated["topics"][topic_index]["topic_id"]
    linked_topic = next(topic for topic in linked["topics"] if topic["topic_id"] == topic_id)
    candidate = validated["topics"][topic_index]["candidates"][0]
    metric = next(item for item in validated["validation_metrics"] if item["topic_id"] == topic_id)

    if tamper_kind == "valid_as_rejected":
        validated["topics"][topic_index]["candidates"] = []
        validated["binding_resolutions"] = [
            item
            for item in validated["binding_resolutions"]
            if candidate["id"] not in item["contributor_candidate_ids"]
        ]
        validated["individual_rejections"].append(
            {"topic_id": topic_id, "candidate_id": candidate["id"]}
        )
        metric.update(
            {
                "valid_count": 0,
                "rejected_count": 1,
                "individual_rejected_count": 1,
                "ambiguous_binding_count": 0,
                "ambiguous_rejected_count": 0,
                "coalesced_count": 0,
                "error_types": {"rewritten_validation": 1},
            }
        )
    elif tamper_kind == "ambiguous_as_individual_rejection":
        alternate = {**candidate, "id": f"{candidate['id']}-alternate", "confidence": 0.5}
        linked_topic["candidates"].append(alternate)
        linked["provenance"]["linked_artifact_hash"] = _test_payload_hash(
            outcome_hash_input(linked)
        )
        linked_path.write_text(
            json.dumps(linked, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validated["provenance"]["linked_artifact_sha256"] = hashlib.sha256(
            linked_path.read_bytes()
        ).hexdigest()
        validated["individual_rejections"].append(
            {"topic_id": topic_id, "candidate_id": alternate["id"]}
        )
        metric.update(
            {
                "candidate_count": 2,
                "valid_count": 1,
                "rejected_count": 1,
                "individual_rejected_count": 1,
                "ambiguous_binding_count": 0,
                "ambiguous_rejected_count": 0,
                "coalesced_count": 0,
                "error_types": {"rewritten_validation": 1},
            }
        )
    elif tamper_kind == "error_types_rewrite":
        metric["error_types"] = {"rewritten_validation": 1}
    elif tamper_kind == "origin_diagnostics_rewrite":
        resolution = next(
            item
            for item in validated["binding_resolutions"]
            if candidate["id"] in item["contributor_candidate_ids"]
        )
        resolution["difference_fields"] = ["text"]
    elif tamper_kind == "linked_status_rewrite":
        linked_candidate = next(
            item for item in linked_topic["candidates"] if item["id"] == candidate["id"]
        )
        linked_candidate["verification_status"] = "not_checked"
        validated["topics"][topic_index]["candidates"][0]["verification_status"] = "not_checked"
    else:
        candidate_topic = next(
            topic for topic in candidate_artifact["topics"] if topic["topic_id"] == topic_id
        )
        if tamper_kind == "full_chain_candidate_delete":
            candidate_topic["candidates"] = [
                item for item in candidate_topic["candidates"] if item["id"] != candidate["id"]
            ]
            linked_topic["candidates"] = [
                item for item in linked_topic["candidates"] if item["id"] != candidate["id"]
            ]
            validated["topics"][topic_index]["candidates"] = []
            validated["binding_resolutions"] = [
                item
                for item in validated["binding_resolutions"]
                if candidate["id"] not in item["contributor_candidate_ids"]
            ]
            metric.update(
                {
                    "candidate_count": 0,
                    "valid_count": 0,
                    "rejected_count": 0,
                    "individual_rejected_count": 0,
                    "ambiguous_binding_count": 0,
                    "ambiguous_rejected_count": 0,
                    "coalesced_count": 0,
                    "error_types": {},
                }
            )
        else:
            alternate = {**candidate, "id": f"{candidate['id']}-fabricated"}
            candidate_topic["candidates"].append(alternate)
        candidate_artifact["provenance"]["outcome_artifact_hash"] = _test_payload_hash(
            outcome_hash_input(candidate_artifact)
        )
        candidate_path.write_text(
            json.dumps(candidate_artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        linked["provenance"]["candidate_artifact_sha256"] = hashlib.sha256(
            candidate_path.read_bytes()
        ).hexdigest()

    if tamper_kind in {
        "ambiguous_as_individual_rejection",
        "linked_status_rewrite",
        "full_chain_candidate_delete",
        "full_chain_candidate_fabricate",
    }:
        linked["provenance"]["linked_artifact_hash"] = _test_payload_hash(
            outcome_hash_input(linked)
        )
        linked_path.write_text(
            json.dumps(linked, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validated["provenance"]["linked_artifact_sha256"] = hashlib.sha256(
            linked_path.read_bytes()
        ).hexdigest()

    validated["individual_rejections"].sort(
        key=lambda item: (item["topic_id"], item["candidate_id"])
    )
    validated["provenance"]["validated_artifact_hash"] = _test_payload_hash(
        validated_outcome_hash_input(validated)
    )
    validated_path.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if tamper_kind.startswith("full_chain"):
        assert not await pipeline._candidate_outcomes_are_current(session_id)
    if tamper_kind in {
        "ambiguous_as_individual_rejection",
        "linked_status_rewrite",
        "full_chain_candidate_delete",
        "full_chain_candidate_fabricate",
    }:
        assert not await pipeline._linked_outcomes_are_current(session_id)
    assert await pipeline._current_validated_outcomes(session_id) is None
    assert provider.outcome_calls == model_calls_before
    if tamper_kind.startswith("full_chain"):
        await pipeline._stage_extract_topic_outcomes(session_id)
    if tamper_kind in {
        "ambiguous_as_individual_rejection",
        "linked_status_rewrite",
        "full_chain_candidate_delete",
        "full_chain_candidate_fabricate",
    }:
        await pipeline._stage_link_evidence(session_id)
    await pipeline._stage_validate_outcomes(session_id)
    assert await pipeline._candidate_outcomes_are_current(session_id)
    assert await pipeline._linked_outcomes_are_current(session_id)
    assert await pipeline._current_validated_outcomes(session_id) is not None
    assert provider.outcome_calls == model_calls_before
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("tamper_kind", ("missing", "duplicate"))
async def test_fix28_candidate_origin_authority_rejects_terminal_mapping_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publication(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("content-safe origin-authority interruption")

    with pytest.raises(TimeoutError, match="origin-authority"):
        await pipeline.run(job, interrupt_before_publication)
    model_calls_before = provider.outcome_calls
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    linked_sets = pipeline._read_outcome_sets(meeting_dir / "linked_outcomes.json")
    manifest = OutcomePartitionManifest.model_validate_json(
        (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
    )
    terminal_results = read_partition_results(
        meeting_dir / "outcome_partition_results.json",
        manifest,
        pipeline_revision=_OUTCOME_PIPELINE_REVISION,
    )
    source_key, source_result = next(
        (key, result) for key, result in terminal_results.items() if result.candidates
    )
    candidate = source_result.candidates[0]
    altered = dict(terminal_results)
    if tamper_kind == "missing":
        altered[source_key] = source_result.model_copy(update={"candidates": ()})
    else:
        target_key, target_result = next(
            (key, result)
            for key, result in terminal_results.items()
            if key != source_key and not result.candidates
        )
        altered[target_key] = target_result.model_copy(update={"candidates": (candidate,)})
    monkeypatch.setattr(pipeline_module, "read_partition_results", lambda *_, **__: altered)

    with pytest.raises(
        OutcomePartitionError,
        match="candidate_origin_authority_invalid",
    ):
        pipeline._candidate_origin_authority(session_id, linked_sets)
    assert provider.outcome_calls == model_calls_before
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_point", "expected_first_stage", "expected_total_model_calls"),
    (
        ("candidate", "EXTRACT_TOPIC_OUTCOMES", 1),
        ("linked", "LINK_EVIDENCE", 1),
        ("speech_revision", "LINK_EVIDENCE", 1),
    ),
)
async def test_outcome_recovery_ladder_preserves_approved_model_call_boundary(
    tmp_path: Path,
    failure_point: str,
    expected_first_stage: str,
    expected_total_model_calls: int,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        current_revision = payload.get("outcome_pipeline_revision")
        if isinstance(current_revision, str):
            revision = current_revision
        current_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(current_speech_revision, str):
            speech_revision = current_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, interrupt_before_publish)
    assert provider.outcome_calls == 1
    assert revision is not None
    assert speech_revision is not None
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    if failure_point == "candidate":
        candidate_path = meeting_dir / "topic_outcome_candidates.json"
        candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
    elif failure_point == "linked":
        linked_path = meeting_dir / "linked_outcomes.json"
        linked_path.write_bytes(linked_path.read_bytes() + b" ")
    else:
        speech_revision = hashlib.sha256(b"lingxu-speech-act-policy-task10r-v1").hexdigest()
        linked_path = meeting_dir / "linked_outcomes.json"
        linked_payload = json.loads(linked_path.read_text(encoding="utf-8"))
        linked_provenance = linked_payload["provenance"]
        linked_provenance["speech_act_policy_revision"] = speech_revision
        linked_provenance["linked_artifact_hash"] = _test_payload_hash(
            outcome_hash_input(linked_payload)
        )
        linked_path.write_text(
            json.dumps(linked_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validated_path = meeting_dir / "validated_outcomes.json"
        validated_payload = json.loads(validated_path.read_text(encoding="utf-8"))
        validated_payload["provenance"]["speech_act_policy_revision"] = speech_revision
        validated_payload["provenance"]["linked_artifact_sha256"] = hashlib.sha256(
            linked_path.read_bytes()
        ).hexdigest()
        validated_payload["provenance"]["validated_artifact_hash"] = _test_payload_hash(
            validated_outcome_hash_input(validated_payload)
        )
        validated_path.write_text(
            json.dumps(validated_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence_path = meeting_dir / "evidence.json"
        evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence_payload["provenance"]["speech_act_policy_revision"] = speech_revision
        evidence_payload["provenance"]["validated_outcomes_sha256"] = hashlib.sha256(
            validated_path.read_bytes()
        ).hexdigest()
        evidence_payload["provenance"]["evidence_payload_hash"] = _test_payload_hash(
            _evidence_hash_input(evidence_payload)
        )
        evidence_path.write_text(
            json.dumps(evidence_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        minutes_path = meeting_dir / "minutes.json"
        minutes_payload = json.loads(minutes_path.read_text(encoding="utf-8"))
        minutes_payload["provenance"]["speech_act_policy_revision"] = speech_revision
        minutes_payload["provenance"]["validated_outcomes_sha256"] = hashlib.sha256(
            validated_path.read_bytes()
        ).hexdigest()
        minutes_payload["provenance"]["minutes_payload_hash"] = _test_payload_hash(
            seed_pipeline._minutes_hash_input(minutes_payload)
        )
        minutes_path.write_text(
            json.dumps(minutes_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for artifact_type, path in (
            ("EVIDENCE_JSON", evidence_path),
            ("MINUTES_JSON", minutes_path),
            ("MINUTES_MARKDOWN", meeting_dir / "minutes.md"),
        ):
            await artifacts.add_file(
                session_id=session_id,
                artifact_type=artifact_type,
                path=path,
                metadata={
                    "pipeline_version": 3,
                    "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                    "speech_act_policy_revision": speech_revision,
                },
            )
    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=revision,
        speech_revision=speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=FakePublisher(expected_content="确定性证据验证"),
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == expected_first_stage
    assert provider.outcome_calls == expected_total_model_calls
    validated = json.loads((meeting_dir / "validated_outcomes.json").read_text(encoding="utf-8"))
    assert validated["provenance"]["speech_act_policy_revision"] == SPEECH_ACT_POLICY_REVISION
    if failure_point == "speech_revision":
        for name in ("evidence.json", "minutes.json"):
            rebuilt = json.loads((meeting_dir / name).read_text(encoding="utf-8"))
            assert rebuilt["provenance"]["speech_act_policy_revision"] == (
                SPEECH_ACT_POLICY_REVISION
            )
        for artifact_type, name in (
            ("EVIDENCE_JSON", "evidence.json"),
            ("MINUTES_JSON", "minutes.json"),
            ("MINUTES_MARKDOWN", "minutes.md"),
        ):
            record = await artifacts.at_path(
                session_id=session_id,
                artifact_type=artifact_type,
                path=meeting_dir / name,
            )
            assert record is not None
            assert record.metadata["speech_act_policy_revision"] == SPEECH_ACT_POLICY_REVISION
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_outcome_revision_restarts_at_extraction_and_invalidates_checkpoint(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, interrupt_before_publish)
    assert provider.outcome_calls == 1
    stale_revision = hashlib.sha256(b"outcome-evidence-render-task9-fix5").hexdigest()
    candidate_path = (
        tmp_path / "sessions" / session_id / "meeting" / "topic_outcome_candidates.json"
    )
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_payload["provenance"]["outcome_pipeline_revision"] = stale_revision
    candidate_payload["provenance"]["outcome_artifact_hash"] = _test_payload_hash(
        outcome_hash_input(candidate_payload)
    )
    candidate_path.write_text(
        json.dumps(candidate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=stale_revision,
        speech_revision=SPEECH_ACT_POLICY_REVISION,
    )
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=FakePublisher(expected_content="确定性证据验证"),
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "EXTRACT_TOPIC_OUTCOMES"
    assert provider.outcome_calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    ("file_bytes", "parse", "self_hash", "provenance", "revision"),
)
async def test_current_revision_evidence_corruption_regenerates_from_validated_outcomes(
    tmp_path: Path,
    corruption: str,
) -> None:
    marker = f"EVIDENCE_CORRUPTION_{corruption}"
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    seen_revision: str | None = None
    seen_speech_revision: str | None = None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal seen_revision, seen_speech_revision
        revision = payload.get("outcome_pipeline_revision")
        if isinstance(revision, str):
            seen_revision = revision
        speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(speech_revision, str):
            seen_speech_revision = speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, seed_checkpoint)
    assert seen_revision is not None
    assert seen_speech_revision is not None
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    evidence_path = meeting_dir / "evidence.json"
    original_bytes = evidence_path.read_bytes()
    evidence_payload = json.loads(original_bytes)
    assert isinstance(evidence_payload, dict)
    assert evidence_payload["schema_version"] == 1
    if corruption == "file_bytes":
        evidence_path.write_bytes(original_bytes + b" ")
    elif corruption == "parse":
        evidence_path.write_text(marker, encoding="utf-8")
    else:
        provenance = evidence_payload["provenance"]
        assert isinstance(provenance, dict)
        if corruption == "self_hash":
            provenance["evidence_payload_hash"] = "0" * 64
        elif corruption == "provenance":
            provenance["validated_outcomes_sha256"] = "0" * 64
            provenance["evidence_payload_hash"] = _test_payload_hash(
                _evidence_hash_input(evidence_payload)
            )
        else:
            provenance["outcome_pipeline_revision"] = "stale-revision"
            provenance["evidence_payload_hash"] = _test_payload_hash(
                _evidence_hash_input(evidence_payload)
            )
        evidence_path.write_text(
            json.dumps(evidence_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=seen_revision,
        speech_revision=seen_speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    recovered_publisher = FakePublisher(expected_content="确定性证据验证")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "GENERATE_EVIDENCE"
    assert evidence_path.read_bytes() == original_bytes
    rebuilt = json.loads(original_bytes)
    assert isinstance(rebuilt, dict)
    items = rebuilt["items"]
    assert isinstance(items, list)
    outcome_items = [item for item in items if item.get("evidence_kind") == "validated_outcome"]
    validated = json.loads((meeting_dir / "validated_outcomes.json").read_text(encoding="utf-8"))
    expected_outcomes = {
        (topic["topic_id"], candidate["id"])
        for topic in validated["topics"]
        for candidate in topic["candidates"]
    }
    assert {(item["topic_id"], item["outcome_id"]) for item in outcome_items} == expected_outcomes
    record = await artifacts.first_of_type(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
    )
    assert record is not None
    assert record.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert record.sha256 == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert marker not in evidence_path.read_text(encoding="utf-8")
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("sha256", "pipeline_version", "size_bytes"))
async def test_current_revision_stale_evidence_artifact_record_regenerates_before_publication(
    tmp_path: Path,
    corruption: str,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    seen_revision: str | None = None
    seen_speech_revision: str | None = None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal seen_revision, seen_speech_revision
        revision = payload.get("outcome_pipeline_revision")
        if isinstance(revision, str):
            seen_revision = revision
        speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(speech_revision, str):
            seen_speech_revision = speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, seed_checkpoint)
    assert seen_revision is not None
    assert seen_speech_revision is not None
    evidence_path = tmp_path / "sessions" / session_id / "meeting" / "evidence.json"
    original_bytes = evidence_path.read_bytes()
    record = await artifacts.first_of_type(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
    )
    assert record is not None
    stale_sha256 = record.sha256
    stale_metadata = dict(record.metadata)
    if corruption == "sha256":
        stale_sha256 = "0" * 64
    elif corruption == "pipeline_version":
        stale_metadata["pipeline_version"] = 2
    else:
        stale_metadata["size_bytes"] = len(original_bytes) + 1
    await artifacts.add(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
        path=evidence_path,
        sha256=stale_sha256,
        metadata=stale_metadata,
    )
    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=seen_revision,
        speech_revision=seen_speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    recovered_publisher = FakePublisher(expected_content="确定性证据验证")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "VALIDATE_MINUTES_CONSISTENCY"
    assert evidence_path.read_bytes() == original_bytes
    rebuilt_record = await artifacts.first_of_type(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
    )
    assert rebuilt_record is not None
    assert rebuilt_record.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert rebuilt_record.sha256 == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    assert rebuilt_record.metadata["pipeline_version"] == 4
    assert rebuilt_record.metadata["size_bytes"] == len(original_bytes)
    assert recovered_publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "artifact_type"),
    (
        ("consent_timeline.json", "CONSENT_TIMELINE_JSON"),
        ("privacy_exclusions.json", "PRIVACY_EXCLUSIONS_JSON"),
        ("canonical_transcript.json", "CANONICAL_TRANSCRIPT_JSON"),
        ("topic_segments.json", "TOPIC_SEGMENTS_JSON"),
        ("transcript.json", "TRANSCRIPT_JSON"),
        ("transcript.txt", "TRANSCRIPT_TEXT"),
        ("events.json", "MEETING_EVENTS_JSON"),
        ("languages.json", "LANGUAGE_DISTRIBUTION_JSON"),
        ("evidence.json", "EVIDENCE_JSON"),
        ("minutes.md", "MINUTES_MARKDOWN"),
        ("minutes.json", "MINUTES_JSON"),
    ),
)
async def test_all_registered_file_artifacts_use_exact_metadata_and_restart_render(
    tmp_path: Path,
    name: str,
    artifact_type: str,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        selected_revision = payload.get("outcome_pipeline_revision")
        if isinstance(selected_revision, str):
            revision = selected_revision
        selected_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(selected_speech_revision, str):
            speech_revision = selected_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, interrupt_before_publish)
    assert revision is not None
    assert speech_revision is not None
    path = tmp_path / "sessions" / session_id / "meeting" / name
    record = await artifacts.at_path(
        session_id=session_id,
        artifact_type=artifact_type,
        path=path,
    )
    assert record is not None
    expected_keys = {"pipeline_version", "outcome_pipeline_revision", "size_bytes"}
    if artifact_type in {"EVIDENCE_JSON", "MINUTES_MARKDOWN", "MINUTES_JSON"}:
        expected_keys.add("speech_act_policy_revision")
    if artifact_type in {"MINUTES_MARKDOWN", "MINUTES_JSON"}:
        expected_keys.update({"minutes_renderer_revision", "renderer_input_sha256"})
    assert set(record.metadata) == expected_keys
    assert not record.encrypted
    await artifacts.add(
        session_id=session_id,
        artifact_type=artifact_type,
        path=path,
        sha256=record.sha256,
        encrypted=True,
        metadata={**record.metadata, "unexpected": True},
    )
    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=revision,
        speech_revision=speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=FakePublisher(expected_content="确定性证据验证"),
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "VALIDATE_MINUTES_CONSISTENCY"
    assert provider.outcome_calls == 1
    repaired = await artifacts.at_path(
        session_id=session_id,
        artifact_type=artifact_type,
        path=path,
    )
    assert repaired is not None
    assert not repaired.encrypted
    assert set(repaired.metadata) == expected_keys
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pretask11_unversioned_minutes_restart_at_generate_minutes_without_model_call(
    tmp_path: Path,
) -> None:
    marker = "PRETASK11_UNVERSIONED_PUBLIC_DOCUMENT"
    provider = FakeSummaryProvider()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        selected_revision = payload.get("outcome_pipeline_revision")
        if isinstance(selected_revision, str):
            revision = selected_revision
        selected_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(selected_speech_revision, str):
            speech_revision = selected_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, interrupt_before_publish)
    assert revision is not None
    assert speech_revision is not None
    assert provider.outcome_calls == 1

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    minutes_path = meeting_dir / "minutes.json"
    markdown_path = meeting_dir / "minutes.md"
    legacy_payload = json.loads(minutes_path.read_text(encoding="utf-8"))
    legacy_payload.pop("meeting_title")
    legacy_payload.pop("full_summary")
    for agenda_payload in legacy_payload["agendas"]:
        agenda_payload.pop("summary")
    legacy_payload["provenance"].pop("minutes_renderer_revision")
    legacy_payload["provenance"].pop("renderer_input_sha256")
    legacy_payload["provenance"].pop("minutes_payload_hash")
    legacy_payload["provenance"].pop("minutes_markdown_sha256")
    minutes_path.write_text(
        json.dumps(legacy_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(marker, encoding="utf-8")
    legacy_metadata = {
        "pipeline_version": 3,
        "outcome_pipeline_revision": revision,
        "speech_act_policy_revision": speech_revision,
    }
    for artifact_type, path in (
        ("MINUTES_JSON", minutes_path),
        ("MINUTES_MARKDOWN", markdown_path),
    ):
        await artifacts.add_file(
            session_id=session_id,
            artifact_type=artifact_type,
            path=path,
            metadata={**legacy_metadata, "size_bytes": path.stat().st_size},
        )

    retry = await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=revision,
        speech_revision=speech_revision,
    )
    recovered_publisher = FakePublisher(expected_content="确定性证据验证")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "GENERATE_MINUTES"
    assert provider.outcome_calls == 1
    rebuilt = MinutesDocumentV2.model_validate_json(minutes_path.read_text(encoding="utf-8"))
    assert rebuilt.provenance.minutes_renderer_revision == MINUTES_RENDERER_REVISION
    assert len(rebuilt.provenance.renderer_input_sha256) == 64
    assert marker not in markdown_path.read_text(encoding="utf-8")
    assert recovered_publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_self_consistent_substituted_minutes_pair_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)

    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    minutes_path = meeting_dir / "minutes.json"
    markdown_path = meeting_dir / "minutes.md"
    payload = json.loads(minutes_path.read_text(encoding="utf-8"))
    original = MinutesDocumentV2.model_validate_json(json.dumps(payload))
    payload["generated_at"] = "2030-01-01T00:00:00Z"
    payload["provenance"]["renderer_input_sha256"] = "0" * 64
    payload["provenance"]["minutes_markdown_sha256"] = None
    payload["provenance"]["minutes_payload_hash"] = None
    substituted = MinutesDocumentV2.model_validate_json(json.dumps(payload))
    _, canonical, labels, _ = await pipeline._build_minutes_document(
        session_id,
        generated_at=original.generated_at,
    )
    payload["provenance"]["renderer_input_sha256"] = renderer_input_sha256(
        substituted,
        canonical,
        labels=labels,
    )
    substituted = MinutesDocumentV2.model_validate_json(json.dumps(payload))
    substituted_markdown = render_minutes(substituted, canonical, labels=labels)
    payload = substituted.model_dump(mode="json")
    payload["provenance"]["minutes_markdown_sha256"] = hashlib.sha256(
        substituted_markdown.encode("utf-8")
    ).hexdigest()
    payload["provenance"]["minutes_payload_hash"] = _test_payload_hash(
        pipeline._minutes_hash_input(payload)
    )
    substituted = MinutesDocumentV2.model_validate_json(json.dumps(payload))
    minutes_path.write_text(
        json.dumps(
            substituted.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(substituted_markdown, encoding="utf-8")
    for artifact_type, path in (
        ("MINUTES_JSON", minutes_path),
        ("MINUTES_MARKDOWN", markdown_path),
    ):
        await artifacts.add_file(
            session_id=session_id,
            artifact_type=artifact_type,
            path=path,
            metadata=pipeline._registered_artifact_metadata(
                artifact_type,
                path.stat().st_size,
                document=substituted,
            ),
        )

    with pytest.raises(PublicationReconciliationRequired, match="publication_record_invalid"):
        await pipeline._stage_publish_discord(session_id)
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_current_revision_wrong_path_evidence_record_does_not_force_retry_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_publisher = FakePublisher(fail=True)
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        failing_publisher,
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    wrong_path = meeting_dir / "stale-evidence.json"
    wrong_bytes = b'{"stale":true}\n'
    wrong_path.write_bytes(wrong_bytes)
    await artifacts.add(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
        path=wrong_path,
        sha256=hashlib.sha256(wrong_bytes).hexdigest(),
        metadata={
            "pipeline_version": 3,
            "outcome_pipeline_revision": "pending-current-revision",
            "size_bytes": len(wrong_bytes),
        },
    )

    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        current_revision = payload.get("outcome_pipeline_revision")
        if isinstance(current_revision, str):
            revision = current_revision
        current_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(current_speech_revision, str):
            speech_revision = current_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, seed_checkpoint)
    assert revision is not None
    assert speech_revision is not None
    await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=revision,
        speech_revision=speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    await artifacts.add(
        session_id=session_id,
        artifact_type="EVIDENCE_JSON",
        path=wrong_path,
        sha256=hashlib.sha256(wrong_bytes).hexdigest(),
        metadata={
            "pipeline_version": 3,
            "outcome_pipeline_revision": revision,
            "speech_act_policy_revision": speech_revision,
            "size_bytes": len(wrong_bytes),
        },
    )
    canonical_path = meeting_dir / "evidence.json"
    canonical_bytes = canonical_path.read_bytes()
    canonical_sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    evidence_runs = 0
    original_generate_evidence = PostMeetingPipeline._stage_generate_evidence

    async def track_generate_evidence(
        pipeline: PostMeetingPipeline,
        tracked_session_id: str,
    ) -> dict[str, Any]:
        nonlocal evidence_runs
        evidence_runs += 1
        return await original_generate_evidence(pipeline, tracked_session_id)

    monkeypatch.setattr(
        PostMeetingPipeline,
        "_stage_generate_evidence",
        track_generate_evidence,
    )
    recovered_publisher = FakePublisher()
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="wrong-path-retry-worker",
        pipeline=recovered_pipeline,
    ).run_once()
    assert evidence_runs == 0
    assert canonical_path.read_bytes() == canonical_bytes
    evidence_records = tuple(
        record
        for record in await artifacts.list(session_id)
        if record.artifact_type == "EVIDENCE_JSON"
    )
    canonical_records = tuple(
        record for record in evidence_records if record.path == canonical_path
    )
    assert len(canonical_records) == 1
    assert canonical_records[0].sha256 == canonical_sha256
    assert canonical_records[0].metadata["outcome_pipeline_revision"] == revision
    assert canonical_records[0].metadata["size_bytes"] == len(canonical_bytes)
    assert wrong_path.read_bytes() == wrong_bytes
    assert recovered_publisher.calls == 1

    assert not await WorkerService(
        jobs=jobs,
        worker_id="wrong-path-second-retry-worker",
        pipeline=recovered_pipeline,
    ).run_once()
    assert evidence_runs == 0
    assert recovered_publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completed_published_job_is_not_reopened_by_revision_validation(
    tmp_path: Path,
) -> None:
    marker = "COMPLETED_PUBLICATION_MUST_REMAIN_UNTOUCHED"
    publisher = FakePublisher()
    database, jobs, pipeline, _, _, session_id, _ = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    worker = WorkerService(jobs=jobs, worker_id="completed-worker", pipeline=pipeline)
    assert await worker.run_once()
    minutes = tmp_path / "sessions" / session_id / "meeting" / "minutes.md"
    minutes.write_text(marker, encoding="utf-8")

    assert not await WorkerService(
        jobs=jobs,
        worker_id="completed-retry-worker",
        pipeline=pipeline,
    ).run_once()
    assert minutes.read_text(encoding="utf-8") == marker
    assert publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema_two_rerender_updates_artifact_integrity_record(
    tmp_path: Path,
) -> None:
    """A privacy rerender replaces stale Artifact integrity and size metadata."""

    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    meeting_dir.mkdir(exist_ok=True)
    transcript_path = meeting_dir / "transcript.json"
    stale_payload = b'[{"raw_text":"private"}]\n'
    transcript_path.write_bytes(stale_payload)
    await artifacts.add_file(
        session_id=session_id,
        artifact_type="TRANSCRIPT_JSON",
        path=transcript_path,
        metadata={"pipeline_version": 2, "size_bytes": len(stale_payload)},
    )
    retry = await _seed_schema_two_unfinished_job(jobs, job_id)

    async def checkpoint(stage: str, payload: dict[str, object]) -> None:
        del stage, payload

    await pipeline.run(retry, checkpoint)

    rewritten_payload = transcript_path.read_bytes()
    assert b"private" not in rewritten_payload
    record = next(
        item for item in await artifacts.list(session_id) if item.artifact_type == "TRANSCRIPT_JSON"
    )
    assert record.path == transcript_path
    assert record.sha256 == hashlib.sha256(rewritten_payload).hexdigest()
    assert record.metadata["pipeline_version"] == 4
    assert record.metadata["size_bytes"] == len(rewritten_payload)
    assert isinstance(record.metadata["outcome_pipeline_revision"], str)
    assert len(record.metadata["outcome_pipeline_revision"]) == 64
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_uses_grounded_deepseek_outcomes_without_nonconsented_text(
    tmp_path: Path,
) -> None:
    summary_provider = FakeSummaryProvider()
    publisher = FakePublisher(expected_content="确定性证据验证")
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        summary_provider,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="summary-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    payload = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["summary_method"] == "deepseek"
    assert payload["summary_model"] == APPROVED_TOPIC_MODEL
    assert (
        payload["validated_outcomes"][0]["candidates"][0]["verification_status"]
        == "evidence_linked"
    )
    assert payload["risks"][0]["text"] == "已提炼风险"
    assert payload["votes"][0]["counts"] == {"abstain": 0, "no": 0, "yes": 2}
    assert "private" not in summary_provider.received_text
    markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert "DeepSeek (结果已通过确定性验证)" in markdown
    assert "entry:" not in markdown
    assert "已提炼风险" in markdown
    assert "private" not in markdown

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_falls_back_and_publishes_when_local_summary_fails(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher(expected_content="安全回退")
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
        FailingSummaryProvider(),
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="fallback-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    payload = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["summary_method"] == "deterministic_fallback"
    assert payload["fallback_reasons"]
    outcome_payload = json.loads(
        (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
    )
    provider_failures = [
        failure for topic in outcome_payload["topics"] for failure in topic["provider_failures"]
    ]
    assert provider_failures
    assert all(
        set(failure)
        == {
            "category",
            "request_kind",
            "attempt",
            "count",
            "size_bucket",
            "capacity_subtype",
            "format_subtype",
        }
        for failure in provider_failures
    )
    assert {failure["category"] for failure in provider_failures} == {"unknown"}
    assert all("simulated local outage" not in reason for reason in payload["fallback_reasons"])
    assert "private" not in (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    assert publisher.calls == 1

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_persists_closed_provider_failure_without_crashing(
    tmp_path: Path,
) -> None:
    provider = ClosedFailingOutcomeProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="安全回退"),
        provider,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="closed-provider-failure-worker",
        pipeline=pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    payload = json.loads(
        (
            tmp_path / "sessions" / session_id / "meeting" / "topic_outcome_candidates.json"
        ).read_text(encoding="utf-8")
    )
    failures = [failure for topic in payload["topics"] for failure in topic["provider_failures"]]
    assert failures
    assert {failure["category"] for failure in failures} == {"json"}
    assert {failure["count"] for failure in failures} == {1}
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_persists_capacity_subtype_in_artifact_and_minutes_fallback(
    tmp_path: Path,
) -> None:
    provider = ClosedCapacityOutcomeProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="安全回退"),
        provider,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="capacity-provider-failure-worker",
        pipeline=pipeline,
    ).run_once()

    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    outcome_payload = json.loads(
        (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
    )
    failures = [
        failure for topic in outcome_payload["topics"] for failure in topic["provider_failures"]
    ]
    assert {failure["capacity_subtype"] for failure in failures} == {"text_chars"}
    minutes_payload = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert (
        "output_capacity:action:initial:1:lte_1k:text_chars:none"
        in minutes_payload["fallback_reasons"]
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_never_renders_an_unsupported_model_outcome(
    tmp_path: Path,
) -> None:
    provider = UnsupportedOutcomeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    checkpoints: list[tuple[str, dict[str, Any]]] = []
    job = await jobs.get(job_id)
    assert job is not None

    async def checkpoint(stage: str, payload: dict[str, Any]) -> None:
        checkpoints.append((stage, payload))

    await pipeline.run(job, checkpoint)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    raw_candidates = (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
    validated = (meeting_dir / "validated_outcomes.json").read_text(encoding="utf-8")
    minutes = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
    evidence = (meeting_dir / "evidence.json").read_text(encoding="utf-8")
    validation_details = next(
        payload["last_stage_details"]
        for stage, payload in checkpoints
        if stage == "VALIDATE_OUTCOMES" and "last_stage_details" in payload
    )

    assert "UNSUPPORTED_MODEL_FACT" in raw_candidates
    assert "UNSUPPORTED_MODEL_FACT" not in validated
    assert "UNSUPPORTED_MODEL_FACT" not in minutes
    assert "UNSUPPORTED_MODEL_FACT" not in evidence
    assert validation_details == {
        "valid_count": 0,
        "rejected_count": 1,
        "coalesced_count": 0,
        "ambiguous_outcome_binding_count": 0,
        "error_types": {
            "decision_not_accepted": 1,
            "unknown_source_entry": 1,
        },
    }
    assert "UNSUPPORTED_MODEL_FACT" not in json.dumps(checkpoints, ensure_ascii=False)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worker_routes_only_deterministically_topic_bound_event_facts(
    tmp_path: Path,
) -> None:
    marker = "EVENT_INJECTION_MARKER"
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(expected_content="确定性证据验证"),
        provider,
    )
    agenda = await AgendaRepository(database.sessions).list(session_id)
    assert agenda
    boundary = await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:event-scope-transition",
        user_id=100,
        raw_text="接下来讨论部署风险。",
        normalised_text="接下来讨论部署风险。",
        started_ms=3_000,
        ended_ms=3_500,
    )
    event_store = MeetingEventStore(database.sessions)
    valid = await event_store.append(
        session_id=session_id,
        event_type="VOTE_CLOSED",
        payload={
            "source_entry_ids": [boundary.id],
            "vote_content": "release",
            "result": "accepted",
            "malicious_instruction": marker,
        },
    )
    generic = await event_store.append(
        session_id=session_id,
        event_type="DECISION_RECORDED",
        payload={
            "source_entry_ids": [boundary.id],
            "subject": "release",
            "result": "accepted",
            "malicious_instruction": marker,
        },
    )
    ambiguous = await event_store.append(
        session_id=session_id,
        event_type="VOTE_CLOSED",
        payload={
            "agenda_item_id": agenda[0].id,
            "vote_content": "release",
            "result": "accepted",
        },
    )
    unrelated = await event_store.append(
        session_id=session_id,
        event_type="VOTE_CLOSED",
        payload={
            "topic_id": "unrelated-topic",
            "vote_content": "release",
            "result": "accepted",
        },
    )
    malformed = await event_store.append(
        session_id=session_id,
        event_type="VOTE_OPENED",
        payload={
            "agenda_item_id": [],
            "vote_content": "release",
            "result": "accepted",
        },
    )

    assert await WorkerService(
        jobs=jobs, worker_id="event-scope-worker", pipeline=pipeline
    ).run_once()

    routed = [
        event
        for topic_id, events in provider.outcome_event_batches
        for event in events
        if event.sequence_number
        in {
            valid.sequence_number,
            ambiguous.sequence_number,
            unrelated.sequence_number,
            malformed.sequence_number,
            generic.sequence_number,
        }
    ]
    assert [event.sequence_number for event in routed] == [valid.sequence_number]
    assert routed[0].source_entry_ids == (boundary.id,)
    assert not hasattr(routed[0], "payload")
    assert not hasattr(routed[0], "event_type")
    assert all(
        event.topic_id == topic_id
        for topic_id, events in provider.outcome_event_batches
        for event in events
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    completed = await jobs.get(job_id)
    assert completed is not None
    rendered_and_checkpoint = "\n".join(
        (
            (meeting_dir / "events.json").read_text(encoding="utf-8"),
            (meeting_dir / "evidence.json").read_text(encoding="utf-8"),
            (meeting_dir / "minutes.json").read_text(encoding="utf-8"),
            (meeting_dir / "minutes.md").read_text(encoding="utf-8"),
            json.dumps(completed.checkpoint, ensure_ascii=False),
            json.dumps(
                [
                    event.model_dump(mode="json")
                    for _, events in provider.outcome_event_batches
                    for event in events
                ],
                ensure_ascii=False,
            ),
        )
    )
    assert marker not in rendered_and_checkpoint
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_privacy_exclusion_journal_reconciles_crash_after_row_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, jobs, pipeline, _, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    excluded = await transcripts.upsert(
        session_id=session_id,
        source_key="legacy:privacy-journal-regression",
        user_id=200,
        raw_text="excluded synthetic utterance",
        normalised_text="excluded synthetic utterance",
        started_ms=1_300,
        ended_ms=2_400,
    )
    original_delete = pipeline._transcripts.delete_ids
    crashed = False

    async def delete_then_crash(*, session_id: str, entry_ids: tuple[str, ...]) -> int:
        nonlocal crashed
        removed = await original_delete(session_id=session_id, entry_ids=entry_ids)
        if not crashed:
            crashed = True
            raise RuntimeError("simulated content-safe privacy journal interruption")
        return removed

    monkeypatch.setattr(pipeline._transcripts, "delete_ids", delete_then_crash)
    assert await WorkerService(
        jobs=jobs,
        worker_id="privacy-journal-crash-worker",
        pipeline=pipeline,
        max_attempts=3,
    ).run_once()
    interrupted = await jobs.get(job_id)
    assert interrupted is not None and interrupted.status is JobStatus.PENDING
    assert interrupted.current_stage == "APPLY_PRIVACY_CUTOFFS"
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    serialized_journal = (meeting_dir / "privacy_exclusions.json").read_text(encoding="utf-8")
    journal = json.loads(serialized_journal)
    assert journal["state"] == "prepared"
    assert journal["candidate_counts"]["transcript_entries"] == 1
    assert excluded.id not in serialized_journal
    assert session_id not in serialized_journal

    monkeypatch.setattr(pipeline._transcripts, "delete_ids", original_delete)
    assert await WorkerService(
        jobs=jobs,
        worker_id="privacy-journal-recovery-worker",
        pipeline=pipeline,
        max_attempts=3,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    minutes = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert minutes["privacy_exclusions"] == {
        "measurement_available": True,
        "segment_files": 1,
        "segment_rows": 1,
        "transcript_entries": 1,
    }
    assert (
        json.loads((meeting_dir / "privacy_exclusions.json").read_text(encoding="utf-8"))["state"]
        == "complete"
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_privacy_exclusion_counts_only_files_that_existed_before_unlink(
    tmp_path: Path,
) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    segments = await AudioSegmentRepository(database.sessions).list_session(session_id)
    excluded_segment = next(segment for segment in segments if segment.user_id == 200)
    SessionFiles(tmp_path / "sessions").resolve(
        session_id,
        excluded_segment.encrypted_path,
    ).unlink()

    assert await WorkerService(
        jobs=jobs,
        worker_id="privacy-file-count-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    minutes = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert minutes["privacy_exclusions"]["segment_rows"] == 1
    assert minutes["privacy_exclusions"]["segment_files"] == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", (False, True))
async def test_new_job_missing_or_corrupt_privacy_journal_is_not_legacy_unavailable(
    tmp_path: Path,
    corrupt: bool,
) -> None:
    database, _, pipeline, _, _, session_id, _ = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    await pipeline._stage_build_consent_timeline(session_id)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    timeline = json.loads((meeting_dir / "consent_timeline.json").read_text(encoding="utf-8"))
    assert timeline["exclusion_journal_version"] == 1
    journal_path = meeting_dir / "privacy_exclusions.json"
    if corrupt:
        journal_path.write_text('{"schema_version":1,"state":"corrupt"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="privacy exclusion journal"):
        pipeline._privacy_exclusions(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("journal_corrupt", (False, True))
@pytest.mark.parametrize("timeline_marker", (None, "invalid"))
async def test_new_job_cannot_downgrade_when_timeline_marker_is_missing_or_invalid(
    tmp_path: Path,
    journal_corrupt: bool,
    timeline_marker: str | None,
) -> None:
    database, _, pipeline, _, _, session_id, _ = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    await pipeline._stage_build_consent_timeline(session_id)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    timeline_path = meeting_dir / "consent_timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if timeline_marker is None:
        timeline.pop("exclusion_journal_version", None)
    else:
        timeline["exclusion_journal_version"] = timeline_marker
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    if journal_corrupt:
        (meeting_dir / "privacy_exclusions.json").write_text(
            '{"schema_version":2,"state":"corrupt"}\n',
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="privacy exclusion journal"):
        pipeline._privacy_exclusions(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_new_job_cannot_claim_legacy_unavailable_without_checkpoint_authority(
    tmp_path: Path,
) -> None:
    database, _, pipeline, _, _, session_id, _ = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    await pipeline._stage_build_consent_timeline(session_id)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    timeline_path = meeting_dir / "consent_timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline.pop("exclusion_journal_version", None)
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
    journal = pipeline._privacy_journal_payload(
        state="legacy_unavailable",
        candidate_hashes={
            "transcript_entries": [],
            "segment_rows": [],
            "segment_files": [],
        },
        measurements=None,
    )
    (meeting_dir / "privacy_exclusions.json").write_text(
        json.dumps(journal),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy"):
        pipeline._privacy_exclusions(session_id)
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_final_withdrawal_preserves_temporally_eligible_participant_identity(
    tmp_path: Path,
) -> None:
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(),
    )
    await MeetingEventStore(database.sessions).append(
        session_id=session_id,
        event_type="CONSENT_WITHDRAWN",
        actor_user_id=100,
        source="CONSENT",
        payload={
            "subject_user_id": 100,
            "consent_kinds": [ConsentKind.MINUTES.value],
            "effective_at_ms": 2_000,
        },
    )
    await ParticipantRepository(database.sessions).set_consent(
        session_id=session_id,
        user_id=100,
        kind=ConsentKind.MINUTES,
        granted=False,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="temporal-identity-worker",
        pipeline=pipeline,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    minutes = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    canonical = json.loads((meeting_dir / "canonical_transcript.json").read_text(encoding="utf-8"))
    assert "主持人甲" in minutes["participants"]["host"]
    assert any(
        entry["speaker_user_id"] == 100 and entry["speaker_name"] == "主持人甲"
        for entry in canonical["entries"]
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("retry_provider_mode", ("none", "unapproved"))
async def test_minutes_metadata_comes_from_validated_artifact_when_provider_is_disabled_on_retry(
    tmp_path: Path,
    retry_provider_mode: str,
) -> None:
    provider = FakeSummaryProvider()
    database, jobs, pipeline, _, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        FakePublisher(fail=True),
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None
    revision: str | None = None
    speech_revision: str | None = None

    async def seed_checkpoint(stage: str, payload: dict[str, object]) -> None:
        nonlocal revision, speech_revision
        current_revision = payload.get("outcome_pipeline_revision")
        if isinstance(current_revision, str):
            revision = current_revision
        current_speech_revision = payload.get("speech_act_policy_revision")
        if isinstance(current_speech_revision, str):
            speech_revision = current_speech_revision
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, seed_checkpoint)
    assert revision is not None
    assert speech_revision is not None
    await _seed_prefingerprint_schema_three_job(
        jobs,
        job_id,
        revision=revision,
        speech_revision=speech_revision,
        minutes_revision=MINUTES_RENDERER_REVISION,
        minutes_input_digest=_minutes_renderer_input(tmp_path, session_id),
    )
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    seeded = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert seeded["summary_method"] == "deepseek"
    (meeting_dir / "minutes.json").unlink()
    (meeting_dir / "minutes.md").unlink()

    retry_provider = (
        None
        if retry_provider_mode == "none"
        else FakeSummaryProvider(
            TopicBoundaryProviderPolicy(
                provider=APPROVED_TOPIC_PROVIDER,
                base_url="https://example.invalid/v1",
                model=APPROVED_TOPIC_MODEL,
                expected_model_digest=APPROVED_TOPIC_MODEL_DIGEST,
                allow_remote=True,
            )
        )
    )
    retry_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=ArtifactRepository(database.sessions),
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=FakePublisher(expected_content="确定性证据验证"),
        retention_hours=24,
        post_meeting_inference_provider=retry_provider,
    )
    assert await WorkerService(
        jobs=jobs,
        worker_id="metadata-retry-worker",
        pipeline=retry_pipeline,
        max_attempts=3,
    ).run_once()
    completed = await jobs.get(job_id)
    assert completed is not None and completed.status is JobStatus.COMPLETED
    recovered = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
    assert recovered["summary_method"] == "deepseek"
    assert recovered["summary_provider"] == APPROVED_TOPIC_PROVIDER
    assert recovered["summary_model"] == APPROVED_TOPIC_MODEL
    assert recovered["summary_model_digest"] == APPROVED_TOPIC_MODEL_DIGEST
    assert isinstance(recovered["summary_policy_identity"], str)
    assert len(recovered["summary_policy_identity"]) == 64
    assert recovered["fallback_reasons"] == []
    if retry_provider is not None:
        assert retry_provider.topic_calls == 0
        assert retry_provider.outcome_calls == 0
    await database.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "tamper",
    (
        {"status": "unknown_policy_state", "provider": None, "model": None},
        {
            "status": "provider_disabled",
            "provider": "ollama",
            "model": APPROVED_TOPIC_MODEL,
        },
    ),
)
def test_fix27_stored_outcome_policy_rejects_self_consistent_unauthorized_states(
    tamper: dict[str, object],
) -> None:
    provenance: dict[str, object] = {
        "status": "approved",
        "provider": "ollama",
        "model": APPROVED_TOPIC_MODEL,
        "approved_model_digest": APPROVED_TOPIC_MODEL_DIGEST,
        "endpoint_policy": "private_local_http_only",
        "remote_allow_authorizes_outcome_calls": False,
        "provider_interface": "single_kind",
    }
    provenance.update(tamper)
    if provenance["status"] != "approved":
        provenance["provider_interface"] = "unsupported"
    policy = {
        key: provenance[key]
        for key in (
            "status",
            "provider",
            "model",
            "approved_model_digest",
            "endpoint_policy",
            "remote_allow_authorizes_outcome_calls",
        )
    }
    provenance["policy_hash"] = _test_payload_hash(policy)

    with pytest.raises(ValueError, match="outcome policy is invalid"):
        stored_outcome_policy_provenance(provenance)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_consistency_is_terminal_before_intent_or_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )

    async def reject_consistency(_session_id: str) -> dict[str, object]:
        raise MinutesConsistencyError(
            "participant_transcript_count",
            expected=1,
            observed=2,
        )

    monkeypatch.setattr(pipeline, "_stage_validate_minutes_consistency", reject_consistency)
    assert await WorkerService(
        jobs=jobs,
        worker_id="consistency-rejection-worker",
        pipeline=pipeline,
    ).run_once()

    failed = await jobs.get(job_id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert failed.checkpoint["schema_version"] == 4
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_schema_three_current_job_resumes_at_consistency_without_model_call(
    tmp_path: Path,
) -> None:
    provider = FakeSummaryProvider()
    seed_publisher = FakePublisher()
    database, jobs, seed_pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        seed_publisher,
        provider,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await seed_pipeline.run(job, interrupt_before_publish)
    model_calls = provider.topic_calls + provider.outcome_calls

    for record in await artifacts.list(session_id):
        if record.artifact_type in {
            "DISCORD_PUBLICATION_INTENT",
            "DISCORD_PUBLICATION",
        }:
            continue
        await artifacts.add(
            session_id=session_id,
            artifact_type=record.artifact_type,
            path=record.path,
            sha256=record.sha256,
            encrypted=record.encrypted,
            metadata={**record.metadata, "pipeline_version": 3},
        )

    worker_id = "schema-three-consistency-worker"
    claimed = await jobs.claim(
        worker_id=worker_id,
        lease_seconds=30,
        job_types=(POST_MEETING_JOB,),
    )
    assert claimed is not None and claimed.id == job_id
    completed = list(
        POST_MEETING_STAGES[: POST_MEETING_STAGES.index("VALIDATE_MINUTES_CONSISTENCY")]
    )
    checkpoint = {
        "schema_version": 3,
        "privacy_exclusion_journal_version": 1,
        "privacy_exclusion_measurement_state": "journal_v2",
        "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
        "speech_act_policy_revision": SPEECH_ACT_POLICY_REVISION,
        "minutes_renderer_revision": MINUTES_RENDERER_REVISION,
        "renderer_input_sha256": _minutes_renderer_input(tmp_path, session_id),
        "completed_stages": completed,
        "active_stage": "VALIDATE_MINUTES_CONSISTENCY",
    }
    await jobs.checkpoint(
        job_id=job_id,
        worker_id=worker_id,
        stage="VALIDATE_MINUTES_CONSISTENCY",
        checkpoint=checkpoint,
        lease_seconds=30,
    )
    await jobs.fail(
        job_id=job_id,
        worker_id=worker_id,
        error_type="InterruptedError@VALIDATE_MINUTES_CONSISTENCY",
        retry_at=utc_now(),
    )
    retry = await jobs.get(job_id)
    assert retry is not None

    recovered_publisher = FakePublisher(expected_content="确定性证据验证")
    recovered_pipeline = PostMeetingPipeline(
        sessions=MeetingSessionRepository(database.sessions),
        events=MeetingEventStore(database.sessions),
        participants=ParticipantRepository(database.sessions),
        agenda=AgendaRepository(database.sessions),
        utterances=RealtimeUtteranceRepository(database.sessions),
        segments=AudioSegmentRepository(database.sessions),
        transcripts=TranscriptRepository(database.sessions),
        artifacts=artifacts,
        files=SessionFiles(tmp_path / "sessions"),
        transcriber=None,
        publisher=recovered_publisher,
        retention_hours=24,
        post_meeting_inference_provider=provider,
    )
    active_stages: list[str] = []
    final_checkpoint: dict[str, object] = {}

    async def recovered_checkpoint(stage: str, payload: dict[str, object]) -> None:
        if payload.get("active_stage") == stage:
            active_stages.append(stage)
        final_checkpoint.clear()
        final_checkpoint.update(payload)

    await recovered_pipeline.run(retry, recovered_checkpoint)

    assert active_stages[0] == "VALIDATE_MINUTES_CONSISTENCY"
    assert provider.topic_calls + provider.outcome_calls == model_calls
    assert final_checkpoint["schema_version"] == 4
    assert recovered_publisher.calls == 1
    current_records = await artifacts.list(session_id)
    for record in current_records:
        if record.artifact_type in {
            "DISCORD_PUBLICATION_INTENT",
            "DISCORD_PUBLICATION",
        }:
            assert record.metadata["pipeline_version"] == 4
        elif record.path.is_absolute():
            assert record.metadata["pipeline_version"] == 4
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "artifact_type"),
    (
        ("consent_timeline.json", "CONSENT_TIMELINE_JSON"),
        ("privacy_exclusions.json", "PRIVACY_EXCLUSIONS_JSON"),
        ("canonical_transcript.json", "CANONICAL_TRANSCRIPT_JSON"),
        ("topic_segments.json", "TOPIC_SEGMENTS_JSON"),
        ("transcript.json", "TRANSCRIPT_JSON"),
        ("transcript.txt", "TRANSCRIPT_TEXT"),
        ("events.json", "MEETING_EVENTS_JSON"),
        ("languages.json", "LANGUAGE_DISTRIBUTION_JSON"),
        ("evidence.json", "EVIDENCE_JSON"),
        ("minutes.md", "MINUTES_MARKDOWN"),
        ("minutes.json", "MINUTES_JSON"),
    ),
)
async def test_attacker_updated_registered_artifact_is_rejected_before_intent(
    tmp_path: Path,
    name: str,
    artifact_type: str,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    job = await jobs.get(job_id)
    assert job is not None

    async def interrupt_before_publish(stage: str, payload: dict[str, object]) -> None:
        if stage == "PUBLISH_DISCORD" and payload.get("active_stage") == stage:
            raise TimeoutError("simulated pre-publication interruption")

    with pytest.raises(TimeoutError):
        await pipeline.run(job, interrupt_before_publish)
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    path = meeting_dir / name
    record = await artifacts.at_path(
        session_id=session_id,
        artifact_type=artifact_type,
        path=path,
    )
    assert record is not None
    # Preserve valid JSON/Markdown/text syntax while changing the authoritative byte identity.
    path.write_bytes(path.read_bytes() + b"\n")
    forged_metadata = {**record.metadata, "size_bytes": path.stat().st_size}
    await artifacts.add(
        session_id=session_id,
        artifact_type=artifact_type,
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata=forged_metadata,
    )

    with pytest.raises((MinutesConsistencyError, PublicationReconciliationRequired)):
        await pipeline._stage_publish_discord(session_id)
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consented_human_without_participant_row_fails_before_publication(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, _, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    async with database.sessions() as db:
        await db.execute(
            delete(MeetingParticipantRow).where(
                MeetingParticipantRow.session_id == session_id,
                MeetingParticipantRow.user_id == 100,
            )
        )
        await db.commit()

    assert await WorkerService(
        jobs=jobs,
        worker_id="missing-participant-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_null_user_non_bot_source_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    original = pipeline._stage_finalise_realtime_utterances

    async def inject_invalid_source(selected_session_id: str) -> dict[str, Any]:
        details = await original(selected_session_id)
        await transcripts.upsert(
            session_id=selected_session_id,
            source_key="segment:synthetic-null-user",
            user_id=None,
            raw_text="synthetic invalid null source",
            normalised_text="synthetic invalid null source",
            started_ms=701,
            ended_ms=800,
        )
        return details

    monkeypatch.setattr(pipeline, "_stage_finalise_realtime_utterances", inject_invalid_source)
    assert await WorkerService(
        jobs=jobs,
        worker_id="null-source-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_human_user_bot_tts_source_fails_before_publication(tmp_path: Path) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    await transcripts.upsert(
        session_id=session_id,
        source_key="BOT_TTS:synthetic-human-mismatch",
        user_id=100,
        raw_text="synthetic invalid bot identity",
        normalised_text="synthetic invalid bot identity",
        started_ms=701,
        ended_ms=800,
    )

    assert await WorkerService(
        jobs=jobs,
        worker_id="bot-human-mismatch-worker",
        pipeline=pipeline,
    ).run_once()
    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legitimate_bot_and_human_identities_validate_with_exact_counts(
    tmp_path: Path,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, _, _, _, job_id = await _pipeline_fixture(tmp_path, publisher)

    assert await WorkerService(
        jobs=jobs,
        worker_id="legitimate-identity-worker",
        pipeline=pipeline,
    ).run_once()
    job = await jobs.get(job_id)
    assert job is not None and job.status is JobStatus.COMPLETED

    report = await pipeline._validate_minutes_consistency(
        job.session_id,
        require_registered_artifacts=True,
    )
    assert report.participant_entries == 1
    assert report.bot_entries == 1
    assert report.readable_entries == 2
    assert len(report.expected_artifact_digests) == 14
    assert publisher.calls == 1
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_entry_type", "source_text"),
    (
        ("backchannel", "嗯嗯"),
        ("noise", "<|noise|>"),
        ("privacy_control", "请停止录音"),
    ),
)
async def test_excluded_human_source_without_minutes_consent_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expected_entry_type: str,
    source_text: str,
) -> None:
    publisher = FakePublisher()
    database, jobs, pipeline, artifacts, transcripts, session_id, job_id = await _pipeline_fixture(
        tmp_path,
        publisher,
    )
    original = pipeline._stage_validate_languages
    injected_ids: list[str] = []

    async def inject_after_privacy_cutoff(selected_session_id: str) -> dict[str, Any]:
        details = await original(selected_session_id)
        injected = await transcripts.upsert(
            session_id=selected_session_id,
            source_key=f"legacy:excluded-{expected_entry_type}",
            user_id=200,
            raw_text=source_text,
            normalised_text=source_text,
            started_ms=2_500,
            ended_ms=2_700,
        )
        injected_ids.append(injected.id)
        return details

    monkeypatch.setattr(pipeline, "_stage_validate_languages", inject_after_privacy_cutoff)
    worker = WorkerService(
        jobs=jobs,
        worker_id=f"excluded-consent-{expected_entry_type}",
        pipeline=pipeline,
    )
    assert await worker.run_once()

    failed = await jobs.get(job_id)
    assert failed is not None and failed.status is JobStatus.FAILED
    assert failed.current_stage == "VALIDATE_MINUTES_CONSISTENCY"
    assert injected_ids
    meeting_dir = tmp_path / "sessions" / session_id / "meeting"
    transcript_payload = json.loads((meeting_dir / "transcript.json").read_text(encoding="utf-8"))
    assert injected_ids[0] in {entry["id"] for entry in transcript_payload}
    canonical_payload = json.loads(
        (meeting_dir / "canonical_transcript.json").read_text(encoding="utf-8")
    )
    matching = [
        entry
        for entry in canonical_payload["entries"]
        if injected_ids[0] in entry["source_entry_ids"]
    ]
    assert len(matching) == 1 and matching[0]["entry_type"] == expected_entry_type
    assert publisher.calls == 0
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION_INTENT",
    )
    assert not await artifacts.all_of_type(
        session_id=session_id,
        artifact_type="DISCORD_PUBLICATION",
    )
    assert source_text not in capsys.readouterr().out
    assert not await worker.run_once()
    await database.close()
