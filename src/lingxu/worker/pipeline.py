"""Checkpointed, deterministic post-Meeting Worker pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from lingxu.config import Settings
from lingxu.domain.enums import ConsentKind
from lingxu.domain.errors import SummaryProviderError
from lingxu.domain.models import (
    AgendaItemRecord,
    AudioSegmentRecord,
    MeetingEventRecord,
    ProcessingJobRecord,
    TranscriptEntryRecord,
)
from lingxu.meeting.prompts import agenda_language
from lingxu.privacy.consent_timeline import ConsentTimeline
from lingxu.storage.component_repositories import AgendaRepository, ParticipantRepository
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    MeetingSessionRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.session_files import SessionFiles, atomic_write_text
from lingxu.storage.worker_repositories import (
    ArtifactRepository,
    PublicationReconciliationRequired,
    TranscriptRepository,
)
from lingxu.transcript.canonical import build_canonical_transcript
from lingxu.transcript.glossary import load_glossaries, normalise_entities
from lingxu.transcript.models import CanonicalTranscript, CanonicalTranscriptEntry
from lingxu.worker.audio import SegmentTranscriber
from lingxu.worker.basic_minutes import render_basic_minutes
from lingxu.worker.checkpoint_recovery import (
    CHECKPOINT_SCHEMA_VERSION,
    recover_attempt_generation,
    recover_completed_stages,
    recover_renderer_input_sha256,
)
from lingxu.worker.consistency import (
    ArtifactDigestExpectation,
    ConsistencyReport,
    MinutesConsistencyContext,
    MinutesConsistencyError,
    validate_minutes_consistency,
)
from lingxu.worker.evidence_linker import (
    EvidenceContext,
    SpeechActReductionCache,
    link_outcome_evidence,
    validate_outcome,
)
from lingxu.worker.full_transcript_preview import (
    DEFAULT_FINAL_MINUTES_MAX_OUTPUT_TOKENS,
    EVIDENCE_NORMALIZATION_REVISION,
    MAX_FINAL_MINUTES_MAX_OUTPUT_TOKENS,
    PREVIEW_SCHEMA_VERSION,
    PUBLIC_MINUTES_REVISION,
    TRANSCRIPT_RECONCILIATION_REVISION,
    FullTranscriptMinutesProvider,
    FullTranscriptPreviewError,
    PreparedAuthoritativeMeetingFacts,
    PreparedFullTranscript,
    PreviewTemplateSource,
    build_preview_messages,
    compose_authoritative_minutes,
    load_preview_template,
    normalize_generated_evidence,
    prepare_authoritative_meeting_facts,
    prepare_full_transcript_records,
    reconcile_generated_transcript,
    render_public_minutes,
    request_sha256,
    validate_generated_minutes,
    validate_public_minutes,
)
from lingxu.worker.minutes_models import (
    CanonicalMetrics,
    DeterministicCounts,
    EvidenceCoverage,
    MinutesAgenda,
    MinutesDocumentV2,
    MinutesParticipants,
    MinutesProvenance,
    MinutesTopic,
    MinutesVote,
    PrivacyExclusionCounts,
    SummaryMethod,
    normalise_vote_payload,
)
from lingxu.worker.minutes_renderer import (
    MINUTES_RENDERER_REVISION,
    render_minutes,
    renderer_input_sha256,
)
from lingxu.worker.mute_facts import project_authoritative_mute_summaries
from lingxu.worker.outcome_artifacts import (
    budget_controller,
    decode_outcome_sets,
    outcome_hash_input,
    outcome_policy_provenance,
    read_budget_journal,
    read_partition_results,
    reusable_outcome_payload,
    sanitized_vote_result,
    stored_outcome_policy_provenance,
    validated_outcome_hash_input,
    write_partition_results,
)
from lingxu.worker.outcome_merge import (
    ValidatedOutcomeBindingResolution,
    ValidatedOutcomeCandidateRejection,
    merge_topic_outcomes,
    resolve_validated_topic_bindings,
    validate_validated_binding_artifact,
)
from lingxu.worker.outcome_partitions import (
    OUTCOME_DISPOSITION_REVISION,
    LlmBudgetController,
    OutcomePartitionError,
    OutcomePartitionManifest,
    OutcomePartitionResult,
    boundary_task_hash,
    build_outcome_partition_manifest,
    digest_task_hash,
    expected_partition_result_keys,
    llm_job_hash,
    outcome_task_hash,
    serialize_outcome_request,
    summarize_outcome_dispositions,
    validate_outcome_partition_manifest,
    validate_partition_result_keys,
)
from lingxu.worker.outcomes import (
    MAX_FINAL_CANDIDATES_PER_TOPIC,
    OUTCOME_EXTRACTION_CONFIG_VERSION,
    OutcomeCandidate,
    OutcomeEventFact,
    OutcomeFailureAttempt,
    OutcomeFailureCategory,
    OutcomeProviderError,
    OutcomeProviderFailure,
    OutcomeRequestKind,
    OutcomeSizeBucket,
    OutcomeTaskDisposition,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
    outcome_failure_request_kind,
    outcome_failure_token,
)
from lingxu.worker.publication_recovery import (
    PUBLICATION_ATTACHMENTS,
    PUBLICATION_CONTRACT_REVISION,
    build_publication_manifest,
    publication_is_terminal,
)
from lingxu.worker.publisher import ArtifactPublisher
from lingxu.worker.semantic_policy import SPEECH_ACT_POLICY_REVISION
from lingxu.worker.speaking_facts import project_authoritative_speaking_facts
from lingxu.worker.speaking_plans import project_authoritative_speaking_plans
from lingxu.worker.stages import (
    POST_MEETING_STAGE_DEFINITIONS,
    POST_MEETING_STAGES,
    truncate_completed_before,
)
from lingxu.worker.structured_minutes import (
    STRUCTURED_MINUTES_REVISION,
    build_structured_messages,
    compose_structured_minutes,
    validate_summary_bindings,
)
from lingxu.worker.summary import PostMeetingInferenceProvider
from lingxu.worker.topic_artifacts import (
    read_topic_segments,
    reusable_topic_payload,
    segment_topics,
    stored_topic_policy_provenance,
)
from lingxu.worker.topics import (
    APPROVED_TOPIC_MODEL,
    APPROVED_TOPIC_MODEL_DIGEST,
    APPROVED_TOPIC_PROVIDER,
    TopicSegment,
    topic_candidates,
)

__all__ = ["POST_MEETING_JOB", "POST_MEETING_STAGES", "PostMeetingPipeline", "SegmentTranscriber"]

POST_MEETING_JOB = "POST_MEETING"
_REALTIME_COVERAGE_TOLERANCE_MS = 1_000
_ARTIFACT_PIPELINE_VERSION = 4
_RETIRED_FIX9_OUTCOME_PIPELINE_REVISION = hashlib.sha256(
    b"outcome-evidence-task13-lossless-fix9"
).hexdigest()
_OUTCOME_PIPELINE_REVISION = hashlib.sha256(
    b"outcome-evidence-task13-topic-boundary-cap-120-fix44"
).hexdigest()
_VALIDATED_BINDING_REVISION = hashlib.sha256(
    b"lingxu-validated-outcome-binding-v2-origin-diagnostics"
).hexdigest()
_SPEECH_ACT_POLICY_REVISION = SPEECH_ACT_POLICY_REVISION
_PREVIOUS_SPEECH_ACT_POLICY_REVISION = hashlib.sha256(
    b"lingxu-speech-act-policy-task10r-v1"
).hexdigest()
_REGISTERED_FILE_ARTIFACTS = (
    ("consent_timeline.json", "CONSENT_TIMELINE_JSON"),
    ("privacy_exclusions.json", "PRIVACY_EXCLUSIONS_JSON"),
    ("canonical_transcript.json", "CANONICAL_TRANSCRIPT_JSON"),
    ("topic_segments.json", "TOPIC_SEGMENTS_JSON"),
    ("outcome_partition_manifest.json", "OUTCOME_PARTITION_MANIFEST_JSON"),
    ("outcome_budget_journal.json", "OUTCOME_BUDGET_JOURNAL_JSON"),
    ("outcome_partition_results.json", "OUTCOME_PARTITION_RESULTS_JSON"),
    ("transcript.json", "TRANSCRIPT_JSON"),
    ("transcript.txt", "TRANSCRIPT_TEXT"),
    ("events.json", "MEETING_EVENTS_JSON"),
    ("languages.json", "LANGUAGE_DISTRIBUTION_JSON"),
    ("evidence.json", "EVIDENCE_JSON"),
    ("minutes.md", "MINUTES_MARKDOWN"),
    ("minutes.json", "MINUTES_JSON"),
)
_SPEECH_DEPENDENT_ARTIFACTS = frozenset({"EVIDENCE_JSON", "MINUTES_MARKDOWN", "MINUTES_JSON"})
_PUBLIC_EVENT_PAYLOAD_KEYS = frozenset(
    {
        "agenda_item_id",
        "consent_kinds",
        "counts",
        "effective_at_ms",
        "eligible_count",
        "from_state",
        "reason",
        "responded_count",
        "result",
        "source_entry_id",
        "source_entry_ids",
        "subject",
        "subject_user_id",
        "to_state",
        "total",
        "user_id",
        "vote_content",
        "vote_id",
    }
)

_SUMMARY_EXCLUDED_ENTRY_TYPES = frozenset(
    {"backchannel", "noise", "excluded", "privacy_control", "privacy-control"}
)

CheckpointCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class _OutcomeExtractionContext:
    topics: tuple[TopicSegment, ...]
    manifest: OutcomePartitionManifest
    outcome_path: Path
    results_path: Path
    expected_provenance: dict[str, Any]
    policy_failure: str
    provider_ready: bool
    kind_extractor: Callable[..., Awaitable[TopicOutcomeSet]] | None
    job_hash: str
    outcome_tasks: tuple[str, ...]
    expected_tasks: tuple[str, ...]
    physical_expected: tuple[tuple[str, OutcomeRequestKind], ...]
    llm_budget: LlmBudgetController
    terminal_results: dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_is_covered_by_realtime(
    segment: AudioSegmentRecord,
    entries: tuple[TranscriptEntryRecord, ...],
) -> bool:
    """Return whether primary realtime transcript intervals fully cover one Segment.

    Realtime VAD and the encrypted recorder consume the same per-user PCM but can place their
    silence boundaries slightly differently. Coalesce only realtime-utterance intervals separated
    by at most one bounded tolerance and require coverage to reach both Segment boundaries within
    that same tolerance. Mere overlap is deliberately insufficient so offline ASR remains a
    fallback for a partially missed Segment.
    """

    if segment.ended_ms <= segment.started_ms:
        return False
    ranges = sorted(
        (entry.started_ms, entry.ended_ms)
        for entry in entries
        if entry.source_key.startswith("utterance:")
        and entry.user_id == segment.user_id
        and (entry.normalised_text or entry.raw_text).strip()
        and entry.ended_ms >= segment.started_ms - _REALTIME_COVERAGE_TOLERANCE_MS
        and entry.started_ms <= segment.ended_ms + _REALTIME_COVERAGE_TOLERANCE_MS
    )
    cursor = segment.started_ms
    for started_ms, ended_ms in ranges:
        if ended_ms < cursor - _REALTIME_COVERAGE_TOLERANCE_MS:
            continue
        if started_ms > cursor + _REALTIME_COVERAGE_TOLERANCE_MS:
            return False
        cursor = max(cursor, ended_ms)
        if cursor + _REALTIME_COVERAGE_TOLERANCE_MS >= segment.ended_ms:
            return True
    return False


class PostMeetingPipeline:
    """Run every stage idempotently and resume after the last durable checkpoint."""

    def __init__(
        self,
        *,
        sessions: MeetingSessionRepository,
        events: MeetingEventStore,
        participants: ParticipantRepository,
        agenda: AgendaRepository,
        utterances: RealtimeUtteranceRepository,
        segments: AudioSegmentRepository,
        transcripts: TranscriptRepository,
        artifacts: ArtifactRepository,
        files: SessionFiles,
        transcriber: SegmentTranscriber | None,
        publisher: ArtifactPublisher | None,
        retention_hours: int,
        post_meeting_inference_provider: PostMeetingInferenceProvider | None = None,
        full_transcript_minutes_provider: FullTranscriptMinutesProvider | None = None,
        full_transcript_minutes_template: PreviewTemplateSource | None = None,
        glossary_settings: Settings | None = None,
    ) -> None:
        self._sessions = sessions
        self._events = events
        self._participants = participants
        self._agenda = agenda
        self._utterances = utterances
        self._segments = segments
        self._transcripts = transcripts
        self._artifacts = artifacts
        self._files = files
        self._artifacts.bind_session_files(files)
        self._transcriber = transcriber
        self._publisher = publisher
        self._retention_hours = retention_hours
        self._post_meeting_inference_provider = post_meeting_inference_provider
        self._full_transcript_minutes_provider = full_transcript_minutes_provider
        self._structured_final_minutes = (
            getattr(full_transcript_minutes_provider, "output_format", "text") == "json_object"
        )
        self._full_transcript_minutes_template = full_transcript_minutes_template
        full_transcript_minutes_max_output_tokens = (
            full_transcript_minutes_provider.max_output_tokens
            if full_transcript_minutes_provider is not None
            else DEFAULT_FINAL_MINUTES_MAX_OUTPUT_TOKENS
        )
        if (
            isinstance(full_transcript_minutes_max_output_tokens, bool)
            or not isinstance(full_transcript_minutes_max_output_tokens, int)
            or not 128
            <= full_transcript_minutes_max_output_tokens
            <= MAX_FINAL_MINUTES_MAX_OUTPUT_TOKENS
        ):
            raise ValueError("full transcript minutes output limit is invalid")
        self._full_transcript_minutes_max_output_tokens = full_transcript_minutes_max_output_tokens
        if (
            full_transcript_minutes_provider is not None
            and full_transcript_minutes_template is None
        ):
            raise ValueError("full transcript minutes template is required")
        self._glossary_settings = glossary_settings
        self._privacy_legacy_schema_allowed: dict[str, bool] = {}
        self._privacy_legacy_unavailable_allowed: dict[str, bool] = {}
        self._privacy_checkpoint_state: dict[str, str] = {}
        self._minutes_renderer_inputs: dict[str, str] = {}
        self._full_transcript_attempt_generations: dict[str, int] = {}

    def _meeting_dir(self, session_id: str) -> Path:
        path = self._files.directory(session_id) / "meeting"
        path.mkdir(exist_ok=True)
        return path

    async def run(
        self,
        job: ProcessingJobRecord,
        checkpoint: CheckpointCallback,
    ) -> None:
        if await self._publication_is_terminal(job.session_id):
            return
        checkpoint_revision = job.checkpoint.get("outcome_pipeline_revision")
        if checkpoint_revision == _RETIRED_FIX9_OUTCOME_PIPELINE_REVISION:
            raise OutcomePartitionError("retired_fix9_checkpoint_requires_matching_rollback")
        stored_renderer_input = recover_renderer_input_sha256(job.checkpoint)
        if stored_renderer_input is not None:
            self._minutes_renderer_inputs[job.session_id] = stored_renderer_input
        else:
            self._minutes_renderer_inputs.pop(job.session_id, None)
        privacy_state = await self._recover_privacy_checkpoint_state(job)
        attempt_generation = recover_attempt_generation(job.checkpoint)
        self._full_transcript_attempt_generations[job.session_id] = attempt_generation
        self._privacy_checkpoint_state[job.session_id] = privacy_state
        self._privacy_legacy_schema_allowed[job.session_id] = privacy_state == "legacy_measured_v1"
        self._privacy_legacy_unavailable_allowed[job.session_id] = (
            privacy_state == "legacy_unavailable"
        )
        completed = recover_completed_stages(job.checkpoint)
        completed = await self._truncate_stale_completed(job, completed)
        if "GENERATE_MINUTES" not in completed:
            self._minutes_renderer_inputs.pop(job.session_id, None)
        for definition in POST_MEETING_STAGE_DEFINITIONS:
            stage = definition.name
            if stage in completed:
                continue
            active_checkpoint: dict[str, object] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "privacy_exclusion_journal_version": 1,
                "privacy_exclusion_measurement_state": privacy_state,
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "speech_act_policy_revision": _SPEECH_ACT_POLICY_REVISION,
                "minutes_renderer_revision": MINUTES_RENDERER_REVISION,
                "completed_stages": list(completed),
                "active_stage": stage,
            }
            if renderer_input := self._minutes_renderer_inputs.get(job.session_id):
                active_checkpoint["renderer_input_sha256"] = renderer_input
            if attempt_generation:
                active_checkpoint["full_transcript_minutes_attempt_generation"] = attempt_generation
            await checkpoint(stage, active_checkpoint)
            handler = getattr(self, definition.handler_name)
            details = await handler(job.session_id)
            if stage == "GENERATE_MINUTES":
                renderer_input = details.get("renderer_input_sha256")
                if (
                    not isinstance(renderer_input, str)
                    or re.fullmatch(r"[0-9a-f]{64}", renderer_input) is None
                ):
                    raise ValueError("minutes renderer input digest is invalid")
                self._minutes_renderer_inputs[job.session_id] = renderer_input
            completed.append(stage)
            completed_checkpoint: dict[str, object] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "privacy_exclusion_journal_version": 1,
                "privacy_exclusion_measurement_state": privacy_state,
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "speech_act_policy_revision": _SPEECH_ACT_POLICY_REVISION,
                "minutes_renderer_revision": MINUTES_RENDERER_REVISION,
                "completed_stages": list(completed),
                "last_stage_details": details,
            }
            if renderer_input := self._minutes_renderer_inputs.get(job.session_id):
                completed_checkpoint["renderer_input_sha256"] = renderer_input
            if attempt_generation:
                completed_checkpoint["full_transcript_minutes_attempt_generation"] = (
                    attempt_generation
                )
            await checkpoint(stage, completed_checkpoint)

    async def _recover_privacy_checkpoint_state(self, job: ProcessingJobRecord) -> str:
        checkpoint_completed = {str(stage) for stage in job.checkpoint.get("completed_stages", [])}
        legacy_checkpoint = (
            job.checkpoint.get("schema_version") == 3
            and "APPLY_PRIVACY_CUTOFFS" in checkpoint_completed
            and "privacy_exclusion_journal_version" not in job.checkpoint
        )
        if legacy_checkpoint:
            privacy_state = self._prepare_legacy_privacy_migration(job.session_id)
            timeline = await self._consent_timeline(job.session_id)
            atomic_write_text(
                self._meeting_dir(job.session_id) / "consent_timeline.json",
                _json_text(self._consent_timeline_payload(timeline)),
            )
            return privacy_state
        if job.checkpoint.get("privacy_exclusion_journal_version") != 1:
            return "journal_v2"
        checkpoint_state = job.checkpoint.get("privacy_exclusion_measurement_state")
        if checkpoint_state is None:
            return "journal_v2"
        if checkpoint_state in {
            "journal_v2",
            "legacy_measured_v1",
            "legacy_unavailable",
        }:
            return str(checkpoint_state)
        raise ValueError("privacy exclusion checkpoint state is invalid")

    async def _truncate_stale_completed(
        self,
        job: ProcessingJobRecord,
        completed: list[str],
    ) -> list[str]:
        if job.checkpoint.get("schema_version") not in {3, CHECKPOINT_SCHEMA_VERSION}:
            return completed
        revision = job.checkpoint.get("outcome_pipeline_revision")
        if revision != _OUTCOME_PIPELINE_REVISION:
            return truncate_completed_before(completed, "EXTRACT_TOPIC_OUTCOMES")
        if "EXTRACT_TOPIC_OUTCOMES" in completed and not await self._candidate_outcomes_are_current(
            job.session_id
        ):
            return truncate_completed_before(completed, "EXTRACT_TOPIC_OUTCOMES")
        if "LINK_EVIDENCE" in completed and not await self._linked_outcomes_are_current(
            job.session_id
        ):
            return truncate_completed_before(completed, "LINK_EVIDENCE")
        if "VALIDATE_OUTCOMES" in completed and (
            job.checkpoint.get("speech_act_policy_revision") != _SPEECH_ACT_POLICY_REVISION
            or not await self._validated_outcomes_are_current(job.session_id)
        ):
            return truncate_completed_before(completed, "VALIDATE_OUTCOMES")
        if "MERGE_MEETING_EVENTS" in completed and not await self._meeting_events_are_current(
            job.session_id
        ):
            return truncate_completed_before(completed, "MERGE_MEETING_EVENTS")
        if "GENERATE_EVIDENCE" in completed and not await self._evidence_is_current(
            job.session_id,
            require_registered="RENDER_ARTIFACTS" in completed,
        ):
            return truncate_completed_before(completed, "GENERATE_EVIDENCE")
        if "GENERATE_MINUTES" in completed and (
            job.checkpoint.get("minutes_renderer_revision") != MINUTES_RENDERER_REVISION
            or job.session_id not in self._minutes_renderer_inputs
            or not await self._minutes_are_current(
                job.session_id,
                expected_renderer_input_sha256=self._minutes_renderer_inputs.get(job.session_id),
            )
        ):
            return truncate_completed_before(completed, "GENERATE_MINUTES")
        if "RENDER_ARTIFACTS" in completed and not await self._registered_outputs_are_current(
            job.session_id
        ):
            return truncate_completed_before(completed, "RENDER_ARTIFACTS")
        return completed

    def _prepare_legacy_privacy_migration(self, session_id: str) -> str:
        """Create/validate the legacy journal before any upgraded checkpoint is emitted."""

        path = self._meeting_dir(session_id) / "privacy_exclusions.json"
        if not path.is_file():
            payload = self._privacy_journal_payload(
                state="legacy_unavailable",
                candidate_hashes={
                    "transcript_entries": [],
                    "segment_rows": [],
                    "segment_files": [],
                },
                measurements=None,
            )
            atomic_write_text(path, _json_text(payload))
            return "legacy_unavailable"
        raw = json.loads(path.read_text(encoding="utf-8"))
        journal = self._read_privacy_journal(session_id, allow_legacy=True)
        if journal is None:
            raise ValueError("privacy exclusion legacy migration is incomplete")
        if isinstance(raw, dict) and raw.get("schema_version") == 1:
            return "legacy_measured_v1"
        if journal["state"] == "legacy_unavailable":
            return "legacy_unavailable"
        if journal["state"] == "complete":
            return "journal_v2"
        raise ValueError("privacy exclusion legacy migration is incomplete")

    async def _stage_verify_session(self, session_id: str) -> dict[str, Any]:
        session = await self._sessions.get(session_id)
        manifest = self._files.read_manifest(session_id)
        if manifest.guild_id != session.guild_id:
            raise ValueError("Session manifest guild differs from database")
        return {"manifest_schema_version": manifest.schema_version}

    async def _consent_timeline(self, session_id: str) -> ConsentTimeline:
        """Project privacy eligibility from the authoritative append-only event stream."""

        return ConsentTimeline.from_events(await self._events.list(session_id))

    async def _stage_build_consent_timeline(self, session_id: str) -> dict[str, Any]:
        timeline = await self._consent_timeline(session_id)
        payload = self._consent_timeline_payload(timeline)
        atomic_write_text(
            self._meeting_dir(session_id) / "consent_timeline.json",
            _json_text(payload),
        )
        return {"consent_intervals": len(timeline.intervals)}

    @staticmethod
    def _consent_timeline_payload(timeline: ConsentTimeline) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "exclusion_journal_version": 1,
            "intervals": [
                {
                    "user_id": interval.user_id,
                    "consent_kind": interval.kind.value,
                    "started_ms": interval.started_ms,
                    "ended_ms": interval.ended_ms,
                }
                for interval in timeline.intervals
            ],
        }

    @staticmethod
    def _allows_recording_segment(
        timeline: ConsentTimeline,
        segment: AudioSegmentRecord,
    ) -> bool:
        """Apply the recorder's exclusive-end interval rule to raw Segment retention."""

        if segment.started_ms < 0 or segment.ended_ms <= segment.started_ms:
            return False
        return any(
            interval.user_id == segment.user_id
            and interval.kind is ConsentKind.RECORDING
            and interval.started_ms <= segment.started_ms
            and (interval.ended_ms is None or segment.ended_ms <= interval.ended_ms)
            for interval in timeline.intervals
        )

    async def _stage_apply_privacy_cutoffs(self, session_id: str) -> dict[str, Any]:
        timeline = await self._consent_timeline(session_id)
        transcript_entries = tuple(
            entry
            for entry in await self._transcripts.list(session_id)
            if not entry.source_key.startswith("BOT_TTS:")
            and (
                entry.user_id is None
                or not timeline.allows_interval(
                    entry.user_id,
                    ConsentKind.MINUTES,
                    entry.started_ms,
                    entry.ended_ms,
                )
            )
        )
        removed_segments = tuple(
            segment
            for segment in await self._segments.list_session(session_id)
            if not self._allows_recording_segment(timeline, segment)
        )
        journal_path = self._meeting_dir(session_id) / "privacy_exclusions.json"
        journal = self._read_privacy_journal(session_id, allow_legacy=False)
        if journal is not None and journal["state"] == "complete":
            measurements = journal["measurements"]
            assert isinstance(measurements, dict)
            return {
                "removed_transcript_entries": measurements["transcript_entries"],
                "removed_segment_rows": measurements["segment_rows"],
                "removed_segment_files": measurements["segment_files"],
            }
        if journal is None:
            file_segments = tuple(
                segment
                for segment in removed_segments
                if self._files.resolve(session_id, segment.encrypted_path).is_file()
            )
            candidate_hashes = {
                "transcript_entries": [
                    self._privacy_candidate_hash("transcript_entry", entry.id)
                    for entry in transcript_entries
                ],
                "segment_rows": [
                    self._privacy_candidate_hash("segment_row", segment.id)
                    for segment in removed_segments
                ],
                "segment_files": [
                    self._privacy_candidate_hash("segment_file", segment.id)
                    for segment in file_segments
                ],
            }
            journal = self._privacy_journal_payload(
                state="prepared",
                candidate_hashes=candidate_hashes,
                measurements=None,
            )
            atomic_write_text(journal_path, _json_text(journal))
        transcript_hashes = set(journal["candidate_hashes"]["transcript_entries"])
        segment_row_hashes = set(journal["candidate_hashes"]["segment_rows"])
        segment_file_hashes = set(journal["candidate_hashes"]["segment_files"])
        await self._transcripts.delete_ids(
            session_id=session_id,
            entry_ids=tuple(
                entry.id
                for entry in transcript_entries
                if self._privacy_candidate_hash("transcript_entry", entry.id) in transcript_hashes
            ),
        )
        for segment in removed_segments:
            if self._privacy_candidate_hash("segment_file", segment.id) not in segment_file_hashes:
                continue
            path = self._files.resolve(session_id, segment.encrypted_path)
            await asyncio.to_thread(path.unlink, missing_ok=True)
            if path.exists():
                raise OSError("privacy exclusion file deletion did not complete")
        await self._segments.delete_ids(
            session_id=session_id,
            segment_ids=tuple(
                segment.id
                for segment in removed_segments
                if self._privacy_candidate_hash("segment_row", segment.id) in segment_row_hashes
            ),
        )
        remaining_transcript_hashes = {
            self._privacy_candidate_hash("transcript_entry", entry.id)
            for entry in await self._transcripts.list(session_id)
        }
        remaining_segment_hashes = {
            self._privacy_candidate_hash("segment_row", segment.id)
            for segment in await self._segments.list_session(session_id)
        }
        if (
            transcript_hashes & remaining_transcript_hashes
            or segment_row_hashes & remaining_segment_hashes
        ):
            raise RuntimeError("privacy exclusion reconciliation is incomplete")
        counts = journal["candidate_counts"]
        complete = self._privacy_journal_payload(
            state="complete",
            candidate_hashes=journal["candidate_hashes"],
            measurements=counts,
        )
        atomic_write_text(journal_path, _json_text(complete))
        details = {
            "removed_transcript_entries": counts["transcript_entries"],
            "removed_segment_rows": counts["segment_rows"],
            "removed_segment_files": counts["segment_files"],
        }
        return details

    async def _stage_verify_audio_segments(self, session_id: str) -> dict[str, Any]:
        segments = await self._segments.list_session(session_id)
        for segment in segments:
            encrypted_path = self._files.resolve(session_id, segment.encrypted_path)
            if not encrypted_path.is_file():
                raise FileNotFoundError(encrypted_path)
            if _sha256(encrypted_path) != segment.sha256:
                raise ValueError(f"recorded Segment hash mismatch: {segment.id}")
        return {"verified_segments": len(segments)}

    async def _stage_finalise_realtime_utterances(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        timeline = await self._consent_timeline(session_id)
        added = 0
        for utterance in await self._utterances.list_session(session_id):
            if not timeline.allows_interval(
                utterance.user_id,
                ConsentKind.MINUTES,
                utterance.started_ms,
                utterance.ended_ms,
            ):
                continue
            await self._transcripts.upsert(
                session_id=session_id,
                source_key=f"utterance:{utterance.id}",
                user_id=utterance.user_id,
                raw_text=utterance.raw_text,
                normalised_text=utterance.normalised_text,
                started_ms=utterance.started_ms,
                ended_ms=utterance.ended_ms,
            )
            added += 1
        return {"finalised_utterances": added}

    async def _stage_transcribe_missing_segments(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        timeline = await self._consent_timeline(session_id)
        entries = await self._transcripts.list(session_id)
        existing = {entry.source_key for entry in entries}
        eligible = [
            segment
            for segment in await self._segments.list_session(session_id)
            if timeline.allows_interval(
                segment.user_id,
                ConsentKind.MINUTES,
                segment.started_ms,
                segment.ended_ms,
            )
            and f"segment:{segment.id}" not in existing
        ]
        covered = [
            segment for segment in eligible if _segment_is_covered_by_realtime(segment, entries)
        ]
        missing = [segment for segment in eligible if segment not in covered]
        if missing and self._transcriber is None:
            raise RuntimeError("offline Segment ASR is not configured")
        added = 0
        for segment in missing:
            assert self._transcriber is not None
            encrypted_path = self._files.resolve(session_id, segment.encrypted_path)
            raw_text, normalised_text = await self._transcriber.transcribe(
                session_id=session_id,
                user_id=segment.user_id,
                encrypted_path=encrypted_path,
                started_ms=segment.started_ms,
                ended_ms=segment.ended_ms,
            )
            if not (normalised_text or raw_text).strip():
                continue
            await self._transcripts.upsert(
                session_id=session_id,
                source_key=f"segment:{segment.id}",
                user_id=segment.user_id,
                raw_text=raw_text,
                normalised_text=normalised_text,
                started_ms=segment.started_ms,
                ended_ms=segment.ended_ms,
            )
            added += 1
        return {
            "eligible_segments": len(eligible),
            "realtime_covered_segments": len(covered),
            "transcribed_segments": len(missing),
            "added_segment_entries": added,
        }

    async def _stage_validate_languages(self, session_id: str) -> dict[str, Any]:
        blank_entries_removed = await self._transcripts.delete_blank(session_id)
        entries = await self._transcripts.list(session_id)
        invalid = [entry.id for entry in entries if entry.ended_ms < entry.started_ms]
        if invalid:
            raise ValueError("transcript entries failed structural validation")
        return {
            "validated_entries": len(entries),
            "blank_entries_removed": blank_entries_removed,
        }

    def _canonical_path(self, session_id: str) -> Path:
        return self._meeting_dir(session_id) / "canonical_transcript.json"

    def _write_canonical_transcript(
        self,
        session_id: str,
        transcript: CanonicalTranscript,
    ) -> None:
        atomic_write_text(self._canonical_path(session_id), _json_text(transcript.to_dict()))

    def _read_canonical_transcript(self, session_id: str) -> CanonicalTranscript:
        payload = json.loads(self._canonical_path(session_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("canonical transcript is malformed")
        return CanonicalTranscript.from_dict(payload)

    async def _build_enriched_canonical(self, session_id: str) -> CanonicalTranscript:
        """Rebuild canonical input from privacy-filtered revision-aware source rows."""

        canonical = build_canonical_transcript(await self._transcripts.list(session_id))
        _, _, labels = await self._speaker_context(session_id)
        return CanonicalTranscript(
            entries=tuple(
                replace(
                    entry,
                    speaker_name=(
                        "聆序机器人"
                        if entry.speaker_user_id is None
                        else labels.get(entry.speaker_user_id)
                    ),
                )
                for entry in canonical.entries
            ),
            metrics=canonical.metrics,
        )

    async def _stage_canonicalise_transcript(self, session_id: str) -> dict[str, Any]:
        enriched = await self._build_enriched_canonical(session_id)
        self._write_canonical_transcript(session_id, enriched)
        return {
            "source_entries": enriched.metrics.source_entry_count,
            "canonical_entries": enriched.metrics.canonical_entry_count,
            "duplicate_entries": enriched.metrics.duplicate_entry_count,
        }

    async def _stage_normalise_entities(self, session_id: str) -> dict[str, Any]:
        normalised, glossary_count = await self._build_normalised_canonical(session_id)
        self._write_canonical_transcript(session_id, normalised)
        return {
            "glossary_layers": glossary_count,
            "normalisation_changes": sum(
                len(entry.normalisation_changes) for entry in normalised.entries
            ),
        }

    async def _build_normalised_canonical(self, session_id: str) -> tuple[CanonicalTranscript, int]:
        canonical = await self._build_enriched_canonical(session_id)
        session = await self._sessions.get(session_id)
        glossaries = (
            load_glossaries(
                self._glossary_settings,
                session_id=session_id,
                guild_id=session.guild_id,
            )
            if self._glossary_settings is not None
            else ()
        )
        normalised = CanonicalTranscript(
            entries=tuple(
                normalise_entities(entry, glossaries=glossaries) for entry in canonical.entries
            ),
            metrics=canonical.metrics,
        )
        return normalised, len(glossaries)

    async def _stage_segment_topics(self, session_id: str) -> dict[str, Any]:
        return await segment_topics(
            canonical=self._read_canonical_transcript(session_id),
            meeting_dir=self._meeting_dir(session_id),
            provider=self._post_meeting_inference_provider,
            budget_controller_factory=budget_controller,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )

    def _read_topic_segments(self, session_id: str) -> tuple[TopicSegment, ...]:
        return read_topic_segments(self._meeting_dir(session_id))

    async def _outcome_events_by_topic(
        self,
        session_id: str,
        topics: tuple[TopicSegment, ...],
        events: tuple[Any, ...],
    ) -> dict[str, tuple[OutcomeEventFact, ...]]:
        """Expose only sanitized, evidence-bound authoritative vote facts."""

        del session_id
        by_topic: dict[str, tuple[OutcomeEventFact, ...]] = {}
        for topic in topics:
            topic_entry_ids = frozenset(topic.entry_ids)
            facts: list[OutcomeEventFact] = []
            for event in events:
                if event.event_type != "VOTE_CLOSED":
                    continue
                payload = event.payload
                if not isinstance(payload, dict):
                    continue
                payload_topic_id = payload.get("topic_id")
                if payload_topic_id is not None and not isinstance(payload_topic_id, str):
                    continue
                if payload_topic_id is not None and payload_topic_id != topic.id:
                    continue
                source_value = payload.get("source_entry_ids")
                if source_value is None and payload.get("source_entry_id") is not None:
                    source_value = [payload["source_entry_id"]]
                if not isinstance(source_value, (list, tuple)) or not all(
                    isinstance(entry_id, str) for entry_id in source_value
                ):
                    continue
                source_entry_ids = frozenset(source_value)
                if (
                    not source_entry_ids
                    or len(source_entry_ids) > 12
                    or not source_entry_ids <= topic_entry_ids
                ):
                    continue
                subject = payload.get("vote_content")
                if not isinstance(subject, str) or not subject.strip() or len(subject) > 500:
                    continue
                facts.append(
                    OutcomeEventFact(
                        sequence_number=event.sequence_number,
                        topic_id=topic.id,
                        subject=subject.strip(),
                        result=sanitized_vote_result(payload),
                        source_entry_ids=tuple(sorted(source_entry_ids)),
                    )
                )
            by_topic[topic.id] = tuple(facts)
        return by_topic

    def _candidate_outcomes_payload(
        self,
        *,
        topics: tuple[TopicSegment, ...],
        manifest: OutcomePartitionManifest,
        terminal_results: dict[tuple[str, OutcomeRequestKind], OutcomePartitionResult],
        expected_provenance: dict[str, object],
        policy_failure: str,
        provider_ready: bool,
    ) -> tuple[dict[str, Any], tuple[TopicOutcomeSet, ...]]:
        """Rebuild the candidate artifact exactly from terminal partition authority."""

        outcome_sets: list[TopicOutcomeSet] = []
        for topic in topics:
            topic_results = tuple(
                (
                    partition,
                    terminal_results[(partition.partition_id, kind)].as_topic_outcome_set(),
                )
                for partition in manifest.partitions
                if partition.topic_id == topic.id
                for kind in OutcomeRequestKind
                if (partition.partition_id, kind) in terminal_results
            )
            if policy_failure != "approved":
                outcome = TopicOutcomeSet(
                    topic_id=topic.id,
                    candidates=(),
                    extraction_method="deterministic_fallback",
                    failure_types=(policy_failure,),
                )
            elif not provider_ready:
                outcome = TopicOutcomeSet(
                    topic_id=topic.id,
                    candidates=(),
                    extraction_method="deterministic_fallback",
                    failure_types=("provider_unsupported",),
                )
            else:
                candidates = tuple(
                    candidate for _, result in topic_results for candidate in result.candidates
                )
                failures = tuple(
                    failure for _, result in topic_results for failure in result.provider_failures
                )
                outcome = TopicOutcomeSet(
                    topic_id=topic.id,
                    candidates=candidates,
                    extraction_method=(
                        "deterministic_fallback"
                        if not candidates
                        and all(
                            result.extraction_method == "deterministic_fallback"
                            for _, result in topic_results
                        )
                        else "deepseek"
                        if not failures
                        else "deepseek_partial"
                    ),
                    failure_types=tuple(
                        sorted(
                            {
                                failure
                                for _, result in topic_results
                                for failure in result.failure_types
                            }
                        )
                    ),
                    provider_failures=failures,
                )
            outcome_sets.append(outcome)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "provenance": expected_provenance,
            "topics": [outcome.model_dump(mode="json") for outcome in outcome_sets],
        }
        payload["provenance"]["outcome_artifact_hash"] = _payload_hash(outcome_hash_input(payload))
        return payload, tuple(outcome_sets)

    def _persist_outcome_manifest(
        self,
        path: Path,
        manifest: OutcomePartitionManifest,
        *,
        canonical_hash: str,
        topic_artifact_hash: str,
        event_facts_hash: str,
        participant_hash: str,
        topics_input: tuple[
            tuple[
                TopicSegment,
                tuple[CanonicalTranscriptEntry, ...],
                tuple[OutcomeEventFact, ...],
            ],
            ...,
        ],
        allowed_participant_ids: tuple[int, ...],
    ) -> None:
        if path.is_file():
            try:
                existing = OutcomePartitionManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                validate_outcome_partition_manifest(
                    existing,
                    canonical_hash=canonical_hash,
                    topic_artifact_hash=topic_artifact_hash,
                    event_facts_hash=event_facts_hash,
                    participant_hash=participant_hash,
                    topics=topics_input,
                    pipeline_revision=_OUTCOME_PIPELINE_REVISION,
                    allowed_participant_ids=allowed_participant_ids,
                )
                if existing != manifest:
                    raise OutcomePartitionError("manifest_tamper")
            except (OSError, UnicodeError, ValueError) as exc:
                raise OutcomePartitionError("manifest_tamper") from exc
        atomic_write_text(path, _json_text(manifest.model_dump(mode="json")))

    @staticmethod
    def _outcome_boundary_tasks(topic_path: Path, job_hash: str) -> tuple[tuple[str, ...], bool]:
        payload = json.loads(topic_path.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        provenance = payload.get("provenance")
        if not isinstance(candidates, list) or not isinstance(provenance, dict):
            raise OutcomePartitionError("topic_artifact_invalid")
        tasks = tuple(
            boundary_task_hash(job_hash, str(candidate.get("entry_id")))
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("entry_id"), str)
        )
        if len(tasks) != len(candidates):
            raise OutcomePartitionError("topic_artifact_invalid")
        return tasks, provenance.get("status") == "approved"

    async def _prepare_outcome_extraction(
        self,
        session_id: str,
    ) -> _OutcomeExtractionContext:
        canonical = self._read_canonical_transcript(session_id)
        topics = self._read_topic_segments(session_id)
        meeting_dir = self._meeting_dir(session_id)
        topic_path = meeting_dir / "topic_segments.json"
        _, _, labels = await self._speaker_context(session_id)
        events_by_topic = await self._outcome_events_by_topic(
            session_id,
            topics,
            await self._events.list(session_id),
        )
        policy = outcome_policy_provenance(self._post_meeting_inference_provider)
        canonical_by_id = {entry.entry_id: entry for entry in canonical.entries}
        event_facts_hash = _payload_hash(
            {
                topic_id: [event.model_dump(mode="json") for event in events]
                for topic_id, events in events_by_topic.items()
            }
        )
        participant_hash = _payload_hash(sorted(labels))
        topic_artifact_hash = _sha256(topic_path)
        allowed_participant_ids = tuple(sorted(labels))
        topics_input = tuple(
            (
                topic,
                tuple(
                    canonical_by_id[entry_id]
                    for entry_id in topic.entry_ids
                    if entry_id in canonical_by_id
                ),
                events_by_topic.get(topic.id, ()),
            )
            for topic in topics
        )
        manifest = build_outcome_partition_manifest(
            canonical_hash=canonical.serialized_hash,
            topic_artifact_hash=topic_artifact_hash,
            event_facts_hash=event_facts_hash,
            participant_hash=participant_hash,
            topics=topics_input,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            allowed_participant_ids=allowed_participant_ids,
        )
        validate_outcome_partition_manifest(
            manifest,
            canonical_hash=canonical.serialized_hash,
            topic_artifact_hash=topic_artifact_hash,
            event_facts_hash=event_facts_hash,
            participant_hash=participant_hash,
            topics=topics_input,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            allowed_participant_ids=allowed_participant_ids,
        )
        self._persist_outcome_manifest(
            meeting_dir / "outcome_partition_manifest.json",
            manifest,
            canonical_hash=canonical.serialized_hash,
            topic_artifact_hash=topic_artifact_hash,
            event_facts_hash=event_facts_hash,
            participant_hash=participant_hash,
            topics_input=topics_input,
            allowed_participant_ids=allowed_participant_ids,
        )

        provider = self._post_meeting_inference_provider
        kind_extractor = (
            getattr(provider, "extract_topic_outcome_kind", None) if provider is not None else None
        )
        binder = getattr(provider, "bind_llm_budget", None) if provider is not None else None
        policy_failure = str(policy["status"])
        provider_ready = (
            policy_failure == "approved" and callable(kind_extractor) and callable(binder)
        )
        job_hash = llm_job_hash(
            canonical_hash=canonical.serialized_hash,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )
        boundary_tasks, boundary_approved = self._outcome_boundary_tasks(topic_path, job_hash)
        outcome_tasks = tuple(
            outcome_task_hash(job_hash, partition, kind)
            for partition in manifest.partitions
            for kind in OutcomeRequestKind
        )
        prior_tasks = (
            (digest_task_hash(job_hash), *boundary_tasks)
            if boundary_approved and boundary_tasks
            else ()
        )
        expected_tasks = (
            (digest_task_hash(job_hash), *boundary_tasks, *outcome_tasks)
            if provider_ready
            else prior_tasks
        )
        llm_budget = budget_controller(
            meeting_dir / "outcome_budget_journal.json",
            job_hash=job_hash,
            expected_task_hashes=expected_tasks,
            prior_task_hashes=prior_tasks,
        )
        if provider_ready:
            cast(Callable[[LlmBudgetController], None], binder)(llm_budget)
        results_path = meeting_dir / "outcome_partition_results.json"
        terminal_results = read_partition_results(
            results_path,
            manifest,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            journal=llm_budget.journal,
        )
        expected_provenance = {
            **policy,
            "provider_interface": "single_kind" if provider_ready else "unsupported",
            "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
            "canonical_input_hash": canonical.serialized_hash,
            "topic_artifact_sha256": topic_artifact_hash,
            "event_facts_hash": event_facts_hash,
            "allowed_participants_hash": participant_hash,
            "outcome_config_version": OUTCOME_EXTRACTION_CONFIG_VERSION,
            "outcome_disposition_revision": OUTCOME_DISPOSITION_REVISION,
            "partition_manifest_hash": manifest.manifest_hash,
        }
        return _OutcomeExtractionContext(
            topics=topics,
            manifest=manifest,
            outcome_path=meeting_dir / "topic_outcome_candidates.json",
            results_path=results_path,
            expected_provenance=expected_provenance,
            policy_failure=policy_failure,
            provider_ready=provider_ready,
            kind_extractor=(
                cast(Callable[..., Awaitable[TopicOutcomeSet]], kind_extractor)
                if callable(kind_extractor)
                else None
            ),
            job_hash=job_hash,
            outcome_tasks=outcome_tasks,
            expected_tasks=expected_tasks,
            physical_expected=(expected_partition_result_keys(manifest) if provider_ready else ()),
            llm_budget=llm_budget,
            terminal_results=terminal_results,
        )

    @staticmethod
    def _closed_outcome_failure(
        exc: Exception,
        kind: OutcomeRequestKind,
    ) -> OutcomeProviderFailure:
        if isinstance(exc, OutcomeProviderError):
            return exc.failure
        return OutcomeProviderFailure(
            category=OutcomeFailureCategory.UNKNOWN,
            request_kind=outcome_failure_request_kind(kind),
            attempt=OutcomeFailureAttempt.PREFLIGHT,
            count=1,
            size_bucket=OutcomeSizeBucket.NOT_OBSERVED,
            capacity_subtype=None,
        )

    @staticmethod
    def _outcome_stage_details(
        outcome_sets: tuple[TopicOutcomeSet, ...],
        manifest: OutcomePartitionManifest,
        disposition_metrics: dict[str, Any],
        *,
        artifact_reused: bool,
    ) -> dict[str, Any]:
        provider_failures = tuple(
            failure for outcome in outcome_sets for failure in outcome.provider_failures
        )
        return {
            "topic_count": len(outcome_sets),
            "candidate_count": sum(len(outcome.candidates) for outcome in outcome_sets),
            "failure_types": sorted(
                {failure for outcome in outcome_sets for failure in outcome.failure_types}
            ),
            "provider_failure_count": sum(failure.count for failure in provider_failures),
            "provider_failure_facts": sorted(
                {outcome_failure_token(failure) for failure in provider_failures}
            ),
            "artifact_reused": artifact_reused,
            **disposition_metrics,
            **manifest.telemetry(),
        }

    @staticmethod
    def _validate_outcome_result_authority(
        context: _OutcomeExtractionContext,
    ) -> dict[str, Any]:
        validate_partition_result_keys(
            tuple(context.terminal_results),
            context.physical_expected,
        )
        context.llm_budget.journal.validate_expected(context.expected_tasks)
        context.llm_budget.journal.validate_final()
        return summarize_outcome_dispositions(
            context.terminal_results,
            context.llm_budget.journal,
            context.manifest,
        ).model_dump(mode="json")

    def _try_reuse_outcome_artifact(
        self,
        context: _OutcomeExtractionContext,
    ) -> dict[str, Any] | None:
        reusable = reusable_outcome_payload(
            path=context.outcome_path,
            expected_provenance=context.expected_provenance,
            topic_ids=tuple(topic.id for topic in context.topics),
        )
        if reusable is None:
            return None
        partitions = {
            partition.partition_id: partition for partition in context.manifest.partitions
        }
        validate_partition_result_keys(
            tuple(context.terminal_results),
            context.physical_expected,
        )
        for (partition_id, kind), result in context.terminal_results.items():
            partition = partitions[partition_id]
            task_hash = outcome_task_hash(context.job_hash, partition, kind)
            if task_hash not in context.llm_budget.journal.result_task_hashes:
                raise OutcomePartitionError("partition_result_tamper")
            result.validate_for(partition, kind, _OUTCOME_PIPELINE_REVISION)
        disposition_metrics = self._validate_outcome_result_authority(context)
        expected_payload, reusable_sets = self._candidate_outcomes_payload(
            topics=context.topics,
            manifest=context.manifest,
            terminal_results=context.terminal_results,
            expected_provenance=context.expected_provenance,
            policy_failure=context.policy_failure,
            provider_ready=context.provider_ready,
        )
        if reusable != expected_payload or context.outcome_path.read_bytes() != _json_text(
            expected_payload
        ).encode("utf-8"):
            return None
        return self._outcome_stage_details(
            reusable_sets,
            context.manifest,
            disposition_metrics,
            artifact_reused=True,
        )

    @staticmethod
    def _reconcile_existing_outcome_results(context: _OutcomeExtractionContext) -> None:
        expected_outcome_set = set(context.outcome_tasks)
        result_tasks: set[str] = set()
        partitions = {
            partition.partition_id: partition for partition in context.manifest.partitions
        }
        for partition_id, kind in context.terminal_results:
            task_hash = outcome_task_hash(context.job_hash, partitions[partition_id], kind)
            if task_hash not in expected_outcome_set:
                raise OutcomePartitionError("unknown_task")
            if task_hash not in context.llm_budget.journal.reserved_task_hashes:
                raise OutcomePartitionError("unreserved_result")
            context.llm_budget.reconcile_result(task_hash)
            result_tasks.add(task_hash)
        completed_without_result = (
            set(context.llm_budget.journal.completed_task_hashes) & expected_outcome_set
        ) - result_tasks
        if completed_without_result:
            raise OutcomePartitionError("completed_result_missing")

    async def _extract_missing_outcome_kind(
        self,
        context: _OutcomeExtractionContext,
        partition: Any,
        kind: OutcomeRequestKind,
    ) -> None:
        task_hash = outcome_task_hash(context.job_hash, partition, kind)
        if task_hash in context.llm_budget.journal.completed_task_hashes:
            raise OutcomePartitionError("completed_result_missing")
        context.llm_budget.reserve_task(task_hash)
        try:
            extract_one = cast(Callable[..., Awaitable[TopicOutcomeSet]], context.kind_extractor)
            extracted = await extract_one(
                partition=partition,
                kind=kind,
                request_bytes=serialize_outcome_request(partition, kind),
            )
            if extracted.topic_id != partition.topic_id:
                raise SummaryProviderError("outcome provider returned a different topic")
        except (OutcomeProviderError, SummaryProviderError) as exc:
            extracted = TopicOutcomeSet(
                topic_id=partition.topic_id,
                candidates=(),
                extraction_method="deterministic_fallback",
                failure_types=(),
                provider_failures=(self._closed_outcome_failure(exc, kind),),
                task_disposition=OutcomeTaskDisposition.TERMINAL_FAILURE,
            )
        context.terminal_results[(partition.partition_id, kind)] = OutcomePartitionResult.terminal(
            partition=partition,
            kind=kind,
            result=extracted,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )
        write_partition_results(
            context.results_path,
            context.manifest,
            context.terminal_results,
        )
        context.llm_budget.reconcile_result(task_hash)

    async def _run_missing_outcome_tasks(self, context: _OutcomeExtractionContext) -> None:
        if not context.provider_ready:
            return
        self._reconcile_existing_outcome_results(context)
        for partition in context.manifest.partitions:
            missing_kinds = tuple(
                kind
                for kind in OutcomeRequestKind
                if (partition.partition_id, kind) not in context.terminal_results
            )
            for kind in missing_kinds:
                await self._extract_missing_outcome_kind(context, partition, kind)

    def _finalize_outcome_extraction(
        self,
        context: _OutcomeExtractionContext,
    ) -> dict[str, Any]:
        disposition_metrics = self._validate_outcome_result_authority(context)
        write_partition_results(
            context.results_path,
            context.manifest,
            context.terminal_results,
        )
        payload, rebuilt_sets = self._candidate_outcomes_payload(
            topics=context.topics,
            manifest=context.manifest,
            terminal_results=context.terminal_results,
            expected_provenance=context.expected_provenance,
            policy_failure=context.policy_failure,
            provider_ready=context.provider_ready,
        )
        atomic_write_text(context.outcome_path, _json_text(payload))
        return self._outcome_stage_details(
            rebuilt_sets,
            context.manifest,
            disposition_metrics,
            artifact_reused=False,
        )

    async def _stage_extract_topic_outcomes(self, session_id: str) -> dict[str, Any]:
        context = await self._prepare_outcome_extraction(session_id)
        reusable = self._try_reuse_outcome_artifact(context)
        if reusable is not None:
            return reusable
        await self._run_missing_outcome_tasks(context)
        return self._finalize_outcome_extraction(context)

    async def _outcome_evidence_contexts(
        self,
        session_id: str,
    ) -> dict[str, EvidenceContext]:
        canonical = self._read_canonical_transcript(session_id)
        _, _, labels = await self._speaker_context(session_id)
        topics = self._read_topic_segments(session_id)
        events_by_topic = await self._outcome_events_by_topic(
            session_id,
            topics,
            await self._events.list(session_id),
        )
        canonical_by_id = {entry.entry_id: entry for entry in canonical.entries}
        return {
            topic.id: EvidenceContext(
                topic=topic,
                canonical_entries=tuple(
                    canonical_by_id[entry_id]
                    for entry_id in topic.entry_ids
                    if entry_id in canonical_by_id
                ),
                authoritative_events=events_by_topic.get(topic.id, ()),
                allowed_participant_ids=frozenset(labels),
                participant_names=labels,
            )
            for topic in topics
        }

    def _read_outcome_sets(self, path: Path) -> tuple[TopicOutcomeSet, ...]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("outcome artifact is malformed")
        return decode_outcome_sets(payload.get("topics"))

    def _candidate_origin_authority(
        self,
        session_id: str,
        outcome_sets: tuple[TopicOutcomeSet, ...],
    ) -> dict[str, str]:
        """Map every candidate to one terminal partition without exposing partition identity."""

        meeting_dir = self._meeting_dir(session_id)
        candidate_payload = json.loads(
            (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
        )
        provenance = (
            candidate_payload.get("provenance") if isinstance(candidate_payload, dict) else None
        )
        if not isinstance(provenance, dict):
            raise OutcomePartitionError("candidate_origin_authority_invalid")
        stored_outcome_policy_provenance(provenance)
        provider_interface = provenance.get("provider_interface")
        manifest = OutcomePartitionManifest.model_validate_json(
            (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
        )
        validate_outcome_partition_manifest(manifest)
        results = read_partition_results(
            meeting_dir / "outcome_partition_results.json",
            manifest,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )
        expected_keys = (
            expected_partition_result_keys(manifest) if provider_interface == "single_kind" else ()
        )
        validate_partition_result_keys(tuple(results), expected_keys)
        origins: dict[str, str] = {}
        for (partition_id, _), result in sorted(
            results.items(), key=lambda item: (item[0][0], item[0][1].value)
        ):
            for candidate in result.candidates:
                if candidate.id in origins:
                    raise OutcomePartitionError("candidate_origin_authority_invalid")
                origins[candidate.id] = partition_id
        observed_ids = tuple(
            candidate.id for outcome_set in outcome_sets for candidate in outcome_set.candidates
        )
        if len(set(observed_ids)) != len(observed_ids) or set(origins) != set(observed_ids):
            raise OutcomePartitionError("candidate_origin_authority_invalid")
        return origins

    async def _candidate_outcomes_are_current(self, session_id: str) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        path = meeting_dir / "topic_outcome_candidates.json"
        topic_path = meeting_dir / "topic_segments.json"
        if not path.is_file() or not topic_path.is_file():
            return False
        try:
            serialized = path.read_text(encoding="utf-8")
            payload = json.loads(serialized)
            if (
                not isinstance(payload, dict)
                or set(payload)
                != {
                    "schema_version",
                    "provenance",
                    "topics",
                }
                or payload.get("schema_version") != 1
            ):
                return False
            provenance = payload.get("provenance")
            expected_keys = {
                "status",
                "provider",
                "model",
                "approved_model_digest",
                "endpoint_policy",
                "remote_allow_authorizes_outcome_calls",
                "policy_hash",
                "provider_interface",
                "outcome_pipeline_revision",
                "canonical_input_hash",
                "topic_artifact_sha256",
                "event_facts_hash",
                "allowed_participants_hash",
                "outcome_config_version",
                "outcome_disposition_revision",
                "partition_manifest_hash",
                "outcome_artifact_hash",
            }
            if not isinstance(provenance, dict) or set(provenance) != expected_keys:
                return False
            stored_policy = stored_outcome_policy_provenance(provenance)
            if (
                provenance.get("outcome_pipeline_revision") != _OUTCOME_PIPELINE_REVISION
                or provenance.get("outcome_config_version") != OUTCOME_EXTRACTION_CONFIG_VERSION
                or provenance.get("outcome_disposition_revision") != OUTCOME_DISPOSITION_REVISION
                or not isinstance(provenance.get("partition_manifest_hash"), str)
                or provenance.get("topic_artifact_sha256") != _sha256(topic_path)
                or provenance.get("canonical_input_hash")
                != self._read_canonical_transcript(session_id).serialized_hash
                or provenance.get("outcome_artifact_hash")
                != _payload_hash(outcome_hash_input(payload))
            ):
                return False
            topics = self._read_topic_segments(session_id)
            manifest = OutcomePartitionManifest.model_validate_json(
                (meeting_dir / "outcome_partition_manifest.json").read_text(encoding="utf-8")
            )
            validate_outcome_partition_manifest(manifest)
            if provenance.get("partition_manifest_hash") != manifest.manifest_hash:
                return False
            outcome_sets = decode_outcome_sets(payload.get("topics"))
            if tuple(item.topic_id for item in outcome_sets) != tuple(item.id for item in topics):
                return False
            _, _, labels = await self._speaker_context(session_id)
            if provenance.get("allowed_participants_hash") != _payload_hash(sorted(labels)):
                return False
            events_by_topic = await self._outcome_events_by_topic(
                session_id,
                topics,
                await self._events.list(session_id),
            )
            if provenance.get("event_facts_hash") != _payload_hash(
                {
                    topic_id: [event.model_dump(mode="json") for event in events]
                    for topic_id, events in events_by_topic.items()
                }
            ):
                return False
            canonical = self._read_canonical_transcript(session_id)
            canonical_by_id = {entry.entry_id: entry for entry in canonical.entries}
            topics_input = tuple(
                (
                    topic,
                    tuple(canonical_by_id[entry_id] for entry_id in topic.entry_ids),
                    events_by_topic.get(topic.id, ()),
                )
                for topic in topics
            )
            validate_outcome_partition_manifest(
                manifest,
                canonical_hash=canonical.serialized_hash,
                topic_artifact_hash=_sha256(topic_path),
                event_facts_hash=str(provenance["event_facts_hash"]),
                participant_hash=str(provenance["allowed_participants_hash"]),
                topics=topics_input,
                pipeline_revision=_OUTCOME_PIPELINE_REVISION,
                allowed_participant_ids=tuple(sorted(labels)),
            )
            topic_payload = json.loads(topic_path.read_text(encoding="utf-8"))
            boundary_candidates = (
                topic_payload.get("candidates") if isinstance(topic_payload, dict) else None
            )
            topic_provenance = (
                topic_payload.get("provenance") if isinstance(topic_payload, dict) else None
            )
            if not isinstance(boundary_candidates, list) or not isinstance(topic_provenance, dict):
                return False
            job_hash = llm_job_hash(
                canonical_hash=canonical.serialized_hash,
                pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            )
            boundary_tasks = tuple(
                boundary_task_hash(job_hash, str(candidate.get("entry_id")))
                for candidate in boundary_candidates
                if isinstance(candidate, dict) and isinstance(candidate.get("entry_id"), str)
            )
            if len(boundary_tasks) != len(boundary_candidates):
                return False
            provider_ready = bool(
                stored_policy["status"] == "approved"
                and stored_policy["provider"] == APPROVED_TOPIC_PROVIDER
                and stored_policy["model"] == APPROVED_TOPIC_MODEL
                and provenance.get("provider_interface") == "single_kind"
            )
            outcome_enabled = provider_ready
            outcome_tasks = tuple(
                outcome_task_hash(job_hash, partition, kind)
                for partition in manifest.partitions
                for kind in OutcomeRequestKind
            )
            prior_tasks = (
                (digest_task_hash(job_hash), *boundary_tasks)
                if topic_provenance.get("status") == "approved" and boundary_tasks
                else ()
            )
            expected_tasks = (
                (digest_task_hash(job_hash), *boundary_tasks, *outcome_tasks)
                if outcome_enabled
                else prior_tasks
            )
            journal = read_budget_journal(
                meeting_dir / "outcome_budget_journal.json",
                job_hash=job_hash,
                expected_task_hashes=expected_tasks,
            )
            results = read_partition_results(
                meeting_dir / "outcome_partition_results.json",
                manifest,
                pipeline_revision=_OUTCOME_PIPELINE_REVISION,
                journal=journal,
            )
            validate_partition_result_keys(
                tuple(results),
                expected_partition_result_keys(manifest) if outcome_enabled else (),
            )
            for (partition_id, kind), result in results.items():
                partition = next(
                    item for item in manifest.partitions if item.partition_id == partition_id
                )
                task_hash = outcome_task_hash(job_hash, partition, kind)
                if task_hash not in journal.result_task_hashes:
                    return False
                result.validate_for(partition, kind, _OUTCOME_PIPELINE_REVISION)
            journal.validate_expected(expected_tasks)
            journal.validate_final()
            summarize_outcome_dispositions(results, journal, manifest)
            expected_provenance = {
                **stored_policy,
                "provider_interface": "single_kind" if provider_ready else "unsupported",
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "canonical_input_hash": canonical.serialized_hash,
                "topic_artifact_sha256": _sha256(topic_path),
                "event_facts_hash": _payload_hash(
                    {
                        topic_id: [event.model_dump(mode="json") for event in events]
                        for topic_id, events in events_by_topic.items()
                    }
                ),
                "allowed_participants_hash": _payload_hash(sorted(labels)),
                "outcome_config_version": OUTCOME_EXTRACTION_CONFIG_VERSION,
                "outcome_disposition_revision": OUTCOME_DISPOSITION_REVISION,
                "partition_manifest_hash": manifest.manifest_hash,
            }
            expected_payload, expected_sets = self._candidate_outcomes_payload(
                topics=topics,
                manifest=manifest,
                terminal_results=results,
                expected_provenance=expected_provenance,
                policy_failure=str(stored_policy["status"]),
                provider_ready=provider_ready,
            )
            return bool(
                payload == expected_payload
                and serialized == _json_text(expected_payload)
                and outcome_sets == expected_sets
            )
        except OutcomePartitionError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    async def _current_candidate_outcomes(
        self, session_id: str
    ) -> tuple[TopicOutcomeSet, ...] | None:
        if not await self._candidate_outcomes_are_current(session_id):
            return None
        try:
            return self._read_outcome_sets(
                self._meeting_dir(session_id) / "topic_outcome_candidates.json"
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    async def _linked_outcomes_are_current(self, session_id: str) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        path = meeting_dir / "linked_outcomes.json"
        candidate_path = meeting_dir / "topic_outcome_candidates.json"
        if not path.is_file() or not candidate_path.is_file():
            return False
        try:
            candidate_sets = await self._current_candidate_outcomes(session_id)
            if candidate_sets is None:
                return False
            serialized = path.read_text(encoding="utf-8")
            payload = json.loads(serialized)
            if (
                not isinstance(payload, dict)
                or set(payload)
                != {
                    "schema_version",
                    "provenance",
                    "topics",
                }
                or payload.get("schema_version") != 1
            ):
                return False
            provenance = payload.get("provenance")
            if not isinstance(provenance, dict) or set(provenance) != {
                "candidate_artifact_sha256",
                "outcome_pipeline_revision",
                "speech_act_policy_revision",
                "linked_artifact_hash",
            }:
                return False
            expected_payload, _, _ = await self._linked_outcomes_payload(
                session_id,
                outcome_sets=candidate_sets,
            )
            return bool(
                provenance.get("candidate_artifact_sha256") == _sha256(candidate_path)
                and provenance.get("outcome_pipeline_revision") == _OUTCOME_PIPELINE_REVISION
                and provenance.get("speech_act_policy_revision") == _SPEECH_ACT_POLICY_REVISION
                and payload == expected_payload
                and serialized == _json_text(expected_payload)
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    async def _current_validated_outcomes(
        self,
        session_id: str,
    ) -> tuple[TopicOutcomeSet, ...] | None:
        meeting_dir = self._meeting_dir(session_id)
        path = meeting_dir / "validated_outcomes.json"
        linked_path = meeting_dir / "linked_outcomes.json"
        candidate_path = meeting_dir / "topic_outcome_candidates.json"
        if not path.is_file() or not linked_path.is_file() or not candidate_path.is_file():
            return None
        try:
            candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            linked_payload = json.loads(linked_path.read_text(encoding="utf-8"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(candidate_payload, dict)
                or candidate_payload.get("schema_version") != 1
                or not isinstance(linked_payload, dict)
                or linked_payload.get("schema_version") != 1
                or not isinstance(payload, dict)
                or payload.get("schema_version") != 3
                or set(payload)
                != {
                    "schema_version",
                    "provenance",
                    "topics",
                    "validation_metrics",
                    "binding_resolutions",
                    "individual_rejections",
                }
            ):
                return None
            candidate_provenance = candidate_payload.get("provenance")
            linked_provenance = linked_payload.get("provenance")
            provenance = payload.get("provenance")
            if not all(
                isinstance(item, dict)
                for item in (candidate_provenance, linked_provenance, provenance)
            ):
                return None
            assert isinstance(candidate_provenance, dict)
            assert isinstance(linked_provenance, dict)
            assert isinstance(provenance, dict)
            if any(
                item.get("outcome_pipeline_revision") != _OUTCOME_PIPELINE_REVISION
                for item in (candidate_provenance, linked_provenance, provenance)
            ):
                return None
            if (
                linked_provenance.get("speech_act_policy_revision")
                not in {
                    _SPEECH_ACT_POLICY_REVISION,
                    _PREVIOUS_SPEECH_ACT_POLICY_REVISION,
                }
                or provenance.get("speech_act_policy_revision") != _SPEECH_ACT_POLICY_REVISION
            ):
                return None
            if set(linked_provenance) != {
                "candidate_artifact_sha256",
                "outcome_pipeline_revision",
                "speech_act_policy_revision",
                "linked_artifact_hash",
            } or set(provenance) != {
                "linked_artifact_sha256",
                "outcome_pipeline_revision",
                "speech_act_policy_revision",
                "binding_resolution_revision",
                "validated_artifact_hash",
            }:
                return None
            if candidate_provenance.get("outcome_artifact_hash") != _payload_hash(
                outcome_hash_input(candidate_payload)
            ):
                return None
            if linked_provenance.get("candidate_artifact_sha256") != _sha256(candidate_path):
                return None
            if linked_provenance.get("linked_artifact_hash") != _payload_hash(
                outcome_hash_input(linked_payload)
            ):
                return None
            if provenance.get("linked_artifact_sha256") != _sha256(linked_path):
                return None
            if provenance.get("binding_resolution_revision") != _VALIDATED_BINDING_REVISION:
                return None
            if provenance.get("validated_artifact_hash") != _payload_hash(
                validated_outcome_hash_input(payload)
            ):
                return None
            candidate_sets = await self._current_candidate_outcomes(session_id)
            if candidate_sets is None:
                return None
            expected_linked_payload, _, linked_sets = await self._linked_outcomes_payload(
                session_id,
                outcome_sets=candidate_sets,
            )
            if linked_payload != expected_linked_payload or linked_path.read_bytes() != _json_text(
                expected_linked_payload
            ).encode("utf-8"):
                return None
            expected_payload, _, expected_sets = await self._validated_outcomes_payload(
                session_id,
                outcome_sets=linked_sets,
            )
            if payload != expected_payload or path.read_bytes() != _json_text(
                expected_payload
            ).encode("utf-8"):
                return None
            return expected_sets
        except OutcomePartitionError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _outcome_generation_metadata(
        self,
        session_id: str,
        outcome_sets: tuple[TopicOutcomeSet, ...],
    ) -> tuple[SummaryMethod, str | None, str | None, str, str, tuple[str, ...]]:
        """Derive publication metadata only from the validated artifact chain."""

        payload = json.loads(
            (self._meeting_dir(session_id) / "topic_outcome_candidates.json").read_text(
                encoding="utf-8"
            )
        )
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("outcome generation provenance is malformed")
        policy = {
            "status": provenance.get("status"),
            "provider": provenance.get("provider"),
            "model": provenance.get("model"),
            "approved_model_digest": provenance.get("approved_model_digest"),
            "endpoint_policy": provenance.get("endpoint_policy"),
            "remote_allow_authorizes_outcome_calls": provenance.get(
                "remote_allow_authorizes_outcome_calls"
            ),
        }
        policy_hash = provenance.get("policy_hash")
        if (
            not isinstance(policy_hash, str)
            or policy_hash != _payload_hash(policy)
            or policy["approved_model_digest"] != APPROVED_TOPIC_MODEL_DIGEST
            or policy["endpoint_policy"] != "official_remote_https_only"
            or policy["remote_allow_authorizes_outcome_calls"] is not True
        ):
            raise ValueError("outcome generation policy provenance is invalid")
        status = policy["status"]
        methods = tuple(outcome.extraction_method for outcome in outcome_sets)
        failure_reasons = tuple(
            sorted(
                {
                    *(failure for outcome in outcome_sets for failure in outcome.failure_types),
                    *(
                        outcome_failure_token(failure)
                        for outcome in outcome_sets
                        for failure in outcome.provider_failures
                    ),
                }
            )
        )
        if status == "provider_disabled" and (
            not methods or all(method == "deterministic_fallback" for method in methods)
        ):
            method: SummaryMethod = "deterministic"
        elif not methods or all(method == "deterministic_fallback" for method in methods):
            method = "deterministic_fallback"
        elif failure_reasons or any(item != "deepseek" for item in methods):
            method = "deepseek_partial"
        else:
            method = "deepseek"
        provider = policy["provider"]
        model = policy["model"]
        if method in {"deepseek", "deepseek_partial"} and (
            status != "approved"
            or not isinstance(provider, str)
            or provider != APPROVED_TOPIC_PROVIDER
            or not isinstance(model, str)
            or model != APPROVED_TOPIC_MODEL
        ):
            raise ValueError("DeepSeek outcome generation provenance is invalid")
        return (
            method,
            provider if isinstance(provider, str) else None,
            model if isinstance(model, str) else None,
            APPROVED_TOPIC_MODEL_DIGEST,
            policy_hash,
            failure_reasons,
        )

    async def _validated_outcomes_are_current(self, session_id: str) -> bool:
        return await self._current_validated_outcomes(session_id) is not None

    @staticmethod
    def _minutes_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("minutes provenance is malformed")
        return {
            **payload,
            "provenance": {
                key: value for key, value in provenance.items() if key != "minutes_payload_hash"
            },
        }

    @staticmethod
    def _evidence_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("evidence provenance is malformed")
        return {
            **payload,
            "provenance": {
                key: value for key, value in provenance.items() if key != "evidence_payload_hash"
            },
        }

    async def _evidence_items(self, session_id: str) -> tuple[list[dict[str, Any]], int]:
        events = await self._events.list(session_id)
        evidence: list[dict[str, Any]] = [
            {
                "event_sequence": event.sequence_number,
                "event_type": event.event_type,
                "actor_user_id": event.actor_user_id,
                "payload": self._public_event_payload(event.payload),
            }
            for event in events
            if event.event_type
            in {
                "VOTE_CLOSED",
                "AGENDA_ITEM_COMPLETED",
                "SPEAKER_FINISHED",
                "MEETING_STOPPED",
            }
        ]
        outcome_evidence = [
            {
                "evidence_kind": "validated_outcome",
                "topic_id": outcome_set.topic_id,
                "outcome_id": candidate.id,
                "outcome_type": candidate.type.value,
                "verification_status": candidate.verification_status.value,
                "source_entry_ids": list(candidate.source_entry_ids),
                "source_event_ids": list(candidate.source_event_ids),
            }
            for outcome_set in await self._validated_outcome_sets(session_id)
            for candidate in outcome_set.candidates
        ]
        evidence.extend(outcome_evidence)
        return evidence, len(outcome_evidence)

    async def _evidence_is_current(
        self,
        session_id: str,
        *,
        require_registered: bool,
    ) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        path = meeting_dir / "evidence.json"
        validated_path = meeting_dir / "validated_outcomes.json"
        if not path.is_file() or not validated_path.is_file():
            return False
        try:
            serialized = path.read_text(encoding="utf-8")
            payload = json.loads(serialized)
            if not isinstance(payload, dict) or set(payload) != {
                "schema_version",
                "provenance",
                "items",
            }:
                return False
            if payload.get("schema_version") != 1:
                return False
            if serialized != _json_text(payload):
                return False
            provenance = payload.get("provenance")
            if not isinstance(provenance, dict) or set(provenance) != {
                "outcome_pipeline_revision",
                "speech_act_policy_revision",
                "validated_outcomes_sha256",
                "evidence_payload_hash",
            }:
                return False
            if provenance.get("outcome_pipeline_revision") != _OUTCOME_PIPELINE_REVISION:
                return False
            if provenance.get("speech_act_policy_revision") != _SPEECH_ACT_POLICY_REVISION:
                return False
            if provenance.get("validated_outcomes_sha256") != _sha256(validated_path):
                return False
            if provenance.get("evidence_payload_hash") != _payload_hash(
                self._evidence_hash_input(payload)
            ):
                return False
            expected_items, _ = await self._evidence_items(session_id)
            if payload.get("items") != expected_items:
                return False
            if require_registered:
                artifact = await self._artifacts.at_path(
                    session_id=session_id,
                    artifact_type="EVIDENCE_JSON",
                    path=path,
                )
                if (
                    artifact is None
                    or artifact.path != path
                    or artifact.sha256 != _sha256(path)
                    or artifact.metadata.get("pipeline_version") != _ARTIFACT_PIPELINE_VERSION
                    or artifact.metadata.get("outcome_pipeline_revision")
                    != _OUTCOME_PIPELINE_REVISION
                    or artifact.metadata.get("speech_act_policy_revision")
                    != _SPEECH_ACT_POLICY_REVISION
                    or artifact.metadata.get("size_bytes") != path.stat().st_size
                ):
                    return False
            return True
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    async def _minutes_are_current(
        self,
        session_id: str,
        *,
        expected_renderer_input_sha256: str | None = None,
    ) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        json_path = meeting_dir / "minutes.json"
        markdown_path = meeting_dir / "minutes.md"
        validated_path = meeting_dir / "validated_outcomes.json"
        if not json_path.is_file() or not markdown_path.is_file() or not validated_path.is_file():
            return False
        try:
            serialized = json_path.read_text(encoding="utf-8")
            document = MinutesDocumentV2.model_validate_json(serialized)
            payload = document.model_dump(mode="json")
            provenance = document.provenance
            if provenance.outcome_pipeline_revision != _OUTCOME_PIPELINE_REVISION:
                return False
            if provenance.speech_act_policy_revision != _SPEECH_ACT_POLICY_REVISION:
                return False
            if provenance.minutes_renderer_revision != MINUTES_RENDERER_REVISION:
                return False
            if (
                expected_renderer_input_sha256 is not None
                and provenance.renderer_input_sha256 != expected_renderer_input_sha256
            ):
                return False
            if provenance.validated_outcomes_sha256 != _sha256(validated_path):
                return False
            if provenance.canonical_transcript_sha256 != _sha256(self._canonical_path(session_id)):
                return False
            if provenance.minutes_markdown_sha256 != _sha256(markdown_path):
                return False
            if provenance.minutes_payload_hash != _payload_hash(self._minutes_hash_input(payload)):
                return False
            expected, canonical, labels, _ = await self._build_minutes_document(
                session_id,
                generated_at=document.generated_at,
            )
            if provenance.renderer_input_sha256 != expected.provenance.renderer_input_sha256:
                return False
            expected_payload = expected.model_dump(mode="json")
            actual_without_output_hashes = document.model_dump(mode="json")
            actual_without_output_hashes["provenance"] = {
                **actual_without_output_hashes["provenance"],
                "minutes_markdown_sha256": None,
                "minutes_payload_hash": None,
            }
            if actual_without_output_hashes != expected_payload:
                return False
            if self._full_transcript_minutes_provider is not None:
                if not await self._full_transcript_minutes_are_current(session_id):
                    return False
            elif markdown_path.read_text(encoding="utf-8") != render_minutes(
                expected,
                canonical,
                labels=labels,
            ):
                return False
            return serialized == _json_text(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

    async def _registered_outputs_are_current(self, session_id: str) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        try:
            document = MinutesDocumentV2.model_validate_json(
                (meeting_dir / "minutes.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError):
            return False
        for name, artifact_type in _REGISTERED_FILE_ARTIFACTS:
            path = meeting_dir / name
            artifact = await self._artifacts.at_path(
                session_id=session_id,
                artifact_type=artifact_type,
                path=path,
            )
            if (
                artifact is None
                or artifact.encrypted
                or artifact.path != path
                or not path.is_file()
                or artifact.sha256 != _sha256(path)
                or artifact.metadata
                != self._registered_artifact_metadata(
                    artifact_type,
                    path.stat().st_size,
                    document=document,
                )
            ):
                return False
        return True

    @staticmethod
    def _registered_artifact_metadata(
        artifact_type: str,
        size_bytes: int,
        *,
        document: MinutesDocumentV2,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "pipeline_version": _ARTIFACT_PIPELINE_VERSION,
            "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
            "size_bytes": size_bytes,
        }
        if artifact_type in _SPEECH_DEPENDENT_ARTIFACTS:
            metadata["speech_act_policy_revision"] = _SPEECH_ACT_POLICY_REVISION
        if artifact_type in {"MINUTES_JSON", "MINUTES_MARKDOWN"}:
            metadata["minutes_renderer_revision"] = MINUTES_RENDERER_REVISION
            metadata["renderer_input_sha256"] = document.provenance.renderer_input_sha256
        return metadata

    async def _linked_outcomes_payload(
        self,
        session_id: str,
        *,
        outcome_sets: tuple[TopicOutcomeSet, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[TopicOutcomeSet, ...]]:
        """Rebuild linked outcomes from authoritative candidate sets and evidence contexts."""

        meeting_dir = self._meeting_dir(session_id)
        source_path = meeting_dir / "topic_outcome_candidates.json"
        outcome_sets = (
            self._read_outcome_sets(source_path) if outcome_sets is None else outcome_sets
        )
        contexts = await self._outcome_evidence_contexts(session_id)
        linked_sets: list[TopicOutcomeSet] = []
        status_counts: Counter[str] = Counter()
        for outcome_set in outcome_sets:
            context = contexts.get(outcome_set.topic_id)
            if context is None:
                raise ValueError("outcome set identifies an unknown topic")
            cache = SpeechActReductionCache(context)
            candidates = tuple(
                link_outcome_evidence(candidate, context, cache)
                for candidate in outcome_set.candidates
            )
            status_counts.update(candidate.verification_status.value for candidate in candidates)
            linked_sets.append(
                TopicOutcomeSet(
                    topic_id=outcome_set.topic_id,
                    candidates=candidates,
                    extraction_method=outcome_set.extraction_method,
                    failure_types=outcome_set.failure_types,
                    provider_failures=outcome_set.provider_failures,
                )
            )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "provenance": {
                "candidate_artifact_sha256": _sha256(source_path),
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "speech_act_policy_revision": _SPEECH_ACT_POLICY_REVISION,
            },
            "topics": [outcome.model_dump(mode="json") for outcome in linked_sets],
        }
        payload["provenance"]["linked_artifact_hash"] = _payload_hash(outcome_hash_input(payload))
        details = {
            "linked_count": sum(len(outcome.candidates) for outcome in linked_sets),
            "verification_status_counts": dict(sorted(status_counts.items())),
        }
        return payload, details, tuple(linked_sets)

    async def _stage_link_evidence(self, session_id: str) -> dict[str, Any]:
        meeting_dir = self._meeting_dir(session_id)
        payload, details, _ = await self._linked_outcomes_payload(session_id)
        atomic_write_text(meeting_dir / "linked_outcomes.json", _json_text(payload))
        return details

    async def _validated_outcomes_payload(
        self,
        session_id: str,
        *,
        outcome_sets: tuple[TopicOutcomeSet, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], tuple[TopicOutcomeSet, ...]]:
        """Rebuild validated authority from linked candidates and current evidence contexts."""

        meeting_dir = self._meeting_dir(session_id)
        source_path = meeting_dir / "linked_outcomes.json"
        outcome_sets = (
            self._read_outcome_sets(source_path) if outcome_sets is None else outcome_sets
        )
        candidate_origins = self._candidate_origin_authority(session_id, outcome_sets)
        contexts = await self._outcome_evidence_contexts(session_id)
        validated_sets: list[TopicOutcomeSet] = []
        resolutions: list[ValidatedOutcomeBindingResolution] = []
        individual_rejections: list[ValidatedOutcomeCandidateRejection] = []
        metrics: list[dict[str, Any]] = []
        all_errors: Counter[str] = Counter()
        for outcome_set in outcome_sets:
            context = contexts.get(outcome_set.topic_id)
            if context is None:
                raise ValueError("linked outcome set identifies an unknown topic")
            valid_candidates: list[OutcomeCandidate] = []
            topic_errors: Counter[str] = Counter()
            cache = SpeechActReductionCache(context)
            for candidate in outcome_set.candidates:
                validation = validate_outcome(candidate, context, cache)
                if validation.valid:
                    valid_candidates.append(candidate)
                    continue
                individual_rejections.append(
                    ValidatedOutcomeCandidateRejection(
                        topic_id=outcome_set.topic_id,
                        candidate_id=candidate.id,
                    )
                )
                topic_errors.update(validation.error_types)
                all_errors.update(validation.error_types)
            resolved = resolve_validated_topic_bindings(
                outcome_set.model_copy(update={"candidates": tuple(valid_candidates)}),
                candidate_origins={
                    candidate.id: candidate_origins[candidate.id] for candidate in valid_candidates
                },
            )
            resolutions.extend(resolved.resolutions)
            if resolved.ambiguous_binding_count:
                topic_errors["ambiguous_outcome_binding"] += resolved.ambiguous_binding_count
                all_errors["ambiguous_outcome_binding"] += resolved.ambiguous_binding_count
            if len(resolved.outcome_set.candidates) > MAX_FINAL_CANDIDATES_PER_TOPIC:
                raise OutcomePartitionError("validated_candidate_capacity")
            validated_sets.append(resolved.outcome_set)
            individual_rejected_count = len(outcome_set.candidates) - len(valid_candidates)
            metrics.append(
                {
                    "topic_id": outcome_set.topic_id,
                    "candidate_count": len(outcome_set.candidates),
                    "valid_count": len(resolved.outcome_set.candidates),
                    "rejected_count": (
                        individual_rejected_count + resolved.ambiguous_candidate_count
                    ),
                    "individual_rejected_count": individual_rejected_count,
                    "ambiguous_binding_count": resolved.ambiguous_binding_count,
                    "ambiguous_rejected_count": resolved.ambiguous_candidate_count,
                    "coalesced_count": resolved.coalesced_candidate_count,
                    "error_types": dict(sorted(topic_errors.items())),
                }
            )
        topic_order = {
            outcome_set.topic_id: ordinal for ordinal, outcome_set in enumerate(outcome_sets)
        }
        individual_rejections.sort(key=lambda item: (topic_order[item.topic_id], item.candidate_id))
        payload: dict[str, Any] = {
            "schema_version": 3,
            "provenance": {
                "linked_artifact_sha256": _sha256(source_path),
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "speech_act_policy_revision": _SPEECH_ACT_POLICY_REVISION,
                "binding_resolution_revision": _VALIDATED_BINDING_REVISION,
            },
            "topics": [outcome.model_dump(mode="json") for outcome in validated_sets],
            "validation_metrics": metrics,
            "binding_resolutions": [item.model_dump(mode="json") for item in resolutions],
            "individual_rejections": [
                item.model_dump(mode="json") for item in individual_rejections
            ],
        }
        payload["provenance"]["validated_artifact_hash"] = _payload_hash(
            validated_outcome_hash_input(payload)
        )
        validate_validated_binding_artifact(
            linked_sets=outcome_sets,
            validated_sets=tuple(validated_sets),
            resolutions=tuple(resolutions),
            individual_rejections=tuple(individual_rejections),
            metrics=tuple(metrics),
            candidate_origins=candidate_origins,
        )
        details = {
            "valid_count": sum(len(outcome.candidates) for outcome in validated_sets),
            "rejected_count": sum(metric["rejected_count"] for metric in metrics),
            "coalesced_count": sum(metric["coalesced_count"] for metric in metrics),
            "ambiguous_outcome_binding_count": sum(
                metric["ambiguous_binding_count"] for metric in metrics
            ),
            "error_types": dict(sorted(all_errors.items())),
        }
        return payload, details, tuple(validated_sets)

    async def _stage_validate_outcomes(self, session_id: str) -> dict[str, Any]:
        meeting_dir = self._meeting_dir(session_id)
        payload, details, _ = await self._validated_outcomes_payload(session_id)
        atomic_write_text(meeting_dir / "validated_outcomes.json", _json_text(payload))
        return details

    def _canonical_records(
        self,
        session_id: str,
        *,
        summary_ready: bool,
        canonical: CanonicalTranscript | None = None,
    ) -> tuple[TranscriptEntryRecord, ...]:
        canonical = canonical or self._read_canonical_transcript(session_id)
        return tuple(
            TranscriptEntryRecord(
                id=entry.entry_id,
                session_id=session_id,
                source_key=(
                    f"BOT_TTS:canonical:{entry.entry_id}"
                    if entry.speaker_user_id is None
                    else f"canonical:{entry.entry_id}"
                ),
                user_id=entry.speaker_user_id,
                raw_text=entry.raw_texts[0] if entry.raw_texts else entry.text,
                normalised_text=entry.text,
                started_ms=entry.started_ms,
                ended_ms=entry.ended_ms,
            )
            for entry in canonical.entries
            if not summary_ready or entry.entry_type not in _SUMMARY_EXCLUDED_ENTRY_TYPES
        )

    @staticmethod
    def _entry_payload(entry: TranscriptEntryRecord) -> dict[str, Any]:
        return {
            "id": entry.id,
            "source_key": entry.source_key,
            "user_id": entry.user_id,
            "started_ms": entry.started_ms,
            "ended_ms": entry.ended_ms,
            "raw_text": entry.raw_text,
            "normalised_text": entry.normalised_text,
            "revision_json": list(entry.revision_json),
        }

    @staticmethod
    def _timestamp(milliseconds: int) -> str:
        hours, remainder = divmod(max(milliseconds, 0), 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1_000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    async def _speaker_context(
        self,
        session_id: str,
    ) -> tuple[str, tuple[str, ...], dict[int, str]]:
        session = await self._sessions.get(session_id)
        timeline = await self._consent_timeline(session_id)
        eligible_ids = {
            interval.user_id
            for interval in timeline.intervals
            if interval.kind is ConsentKind.MINUTES
        }
        participants = await self._participants.list_session(session_id)
        labels: dict[int, str] = {}
        host_label = f"user:{session.creator_user_id}"
        attendee_labels: list[str] = []
        for participant in participants:
            label = participant.display_name or f"user:{participant.user_id}"
            if participant.is_host or participant.user_id == session.creator_user_id:
                host_label = f"{label} (user:{participant.user_id})"
            if participant.user_id not in eligible_ids:
                continue
            labels[participant.user_id] = label
            if not (participant.is_host or participant.user_id == session.creator_user_id):
                attendee_labels.append(f"{label} (user:{participant.user_id})")
        return host_label, tuple(attendee_labels), labels

    @staticmethod
    def _speaker_label(entry: TranscriptEntryRecord, labels: dict[int, str]) -> str:
        if entry.source_key.startswith("BOT_TTS:"):
            return "聆序机器人"
        if entry.user_id is None:
            return "未知发言者"
        return labels.get(entry.user_id, f"user:{entry.user_id}")

    async def _stage_merge_transcript(self, session_id: str) -> dict[str, Any]:
        raw_entries = await self._transcripts.list(session_id)
        entries = self._canonical_records(session_id, summary_ready=True)
        host, attendees, labels = await self._speaker_context(session_id)
        meeting_dir = self._meeting_dir(session_id)
        atomic_write_text(
            meeting_dir / "transcript.json",
            _json_text([self._entry_payload(entry) for entry in raw_entries]),
        )
        transcript_text = self._render_transcript_text(
            session_id=session_id,
            entries=entries,
            host=host,
            attendees=attendees,
            labels=labels,
        )
        atomic_write_text(meeting_dir / "transcript.txt", transcript_text)
        return {
            "raw_entries": len(raw_entries),
            "merged_entries": len(entries),
        }

    def _render_transcript_text(
        self,
        *,
        session_id: str,
        entries: tuple[TranscriptEntryRecord, ...],
        host: str,
        attendees: tuple[str, ...],
        labels: dict[int, str],
    ) -> str:
        lines = [
            "会议逐字稿",
            f"会议 ID: {session_id}",
            f"主持人: {host}",
            f"参会者: {', '.join(attendees) if attendees else '无其他已授权参会者'}",
            "",
        ]
        lines.extend(
            (
                f"[{self._timestamp(entry.started_ms)}-{self._timestamp(entry.ended_ms)}] "
                f"{self._speaker_label(entry, labels)}: "
                f"{entry.normalised_text or entry.raw_text}"
            )
            for entry in entries
        )
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _agenda_groups(
        agenda: tuple[AgendaItemRecord, ...],
        entries: tuple[TranscriptEntryRecord, ...],
    ) -> dict[str, tuple[TranscriptEntryRecord, ...]]:
        """Group entries by the Bot's durable spoken agenda-title markers."""

        markers: list[tuple[str, int]] = []
        search_from = 0
        for item in agenda:
            marker = next(
                (
                    index
                    for index in range(search_from, len(entries))
                    if entries[index].source_key.startswith("BOT_TTS:")
                    and item.title in (entries[index].normalised_text or entries[index].raw_text)
                ),
                None,
            )
            if marker is not None:
                markers.append((item.id, marker))
                search_from = marker + 1
        grouped: dict[str, tuple[TranscriptEntryRecord, ...]] = {}
        for index, (item_id, start) in enumerate(markers):
            end = markers[index + 1][1] if index + 1 < len(markers) else len(entries)
            grouped[item_id] = entries[start:end]
        return grouped

    async def _validated_outcome_sets(self, session_id: str) -> tuple[TopicOutcomeSet, ...]:
        outcomes = await self._current_validated_outcomes(session_id)
        if outcomes is None:
            raise ValueError("validated outcome artifact provenance is invalid")
        return outcomes

    @staticmethod
    def _privacy_candidate_hash(kind: str, identifier: str) -> str:
        return _payload_hash({"kind": kind, "identifier": identifier})

    @staticmethod
    def _privacy_journal_hash_input(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "journal_hash"}

    @classmethod
    def _privacy_journal_payload(
        cls,
        *,
        state: str,
        candidate_hashes: dict[str, list[str]],
        measurements: dict[str, int] | None,
    ) -> dict[str, Any]:
        counts = {key: len(value) for key, value in candidate_hashes.items()}
        payload: dict[str, Any] = {
            "schema_version": 2,
            "state": state,
            "candidate_hashes": candidate_hashes,
            "candidate_counts": counts,
            "measurements": measurements,
        }
        payload["journal_hash"] = _payload_hash(cls._privacy_journal_hash_input(payload))
        return payload

    def _read_privacy_journal(
        self,
        session_id: str,
        *,
        allow_legacy: bool,
    ) -> dict[str, Any] | None:
        path = self._meeting_dir(session_id) / "privacy_exclusions.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("privacy exclusion journal is malformed") from exc
        if allow_legacy and isinstance(payload, dict) and payload.get("schema_version") == 1:
            expected = {
                "schema_version",
                "measurement_available",
                "removed_transcript_entries",
                "removed_segment_rows",
                "removed_segment_files",
            }
            if set(payload) != expected or payload["measurement_available"] is not True:
                raise ValueError("privacy exclusion journal is malformed")
            values = [
                payload["removed_transcript_entries"],
                payload["removed_segment_rows"],
                payload["removed_segment_files"],
            ]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise ValueError("privacy exclusion journal is malformed")
            hashes: dict[str, list[str]] = {
                "transcript_entries": [],
                "segment_rows": [],
                "segment_files": [],
            }
            legacy = self._privacy_journal_payload(
                state="complete",
                candidate_hashes=hashes,
                measurements={
                    "transcript_entries": values[0],
                    "segment_rows": values[1],
                    "segment_files": values[2],
                },
            )
            legacy["candidate_counts"] = legacy["measurements"]
            legacy["journal_hash"] = _payload_hash(self._privacy_journal_hash_input(legacy))
            return legacy
        keys = {
            "schema_version",
            "state",
            "candidate_hashes",
            "candidate_counts",
            "measurements",
            "journal_hash",
        }
        if not isinstance(payload, dict) or set(payload) != keys or payload["schema_version"] != 2:
            raise ValueError("privacy exclusion journal is malformed")
        if payload["state"] not in {"prepared", "complete", "legacy_unavailable"}:
            raise ValueError("privacy exclusion journal state is invalid")
        names = ("transcript_entries", "segment_rows", "segment_files")
        hashes = payload["candidate_hashes"]
        counts = payload["candidate_counts"]
        if not isinstance(hashes, dict) or set(hashes) != set(names):
            raise ValueError("privacy exclusion journal candidates are malformed")
        if not isinstance(counts, dict) or set(counts) != set(names):
            raise ValueError("privacy exclusion journal counts are malformed")
        for name in names:
            values = hashes[name]
            if (
                not isinstance(values, list)
                or any(
                    not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in values
                )
                or len(values) != len(set(values))
                or counts[name] != len(values)
            ):
                raise ValueError("privacy exclusion journal candidates are malformed")
        measurements = payload["measurements"]
        if payload["state"] == "prepared" and measurements is not None:
            raise ValueError("prepared privacy exclusion journal has measurements")
        if payload["state"] == "legacy_unavailable" and (
            measurements is not None or any(counts.values())
        ):
            raise ValueError("legacy privacy exclusion journal is inconsistent")
        if payload["state"] == "complete":
            if not isinstance(measurements, dict) or measurements != counts:
                raise ValueError("complete privacy exclusion journal is inconsistent")
        if payload["journal_hash"] != _payload_hash(self._privacy_journal_hash_input(payload)):
            raise ValueError("privacy exclusion journal hash mismatch")
        return payload

    def _privacy_exclusions(self, session_id: str) -> PrivacyExclusionCounts:
        """Read durable counts; only genuinely old jobs may report unavailable."""

        journal = self._read_privacy_journal(
            session_id,
            allow_legacy=self._privacy_legacy_schema_allowed.get(session_id, False),
        )
        if journal is None:
            if not self._privacy_legacy_unavailable_allowed.get(session_id, False):
                raise ValueError("privacy exclusion journal is missing")
            unavailable = PrivacyExclusionCounts(
                transcript_entries=0,
                segment_rows=0,
                segment_files=0,
                measurement_available=False,
            )
            payload = self._privacy_journal_payload(
                state="legacy_unavailable",
                candidate_hashes={
                    "transcript_entries": [],
                    "segment_rows": [],
                    "segment_files": [],
                },
                measurements=None,
            )
            atomic_write_text(
                self._meeting_dir(session_id) / "privacy_exclusions.json",
                _json_text(payload),
            )
            return unavailable
        if journal["state"] == "legacy_unavailable":
            if not self._privacy_legacy_unavailable_allowed.get(session_id, False):
                raise ValueError("legacy privacy exclusion state lacks checkpoint authority")
            return PrivacyExclusionCounts(
                transcript_entries=0,
                segment_rows=0,
                segment_files=0,
                measurement_available=False,
            )
        if journal["state"] != "complete":
            raise ValueError("privacy exclusion journal is incomplete")
        measurements = journal["measurements"]
        assert isinstance(measurements, dict)
        return PrivacyExclusionCounts(
            transcript_entries=measurements["transcript_entries"],
            segment_rows=measurements["segment_rows"],
            segment_files=measurements["segment_files"],
            measurement_available=True,
        )

    @staticmethod
    def _deterministic_summary_v1_payload() -> dict[str, Any]:
        return {
            "executive_summary": "已验证议题结果见 topic_outcomes; 结构化会议摘要尚未生成。",
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "risks": [],
            "disagreements": [],
            "host_overrides": [],
            "source_entry_ids": [],
            "source_event_ids": [],
        }

    @staticmethod
    def _render_topic_outcomes(
        lines: list[str],
        outcome_sets: tuple[TopicOutcomeSet, ...],
        topics: tuple[TopicSegment, ...],
    ) -> None:
        lines.extend(["## 议题结果", ""])
        topic_titles = {topic.id: topic.title for topic in topics}
        headings = (
            (OutcomeType.DECISION, "决议"),
            (OutcomeType.AGREEMENT, "一致意见"),
            (OutcomeType.ACTION_ITEM, "行动项"),
            (OutcomeType.OPEN_QUESTION, "未决问题"),
            (OutcomeType.PROPOSAL, "提议"),
            (OutcomeType.RISK, "风险"),
            (OutcomeType.DISAGREEMENT, "分歧"),
        )
        rendered = 0
        for outcome_set in outcome_sets:
            if not outcome_set.candidates:
                continue
            rendered += len(outcome_set.candidates)
            lines.extend(
                [
                    f"### {topic_titles.get(outcome_set.topic_id, outcome_set.topic_id)}",
                    "",
                ]
            )
            for outcome_type, heading in headings:
                candidates = tuple(
                    candidate
                    for candidate in outcome_set.candidates
                    if candidate.type is outcome_type
                )
                if not candidates:
                    continue
                lines.extend([f"#### {heading}", ""])
                for candidate in candidates:
                    owner = (
                        f"{candidate.owner_name} (user:{candidate.owner_user_id})"
                        if candidate.owner_user_id is not None
                        else "未明确"
                    )
                    deadline = candidate.deadline or "未明确"
                    status = {
                        VerificationStatus.EVIDENCE_LINKED: "已链接证据",
                        VerificationStatus.UNCONFIRMED: "待确认",
                        VerificationStatus.NOT_CHECKED: "发言者/第三方主张未核验",
                        VerificationStatus.UNVERIFIED: "未验证",
                    }[candidate.verification_status]
                    sources = [f"entry:{entry_id}" for entry_id in candidate.source_entry_ids]
                    sources.extend(f"event:{event_id}" for event_id in candidate.source_event_ids)
                    lines.append(
                        f"- {candidate.text} (负责人: {owner}; 截止: {deadline}; "
                        f"状态: {status}; 来源: {', '.join(sources)})"
                    )
                lines.append("")
        if rendered == 0:
            lines.extend(["- 未提取到通过证据验证的议题结果。", ""])

    async def _stage_merge_meeting_events(self, session_id: str) -> dict[str, Any]:
        events = await self._events.list(session_id)
        payload = self._meeting_events_payload(events)
        atomic_write_text(self._meeting_dir(session_id) / "events.json", _json_text(payload))
        return {"merged_events": len(events)}

    async def _meeting_events_are_current(self, session_id: str) -> bool:
        """Bind the exported event view to the append-only event-log source of truth."""

        path = self._meeting_dir(session_id) / "events.json"
        try:
            if path.is_symlink() or not path.is_file():
                return False
            expected = _json_text(self._meeting_events_payload(await self._events.list(session_id)))
            return path.read_text(encoding="utf-8") == expected
        except (OSError, UnicodeError):
            return False

    def _meeting_events_payload(
        self, events: tuple[MeetingEventRecord, ...]
    ) -> list[dict[str, Any]]:
        return [
            {
                "sequence_number": event.sequence_number,
                "event_type": event.event_type,
                "actor_user_id": event.actor_user_id,
                "source": event.source,
                "payload": self._public_event_payload(event.payload),
                "state_before": event.state_before.value,
                "state_after": event.state_after.value,
                "occurred_at": event.occurred_at.isoformat(),
            }
            for event in events
        ]

    @staticmethod
    def _public_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Drop unknown event fields at the public-artifact boundary."""

        return {key: payload[key] for key in sorted(payload) if key in _PUBLIC_EVENT_PAYLOAD_KEYS}

    async def _stage_calculate_language_distribution(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        counts = self._language_counts(self._canonical_records(session_id, summary_ready=True))
        atomic_write_text(
            self._meeting_dir(session_id) / "languages.json",
            _json_text(dict(sorted(counts.items()))),
        )
        return {"language_counts": dict(counts)}

    @staticmethod
    def _language_counts(entries: tuple[TranscriptEntryRecord, ...]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for entry in entries:
            text = entry.normalised_text or entry.raw_text
            if re.search(r"[\u3400-\u9fff]", text):
                counts["zh"] += 1
            if re.search(r"[A-Za-z]", text):
                counts["en"] += 1
            if not re.search(r"[\u3400-\u9fffA-Za-z]", text):
                counts["other"] += 1
        return counts

    async def _stage_generate_evidence(self, session_id: str) -> dict[str, Any]:
        evidence, outcome_evidence_count = await self._evidence_items(session_id)
        payload = self._evidence_payload(session_id, evidence)
        atomic_write_text(
            self._meeting_dir(session_id) / "evidence.json",
            _json_text(payload),
        )
        return {
            "evidence_items": len(evidence),
            "validated_outcome_evidence_items": outcome_evidence_count,
        }

    def _evidence_payload(self, session_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "provenance": {
                "outcome_pipeline_revision": _OUTCOME_PIPELINE_REVISION,
                "speech_act_policy_revision": _SPEECH_ACT_POLICY_REVISION,
                "validated_outcomes_sha256": _sha256(
                    self._meeting_dir(session_id) / "validated_outcomes.json"
                ),
            },
            "items": evidence,
        }
        payload["provenance"]["evidence_payload_hash"] = _payload_hash(
            self._evidence_hash_input(payload)
        )
        return payload

    @staticmethod
    def _bounded_fact_summary(
        prefix: str,
        facts: tuple[str, ...],
        *,
        empty: str,
        limit: int = 2_000,
    ) -> str:
        if not facts:
            return empty
        selected: list[str] = []
        for fact in facts:
            cleaned = " ".join(fact.split())
            remaining = len(facts) - len(selected) - 1
            suffix = f"; 另有 {remaining} 项。" if remaining else "。"
            candidate = f"{prefix}{'; '.join((*selected, cleaned))}{suffix}"
            if len(candidate) > limit:
                break
            selected.append(cleaned)
        if not selected:
            return f"{prefix}事实内容超过公开摘要长度限制; 详见结构化结果。"
        remaining = len(facts) - len(selected)
        suffix = f"; 另有 {remaining} 项。" if remaining else "。"
        return f"{prefix}{'; '.join(selected)}{suffix}"

    async def _build_minutes_document(
        self,
        session_id: str,
        *,
        generated_at: datetime,
        canonical_override: CanonicalTranscript | None = None,
    ) -> tuple[
        MinutesDocumentV2,
        CanonicalTranscript,
        dict[int, str],
        dict[str, int | str],
    ]:
        canonical = canonical_override or self._read_canonical_transcript(session_id)
        entries = self._canonical_records(
            session_id,
            summary_ready=True,
            canonical=canonical,
        )
        events = await self._events.list(session_id)
        agenda = await self._agenda.list(session_id)
        host, attendees, labels = await self._speaker_context(session_id)
        speaking_plans = project_authoritative_speaking_plans(agenda, labels=labels)
        mute_enforcement_summaries = project_authoritative_mute_summaries(agenda, events)
        speaking_facts = project_authoritative_speaking_facts(events, labels=labels)
        agenda_ids = {item.id for item in agenda}
        unknown_speaking_agendas = {
            value.agenda_item_id
            for value in speaking_facts.turns
            if value.agenda_item_id not in agenda_ids
        }
        unknown_speaking_agendas.update(
            value.agenda_item_id
            for value in speaking_facts.followups
            if value.agenda_item_id not in agenda_ids
        )
        if unknown_speaking_agendas:
            raise ValueError("authoritative speaking facts reference an unknown agenda")
        agenda_groups = self._agenda_groups(agenda, entries)
        votes = [event for event in events if event.event_type == "VOTE_CLOSED"]
        topic_outcomes = await self._validated_outcome_sets(session_id)
        topics = self._read_topic_segments(session_id)
        merged_outcomes = merge_topic_outcomes(topic_outcomes)
        failure_count = sum(
            len(outcome.failure_types) + sum(failure.count for failure in outcome.provider_failures)
            for outcome in topic_outcomes
        )
        (
            summary_method,
            summary_provider,
            summary_model,
            summary_model_digest,
            summary_policy_identity,
            fallback_reasons,
        ) = self._outcome_generation_metadata(session_id, topic_outcomes)

        participant_entry_count = sum(entry.user_id is not None for entry in entries)
        bot_entry_count = sum(entry.source_key.startswith("BOT_TTS:") for entry in entries)
        meeting_dir = self._meeting_dir(session_id)
        topic_entry_ids = {topic.id: frozenset(topic.entry_ids) for topic in topics}
        agenda_topic_ids = {
            item.id: tuple(
                topic.id
                for topic in topics
                if topic_entry_ids[topic.id]
                & frozenset(entry.id for entry in agenda_groups.get(item.id, ()))
            )
            for item in agenda
        }

        minute_votes: list[MinutesVote] = []
        for event in votes:
            minute_votes.append(normalise_vote_payload(event.sequence_number, event.payload))

        topic_outcomes_by_id = {outcome.topic_id: outcome for outcome in topic_outcomes}
        topic_summaries = {
            topic.id: self._bounded_fact_summary(
                "已记录结果: ",
                tuple(
                    f"[{candidate.verification_status.value}] "
                    f"{candidate.type.value}: {candidate.text}"
                    for candidate in topic_outcomes_by_id.get(
                        topic.id,
                        TopicOutcomeSet(
                            topic_id=topic.id,
                            candidates=(),
                            extraction_method="deterministic_fallback",
                            failure_types=(),
                        ),
                    ).candidates
                ),
                empty="该议题未形成可记录结果。",
            )
            for topic in topics
        }
        agenda_summaries: dict[str, str] = {}
        for item in agenda:
            item_topic_ids = agenda_topic_ids[item.id]
            outcome_count = sum(
                len(topic_outcomes_by_id[topic_id].candidates)
                for topic_id in item_topic_ids
                if topic_id in topic_outcomes_by_id
            )
            item_votes = tuple(vote for vote in minute_votes if vote.agenda_item_id == item.id)
            agenda_summaries[item.id] = (
                f"该议程关联 {len(item_topic_ids)} 个议题、{outcome_count} 个结构化结果和 "
                f"{len(item_votes)} 个已完成投票。"
            )

        evidence_required = (
            *merged_outcomes.decisions,
            *merged_outcomes.actions,
            *merged_outcomes.questions,
            *merged_outcomes.risks,
        )
        summary_facts = tuple(
            f"[{candidate.verification_status.value}] {candidate.type.value}: {candidate.text}"
            for section in (
                merged_outcomes.decisions,
                merged_outcomes.agreements,
                merged_outcomes.actions,
                merged_outcomes.questions,
                merged_outcomes.proposals,
                merged_outcomes.risks,
                merged_outcomes.disagreements,
                merged_outcomes.deferred_items,
            )
            for candidate in section
        )
        vote_facts = tuple(
            f"投票 {vote.vote_content or vote.vote_id or '未命名'}: "
            f"{json.dumps(vote.counts, ensure_ascii=False, sort_keys=True)}; "
            f"结果 {vote.result if vote.result is not None else '未知'}"
            for vote in minute_votes
        )
        executive_summary = (
            f"会议包含 {len(agenda)} 个议程、{len(topics)} 个议题、"
            f"{len(summary_facts)} 个已确认或待确认结果和 {len(minute_votes)} 个已完成投票。"
        )
        full_summary = self._bounded_fact_summary(
            "会议结构化事实: ",
            (*summary_facts, *vote_facts),
            empty="会议未形成已确认/待确认结果或已完成投票。",
        )
        document = MinutesDocumentV2(
            provenance=MinutesProvenance(
                outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
                speech_act_policy_revision=_SPEECH_ACT_POLICY_REVISION,
                minutes_renderer_revision=MINUTES_RENDERER_REVISION,
                renderer_input_sha256="0" * 64,
                validated_outcomes_sha256=_sha256(meeting_dir / "validated_outcomes.json"),
                canonical_transcript_sha256=hashlib.sha256(
                    _json_text(canonical.to_dict()).encode("utf-8")
                ).hexdigest(),
            ),
            session_id=session_id,
            generated_at=generated_at,
            meeting_title=None,
            participants=MinutesParticipants(host=host, attendees=attendees),
            agendas=tuple(
                MinutesAgenda(
                    agenda_id=item.id,
                    position=item.position,
                    title=item.title,
                    language=agenda_language(item),
                    status=item.status.value,
                    duration_minutes=item.duration_minutes,
                    requires_vote=item.requires_vote,
                    vote_content=item.vote_content,
                    vote_duration_seconds=item.vote_duration_seconds,
                    topic_ids=agenda_topic_ids[item.id],
                    summary=agenda_summaries[item.id],
                )
                for item in agenda
            ),
            topics=tuple(
                MinutesTopic(
                    topic_id=topic.id,
                    title=topic.title,
                    started_ms=topic.started_ms,
                    ended_ms=topic.ended_ms,
                    method=topic.method,
                    source_entry_ids=topic.entry_ids,
                    summary=topic_summaries[topic.id],
                )
                for topic in topics
            ),
            validated_outcomes=topic_outcomes,
            decisions=merged_outcomes.decisions,
            agreements=merged_outcomes.agreements,
            actions=merged_outcomes.actions,
            questions=merged_outcomes.questions,
            proposals=merged_outcomes.proposals,
            risks=merged_outcomes.risks,
            disagreements=merged_outcomes.disagreements,
            deferred_items=merged_outcomes.deferred_items,
            speaker_claims=merged_outcomes.speaker_claims,
            votes=tuple(minute_votes),
            speaking_plans=speaking_plans,
            mute_enforcement_summaries=mute_enforcement_summaries,
            speaking_turns=speaking_facts.turns,
            speaking_followups=speaking_facts.followups,
            deterministic_counts=DeterministicCounts(
                agenda_total=len(agenda),
                agenda_completed=sum(item.status.value == "COMPLETED" for item in agenda),
                participant_transcript_entries=participant_entry_count,
                bot_tts_entries=bot_entry_count,
                closed_votes=len(votes),
                topic_total=len(topics),
                validated_outcomes=sum(len(outcome.candidates) for outcome in topic_outcomes),
                agenda_speaking_plans=len(speaking_plans),
                mute_enforcement_summaries=len(mute_enforcement_summaries),
                controlled_speaking_turns=len(speaking_facts.turns),
                controlled_speaking_followups=len(speaking_facts.followups),
            ),
            evidence_coverage=EvidenceCoverage(
                evidence_required=len(evidence_required),
                evidence_linked=sum(
                    bool(
                        candidate.provenance.source_entry_ids
                        or candidate.provenance.source_event_ids
                    )
                    for candidate in evidence_required
                ),
                transcript_sources=len(
                    {
                        source
                        for candidate in evidence_required
                        for source in candidate.provenance.source_entry_ids
                    }
                ),
                event_sources=len(
                    {
                        source
                        for candidate in evidence_required
                        for source in candidate.provenance.source_event_ids
                    }
                ),
            ),
            privacy_exclusions=self._privacy_exclusions(session_id),
            canonical_metrics=CanonicalMetrics(**asdict(canonical.metrics)),
            summary_method=summary_method,
            summary_model=summary_model,
            summary_provider=summary_provider,
            summary_model_digest=summary_model_digest,
            summary_policy_identity=summary_policy_identity,
            fallback_reasons=fallback_reasons,
            executive_summary=executive_summary,
            full_summary=full_summary,
        )

        document_payload = document.model_dump(mode="python")
        document_payload["provenance"] = {
            **document_payload["provenance"],
            "renderer_input_sha256": renderer_input_sha256(
                document,
                canonical,
                labels=labels,
            ),
        }
        document = MinutesDocumentV2.model_validate(document_payload)
        return (
            document,
            canonical,
            labels,
            {
                "minutes_entries": len(entries),
                "agenda_summaries": len(agenda),
                "vote_results": len(votes),
                "summary_method": summary_method,
                "summary_failures": failure_count,
                "validated_outcomes": sum(len(outcome.candidates) for outcome in topic_outcomes),
                "renderer_input_sha256": document.provenance.renderer_input_sha256,
            },
        )

    async def _full_transcript_minutes_request(
        self,
        session_id: str,
        document: MinutesDocumentV2,
    ) -> tuple[
        PreparedFullTranscript,
        PreparedAuthoritativeMeetingFacts,
        str,
        str,
        list[dict[str, str]],
    ]:
        template_path = self._full_transcript_minutes_template
        if template_path is None:
            raise ValueError("full transcript minutes template is unavailable")
        _, _, labels = await self._speaker_context(session_id)
        prepared = prepare_full_transcript_records(
            await self._transcripts.list(session_id),
            speaker_names=labels,
            allow_empty=True,
        )
        authoritative_facts = prepare_authoritative_meeting_facts(
            participant_labels=tuple(f"[{labels[user_id]}]" for user_id in sorted(labels)),
            agendas=document.agendas,
            votes=document.votes,
            speaking_plans=document.speaking_plans,
            mute_enforcement_summaries=document.mute_enforcement_summaries,
            speaking_turns=document.speaking_turns,
            speaking_followups=document.speaking_followups,
        )
        template, template_sha256 = load_preview_template(template_path)
        messages = (
            build_structured_messages(prepared=prepared, authoritative_facts=authoritative_facts)
            if self._structured_final_minutes
            else build_preview_messages(
                prepared=prepared, template=template, authoritative_facts=authoritative_facts
            )
        )
        return prepared, authoritative_facts, template, template_sha256, messages

    def _full_transcript_manifest_base(
        self,
        prepared: PreparedFullTranscript,
        authoritative_facts: PreparedAuthoritativeMeetingFacts,
        *,
        template_sha256: str,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        return {
            **(
                {
                    "structured_minutes_revision": STRUCTURED_MINUTES_REVISION,
                    "output_format": "json_object",
                }
                if self._structured_final_minutes
                else {}
            ),
            "schema_version": PREVIEW_SCHEMA_VERSION,
            "provider": APPROVED_TOPIC_PROVIDER,
            "model": APPROVED_TOPIC_MODEL,
            "model_digest": APPROVED_TOPIC_MODEL_DIGEST,
            "thinking": "high",
            "max_output_tokens": self._full_transcript_minutes_max_output_tokens,
            "transcript_reconciliation_revision": TRANSCRIPT_RECONCILIATION_REVISION,
            "evidence_normalization_revision": EVIDENCE_NORMALIZATION_REVISION,
            "public_minutes_revision": PUBLIC_MINUTES_REVISION,
            "template_sha256": template_sha256,
            "source_sha256": prepared.source_sha256,
            "cleaned_payload_sha256": prepared.cleaned_payload_sha256,
            "authoritative_facts_sha256": authoritative_facts.payload_sha256,
            "authoritative_participant_count": authoritative_facts.participant_count,
            "authoritative_agenda_count": authoritative_facts.agenda_count,
            "authoritative_speaking_plan_count": authoritative_facts.speaking_plan_count,
            "authoritative_mute_enforcement_count": authoritative_facts.mute_enforcement_count,
            "authoritative_vote_count": authoritative_facts.vote_count,
            "authoritative_speaking_row_count": authoritative_facts.speaking_row_count,
            "request_sha256": request_sha256(
                messages,
                max_output_tokens=self._full_transcript_minutes_max_output_tokens,
                output_format="json_object" if self._structured_final_minutes else "text",
            ),
            "row_count": prepared.row_count,
            "raw_text_chars": prepared.raw_text_chars,
            "cleaned_text_chars": prepared.cleaned_text_chars,
            "noise_removed": {
                "japanese": prepared.removal_counts.japanese,
                "korean": prepared.removal_counts.korean,
                "emoji": prepared.removal_counts.emoji,
                "asr_control": prepared.removal_counts.asr_control,
                "total": prepared.removal_counts.total,
            },
        }

    async def _generate_full_transcript_minutes(
        self,
        session_id: str,
        document: MinutesDocumentV2,
    ) -> tuple[str, dict[str, object], dict[str, int | str]]:
        try:
            return await self._generate_model_full_transcript_minutes(session_id, document)
        except FullTranscriptPreviewError as exc:
            # Input, policy and journal corruption must retain their original failure boundary.
            if (
                not (
                    exc.code.startswith(
                        ("final_minutes_", "public_minutes_", "preview_", "model_inventory_")
                    )
                    or exc.code
                    in {
                        "provider_response_too_large",
                        "full_transcript_minutes_request_already_attempted",
                    }
                )
                or re.fullmatch(r"[a-z0-9_]{1,96}", exc.code) is None
            ):
                raise
            (
                prepared,
                facts,
                _,
                template_hash,
                messages,
            ) = await self._full_transcript_minutes_request(session_id, document)
            content, output = render_basic_minutes(prepared, facts)
            manifest = {
                **self._full_transcript_manifest_base(
                    prepared, facts, template_sha256=template_hash, messages=messages
                ),
                **output,
                "fallback_reason": exc.code,
                **(
                    {"validation_details": exc.validation_details.as_dict()}
                    if exc.validation_details is not None
                    else {}
                ),
                "evidence_audit_sha256": _payload_hash(output["evidence_audit"]),
            }
            return (
                content,
                manifest,
                {
                    "final_output_method": "deterministic_basic",
                    "final_output_failure_code": exc.code,
                    "final_output_rows": prepared.row_count,
                    "final_output_bytes": len(content.encode("utf-8")),
                },
            )

    async def _generate_model_full_transcript_minutes(
        self,
        session_id: str,
        document: MinutesDocumentV2,
    ) -> tuple[str, dict[str, object], dict[str, int | str]]:
        provider = self._full_transcript_minutes_provider
        if provider is None:
            raise ValueError("full transcript minutes provider is unavailable")
        (
            prepared,
            authoritative_facts,
            _,
            template_sha256,
            messages,
        ) = await self._full_transcript_minutes_request(session_id, document)
        if not any(label != "[聆序]" for _, label in prepared.source_speaker_labels):
            raise FullTranscriptPreviewError("final_minutes_no_authorized_speech")
        manifest_base = self._full_transcript_manifest_base(
            prepared,
            authoritative_facts,
            template_sha256=template_sha256,
            messages=messages,
        )
        attempt_generation = self._full_transcript_attempt_generations.get(session_id, 0)
        attempt_name = (
            "full_transcript_minutes_attempt.json"
            if attempt_generation == 0
            else f"full_transcript_minutes_attempt.{attempt_generation}.json"
        )
        attempt_path = self._meeting_dir(session_id) / attempt_name
        if attempt_path.is_file():
            try:
                prior_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise FullTranscriptPreviewError(
                    "full_transcript_minutes_attempt_invalid"
                ) from None
            if (
                isinstance(prior_attempt, dict)
                and all(prior_attempt.get(key) == value for key, value in manifest_base.items())
                and prior_attempt.get("attempt_generation", 0) == attempt_generation
            ):
                failure_code = prior_attempt.get("failure_code")
                if (
                    prior_attempt.get("status") == "failed"
                    and isinstance(failure_code, str)
                    and re.fullmatch(r"[a-z0-9_]{1,96}", failure_code)
                ):
                    raise FullTranscriptPreviewError(failure_code)
                raise FullTranscriptPreviewError(
                    "full_transcript_minutes_request_already_attempted"
                )
            # A revision or input change is not authority to repeat this generation's request.
            if not isinstance(prior_attempt, dict) or prior_attempt.get("status") not in {
                "started",
                "failed",
                "complete",
            }:
                raise FullTranscriptPreviewError("full_transcript_minutes_attempt_invalid")
            raise FullTranscriptPreviewError("full_transcript_minutes_request_already_attempted")
        started_attempt = {
            **manifest_base,
            "attempt_generation": attempt_generation,
            "status": "started",
        }
        atomic_write_text(
            attempt_path,
            _json_text(started_attempt),
        )
        structured_metadata: dict[str, object] = {}
        result = None
        try:
            result = await provider.generate(messages)
            if result.status != "complete" or result.finish_reason != "stop":
                raise FullTranscriptPreviewError(
                    "final_minutes_generation_incomplete",
                    diagnostics=result.diagnostics,
                )
            raw_markdown = result.content
            if self._structured_final_minutes:
                draft = compose_structured_minutes(
                    result.content, prepared=prepared, authoritative_facts=authoritative_facts
                )
                raw_markdown = draft.content
                structured_metadata = {
                    "generation_method": "deepseek_structured",
                    "summary_bindings": draft.summary_bindings,
                    "summary_bindings_sha256": _payload_hash(draft.summary_bindings),
                }
            content = compose_authoritative_minutes(
                raw_markdown,
                expected_speaker_labels=prepared.expected_speaker_labels,
                authoritative_facts=authoritative_facts,
            )
            content, reconciliation = reconcile_generated_transcript(
                content,
                prepared=prepared,
            )
            content, evidence_normalization = normalize_generated_evidence(
                content,
                expected_speaker_labels=prepared.expected_speaker_labels,
                expected_source_speaker_labels=prepared.source_speaker_labels,
                authoritative_facts=authoritative_facts,
            )
            validate_generated_minutes(
                content,
                expected_speaker_labels=prepared.expected_speaker_labels,
                expected_source_speaker_labels=prepared.source_speaker_labels,
                authoritative_facts=authoritative_facts,
            )
            validated_content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            content, evidence_audit = render_public_minutes(content)
            validate_public_minutes(
                content,
                evidence_audit=evidence_audit,
                expected_speaker_labels=prepared.expected_speaker_labels,
                expected_source_speaker_labels=prepared.source_speaker_labels,
                authoritative_facts=authoritative_facts,
            )
        except FullTranscriptPreviewError as exc:
            failed_attempt: dict[str, object] = {
                **started_attempt,
                "status": "failed",
                "failure_code": exc.code,
            }
            if exc.diagnostics is not None:
                failed_attempt["failure_diagnostics"] = exc.diagnostics.as_dict()
            elif result is not None and result.diagnostics is not None:
                failed_attempt["failure_diagnostics"] = result.diagnostics.as_dict()
            if result is not None:
                failed_attempt["provider_response_content_bytes"] = len(
                    result.content.encode("utf-8")
                )
                failed_attempt["provider_response_sha256"] = hashlib.sha256(
                    result.content.encode("utf-8")
                ).hexdigest()
            if exc.validation_details is not None:
                failed_attempt["validation_details"] = exc.validation_details.as_dict()
            atomic_write_text(
                attempt_path,
                _json_text(failed_attempt),
            )
            raise
        except Exception:
            failure_code = "final_minutes_unexpected_error"
            atomic_write_text(
                attempt_path,
                _json_text(
                    {
                        **started_attempt,
                        "status": "failed",
                        "failure_code": failure_code,
                    }
                ),
            )
            raise FullTranscriptPreviewError(failure_code) from None
        atomic_write_text(
            attempt_path,
            _json_text(
                {
                    **started_attempt,
                    **structured_metadata,
                    "status": "complete",
                    "finish_reason": result.finish_reason,
                    "response_content_bytes": len(content.encode("utf-8")),
                    "provider_response_content_bytes": result.response_content_bytes,
                    "provider_response_sha256": hashlib.sha256(
                        result.content.encode("utf-8")
                    ).hexdigest(),
                    "validated_minutes_sha256": validated_content_sha256,
                    "minutes_markdown_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "evidence_audit_sha256": _payload_hash(evidence_audit),
                    "evidence_audit": evidence_audit,
                    "transcript_reconciliation": reconciliation.as_dict(),
                    "evidence_normalization": evidence_normalization.as_dict(),
                    "usage": result.usage,
                }
            ),
        )
        manifest = {
            **manifest_base,
            **structured_metadata,
            "status": result.status,
            "finish_reason": result.finish_reason,
            "response_content_bytes": len(content.encode("utf-8")),
            "provider_response_content_bytes": result.response_content_bytes,
            "provider_response_sha256": hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
            "validated_minutes_sha256": validated_content_sha256,
            "minutes_markdown_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "evidence_audit_sha256": _payload_hash(evidence_audit),
            "evidence_audit": evidence_audit,
            "transcript_reconciliation": reconciliation.as_dict(),
            "evidence_normalization": evidence_normalization.as_dict(),
            "usage": result.usage,
        }
        details: dict[str, int | str] = {
            "final_output_method": (
                "deepseek_structured"
                if self._structured_final_minutes
                else "deepseek_full_transcript"
            ),
            "final_output_rows": prepared.row_count,
            "final_output_noise_removed": prepared.removal_counts.total,
            "final_output_bytes": len(content.encode("utf-8")),
            "final_output_agendas": authoritative_facts.agenda_count,
            "final_output_speaking_plans": authoritative_facts.speaking_plan_count,
            "final_output_mute_summaries": authoritative_facts.mute_enforcement_count,
            "final_output_votes": authoritative_facts.vote_count,
            "final_output_speaking_rows": authoritative_facts.speaking_row_count,
            "final_output_model_transcript_rows": reconciliation.accepted_model_source_count,
            "final_output_fallback_transcript_rows": reconciliation.fallback_source_count,
            "final_output_rejected_model_transcript_lines": (
                reconciliation.rejected_model_line_count
            ),
            "final_output_normalized_evidence_lines": (
                evidence_normalization.ordinal_lines_normalized
            ),
            "final_output_unproven_action_owners": (
                evidence_normalization.action_owners_downgraded
            ),
        }
        return content, manifest, details

    async def _full_transcript_minutes_are_current(self, session_id: str) -> bool:
        meeting_dir = self._meeting_dir(session_id)
        manifest_path = meeting_dir / "full_transcript_minutes.json"
        markdown_path = meeting_dir / "minutes.md"
        document_path = meeting_dir / "minutes.json"
        if (
            not manifest_path.is_file()
            or not markdown_path.is_file()
            or not document_path.is_file()
        ):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False
            document = MinutesDocumentV2.model_validate_json(
                document_path.read_text(encoding="utf-8")
            )
            (
                prepared,
                authoritative_facts,
                _,
                template_sha256,
                messages,
            ) = await self._full_transcript_minutes_request(
                session_id,
                document,
            )
            expected_base = self._full_transcript_manifest_base(
                prepared,
                authoritative_facts,
                template_sha256=template_sha256,
                messages=messages,
            )
            if any(manifest.get(key) != value for key, value in expected_base.items()):
                return False
            content = markdown_path.read_text(encoding="utf-8")
            if manifest.get("generation_method") == "deterministic_basic":
                expected_content, expected_output = render_basic_minutes(
                    prepared, authoritative_facts
                )
                reason = manifest.get("fallback_reason")
                return (
                    isinstance(reason, str)
                    and re.fullmatch(r"[a-z0-9_]{1,96}", reason) is not None
                    and content == expected_content
                    and all(manifest.get(key) == value for key, value in expected_output.items())
                    and manifest.get("evidence_audit_sha256")
                    == _payload_hash(expected_output["evidence_audit"])
                )
            evidence_audit = manifest.get("evidence_audit")
            if self._structured_final_minutes and (
                manifest.get("generation_method") != "deepseek_structured"
                or not validate_summary_bindings(
                    content, manifest.get("summary_bindings"), prepared
                )
                or manifest.get("summary_bindings_sha256")
                != _payload_hash(manifest.get("summary_bindings"))
            ):
                return False
            if not isinstance(evidence_audit, dict):
                return False
            validate_public_minutes(
                content,
                evidence_audit=evidence_audit,
                expected_speaker_labels=prepared.expected_speaker_labels,
                expected_source_speaker_labels=prepared.source_speaker_labels,
                authoritative_facts=authoritative_facts,
            )
            if manifest.get("evidence_audit_sha256") != _payload_hash(evidence_audit):
                return False
            if manifest.get("validated_minutes_sha256") != evidence_audit.get(
                "annotated_minutes_sha256"
            ):
                return False
            usage = manifest.get("usage")
            if not isinstance(usage, dict) or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in usage.items()
            ):
                return False
            evidence_normalization = manifest.get("evidence_normalization")
            if (
                not isinstance(evidence_normalization, dict)
                or set(evidence_normalization)
                != {"ordinal_lines_normalized", "action_owners_downgraded"}
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in evidence_normalization.values()
                )
            ):
                return False
            reconciliation = manifest.get("transcript_reconciliation")
            if not isinstance(reconciliation, dict):
                return False
            reconciliation_keys = {
                "source_row_count",
                "model_line_count",
                "accepted_model_group_count",
                "accepted_model_source_count",
                "fallback_source_count",
                "rejected_model_line_count",
            }
            if set(reconciliation) != reconciliation_keys or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in reconciliation.values()
            ):
                return False
            source_row_count = len(prepared.source_rows)
            if (
                reconciliation["source_row_count"] != source_row_count
                or reconciliation["accepted_model_source_count"]
                + reconciliation["fallback_source_count"]
                != source_row_count
                or reconciliation["accepted_model_group_count"]
                > reconciliation["accepted_model_source_count"]
                or reconciliation["accepted_model_group_count"]
                + reconciliation["rejected_model_line_count"]
                != reconciliation["model_line_count"]
                or reconciliation["rejected_model_line_count"] > reconciliation["model_line_count"]
            ):
                return False
            expected_output = {
                "status": "complete",
                "finish_reason": "stop",
                "response_content_bytes": len(content.encode("utf-8")),
                "minutes_markdown_sha256": _sha256(markdown_path),
                "public_minutes_revision": PUBLIC_MINUTES_REVISION,
            }
            return all(manifest.get(key) == value for key, value in expected_output.items())
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            FullTranscriptPreviewError,
        ):
            return False

    async def _stage_generate_minutes(self, session_id: str) -> dict[str, Any]:
        if self._full_transcript_minutes_provider is not None and await self._minutes_are_current(
            session_id
        ):
            document = MinutesDocumentV2.model_validate_json(
                (self._meeting_dir(session_id) / "minutes.json").read_text(encoding="utf-8")
            )
            return {
                "final_output_method": json.loads(
                    (self._meeting_dir(session_id) / "full_transcript_minutes.json").read_text(
                        encoding="utf-8"
                    )
                ).get("generation_method", "deepseek_full_transcript"),
                "artifact_reused": True,
                "renderer_input_sha256": document.provenance.renderer_input_sha256,
            }
        document, canonical, labels, details = await self._build_minutes_document(
            session_id,
            generated_at=datetime.now(UTC),
        )
        meeting_dir = self._meeting_dir(session_id)

        manifest: dict[str, object] | None = None
        if self._full_transcript_minutes_provider is None:
            markdown_text = render_minutes(document, canonical, labels=labels)
        else:
            (
                markdown_text,
                manifest,
                generation_details,
            ) = await self._generate_full_transcript_minutes(session_id, document)
            details.update(generation_details)
        document_payload = document.model_dump(mode="python")
        document_payload["provenance"] = {
            **document_payload["provenance"],
            "minutes_markdown_sha256": hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
        }
        document = MinutesDocumentV2.model_validate(document_payload)
        minutes_payload = document.model_dump(mode="json")
        document_payload = document.model_dump(mode="python")
        document_payload["provenance"] = {
            **document_payload["provenance"],
            "minutes_payload_hash": _payload_hash(self._minutes_hash_input(minutes_payload)),
        }
        document = MinutesDocumentV2.model_validate(document_payload)
        minutes_payload = document.model_dump(mode="json")
        atomic_write_text(meeting_dir / "minutes.json", _json_text(minutes_payload))
        atomic_write_text(meeting_dir / "minutes.md", markdown_text)
        if manifest is not None:
            atomic_write_text(
                meeting_dir / "full_transcript_minutes.json",
                _json_text(manifest),
            )
        return details

    async def _minutes_consistency_context(
        self,
        session_id: str,
        *,
        require_registered_artifacts: bool = False,
    ) -> MinutesConsistencyContext:
        """Rebuild authoritative projections without making any model call."""

        meeting_dir = self._meeting_dir(session_id)
        source_entries = await self._transcripts.list(session_id)
        participants = await self._participants.list_session(session_id)
        session = await self._sessions.get(session_id)
        timeline = await self._consent_timeline(session_id)
        expected_canonical, _ = await self._build_normalised_canonical(session_id)
        candidates = topic_candidates(expected_canonical.entries)
        candidate_payload = [
            {**asdict(candidate), "reason": list(candidate.reason)} for candidate in candidates
        ]
        topic_path = meeting_dir / "topic_segments.json"
        topic_payload = reusable_topic_payload(
            path=topic_path,
            canonical=expected_canonical,
            candidate_payload=candidate_payload,
            policy_provenance=stored_topic_policy_provenance(topic_path),
        )
        if topic_payload is None:
            raise ValueError("topic segment artifact provenance is invalid")
        document = MinutesDocumentV2.model_validate_json(
            (meeting_dir / "minutes.json").read_text(encoding="utf-8")
        )
        expected, canonical, labels, _ = await self._build_minutes_document(
            session_id,
            generated_at=document.generated_at,
            canonical_override=expected_canonical,
        )
        if self._full_transcript_minutes_provider is not None:
            if not await self._full_transcript_minutes_are_current(session_id):
                raise ValueError("full transcript minutes provenance is invalid")
            expected_markdown = (meeting_dir / "minutes.md").read_text(encoding="utf-8")
        else:
            expected_markdown = render_minutes(expected, canonical, labels=labels)
        expected_payload = expected.model_dump(mode="python")
        expected_payload["provenance"] = {
            **expected_payload["provenance"],
            "minutes_markdown_sha256": hashlib.sha256(
                expected_markdown.encode("utf-8")
            ).hexdigest(),
        }
        expected = MinutesDocumentV2.model_validate(expected_payload)
        expected_json_payload = expected.model_dump(mode="json")
        expected_payload = expected.model_dump(mode="python")
        expected_payload["provenance"] = {
            **expected_payload["provenance"],
            "minutes_payload_hash": _payload_hash(self._minutes_hash_input(expected_json_payload)),
        }
        expected = MinutesDocumentV2.model_validate(expected_payload)

        evidence_payload = json.loads((meeting_dir / "evidence.json").read_text(encoding="utf-8"))
        observed_evidence = evidence_payload.get("items")
        if not isinstance(observed_evidence, list) or any(
            not isinstance(item, dict) for item in observed_evidence
        ):
            raise ValueError("evidence items are malformed")
        expected_evidence, _ = await self._evidence_items(session_id)

        language_payload = json.loads((meeting_dir / "languages.json").read_text(encoding="utf-8"))
        if not isinstance(language_payload, dict) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in language_payload.items()
        ):
            raise ValueError("language metrics are malformed")
        expected_records = self._canonical_records(
            session_id,
            summary_ready=True,
            canonical=expected_canonical,
        )
        expected_languages = self._language_counts(expected_records)
        host, attendees, render_labels = await self._speaker_context(session_id)
        events = await self._events.list(session_id)
        topics = self._read_topic_segments(session_id)
        events_by_topic = await self._outcome_events_by_topic(session_id, topics, events)
        canonical_by_id = {entry.entry_id: entry for entry in expected_canonical.entries}
        topics_input = tuple(
            (
                topic,
                tuple(canonical_by_id[entry_id] for entry_id in topic.entry_ids),
                events_by_topic.get(topic.id, ()),
            )
            for topic in topics
        )
        event_facts_hash = _payload_hash(
            {
                topic_id: [event.model_dump(mode="json") for event in facts]
                for topic_id, facts in events_by_topic.items()
            }
        )
        participant_hash = _payload_hash(sorted(render_labels))
        privacy_journal = self._read_privacy_journal(
            session_id,
            allow_legacy=self._privacy_legacy_schema_allowed.get(session_id, False),
        )
        if privacy_journal is None:
            raise ValueError("privacy exclusion journal is missing")
        evidence_expected_payload = self._evidence_payload(session_id, expected_evidence)
        partition_manifest_path = meeting_dir / "outcome_partition_manifest.json"
        partition_manifest = OutcomePartitionManifest.model_validate_json(
            partition_manifest_path.read_text(encoding="utf-8")
        )
        validate_outcome_partition_manifest(
            partition_manifest,
            canonical_hash=expected_canonical.serialized_hash,
            topic_artifact_hash=_sha256(meeting_dir / "topic_segments.json"),
            event_facts_hash=event_facts_hash,
            participant_hash=participant_hash,
            topics=topics_input,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            allowed_participant_ids=tuple(sorted(render_labels)),
        )
        boundary_candidates = topic_payload.get("candidates")
        topic_provenance = topic_payload.get("provenance")
        outcome_payload = json.loads(
            (meeting_dir / "topic_outcome_candidates.json").read_text(encoding="utf-8")
        )
        outcome_provenance = (
            outcome_payload.get("provenance") if isinstance(outcome_payload, dict) else None
        )
        if (
            not isinstance(boundary_candidates, list)
            or not isinstance(topic_provenance, dict)
            or not isinstance(outcome_provenance, dict)
        ):
            raise ValueError("topic candidate inventory is malformed")
        job_hash = llm_job_hash(
            canonical_hash=expected_canonical.serialized_hash,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )
        boundary_tasks = tuple(
            boundary_task_hash(job_hash, str(candidate.get("entry_id")))
            for candidate in boundary_candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("entry_id"), str)
        )
        if len(boundary_tasks) != len(boundary_candidates):
            raise ValueError("topic candidate inventory is malformed")
        outcome_tasks = tuple(
            outcome_task_hash(job_hash, partition, kind)
            for partition in partition_manifest.partitions
            for kind in OutcomeRequestKind
        )
        outcome_enabled = outcome_provenance.get("provider_interface") == "single_kind"
        prior_tasks = (
            (digest_task_hash(job_hash), *boundary_tasks)
            if topic_provenance.get("status") == "approved" and boundary_tasks
            else ()
        )
        expected_tasks = (
            (digest_task_hash(job_hash), *boundary_tasks, *outcome_tasks)
            if outcome_enabled
            else prior_tasks
        )
        journal = read_budget_journal(
            meeting_dir / "outcome_budget_journal.json",
            job_hash=job_hash,
            expected_task_hashes=expected_tasks,
        )
        partition_results = read_partition_results(
            meeting_dir / "outcome_partition_results.json",
            partition_manifest,
            pipeline_revision=_OUTCOME_PIPELINE_REVISION,
        )
        validate_partition_result_keys(
            tuple(partition_results),
            expected_partition_result_keys(partition_manifest) if outcome_enabled else (),
        )
        for (partition_id, kind), result in partition_results.items():
            partition = next(
                item for item in partition_manifest.partitions if item.partition_id == partition_id
            )
            task_hash = outcome_task_hash(job_hash, partition, kind)
            if task_hash not in journal.result_task_hashes:
                raise ValueError("outcome result ledger binding is invalid")
            result.validate_for(partition, kind, _OUTCOME_PIPELINE_REVISION)
        journal.validate_expected(expected_tasks)
        journal.validate_final()
        expected_texts = {
            "CONSENT_TIMELINE_JSON": _json_text(self._consent_timeline_payload(timeline)),
            "PRIVACY_EXCLUSIONS_JSON": _json_text(privacy_journal),
            "CANONICAL_TRANSCRIPT_JSON": _json_text(expected_canonical.to_dict()),
            "TOPIC_SEGMENTS_JSON": _json_text(topic_payload),
            "OUTCOME_PARTITION_MANIFEST_JSON": _json_text(
                partition_manifest.model_dump(mode="json")
            ),
            "OUTCOME_BUDGET_JOURNAL_JSON": (meeting_dir / "outcome_budget_journal.json").read_text(
                encoding="utf-8"
            ),
            "OUTCOME_PARTITION_RESULTS_JSON": (
                meeting_dir / "outcome_partition_results.json"
            ).read_text(encoding="utf-8"),
            "TRANSCRIPT_JSON": _json_text([self._entry_payload(entry) for entry in source_entries]),
            "TRANSCRIPT_TEXT": self._render_transcript_text(
                session_id=session_id,
                entries=expected_records,
                host=host,
                attendees=attendees,
                labels=render_labels,
            ),
            "MEETING_EVENTS_JSON": _json_text(self._meeting_events_payload(events)),
            "LANGUAGE_DISTRIBUTION_JSON": _json_text(dict(sorted(expected_languages.items()))),
            "EVIDENCE_JSON": _json_text(evidence_expected_payload),
            "MINUTES_MARKDOWN": expected_markdown,
            "MINUTES_JSON": _json_text(expected.model_dump(mode="json")),
        }
        internal_artifacts: list[ArtifactDigestExpectation] = []
        for name, artifact_type in _REGISTERED_FILE_ARTIFACTS:
            expected_bytes = expected_texts[artifact_type].encode("utf-8")
            expected_size = len(expected_bytes)
            path = meeting_dir / name
            registered = await self._artifacts.at_path(
                session_id=session_id,
                artifact_type=artifact_type,
                path=path,
            )
            internal_artifacts.append(
                ArtifactDigestExpectation(
                    artifact_type=artifact_type,
                    path_identity=name,
                    path=path,
                    expected_sha256=hashlib.sha256(expected_bytes).hexdigest(),
                    expected_size=expected_size,
                    registered_sha256=registered.sha256 if registered is not None else None,
                    registered_metadata=registered.metadata if registered is not None else None,
                    registered_encrypted=registered.encrypted if registered is not None else None,
                    expected_metadata=self._registered_artifact_metadata(
                        artifact_type,
                        expected_size,
                        document=expected,
                    ),
                    registration_required=require_registered_artifacts,
                )
            )

        readable_ids = tuple(
            entry.entry_id
            for entry in expected_canonical.entries
            if entry.entry_type not in _SUMMARY_EXCLUDED_ENTRY_TYPES
        )
        eligible_user_ids = tuple(
            sorted(
                {
                    interval.user_id
                    for interval in timeline.intervals
                    if interval.kind is ConsentKind.MINUTES
                }
            )
        )
        eligible_source_ids = tuple(
            entry.id
            for entry in source_entries
            if entry.user_id is not None
            and timeline.allows_interval(
                entry.user_id,
                ConsentKind.MINUTES,
                entry.started_ms,
                entry.ended_ms,
            )
        )
        return MinutesConsistencyContext(
            document=document,
            expected_document=expected,
            canonical=canonical,
            labels=labels,
            participants=participants,
            host_user_id=session.creator_user_id,
            consent_eligible_user_ids=eligible_user_ids,
            consent_eligible_source_entry_ids=eligible_source_ids,
            source_entries=source_entries,
            agendas=await self._agenda.list(session_id),
            topics=self._read_topic_segments(session_id),
            validated_outcomes=await self._validated_outcome_sets(session_id),
            events=events,
            expected_evidence_items=tuple(expected_evidence),
            observed_evidence_items=tuple(observed_evidence),
            privacy_exclusions=self._privacy_exclusions(session_id),
            expected_language_counts=dict(sorted(expected_languages.items())),
            observed_language_counts=dict(sorted(language_payload.items())),
            public_transcript_entry_ids=readable_ids,
            public_files=(meeting_dir / "minutes.md",),
            internal_artifacts=tuple(internal_artifacts),
            meeting_dir=meeting_dir,
            publication_contract_revision=PUBLICATION_CONTRACT_REVISION,
            publication_manifest=PUBLICATION_ATTACHMENTS,
        )

    async def _stage_validate_minutes_consistency(self, session_id: str) -> dict[str, Any]:
        report = await self._validate_minutes_consistency(session_id)
        return asdict(report)

    async def _validate_minutes_consistency(
        self,
        session_id: str,
        *,
        require_registered_artifacts: bool = False,
    ) -> ConsistencyReport:
        try:
            report = validate_minutes_consistency(
                await self._minutes_consistency_context(
                    session_id,
                    require_registered_artifacts=require_registered_artifacts,
                )
            )
        except MinutesConsistencyError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise MinutesConsistencyError(
                "consistency_inputs",
                expected=1,
                observed=0,
            ) from None
        return report

    async def _stage_render_artifacts(self, session_id: str) -> dict[str, Any]:
        meeting_dir = self._meeting_dir(session_id)
        document = MinutesDocumentV2.model_validate_json(
            (meeting_dir / "minutes.json").read_text(encoding="utf-8")
        )
        if self._privacy_exclusions(session_id) != document.privacy_exclusions:
            raise ValueError("privacy exclusion journal differs from minutes")
        for name, artifact_type in _REGISTERED_FILE_ARTIFACTS:
            path = meeting_dir / name
            metadata = self._registered_artifact_metadata(
                artifact_type,
                path.stat().st_size,
                document=document,
            )
            metadata.pop("size_bytes")
            await self._artifacts.add_file(
                session_id=session_id,
                artifact_type=artifact_type,
                path=meeting_dir / name,
                metadata=metadata,
            )
        return {"rendered_artifacts": len(_REGISTERED_FILE_ARTIFACTS)}

    async def _publication_manifest(
        self,
        session_id: str,
        *,
        allow_legacy_speech_revision: bool = False,
        publication_contract_revision: str = PUBLICATION_CONTRACT_REVISION,
        pipeline_version: int = _ARTIFACT_PIPELINE_VERSION,
    ) -> list[dict[str, Any]]:
        return await build_publication_manifest(
            artifacts=self._artifacts,
            meeting_dir=self._meeting_dir(session_id),
            session_id=session_id,
            outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            speech_act_policy_revision=_SPEECH_ACT_POLICY_REVISION,
            artifact_pipeline_version=_ARTIFACT_PIPELINE_VERSION,
            allow_legacy_speech_revision=allow_legacy_speech_revision,
            publication_contract_revision=publication_contract_revision,
            pipeline_version=pipeline_version,
        )

    async def _publication_is_terminal(self, session_id: str) -> bool:
        return await publication_is_terminal(
            sessions=self._sessions,
            artifacts=self._artifacts,
            meeting_dir=self._meeting_dir(session_id),
            session_id=session_id,
            outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            speech_act_policy_revision=_SPEECH_ACT_POLICY_REVISION,
            artifact_pipeline_version=_ARTIFACT_PIPELINE_VERSION,
        )

    async def _stage_publish_discord(self, session_id: str) -> dict[str, Any]:
        if await self._publication_is_terminal(session_id):
            return {"published": False, "deduplicated": True}
        if not await self._minutes_are_current(
            session_id,
            expected_renderer_input_sha256=self._minutes_renderer_inputs.get(session_id),
        ) or not await self._registered_outputs_are_current(session_id):
            raise PublicationReconciliationRequired("publication_record_invalid")
        await self._validate_minutes_consistency(
            session_id,
            require_registered_artifacts=True,
        )
        if self._publisher is None:
            return {"published": False, "reason": "publisher_disabled"}
        session = await self._sessions.get(session_id)
        meeting_dir = self._meeting_dir(session_id)
        manifest = await self._publication_manifest(session_id)
        preparation = await self._artifacts.prepare_discord_publication(
            session_id=session_id,
            pipeline_version=_ARTIFACT_PIPELINE_VERSION,
            outcome_pipeline_revision=_OUTCOME_PIPELINE_REVISION,
            speech_act_policy_revision=_SPEECH_ACT_POLICY_REVISION,
            publication_contract_revision=PUBLICATION_CONTRACT_REVISION,
            attachments=tuple(manifest),
        )
        if not preparation.created_by_this_call:
            raise PublicationReconciliationRequired("publication_prepared_unresolved")
        minutes_payload = json.loads((meeting_dir / "minutes.json").read_text(encoding="utf-8"))
        summary_method = str(minutes_payload.get("summary_method", "deterministic_fallback"))
        if self._full_transcript_minutes_provider is not None:
            generation = json.loads(
                (meeting_dir / "full_transcript_minutes.json").read_text(encoding="utf-8")
            )
            summary_note = (
                "智能整理未完成, 已生成基础纪要, 包含会议事实、发言摘录和完整记录; "
                "决策与行动事项请人工核对。"
                if generation.get("generation_method") == "deterministic_basic"
                else ""
            )
        elif summary_method == "deterministic":
            summary_note = "当前总结由确定性模板生成 (未使用 LLM)。"
        elif summary_method == "deterministic_fallback":
            summary_note = "DeepSeek 结果未通过验证, 已安全回退到确定性纪要。"
        elif summary_method in {"deepseek_partial", "local_llm_partial"}:
            summary_note = "部分模型候选已通过确定性证据验证; 其余内容安全回退。"
        else:
            summary_note = "模型候选已通过确定性证据验证; 未发布未经验证的叙事。"
        selected_channel, message_id = await self._publisher.publish(
            guild_id=session.guild_id,
            channel_id=session.publication_channel_id,
            content=(f"聆序会议 `{session_id}` 的逐字稿和已验证议题结果已生成。{summary_note}"),
            files=(meeting_dir / "minutes.md",),
        )
        if (
            isinstance(selected_channel, bool)
            or not isinstance(selected_channel, int)
            or selected_channel <= 0
            or not isinstance(message_id, str)
            or not message_id.isdecimal()
            or int(message_id) <= 0
        ):
            raise PublicationReconciliationRequired("publication_record_invalid")
        await self._artifacts.finalize_discord_publication(
            session_id=session_id,
            selected_channel_id=selected_channel,
            message_id=message_id,
        )
        return {"published": True, "channel_id": selected_channel}

    async def _stage_schedule_cleanup(self, session_id: str) -> dict[str, Any]:
        cleanup_at = datetime.now(UTC) + timedelta(hours=self._retention_hours)
        atomic_write_text(
            self._meeting_dir(session_id) / "cleanup.json",
            _json_text({"audio_cleanup_after": cleanup_at.isoformat()}),
        )
        return {"audio_cleanup_after": cleanup_at.isoformat()}
