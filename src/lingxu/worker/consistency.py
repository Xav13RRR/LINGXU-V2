"""Deterministic, content-safe cross-artifact validation for public minutes."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from lingxu.domain.models import (
    AgendaItemRecord,
    MeetingEventRecord,
    ParticipantRecord,
    TranscriptEntryRecord,
)
from lingxu.transcript.models import CanonicalTranscript
from lingxu.worker.minutes_models import MinutesDocumentV2, PrivacyExclusionCounts
from lingxu.worker.minutes_renderer import MINUTES_RENDERER_REVISION
from lingxu.worker.outcome_merge import merge_topic_outcomes
from lingxu.worker.outcomes import OutcomeType, TopicOutcomeSet, VerificationStatus
from lingxu.worker.topics import TopicSegment

_READABLE_EXCLUDED_ENTRY_TYPES = frozenset(
    {"backchannel", "noise", "excluded", "privacy_control", "privacy-control"}
)
_PUBLICATION_CONTRACT_REVISION = "task11-one-file-v1"
_INTERNAL_FILES = (
    ("CONSENT_TIMELINE_JSON", "consent_timeline.json"),
    ("PRIVACY_EXCLUSIONS_JSON", "privacy_exclusions.json"),
    ("CANONICAL_TRANSCRIPT_JSON", "canonical_transcript.json"),
    ("TOPIC_SEGMENTS_JSON", "topic_segments.json"),
    ("OUTCOME_PARTITION_MANIFEST_JSON", "outcome_partition_manifest.json"),
    ("OUTCOME_BUDGET_JOURNAL_JSON", "outcome_budget_journal.json"),
    ("OUTCOME_PARTITION_RESULTS_JSON", "outcome_partition_results.json"),
    ("TRANSCRIPT_JSON", "transcript.json"),
    ("TRANSCRIPT_TEXT", "transcript.txt"),
    ("MEETING_EVENTS_JSON", "events.json"),
    ("LANGUAGE_DISTRIBUTION_JSON", "languages.json"),
    ("EVIDENCE_JSON", "evidence.json"),
    ("MINUTES_MARKDOWN", "minutes.md"),
    ("MINUTES_JSON", "minutes.json"),
)
_OUTCOME_SECTIONS = {
    "decisions": OutcomeType.DECISION,
    "agreements": OutcomeType.AGREEMENT,
    "actions": OutcomeType.ACTION_ITEM,
    "questions": OutcomeType.OPEN_QUESTION,
    "proposals": OutcomeType.PROPOSAL,
    "risks": OutcomeType.RISK,
    "disagreements": OutcomeType.DISAGREEMENT,
}


class MinutesConsistencyError(RuntimeError):
    """A content-safe invariant failure containing counts/identities, never Meeting text."""

    def __init__(self, invariant: str, *, expected: int, observed: int) -> None:
        if not invariant or any(
            character not in "abcdefghijklmnopqrstuvwxyz_" for character in invariant
        ):
            raise ValueError("consistency invariant name is invalid")
        self.invariant = invariant
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 0
            or isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < 0
        ):
            raise ValueError("consistency observations must be non-negative counts")
        self.expected = expected
        self.observed = observed
        super().__init__(f"{invariant}:expected={expected}:observed={observed}")


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """Content-free aggregate proof emitted by a successful validation pass."""

    participant_entries: int
    bot_entries: int
    readable_entries: int
    excluded_entries: int
    agendas: int
    topics: int
    validated_outcomes: int
    votes: int
    evidence_items: int
    public_attachments: int
    internal_artifacts: int
    expected_artifact_digests: tuple[tuple[str, str, str, int], ...]
    report_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactDigestExpectation:
    """Content-free comparison inputs for one canonical internal artifact."""

    artifact_type: str
    path_identity: str
    path: Path
    expected_sha256: str
    expected_size: int
    registered_sha256: str | None
    registered_metadata: Mapping[str, Any] | None
    registered_encrypted: bool | None
    expected_metadata: Mapping[str, Any]
    registration_required: bool


@dataclass(frozen=True, slots=True)
class MinutesConsistencyContext:
    """Authoritative and generated structures compared immediately before publication."""

    document: MinutesDocumentV2
    expected_document: MinutesDocumentV2
    canonical: CanonicalTranscript
    labels: Mapping[int, str]
    participants: tuple[ParticipantRecord, ...]
    host_user_id: int
    consent_eligible_user_ids: tuple[int, ...]
    consent_eligible_source_entry_ids: tuple[str, ...]
    source_entries: tuple[TranscriptEntryRecord, ...]
    agendas: tuple[AgendaItemRecord, ...]
    topics: tuple[TopicSegment, ...]
    validated_outcomes: tuple[TopicOutcomeSet, ...]
    events: tuple[MeetingEventRecord, ...]
    expected_evidence_items: tuple[Mapping[str, Any], ...]
    observed_evidence_items: tuple[Mapping[str, Any], ...]
    privacy_exclusions: PrivacyExclusionCounts
    expected_language_counts: Mapping[str, int]
    observed_language_counts: Mapping[str, int]
    public_transcript_entry_ids: tuple[str, ...]
    public_files: tuple[Path, ...]
    internal_artifacts: tuple[ArtifactDigestExpectation, ...]
    meeting_dir: Path
    publication_contract_revision: str
    publication_manifest: tuple[tuple[str, str], ...]


def _fail(invariant: str, expected: int, observed: int) -> NoReturn:
    raise MinutesConsistencyError(invariant, expected=expected, observed=observed)


def validate_publication_files(files: Sequence[Path], *, meeting_dir: Path) -> None:
    """Require the exact canonical public file without disclosing paths in failures."""

    if len(files) != 1:
        _fail("public_attachment_count", 1, len(files))
    expected = (meeting_dir / "minutes.md").resolve()
    try:
        observed = files[0].resolve()
    except OSError:
        _fail("public_attachment_identity", 1, 0)
    if observed != expected:
        _fail("public_attachment_identity", 1, 0)


def _streaming_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _validate_internal_files(
    artifacts: tuple[ArtifactDigestExpectation, ...], *, meeting_dir: Path
) -> tuple[tuple[str, str, str, int], ...]:
    if len(artifacts) != len(_INTERNAL_FILES):
        _fail("internal_artifact_count", len(_INTERNAL_FILES), len(artifacts))
    expected = tuple(
        (artifact_type, filename, (meeting_dir / filename).resolve())
        for artifact_type, filename in _INTERNAL_FILES
    )
    observed = tuple(
        (artifact.artifact_type, artifact.path_identity, artifact.path.resolve())
        for artifact in artifacts
    )
    if observed != expected or len(set(observed)) != len(observed):
        _fail("internal_artifact_identity", len(expected), len(set(observed)))
    available = 0
    expected_digests: list[tuple[str, str, str, int]] = []
    for artifact in artifacts:
        try:
            if not stat.S_ISREG(artifact.path.lstat().st_mode):
                continue
            actual_sha256, actual_size = _streaming_digest(artifact.path)
        except OSError:
            continue
        available += 1
        if actual_sha256 != artifact.expected_sha256:
            _fail("internal_artifact_expected_digest", 1, 0)
        if actual_size != artifact.expected_size:
            _fail("internal_artifact_expected_size", artifact.expected_size, actual_size)
        if artifact.registration_required:
            if artifact.registered_sha256 != artifact.expected_sha256:
                _fail("internal_artifact_registered_digest", 1, 0)
            if artifact.registered_metadata != artifact.expected_metadata:
                _fail("internal_artifact_registered_metadata", 1, 0)
            if artifact.registered_encrypted is not False:
                _fail("internal_artifact_registered_encryption", 0, 1)
        expected_digests.append(
            (
                artifact.artifact_type,
                artifact.path_identity,
                artifact.expected_sha256,
                artifact.expected_size,
            )
        )
    if available != len(artifacts):
        _fail("internal_artifact_availability", len(artifacts), available)
    return tuple(expected_digests)


def _validate_speaker_identities(context: MinutesConsistencyContext) -> None:
    participants_by_user = {
        participant.user_id: participant for participant in context.participants
    }
    if len(participants_by_user) != len(context.participants):
        _fail("participant_roster_uniqueness", len(context.participants), len(participants_by_user))
    host = participants_by_user.get(context.host_user_id)
    if host is None:
        _fail("host_roster_projection", 1, 0)
    expected_host = f"{host.display_name or f'user:{host.user_id}'} (user:{host.user_id})"
    if context.expected_document.participants.host != expected_host:
        _fail("host_label_projection", 1, 0)
    unexpected_hosts = sum(
        participant.is_host and participant.user_id != context.host_user_id
        for participant in context.participants
    )
    if unexpected_hosts:
        _fail("host_roster_projection", 0, unexpected_hosts)

    eligible = set(context.consent_eligible_user_ids)
    eligible_sources = set(context.consent_eligible_source_entry_ids)
    sources = {entry.id: entry for entry in context.source_entries}
    if len(sources) != len(context.source_entries):
        _fail("source_identity_uniqueness", len(context.source_entries), len(sources))
    mismatched_source_kinds = 0
    missing_source_roster: set[int] = set()
    missing_source_consent: set[str] = set()
    for source in context.source_entries:
        is_bot_source = source.source_key.startswith("BOT_TTS:")
        if is_bot_source:
            mismatched_source_kinds += int(source.user_id is not None)
            continue
        if source.user_id is None:
            mismatched_source_kinds += 1
            continue
        if source.user_id not in participants_by_user:
            missing_source_roster.add(source.user_id)
        if source.id not in eligible_sources:
            missing_source_consent.add(source.id)
    if mismatched_source_kinds:
        _fail("source_speaker_kind", 0, mismatched_source_kinds)
    if missing_source_roster:
        _fail("source_participant_roster", 0, len(missing_source_roster))
    if missing_source_consent:
        _fail("source_consent_coverage", 0, len(missing_source_consent))

    missing_roster = {
        entry.speaker_user_id
        for entry in context.canonical.entries
        if entry.speaker_user_id is not None and entry.speaker_user_id not in participants_by_user
    }
    if missing_roster:
        _fail("participant_roster_coverage", 0, len(missing_roster))
    missing_consent = {
        entry.speaker_user_id
        for entry in context.canonical.entries
        if entry.speaker_user_id is not None and entry.speaker_user_id not in eligible
    }
    if missing_consent:
        _fail("participant_consent_coverage", 0, len(missing_consent))

    missing_sources = 0
    source_mismatches = 0
    source_consent_mismatches = 0
    label_mismatches = 0
    duplicate_source_references = 0
    referenced_source_ids: list[str] = []
    canonical_entry_ids: set[str] = set()
    for entry in context.canonical.entries:
        if entry.entry_id in canonical_entry_ids:
            _fail(
                "canonical_entry_identity", len(context.canonical.entries), len(canonical_entry_ids)
            )
        canonical_entry_ids.add(entry.entry_id)
        duplicate_source_references += len(entry.source_entry_ids) - len(
            set(entry.source_entry_ids)
        )
        referenced_source_ids.extend(entry.source_entry_ids)
        entry_sources = [sources.get(source_id) for source_id in entry.source_entry_ids]
        missing_sources += sum(source is None for source in entry_sources)
        present_sources = [source for source in entry_sources if source is not None]
        if entry.speaker_user_id is None:
            source_mismatches += sum(
                source.user_id is not None or not source.source_key.startswith("BOT_TTS:")
                for source in present_sources
            )
            if entry.speaker_name != "聆序机器人":
                label_mismatches += 1
        else:
            source_mismatches += sum(
                source.user_id != entry.speaker_user_id or source.source_key.startswith("BOT_TTS:")
                for source in present_sources
            )
            source_consent_mismatches += sum(
                source.id not in eligible_sources for source in present_sources
            )
            participant = participants_by_user.get(entry.speaker_user_id)
            if participant is not None:
                expected_label = participant.display_name or f"user:{participant.user_id}"
                if entry.speaker_name != expected_label:
                    label_mismatches += 1
    if missing_sources:
        _fail("canonical_source_existence", 0, missing_sources)
    if source_mismatches:
        _fail("canonical_source_identity", 0, source_mismatches)
    if source_consent_mismatches:
        _fail("canonical_source_consent", 0, source_consent_mismatches)
    if label_mismatches:
        _fail("canonical_speaker_label", 0, label_mismatches)
    if duplicate_source_references or len(referenced_source_ids) != len(set(referenced_source_ids)):
        _fail(
            "canonical_source_uniqueness",
            len(set(referenced_source_ids)),
            len(referenced_source_ids),
        )
    expected_source_ids = set(sources)
    observed_source_ids = set(referenced_source_ids)
    if observed_source_ids != expected_source_ids:
        _fail("canonical_source_completeness", len(expected_source_ids), len(observed_source_ids))


def _validate_document_structure(context: MinutesConsistencyContext) -> None:
    document = context.document
    canonical = context.canonical
    readable = tuple(
        entry
        for entry in canonical.entries
        if entry.entry_type not in _READABLE_EXCLUDED_ENTRY_TYPES
    )
    expected_public_ids = tuple(entry.entry_id for entry in readable)
    if context.public_transcript_entry_ids != expected_public_ids:
        _fail(
            "public_transcript_entries",
            len(expected_public_ids),
            len(context.public_transcript_entry_ids),
        )
    if len(set(context.public_transcript_entry_ids)) != len(context.public_transcript_entry_ids):
        _fail(
            "public_transcript_uniqueness",
            len(context.public_transcript_entry_ids),
            len(set(context.public_transcript_entry_ids)),
        )
    excluded_ids = {
        entry.entry_id
        for entry in canonical.entries
        if entry.entry_type in _READABLE_EXCLUDED_ENTRY_TYPES
    }
    overlap = excluded_ids.intersection(context.public_transcript_entry_ids)
    if overlap:
        _fail("public_transcript_exclusions", 0, len(overlap))

    participant_count = sum(entry.speaker_user_id is not None for entry in readable)
    bot_count = sum(entry.speaker_user_id is None for entry in readable)
    if document.deterministic_counts.participant_transcript_entries != participant_count:
        _fail(
            "participant_transcript_count",
            participant_count,
            document.deterministic_counts.participant_transcript_entries,
        )
    if document.deterministic_counts.bot_tts_entries != bot_count:
        _fail("bot_transcript_count", bot_count, document.deterministic_counts.bot_tts_entries)
    missing_labels = {
        entry.speaker_user_id
        for entry in readable
        if entry.speaker_user_id is not None and entry.speaker_user_id not in context.labels
    }
    if missing_labels:
        _fail("participant_label_coverage", 0, len(missing_labels))
    if document.participants != context.expected_document.participants:
        _fail("participant_projection", 1, 0)

    if len(document.agendas) != len(context.agendas):
        _fail("agenda_count", len(context.agendas), len(document.agendas))
    completed = sum(item.status.value == "COMPLETED" for item in context.agendas)
    if document.deterministic_counts.agenda_completed != completed:
        _fail("completed_agenda_count", completed, document.deterministic_counts.agenda_completed)
    if document.agendas != context.expected_document.agendas:
        _fail("agenda_projection", len(context.agendas), 0)

    if len(document.topics) != len(context.topics):
        _fail("topic_count", len(context.topics), len(document.topics))
    if document.topics != context.expected_document.topics:
        _fail("topic_projection", len(context.topics), 0)
    canonical_ids = {entry.entry_id for entry in canonical.entries}
    unknown_topic_sources = {
        source_id
        for topic in document.topics
        for source_id in topic.source_entry_ids
        if source_id not in canonical_ids
    }
    if unknown_topic_sources:
        _fail("topic_source_existence", 0, len(unknown_topic_sources))
    topic_ids = {topic.topic_id for topic in document.topics}
    bound_topic_ids = {topic_id for agenda in document.agendas for topic_id in agenda.topic_ids}
    if not bound_topic_ids.issubset(topic_ids):
        _fail("topic_agenda_binding", 0, len(bound_topic_ids - topic_ids))

    if document.validated_outcomes != context.validated_outcomes:
        _fail(
            "validated_outcome_projection",
            sum(len(item.candidates) for item in context.validated_outcomes),
            sum(len(item.candidates) for item in document.validated_outcomes),
        )
    event_ids = {event.sequence_number for event in context.events}
    readable_ids = set(expected_public_ids)
    unknown_entry_sources: set[str] = set()
    unknown_event_sources: set[int] = set()
    for outcome_set in context.validated_outcomes:
        if outcome_set.topic_id not in topic_ids:
            _fail("outcome_topic_binding", 1, 0)
        for candidate in outcome_set.candidates:
            unknown_entry_sources.update(set(candidate.source_entry_ids) - readable_ids)
            unknown_event_sources.update(set(candidate.source_event_ids) - event_ids)
    if unknown_entry_sources:
        _fail("outcome_entry_source_existence", 0, len(unknown_entry_sources))
    if unknown_event_sources:
        _fail("outcome_event_source_existence", 0, len(unknown_event_sources))

    merged = merge_topic_outcomes(context.validated_outcomes)
    for field, outcome_type in _OUTCOME_SECTIONS.items():
        observed = getattr(document, field)
        expected = getattr(merged, field)
        if observed != expected or any(item.type is not outcome_type for item in observed):
            _fail(f"{field}_routing", len(expected), len(observed))
        if any(
            item.verification_status
            in {VerificationStatus.NOT_CHECKED, VerificationStatus.UNVERIFIED}
            for item in observed
        ):
            _fail(f"{field}_status", 0, len(observed))
    for field in ("deferred_items", "speaker_claims"):
        observed = getattr(document, field)
        expected = getattr(merged, field)
        if observed != expected:
            _fail(f"{field}_routing", len(expected), len(observed))
    if any(
        item.verification_status is not VerificationStatus.NOT_CHECKED
        for item in document.speaker_claims
    ):
        _fail("speaker_claims_status", 0, len(document.speaker_claims))

    if document.votes != context.expected_document.votes:
        _fail("vote_projection", len(context.expected_document.votes), len(document.votes))
    if document.speaking_plans != context.expected_document.speaking_plans:
        _fail(
            "speaking_plan_projection",
            len(context.expected_document.speaking_plans),
            len(document.speaking_plans),
        )
    if document.deterministic_counts.agenda_speaking_plans != len(
        context.expected_document.speaking_plans
    ):
        _fail(
            "speaking_plan_count",
            len(context.expected_document.speaking_plans),
            document.deterministic_counts.agenda_speaking_plans,
        )
    if document.mute_enforcement_summaries != context.expected_document.mute_enforcement_summaries:
        _fail(
            "mute_enforcement_projection",
            len(context.expected_document.mute_enforcement_summaries),
            len(document.mute_enforcement_summaries),
        )
    if document.deterministic_counts.mute_enforcement_summaries != len(
        context.expected_document.mute_enforcement_summaries
    ):
        _fail(
            "mute_enforcement_count",
            len(context.expected_document.mute_enforcement_summaries),
            document.deterministic_counts.mute_enforcement_summaries,
        )
    if document.speaking_turns != context.expected_document.speaking_turns:
        _fail(
            "speaking_turn_projection",
            len(context.expected_document.speaking_turns),
            len(document.speaking_turns),
        )
    if document.speaking_followups != context.expected_document.speaking_followups:
        _fail(
            "speaking_followup_projection",
            len(context.expected_document.speaking_followups),
            len(document.speaking_followups),
        )
    if document.deterministic_counts.controlled_speaking_turns != len(
        context.expected_document.speaking_turns
    ):
        _fail(
            "speaking_turn_count",
            len(context.expected_document.speaking_turns),
            document.deterministic_counts.controlled_speaking_turns,
        )
    if document.deterministic_counts.controlled_speaking_followups != len(
        context.expected_document.speaking_followups
    ):
        _fail(
            "speaking_followup_count",
            len(context.expected_document.speaking_followups),
            document.deterministic_counts.controlled_speaking_followups,
        )
    if context.observed_evidence_items != context.expected_evidence_items:
        _fail(
            "evidence_items",
            len(context.expected_evidence_items),
            len(context.observed_evidence_items),
        )
    if document.evidence_coverage != context.expected_document.evidence_coverage:
        _fail("evidence_coverage", 1, 0)
    if document.privacy_exclusions != context.privacy_exclusions:
        _fail("privacy_exclusions", 1, 0)
    if document.canonical_metrics != context.expected_document.canonical_metrics:
        _fail("canonical_metrics", 1, 0)
    if dict(context.observed_language_counts) != dict(context.expected_language_counts):
        _fail(
            "language_metrics",
            sum(context.expected_language_counts.values()),
            sum(context.observed_language_counts.values()),
        )

    if document.executive_summary != context.expected_document.executive_summary or (
        document.full_summary != context.expected_document.full_summary
    ):
        _fail("summary_facts", 1, 0)
    if document.provenance.minutes_renderer_revision != MINUTES_RENDERER_REVISION:
        _fail("renderer_revision", 1, 0)
    if document.provenance.renderer_input_sha256 != (
        context.expected_document.provenance.renderer_input_sha256
    ):
        _fail("renderer_input_digest", 1, 0)
    if document != context.expected_document:
        _fail("minutes_document", 1, 0)


def validate_minutes_consistency(context: MinutesConsistencyContext) -> ConsistencyReport:
    """Validate all authoritative projections and return content-free aggregate evidence."""

    _validate_speaker_identities(context)
    _validate_document_structure(context)
    validate_publication_files(context.public_files, meeting_dir=context.meeting_dir)
    expected_artifact_digests = _validate_internal_files(
        context.internal_artifacts, meeting_dir=context.meeting_dir
    )
    if context.publication_contract_revision != _PUBLICATION_CONTRACT_REVISION:
        _fail("publication_contract_revision", 1, 0)
    if context.publication_manifest != (("MINUTES_MARKDOWN", "minutes.md"),):
        _fail("publication_manifest", 1, len(context.publication_manifest))

    readable = tuple(
        entry
        for entry in context.canonical.entries
        if entry.entry_type not in _READABLE_EXCLUDED_ENTRY_TYPES
    )
    values = {
        "participant_entries": sum(entry.speaker_user_id is not None for entry in readable),
        "bot_entries": sum(entry.speaker_user_id is None for entry in readable),
        "readable_entries": len(readable),
        "excluded_entries": len(context.canonical.entries) - len(readable),
        "agendas": len(context.agendas),
        "topics": len(context.topics),
        "validated_outcomes": sum(len(item.candidates) for item in context.validated_outcomes),
        "votes": len(context.document.votes),
        "evidence_items": len(context.observed_evidence_items),
        "public_attachments": len(context.public_files),
        "internal_artifacts": len(context.internal_artifacts),
    }
    report_input = {
        **values,
        "expected_artifact_digests": expected_artifact_digests,
    }
    report_sha256 = hashlib.sha256(
        json.dumps(report_input, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ConsistencyReport(
        **values,
        expected_artifact_digests=expected_artifact_digests,
        report_sha256=report_sha256,
    )
