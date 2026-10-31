import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lingxu.domain.enums import AgendaStatus, MeetingState
from lingxu.domain.models import (
    AgendaItemRecord,
    MeetingEventRecord,
    ParticipantRecord,
    TranscriptEntryRecord,
)
from lingxu.transcript.models import (
    CanonicalTranscript,
    CanonicalTranscriptEntry,
    CanonicalTranscriptMetrics,
)
from lingxu.worker.consistency import (
    ArtifactDigestExpectation,
    MinutesConsistencyContext,
    MinutesConsistencyError,
    validate_minutes_consistency,
    validate_publication_files,
)
from lingxu.worker.minutes_models import (
    CanonicalMetrics,
    DeterministicCounts,
    EvidenceCoverage,
    MinutesAgenda,
    MinutesDocumentV2,
    MinutesParticipants,
    MinutesProvenance,
    MinutesSpeakingInterval,
    MinutesSpeakingTurn,
    MinutesTopic,
    MinutesVote,
    PrivacyExclusionCounts,
)
from lingxu.worker.minutes_renderer import MINUTES_RENDERER_REVISION
from lingxu.worker.outcome_merge import merge_topic_outcomes
from lingxu.worker.outcomes import (
    OutcomeCandidate,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)
from lingxu.worker.topics import TopicSegment

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


def _document() -> MinutesDocumentV2:
    decision = OutcomeCandidate(
        id="decision-1",
        type=OutcomeType.DECISION,
        text="Release is approved",
        owner_user_id=None,
        owner_name=None,
        deadline=None,
        source_entry_ids=("e1",),
        source_event_ids=(9,),
        confidence=0.95,
        requires_confirmation=False,
        verification_status=VerificationStatus.EVIDENCE_LINKED,
    )
    outcome_set = TopicOutcomeSet(
        topic_id="topic-001",
        candidates=(decision,),
        extraction_method="local_llm",
        failure_types=(),
    )
    merged = merge_topic_outcomes((outcome_set,))
    return MinutesDocumentV2(
        provenance=MinutesProvenance(
            outcome_pipeline_revision="revision",
            speech_act_policy_revision="e" * 64,
            minutes_renderer_revision=MINUTES_RENDERER_REVISION,
            renderer_input_sha256="f" * 64,
            validated_outcomes_sha256="a" * 64,
            canonical_transcript_sha256="b" * 64,
        ),
        session_id="session-1",
        generated_at=datetime(2026, 8, 19, tzinfo=UTC),
        meeting_title="Release review",
        participants=MinutesParticipants(host="host (user:1)", attendees=("participant (user:7)",)),
        agendas=(
            MinutesAgenda(
                agenda_id="agenda-1",
                position=1,
                title="Release",
                language="en",
                status="COMPLETED",
                duration_minutes=10,
                requires_vote=True,
                vote_content="Release?",
                vote_duration_seconds=60,
                topic_ids=("topic-001",),
                summary="Reviewed the release evidence.",
            ),
        ),
        topics=(
            MinutesTopic(
                topic_id="topic-001",
                title="Release",
                started_ms=1_000,
                ended_ms=2_000,
                method="deterministic_fallback",
                source_entry_ids=("e1",),
                summary=None,
            ),
        ),
        validated_outcomes=(outcome_set,),
        decisions=merged.decisions,
        agreements=(),
        actions=(),
        questions=(),
        proposals=(),
        risks=(),
        disagreements=(),
        deferred_items=(),
        speaker_claims=(),
        votes=(
            MinutesVote(
                source_event_id=9,
                vote_id="vote-1",
                agenda_item_id="agenda-1",
                vote_content="Release?",
                counts={"yes": 2, "no": 0, "abstain": 1},
                eligible_count=3,
                responded_count=2,
                result="yes",
            ),
        ),
        deterministic_counts=DeterministicCounts(
            agenda_total=1,
            agenda_completed=1,
            participant_transcript_entries=1,
            bot_tts_entries=0,
            closed_votes=1,
            topic_total=1,
            validated_outcomes=1,
        ),
        evidence_coverage=EvidenceCoverage(
            evidence_required=1,
            evidence_linked=1,
            transcript_sources=1,
            event_sources=1,
        ),
        privacy_exclusions=PrivacyExclusionCounts(
            transcript_entries=1,
            segment_rows=2,
            segment_files=2,
        ),
        canonical_metrics=CanonicalMetrics(
            source_entry_count=2,
            canonical_entry_count=1,
            duplicate_entry_count=1,
            backchannel_entry_count=0,
            asr_tag_count=0,
        ),
        summary_method="local_llm",
        summary_model="qwen3.5:9b",
        summary_provider="ollama",
        summary_model_digest="c" * 64,
        summary_policy_identity="d" * 64,
        fallback_reasons=(),
        executive_summary=None,
        full_summary="The release facts were reviewed.",
    )


def _context(tmp_path: Path) -> MinutesConsistencyContext:
    document = _document()
    entry = CanonicalTranscriptEntry(
        entry_id="e1",
        source_entry_ids=("raw-e1",),
        speaker_user_id=7,
        speaker_name="participant",
        started_ms=1_000,
        ended_ms=2_000,
        raw_texts=("synthetic release statement",),
        text="synthetic release statement",
        entry_type="speech",
        language="en",
        confidence=0.9,
        utterance_group_id="group-e1",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )
    canonical = CanonicalTranscript(
        entries=(entry,),
        metrics=CanonicalTranscriptMetrics(
            source_entry_count=2,
            canonical_entry_count=1,
            duplicate_entry_count=1,
            backchannel_entry_count=0,
            asr_tag_count=0,
        ),
    )
    event = MeetingEventRecord(
        id="event-9",
        session_id=document.session_id,
        sequence_number=9,
        event_type="VOTE_CLOSED",
        actor_user_id=1,
        source="synthetic",
        payload={
            "vote_id": "vote-1",
            "agenda_item_id": "agenda-1",
            "vote_content": "Release?",
            "counts": {"yes": 2, "no": 0, "abstain": 1},
            "eligible_count": 3,
            "responded_count": 2,
            "result": "yes",
        },
        evidence_utterance_ids=(),
        state_before=MeetingState.VOTE_OPEN,
        state_after=MeetingState.VOTE_RESULT,
        reversible=False,
        correlation_id="correlation-9",
        occurred_at=datetime(2026, 8, 19, tzinfo=UTC),
    )
    expected_bytes_by_type = {
        artifact_type: (
            b"# Deterministic synthetic minutes\n"
            if artifact_type == "MINUTES_MARKDOWN"
            else b"Deterministic synthetic transcript\n"
            if artifact_type == "TRANSCRIPT_TEXT"
            else (
                json.dumps(
                    {"artifact_type": artifact_type, "schema_version": 1},
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
        for artifact_type, _ in _INTERNAL_FILES
    }
    internal_artifacts = []
    for artifact_type, filename in _INTERNAL_FILES:
        path = tmp_path / filename
        payload = expected_bytes_by_type[artifact_type]
        path.write_bytes(payload)
        internal_artifacts.append(
            ArtifactDigestExpectation(
                artifact_type=artifact_type,
                path_identity=filename,
                path=path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size=len(payload),
                registered_sha256=hashlib.sha256(payload).hexdigest(),
                registered_metadata={},
                registered_encrypted=False,
                expected_metadata={},
                registration_required=True,
            )
        )
    return MinutesConsistencyContext(
        document=document,
        expected_document=document,
        canonical=canonical,
        labels={7: "participant"},
        participants=(
            ParticipantRecord(
                id="participant-host",
                session_id=document.session_id,
                user_id=1,
                present=True,
                display_name="host",
                is_host=True,
            ),
            ParticipantRecord(
                id="participant-7",
                session_id=document.session_id,
                user_id=7,
                present=True,
                display_name="participant",
            ),
        ),
        host_user_id=1,
        consent_eligible_user_ids=(7,),
        consent_eligible_source_entry_ids=("raw-e1",),
        source_entries=(
            TranscriptEntryRecord(
                id="raw-e1",
                session_id=document.session_id,
                source_key="utterance:raw-e1",
                user_id=7,
                raw_text="synthetic release statement",
                normalised_text="synthetic release statement",
                started_ms=1_000,
                ended_ms=2_000,
            ),
        ),
        agendas=(
            AgendaItemRecord(
                id="agenda-1",
                session_id=document.session_id,
                position=1,
                title="Release",
                status=AgendaStatus.COMPLETED,
                duration_minutes=10,
                requires_vote=True,
                vote_content="Release?",
                vote_duration_seconds=60,
                language="en",
            ),
        ),
        topics=(
            TopicSegment(
                id="topic-001",
                title="Release",
                entry_ids=("e1",),
                started_ms=1_000,
                ended_ms=2_000,
                method="deterministic_fallback",
            ),
        ),
        validated_outcomes=document.validated_outcomes,
        events=(event,),
        expected_evidence_items=({"kind": "synthetic"},),
        observed_evidence_items=({"kind": "synthetic"},),
        privacy_exclusions=document.privacy_exclusions,
        expected_language_counts={"en": 1},
        observed_language_counts={"en": 1},
        public_transcript_entry_ids=("e1",),
        public_files=(tmp_path / "minutes.md",),
        internal_artifacts=tuple(internal_artifacts),
        meeting_dir=tmp_path,
        publication_contract_revision="task11-one-file-v1",
        publication_manifest=(("MINUTES_MARKDOWN", "minutes.md"),),
    )


def test_publication_files_require_exactly_one_minutes_markdown(tmp_path: Path) -> None:
    minutes = tmp_path / "minutes.md"
    evidence = tmp_path / "evidence.json"

    with pytest.raises(MinutesConsistencyError) as error:
        validate_publication_files((minutes, evidence), meeting_dir=tmp_path)

    assert error.value.invariant == "public_attachment_count"
    assert str(error.value) == "public_attachment_count:expected=1:observed=2"


@pytest.mark.parametrize(
    "files, invariant",
    [
        ((), "public_attachment_count"),
        ((Path("other.md"),), "public_attachment_identity"),
        ((Path("minutes.md"), Path("minutes.md")), "public_attachment_count"),
    ],
)
def test_publication_files_fail_closed_with_content_safe_reasons(
    tmp_path: Path,
    files: tuple[Path, ...],
    invariant: str,
) -> None:
    resolved = tuple(tmp_path / path for path in files)
    with pytest.raises(MinutesConsistencyError) as error:
        validate_publication_files(resolved, meeting_dir=tmp_path)
    assert error.value.invariant == invariant
    assert all(str(path) not in str(error.value) for path in resolved)


def test_complete_consistency_context_returns_content_free_report(tmp_path: Path) -> None:
    report = validate_minutes_consistency(_context(tmp_path))

    assert report.participant_entries == 1
    assert report.bot_entries == 0
    assert report.public_attachments == 1
    assert report.internal_artifacts == 14
    assert len(report.report_sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "invariant"),
    (
        ("orphan_source", "canonical_source_completeness"),
        ("duplicate_source", "canonical_source_uniqueness"),
    ),
)
def test_canonical_source_contract_rejects_orphans_and_duplicate_references(
    tmp_path: Path,
    mutation: str,
    invariant: str,
) -> None:
    context = _context(tmp_path)
    if mutation == "orphan_source":
        orphan = TranscriptEntryRecord(
            id="orphan-bot-source",
            session_id=context.document.session_id,
            source_key="BOT_TTS:orphan",
            user_id=None,
            raw_text="synthetic bot source",
            normalised_text="synthetic bot source",
            started_ms=3_000,
            ended_ms=3_100,
        )
        context = replace(context, source_entries=(*context.source_entries, orphan))
    else:
        duplicate = replace(
            context.canonical.entries[0],
            entry_id="e2",
            started_ms=3_000,
            ended_ms=3_100,
        )
        context = replace(
            context,
            canonical=replace(
                context.canonical,
                entries=(*context.canonical.entries, duplicate),
            ),
        )

    with pytest.raises(MinutesConsistencyError) as error:
        validate_minutes_consistency(context)
    assert error.value.invariant == invariant


@pytest.mark.parametrize(
    "mutation, invariant",
    [
        ("participant_count", "participant_transcript_count"),
        ("agenda", "agenda_projection"),
        ("topic_source", "topic_projection"),
        ("outcome_source", "validated_outcome_projection"),
        ("vote", "vote_projection"),
        ("evidence", "evidence_items"),
        ("privacy", "privacy_exclusions"),
        ("language", "language_metrics"),
        ("summary", "summary_facts"),
        ("renderer", "renderer_input_digest"),
        ("contract", "publication_contract_revision"),
        ("manifest", "publication_manifest"),
    ],
)
def test_consistency_matrix_fails_with_only_invariant_and_counts(
    tmp_path: Path,
    mutation: str,
    invariant: str,
) -> None:
    context = _context(tmp_path)
    document = context.document
    if mutation == "participant_count":
        counts = document.deterministic_counts.model_copy(
            update={"participant_transcript_entries": 0}
        )
        context = replace(
            context, document=document.model_copy(update={"deterministic_counts": counts})
        )
    elif mutation == "agenda":
        agenda = document.agendas[0].model_copy(update={"status": "ACTIVE"})
        context = replace(context, document=document.model_copy(update={"agendas": (agenda,)}))
    elif mutation == "topic_source":
        topic = document.topics[0].model_copy(update={"source_entry_ids": ("missing",)})
        context = replace(context, document=document.model_copy(update={"topics": (topic,)}))
    elif mutation == "outcome_source":
        context = replace(context, validated_outcomes=())
    elif mutation == "vote":
        context = replace(context, document=document.model_copy(update={"votes": ()}))
    elif mutation == "evidence":
        context = replace(context, observed_evidence_items=())
    elif mutation == "privacy":
        privacy = document.privacy_exclusions.model_copy(update={"transcript_entries": 99})
        context = replace(
            context, document=document.model_copy(update={"privacy_exclusions": privacy})
        )
    elif mutation == "language":
        context = replace(context, observed_language_counts={"en": 2})
    elif mutation == "summary":
        context = replace(context, document=document.model_copy(update={"full_summary": "changed"}))
    elif mutation == "renderer":
        provenance = document.provenance.model_copy(update={"renderer_input_sha256": "0" * 64})
        context = replace(context, document=document.model_copy(update={"provenance": provenance}))
    elif mutation == "contract":
        context = replace(context, publication_contract_revision="unknown")
    else:
        context = replace(context, publication_manifest=())

    with pytest.raises(MinutesConsistencyError) as error:
        validate_minutes_consistency(context)
    assert error.value.invariant == invariant
    assert "synthetic release statement" not in str(error.value)


def test_consistency_rejects_removed_authoritative_speaking_turn(tmp_path: Path) -> None:
    context = _context(tmp_path)
    started = datetime(2026, 8, 27, tzinfo=UTC)
    finished = datetime(2026, 8, 27, 0, 0, 30, tzinfo=UTC)
    turn = MinutesSpeakingTurn(
        agenda_item_id="agenda-1",
        turn_index=0,
        mode="ORDERED",
        speaker_user_ids=(7,),
        speaker_names=("participant (user:7)",),
        planned_seconds=30,
        extension_seconds=0,
        started_at=started,
        finished_at=finished,
        active_intervals=(
            MinutesSpeakingInterval(
                started_at=started,
                ended_at=finished,
                duration_ms=30_000,
            ),
        ),
        pause_intervals=(),
        actual_floor_ms=30_000,
        finish_reason="TIME_EXPIRED",
        finished_early=False,
        skipped=False,
        absent=False,
        overtime_ms=0,
    )
    counts = context.document.deterministic_counts.model_copy(
        update={"controlled_speaking_turns": 1}
    )
    authoritative = context.document.model_copy(
        update={"speaking_turns": (turn,), "deterministic_counts": counts}
    )
    context = replace(
        context,
        document=authoritative.model_copy(update={"speaking_turns": ()}),
        expected_document=authoritative,
    )

    with pytest.raises(MinutesConsistencyError) as error:
        validate_minutes_consistency(context)
    assert error.value.invariant == "speaking_turn_projection"
