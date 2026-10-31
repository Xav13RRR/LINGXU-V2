from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

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
    normalise_vote_payload,
)
from lingxu.worker.minutes_renderer import MINUTES_RENDERER_REVISION
from lingxu.worker.outcome_merge import merge_outcomes
from lingxu.worker.outcomes import (
    OutcomeCandidate,
    OutcomeType,
    TopicOutcomeSet,
    VerificationStatus,
)


def _decision() -> OutcomeCandidate:
    return OutcomeCandidate(
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


def _document() -> MinutesDocumentV2:
    decision = _decision()
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
        participants=MinutesParticipants(host="host", attendees=("participant",)),
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
        validated_outcomes=(
            TopicOutcomeSet(
                topic_id="topic-001",
                candidates=(decision,),
                extraction_method="deepseek",
                failure_types=(),
            ),
        ),
        decisions=merge_outcomes((decision,)).decisions,
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
        summary_method="deepseek",
        summary_model="deepseek-v4-flash",
        summary_provider="deepseek",
        summary_model_digest="c" * 64,
        summary_policy_identity="d" * 64,
        fallback_reasons=(),
        executive_summary=None,
        full_summary="The release facts were reviewed.",
    )


def test_document_v2_contains_all_evidence_linked_sections() -> None:
    payload = _document().model_dump(mode="json")

    assert payload["schema_version"] == 2
    assert payload["topics"][0]["topic_id"] == "topic-001"
    assert payload["validated_outcomes"][0]["candidates"][0]["id"] == "decision-1"
    assert payload["decisions"][0]["provenance"]["candidate_sources"][0]["source_entry_ids"] == [
        "e1"
    ]
    assert payload["privacy_exclusions"]["transcript_entries"] == 1
    assert payload["canonical_metrics"]["duplicate_entry_count"] == 1
    assert payload["votes"][0]["counts"] == {"yes": 2, "no": 0, "abstain": 1}
    assert payload["summary_method"] == "deepseek"
    assert payload["speaking_turns"] == []
    assert payload["speaking_followups"] == []


def test_speaking_turn_rejects_modified_actual_duration_and_flags() -> None:
    started = datetime(2026, 8, 27, tzinfo=UTC)
    ended = started + timedelta(seconds=12)
    payload = {
        "agenda_item_id": "agenda-1",
        "turn_index": 0,
        "mode": "ORDERED",
        "speaker_user_ids": (7,),
        "speaker_names": ("participant (user:7)",),
        "planned_seconds": 10,
        "extension_seconds": 0,
        "started_at": started,
        "finished_at": ended,
        "active_intervals": (
            MinutesSpeakingInterval(started_at=started, ended_at=ended, duration_ms=12_000),
        ),
        "pause_intervals": (),
        "actual_floor_ms": 12_000,
        "finish_reason": "TIME_EXPIRED",
        "finished_early": False,
        "skipped": False,
        "absent": False,
        "overtime_ms": 2_000,
    }

    assert MinutesSpeakingTurn.model_validate(payload).overtime_ms == 2_000
    with pytest.raises(ValidationError, match="actual speaking-floor duration"):
        MinutesSpeakingTurn.model_validate({**payload, "actual_floor_ms": 11_000})
    with pytest.raises(ValidationError, match="early-finish"):
        MinutesSpeakingTurn.model_validate({**payload, "finished_early": True})


def test_document_v2_reads_legacy_local_llm_provenance() -> None:
    payload = _document().model_dump(mode="python")
    payload["summary_method"] = "local_llm"
    payload["summary_model"] = "qwen3.5:9b"
    payload["summary_provider"] = "ollama"
    payload["validated_outcomes"][0]["extraction_method"] = "local_llm"

    legacy = MinutesDocumentV2.model_validate(payload)

    assert legacy.summary_method == "local_llm"
    assert legacy.validated_outcomes[0].extraction_method == "local_llm"


def test_document_v2_forbids_unknown_fields() -> None:
    payload = _document().model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        MinutesDocumentV2.model_validate(payload)


def test_document_v2_rejects_wrongly_typed_outcome_section() -> None:
    payload = _document().model_dump()
    payload["actions"] = payload["decisions"]

    with pytest.raises(ValidationError, match="actions"):
        MinutesDocumentV2.model_validate(payload)


def test_document_v2_rejects_unverified_outcome_from_final_sections() -> None:
    payload = _document().model_dump()
    payload["decisions"][0]["verification_status"] = VerificationStatus.UNVERIFIED

    with pytest.raises(ValidationError, match="unverified outcome"):
        MinutesDocumentV2.model_validate(payload)


def test_document_v2_strictly_rejects_coercive_scalars() -> None:
    payload = _document().model_dump()
    payload["deterministic_counts"]["agenda_total"] = "1"

    with pytest.raises(ValidationError, match="agenda_total"):
        MinutesDocumentV2.model_validate(payload)


def test_document_v2_revalidates_mutated_nested_models() -> None:
    document = _document()
    object.__setattr__(document.decisions[0], "text", "")

    with pytest.raises(ValidationError, match="text"):
        MinutesDocumentV2.model_validate(document)


def test_document_v2_revalidates_mutated_validated_topic_sets() -> None:
    document = _document()
    object.__setattr__(document.validated_outcomes[0], "topic_id", "")

    with pytest.raises(ValidationError, match="topic_id"):
        MinutesDocumentV2.model_validate(document)


def test_vote_rejects_counts_beyond_eligible_participants() -> None:
    with pytest.raises(ValidationError, match="eligible"):
        MinutesVote(
            source_event_id=9,
            vote_id="vote-1",
            agenda_item_id="agenda-1",
            vote_content="Release?",
            counts={"yes": 2, "no": 1, "abstain": 1},
            eligible_count=3,
            responded_count=4,
            result="yes",
        )


def test_vote_allows_unavailable_eligibility_for_legacy_generic_vote() -> None:
    vote = MinutesVote(
        source_event_id=9,
        vote_id="vote-1",
        agenda_item_id=None,
        vote_content="Release?",
        counts={"approve": 1, "reject": 1},
        eligible_count=None,
        responded_count=2,
        result="tied",
    )

    assert vote.eligible_count is None
    assert vote.counts == {"approve": 1, "reject": 1}


@pytest.mark.parametrize(
    ("vote_content", "vote_duration_seconds"),
    ((None, None), (None, 0), ("Release?", 0)),
)
def test_voting_agenda_preserves_unavailable_or_zero_authoritative_configuration(
    vote_content: str | None,
    vote_duration_seconds: int | None,
) -> None:
    agenda = MinutesAgenda(
        agenda_id="agenda-legacy",
        position=1,
        title="Legacy vote",
        language="en",
        status="COMPLETED",
        duration_minutes=None,
        requires_vote=True,
        vote_content=vote_content,
        vote_duration_seconds=vote_duration_seconds,
        topic_ids=(),
        summary=None,
    )

    assert agenda.vote_content == vote_content
    assert agenda.vote_duration_seconds == vote_duration_seconds


@pytest.mark.parametrize("result", (True, False))
def test_vote_payload_normalises_supported_boolean_result(result: bool) -> None:
    vote = normalise_vote_payload(
        source_event_id=9,
        payload={
            "counts": {"yes": 1, "no": 0},
            "eligible_count": None,
            "responded_count": 1,
            "result": result,
        },
    )

    assert vote.eligible_count is None
    assert vote.result == ("yes" if result else "no")


def test_vote_payload_never_fabricates_missing_eligibility() -> None:
    vote = normalise_vote_payload(
        source_event_id=9,
        payload={
            "counts": {"yes": 1, "no": 1},
            "total": 2,
            "responded_count": 2,
            "result": "tied",
        },
    )

    assert vote.eligible_count is None


def test_vote_payload_maps_legacy_generic_zero_eligibility_sentinel_to_unknown() -> None:
    vote = normalise_vote_payload(
        source_event_id=9,
        payload={
            "counts": {"yes": 1, "no": 1},
            "eligible_count": 0,
            "responded_count": 2,
            "result": "tied",
        },
    )

    assert vote.eligible_count is None


@pytest.mark.parametrize(
    "payload",
    (
        {"counts": {"yes": True}, "responded_count": 1},
        {"counts": {"yes": 1}, "eligible_count": "1", "responded_count": 1},
        {"counts": {"yes": 1}, "eligible_count": 1, "responded_count": True},
        {"counts": {"yes": 1}, "eligible_count": 1, "responded_count": 1, "result": 7},
        {"counts": {"yes": 1}, "eligible_count": 1, "responded_count": 1, "vote_id": False},
    ),
)
def test_vote_payload_rejects_malformed_scalar_types(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="vote"):
        normalise_vote_payload(source_event_id=9, payload=payload)
