import pytest
from pydantic import ValidationError

from lingxu.transcript.models import CanonicalTranscriptEntry
from lingxu.worker.evidence_linker import (
    EvidenceContext,
    SpeechActReductionCache,
    link_outcome_evidence,
    validate_outcome,
)
from lingxu.worker.outcomes import (
    OutcomeCandidate,
    OutcomeEventFact,
    OutcomeType,
    VerificationStatus,
)
from lingxu.worker.topics import TopicSegment


def _entry(
    entry_id: str,
    text: str,
    *,
    user_id: int = 7,
    speaker_name: str = "participant",
) -> CanonicalTranscriptEntry:
    return CanonicalTranscriptEntry(
        entry_id=entry_id,
        source_entry_ids=(f"source-{entry_id}",),
        speaker_user_id=user_id,
        speaker_name=speaker_name,
        started_ms=1_000,
        ended_ms=1_800,
        raw_texts=(text,),
        text=text,
        entry_type="statement",
        language="en",
        confidence=0.95,
        utterance_group_id=f"group-{entry_id}",
        normalisation_changes=(),
        asr_tags=(),
        emotion=None,
    )


def _context(
    *entries: CanonicalTranscriptEntry,
    events: tuple[OutcomeEventFact, ...] = (),
) -> EvidenceContext:
    return EvidenceContext(
        topic=TopicSegment(
            "topic-001",
            "Topic 1",
            tuple(entry.entry_id for entry in entries),
            1_000,
            1_800,
            "deterministic_fallback",
        ),
        canonical_entries=entries,
        authoritative_events=events,
        allowed_participant_ids=frozenset({7, 8}),
        participant_names={7: "participant", 8: "reviewer"},
    )


def _candidate(**changes: object) -> OutcomeCandidate:
    values: dict[str, object] = {
        "id": "o1",
        "type": OutcomeType.ACTION_ITEM,
        "text": "prepare release",
        "owner_user_id": 7,
        "owner_name": "participant",
        "deadline": None,
        "source_entry_ids": ("e1",),
        "source_event_ids": (),
        "confidence": 0.9,
        "requires_confirmation": False,
        "verification_status": VerificationStatus.EVIDENCE_LINKED,
    }
    values.update(changes)
    return OutcomeCandidate(**values)


def _event(
    *,
    sequence_number: int = 11,
    topic_id: str | None = "topic-001",
    result: str = "accepted",
    subject: str = "release",
    source_entry_ids: tuple[str, ...] = ("e1",),
) -> OutcomeEventFact:
    return OutcomeEventFact(
        sequence_number=sequence_number,
        topic_id=topic_id,
        subject=subject,
        result=result,
        source_entry_ids=source_entry_ids,
    )


def test_direct_owner_action_with_topic_evidence_is_valid() -> None:
    context = _context(_entry("e1", "I will prepare release"))

    assert validate_outcome(_candidate(), context).valid


def test_unknown_source_is_rejected() -> None:
    result = validate_outcome(
        _candidate(source_entry_ids=("fabricated",)),
        _context(_entry("e1", "I will prepare release")),
    )

    assert not result.valid
    assert "unknown_source_entry" in result.error_types


def test_known_source_outside_topic_is_rejected() -> None:
    e1 = _entry("e1", "I will prepare release")
    e2 = _entry("e2", "another topic")
    context = EvidenceContext(
        topic=TopicSegment("topic-001", "Topic 1", ("e1",), 1_000, 1_800, "deterministic_fallback"),
        canonical_entries=(e1, e2),
        authoritative_events=(),
        allowed_participant_ids=frozenset({7}),
        participant_names={7: "participant"},
    )

    result = validate_outcome(_candidate(source_entry_ids=("e2",)), context)

    assert not result.valid
    assert "source_outside_topic" in result.error_types


def test_owner_outside_participants_is_rejected() -> None:
    result = validate_outcome(
        _candidate(owner_user_id=99, owner_name="outsider"),
        _context(_entry("e1", "I will prepare release")),
    )

    assert not result.valid
    assert "owner_not_allowed" in result.error_types


def test_deadline_must_occur_verbatim_in_cited_evidence() -> None:
    result = validate_outcome(
        _candidate(deadline="Friday"),
        _context(_entry("e1", "I will prepare release")),
    )

    assert not result.valid
    assert "unsupported_deadline" in result.error_types


def test_proposal_cannot_be_promoted_to_decision() -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", "I propose that we prepare release")),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "text",
    (
        "approved plan",
        "accepted format",
        "approved proposal",
        "accepted deployment",
        "approved decision",
        "approved yesterday",
        "Alice approved",
        "已批准的方案",
    ),
)
def test_bare_descriptive_acceptance_cannot_validate_a_decision(text: str) -> None:
    context = _context(_entry("e1", text))
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text="plan",
        owner_user_id=None,
        owner_name=None,
    )

    linked = link_outcome_evidence(candidate, context)
    result = validate_outcome(linked, context)

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize("text", ("We agreed not", "我们同意不"))
def test_unresolved_post_acceptance_direction_invalidates_decision(text: str) -> None:
    context = _context(_entry("e1", text))
    candidate = link_outcome_evidence(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy",
            owner_user_id=None,
            owner_name=None,
        ),
        context,
    )

    result = validate_outcome(candidate, context)

    assert not result.valid
    assert "semantic_ambiguous" in result.error_types
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    ("evidence_text", "decision_text"),
    (
        ("We decided to stop the deployment", "stop the deployment"),
        ("We agreed to revoke the temporary permission", "revoke the temporary permission"),
        ("We decided to terminate the legacy service", "terminate the legacy service"),
        ("我们决定停止部署", "停止部署"),
        ("我们同意撤销临时权限", "撤销临时权限"),
        ("我们决定终止旧服务", "终止旧服务"),
    ),
)
def test_accepted_object_level_reversal_action_remains_a_valid_decision(
    evidence_text: str,
    decision_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text=decision_text,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    ("evidence_text", "decision_text"),
    (
        ("We decided not to deploy the release", "not deploy the release"),
        ("We agreed not to enable the temporary permission", "not enable the permission"),
        ("我们决定不部署新版本", "不部署新版本"),
        ("我们同意不启用临时权限", "不启用临时权限"),
    ),
)
def test_accepted_negative_action_remains_a_valid_decision(
    evidence_text: str,
    decision_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text=decision_text,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We did not, after discussion, agree to deploy the release",
        "We did not \u2014 after discussion \u2014 agree to deploy the release",
        "\u6211\u4eec\u672a\u5728\u8ba8\u8bba\u540e\u51b3\u5b9a\u90e8\u7f72\u65b0\u7248\u672c",
        "\u6211\u4eec\u4e0d\u5728\u8ba8\u8bba\u540e\u540c\u610f\u542f\u7528\u4e34\u65f6\u6743\u9650",
    ),
)
def test_negated_acceptance_survives_neutral_bounded_clause_insertions(
    evidence_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    ("evidence_text", "decision_text"),
    (
        ("We decided, after discussion, not to deploy the release", "not deploy the release"),
        (
            "\u6211\u4eec\u51b3\u5b9a\u5728\u8ba8\u8bba\u540e\u4e0d\u90e8\u7f72\u65b0\u7248\u672c",
            "\u4e0d\u90e8\u7f72\u65b0\u7248\u672c",
        ),
    ),
)
def test_directional_negation_after_acceptance_remains_an_accepted_negative_action(
    evidence_text: str,
    decision_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text=decision_text,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We did not discuss the rollout. We decided to deploy the release",
        "\u6211\u4eec\u672a\u8ba8\u8bba\u53d1\u5e03\uff1b\u6211\u4eec\u51b3\u5b9a\u90e8\u7f72\u65b0\u7248\u672c",
    ),
)
def test_negation_cannot_cross_a_strong_clause_boundary(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We did not only agree to deploy, we also approved the release",
        "\u6211\u4eec\u4e0d\u4ec5\u8ba8\u8bba\u4e86\u53d1\u5e03\uff0c\u4e5f\u51b3\u5b9a\u90e8\u7f72\u65b0\u7248\u672c",
    ),
)
def test_non_negating_idioms_do_not_create_broad_false_positives(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We did not discuss the rollout, and after review agreed to deploy the release",
        "\u6211\u4eec\u672a\u8ba8\u8bba\u7ec6\u8282\uff0c\u4e4b\u540e\u51b3\u5b9a\u90e8\u7f72\u65b0\u7248\u672c",
    ),
)
def test_ambiguous_negation_inside_one_bounded_clause_fails_closed(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "evidence_text",
    (
        (
            "None of the voting members, after a careful review of all technical and "
            "operational evidence presented during the meeting, agreed to deploy the release"
        ),
        (
            "No participant \u2014 after the chair presented every security, operations, and "
            "rollback consideration in detail \u2014 decided to deploy the release"
        ),
    ),
)
def test_negative_quantifier_subject_governs_acceptance_across_neutral_insertions(
    evidence_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


def test_negative_object_in_prior_strong_clause_does_not_negate_later_acceptance() -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", "No objections were raised; the group agreed to deploy the release")),
    )

    assert result.valid


@pytest.mark.parametrize(
    ("evidence_text", "decision_text"),
    (
        ("We decided not to reject the release proposal", "not reject the release proposal"),
        ("We agreed not to deny the deployment request", "not deny the deployment request"),
        ("We decided not to refuse the approved release", "not refuse the approved release"),
        ("We agreed not to prohibit deployment", "not prohibit deployment"),
        ("We agreed not to withdraw the release proposal", "not withdraw the release proposal"),
        (
            "\u6211\u4eec\u51b3\u5b9a\u4e0d\u62d2\u7edd\u53d1\u5e03\u65b9\u6848",
            "\u4e0d\u62d2\u7edd\u53d1\u5e03\u65b9\u6848",
        ),
        (
            "\u6211\u4eec\u540c\u610f\u4e0d\u7981\u6b62\u90e8\u7f72",
            "\u4e0d\u7981\u6b62\u90e8\u7f72",
        ),
        (
            "\u6211\u4eec\u51b3\u5b9a\u4e0d\u64a4\u56de\u53d1\u5e03\u65b9\u6848",
            "\u4e0d\u64a4\u56de\u53d1\u5e03\u65b9\u6848",
        ),
    ),
)
def test_accepted_negated_rejection_action_remains_a_valid_decision(
    evidence_text: str,
    decision_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text=decision_text,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    "evidence_text",
    (
        "The release proposal was rejected",
        "The deployment request was denied",
        "The release proposal was refused",
        "Deployment was prohibited",
        "\u53d1\u5e03\u65b9\u6848\u88ab\u62d2\u7edd",
        "\u8bf7\u6c42\u88ab\u5426\u51b3",
        "\u53d1\u5e03\u88ab\u7981\u6b62",
    ),
)
def test_affirmative_rejection_actions_remain_invalid(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "evidence_text",
    (
        "The group did not reject the proposal; the group agreed to deploy the release",
        "The group did not deny the request. The group decided to deploy the release",
        "\u6211\u4eec\u6ca1\u6709\u62d2\u7edd\u65b9\u6848\uff1b\u6211\u4eec\u540c\u610f\u90e8\u7f72\u65b0\u7248\u672c",
        "\u6211\u4eec\u672a\u7981\u6b62\u90e8\u7f72\u3002\u6211\u4eec\u51b3\u5b9a\u53d1\u5e03\u65b0\u7248\u672c",
    ),
)
def test_negated_rejection_does_not_contaminate_later_strong_clause_acceptance(
    evidence_text: str,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


@pytest.mark.parametrize(
    ("evidence_text", "decision_text", "expected_valid"),
    (
        ("We did not, after discussion, agree to deploy", "deploy", False),
        ("We did not \u2014 after discussion \u2014 decide to deploy", "deploy", False),
        ("We did not agree to deploy", "deploy", False),
        ("We haven't agreed to deploy", "deploy", False),
        ("None of us agreed to deploy", "deploy", False),
        ("We did not only agree to deploy, we also approved it", "deploy", True),
        ("We decided to stop the legacy service", "stop the legacy service", True),
        ("We agreed to revoke the temporary permission", "revoke the permission", True),
        ("We decided to terminate the legacy service", "terminate the service", True),
        (
            "\u6211\u4eec\u672a\u5728\u8ba8\u8bba\u540e\u51b3\u5b9a\u90e8\u7f72",
            "\u90e8\u7f72",
            False,
        ),
        (
            "\u6211\u4eec\u4e0d\u4ec5\u8ba8\u8bba\uff0c\u4e5f\u51b3\u5b9a\u90e8\u7f72",
            "\u90e8\u7f72",
            True,
        ),
        (
            "\u6211\u4eec\u51b3\u5b9a\u505c\u6b62\u65e7\u670d\u52a1",
            "\u505c\u6b62\u65e7\u670d\u52a1",
            True,
        ),
        (
            "\u6211\u4eec\u540c\u610f\u64a4\u9500\u4e34\u65f6\u6743\u9650",
            "\u64a4\u9500\u4e34\u65f6\u6743\u9650",
            True,
        ),
        (
            "\u6211\u4eec\u51b3\u5b9a\u7ec8\u6b62\u65e7\u670d\u52a1",
            "\u7ec8\u6b62\u65e7\u670d\u52a1",
            True,
        ),
    ),
)
def test_directional_speech_act_regression_matrix(
    evidence_text: str,
    decision_text: str,
    expected_valid: bool,
) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text=decision_text,
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid is expected_valid


def test_speech_act_refactor_preserves_attribution_and_vote_event_authority() -> None:
    attributed = _candidate(
        type=OutcomeType.RISK,
        text="certificate validity",
        owner_user_id=None,
        owner_name=None,
    )
    linked = link_outcome_evidence(
        attributed,
        _context(_entry("e1", "The supplier reported that the certificate is valid")),
    )
    assert linked.verification_status is VerificationStatus.NOT_CHECKED
    assert linked.requires_confirmation is True

    accepted_event = _event(result="accepted")
    accepted = _candidate(
        type=OutcomeType.DECISION,
        text="release",
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(accepted_event.sequence_number,),
    )
    rejected_event = _event(result="rejected")
    rejected = accepted.model_copy(update={"source_event_ids": (rejected_event.sequence_number,)})

    assert validate_outcome(
        accepted,
        _context(_entry("e1", "discussion"), events=(accepted_event,)),
    ).valid
    rejected_result = validate_outcome(
        rejected,
        _context(_entry("e1", "discussion"), events=(rejected_event,)),
    )
    assert not rejected_result.valid
    assert "decision_not_accepted" in rejected_result.error_types


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We did not decide to deploy the release",
        "We never agreed to enable the permission",
        "None of us agreed to deploy the release",
        "The release wasn't approved",
        "The proposal was not accepted",
        "The proposal was rejected",
        "我们未决定部署新版本",
        "我们不同意启用临时权限",
        "提案未获通过",
        "提案被否决",
    ),
)
def test_negative_or_rejected_speech_act_remains_invalid(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy the release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "text",
    (
        "We did not agree to release",
        "The release was not approved",
        "The release was never approved",
        "We haven't agreed to release",
        "The release is not yet approved",
        "The release remains pending",
        "The group cannot agree to release",
        "The release proposal was rejected",
        "We agreed earlier, but they reject the proposal",
        "The vote failed",
        "We agreed earlier, but the proposal may fail",
        "We agreed but the final result was a failure",
        "We agreed before the request was denied",
        "We agreed earlier, but they prohibit the release",
        "We agreed before release was prohibited",
        "We agreed earlier, but the proposal was voted down",
        "We agreed earlier, but the proposal was defeated",
        "We agreed earlier, but the proposal was disapproved",
        "We agreed earlier, but the proposal was vetoed",
        "We agreed earlier, but the proposal was overruled",
        "We agreed earlier, but the proposal lost",
        "We agreed earlier, but they withdraw the proposal",
        "We agreed earlier, but the proposal was withdrawn",
        "We agreed earlier, but they cancel the proposal",
        "我们先前同意, 但最终投票否决",
        "我们先前同意, 但方案被撤回",
        "大家不同意发布",
        "发布方案未通过",
        "发布方案不得通过",
        "发布方案不予通过",
        "发布方案未获批准",
        "大家未确认发布",
        "发布方案被否决",
    ),
)
def test_negated_or_rejected_language_never_accepts_a_decision(text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "marker",
    (
        "denial",
        "denials",
        "disapproval",
        "disapprovals",
        "loss",
        "losses",
        "withdrawal",
        "withdrawals",
    ),
)
def test_rejection_nominal_morphology_overrides_earlier_acceptance(marker: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", f"We agreed earlier, but the {marker} was final")),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "text",
    (
        "已同意\uff0c但最后是denial。",
        "已同意\uff0c但withdrawal是最终结果",
        "已同意\uff0c但最后是DENIALS。",
        "已同意\uff0c但WITHDRAWALS是最终结果",
        "已同意\uff0c但最后是DISAPPROVALS。",
        "已同意\uff0c但LOSSES是最终结果",
    ),
)
def test_rejection_morphology_detects_latin_words_adjacent_to_cjk(text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "word",
    (
        "denialist",
        "antidisapprovalism",
        "lossless",
        "withdrawalism",
    ),
)
def test_rejection_morphology_does_not_match_inside_longer_latin_words(word: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", f"已同意采用{word}分类")),
    )

    assert result.valid


def test_rejection_morphology_does_not_match_inside_ordinary_words() -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="release",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", "We agreed to use lossless compression for the glossary")),
    )

    assert result.valid


@pytest.mark.parametrize(
    ("event", "expected_valid"),
    (
        (_event(result="rejected"), False),
        (_event(result="unknown"), False),
        (_event(result="pending"), False),
        (_event(topic_id="topic-else"), False),
        (_event(subject="unrelated budget"), False),
        (_event(source_entry_ids=("outside-topic",)), False),
        (_event(topic_id=None), False),
        (_event(), True),
    ),
)
def test_vote_decision_support_is_explicit_topic_bound_and_subject_matched(
    event: OutcomeEventFact,
    expected_valid: bool,
) -> None:
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text="release",
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(event.sequence_number,),
    )

    result = validate_outcome(candidate, _context(_entry("e1", "discussion"), events=(event,)))

    assert result.valid is expected_valid
    if not expected_valid:
        assert "decision_not_accepted" in result.error_types


def test_vote_subject_is_derived_only_from_rendered_candidate_text() -> None:
    event = _event(subject="release")
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text="unrelated decision",
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(event.sequence_number,),
    )

    result = validate_outcome(candidate, _context(_entry("e1", "discussion"), events=(event,)))

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


def test_model_cannot_forge_a_separate_vote_subject() -> None:
    with pytest.raises(ValidationError):
        _candidate(
            type=OutcomeType.DECISION,
            text="unrelated decision",
            subject="release",
            owner_user_id=None,
            owner_name=None,
        )


def test_vote_exact_rendered_text_with_affirmative_result_is_valid() -> None:
    event = _event(subject="release")
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text="release",
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(event.sequence_number,),
    )

    assert validate_outcome(
        candidate,
        _context(_entry("e1", "discussion"), events=(event,)),
    ).valid


def test_vote_event_cannot_be_the_evidence_for_a_nondecision_outcome() -> None:
    event = _event()
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="release risk",
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(event.sequence_number,),
    )

    result = validate_outcome(candidate, _context(_entry("e1", "discussion"), events=(event,)))

    assert not result.valid
    assert "event_evidence_not_allowed" in result.error_types
    assert "transcript_evidence_required" in result.error_types


@pytest.mark.parametrize(
    "evidence_text",
    (
        "We discussed an approved plan",
        "We reviewed an accepted format",
        "A consensus proposal was discussed",
        "讨论已批准的方案",
        "审阅通过的提案",
        "讨论共识方案",
    ),
)
def test_descriptive_acceptance_cannot_authorise_a_decision(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "evidence_text",
    (
        "The group reached consensus to deploy",
        "Consensus was reached on deployment",
        "Agreed to deploy",
        "团队达成共识并部署",
        "已同意部署",
    ),
)
def test_closed_acceptance_frames_authorise_transcript_decision(evidence_text: str) -> None:
    result = validate_outcome(
        _candidate(
            type=OutcomeType.DECISION,
            text="deploy",
            owner_user_id=None,
            owner_name=None,
        ),
        _context(_entry("e1", evidence_text)),
    )

    assert result.valid


def test_event_authority_bypasses_all_speech_scanning(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(subject='Synthetic Person " unmatched', result="accepted")
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text='Synthetic Person " unmatched',
        owner_user_id=None,
        owner_name=None,
        source_entry_ids=(),
        source_event_ids=(event.sequence_number,),
    )

    def fail_scan(text: str) -> object:
        raise AssertionError(f"speech scan should not run: {len(text)}")

    monkeypatch.setattr("lingxu.worker.evidence_linker.classify_speech_acts", fail_scan)

    context = _context(_entry("e1", "positive looking text"), events=(event,))
    linked = link_outcome_evidence(candidate, context)
    assert linked.verification_status is VerificationStatus.EVIDENCE_LINKED
    assert validate_outcome(linked, context).valid


def test_failed_event_binding_has_no_transcript_or_claim_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _event(subject="other", result="rejected")
    candidate = _candidate(
        type=OutcomeType.DECISION,
        text="deploy",
        owner_user_id=None,
        owner_name=None,
        source_event_ids=(event.sequence_number,),
    )

    def fail_scan(text: str) -> object:
        raise AssertionError(f"speech scan should not run: {len(text)}")

    monkeypatch.setattr("lingxu.worker.evidence_linker.classify_speech_acts", fail_scan)
    result = validate_outcome(
        candidate,
        _context(_entry("e1", "We agreed to deploy"), events=(event,)),
    )

    assert not result.valid
    assert "decision_not_accepted" in result.error_types


@pytest.mark.parametrize(
    "outcome_type",
    (
        OutcomeType.ACTION_ITEM,
        OutcomeType.OPEN_QUESTION,
        OutcomeType.PROPOSAL,
        OutcomeType.RISK,
        OutcomeType.DISAGREEMENT,
    ),
)
def test_every_semantic_path_fails_closed_on_quote_error(outcome_type: OutcomeType) -> None:
    candidate = _candidate(
        type=outcome_type,
        text="synthetic item",
        owner_user_id=None,
        owner_name=None,
    )
    result = validate_outcome(candidate, _context(_entry("e1", '"unclosed')))

    assert not result.valid
    assert "semantic_quote_mismatch" in result.error_types


def test_reduction_cache_scans_unique_entry_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import lingxu.worker.evidence_linker as module

    real_classifier = module.classify_speech_acts
    calls = 0

    def counted(text: str) -> object:
        nonlocal calls
        calls += 1
        return real_classifier(text)

    monkeypatch.setattr(module, "classify_speech_acts", counted)
    context = _context(_entry("e1", "We agreed to deploy"))
    cache = SpeechActReductionCache(context)

    first = cache.reduce_entry(context.canonical_entries[0])
    second = cache.reduce_entry(context.canonical_entries[0])

    assert first == second
    assert calls == 1
    assert cache.entry_misses == 1
    assert cache.entry_hits == 1
    assert cache.classified_codepoints == len(context.canonical_entries[0].text)


def test_topic_cache_builds_structural_maps_once_and_reduces_each_cited_entry_once() -> None:
    entries = tuple(_entry(f"e{index}", f"statement {index}") for index in range(12))
    context = _context(*entries)
    cache = SpeechActReductionCache(context)

    for index in range(80):
        candidate = _candidate(
            id=f"candidate-{index}",
            text=f"prepare release {index}",
            owner_user_id=None,
            owner_name=None,
            source_entry_ids=tuple(entry.entry_id for entry in entries),
        )
        linked = link_outcome_evidence(candidate, context, cache)
        assert validate_outcome(linked, context, cache).valid

    assert cache.structural_map_builds == 1
    assert cache.entry_misses == 12
    assert cache.candidate_misses == 80
    assert cache.entry_hits > cache.entry_misses


def test_candidate_external_claim_text_cannot_self_authorize_not_checked_status() -> None:
    context = _context(_entry("e1", "The release has a risk"))
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="The vendor said the release has a risk",
        owner_user_id=None,
        owner_name=None,
    )

    linked = link_outcome_evidence(candidate, context)

    assert linked.verification_status is VerificationStatus.EVIDENCE_LINKED
    assert not linked.requires_confirmation
    assert validate_outcome(linked, context).valid


def test_duplicate_canonical_entry_ids_fail_structurally() -> None:
    duplicate = _entry("e1", "another statement")
    context = _context(_entry("e1", "statement"), duplicate)
    result = validate_outcome(_candidate(), context)

    assert not result.valid
    assert "duplicate_source_entry" in result.error_types


def test_third_party_claim_cannot_be_presented_as_verified_fact() -> None:
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="supplier certification may be invalid",
        owner_user_id=None,
        owner_name=None,
    )
    context = _context(_entry("e1", "The vendor claims that the supplier is certified"))

    raw_validation = validate_outcome(candidate, context)
    linked = link_outcome_evidence(candidate, context)

    assert not raw_validation.valid
    assert "speaker_claim_not_marked" in raw_validation.error_types
    assert linked.verification_status is VerificationStatus.NOT_CHECKED
    assert linked.requires_confirmation is True
    assert validate_outcome(linked, context).valid


@pytest.mark.parametrize(
    "text",
    (
        "The vendor says the certificate is valid",
        "The vendor said the certificate is valid",
        "The vendor claims the certificate is valid",
        "The vendor claimed the certificate is valid",
        "The vendor tells us the certificate is valid",
        "The vendor told us the certificate is valid",
        "The vendor reports the certificate is valid",
        "The vendor reported the certificate is valid",
        "The vendor confirms the certificate is valid",
        "The vendor confirmed the certificate is valid",
        "The vendor is saying the certificate is valid",
        "The vendor is telling us the certificate is valid",
        "The vendor is claiming the certificate is valid",
        "The vendor is reporting the certificate is valid",
        "The vendor is confirming the certificate is valid",
        "They are reporting the certificate is valid",
        "Our vendor is progressively reporting the certificate is valid",
        "According to the vendor, the certificate is valid",
        "According to our vendor, the certificate is valid",
        "It was reported that the certificate is valid",
        "The certificate is reportedly valid",
        "供应商说证书有效",
        "供应商声称证书有效",
        "供应商告诉我们证书有效",
        "据供应商称证书有效",
        "据说证书有效",
        "听说证书有效",
        "供应商报道称证书有效",
        "供应商正在报告证书有效",
        "供应商正在确认该证书有效",
        "他们表示证书有效",
        "我的供应商报告证书有效",
        "我们的供应商正在报告证书有效",
        "我司报告证书有效",
        "我同事报告证书有效",
        "我们同事报告证书有效",
        "我经理报告证书有效",
        "我们主管报告证书有效",
        "我朋友说证书有效",
        "我们律师确认该证书有效",
        "我同事明确地报告证书有效",
        "我们经理正式地表示证书有效",
        "我律师如实地确认该证书有效",
        "我们供应商清楚地报告证书有效",
        "我项目顾问明确地报告证书有效",
    ),
)
def test_third_party_attribution_variants_are_not_checked(text: str) -> None:
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="certificate validity",
        owner_user_id=None,
        owner_name=None,
    )

    linked = link_outcome_evidence(candidate, _context(_entry("e1", text)))

    assert linked.verification_status is VerificationStatus.NOT_CHECKED
    assert linked.requires_confirmation is True


@pytest.mark.parametrize(
    "text",
    (
        "I say the release has a risk",
        "I said the release has a risk",
        "I am saying the release has a risk",
        "I am telling this meeting about the release risk",
        "I am claiming the release risk",
        "I am reporting the release risk",
        "I am confirming the release risk",
        "I confirmed the release risk",
        "I personally confirm the release risk",
        "I will report the release risk",
        "I clearly said the release has a risk",
        "We report a release risk to this meeting",
        "We are reporting a release risk to this meeting",
        "We directly reported a release risk to this meeting",
        "我说发布存在风险",
        "我正在报告发布存在风险",
        "我明确报告发布存在风险",
        "我明确地报告发布存在风险",
        "我们正式地表示发布存在风险",
        "本人如实地确认发布存在风险",
        "我会明确地报告发布存在风险",
        "我们应该正式地表示发布存在风险",
        "本人已经如实地确认发布存在风险",
        "本人正在确认发布存在风险",
        "我们报告发布存在风险",
        "我们将报告发布存在风险",
        "我们向会议报告发布存在风险",
    ),
)
def test_direct_first_person_meeting_statements_are_not_external_claims(text: str) -> None:
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="release risk",
        owner_user_id=None,
        owner_name=None,
    )

    linked = link_outcome_evidence(candidate, _context(_entry("e1", text)))

    assert linked.verification_status is VerificationStatus.EVIDENCE_LINKED
    assert linked.requires_confirmation is False


def test_vote_fact_cannot_verify_an_attributed_nondecision_claim() -> None:
    event = _event(subject="certificate validity")
    candidate = _candidate(
        type=OutcomeType.RISK,
        text="certificate validity",
        owner_user_id=None,
        owner_name=None,
        source_event_ids=(event.sequence_number,),
    )
    context = _context(
        _entry("e1", "The vendor claims the certificate is valid"),
        events=(event,),
    )

    linked = link_outcome_evidence(candidate, context)

    assert linked.verification_status is VerificationStatus.NOT_CHECKED
    assert linked.requires_confirmation is True
    validation = validate_outcome(linked, context)
    assert not validation.valid
    assert "event_evidence_not_allowed" in validation.error_types


def test_indirect_owner_assignment_is_explicitly_unconfirmed() -> None:
    candidate = _candidate(owner_user_id=8, owner_name="reviewer")
    context = _context(_entry("e1", "reviewer will prepare release"))

    linked = link_outcome_evidence(candidate, context)

    assert linked.requires_confirmation is True
    assert linked.verification_status is VerificationStatus.UNCONFIRMED
    assert validate_outcome(linked, context).valid
