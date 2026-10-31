import pytest

from lingxu.domain.enums import MeetingState
from lingxu.domain.errors import InvalidStateTransitionError
from lingxu.meeting.state import validate_transition


def test_state_validator_accepts_plan_edge_and_rejects_skip() -> None:
    validate_transition(MeetingState.CREATED, MeetingState.AWAITING_PARTICIPANTS)

    with pytest.raises(InvalidStateTransitionError):
        validate_transition(MeetingState.CREATED, MeetingState.OPENING)


def test_paused_session_only_resumes_to_persisted_state() -> None:
    validate_transition(
        MeetingState.PAUSED,
        MeetingState.DISCUSSION_READY,
        resume_state=MeetingState.DISCUSSION_READY,
    )

    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            MeetingState.PAUSED,
            MeetingState.SPEAKER_ACTIVE,
            resume_state=MeetingState.DISCUSSION_READY,
        )


def test_automatic_edges_require_explicit_runtime_authority() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(MeetingState.NEXT_AGENDA_ITEM, MeetingState.AGENDA_ITEM_INTRO)

    validate_transition(
        MeetingState.NEXT_AGENDA_ITEM,
        MeetingState.AGENDA_ITEM_INTRO,
        allow_auto=True,
    )


def test_failed_post_processing_only_recovers_to_published_with_explicit_authority() -> None:
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            MeetingState.FAILED_POST_PROCESSING,
            MeetingState.PUBLISHED,
        )

    validate_transition(
        MeetingState.FAILED_POST_PROCESSING,
        MeetingState.PUBLISHED,
        allow_post_processing_recovery=True,
    )

    with pytest.raises(InvalidStateTransitionError):
        validate_transition(
            MeetingState.FAILED_POST_PROCESSING,
            MeetingState.POST_PROCESSING,
            allow_post_processing_recovery=True,
        )
