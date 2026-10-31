from lingxu.domain.enums import MeetingEventType, MeetingState
from lingxu.domain.meeting_state import forward_state, validate_transition
from lingxu.meeting.events import MeetingEventType as CompatibilityMeetingEventType


def test_meeting_contracts_are_owned_by_the_domain_layer() -> None:
    assert CompatibilityMeetingEventType is MeetingEventType
    assert MeetingEventType.VOTE_CLOSED == "VOTE_CLOSED"
    assert forward_state(MeetingState.CREATED) is MeetingState.AWAITING_PARTICIPANTS
    validate_transition(MeetingState.CREATED, MeetingState.AWAITING_PARTICIPANTS)
