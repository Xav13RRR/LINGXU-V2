"""Pure Meeting Session transition validation owned by the domain layer."""

from __future__ import annotations

from itertools import pairwise

from lingxu.domain.enums import MeetingState
from lingxu.domain.errors import InvalidStateTransitionError

_FORWARD_SEQUENCE = (
    MeetingState.CREATED,
    MeetingState.AWAITING_PARTICIPANTS,
    MeetingState.CONSENT_CHECK,
    MeetingState.OPENING,
    MeetingState.AGENDA_OVERVIEW,
    MeetingState.AGENDA_ITEM_INTRO,
    MeetingState.DISCUSSION_READY,
    MeetingState.SPEAKER_ACTIVE,
    MeetingState.DISCUSSION_COMPLETE,
    MeetingState.VOTE_OPEN,
    MeetingState.VOTE_RESULT,
    MeetingState.NEXT_AGENDA_ITEM,
    MeetingState.CLOSING,
    MeetingState.COMPLETED,
    MeetingState.POST_PROCESSING,
    MeetingState.PUBLISHED,
)

TERMINAL_STATES = frozenset(
    {
        MeetingState.PUBLISHED,
        MeetingState.STOPPED,
        MeetingState.FAILED_CAPTURE,
        MeetingState.FAILED_REALTIME_ASR,
        MeetingState.FAILED_TTS,
        MeetingState.FAILED_POST_PROCESSING,
        MeetingState.DELETED,
    }
)

ACTIVE_STATES = frozenset(set(MeetingState) - TERMINAL_STATES)

_INTERRUPTION_STATES = frozenset(
    {
        MeetingState.PAUSED,
        MeetingState.STOPPED,
        MeetingState.INTERRUPTED,
        MeetingState.FAILED_CAPTURE,
        MeetingState.FAILED_REALTIME_ASR,
        MeetingState.FAILED_TTS,
        MeetingState.FAILED_POST_PROCESSING,
    }
)

_FORWARD_TRANSITIONS = {current: following for current, following in pairwise(_FORWARD_SEQUENCE)}
_BACK_TRANSITIONS = {following: current for current, following in pairwise(_FORWARD_SEQUENCE)}

# Explicit system-only edges required by the automatic agenda loop. Keeping these
# separate from the public fixed sequence prevents slash commands from jumping
# across states while still allowing the Runtime to model timed stages faithfully.
_AUTOMATIC_TRANSITIONS = frozenset(
    {
        (MeetingState.AGENDA_OVERVIEW, MeetingState.CLOSING),
        (MeetingState.DISCUSSION_READY, MeetingState.DISCUSSION_COMPLETE),
        (MeetingState.DISCUSSION_COMPLETE, MeetingState.NEXT_AGENDA_ITEM),
        (MeetingState.NEXT_AGENDA_ITEM, MeetingState.AGENDA_ITEM_INTRO),
    }
)


def forward_state(current: MeetingState) -> MeetingState | None:
    """Return the one fixed next state, if any."""

    return _FORWARD_TRANSITIONS.get(current)


def back_state(current: MeetingState) -> MeetingState | None:
    """Return the one fixed prior state without mutating event history."""

    return _BACK_TRANSITIONS.get(current)


def validate_transition(
    current: MeetingState,
    target: MeetingState,
    *,
    resume_state: MeetingState | None = None,
    allow_back: bool = False,
    allow_auto: bool = False,
    allow_post_processing_recovery: bool = False,
) -> None:
    """Reject transitions that are not deterministic plan edges or explicit recovery."""

    if current == target:
        return
    if target is MeetingState.DELETED and current is not MeetingState.DELETED:
        return
    if (
        allow_post_processing_recovery
        and current is MeetingState.FAILED_POST_PROCESSING
        and target is MeetingState.PUBLISHED
    ):
        return
    if current in TERMINAL_STATES:
        raise InvalidStateTransitionError(f"terminal state {current} cannot transition to {target}")
    if current in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
        if resume_state is not None and target is resume_state:
            return
        if target in _INTERRUPTION_STATES - {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
            return
        raise InvalidStateTransitionError(
            f"{current} may only resume to its persisted prior state or terminate"
        )
    if _FORWARD_TRANSITIONS.get(current) is target:
        return
    if allow_back and _BACK_TRANSITIONS.get(current) is target:
        return
    if allow_auto and (current, target) in _AUTOMATIC_TRANSITIONS:
        return
    if target in _INTERRUPTION_STATES:
        return
    raise InvalidStateTransitionError(f"illegal Meeting Session transition: {current} -> {target}")
