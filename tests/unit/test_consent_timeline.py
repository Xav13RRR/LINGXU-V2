"""Unit tests for the deterministic consent-interval projection."""

from datetime import UTC, datetime

import pytest

from lingxu.domain.enums import ConsentKind, MeetingState
from lingxu.domain.models import MeetingEventRecord
from lingxu.meeting.events import MeetingEventType
from lingxu.privacy.consent_timeline import ConsentTimeline


def event(
    event_type: MeetingEventType | str,
    *,
    user: int = 7,
    at: int = 0,
    kinds: tuple[ConsentKind, ...] = (ConsentKind.MINUTES,),
    sequence: int = 1,
) -> MeetingEventRecord:
    """Build one synthetic append-only consent event without Meeting content."""

    return MeetingEventRecord(
        id=f"event-{sequence}",
        session_id="session-1",
        sequence_number=sequence,
        event_type=str(event_type),
        actor_user_id=user,
        source="TEST",
        payload={
            "subject_user_id": user,
            "effective_at_ms": at,
            "consent_kinds": [kind.value for kind in kinds],
        },
        evidence_utterance_ids=(),
        state_before=MeetingState.CONSENT_CHECK,
        state_after=MeetingState.CONSENT_CHECK,
        reversible=False,
        correlation_id=f"correlation-{sequence}",
        occurred_at=datetime(2026, 8, 19, tzinfo=UTC),
    )


def test_consent_timeline_is_half_open_and_supports_regrant() -> None:
    """A withdrawal cutoff excludes the cutoff and later gap until re-granted."""

    timeline = ConsentTimeline.from_events(
        (
            event(MeetingEventType.CONSENT_GRANTED, user=7, at=0, sequence=1),
            event(MeetingEventType.CONSENT_WITHDRAWN, user=7, at=10_000, sequence=2),
            event(MeetingEventType.CONSENT_GRANTED, user=7, at=20_000, sequence=3),
        )
    )

    assert timeline.allows_interval(7, ConsentKind.MINUTES, 0, 9_999)
    assert not timeline.allows_interval(7, ConsentKind.MINUTES, 9_500, 10_500)
    assert not timeline.allows_interval(7, ConsentKind.MINUTES, 10_000, 19_999)
    assert timeline.allows_interval(7, ConsentKind.MINUTES, 20_000, 21_000)


def test_consent_timeline_denies_an_interval_ending_at_withdrawal_cutoff() -> None:
    """A closed transcript must end strictly before the half-open withdrawal cutoff."""

    timeline = ConsentTimeline.from_events(
        (
            event(MeetingEventType.CONSENT_GRANTED, at=0, sequence=1),
            event(MeetingEventType.CONSENT_WITHDRAWN, at=10_000, sequence=2),
        )
    )

    assert not timeline.allows_interval(7, ConsentKind.MINUTES, 0, 10_000)


def test_consent_timeline_denies_a_user_without_this_meetings_grant() -> None:
    """A projection contains no implied consent from a different user or Guild."""

    timeline = ConsentTimeline.from_events((event(MeetingEventType.CONSENT_GRANTED, user=7, at=0),))

    assert not timeline.allows_interval(8, ConsentKind.MINUTES, 0, 1)


def test_consent_timeline_rejects_malformed_or_non_monotonic_payloads() -> None:
    """Malformed purposes and backwards withdrawal cutoffs cannot create eligibility."""

    malformed = event(MeetingEventType.CONSENT_GRANTED)
    malformed.payload["consent_kinds"] = ["UNKNOWN"]
    with pytest.raises(ValueError, match="UNKNOWN"):
        ConsentTimeline.from_events((malformed,))

    with pytest.raises(ValueError, match="not monotonic"):
        ConsentTimeline.from_events(
            (
                event(MeetingEventType.CONSENT_GRANTED, at=10, sequence=1),
                event(MeetingEventType.CONSENT_WITHDRAWN, at=9, sequence=2),
            )
        )


@pytest.mark.parametrize(
    ("events", "message"),
    (
        (
            (event(MeetingEventType.CONSENT_WITHDRAWN, at=100),),
            "unmatched withdrawal",
        ),
        (
            (
                event(MeetingEventType.CONSENT_GRANTED, at=100, sequence=1),
                event(MeetingEventType.CONSENT_GRANTED, at=200, sequence=2),
            ),
            "repeated grant at a different cutoff",
        ),
        (
            (
                event(MeetingEventType.CONSENT_GRANTED, at=100, sequence=1),
                event(MeetingEventType.CONSENT_WITHDRAWN, at=200, sequence=2),
                event(MeetingEventType.CONSENT_WITHDRAWN, at=300, sequence=3),
            ),
            "repeated withdrawal at a different cutoff",
        ),
        (
            (
                event(MeetingEventType.CONSENT_GRANTED, at=100, sequence=1),
                event(MeetingEventType.CONSENT_WITHDRAWN, at=200, sequence=2),
                event(MeetingEventType.CONSENT_GRANTED, at=50, sequence=3),
            ),
            "consent events are not monotonic",
        ),
    ),
)
def test_consent_timeline_rejects_invalid_transitions(
    events: tuple[MeetingEventRecord, ...], message: str
) -> None:
    """Each consent purpose has monotonic timestamps and matching state transitions."""

    with pytest.raises(ValueError, match=message):
        ConsentTimeline.from_events(events)


def test_consent_timeline_treats_identical_duplicate_changes_as_idempotent() -> None:
    """Only repeated transitions at the exact same cutoff leave one interval."""

    timeline = ConsentTimeline.from_events(
        (
            event(MeetingEventType.CONSENT_GRANTED, at=0, sequence=1),
            event(MeetingEventType.CONSENT_GRANTED, at=0, sequence=2),
            event(MeetingEventType.CONSENT_WITHDRAWN, at=10, sequence=3),
            event(MeetingEventType.CONSENT_WITHDRAWN, at=10, sequence=4),
        )
    )

    assert len(timeline.intervals) == 1
    assert timeline.allows_interval(7, ConsentKind.MINUTES, 0, 9)
    assert not timeline.allows_interval(7, ConsentKind.MINUTES, 10, 11)


def test_consent_event_types_define_the_privacy_event_vocabulary() -> None:
    """The append-only event vocabulary includes grant and withdrawal facts."""

    assert MeetingEventType.CONSENT_GRANTED.value == "CONSENT_GRANTED"
    assert MeetingEventType.CONSENT_WITHDRAWN.value == "CONSENT_WITHDRAWN"
