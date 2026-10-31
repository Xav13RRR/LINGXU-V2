from datetime import UTC, datetime, timedelta

import pytest

from lingxu.domain.enums import MeetingEventType, MeetingState
from lingxu.domain.models import MeetingEventRecord
from lingxu.worker.speaking_facts import project_authoritative_speaking_facts


def _event(
    sequence: int,
    event_type: MeetingEventType,
    at: datetime,
    payload: dict[str, object],
) -> MeetingEventRecord:
    return MeetingEventRecord(
        id=f"event-{sequence}",
        session_id="session-1",
        sequence_number=sequence,
        event_type=event_type.value,
        actor_user_id=None,
        source="TEST",
        payload=payload,
        evidence_utterance_ids=(),
        state_before=MeetingState.DISCUSSION_READY,
        state_after=MeetingState.DISCUSSION_READY,
        reversible=False,
        correlation_id=f"correlation-{sequence}",
        occurred_at=at,
    )


def test_projects_active_time_pauses_extension_absence_and_followup() -> None:
    baseline = datetime(2026, 8, 27, 1, 0, tzinfo=UTC)
    events = (
        _event(
            1,
            MeetingEventType.SPEAKING_FLOW_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-1",
                "mode": "ORDERED",
                "turns": [
                    {"user_id": 100, "planned_seconds": 60},
                    {"user_id": 200, "planned_seconds": 45},
                ],
                "presenter_user_ids": [],
                "identities": [
                    {"user_id": 100, "display_name": "Host"},
                    {"user_id": 200, "display_name": "Alice"},
                ],
                "followup_mode": "SUPPLEMENT_TURNS",
                "prep_seconds": 10,
                "presentation_seconds": None,
            },
        ),
        _event(
            2,
            MeetingEventType.SPEAKING_TURN_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 0,
                "current_user_ids": [100],
                "next_user_ids": [200],
                "planned_seconds": 60,
            },
        ),
        _event(
            3,
            MeetingEventType.TIMER_STARTED,
            baseline,
            {
                "timer_id": "timer-1",
                "timer_type": "FLOW:SPEAKING_TURN:agenda-1:0",
                "deadline_at": (baseline + timedelta(seconds=60)).isoformat(),
                "duration_seconds": 60,
            },
        ),
        _event(
            4,
            MeetingEventType.TIMER_PAUSED,
            baseline + timedelta(seconds=10),
            {"timer_id": "timer-1", "remaining_ms": 50_000},
        ),
        _event(
            5,
            MeetingEventType.TIMER_RESUMED,
            baseline + timedelta(seconds=30),
            {"timer_id": "timer-1", "deadline_at": baseline.isoformat()},
        ),
        _event(
            6,
            MeetingEventType.SPEAKING_TURN_EXTENDED,
            baseline + timedelta(seconds=40),
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 0,
                "current_user_ids": [100],
                "additional_seconds": 30,
            },
        ),
        _event(
            7,
            MeetingEventType.TIMER_PAUSED,
            baseline + timedelta(seconds=50),
            {
                "timer_id": "timer-1",
                "timer_type": "FLOW:SPEAKING_TURN:agenda-1:0",
                "remaining_ms": 40_000,
                "reason": "HOST_INTERVENTION",
            },
        ),
        _event(
            8,
            MeetingEventType.TIMER_RESUMED,
            baseline + timedelta(seconds=60),
            {
                "timer_id": "timer-1",
                "timer_type": "FLOW:SPEAKING_TURN:agenda-1:0",
                "deadline_at": baseline.isoformat(),
                "reason": "HOST_INTERVENTION_ENDED",
            },
        ),
        _event(
            9,
            MeetingEventType.SPEAKING_TURN_FINISHED,
            baseline + timedelta(seconds=80),
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 0,
                "current_user_ids": [100],
                "planned_seconds": 60,
                "extension_seconds": 30,
                "reason": "HOST_ENDED_EARLY",
            },
        ),
        _event(
            10,
            MeetingEventType.SPEAKING_TURN_FINISHED,
            baseline + timedelta(seconds=81),
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 1,
                "current_user_ids": [200],
                "planned_seconds": 45,
                "reason": "ABSENT",
            },
        ),
        _event(
            11,
            MeetingEventType.SPEAKER_SELECTED,
            baseline + timedelta(seconds=81, milliseconds=500),
            {"user_id": 200},
        ),
        _event(
            12,
            MeetingEventType.SPEAKING_FOLLOWUP_STARTED,
            baseline + timedelta(seconds=82),
            {"agenda_item_id": "agenda-1", "followup_mode": "SUPPLEMENT_TURNS"},
        ),
        _event(
            13,
            MeetingEventType.SPEAKER_FINISHED,
            baseline + timedelta(seconds=100),
            {"user_id": 200},
        ),
        _event(
            14,
            MeetingEventType.SPEAKING_FOLLOWUP_FINISHED,
            baseline + timedelta(seconds=105),
            {"agenda_item_id": "agenda-1", "followup_mode": "SUPPLEMENT_TURNS"},
        ),
    )

    facts = project_authoritative_speaking_facts(events, labels={100: "New Host"})

    assert len(facts.turns) == 2
    first = facts.turns[0]
    assert first.speaker_names == ("Host (user:100)",)
    assert first.extension_seconds == 30
    assert first.actual_floor_ms == 50_000
    assert [value.duration_ms for value in first.active_intervals] == [10_000, 20_000, 20_000]
    assert [value.duration_ms for value in first.pause_intervals] == [20_000, 10_000]
    assert [value.reason for value in first.pause_intervals] == [
        "MEETING_PAUSED",
        "HOST_INTERVENTION",
    ]
    assert first.finished_early is True
    assert first.overtime_ms == 0
    absent = facts.turns[1]
    assert absent.absent is True
    assert absent.speaker_user_ids == (None,)
    assert absent.speaker_names == ("未获纪要授权成员",)
    assert absent.absent_user_ids == (None,)
    assert absent.absent_names == ("未获纪要授权成员",)
    assert absent.started_at is None
    assert absent.actual_floor_ms == 0
    assert len(facts.followups) == 1
    assert facts.followups[0].standing_user_ids == ()
    assert [
        (value.user_id, value.finish_reason) for value in facts.followups[0].participations
    ] == [(None, "SPEAKER_FINISHED")]


def test_presentation_q_and_a_tracks_standing_presenters_and_overtime() -> None:
    baseline = datetime(2026, 8, 27, 2, 0, tzinfo=UTC)
    events = (
        _event(
            1,
            MeetingEventType.SPEAKING_FLOW_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-2",
                "mode": "PRESENTATION",
                "turns": [],
                "presenter_user_ids": [100, 200],
                "identities": [],
                "followup_mode": "Q_AND_A",
                "prep_seconds": 10,
                "presentation_seconds": 60,
            },
        ),
        _event(
            2,
            MeetingEventType.SPEAKING_TURN_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-2",
                "turn_index": 0,
                "current_user_ids": [100, 200],
                "next_user_ids": [],
                "planned_seconds": 60,
            },
        ),
        _event(
            3,
            MeetingEventType.SPEAKING_TURN_FINISHED,
            baseline + timedelta(seconds=65),
            {
                "agenda_item_id": "agenda-2",
                "turn_index": 0,
                "current_user_ids": [100, 200],
                "planned_seconds": 60,
                "extension_seconds": 0,
                "reason": "TIME_EXPIRED",
            },
        ),
        _event(
            4,
            MeetingEventType.SPEAKING_FOLLOWUP_STARTED,
            baseline + timedelta(seconds=65),
            {"agenda_item_id": "agenda-2", "followup_mode": "Q_AND_A"},
        ),
        _event(
            5,
            MeetingEventType.SPEAKING_FOLLOWUP_FINISHED,
            baseline + timedelta(seconds=75),
            {"agenda_item_id": "agenda-2", "followup_mode": "Q_AND_A"},
        ),
    )

    facts = project_authoritative_speaking_facts(
        events,
        labels={100: "Host", 200: "Presenter"},
    )

    assert facts.turns[0].overtime_ms == 5_000
    assert facts.turns[0].finish_reason == "TIME_EXPIRED"
    assert facts.followups[0].standing_names == (
        "Host (user:100)",
        "Presenter (user:200)",
    )


def test_presentation_records_partial_absence_without_leaking_identity() -> None:
    baseline = datetime(2026, 8, 27, 2, 30, tzinfo=UTC)
    events = (
        _event(
            1,
            MeetingEventType.SPEAKING_FLOW_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-2",
                "mode": "PRESENTATION",
                "turns": [],
                "presenter_user_ids": [100, 200],
                "identities": [
                    {"user_id": 100, "display_name": "Host"},
                    {"user_id": 200, "display_name": "Private presenter"},
                ],
                "followup_mode": "NONE",
                "prep_seconds": 10,
                "presentation_seconds": 60,
            },
        ),
        _event(
            2,
            MeetingEventType.SPEAKING_TURN_STARTED,
            baseline + timedelta(seconds=10),
            {
                "agenda_item_id": "agenda-2",
                "turn_index": 0,
                "current_user_ids": [100],
                "next_user_ids": [],
                "planned_seconds": 60,
            },
        ),
        _event(
            3,
            MeetingEventType.SPEAKING_TURN_FINISHED,
            baseline + timedelta(seconds=70),
            {
                "agenda_item_id": "agenda-2",
                "turn_index": 0,
                "current_user_ids": [100],
                "planned_seconds": 60,
                "extension_seconds": 0,
                "reason": "TIME_EXPIRED",
            },
        ),
    )

    turn = project_authoritative_speaking_facts(events, labels={100: "Current host"}).turns[0]

    assert turn.speaker_user_ids == (100,)
    assert turn.speaker_names == ("Host (user:100)",)
    assert turn.absent_user_ids == (None,)
    assert turn.absent_names == ("未获纪要授权成员",)


def test_rejects_malformed_timer_pause_sequence() -> None:
    baseline = datetime(2026, 8, 27, 2, 45, tzinfo=UTC)
    events = (
        _event(
            1,
            MeetingEventType.SPEAKING_FLOW_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-1",
                "mode": "ORDERED",
                "turns": [{"user_id": 100, "planned_seconds": 30}],
                "presenter_user_ids": [],
                "identities": [],
                "followup_mode": "NONE",
                "prep_seconds": 0,
                "presentation_seconds": None,
            },
        ),
        _event(
            2,
            MeetingEventType.SPEAKING_TURN_STARTED,
            baseline,
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 0,
                "current_user_ids": [100],
                "next_user_ids": [],
                "planned_seconds": 30,
            },
        ),
        _event(
            3,
            MeetingEventType.TIMER_STARTED,
            baseline,
            {
                "timer_id": "timer-1",
                "timer_type": "FLOW:SPEAKING_TURN:agenda-1:0",
            },
        ),
        _event(
            4,
            MeetingEventType.TIMER_PAUSED,
            baseline + timedelta(seconds=5),
            {"timer_id": "timer-1", "remaining_ms": 25_000},
        ),
        _event(
            5,
            MeetingEventType.TIMER_PAUSED,
            baseline + timedelta(seconds=6),
            {"timer_id": "timer-1", "remaining_ms": 24_000},
        ),
        _event(
            6,
            MeetingEventType.SPEAKING_TURN_FINISHED,
            baseline + timedelta(seconds=10),
            {
                "agenda_item_id": "agenda-1",
                "turn_index": 0,
                "current_user_ids": [100],
                "planned_seconds": 30,
                "extension_seconds": 0,
                "reason": "HOST_ENDED_EARLY",
            },
        ),
    )

    with pytest.raises(ValueError, match="paused more than once"):
        project_authoritative_speaking_facts(events, labels={})


def test_rejects_duplicate_turn_completion() -> None:
    baseline = datetime(2026, 8, 27, 3, 0, tzinfo=UTC)
    start = _event(
        1,
        MeetingEventType.SPEAKING_FLOW_STARTED,
        baseline,
        {
            "agenda_item_id": "agenda-1",
            "mode": "ORDERED",
            "turns": [{"user_id": 100, "planned_seconds": 30}],
            "presenter_user_ids": [],
            "identities": [],
            "followup_mode": "NONE",
            "prep_seconds": 0,
            "presentation_seconds": None,
        },
    )
    payload = {
        "agenda_item_id": "agenda-1",
        "turn_index": 0,
        "current_user_ids": [100],
        "planned_seconds": 30,
        "reason": "ABSENT",
    }

    with pytest.raises(ValueError, match="more than once"):
        project_authoritative_speaking_facts(
            (
                start,
                _event(2, MeetingEventType.SPEAKING_TURN_FINISHED, baseline, payload),
                _event(3, MeetingEventType.SPEAKING_TURN_FINISHED, baseline, payload),
            ),
            labels={},
        )
