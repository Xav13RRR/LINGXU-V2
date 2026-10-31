from datetime import UTC, datetime, timedelta

import pytest

from lingxu.domain.enums import AgendaStatus, MeetingState
from lingxu.domain.models import AgendaItemRecord, MeetingEventRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.worker.mute_facts import project_authoritative_mute_summaries


def _agenda(*, requested: bool) -> AgendaItemRecord:
    return AgendaItemRecord(
        id="agenda-1",
        session_id="session-1",
        position=1,
        title="顺序发言",
        status=AgendaStatus.COMPLETED,
        duration_minutes=10,
        requires_vote=False,
        speaking_policy=AgendaSpeakingPolicy.ordered_explicit(
            turns=(SpeakerTurnPlan(user_id=10, planned_seconds=60),),
            mute_enforcement_requested=requested,
        ),
    )


def _event(sequence: int, event_type: str) -> MeetingEventRecord:
    at = datetime(2026, 8, 27, tzinfo=UTC) + timedelta(seconds=sequence)
    return MeetingEventRecord(
        id=f"event-{sequence}",
        session_id="session-1",
        sequence_number=sequence,
        event_type=event_type,
        actor_user_id=None,
        source="SYSTEM",
        payload={"agenda_item_id": "agenda-1"},
        evidence_utterance_ids=(),
        state_before=MeetingState.DISCUSSION_READY,
        state_after=MeetingState.DISCUSSION_READY,
        reversible=False,
        correlation_id=f"event-{sequence}",
        occurred_at=at,
    )


def test_mute_summary_records_permission_recovery_reapplication_and_terminal_state() -> None:
    summary = project_authoritative_mute_summaries(
        (_agenda(requested=True),),
        tuple(
            _event(index, event_type)
            for index, event_type in enumerate(
                (
                    "MUTE_ENFORCEMENT_STARTED",
                    "MUTE_PERMISSION_LOST",
                    "MUTE_ENFORCEMENT_PAUSED",
                    "MUTE_ENFORCEMENT_RESUMED",
                    "MUTE_REAPPLIED",
                    "HOST_INTERVENTION_STARTED",
                    "MUTE_ENFORCEMENT_ENDED",
                ),
                start=1,
            )
        ),
    )[0]

    assert summary.requested is True
    assert summary.started is True
    assert summary.permission_loss_count == 1
    assert summary.pause_count == 1
    assert summary.resume_count == 1
    assert summary.reapplication_count == 1
    assert summary.host_intervention_count == 1
    assert summary.final_status == "ENDED"


def test_unstarted_request_and_unrequested_event_are_distinguished() -> None:
    summary = project_authoritative_mute_summaries((_agenda(requested=True),), ())[0]
    assert summary.final_status == "REQUESTED_NOT_STARTED"

    with pytest.raises(ValueError, match="unrequested agenda"):
        project_authoritative_mute_summaries(
            (_agenda(requested=False),),
            (_event(1, "MUTE_ENFORCEMENT_STARTED"),),
        )
