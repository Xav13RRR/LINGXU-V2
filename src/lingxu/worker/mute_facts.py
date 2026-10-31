"""Project content-safe mute-enforcement outcomes from agenda policies and Meeting events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.models import AgendaItemRecord, MeetingEventRecord
from lingxu.meeting.mute_enforcement import project_mute_enforcement
from lingxu.worker.minutes_models import MinutesMuteEnforcementSummary, MuteEnforcementFinalStatus

_MUTE_EVENT_TYPES = frozenset(
    {
        MeetingEventType.MUTE_ENFORCEMENT_STARTED.value,
        MeetingEventType.MUTE_ENFORCEMENT_PAUSED.value,
        MeetingEventType.MUTE_ENFORCEMENT_RESUMED.value,
        MeetingEventType.MUTE_ENFORCEMENT_ENDED.value,
        MeetingEventType.MUTE_ENFORCEMENT_DOWNGRADED.value,
        MeetingEventType.MUTE_PERMISSION_LOST.value,
        MeetingEventType.MUTE_REAPPLIED.value,
        MeetingEventType.HOST_INTERVENTION_STARTED.value,
        MeetingEventType.EMERGENCY_MUTE_RELEASED.value,
    }
)


def project_authoritative_mute_summaries(
    agendas: Sequence[AgendaItemRecord],
    events: Sequence[MeetingEventRecord],
) -> tuple[MinutesMuteEnforcementSummary, ...]:
    """Summarize lifecycle counts without exposing leases, users, or Discord operations."""

    ordered_agendas = tuple(sorted(agendas, key=lambda item: (item.position, item.id)))
    agenda_ids = {item.id for item in ordered_agendas}
    relevant: dict[str, list[MeetingEventRecord]] = {item.id: [] for item in ordered_agendas}
    for event in events:
        if event.event_type not in _MUTE_EVENT_TYPES:
            continue
        agenda_item_id = event.payload.get("agenda_item_id")
        if not isinstance(agenda_item_id, str) or agenda_item_id not in agenda_ids:
            raise ValueError("mute-enforcement event references an unknown agenda")
        relevant[agenda_item_id].append(event)

    results: list[MinutesMuteEnforcementSummary] = []
    event_tuple = tuple(events)
    for agenda in ordered_agendas:
        requested = agenda.speaking_policy.mute_enforcement_requested
        selected = relevant[agenda.id]
        if selected and not requested:
            raise ValueError("unrequested agenda contains mute-enforcement events")
        counts = Counter(value.event_type for value in selected)
        started = counts[MeetingEventType.MUTE_ENFORCEMENT_STARTED.value] > 0
        if not requested:
            final_status: MuteEnforcementFinalStatus = "NOT_REQUESTED"
        elif not started:
            final_status = "REQUESTED_NOT_STARTED"
        elif counts[MeetingEventType.EMERGENCY_MUTE_RELEASED.value]:
            final_status = "EMERGENCY_RELEASED"
        elif counts[MeetingEventType.MUTE_ENFORCEMENT_DOWNGRADED.value]:
            final_status = "DOWNGRADED"
        else:
            state = project_mute_enforcement(event_tuple, agenda.id)
            if state is None:
                raise ValueError("started mute enforcement cannot be projected")
            if state.enabled and state.paused:
                final_status = "PAUSED"
            elif state.enabled:
                final_status = "ACTIVE"
            else:
                final_status = "ENDED"
        results.append(
            MinutesMuteEnforcementSummary(
                agenda_item_id=agenda.id,
                requested=requested,
                started=started,
                pause_count=counts[MeetingEventType.MUTE_ENFORCEMENT_PAUSED.value],
                resume_count=counts[MeetingEventType.MUTE_ENFORCEMENT_RESUMED.value],
                permission_loss_count=counts[MeetingEventType.MUTE_PERMISSION_LOST.value],
                reapplication_count=counts[MeetingEventType.MUTE_REAPPLIED.value],
                host_intervention_count=counts[MeetingEventType.HOST_INTERVENTION_STARTED.value],
                final_status=final_status,
            )
        )
    return tuple(results)
