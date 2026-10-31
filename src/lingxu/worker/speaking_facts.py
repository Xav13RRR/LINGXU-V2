"""Project controlled-speaking facts exclusively from append-only Meeting events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.models import MeetingEventRecord
from lingxu.worker.minutes_models import (
    MinutesFollowupParticipation,
    MinutesSpeakingFollowup,
    MinutesSpeakingInterval,
    MinutesSpeakingPauseInterval,
    MinutesSpeakingTurn,
)

_FINISH_REASONS = frozenset(
    {
        "TIME_EXPIRED",
        "HOST_ENDED_EARLY",
        "HOST_SKIPPED",
        "PARTICIPANT_ABANDONED",
        "ABSENT",
    }
)


@dataclass(frozen=True, slots=True)
class SpeakingFactsProjection:
    turns: tuple[MinutesSpeakingTurn, ...]
    followups: tuple[MinutesSpeakingFollowup, ...]


@dataclass(frozen=True, slots=True)
class _Flow:
    sequence_number: int
    mode: str
    followup_mode: str
    turn_user_ids: tuple[int, ...]
    turn_planned_seconds: tuple[int, ...]
    presenter_user_ids: tuple[int, ...]
    presentation_seconds: int | None
    names: Mapping[int, str]


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"speaking fact field {key} is invalid")
    return value


def _user_ids(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"speaking fact field {key} is invalid")
    selected = tuple(value)
    if (
        not selected
        or len(selected) > 50
        or len(set(selected)) != len(selected)
        or any(item <= 0 for item in selected)
    ):
        raise ValueError(f"speaking fact field {key} is invalid")
    return selected


def _milliseconds(started_at: datetime, ended_at: datetime) -> int:
    if ended_at < started_at:
        raise ValueError("speaking fact event time is reversed")
    return max(0, int((ended_at - started_at).total_seconds() * 1_000))


def _flow(event: MeetingEventRecord, labels: Mapping[int, str]) -> _Flow:
    payload = event.payload
    mode = str(payload.get("mode", ""))
    followup_mode = str(payload.get("followup_mode", ""))
    if mode not in {"ORDERED", "PRESENTATION"} or followup_mode not in {
        "NONE",
        "SUPPLEMENT_TURNS",
        "Q_AND_A",
    }:
        raise ValueError("speaking flow metadata is invalid")
    raw_presenters = payload.get("presenter_user_ids", [])
    raw_turns = payload.get("turns", [])
    raw_identities = payload.get("identities", [])
    if not isinstance(raw_presenters, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_presenters
    ):
        raise ValueError("speaking presenter identities are invalid")
    if len(set(raw_presenters)) != len(raw_presenters):
        raise ValueError("speaking presenter identities are invalid")
    if any(value <= 0 for value in raw_presenters):
        raise ValueError("speaking presenter identities are invalid")
    if not isinstance(raw_turns, list) or not isinstance(raw_identities, list):
        raise ValueError("speaking identity snapshot is invalid")
    turn_user_ids: list[int] = []
    turn_planned_seconds: list[int] = []
    for value in raw_turns:
        if not isinstance(value, dict):
            raise ValueError("speaking turn plan is invalid")
        user_id = _integer(value, "user_id")
        planned_seconds = _integer(value, "planned_seconds")
        if user_id <= 0 or planned_seconds <= 0:
            raise ValueError("speaking turn plan is invalid")
        turn_user_ids.append(user_id)
        turn_planned_seconds.append(planned_seconds)
    if len(set(turn_user_ids)) != len(turn_user_ids):
        raise ValueError("speaking turn plan is invalid")
    presentation_value = payload.get("presentation_seconds")
    presentation_seconds = (
        None if presentation_value is None else _integer(payload, "presentation_seconds")
    )
    if mode == "ORDERED" and (
        not turn_user_ids or raw_presenters or presentation_seconds is not None
    ):
        raise ValueError("ordered speaking flow plan is invalid")
    if mode == "PRESENTATION" and (
        turn_user_ids
        or not raw_presenters
        or presentation_seconds is None
        or presentation_seconds <= 0
    ):
        raise ValueError("presentation speaking flow plan is invalid")
    names = dict(labels)
    identity_user_ids: set[int] = set()
    for value in raw_identities:
        if not isinstance(value, dict):
            raise ValueError("speaking identity snapshot is invalid")
        user_id = _integer(value, "user_id")
        if user_id <= 0:
            raise ValueError("speaking identity snapshot is invalid")
        if user_id in identity_user_ids:
            raise ValueError("speaking identity snapshot is invalid")
        identity_user_ids.add(user_id)
        display_name = value.get("display_name")
        if display_name is not None:
            if not isinstance(display_name, str) or not display_name or len(display_name) > 256:
                raise ValueError("speaking display identity is invalid")
            if user_id in labels:
                names[user_id] = display_name
    return _Flow(
        sequence_number=event.sequence_number,
        mode=mode,
        followup_mode=followup_mode,
        turn_user_ids=tuple(turn_user_ids),
        turn_planned_seconds=tuple(turn_planned_seconds),
        presenter_user_ids=tuple(raw_presenters),
        presentation_seconds=presentation_seconds,
        names=names,
    )


def _name(flow: _Flow, user_id: int) -> str:
    if user_id not in flow.names:
        return "未获纪要授权成员"
    return f"{flow.names[user_id]} (user:{user_id})"


def _public_user_id(flow: _Flow, user_id: int) -> int | None:
    return user_id if user_id in flow.names else None


def _timer_type(
    event: MeetingEventRecord,
    timer_types: Mapping[str, str],
) -> str | None:
    value = event.payload.get("timer_type")
    timer_id = event.payload.get("timer_id")
    mapped = timer_types.get(timer_id) if isinstance(timer_id, str) else None
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("speaking timer type is invalid")
    if isinstance(value, str) and mapped is not None and value != mapped:
        raise ValueError("speaking timer identity is inconsistent")
    return value if isinstance(value, str) else mapped


def _turn_intervals(
    events: Sequence[MeetingEventRecord],
    *,
    started: MeetingEventRecord,
    finished: MeetingEventRecord,
    timer_type: str,
    timer_types: Mapping[str, str],
) -> tuple[tuple[MinutesSpeakingInterval, ...], tuple[MinutesSpeakingPauseInterval, ...]]:
    active: list[MinutesSpeakingInterval] = []
    paused: list[MinutesSpeakingPauseInterval] = []
    active_started = started.occurred_at
    pause_started: datetime | None = None
    pause_reason: str | None = None
    for event in events:
        if not started.sequence_number < event.sequence_number < finished.sequence_number:
            continue
        if event.event_type == MeetingEventType.TIMER_PAUSED.value:
            if _timer_type(event, timer_types) != timer_type:
                continue
            if pause_started is not None:
                raise ValueError("speaking timer was paused more than once")
            active.append(
                MinutesSpeakingInterval(
                    started_at=active_started,
                    ended_at=event.occurred_at,
                    duration_ms=_milliseconds(active_started, event.occurred_at),
                )
            )
            pause_started = event.occurred_at
            reason = event.payload.get("reason", "MEETING_PAUSED")
            if not isinstance(reason, str) or not reason or len(reason) > 128:
                raise ValueError("speaking timer pause reason is invalid")
            pause_reason = reason
        elif event.event_type == MeetingEventType.TIMER_RESUMED.value:
            if _timer_type(event, timer_types) != timer_type:
                continue
            if pause_started is None:
                raise ValueError("speaking timer resumed without a pause")
            paused.append(
                MinutesSpeakingPauseInterval(
                    started_at=pause_started,
                    ended_at=event.occurred_at,
                    duration_ms=_milliseconds(pause_started, event.occurred_at),
                    reason=pause_reason or "MEETING_PAUSED",
                )
            )
            active_started = event.occurred_at
            pause_started = None
            pause_reason = None
    if pause_started is None:
        active.append(
            MinutesSpeakingInterval(
                started_at=active_started,
                ended_at=finished.occurred_at,
                duration_ms=_milliseconds(active_started, finished.occurred_at),
            )
        )
    else:
        paused.append(
            MinutesSpeakingPauseInterval(
                started_at=pause_started,
                ended_at=finished.occurred_at,
                duration_ms=_milliseconds(pause_started, finished.occurred_at),
                reason=pause_reason or "MEETING_PAUSED",
            )
        )
    return tuple(active), tuple(paused)


def _project_turns(
    events: Sequence[MeetingEventRecord],
    *,
    flows: Mapping[str, _Flow],
    timer_types: Mapping[str, str],
) -> tuple[MinutesSpeakingTurn, ...]:
    results: list[MinutesSpeakingTurn] = []
    seen: set[tuple[str, int]] = set()
    for finished in events:
        if finished.event_type != MeetingEventType.SPEAKING_TURN_FINISHED.value:
            continue
        agenda_item_id = finished.payload.get("agenda_item_id")
        if not isinstance(agenda_item_id, str) or agenda_item_id not in flows:
            raise ValueError("speaking turn is not bound to a flow")
        turn_index = _integer(finished.payload, "turn_index")
        if turn_index < 0:
            raise ValueError("speaking turn index is invalid")
        key = (agenda_item_id, turn_index)
        if key in seen:
            raise ValueError("speaking turn was finished more than once")
        seen.add(key)
        flow = flows[agenda_item_id]
        if finished.sequence_number <= flow.sequence_number:
            raise ValueError("speaking turn precedes its flow")
        speaker_user_ids = _user_ids(finished.payload, "current_user_ids")
        planned_seconds = _integer(finished.payload, "planned_seconds")
        if planned_seconds <= 0:
            raise ValueError("speaking turn planned duration is invalid")
        if flow.mode == "ORDERED":
            if (
                turn_index >= len(flow.turn_user_ids)
                or speaker_user_ids != (flow.turn_user_ids[turn_index],)
                or planned_seconds != flow.turn_planned_seconds[turn_index]
            ):
                raise ValueError("speaking turn differs from its immutable plan")
        elif (
            turn_index != 0
            or not set(speaker_user_ids).issubset(flow.presenter_user_ids)
            or planned_seconds != flow.presentation_seconds
        ):
            raise ValueError("presentation turn differs from its immutable plan")
        reason = str(finished.payload.get("reason", ""))
        if reason not in _FINISH_REASONS:
            raise ValueError("speaking finish reason is invalid")
        started = next(
            (
                event
                for event in reversed(events)
                if event.sequence_number < finished.sequence_number
                and event.event_type == MeetingEventType.SPEAKING_TURN_STARTED.value
                and event.payload.get("agenda_item_id") == agenda_item_id
                and event.payload.get("turn_index") == turn_index
            ),
            None,
        )
        starts = tuple(
            event
            for event in events
            if event.sequence_number < finished.sequence_number
            and event.event_type == MeetingEventType.SPEAKING_TURN_STARTED.value
            and event.payload.get("agenda_item_id") == agenda_item_id
            and event.payload.get("turn_index") == turn_index
        )
        if len(starts) > 1:
            raise ValueError("speaking turn was started more than once")
        matching_extensions = tuple(
            event
            for event in events
            if event.event_type == MeetingEventType.SPEAKING_TURN_EXTENDED.value
            and event.payload.get("agenda_item_id") == agenda_item_id
            and event.payload.get("turn_index") == turn_index
            and event.sequence_number < finished.sequence_number
        )
        if matching_extensions and (
            started is None
            or any(
                event.sequence_number <= started.sequence_number for event in matching_extensions
            )
        ):
            raise ValueError("speaking turn was extended outside its active interval")
        extension_seconds = sum(
            _integer(event.payload, "additional_seconds") for event in matching_extensions
        )
        if extension_seconds < 0 or extension_seconds > 28_800:
            raise ValueError("speaking extension total is invalid")
        stored_extension = finished.payload.get("extension_seconds", 0)
        if (
            isinstance(stored_extension, bool)
            or not isinstance(stored_extension, int)
            or stored_extension != extension_seconds
        ):
            raise ValueError("speaking extension facts are inconsistent")
        active: tuple[MinutesSpeakingInterval, ...] = ()
        paused: tuple[MinutesSpeakingPauseInterval, ...] = ()
        if started is not None:
            started_ids = _user_ids(started.payload, "current_user_ids")
            if (
                started_ids != speaker_user_ids
                or _integer(started.payload, "planned_seconds") != planned_seconds
            ):
                raise ValueError("speaking turn identity changed after floor grant")
            timer_type = f"FLOW:SPEAKING_TURN:{agenda_item_id}:{turn_index}"
            active, paused = _turn_intervals(
                events,
                started=started,
                finished=finished,
                timer_type=timer_type,
                timer_types=timer_types,
            )
        actual_floor_ms = sum(value.duration_ms for value in active)
        if flow.mode == "PRESENTATION":
            absent_user_ids = tuple(
                value for value in flow.presenter_user_ids if value not in speaker_user_ids
            )
            if reason == "ABSENT":
                absent_user_ids = flow.presenter_user_ids
        else:
            absent_user_ids = speaker_user_ids if reason == "ABSENT" else ()
        results.append(
            MinutesSpeakingTurn(
                agenda_item_id=agenda_item_id,
                turn_index=turn_index,
                mode=cast(Literal["ORDERED", "PRESENTATION"], flow.mode),
                speaker_user_ids=tuple(_public_user_id(flow, value) for value in speaker_user_ids),
                speaker_names=tuple(_name(flow, value) for value in speaker_user_ids),
                absent_user_ids=tuple(_public_user_id(flow, value) for value in absent_user_ids),
                absent_names=tuple(_name(flow, value) for value in absent_user_ids),
                planned_seconds=planned_seconds,
                extension_seconds=extension_seconds,
                started_at=started.occurred_at if started is not None else None,
                finished_at=finished.occurred_at,
                active_intervals=active,
                pause_intervals=paused,
                actual_floor_ms=actual_floor_ms,
                finish_reason=cast(
                    Literal[
                        "TIME_EXPIRED",
                        "HOST_ENDED_EARLY",
                        "HOST_SKIPPED",
                        "PARTICIPANT_ABANDONED",
                        "ABSENT",
                    ],
                    reason,
                ),
                finished_early=reason in {"HOST_ENDED_EARLY", "PARTICIPANT_ABANDONED"},
                skipped=reason == "HOST_SKIPPED",
                absent=reason == "ABSENT",
                overtime_ms=max(
                    0,
                    actual_floor_ms - (planned_seconds + extension_seconds) * 1_000,
                ),
            )
        )
    return tuple(results)


def _project_followups(
    events: Sequence[MeetingEventRecord],
    *,
    flows: Mapping[str, _Flow],
) -> tuple[MinutesSpeakingFollowup, ...]:
    results: list[MinutesSpeakingFollowup] = []
    seen: set[str] = set()
    for started in events:
        if started.event_type != MeetingEventType.SPEAKING_FOLLOWUP_STARTED.value:
            continue
        agenda_item_id = started.payload.get("agenda_item_id")
        if not isinstance(agenda_item_id, str) or agenda_item_id not in flows:
            raise ValueError("follow-up is not bound to a speaking flow")
        if agenda_item_id in seen:
            raise ValueError("speaking follow-up was started more than once")
        seen.add(agenda_item_id)
        flow = flows[agenda_item_id]
        mode = str(started.payload.get("followup_mode", ""))
        if mode not in {"SUPPLEMENT_TURNS", "Q_AND_A"} or mode != flow.followup_mode:
            raise ValueError("follow-up mode is inconsistent")
        finished = next(
            (
                event
                for event in events
                if event.sequence_number > started.sequence_number
                and event.event_type == MeetingEventType.SPEAKING_FOLLOWUP_FINISHED.value
                and event.payload.get("agenda_item_id") == agenda_item_id
            ),
            None,
        )
        if finished is None:
            raise ValueError("completed minutes require a closed speaking follow-up")
        participations: list[MinutesFollowupParticipation] = []
        current_user_id: int | None = None
        for event in events:
            if event.sequence_number >= started.sequence_number:
                break
            user_id = event.payload.get("user_id")
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                continue
            if user_id <= 0:
                raise ValueError("follow-up speaker identity is invalid")
            if event.event_type == MeetingEventType.SPEAKER_SELECTED.value:
                current_user_id = user_id
            elif (
                event.event_type
                in {
                    MeetingEventType.SPEAKER_FINISHED.value,
                    MeetingEventType.SPEAKER_REMOVED.value,
                }
                and current_user_id == user_id
            ):
                current_user_id = None
        selected: tuple[int, datetime] | None = (
            (current_user_id, started.occurred_at) if current_user_id is not None else None
        )
        for event in events:
            if not started.sequence_number < event.sequence_number < finished.sequence_number:
                continue
            user_id = event.payload.get("user_id")
            if isinstance(user_id, bool) or not isinstance(user_id, int):
                continue
            if user_id <= 0:
                raise ValueError("follow-up speaker identity is invalid")
            if event.event_type == MeetingEventType.SPEAKER_SELECTED.value:
                if selected is not None:
                    raise ValueError("follow-up selected overlapping speakers")
                selected = (user_id, event.occurred_at)
            elif event.event_type in {
                MeetingEventType.SPEAKER_FINISHED.value,
                MeetingEventType.SPEAKER_REMOVED.value,
            }:
                if selected is None or selected[0] != user_id:
                    continue
                participations.append(
                    MinutesFollowupParticipation(
                        user_id=_public_user_id(flow, user_id),
                        name=_name(flow, user_id),
                        started_at=selected[1],
                        finished_at=event.occurred_at,
                        finish_reason=(
                            "SPEAKER_FINISHED"
                            if event.event_type == MeetingEventType.SPEAKER_FINISHED.value
                            else "SPEAKER_REMOVED"
                        ),
                    )
                )
                selected = None
        if selected is not None:
            participations.append(
                MinutesFollowupParticipation(
                    user_id=_public_user_id(flow, selected[0]),
                    name=_name(flow, selected[0]),
                    started_at=selected[1],
                    finished_at=finished.occurred_at,
                    finish_reason="FOLLOWUP_FINISHED",
                )
            )
        standing = flow.presenter_user_ids if mode == "Q_AND_A" else ()
        results.append(
            MinutesSpeakingFollowup(
                agenda_item_id=agenda_item_id,
                mode=cast(Literal["SUPPLEMENT_TURNS", "Q_AND_A"], mode),
                started_at=started.occurred_at,
                finished_at=finished.occurred_at,
                standing_user_ids=tuple(_public_user_id(flow, value) for value in standing),
                standing_names=tuple(_name(flow, value) for value in standing),
                participations=tuple(participations),
            )
        )
    return tuple(results)


def project_authoritative_speaking_facts(
    events: Sequence[MeetingEventRecord],
    *,
    labels: Mapping[int, str],
) -> SpeakingFactsProjection:
    """Build exact completed turn/follow-up facts without transcript or model inference."""

    ordered = tuple(sorted(events, key=lambda value: value.sequence_number))
    if len({value.sequence_number for value in ordered}) != len(ordered):
        raise ValueError("Meeting event sequence is not unique")
    flows: dict[str, _Flow] = {}
    timer_types: dict[str, str] = {}
    for event in ordered:
        if event.event_type == MeetingEventType.SPEAKING_FLOW_STARTED.value:
            agenda_item_id = event.payload.get("agenda_item_id")
            if not isinstance(agenda_item_id, str) or not agenda_item_id or agenda_item_id in flows:
                raise ValueError("speaking flow identity is invalid")
            flows[agenda_item_id] = _flow(event, labels)
        elif event.event_type == MeetingEventType.TIMER_STARTED.value:
            timer_id = event.payload.get("timer_id")
            timer_type = event.payload.get("timer_type")
            if isinstance(timer_id, str) and isinstance(timer_type, str):
                if timer_id in timer_types and timer_types[timer_id] != timer_type:
                    raise ValueError("speaking timer identity was reused")
                timer_types[timer_id] = timer_type
    return SpeakingFactsProjection(
        turns=_project_turns(ordered, flows=flows, timer_types=timer_types),
        followups=_project_followups(ordered, flows=flows),
    )
