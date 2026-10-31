"""Event-sourced advisory speaking flows for controlled agenda items."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    MeetingEventType,
    SpeakingFlowStage,
)
from lingxu.domain.models import AgendaItemRecord, MeetingEventRecord, MeetingSessionRecord
from lingxu.domain.speaking import (
    FloorParticipant,
    SpeakerTurnPlan,
    planned_turns,
)
from lingxu.meeting.components import TimerNotice, TimerService
from lingxu.storage.component_repositories import ParticipantRepository
from lingxu.storage.repositories import MeetingEventStore


class SpeakingFlowNoticeKind(StrEnum):
    PREPARING = "PREPARING"
    STARTED = "STARTED"
    REMINDER = "REMINDER"
    EXTENDED = "EXTENDED"
    FINISHED = "FINISHED"
    FOLLOWUP_STARTED = "FOLLOWUP_STARTED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True, slots=True)
class SpeakingIdentity:
    user_id: int
    display_name: str | None


@dataclass(frozen=True, slots=True)
class SpeakingFlowState:
    """Projection of one controlled agenda's immutable event facts."""

    agenda_item_id: str
    mode: AgendaSpeakingMode
    stage: SpeakingFlowStage
    turns: tuple[SpeakerTurnPlan, ...]
    presenter_user_ids: tuple[int, ...]
    identities: tuple[SpeakingIdentity, ...]
    followup_mode: AgendaFollowupMode
    prep_seconds: int
    presentation_seconds: int | None
    current_index: int
    current_user_ids: tuple[int, ...]
    next_user_ids: tuple[int, ...]
    planned_seconds: int | None
    extension_seconds: int
    stage_started_at: datetime

    @property
    def timer_type(self) -> str | None:
        if self.stage is SpeakingFlowStage.PREPARING:
            return f"FLOW:SPEAKING_PREP:{self.agenda_item_id}:{self.current_index}"
        if self.stage is SpeakingFlowStage.ACTIVE:
            return f"FLOW:SPEAKING_TURN:{self.agenda_item_id}:{self.current_index}"
        return None

    @property
    def timer_seconds(self) -> int | None:
        if self.stage is SpeakingFlowStage.PREPARING:
            return self.prep_seconds
        if self.stage is SpeakingFlowStage.ACTIVE and self.planned_seconds is not None:
            return self.planned_seconds + self.extension_seconds
        return None

    def display_name(self, user_id: int) -> str:
        return next(
            (
                identity.display_name
                for identity in self.identities
                if identity.user_id == user_id and identity.display_name
            ),
            f"成员{user_id}",
        )


@dataclass(frozen=True, slots=True)
class SpeakingFlowNotice:
    kind: SpeakingFlowNoticeKind
    agenda_item_id: str
    mode: AgendaSpeakingMode
    current_user_ids: tuple[int, ...] = ()
    current_display_names: tuple[str, ...] = ()
    next_user_ids: tuple[int, ...] = ()
    next_display_names: tuple[str, ...] = ()
    planned_seconds: int | None = None
    remaining_seconds: int | None = None
    additional_seconds: int | None = None
    prep_seconds: int | None = None
    followup_mode: AgendaFollowupMode = AgendaFollowupMode.NONE
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakingFlowUpdate:
    state: SpeakingFlowState | None
    notices: tuple[SpeakingFlowNotice, ...] = ()
    changed: bool = False


def _integer(payload: dict[str, object], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"invalid speaking flow event field: {key}")
    return value


def project_speaking_flow(
    events: tuple[MeetingEventRecord, ...],
    agenda_item_id: str,
) -> SpeakingFlowState | None:
    """Rebuild one agenda flow without consulting mutable process state."""

    state: SpeakingFlowState | None = None
    for event in events:
        if event.payload.get("agenda_item_id") != agenda_item_id:
            continue
        payload = event.payload
        if event.event_type == MeetingEventType.SPEAKING_FLOW_STARTED.value:
            raw_identities = payload.get("identities", [])
            raw_turns = payload.get("turns", [])
            raw_presenters = payload.get("presenter_user_ids", [])
            if not isinstance(raw_identities, list):
                raise ValueError("invalid speaking identities event payload")
            if not isinstance(raw_turns, list) or not isinstance(raw_presenters, list):
                raise ValueError("invalid speaking plan event payload")
            identities = tuple(
                SpeakingIdentity(
                    user_id=_integer(identity, "user_id"),
                    display_name=(
                        identity.get("display_name")
                        if isinstance(identity.get("display_name"), str)
                        else None
                    ),
                )
                for identity in raw_identities
                if isinstance(identity, dict)
            )
            state = SpeakingFlowState(
                agenda_item_id=agenda_item_id,
                mode=AgendaSpeakingMode(str(payload["mode"])),
                stage=SpeakingFlowStage.STARTING,
                turns=tuple(SpeakerTurnPlan.from_payload(turn) for turn in raw_turns),
                presenter_user_ids=tuple(int(value) for value in raw_presenters),
                identities=identities,
                followup_mode=AgendaFollowupMode(str(payload["followup_mode"])),
                prep_seconds=_integer(payload, "prep_seconds"),
                presentation_seconds=(
                    _integer(payload, "presentation_seconds")
                    if payload.get("presentation_seconds") is not None
                    else None
                ),
                current_index=-1,
                current_user_ids=(),
                next_user_ids=(),
                planned_seconds=None,
                extension_seconds=0,
                stage_started_at=event.occurred_at,
            )
            continue
        if state is None:
            continue
        if event.event_type == MeetingEventType.SPEAKING_PREPARATION_STARTED.value:
            state = replace(
                state,
                stage=SpeakingFlowStage.PREPARING,
                current_index=_integer(payload, "turn_index"),
                current_user_ids=tuple(int(v) for v in payload["current_user_ids"]),
                next_user_ids=tuple(int(v) for v in payload["next_user_ids"]),
                planned_seconds=_integer(payload, "planned_seconds"),
                extension_seconds=0,
                stage_started_at=event.occurred_at,
            )
        elif event.event_type == MeetingEventType.SPEAKING_TURN_STARTED.value:
            state = replace(
                state,
                stage=SpeakingFlowStage.ACTIVE,
                current_user_ids=tuple(
                    int(v) for v in payload.get("current_user_ids", state.current_user_ids)
                ),
                next_user_ids=tuple(
                    int(v) for v in payload.get("next_user_ids", state.next_user_ids)
                ),
                stage_started_at=event.occurred_at,
            )
        elif event.event_type == MeetingEventType.SPEAKING_TURN_EXTENDED.value:
            state = replace(
                state,
                extension_seconds=state.extension_seconds + _integer(payload, "additional_seconds"),
            )
        elif event.event_type == MeetingEventType.SPEAKING_TURN_FINISHED.value:
            state = replace(
                state,
                stage=SpeakingFlowStage.STARTING,
                current_index=_integer(payload, "turn_index"),
                current_user_ids=tuple(int(v) for v in payload.get("current_user_ids", [])),
                next_user_ids=(),
                planned_seconds=(
                    _integer(payload, "planned_seconds")
                    if payload.get("planned_seconds") is not None
                    else state.planned_seconds
                ),
                stage_started_at=event.occurred_at,
            )
        elif event.event_type == MeetingEventType.SPEAKING_FOLLOWUP_STARTED.value:
            state = replace(
                state,
                stage=SpeakingFlowStage.FOLLOWUP,
                current_user_ids=(),
                next_user_ids=(),
                planned_seconds=None,
                extension_seconds=0,
                stage_started_at=event.occurred_at,
            )
        elif event.event_type == MeetingEventType.SPEAKING_FLOW_COMPLETED.value:
            state = replace(
                state,
                stage=SpeakingFlowStage.COMPLETED,
                current_user_ids=(),
                next_user_ids=(),
                planned_seconds=None,
                extension_seconds=0,
                stage_started_at=event.occurred_at,
            )
    return state


def _notice(
    kind: SpeakingFlowNoticeKind,
    state: SpeakingFlowState,
    *,
    remaining_seconds: int | None = None,
    additional_seconds: int | None = None,
    reason: str | None = None,
) -> SpeakingFlowNotice:
    return SpeakingFlowNotice(
        kind=kind,
        agenda_item_id=state.agenda_item_id,
        mode=state.mode,
        current_user_ids=state.current_user_ids,
        current_display_names=tuple(
            state.display_name(user_id) for user_id in state.current_user_ids
        ),
        next_user_ids=state.next_user_ids,
        next_display_names=tuple(state.display_name(user_id) for user_id in state.next_user_ids),
        planned_seconds=state.planned_seconds,
        remaining_seconds=remaining_seconds,
        additional_seconds=additional_seconds,
        prep_seconds=state.prep_seconds,
        followup_mode=state.followup_mode,
        reason=reason,
    )


class ControlledSpeakingFlowService:
    """Persist and recover ORDERED/PRESENTATION advisory flow transitions."""

    def __init__(
        self,
        *,
        participants: ParticipantRepository,
        events: MeetingEventStore,
        timers: TimerService,
    ) -> None:
        self._participants = participants
        self._events = events
        self._timers = timers
        self._lock = asyncio.Lock()

    async def state(self, session_id: str, agenda_item_id: str) -> SpeakingFlowState | None:
        return project_speaking_flow(await self._events.list(session_id), agenda_item_id)

    async def _append(
        self,
        *,
        session_id: str,
        event_type: MeetingEventType,
        agenda_item_id: str,
        payload: dict[str, object] | None = None,
        actor_user_id: int | None = None,
        source: str = "SYSTEM",
    ) -> None:
        await self._events.append(
            session_id=session_id,
            event_type=event_type.value,
            actor_user_id=actor_user_id,
            source=source,
            payload={"agenda_item_id": agenda_item_id, **(payload or {})},
        )

    async def activate(
        self,
        *,
        session: MeetingSessionRecord,
        item: AgendaItemRecord,
    ) -> SpeakingFlowUpdate:
        if item.speaking_policy.mode is AgendaSpeakingMode.FREE:
            return SpeakingFlowUpdate(state=None)
        async with self._lock:
            existing = await self.state(session.id, item.id)
            if existing is not None:
                if existing.stage is SpeakingFlowStage.STARTING:
                    return await self._prepare_next_locked(session.id, existing)
                if existing.stage is SpeakingFlowStage.PREPARING and existing.prep_seconds == 0:
                    return await self._start_current_locked(session.id, existing)
                await self._ensure_timer_locked(session.id, existing)
                return SpeakingFlowUpdate(state=existing)
            participants = await self._participants.list_session(session.id)
            identities = tuple(
                FloorParticipant(value.user_id, value.display_name) for value in participants
            )
            policy = item.speaking_policy
            turns = planned_turns(policy, identities, session.creator_user_id)
            presentation_seconds = (
                item.duration_minutes * 60
                if policy.mode is AgendaSpeakingMode.PRESENTATION
                and item.duration_minutes is not None
                else None
            )
            if policy.mode is AgendaSpeakingMode.PRESENTATION and presentation_seconds is None:
                raise ValueError("presentation agenda requires a duration")
            await self._append(
                session_id=session.id,
                event_type=MeetingEventType.SPEAKING_FLOW_STARTED,
                agenda_item_id=item.id,
                payload={
                    "mode": policy.mode.value,
                    "turns": [turn.to_payload() for turn in turns],
                    "presenter_user_ids": list(policy.presenter_user_ids),
                    "identities": [
                        {"user_id": value.user_id, "display_name": value.display_name}
                        for value in identities
                    ],
                    "followup_mode": policy.followup_mode.value,
                    "prep_seconds": policy.prep_seconds,
                    "presentation_seconds": presentation_seconds,
                },
            )
            state = await self.state(session.id, item.id)
            if state is None:
                raise RuntimeError("speaking flow start event was not projected")
            return await self._prepare_next_locked(session.id, state)

    async def _ensure_timer_locked(self, session_id: str, state: SpeakingFlowState) -> bool:
        timer_type = state.timer_type
        duration_seconds = state.timer_seconds
        if timer_type is None or duration_seconds is None or duration_seconds <= 0:
            return False
        elapsed = max(
            0,
            int((datetime.now(UTC) - state.stage_started_at).total_seconds()),
        )
        return (
            await self._timers.start_after_handoff_barrier(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=max(1, duration_seconds - elapsed),
            )
            is not None
        )

    async def _reconcile_current_stage_locked(
        self,
        session_id: str,
        state: SpeakingFlowState,
    ) -> SpeakingFlowUpdate:
        if state.stage is SpeakingFlowStage.STARTING:
            return await self._prepare_next_locked(session_id, state)
        if state.stage is SpeakingFlowStage.PREPARING and state.prep_seconds == 0:
            return await self._start_current_locked(session_id, state)
        await self._ensure_timer_locked(session_id, state)
        return SpeakingFlowUpdate(state=state)

    async def reconcile_current_timer(
        self,
        session_id: str,
        agenda_item_id: str,
    ) -> SpeakingFlowUpdate:
        """Idempotently continue the current stage and restore its exact timer.

        Speaking stage events and timer creation intentionally use separate durable writes.  A
        restart or transient failure can therefore leave STARTING without its following stage, or
        PREPARING/ACTIVE without its timer. Reconciliation advances only a durable STARTING stage;
        otherwise it reuses (or creates) the timer identity derived from the current projection.
        Any newly derived stage notices are returned to the caller for normal dispatch.
        """

        async with self._lock:
            state = await self.state(session_id, agenda_item_id)
            if state is None:
                return SpeakingFlowUpdate(state=None)
            return await self._reconcile_current_stage_locked(session_id, state)

    async def _finish_main_locked(
        self,
        session_id: str,
        state: SpeakingFlowState,
    ) -> SpeakingFlowUpdate:
        if state.followup_mode is not AgendaFollowupMode.NONE:
            await self._append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKING_FOLLOWUP_STARTED,
                agenda_item_id=state.agenda_item_id,
                payload={"followup_mode": state.followup_mode.value},
            )
            current = await self.state(session_id, state.agenda_item_id)
            if current is None:
                raise RuntimeError("follow-up event was not projected")
            return SpeakingFlowUpdate(
                state=current,
                notices=(_notice(SpeakingFlowNoticeKind.FOLLOWUP_STARTED, current),),
                changed=True,
            )
        await self._append(
            session_id=session_id,
            event_type=MeetingEventType.SPEAKING_FLOW_COMPLETED,
            agenda_item_id=state.agenda_item_id,
            payload={"reason": "MAIN_COMPLETE"},
        )
        current = await self.state(session_id, state.agenda_item_id)
        if current is None:
            raise RuntimeError("completion event was not projected")
        return SpeakingFlowUpdate(
            state=current,
            notices=(_notice(SpeakingFlowNoticeKind.COMPLETED, current),),
            changed=True,
        )

    async def _prepare_next_locked(
        self,
        session_id: str,
        state: SpeakingFlowState,
    ) -> SpeakingFlowUpdate:
        present = frozenset(await self._participants.present_user_ids(session_id))
        prior_notices: list[SpeakingFlowNotice] = []
        if state.mode is AgendaSpeakingMode.PRESENTATION:
            if state.current_index >= 0:
                return await self._finish_main_locked(session_id, state)
            current_user_ids = tuple(
                user_id for user_id in state.presenter_user_ids if user_id in present
            )
            absent_user_ids = tuple(
                user_id for user_id in state.presenter_user_ids if user_id not in present
            )
            if not current_user_ids:
                await self._append(
                    session_id=session_id,
                    event_type=MeetingEventType.SPEAKING_TURN_FINISHED,
                    agenda_item_id=state.agenda_item_id,
                    payload={
                        "turn_index": 0,
                        "current_user_ids": list(state.presenter_user_ids),
                        "planned_seconds": state.presentation_seconds,
                        "reason": "ABSENT",
                    },
                )
                current = await self.state(session_id, state.agenda_item_id)
                if current is None:
                    raise RuntimeError("absence event was not projected")
                completed = await self._finish_main_locked(session_id, current)
                absent_notice = SpeakingFlowNotice(
                    kind=SpeakingFlowNoticeKind.FINISHED,
                    agenda_item_id=state.agenda_item_id,
                    mode=state.mode,
                    current_user_ids=state.presenter_user_ids,
                    current_display_names=tuple(
                        state.display_name(user_id) for user_id in state.presenter_user_ids
                    ),
                    planned_seconds=state.presentation_seconds,
                    reason="ABSENT",
                )
                return SpeakingFlowUpdate(
                    state=completed.state,
                    notices=(absent_notice, *completed.notices),
                    changed=True,
                )
            index = 0
            planned_seconds = state.presentation_seconds
            next_user_ids: tuple[int, ...] = ()
        else:
            index = state.current_index + 1
            while index < len(state.turns) and state.turns[index].user_id not in present:
                turn = state.turns[index]
                await self._append(
                    session_id=session_id,
                    event_type=MeetingEventType.SPEAKING_TURN_FINISHED,
                    agenda_item_id=state.agenda_item_id,
                    payload={
                        "turn_index": index,
                        "current_user_ids": [turn.user_id],
                        "planned_seconds": turn.planned_seconds,
                        "reason": "ABSENT",
                    },
                )
                prior_notices.append(
                    SpeakingFlowNotice(
                        kind=SpeakingFlowNoticeKind.FINISHED,
                        agenda_item_id=state.agenda_item_id,
                        mode=state.mode,
                        current_user_ids=(turn.user_id,),
                        current_display_names=(state.display_name(turn.user_id),),
                        planned_seconds=turn.planned_seconds,
                        reason="ABSENT",
                    )
                )
                index += 1
            if index >= len(state.turns):
                current = await self.state(session_id, state.agenda_item_id)
                if current is None:
                    raise RuntimeError("ordered completion state was not projected")
                completed = await self._finish_main_locked(session_id, current)
                return SpeakingFlowUpdate(
                    state=completed.state,
                    notices=(*prior_notices, *completed.notices),
                    changed=True,
                )
            turn = state.turns[index]
            current_user_ids = (turn.user_id,)
            planned_seconds = turn.planned_seconds
            next_user_ids = next(
                (
                    (candidate.user_id,)
                    for candidate in state.turns[index + 1 :]
                    if candidate.user_id in present
                ),
                (),
            )
            absent_user_ids = ()

        if planned_seconds is None:
            raise RuntimeError("controlled turn has no planned duration")
        await self._append(
            session_id=session_id,
            event_type=MeetingEventType.SPEAKING_PREPARATION_STARTED,
            agenda_item_id=state.agenda_item_id,
            payload={
                "turn_index": index,
                "current_user_ids": list(current_user_ids),
                "next_user_ids": list(next_user_ids),
                "planned_seconds": planned_seconds,
                "absent_user_ids": list(absent_user_ids),
            },
        )
        current = await self.state(session_id, state.agenda_item_id)
        if current is None:
            raise RuntimeError("preparation event was not projected")
        if current.prep_seconds == 0:
            started = await self._start_current_locked(session_id, current)
            return SpeakingFlowUpdate(
                state=started.state,
                notices=(*prior_notices, *started.notices),
                changed=True,
            )
        await self._timers.start_after_handoff_barrier(
            session_id=session_id,
            timer_type=current.timer_type or "",
            duration_seconds=current.prep_seconds,
        )
        return SpeakingFlowUpdate(
            state=current,
            notices=(
                *prior_notices,
                _notice(SpeakingFlowNoticeKind.PREPARING, current),
            ),
            changed=True,
        )

    async def _start_current_locked(
        self,
        session_id: str,
        state: SpeakingFlowState,
    ) -> SpeakingFlowUpdate:
        present = frozenset(await self._participants.present_user_ids(session_id))
        available_user_ids = tuple(
            user_id for user_id in state.current_user_ids if user_id in present
        )
        if not available_user_ids:
            return await self._finish_current_locked(
                session_id,
                state,
                actor_user_id=None,
                source="VOICE_STATE",
                reason="ABSENT",
                cancel_timer=False,
            )
        await self._append(
            session_id=session_id,
            event_type=MeetingEventType.SPEAKING_TURN_STARTED,
            agenda_item_id=state.agenda_item_id,
            payload={
                "turn_index": state.current_index,
                "current_user_ids": list(available_user_ids),
                "next_user_ids": list(state.next_user_ids),
                "planned_seconds": state.planned_seconds,
            },
        )
        current = await self.state(session_id, state.agenda_item_id)
        if current is None or current.timer_type is None or current.timer_seconds is None:
            raise RuntimeError("active turn event was not projected")
        await self._timers.start_after_handoff_barrier(
            session_id=session_id,
            timer_type=current.timer_type,
            duration_seconds=current.timer_seconds,
        )
        return SpeakingFlowUpdate(
            state=current,
            notices=(_notice(SpeakingFlowNoticeKind.STARTED, current),),
            changed=True,
        )

    async def handle_timer_notice(self, notice: TimerNotice) -> SpeakingFlowUpdate:
        marker = "FLOW:SPEAKING_"
        if not notice.timer_type.startswith(marker):
            return SpeakingFlowUpdate(state=None)
        parts = notice.timer_type.split(":")
        if len(parts) != 4:
            return SpeakingFlowUpdate(state=None)
        agenda_item_id = parts[2]
        async with self._lock:
            state = await self.state(notice.session_id, agenda_item_id)
            if state is None:
                return SpeakingFlowUpdate(state=None)
            if state.timer_type != notice.timer_type:
                return await self._reconcile_current_stage_locked(notice.session_id, state)
            if notice.event_type in {
                MeetingEventType.TIMER_REMINDER_60,
                MeetingEventType.TIMER_REMINDER_10,
            }:
                if state.stage is not SpeakingFlowStage.ACTIVE:
                    return SpeakingFlowUpdate(state=state)
                return SpeakingFlowUpdate(
                    state=state,
                    notices=(
                        _notice(
                            SpeakingFlowNoticeKind.REMINDER,
                            state,
                            remaining_seconds=notice.remaining_seconds,
                        ),
                    ),
                )
            if notice.event_type is not MeetingEventType.TIMER_EXPIRED:
                return SpeakingFlowUpdate(state=state)
            if state.stage is SpeakingFlowStage.PREPARING:
                return await self._start_current_locked(notice.session_id, state)
            if state.stage is SpeakingFlowStage.ACTIVE:
                return await self._finish_current_locked(
                    notice.session_id,
                    state,
                    actor_user_id=None,
                    source="TIMER",
                    reason="TIME_EXPIRED",
                    cancel_timer=False,
                )
            return SpeakingFlowUpdate(state=state)

    async def _finish_current_locked(
        self,
        session_id: str,
        state: SpeakingFlowState,
        *,
        actor_user_id: int | None,
        source: str,
        reason: str,
        cancel_timer: bool,
    ) -> SpeakingFlowUpdate:
        if state.stage not in {SpeakingFlowStage.PREPARING, SpeakingFlowStage.ACTIVE}:
            return SpeakingFlowUpdate(state=state)
        if cancel_timer and state.timer_type is not None:
            await self._timers.cancel_type(
                session_id=session_id,
                timer_type=state.timer_type,
            )
        await self._append(
            session_id=session_id,
            event_type=MeetingEventType.SPEAKING_TURN_FINISHED,
            agenda_item_id=state.agenda_item_id,
            actor_user_id=actor_user_id,
            source=source,
            payload={
                "turn_index": state.current_index,
                "current_user_ids": list(state.current_user_ids),
                "planned_seconds": state.planned_seconds,
                "extension_seconds": state.extension_seconds,
                "reason": reason,
            },
        )
        finished_notice = _notice(
            SpeakingFlowNoticeKind.FINISHED,
            state,
            reason=reason,
        )
        current = await self.state(session_id, state.agenda_item_id)
        if current is None:
            raise RuntimeError("turn finish event was not projected")
        following = await self._prepare_next_locked(session_id, current)
        return SpeakingFlowUpdate(
            state=following.state,
            notices=(finished_notice, *following.notices),
            changed=True,
        )

    async def finish_current(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        reason: str,
        source: str,
    ) -> SpeakingFlowUpdate:
        async with self._lock:
            state = await self.state(session_id, agenda_item_id)
            if state is None:
                return SpeakingFlowUpdate(state=None)
            return await self._finish_current_locked(
                session_id,
                state,
                actor_user_id=actor_user_id,
                source=source,
                reason=reason,
                cancel_timer=True,
            )

    async def extend_current(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        additional_seconds: int,
        source: str,
    ) -> SpeakingFlowUpdate:
        if isinstance(additional_seconds, bool) or not 1 <= additional_seconds <= 3600:
            raise ValueError("extension must be between 1 and 3600 seconds")
        async with self._lock:
            state = await self.state(session_id, agenda_item_id)
            if (
                state is None
                or state.stage is not SpeakingFlowStage.ACTIVE
                or state.timer_type is None
            ):
                return SpeakingFlowUpdate(state=state)
            timer = await self._timers.extend_type(
                session_id=session_id,
                timer_type=state.timer_type,
                additional_seconds=additional_seconds,
            )
            if timer is None:
                return SpeakingFlowUpdate(state=state)
            await self._append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKING_TURN_EXTENDED,
                agenda_item_id=agenda_item_id,
                actor_user_id=actor_user_id,
                source=source,
                payload={
                    "turn_index": state.current_index,
                    "current_user_ids": list(state.current_user_ids),
                    "additional_seconds": additional_seconds,
                    "deadline_at": timer.deadline_at.isoformat(),
                },
            )
            current = await self.state(session_id, agenda_item_id)
            if current is None:
                raise RuntimeError("extension event was not projected")
            return SpeakingFlowUpdate(
                state=current,
                notices=(
                    _notice(
                        SpeakingFlowNoticeKind.EXTENDED,
                        current,
                        additional_seconds=additional_seconds,
                    ),
                ),
                changed=True,
            )

    async def finish_followup(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        source: str,
    ) -> SpeakingFlowUpdate:
        async with self._lock:
            state = await self.state(session_id, agenda_item_id)
            if state is None or state.stage is not SpeakingFlowStage.FOLLOWUP:
                return SpeakingFlowUpdate(state=state)
            await self._append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKING_FOLLOWUP_FINISHED,
                agenda_item_id=agenda_item_id,
                actor_user_id=actor_user_id,
                source=source,
                payload={"followup_mode": state.followup_mode.value},
            )
            await self._append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKING_FLOW_COMPLETED,
                agenda_item_id=agenda_item_id,
                actor_user_id=actor_user_id,
                source=source,
                payload={"reason": "FOLLOWUP_COMPLETE"},
            )
            current = await self.state(session_id, agenda_item_id)
            if current is None:
                raise RuntimeError("follow-up completion was not projected")
            return SpeakingFlowUpdate(
                state=current,
                notices=(_notice(SpeakingFlowNoticeKind.COMPLETED, current),),
                changed=True,
            )

    async def abort(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        source: str,
        completion_reason: str,
    ) -> SpeakingFlowUpdate:
        """Close the current floor without advancing into another controlled turn."""

        if completion_reason not in {"AGENDA_ADVANCED", "MEETING_STOPPED"}:
            raise ValueError("controlled speaking abort reason is invalid")
        async with self._lock:
            state = await self.state(session_id, agenda_item_id)
            if state is None or state.stage is SpeakingFlowStage.COMPLETED:
                return SpeakingFlowUpdate(state=state)
            if state.timer_type is not None:
                await self._timers.cancel_type(
                    session_id=session_id,
                    timer_type=state.timer_type,
                )
            if state.stage in {SpeakingFlowStage.PREPARING, SpeakingFlowStage.ACTIVE}:
                await self._append(
                    session_id=session_id,
                    event_type=MeetingEventType.SPEAKING_TURN_FINISHED,
                    agenda_item_id=agenda_item_id,
                    actor_user_id=actor_user_id,
                    source=source,
                    payload={
                        "turn_index": state.current_index,
                        "current_user_ids": list(state.current_user_ids),
                        "planned_seconds": state.planned_seconds,
                        "extension_seconds": state.extension_seconds,
                        "reason": (
                            "HOST_ENDED_EARLY"
                            if state.stage is SpeakingFlowStage.ACTIVE
                            else "HOST_SKIPPED"
                        ),
                    },
                )
            elif state.stage is SpeakingFlowStage.FOLLOWUP:
                await self._append(
                    session_id=session_id,
                    event_type=MeetingEventType.SPEAKING_FOLLOWUP_FINISHED,
                    agenda_item_id=agenda_item_id,
                    actor_user_id=actor_user_id,
                    source=source,
                    payload={"followup_mode": state.followup_mode.value},
                )
            await self._append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKING_FLOW_COMPLETED,
                agenda_item_id=agenda_item_id,
                actor_user_id=actor_user_id,
                source=source,
                payload={"reason": completion_reason},
            )
            current = await self.state(session_id, agenda_item_id)
            if current is None:
                raise RuntimeError("aborted speaking flow was not projected")
            return SpeakingFlowUpdate(state=current, changed=True)
