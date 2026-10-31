"""Event-sourced agenda mute-enforcement control state."""

from __future__ import annotations

from dataclasses import dataclass, replace

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.models import AgendaItemRecord, MeetingEventRecord, MeetingMuteLeaseRecord
from lingxu.meeting.components import TimerService
from lingxu.storage.repositories import MeetingEventStore


@dataclass(frozen=True, slots=True)
class MuteEnforcementState:
    """Recoverable host operations kept separate from Meeting pause state."""

    agenda_item_id: str
    enabled: bool = True
    paused: bool = False
    downgraded: bool = False
    pause_reason: str | None = None
    pause_timer_type: str | None = None
    pause_timer_owned: bool = False
    exception_user_ids: tuple[int, ...] = ()
    host_intervention_active: bool = False
    intervention_timer_type: str | None = None
    intervention_timer_owned: bool = False


def project_mute_enforcement(
    events: tuple[MeetingEventRecord, ...],
    agenda_item_id: str,
) -> MuteEnforcementState | None:
    """Replay only immutable control facts for one agenda item."""

    state: MuteEnforcementState | None = None
    for event in events:
        payload = event.payload
        timer_type = str(payload.get("timer_type", ""))
        if (
            state is not None
            and event.event_type == MeetingEventType.TIMER_PAUSED.value
            and f":{agenda_item_id}:" in timer_type
        ):
            reason = str(payload.get("reason", ""))
            if reason.startswith("MUTE_ENFORCEMENT:"):
                state = replace(
                    state,
                    paused=True,
                    pause_reason=reason.removeprefix("MUTE_ENFORCEMENT:"),
                    pause_timer_type=timer_type,
                    pause_timer_owned=True,
                )
            elif reason == "HOST_INTERVENTION":
                state = replace(
                    state,
                    host_intervention_active=True,
                    intervention_timer_type=timer_type,
                    intervention_timer_owned=True,
                )
        if payload.get("agenda_item_id") != agenda_item_id:
            continue
        if event.event_type == MeetingEventType.MUTE_ENFORCEMENT_STARTED.value:
            state = MuteEnforcementState(agenda_item_id=agenda_item_id)
        elif state is None:
            continue
        elif event.event_type == MeetingEventType.MUTE_ENFORCEMENT_PAUSED.value:
            state = replace(
                state,
                paused=True,
                pause_reason=str(payload.get("reason", "HOST")),
                pause_timer_type=(
                    str(payload["timer_type"]) if payload.get("timer_type") else None
                ),
                pause_timer_owned=bool(payload.get("timer_owned", False)),
            )
        elif event.event_type == MeetingEventType.MUTE_ENFORCEMENT_RESUMED.value:
            state = replace(
                state,
                paused=False,
                pause_reason=None,
                pause_timer_type=None,
                pause_timer_owned=False,
            )
        elif event.event_type in {
            MeetingEventType.MUTE_ENFORCEMENT_ENDED.value,
            MeetingEventType.MUTE_ENFORCEMENT_DOWNGRADED.value,
            MeetingEventType.EMERGENCY_MUTE_RELEASED.value,
        }:
            state = replace(
                state,
                enabled=False,
                paused=False,
                downgraded=event.event_type != MeetingEventType.MUTE_ENFORCEMENT_ENDED.value,
                pause_reason=None,
                pause_timer_type=None,
                pause_timer_owned=False,
                host_intervention_active=False,
                intervention_timer_type=None,
                intervention_timer_owned=False,
            )
        elif event.event_type == MeetingEventType.FLOOR_EXCEPTION_GRANTED.value:
            user_id = int(payload["user_id"])
            state = replace(
                state,
                exception_user_ids=tuple(dict.fromkeys((*state.exception_user_ids, user_id))),
            )
        elif event.event_type == MeetingEventType.FLOOR_EXCEPTION_REVOKED.value:
            user_id = int(payload["user_id"])
            state = replace(
                state,
                exception_user_ids=tuple(
                    value for value in state.exception_user_ids if value != user_id
                ),
            )
        elif event.event_type == MeetingEventType.HOST_INTERVENTION_STARTED.value:
            state = replace(
                state,
                host_intervention_active=True,
                intervention_timer_type=(
                    str(payload["timer_type"]) if payload.get("timer_type") else None
                ),
                intervention_timer_owned=bool(payload.get("timer_owned", False)),
            )
        elif event.event_type == MeetingEventType.HOST_INTERVENTION_ENDED.value:
            state = replace(
                state,
                host_intervention_active=False,
                intervention_timer_type=None,
                intervention_timer_owned=False,
            )
    return state


class MuteEnforcementService:
    """Persist host controls and exact timer ownership without Discord access."""

    def __init__(self, *, events: MeetingEventStore, timers: TimerService) -> None:
        self._events = events
        self._timers = timers

    async def state(
        self,
        session_id: str,
        agenda_item_id: str,
    ) -> MuteEnforcementState | None:
        return project_mute_enforcement(await self._events.list(session_id), agenda_item_id)

    async def _append(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        event_type: MeetingEventType,
        actor_user_id: int | None = None,
        source: str = "SYSTEM",
        payload: dict[str, object] | None = None,
    ) -> None:
        await self._events.append(
            session_id=session_id,
            event_type=event_type.value,
            actor_user_id=actor_user_id,
            source=source,
            payload={"agenda_item_id": agenda_item_id, **(payload or {})},
        )

    async def ensure_started(
        self,
        *,
        session_id: str,
        item: AgendaItemRecord,
    ) -> MuteEnforcementState | None:
        if not item.speaking_policy.mute_enforcement_requested:
            return None
        state = await self.state(session_id, item.id)
        if state is None or (not state.enabled and not state.downgraded):
            await self._append(
                session_id=session_id,
                agenda_item_id=item.id,
                event_type=MeetingEventType.MUTE_ENFORCEMENT_STARTED,
                payload={"mode": item.speaking_policy.mode.value},
            )
            state = await self.state(session_id, item.id)
        return state

    async def pause(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        reason: str,
        timer_type: str | None,
        actor_user_id: int | None,
        source: str,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled or state.paused:
            return False
        owned = (
            await self._timers.pause_type(
                session_id=session_id,
                timer_type=timer_type,
                reason=f"MUTE_ENFORCEMENT:{reason}",
            )
            if timer_type is not None
            else False
        )
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.MUTE_ENFORCEMENT_PAUSED,
            actor_user_id=actor_user_id,
            source=source,
            payload={"reason": reason, "timer_type": timer_type, "timer_owned": owned},
        )
        return True

    async def resume(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        source: str,
        resume_timer: bool,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled or not state.paused:
            return False
        if resume_timer and state.pause_timer_owned and state.pause_timer_type is not None:
            await self._timers.resume_type(
                session_id=session_id,
                timer_type=state.pause_timer_type,
                reason="MUTE_ENFORCEMENT_RESUMED",
            )
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.MUTE_ENFORCEMENT_RESUMED,
            actor_user_id=actor_user_id,
            source=source,
        )
        return True

    async def downgrade(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        source: str,
        emergency: bool = False,
        resume_timer: bool = True,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled:
            return False
        if resume_timer and state.pause_timer_owned and state.pause_timer_type is not None:
            await self._timers.resume_type(
                session_id=session_id,
                timer_type=state.pause_timer_type,
                reason="MUTE_ENFORCEMENT_DOWNGRADED",
            )
        if (
            resume_timer
            and state.intervention_timer_owned
            and state.intervention_timer_type is not None
        ):
            await self._timers.resume_type(
                session_id=session_id,
                timer_type=state.intervention_timer_type,
                reason="HOST_INTERVENTION_DOWNGRADED",
            )
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=(
                MeetingEventType.EMERGENCY_MUTE_RELEASED
                if emergency
                else MeetingEventType.MUTE_ENFORCEMENT_DOWNGRADED
            ),
            actor_user_id=actor_user_id,
            source=source,
        )
        return True

    async def maintain_owned_timer_pauses(
        self,
        *,
        session_id: str,
        state: MuteEnforcementState,
    ) -> None:
        """Re-freeze operation-owned timers after an independent Meeting resume."""

        if state.paused and state.pause_timer_owned and state.pause_timer_type is not None:
            await self._timers.pause_type(
                session_id=session_id,
                timer_type=state.pause_timer_type,
                reason=f"MUTE_ENFORCEMENT:{state.pause_reason or 'PAUSED'}",
            )
        if (
            state.host_intervention_active
            and state.intervention_timer_owned
            and state.intervention_timer_type is not None
        ):
            await self._timers.pause_type(
                session_id=session_id,
                timer_type=state.intervention_timer_type,
                reason="HOST_INTERVENTION",
            )

    async def end(self, *, session_id: str, agenda_item_id: str) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled:
            return False
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.MUTE_ENFORCEMENT_ENDED,
        )
        return True

    async def set_exception(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        user_id: int,
        granted: bool,
        actor_user_id: int,
        source: str,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled:
            return False
        exists = user_id in state.exception_user_ids
        if exists == granted:
            return False
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=(
                MeetingEventType.FLOOR_EXCEPTION_GRANTED
                if granted
                else MeetingEventType.FLOOR_EXCEPTION_REVOKED
            ),
            actor_user_id=actor_user_id,
            source=source,
            payload={"user_id": user_id},
        )
        return True

    async def start_intervention(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        timer_type: str | None,
        actor_user_id: int,
        source: str,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.enabled or state.paused or state.host_intervention_active:
            return False
        owned = (
            await self._timers.pause_type(
                session_id=session_id,
                timer_type=timer_type,
                reason="HOST_INTERVENTION",
            )
            if timer_type is not None
            else False
        )
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.HOST_INTERVENTION_STARTED,
            actor_user_id=actor_user_id,
            source=source,
            payload={"timer_type": timer_type, "timer_owned": owned},
        )
        return True

    async def end_intervention(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        actor_user_id: int,
        source: str,
        resume_timer: bool,
    ) -> bool:
        state = await self.state(session_id, agenda_item_id)
        if state is None or not state.host_intervention_active:
            return False
        if (
            resume_timer
            and state.intervention_timer_owned
            and state.intervention_timer_type is not None
        ):
            await self._timers.resume_type(
                session_id=session_id,
                timer_type=state.intervention_timer_type,
                reason="HOST_INTERVENTION_ENDED",
            )
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.HOST_INTERVENTION_ENDED,
            actor_user_id=actor_user_id,
            source=source,
        )
        return True

    async def record_lease(
        self,
        *,
        lease: MeetingMuteLeaseRecord,
        event_type: MeetingEventType,
        reason: str,
    ) -> None:
        await self._append(
            session_id=lease.session_id,
            agenda_item_id=lease.agenda_item_id,
            event_type=event_type,
            payload={
                "lease_id": lease.id,
                "user_id": lease.user_id,
                "original_server_mute": lease.original_server_mute,
                "ownership": lease.ownership.value,
                "reason": reason,
            },
        )

    async def record_permission_lost(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        operation: str,
    ) -> None:
        await self._append(
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            event_type=MeetingEventType.MUTE_PERMISSION_LOST,
            payload={"operation": operation},
        )
