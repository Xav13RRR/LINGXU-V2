"""Durable Discord panel bindings projected from the Meeting event stream."""

from __future__ import annotations

from dataclasses import dataclass, replace

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.errors import MeetingPermissionError, SessionNotFoundError
from lingxu.domain.models import MeetingEventRecord, MeetingSessionRecord
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository


def _snowflake(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive Discord snowflake")
    return value


@dataclass(frozen=True, slots=True)
class MeetingPanelBinding:
    session_id: str
    public_channel_id: int | None = None
    public_message_id: int | None = None
    host_user_id: int | None = None
    host_dm_channel_id: int | None = None
    host_dm_message_id: int | None = None


def project_panel_binding(
    events: tuple[MeetingEventRecord, ...],
    session_id: str,
) -> MeetingPanelBinding:
    """Use only the latest binding fact for each independently reissuable panel."""

    binding = MeetingPanelBinding(session_id=session_id)
    for event in events:
        if event.event_type == MeetingEventType.PUBLIC_STATUS_PANEL_BOUND.value:
            binding = replace(
                binding,
                public_channel_id=_snowflake(event.payload.get("channel_id"), "channel_id"),
                public_message_id=_snowflake(event.payload.get("message_id"), "message_id"),
            )
        elif event.event_type == MeetingEventType.HOST_CONTROL_PANEL_BOUND.value:
            binding = replace(
                binding,
                host_user_id=_snowflake(event.payload.get("host_user_id"), "host_user_id"),
                host_dm_channel_id=_snowflake(
                    event.payload.get("channel_id"),
                    "channel_id",
                ),
                host_dm_message_id=_snowflake(
                    event.payload.get("message_id"),
                    "message_id",
                ),
            )
    return binding


class MeetingPanelBindingService:
    """Append panel bindings and authorize buttons against the latest host DM."""

    def __init__(
        self,
        *,
        sessions: MeetingSessionRepository,
        events: MeetingEventStore,
    ) -> None:
        self._sessions = sessions
        self._events = events

    async def binding(self, session_id: str) -> MeetingPanelBinding:
        return project_panel_binding(await self._events.list(session_id), session_id)

    async def bind_public(
        self,
        *,
        session_id: str,
        channel_id: int,
        message_id: int,
        actor_user_id: int,
    ) -> MeetingPanelBinding:
        session = await self._sessions.get(session_id)
        if actor_user_id != session.creator_user_id:
            raise MeetingPermissionError("only the Meeting host may bind the status panel")
        await self._events.append(
            session_id=session_id,
            event_type=MeetingEventType.PUBLIC_STATUS_PANEL_BOUND.value,
            actor_user_id=actor_user_id,
            source="DISCORD",
            payload={
                "channel_id": _snowflake(channel_id, "channel_id"),
                "message_id": _snowflake(message_id, "message_id"),
            },
        )
        return await self.binding(session_id)

    async def bind_host(
        self,
        *,
        session_id: str,
        host_user_id: int,
        channel_id: int,
        message_id: int,
    ) -> MeetingPanelBinding:
        session = await self._sessions.get(session_id)
        if host_user_id != session.creator_user_id:
            raise MeetingPermissionError("host panel recipient does not own this Meeting")
        await self._events.append(
            session_id=session_id,
            event_type=MeetingEventType.HOST_CONTROL_PANEL_BOUND.value,
            actor_user_id=host_user_id,
            source="DISCORD_DM",
            payload={
                "host_user_id": _snowflake(host_user_id, "host_user_id"),
                "channel_id": _snowflake(channel_id, "channel_id"),
                "message_id": _snowflake(message_id, "message_id"),
            },
        )
        return await self.binding(session_id)

    async def resolve_host_panel(
        self,
        *,
        message_id: int,
        user_id: int,
    ) -> MeetingSessionRecord:
        selected_message_id = _snowflake(message_id, "message_id")
        selected_user_id = _snowflake(user_id, "user_id")
        for session in await self._sessions.list_recoverable():
            if session.creator_user_id != selected_user_id:
                continue
            binding = await self.binding(session.id)
            if (
                binding.host_user_id == selected_user_id
                and binding.host_dm_message_id == selected_message_id
            ):
                return session
        raise SessionNotFoundError("this is not the current host control panel")
