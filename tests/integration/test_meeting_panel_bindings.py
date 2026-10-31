from pathlib import Path

import pytest

from lingxu.domain.enums import MeetingEventType
from lingxu.domain.errors import MeetingPermissionError, SessionNotFoundError
from lingxu.meeting.panels import MeetingPanelBindingService
from lingxu.storage.database import Database
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_panel_bindings_are_event_sourced_and_latest_host_dm_wins(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "panels.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session = await MeetingSessionService(sessions, SessionFiles(tmp_path / "sessions")).create(
        guild_id=77,
        creator_user_id=100,
    )
    events = MeetingEventStore(database.sessions)
    panels = MeetingPanelBindingService(sessions=sessions, events=events)

    with pytest.raises(MeetingPermissionError):
        await panels.bind_public(
            session_id=session.id,
            channel_id=700,
            message_id=701,
            actor_user_id=999,
        )
    await panels.bind_public(
        session_id=session.id,
        channel_id=700,
        message_id=701,
        actor_user_id=100,
    )
    with pytest.raises(MeetingPermissionError):
        await panels.bind_host(
            session_id=session.id,
            host_user_id=999,
            channel_id=800,
            message_id=801,
        )
    await panels.bind_host(
        session_id=session.id,
        host_user_id=100,
        channel_id=800,
        message_id=801,
    )
    await panels.bind_host(
        session_id=session.id,
        host_user_id=100,
        channel_id=800,
        message_id=802,
    )

    binding = await panels.binding(session.id)
    assert binding.public_channel_id == 700
    assert binding.public_message_id == 701
    assert binding.host_user_id == 100
    assert binding.host_dm_message_id == 802
    assert (await panels.resolve_host_panel(message_id=802, user_id=100)).guild_id == 77
    with pytest.raises(SessionNotFoundError):
        await panels.resolve_host_panel(message_id=801, user_id=100)
    with pytest.raises(SessionNotFoundError):
        await panels.resolve_host_panel(message_id=802, user_id=999)

    event_types = [event.event_type for event in await events.list(session.id)]
    assert event_types == [
        MeetingEventType.PUBLIC_STATUS_PANEL_BOUND.value,
        MeetingEventType.HOST_CONTROL_PANEL_BOUND.value,
        MeetingEventType.HOST_CONTROL_PANEL_BOUND.value,
    ]
    await database.close()
