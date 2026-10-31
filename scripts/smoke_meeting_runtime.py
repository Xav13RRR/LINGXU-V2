"""Content-safe deployed Phase 6 runtime/recovery smoke with cleanup."""

from __future__ import annotations

import asyncio
import json
import time

from lingxu.config import Settings
from lingxu.domain.enums import MeetingState
from lingxu.domain.errors import SessionNotFoundError
from lingxu.meeting.runtime import MeetingRuntime
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles


async def _run() -> dict[str, object]:
    settings = Settings()
    database = Database.open(settings.database_url)
    sessions = MeetingSessionRepository(database.sessions)
    events = MeetingEventStore(database.sessions)
    snapshots = MeetingSnapshotStore(database.sessions)
    service = MeetingSessionService(sessions, SessionFiles(settings.data_root))
    session = await service.create(
        guild_id=8_000_000_000_000_000 + time.time_ns() % 999_999_999_999_999,
        creator_user_id=7_000_000_000_000_001,
    )
    runtime = MeetingRuntime(
        sessions=sessions,
        events=events,
        snapshots=snapshots,
        snapshot_interval=3,
    )
    state = MeetingState.CREATED
    event_count = 0
    recovered = False
    cleaned = False
    try:
        await runtime.initialise(session.id, actor_user_id=session.creator_user_id)
        current = await runtime.start(
            session.id,
            actor_user_id=session.creator_user_id,
            source="SLASH_COMMAND",
        )
        while current.session.state is not MeetingState.PUBLISHED:
            current = await runtime.advance(session.id)
        state = current.session.state
        event_count = len(await events.list(session.id))
        recovered = (await runtime.recover(session.id)).session.state is state
    finally:
        await service.delete(session_id=session.id, actor_user_id=session.creator_user_id)
        try:
            await sessions.get(session.id)
        except SessionNotFoundError:
            cleaned = not await asyncio.to_thread(
                SessionFiles(settings.data_root).directory(session.id).exists
            )
        await database.close()
    return {
        "final_state": state.value,
        "event_count": event_count,
        "recovered": recovered,
        "cleaned": cleaned,
    }


def main() -> None:
    result = asyncio.run(_run())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not (
        result["final_state"] == MeetingState.PUBLISHED.value
        and result["event_count"] == 15
        and result["recovered"]
        and result["cleaned"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
