from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from lingxu.domain.enums import ResidentAssistantStatus
from lingxu.domain.errors import VoiceModeConflictError
from lingxu.storage.database import Database
from lingxu.storage.resident_assistant_repository import ResidentAssistantRepository


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@pytest.mark.asyncio
async def test_resident_lease_is_event_sourced_recoverable_and_append_only(tmp_path: Path) -> None:
    database = Database.open(_database_url(tmp_path / "resident.db"))
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)

    created = await repository.create_active(
        session_id="01K00000000000000000000000",
        guild_id=11,
        channel_id=22,
        creator_user_id=33,
    )
    assert created.status is ResidentAssistantStatus.ACTIVE
    assert created.next_sequence == 1
    assert await repository.active_for_guild(11) == created
    with pytest.raises(VoiceModeConflictError):
        await repository.create_active(
            session_id="01K00000000000000000000001",
            guild_id=11,
            channel_id=23,
            creator_user_id=34,
        )

    recovered = await repository.record_recovered(created.id)
    assert recovered.sequence_number == 2
    stopped = await repository.stop_active(
        session_id=created.id,
        actor_user_id=33,
        reason="COMMAND",
    )
    assert stopped is not None
    assert stopped.status is ResidentAssistantStatus.STOPPED
    assert await repository.active_for_guild(11) is None
    events = await repository.events(created.id)
    assert [(event.sequence_number, event.event_type) for event in events] == [
        (1, "STARTED"),
        (2, "RECOVERED"),
        (3, "STOPPED"),
    ]

    async with database.sessions() as db:
        with pytest.raises(DBAPIError):
            await db.execute(
                text(
                    "UPDATE resident_assistant_events SET reason='MUTATED' "
                    "WHERE session_id=:session_id"
                ),
                {"session_id": created.id},
            )
            await db.commit()
        await db.rollback()
        with pytest.raises(DBAPIError):
            await db.execute(
                text("DELETE FROM resident_assistant_events WHERE session_id=:session_id"),
                {"session_id": created.id},
            )
            await db.commit()
        await db.rollback()

    await database.close()
