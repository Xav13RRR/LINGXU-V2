# ruff: noqa: RUF001
import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from lingxu.domain.enums import ServiceCapacityMode
from lingxu.domain.errors import ServiceCapacityFullError
from lingxu.storage.database import Database
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository


async def _repository(tmp_path: Path) -> tuple[Database, ServiceCapacityRepository]:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'capacity.db').as_posix()}")
    await database.create_schema()
    return database, ServiceCapacityRepository(database.sessions)


@pytest.mark.asyncio
async def test_capacity_is_global_idempotent_and_token_safe(tmp_path: Path) -> None:
    database, repository = await _repository(tmp_path)
    try:
        lease = await repository.claim(
            guild_id=101,
            mode=ServiceCapacityMode.MEETING,
            owner_id="meeting-one",
        )
        assert (
            await repository.claim(
                guild_id=101,
                mode=ServiceCapacityMode.MEETING,
                owner_id="meeting-one",
            )
            == lease
        )

        with pytest.raises(ServiceCapacityFullError) as captured:
            await repository.claim(
                guild_id=202,
                mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
                owner_id="assistant-two",
            )
        assert str(captured.value) == (
            "服务容量已满。聆序目前正在其它 Discord 服务器提供服务，请稍后再试。"
        )

        assert not await repository.release_owner(
            guild_id=202,
            mode=ServiceCapacityMode.MEETING,
            owner_id="meeting-one",
            reason="WRONG_GUILD",
        )
        assert await repository.current() == lease
        assert await repository.release(lease, reason="PUBLISHED")
        assert await repository.current() is None
        assert [event.event_type for event in await repository.events(lease.id)] == [
            "ACQUIRED",
            "RELEASED",
        ]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_parallel_guild_claims_have_exactly_one_winner(tmp_path: Path) -> None:
    database, repository = await _repository(tmp_path)
    try:
        results = await asyncio.gather(
            repository.claim(
                guild_id=101,
                mode=ServiceCapacityMode.PROBE,
                owner_id="probe-one",
            ),
            repository.claim(
                guild_id=202,
                mode=ServiceCapacityMode.PROBE,
                owner_id="probe-two",
            ),
            return_exceptions=True,
        )
        winners = [result for result in results if not isinstance(result, BaseException)]
        failures = [result for result in results if isinstance(result, BaseException)]
        assert len(winners) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ServiceCapacityFullError)
        assert (await repository.current()).guild_id == winners[0].guild_id  # type: ignore[union-attr]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_capacity_event_stream_recovery_and_append_only_guards(tmp_path: Path) -> None:
    database, repository = await _repository(tmp_path)
    try:
        lease = await repository.claim(
            guild_id=101,
            mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
            owner_id="resident-one",
        )
        recovered = await repository.record_recovered(lease)
        assert recovered.event_type == "RECOVERED"
        assert [event.sequence_number for event in await repository.events(lease.id)] == [1, 2]

        async with database.sessions() as session:
            with pytest.raises(DatabaseError):
                await session.execute(
                    text(
                        "UPDATE service_capacity_events "
                        "SET reason = 'MUTATED' WHERE lease_id = :lease_id"
                    ),
                    {"lease_id": lease.id},
                )
    finally:
        await database.close()
