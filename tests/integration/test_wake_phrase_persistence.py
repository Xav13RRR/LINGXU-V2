from pathlib import Path

import pytest

from lingxu.bot.wakeup_commands import GuildWakePhraseCoordinator
from lingxu.domain.wake_phrase import DEFAULT_WAKE_PHRASE
from lingxu.storage.database import Database
from lingxu.storage.guild_wake_phrase_repository import GuildWakePhraseRepository
from lingxu.storage.tables import GuildWakeupSettingsRow


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wake_phrase_default_override_reset_and_guild_isolation(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'wake-phrases.db').as_posix()}")
    await database.create_schema()
    repository = GuildWakePhraseRepository(database.sessions)

    initial = await repository.get(10)
    assert initial.phrase == DEFAULT_WAKE_PHRASE
    assert not initial.is_override
    assert initial.updated_by_user_id is None
    assert initial.updated_at is None
    async with database.sessions() as db:
        assert await db.get(GuildWakeupSettingsRow, 10) is None

    first = await repository.set(
        guild_id=10,
        phrase="呼叫会议助手",
        updated_by_user_id=20,
    )
    second = await repository.set(
        guild_id=11,
        phrase="请联系会议助手",
        updated_by_user_id=21,
    )
    assert first.phrase == "呼叫会议助手"
    assert first.is_override
    assert first.updated_by_user_id == 20
    assert first.updated_at is not None
    assert second.guild_id == 11
    assert (await repository.get(10)).phrase == "呼叫会议助手"
    assert (await repository.get(11)).phrase == "请联系会议助手"

    updated = await repository.set(
        guild_id=10,
        phrase="新的指挥中心",
        updated_by_user_id=None,
    )
    assert updated.phrase == "新的指挥中心"
    assert updated.updated_by_user_id is None

    reset = await repository.reset(guild_id=10)
    assert reset.phrase == DEFAULT_WAKE_PHRASE
    assert not reset.is_override
    assert (await repository.get(10)) == reset
    assert (await repository.get(11)).phrase == "请联系会议助手"
    assert await repository.reset(guild_id=10) == reset
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_override_is_rejected_and_reset_remains_repairable(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'wake-corrupt.db').as_posix()}")
    await database.create_schema()
    repository = GuildWakePhraseRepository(database.sessions)
    async with database.sessions() as db, db.begin():
        db.add(
            GuildWakeupSettingsRow(
                guild_id=10,
                wake_phrase="unsupported phrase",
                updated_by_user_id=20,
            )
        )

    with pytest.raises(ValueError, match="只能包含连续汉字"):
        await repository.get(10)

    coordinator = GuildWakePhraseCoordinator(repository=repository, runtimes=())
    repaired = await coordinator.reset(guild_id=10)

    assert repaired.phrase == DEFAULT_WAKE_PHRASE
    assert not repaired.is_override
    assert await repository.get(10) == repaired
    await database.close()
