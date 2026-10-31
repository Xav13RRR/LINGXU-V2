from pathlib import Path

import pytest

from lingxu.storage.database import Database
from lingxu.storage.pronunciation_repository import (
    PronunciationAliasRepository,
    normalise_pronunciation_alias,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pronunciation_alias_is_self_record_scoped_by_guild(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'pronunciations.db').as_posix()}")
    await database.create_schema()
    repository = PronunciationAliasRepository(database.sessions)

    first = await repository.set(guild_id=10, user_id=20, alias="  XIAO   Ming  ")
    await repository.set(guild_id=11, user_id=20, alias="小明")
    await repository.set(guild_id=10, user_id=21, alias="另一位")

    assert first.alias == "XIAO Ming"
    assert await repository.names_for(guild_id=10, user_ids=(20, 21, 20)) == {
        20: "XIAO Ming",
        21: "另一位",
    }
    assert (await repository.get(guild_id=11, user_id=20)).alias == "小明"  # type: ignore[union-attr]
    assert await repository.clear(guild_id=10, user_id=20)
    assert await repository.get(guild_id=10, user_id=20) is None
    assert not await repository.clear(guild_id=10, user_id=20)
    await database.close()


def test_pronunciation_alias_rejects_invisible_format_controls() -> None:
    with pytest.raises(ValueError, match="不可见"):
        normalise_pronunciation_alias("小\u200b明")
    with pytest.raises(ValueError, match="1 到 80"):
        normalise_pronunciation_alias("   ")
