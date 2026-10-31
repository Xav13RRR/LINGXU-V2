"""Guild-scoped, user-managed pronunciation aliases used only at TTS rendering."""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.models import PronunciationAliasRecord
from lingxu.storage.tables import GuildMemberPronunciationRow


def normalise_pronunciation_alias(value: str) -> str:
    """Return a bounded TTS phrase while excluding invisible/control payloads."""

    alias = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 1 <= len(alias) <= 80:
        raise ValueError("读音别名必须为 1 到 80 个字符。")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in alias):
        raise ValueError("读音别名不能包含控制字符或不可见格式字符。")
    return alias


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _record(row: GuildMemberPronunciationRow) -> PronunciationAliasRecord:
    return PronunciationAliasRecord(
        guild_id=row.guild_id,
        user_id=row.user_id,
        alias=row.alias,
        updated_at=_aware(row.updated_at),
    )


class PronunciationAliasRepository:
    """Persist only the subject user's alias inside one Discord Guild."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, *, guild_id: int, user_id: int) -> PronunciationAliasRecord | None:
        async with self._sessions() as db:
            row = await db.get(GuildMemberPronunciationRow, (guild_id, user_id))
            return _record(row) if row is not None else None

    async def names_for(self, *, guild_id: int, user_ids: tuple[int, ...]) -> dict[int, str]:
        selected = tuple(dict.fromkeys(user_ids))
        if not selected:
            return {}
        async with self._sessions() as db:
            rows = await db.scalars(
                select(GuildMemberPronunciationRow).where(
                    GuildMemberPronunciationRow.guild_id == guild_id,
                    GuildMemberPronunciationRow.user_id.in_(selected),
                )
            )
            return {row.user_id: row.alias for row in rows}

    async def set(self, *, guild_id: int, user_id: int, alias: str) -> PronunciationAliasRecord:
        selected = normalise_pronunciation_alias(alias)
        now = datetime.now(UTC)
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(GuildMemberPronunciationRow)
                .values(
                    guild_id=guild_id,
                    user_id=user_id,
                    alias=selected,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=("guild_id", "user_id"),
                    set_={"alias": selected, "updated_at": now},
                )
            )
            row = await db.get(GuildMemberPronunciationRow, (guild_id, user_id))
            if row is None:
                raise RuntimeError("pronunciation alias upsert did not persist")
            return _record(row)

    async def clear(self, *, guild_id: int, user_id: int) -> bool:
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                delete(GuildMemberPronunciationRow)
                .where(
                    GuildMemberPronunciationRow.guild_id == guild_id,
                    GuildMemberPronunciationRow.user_id == user_id,
                )
                .returning(GuildMemberPronunciationRow.user_id)
            )
            return result.scalar_one_or_none() is not None
