"""Guild-scoped wake-phrase persistence without voice or Meeting content."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.wake_phrase import (
    DEFAULT_WAKE_PHRASE,
    GuildWakePhraseRecord,
    normalise_wake_phrase,
)
from lingxu.storage.tables import GuildWakeupSettingsRow


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _default_record(guild_id: int) -> GuildWakePhraseRecord:
    return GuildWakePhraseRecord(
        guild_id=guild_id,
        phrase=DEFAULT_WAKE_PHRASE,
        is_override=False,
        updated_by_user_id=None,
        updated_at=None,
    )


def _record(row: GuildWakeupSettingsRow) -> GuildWakePhraseRecord:
    selected = normalise_wake_phrase(row.wake_phrase)
    return GuildWakePhraseRecord(
        guild_id=row.guild_id,
        phrase=selected,
        is_override=True,
        updated_by_user_id=row.updated_by_user_id,
        updated_at=_aware(row.updated_at),
    )


class GuildWakePhraseRepository:
    """Persist one optional wake-phrase override per Discord Guild."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, guild_id: int) -> GuildWakePhraseRecord:
        """Return the Guild override, or the unpersisted global default."""

        async with self._sessions() as db:
            row = await db.get(GuildWakeupSettingsRow, guild_id)
            return _record(row) if row is not None else _default_record(guild_id)

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int | None,
    ) -> GuildWakePhraseRecord:
        """Normalize and atomically upsert this Guilds override."""

        selected = normalise_wake_phrase(phrase)
        now = datetime.now(UTC)
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(GuildWakeupSettingsRow)
                .values(
                    guild_id=guild_id,
                    wake_phrase=selected,
                    updated_by_user_id=updated_by_user_id,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=("guild_id",),
                    set_={
                        "wake_phrase": selected,
                        "updated_by_user_id": updated_by_user_id,
                        "updated_at": now,
                    },
                )
            )
            row = await db.get(GuildWakeupSettingsRow, guild_id)
            if row is None:
                raise RuntimeError("Guild wake phrase upsert did not persist")
            return _record(row)

    async def reset(self, *, guild_id: int) -> GuildWakePhraseRecord:
        """Delete this Guilds override and return the effective default."""

        async with self._sessions() as db, db.begin():
            await db.execute(
                delete(GuildWakeupSettingsRow).where(GuildWakeupSettingsRow.guild_id == guild_id)
            )
        return _default_record(guild_id)
