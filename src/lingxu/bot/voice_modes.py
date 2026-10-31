"""Process-local arbitration for mutually exclusive Discord voice tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from lingxu.domain.errors import VoiceModeConflictError
from lingxu.utils import new_ulid


class GuildVoiceMode(StrEnum):
    """The only supported owners of one Guild's Discord voice state."""

    MEETING = "MEETING"
    RESIDENT_ASSISTANT = "RESIDENT_ASSISTANT"


@dataclass(frozen=True, slots=True)
class GuildVoiceLease:
    """Opaque claim released only by the exact task that acquired it."""

    token: str
    guild_id: int
    mode: GuildVoiceMode
    owner_id: str


class GuildVoiceModeRegistry:
    """Serialise voice ownership before Discord connection work begins."""

    def __init__(self) -> None:
        self._claims: dict[int, GuildVoiceLease] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self, guild_id: int) -> asyncio.Lock:
        return self._locks.setdefault(guild_id, asyncio.Lock())

    async def claim(
        self,
        *,
        guild_id: int,
        mode: GuildVoiceMode,
        owner_id: str,
    ) -> GuildVoiceLease:
        async with self._lock(guild_id):
            existing = self._claims.get(guild_id)
            if existing is not None:
                if existing.mode is mode and existing.owner_id == owner_id:
                    return existing
                raise VoiceModeConflictError(
                    f"该服务器已有语音任务正在运行: {existing.mode.value}。请先结束该任务。"
                )
            lease = GuildVoiceLease(
                token=new_ulid(),
                guild_id=guild_id,
                mode=mode,
                owner_id=owner_id,
            )
            self._claims[guild_id] = lease
            return lease

    async def release(self, lease: GuildVoiceLease) -> bool:
        async with self._lock(lease.guild_id):
            existing = self._claims.get(lease.guild_id)
            if existing is None or existing.token != lease.token:
                return False
            self._claims.pop(lease.guild_id, None)
            return True

    def current(self, guild_id: int) -> GuildVoiceLease | None:
        return self._claims.get(guild_id)
