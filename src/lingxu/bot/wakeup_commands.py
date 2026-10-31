"""Guild-scoped wake-phrase commands and live-runtime coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, Protocol

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from lingxu.bot.permissions import interaction_member
from lingxu.domain.wake_phrase import (
    DEFAULT_WAKE_PHRASE,
    GuildWakePhraseRecord,
    normalise_wake_phrase,
)


class WakePhraseRepository(Protocol):
    async def get(self, guild_id: int) -> GuildWakePhraseRecord: ...

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int | None,
    ) -> GuildWakePhraseRecord: ...

    async def reset(self, *, guild_id: int) -> GuildWakePhraseRecord: ...


class WakePhraseRuntime(Protocol):
    async def update_wake_phrase(self, *, guild_id: int, phrase: str) -> bool: ...


class WakePhraseControl(Protocol):
    async def get(self, guild_id: int) -> GuildWakePhraseRecord: ...

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int,
    ) -> GuildWakePhraseRecord: ...

    async def reset(self, *, guild_id: int) -> GuildWakePhraseRecord: ...


class WakePhraseRollbackIncompleteError(RuntimeError):
    """Raised when storage and every active runtime could not be reconverged."""


class GuildWakePhraseCoordinator:
    """Persist one Guild value and converge any active voice runtime."""

    def __init__(
        self,
        *,
        repository: WakePhraseRepository,
        runtimes: Sequence[WakePhraseRuntime],
        preflight: Callable[[str], Awaitable[None]] | None = None,
        retry_recovery: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        self._repository = repository
        self._runtimes = tuple(runtimes)
        self._preflight = preflight
        self._retry_recovery = retry_recovery
        self._locks: dict[int, asyncio.Lock] = {}
        self._log = structlog.get_logger(__name__)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        return self._locks.setdefault(guild_id, asyncio.Lock())

    async def get(self, guild_id: int) -> GuildWakePhraseRecord:
        async with self._lock(guild_id):
            return await self._repository.get(guild_id)

    async def _update_runtimes(
        self,
        *,
        guild_id: int,
        phrase: str,
    ) -> tuple[WakePhraseRuntime, ...]:
        updated: list[WakePhraseRuntime] = []
        for runtime in self._runtimes:
            if await runtime.update_wake_phrase(guild_id=guild_id, phrase=phrase):
                updated.append(runtime)
        return tuple(updated)

    async def _restore(
        self,
        *,
        guild_id: int,
        previous: GuildWakePhraseRecord | None,
    ) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        try:
            if previous is not None and previous.is_override:
                await self._repository.set(
                    guild_id=previous.guild_id,
                    phrase=previous.phrase,
                    updated_by_user_id=previous.updated_by_user_id,
                )
            else:
                await self._repository.reset(guild_id=guild_id)
        except BaseException as exc:
            errors.append(exc)
        for runtime in reversed(self._runtimes):
            try:
                await runtime.update_wake_phrase(
                    guild_id=guild_id,
                    phrase=previous.phrase if previous is not None else DEFAULT_WAKE_PHRASE,
                )
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)

    async def _retry_failed_recovery(self, guild_id: int) -> None:
        if self._retry_recovery is None:
            return
        try:
            await self._retry_recovery(guild_id)
        except Exception as exc:
            self._log.error(
                "guild_wake_phrase_recovery_retry_failed",
                guild_id=guild_id,
                error_type=type(exc).__name__,
            )

    async def _complete_change(
        self,
        operation: Coroutine[Any, Any, GuildWakePhraseRecord],
        *,
        guild_id: int,
    ) -> tuple[GuildWakePhraseRecord, asyncio.CancelledError | None]:
        task: asyncio.Task[GuildWakePhraseRecord] = asyncio.create_task(
            operation,
            name=f"guild-wake-phrase-converge-{guild_id}",
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                if task.done():
                    break
        result = task.result()
        return result, cancellation

    async def _set_converged(
        self,
        *,
        guild_id: int,
        selected: str,
        updated_by_user_id: int,
        previous: GuildWakePhraseRecord,
    ) -> GuildWakePhraseRecord:
        current = await self._repository.set(
            guild_id=guild_id,
            phrase=selected,
            updated_by_user_id=updated_by_user_id,
        )
        try:
            updated = await self._update_runtimes(
                guild_id=guild_id,
                phrase=current.phrase,
            )
        except BaseException as exc:
            restore_errors = await self._restore(guild_id=guild_id, previous=previous)
            if restore_errors:
                self._log.error(
                    "guild_wake_phrase_rollback_incomplete",
                    guild_id=guild_id,
                    error_count=len(restore_errors),
                )
                raise WakePhraseRollbackIncompleteError(
                    "活动语音助手未能切换唤醒词, 且设置回滚不完整。"
                ) from exc
            raise RuntimeError("活动语音助手未能切换唤醒词, 设置已回滚。") from exc
        self._log.info(
            "guild_wake_phrase_updated",
            guild_id=guild_id,
            is_override=True,
            active_runtime_count=len(updated),
        )
        return current

    async def _reset_converged(
        self,
        *,
        guild_id: int,
        previous: GuildWakePhraseRecord | None,
    ) -> GuildWakePhraseRecord:
        current = await self._repository.reset(guild_id=guild_id)
        try:
            updated = await self._update_runtimes(
                guild_id=guild_id,
                phrase=current.phrase,
            )
        except BaseException as exc:
            restore_errors = await self._restore(guild_id=guild_id, previous=previous)
            if restore_errors:
                self._log.error(
                    "guild_wake_phrase_rollback_incomplete",
                    guild_id=guild_id,
                    error_count=len(restore_errors),
                )
                raise WakePhraseRollbackIncompleteError(
                    "活动语音助手未能恢复默认唤醒词, 且设置回滚不完整。"
                ) from exc
            raise RuntimeError("活动语音助手未能恢复默认唤醒词, 设置已回滚。") from exc
        self._log.info(
            "guild_wake_phrase_updated",
            guild_id=guild_id,
            is_override=False,
            active_runtime_count=len(updated),
        )
        return current

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int,
    ) -> GuildWakePhraseRecord:
        selected = normalise_wake_phrase(phrase)
        async with self._lock(guild_id):
            if self._preflight is not None:
                await self._preflight(selected)
            previous = await self._repository.get(guild_id)
            current, cancellation = await self._complete_change(
                self._set_converged(
                    guild_id=guild_id,
                    selected=selected,
                    updated_by_user_id=updated_by_user_id,
                    previous=previous,
                ),
                guild_id=guild_id,
            )
        await self._retry_failed_recovery(guild_id)
        if cancellation is not None:
            raise cancellation
        return current

    async def reset(self, *, guild_id: int) -> GuildWakePhraseRecord:
        async with self._lock(guild_id):
            if self._preflight is not None:
                await self._preflight(DEFAULT_WAKE_PHRASE)
            try:
                previous: GuildWakePhraseRecord | None = await self._repository.get(guild_id)
            except ValueError as exc:
                previous = None
                self._log.warning(
                    "guild_wake_phrase_corrupt_override_reset",
                    guild_id=guild_id,
                    error_type=type(exc).__name__,
                )
            current, cancellation = await self._complete_change(
                self._reset_converged(guild_id=guild_id, previous=previous),
                guild_id=guild_id,
            )
        await self._retry_failed_recovery(guild_id)
        if cancellation is not None:
            raise cancellation
        return current


def _guild_id(interaction: discord.Interaction[discord.Client]) -> int:
    if interaction.guild_id is None:
        raise ValueError("该命令只能在 Discord 服务器中使用。")
    return interaction.guild_id


async def _send_error(
    interaction: discord.Interaction[discord.Client],
    message: str,
) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


@app_commands.guild_only()
class WakeupCommands(commands.GroupCog, group_name="wakeup"):
    """Read or change the wake phrase for exactly one Discord Guild."""

    def __init__(self, control: WakePhraseControl) -> None:
        self._control = control

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction[discord.Client],
        error: app_commands.AppCommandError,
    ) -> None:
        cause = error.__cause__ or error
        if isinstance(cause, (ValueError, RuntimeError)):
            await _send_error(interaction, str(cause))
            return
        await _send_error(interaction, f"唤醒词操作失败: {type(cause).__name__}")

    @app_commands.command(name="show", description="查看本服务器当前的语音助手唤醒词")
    async def show(self, interaction: discord.Interaction[discord.Client]) -> None:
        record = await self._control.get(_guild_id(interaction))
        source = "服务器自定义" if record.is_override else "系统默认"
        await interaction.response.send_message(
            f"当前唤醒词: `{record.phrase}` ({source})。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(name="set", description="设置本服务器会议与常驻助手共用的唤醒词")
    @app_commands.describe(phrase="连续 4-16 个中文汉字")
    @app_commands.default_permissions(manage_guild=True)
    async def set_phrase(
        self,
        interaction: discord.Interaction[discord.Client],
        phrase: str,
    ) -> None:
        member = interaction_member(interaction)
        if not member.guild_permissions.manage_guild:
            raise ValueError("只有服务器管理者可以修改唤醒词。")
        selected = normalise_wake_phrase(phrase)
        await interaction.response.defer(ephemeral=True, thinking=True)
        record = await self._control.set(
            guild_id=_guild_id(interaction),
            phrase=selected,
            updated_by_user_id=member.id,
        )
        await interaction.followup.send(
            f"本服务器的唤醒词已设置为 `{record.phrase}`; 会议助手和常驻助手立即共用此设置。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(name="reset", description="恢复本服务器的默认语音助手唤醒词")
    @app_commands.default_permissions(manage_guild=True)
    async def reset(self, interaction: discord.Interaction[discord.Client]) -> None:
        member = interaction_member(interaction)
        if not member.guild_permissions.manage_guild:
            raise ValueError("只有服务器管理者可以恢复默认唤醒词。")
        await interaction.response.defer(ephemeral=True, thinking=True)
        record = await self._control.reset(guild_id=_guild_id(interaction))
        await interaction.followup.send(
            f"本服务器已恢复默认唤醒词 `{record.phrase}`。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
