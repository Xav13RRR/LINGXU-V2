from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import discord
import pytest
from discord.ext import commands as discord_commands

from lingxu.bot.wakeup_commands import (
    GuildWakePhraseCoordinator,
    WakePhraseControl,
    WakePhraseRollbackIncompleteError,
    WakeupCommands,
)
from lingxu.domain.wake_phrase import DEFAULT_WAKE_PHRASE, GuildWakePhraseRecord


def _record(
    guild_id: int,
    phrase: str = DEFAULT_WAKE_PHRASE,
    *,
    is_override: bool = False,
    updated_by_user_id: int | None = None,
) -> GuildWakePhraseRecord:
    return GuildWakePhraseRecord(
        guild_id=guild_id,
        phrase=phrase,
        is_override=is_override,
        updated_by_user_id=updated_by_user_id,
        updated_at=datetime.now(UTC) if is_override else None,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.overrides: dict[int, GuildWakePhraseRecord] = {}

    async def get(self, guild_id: int) -> GuildWakePhraseRecord:
        return self.overrides.get(guild_id, _record(guild_id))

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int | None,
    ) -> GuildWakePhraseRecord:
        selected = _record(
            guild_id,
            phrase,
            is_override=True,
            updated_by_user_id=updated_by_user_id,
        )
        self.overrides[guild_id] = selected
        return selected

    async def reset(self, *, guild_id: int) -> GuildWakePhraseRecord:
        self.overrides.pop(guild_id, None)
        return _record(guild_id)


class FakeRuntime:
    def __init__(self, *active_guild_ids: int, fail_phrase: str | None = None) -> None:
        self.active_guild_ids = set(active_guild_ids)
        self.fail_phrase = fail_phrase
        self.updates: list[tuple[int, str]] = []

    async def update_wake_phrase(self, *, guild_id: int, phrase: str) -> bool:
        self.updates.append((guild_id, phrase))
        if phrase == self.fail_phrase:
            raise RuntimeError("runtime update failed")
        return guild_id in self.active_guild_ids


class BlockingRuntime(FakeRuntime):
    def __init__(self, guild_id: int) -> None:
        super().__init__(guild_id)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def update_wake_phrase(self, *, guild_id: int, phrase: str) -> bool:
        self.updates.append((guild_id, phrase))
        self.entered.set()
        await self.release.wait()
        return guild_id in self.active_guild_ids


class RestoreFailingRepository(FakeRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_phrase: str | None = None

    async def set(
        self,
        *,
        guild_id: int,
        phrase: str,
        updated_by_user_id: int | None,
    ) -> GuildWakePhraseRecord:
        if phrase == self.fail_phrase:
            raise RuntimeError("synthetic storage restore failure")
        return await super().set(
            guild_id=guild_id,
            phrase=phrase,
            updated_by_user_id=updated_by_user_id,
        )


def test_wakeup_command_surface_and_default_permissions() -> None:
    cog = WakeupCommands(cast(WakePhraseControl, object()))
    commands = {command.name: command for command in cog.__cog_app_commands__}

    assert set(commands) == {"show", "set", "reset"}
    assert cast(Any, commands["show"]).default_permissions is None
    assert cast(Any, commands["set"]).default_permissions.manage_guild is True
    assert cast(Any, commands["reset"]).default_permissions.manage_guild is True

    bot = discord_commands.Bot(command_prefix="!", intents=discord.Intents.none())
    group = cog.__cog_app_commands_group__
    assert group is not None
    payload = group.to_dict(bot.tree)
    assert group.guild_only is True
    assert payload["dm_permission"] is False
    assert payload["default_member_permissions"] is None


@pytest.mark.asyncio
async def test_non_manager_callbacks_reject_before_any_setting_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = SimpleNamespace(
        get=AsyncMock(),
        set=AsyncMock(),
        reset=AsyncMock(),
    )
    cog = WakeupCommands(cast(WakePhraseControl, control))
    commands = {command.name: command for command in cog.__cog_app_commands__}
    member = SimpleNamespace(
        id=7,
        guild_permissions=SimpleNamespace(manage_guild=False),
    )
    monkeypatch.setattr(
        "lingxu.bot.wakeup_commands.interaction_member",
        lambda _interaction: member,
    )
    interaction = cast(Any, SimpleNamespace())

    with pytest.raises(ValueError, match="只有服务器管理者"):
        await cast(Any, commands["set"]).callback(cog, interaction, "灵序语音助手")
    with pytest.raises(ValueError, match="只有服务器管理者"):
        await cast(Any, commands["reset"]).callback(cog, interaction)

    control.set.assert_not_awaited()
    control.reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_coordinator_isolates_guilds_and_updates_active_runtimes() -> None:
    repository = FakeRepository()
    meeting = FakeRuntime(100)
    resident = FakeRuntime(200)
    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(meeting, resident),
    )

    selected = await coordinator.set(
        guild_id=100,
        phrase="灵序语音助手",
        updated_by_user_id=7,
    )

    assert selected.phrase == "灵序语音助手"
    assert (await coordinator.get(100)).phrase == "灵序语音助手"
    assert (await coordinator.get(200)).phrase == DEFAULT_WAKE_PHRASE
    assert meeting.updates == [(100, "灵序语音助手")]
    assert resident.updates == [(100, "灵序语音助手")]

    reset = await coordinator.reset(guild_id=100)

    assert reset.phrase == DEFAULT_WAKE_PHRASE
    assert not reset.is_override
    assert (await coordinator.get(200)).phrase == DEFAULT_WAKE_PHRASE
    assert meeting.updates[-1] == (100, DEFAULT_WAKE_PHRASE)
    assert resident.updates[-1] == (100, DEFAULT_WAKE_PHRASE)


@pytest.mark.asyncio
async def test_coordinator_rolls_back_storage_and_every_runtime_on_failure() -> None:
    repository = FakeRepository()
    previous = await repository.set(
        guild_id=100,
        phrase="旧版语音助手",
        updated_by_user_id=3,
    )
    meeting = FakeRuntime(100)
    resident = FakeRuntime(100, fail_phrase="新版语音助手")
    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(meeting, resident),
    )

    with pytest.raises(RuntimeError, match="设置已回滚"):
        await coordinator.set(
            guild_id=100,
            phrase="新版语音助手",
            updated_by_user_id=7,
        )

    restored = await coordinator.get(100)
    assert restored.phrase == previous.phrase
    assert restored.updated_by_user_id == previous.updated_by_user_id
    assert meeting.updates == [
        (100, "新版语音助手"),
        (100, "旧版语音助手"),
    ]
    assert resident.updates == [
        (100, "新版语音助手"),
        (100, "旧版语音助手"),
    ]


@pytest.mark.asyncio
async def test_preflight_rejects_unsupported_phrase_before_inactive_persistence() -> None:
    repository = FakeRepository()
    runtime = FakeRuntime()
    checked: list[str] = []

    async def reject(phrase: str) -> None:
        checked.append(phrase)
        raise ValueError("唤醒词包含当前本地关键词模型不支持的读音。")

    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(runtime,),
        preflight=reject,
    )

    with pytest.raises(ValueError, match="不支持的读音"):
        await coordinator.set(
            guild_id=100,
            phrase="灵序语音助手",
            updated_by_user_id=7,
        )

    assert checked == ["灵序语音助手"]
    assert repository.overrides == {}
    assert runtime.updates == []


@pytest.mark.asyncio
async def test_reset_preflights_pinned_default_before_deleting_override() -> None:
    repository = FakeRepository()
    previous = await repository.set(
        guild_id=100,
        phrase="灵序语音助手",
        updated_by_user_id=7,
    )

    async def reject_default(phrase: str) -> None:
        assert phrase == DEFAULT_WAKE_PHRASE
        raise RuntimeError("approved Meeting wake-word model is not installed")

    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(),
        preflight=reject_default,
    )

    with pytest.raises(RuntimeError, match="model is not installed"):
        await coordinator.reset(guild_id=100)

    assert repository.overrides[100] == previous


@pytest.mark.asyncio
async def test_cancelled_set_waits_for_storage_and_runtime_convergence() -> None:
    repository = FakeRepository()
    runtime = BlockingRuntime(100)
    coordinator = GuildWakePhraseCoordinator(repository=repository, runtimes=(runtime,))

    change = asyncio.create_task(
        coordinator.set(
            guild_id=100,
            phrase="灵序语音助手",
            updated_by_user_id=7,
        )
    )
    await asyncio.wait_for(runtime.entered.wait(), timeout=1)
    read = asyncio.create_task(coordinator.get(100))
    await asyncio.sleep(0)
    assert not read.done()

    change.cancel()
    await asyncio.sleep(0)
    assert not change.done()
    runtime.release.set()

    with pytest.raises(asyncio.CancelledError):
        await change
    assert (await read).phrase == "灵序语音助手"
    assert repository.overrides[100].phrase == "灵序语音助手"
    assert runtime.updates == [(100, "灵序语音助手")]


@pytest.mark.asyncio
async def test_incomplete_rollback_still_attempts_every_runtime_restore() -> None:
    repository = RestoreFailingRepository()
    await repository.set(
        guild_id=100,
        phrase="旧版语音助手",
        updated_by_user_id=3,
    )
    repository.fail_phrase = "旧版语音助手"
    meeting = FakeRuntime(100)
    resident = FakeRuntime(100, fail_phrase="新版语音助手")
    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(meeting, resident),
    )

    with pytest.raises(WakePhraseRollbackIncompleteError, match="回滚不完整"):
        await coordinator.set(
            guild_id=100,
            phrase="新版语音助手",
            updated_by_user_id=7,
        )

    assert meeting.updates == [
        (100, "新版语音助手"),
        (100, "旧版语音助手"),
    ]
    assert resident.updates == [
        (100, "新版语音助手"),
        (100, "旧版语音助手"),
    ]


@pytest.mark.asyncio
async def test_success_schedules_recovery_outside_wake_lock() -> None:
    repository = FakeRepository()
    recovered = asyncio.Event()
    observed: list[str] = []
    coordinator: GuildWakePhraseCoordinator

    async def retry(guild_id: int) -> None:
        current = await asyncio.wait_for(coordinator.get(guild_id), timeout=1)
        observed.append(current.phrase)
        recovered.set()

    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(),
        retry_recovery=retry,
    )

    await coordinator.set(
        guild_id=100,
        phrase="灵序语音助手",
        updated_by_user_id=7,
    )
    await asyncio.wait_for(recovered.wait(), timeout=1)

    assert observed == ["灵序语音助手"]


@pytest.mark.asyncio
async def test_recovery_retry_cancellation_propagates_after_committed_change() -> None:
    repository = FakeRepository()
    entered = asyncio.Event()

    async def retry(_guild_id: int) -> None:
        entered.set()
        await asyncio.Event().wait()

    coordinator = GuildWakePhraseCoordinator(
        repository=repository,
        runtimes=(),
        retry_recovery=retry,
    )
    change = asyncio.create_task(
        coordinator.set(
            guild_id=100,
            phrase="灵序语音助手",
            updated_by_user_id=7,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    change.cancel()

    with pytest.raises(asyncio.CancelledError):
        await change
    assert repository.overrides[100].phrase == "灵序语音助手"
