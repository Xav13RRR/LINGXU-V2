"""Discord slash commands for the resident, content-ephemeral voice assistant."""

from __future__ import annotations

from typing import Protocol

import discord
from discord import app_commands
from discord.ext import commands

from lingxu.bot.permissions import (
    caller_voice_channel,
    interaction_member,
    missing_meeting_voice_permissions,
)
from lingxu.domain.errors import LingxuError, ResidentAssistantPermissionError
from lingxu.domain.models import ResidentAssistantSessionRecord


class ResidentAssistantControl(Protocol):
    async def start(
        self,
        *,
        channel: discord.VoiceChannel,
        creator_user_id: int,
    ) -> ResidentAssistantSessionRecord: ...

    async def status(self, guild_id: int) -> ResidentAssistantSessionRecord: ...

    async def stop(self, *, guild_id: int, caller: discord.Member) -> None: ...


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


class AssistantCommands(commands.GroupCog, group_name="assistant"):
    """Manage one resident voice Q&A connection without Meeting behavior."""

    def __init__(self, coordinator: ResidentAssistantControl) -> None:
        self._coordinator = coordinator

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction[discord.Client],
        error: app_commands.AppCommandError,
    ) -> None:
        cause = error.__cause__ or error
        if isinstance(cause, (LingxuError, ValueError)):
            await _send_error(interaction, str(cause))
            return
        await _send_error(interaction, f"常驻助手操作失败: {type(cause).__name__}")

    @app_commands.command(name="join", description="让语音助手进入你的频道并常驻回答问题")
    @app_commands.default_permissions(manage_guild=True)
    async def join(self, interaction: discord.Interaction[discord.Client]) -> None:
        member = interaction_member(interaction)
        if not member.guild_permissions.manage_guild:
            raise ResidentAssistantPermissionError("只有服务器管理者可以启动常驻助手。")
        channel = caller_voice_channel(interaction)
        missing = missing_meeting_voice_permissions(channel)
        if missing:
            raise ValueError(f"机器人缺少语音频道权限: {', '.join(missing)}")
        await interaction.response.defer(thinking=True)
        record = await self._coordinator.start(
            channel=channel,
            creator_user_id=member.id,
        )
        await interaction.followup.send(
            f"语音助手已进入 {channel.mention} 并开始常驻。会话 `{record.id}`。\n"
            "频道语音只用于本地、内存态唤醒和当前问题处理。不录音、不生成逐字稿、"
            "不生成会议纪要。不保存问题或回答。使用 `/wakeup show` 查看本服务器唤醒词。",
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(name="status", description="查看常驻语音助手状态")
    async def status(self, interaction: discord.Interaction[discord.Client]) -> None:
        record = await self._coordinator.status(_guild_id(interaction))
        runtime = "活动中" if record.status.value == "ACTIVE" else record.status.value
        await interaction.response.send_message(
            f"状态: {runtime}; 语音频道: <#{record.channel_id}>; 会话: `{record.id}`。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(name="stop", description="终止常驻语音助手并使其离开频道")
    @app_commands.default_permissions(manage_guild=True)
    async def stop(self, interaction: discord.Interaction[discord.Client]) -> None:
        member = interaction_member(interaction)
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self._coordinator.stop(guild_id=_guild_id(interaction), caller=member)
        await interaction.followup.send("常驻语音助手已终止并离开频道。", ephemeral=True)
