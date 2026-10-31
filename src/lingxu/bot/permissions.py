"""Discord permission and voice-channel checks."""

from __future__ import annotations

import discord


def interaction_member(
    interaction: discord.Interaction[discord.Client],
) -> discord.Member:
    """Return a guild member or reject non-guild interactions."""

    if not isinstance(interaction.user, discord.Member):
        raise ValueError("该命令只能由 Discord 服务器成员使用。")
    return interaction.user


def caller_voice_channel(
    interaction: discord.Interaction[discord.Client],
) -> discord.VoiceChannel:
    """Return the caller's ordinary voice channel."""

    member = interaction_member(interaction)
    channel = member.voice.channel if member.voice else None
    if not isinstance(channel, discord.VoiceChannel):
        raise ValueError("请先加入一个普通 Discord 语音频道。")
    return channel


def require_same_probe_channel(
    member: discord.Member,
    channel: discord.VoiceChannel,
) -> None:
    """Require the member to be present in the active probe channel."""

    member_channel = member.voice.channel if member.voice else None
    if member_channel is None or member_channel.id != channel.id:
        raise ValueError("你必须在当前探针使用的语音频道中。")


def missing_probe_permissions(
    voice_channel: discord.VoiceChannel,
    text_channel: discord.TextChannel,
) -> tuple[str, ...]:
    """Return the exact Discord permissions the Bot lacks."""

    bot_member = voice_channel.guild.me
    voice = voice_channel.permissions_for(bot_member)
    text = text_channel.permissions_for(bot_member)
    missing: list[str] = []
    if not voice.view_channel:
        missing.append("View Channel")
    if not voice.connect:
        missing.append("Connect")
    if not voice.speak:
        missing.append("Speak")
    if not text.view_channel:
        missing.append("View Channel (text)")
    if not text.send_messages:
        missing.append("Send Messages")
    return tuple(missing)


def missing_meeting_voice_permissions(
    voice_channel: discord.VoiceChannel,
    *,
    require_mute_members: bool = False,
) -> tuple[str, ...]:
    """Return the exact permissions required by a formal Meeting voice client."""

    voice = voice_channel.permissions_for(voice_channel.guild.me)
    missing: list[str] = []
    if not voice.view_channel:
        missing.append("View Channel")
    if not voice.connect:
        missing.append("Connect")
    if not voice.speak:
        missing.append("Speak")
    if require_mute_members and not voice.mute_members:
        missing.append("Mute Members")
    return tuple(missing)


def can_control_probe(member: discord.Member, creator_user_id: int) -> bool:
    """Allow the creator or a member with Manage Guild to control the probe."""

    return member.id == creator_user_id or member.guild_permissions.manage_guild
