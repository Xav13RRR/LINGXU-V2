"""Discord slash-command surface for deterministic Meeting components."""

from __future__ import annotations

from typing import Literal

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from lingxu.bot.meeting_panels import MeetingPanelCoordinator
from lingxu.bot.meeting_ui import AgendaVoteModal, HostControlView, VoteView
from lingxu.bot.meeting_voice import MeetingVoiceCoordinator
from lingxu.bot.permissions import caller_voice_channel, missing_meeting_voice_permissions
from lingxu.domain.enums import AgendaFollowupMode, HostParticipationMode
from lingxu.domain.errors import LingxuError
from lingxu.meeting.agenda_templates import AgendaTemplateKind, build_agenda_template
from lingxu.meeting.controller import MeetingController
from lingxu.meeting.intent_service import (
    IntentResolution,
    IntentResolutionStatus,
    MeetingIntentService,
)
from lingxu.storage.pronunciation_repository import PronunciationAliasRepository


def _guild_id(interaction: discord.Interaction[discord.Client]) -> int:
    if interaction.guild_id is None:
        raise ValueError("该命令只能在 Discord 服务器中使用。")
    return interaction.guild_id


async def _error(
    interaction: discord.Interaction[discord.Client],
    message: str,
) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


class MeetingCommands(commands.GroupCog, group_name="meeting"):
    """Commands for the fixed deterministic Meeting template."""

    agenda = app_commands.Group(name="agenda", description="管理会议议程")
    speaker = app_commands.Group(name="speaker", description="管理发言队列")
    vote = app_commands.Group(name="vote", description="管理公开投票")
    intent = app_commands.Group(name="intent", description="本地结构化会议意图")
    pronunciation = app_commands.Group(name="pronunciation", description="管理自己的会议读音别名")

    def __init__(
        self,
        controller: MeetingController,
        intents: MeetingIntentService,
        voice: MeetingVoiceCoordinator,
        panels: MeetingPanelCoordinator,
        pronunciations: PronunciationAliasRepository | None = None,
    ) -> None:
        self._controller = controller
        self._intents = intents
        self._voice = voice
        self._panels = panels
        self._pronunciations = pronunciations
        self._log = structlog.get_logger(__name__)

    def _pronunciations_required(self) -> PronunciationAliasRepository:
        if self._pronunciations is None:
            raise RuntimeError("pronunciation alias repository is unavailable")
        return self._pronunciations

    @staticmethod
    def _intent_message(result: IntentResolution) -> str:
        if result.status is IntentResolutionStatus.NO_MATCH:
            return "未识别出可执行的会议意图，会议状态未改变。"
        if result.status is IntentResolutionStatus.PROVIDER_FAILED:
            return "本地模型暂时不可用或输出无效；规则模式仍可使用，会议状态未改变。"
        if result.status is IntentResolutionStatus.REJECTED:
            return "候选意图已拒绝，会议状态未改变。"
        candidate = result.candidate
        if candidate is None:
            return "未生成候选意图。"
        if result.status is IntentResolutionStatus.CONFIRMATION_REQUIRED:
            return (
                f"候选意图：`{candidate.intent.value}`，置信度 "
                f"`{candidate.confidence:.2f}`。\n"
                f"请运行 `/meeting intent confirm candidate_id:{result.candidate_id}` "
                "确认，或使用同组的 reject 命令拒绝。"
            )
        return (
            f"已由确定性会议控制器执行：`{candidate.intent.value}`（来源 `{candidate.source}`）。"
        )

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction[discord.Client],
        error: app_commands.AppCommandError,
    ) -> None:
        cause = error.__cause__ or error
        if isinstance(cause, (LingxuError, ValueError)):
            await _error(interaction, str(cause))
            return
        command = interaction.command
        self._log.error(
            "meeting_command_failed",
            command=command.qualified_name if command is not None else "unknown",
            error_type=type(cause).__name__,
        )
        await _error(interaction, f"会议操作失败：{type(cause).__name__}")

    @app_commands.command(name="create", description="创建会议并发布只读状态面板")
    async def create(self, interaction: discord.Interaction[discord.Client]) -> None:
        guild_id = _guild_id(interaction)
        await interaction.response.defer(thinking=True)
        voice_state = (
            interaction.user.voice if isinstance(interaction.user, discord.Member) else None
        )
        voice_channel = voice_state.channel if voice_state else None
        session = await self._controller.create(
            guild_id=guild_id,
            host_user_id=interaction.user.id,
            publication_channel_id=(
                voice_channel.id if voice_channel is not None else interaction.channel_id
            ),
            setup_channel_id=interaction.channel_id,
            host_display_name=getattr(interaction.user, "display_name", interaction.user.name),
        )
        if voice_channel is not None:
            for member in voice_channel.members:
                if not member.bot:
                    await self._controller.participant_join(
                        guild_id=guild_id,
                        user_id=member.id,
                        display_name=member.display_name,
                    )
        content = await self._panels.content(session.id, private=False)
        public_message = await interaction.edit_original_response(
            content=content,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        await self._panels.bind_public(
            session=session,
            channel_id=public_message.channel.id,
            message_id=public_message.id,
            content=content,
        )
        try:
            await self._panels.deliver_host(session=session, user=interaction.user)
        except (discord.Forbidden, discord.HTTPException):
            status = await self._controller.status_for_session(session.id)
            await interaction.followup.send(
                "无法发送主持私信；以下临时私有面板可用，也可稍后运行 `/meeting controls`。",
                view=HostControlView(
                    self._controller,
                    self._panels,
                    status=status,
                    fallback_guild_id=guild_id,
                ),
                ephemeral=True,
            )
        else:
            await interaction.followup.send(
                "会议已创建；主持控制面板已通过私信发送。先添加议程，再运行 `/meeting start`。",
                ephemeral=True,
            )

    @app_commands.command(name="controls", description="重新显示仅主持人可见的临时控制面板")
    async def controls(self, interaction: discord.Interaction[discord.Client]) -> None:
        guild_id = _guild_id(interaction)
        session = await self._controller.host_session(
            guild_id=guild_id,
            actor_user_id=interaction.user.id,
        )
        status = await self._controller.status_for_session(session.id)
        await interaction.response.send_message(
            await self._panels.content(session.id, private=True),
            view=HostControlView(
                self._controller,
                self._panels,
                status=status,
                fallback_guild_id=guild_id,
            ),
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @app_commands.command(name="start", description="由主持人开始会议")
    async def start(self, interaction: discord.Interaction[discord.Client]) -> None:
        guild_id = _guild_id(interaction)
        session = await self._controller.ensure_start_ready(
            guild_id=guild_id,
            actor_user_id=interaction.user.id,
        )
        voice_channel = caller_voice_channel(interaction)
        status = await self._controller.status_for_session(session.id)
        strict_requested = any(
            item.speaking_policy.mute_enforcement_requested for item in status.agenda
        )
        missing = missing_meeting_voice_permissions(
            voice_channel,
            require_mute_members=strict_requested,
        )
        if missing:
            raise ValueError(f"机器人缺少语音频道权限：{', '.join(missing)}")

        await interaction.response.defer(thinking=True)
        await self._controller.sync_participants(
            guild_id=guild_id,
            user_ids={member.id for member in voice_channel.members if not member.bot},
            display_names={
                member.id: member.display_name for member in voice_channel.members if not member.bot
            },
        )
        session = await self._controller.set_publication_channel(
            guild_id=guild_id,
            channel_id=voice_channel.id,
        )
        connected = await self._voice.start(session=session, channel=voice_channel)
        try:
            result = await self._controller.start(
                guild_id=guild_id,
                actor_user_id=interaction.user.id,
            )
        except BaseException:
            if connected:
                await self._voice.stop(guild_id=guild_id)
            await self._voice.release_capacity(
                session=session,
                reason="MEETING_START_FAILED",
            )
            raise
        await interaction.followup.send(
            f"聆序已部署至 {voice_channel.mention}；当前会议状态：{result.session.state.value}"
        )
        await self._panels.refresh(result.session.id)

    @app_commands.command(name="status", description="查看当前会议的安全状态摘要")
    async def status(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        status = await self._controller.status(_guild_id(interaction))
        agenda = (
            "\n".join(
                f"{item.position}. [{item.status.value}] {item.title}" for item in status.agenda
            )
            or "（未添加议程）"
        )
        queued = "、".join(f"<@{user_id}>" for user_id in status.speaking.queued_user_ids) or "无"
        vote = status.vote.title if status.vote else "无"
        await interaction.edit_original_response(
            content=(
                f"状态：`{status.session.state.value}`\n"
                f"议程：\n{agenda}\n"
                f"发言队列：{queued}\n"
                f"进行中投票：{vote}"
            )
        )

    @app_commands.command(name="pause", description="由主持人暂停会议和全部计时器")
    async def pause(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.pause(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")

    @app_commands.command(name="resume", description="由主持人恢复会议和全部计时器")
    async def resume(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.resume(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")

    @app_commands.command(name="next", description="由主持人进入固定流程的下一状态")
    async def next(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.next(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")

    @app_commands.command(name="back", description="由主持人返回固定流程的上一状态")
    async def back(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.back(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")

    @app_commands.command(name="skip", description="由主持人跳过当前固定流程状态")
    async def skip(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.skip(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")

    @app_commands.command(name="stop", description="由主持人停止会议并取消全部计时器")
    async def stop(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        result = await self._controller.stop(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(f"会议状态：{result.session.state.value}")
        await self._panels.refresh(result.session.id)

    @app_commands.command(name="speak", description="申请加入发言队列（重复申请不会重复入队）")
    async def speak(self, interaction: discord.Interaction[discord.Client]) -> None:
        changed = await self._controller.speak(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
        )
        message = "已加入发言队列。" if changed else "你已在发言队列或正在发言。"
        await interaction.response.send_message(message, ephemeral=True)

    @app_commands.command(name="consent", description="同意实时识别、录音和会后纪要三项用途")
    async def consent(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        state = await self._controller.consent_all(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
        )
        await interaction.followup.send(
            (
                "已记录，该授权在本 Discord 服务器内跨会议有效，"
                "可运行 `/meeting withdraw` 撤回。"
                f"\n当前会议状态：`{state.value}`"
            ),
            ephemeral=True,
        )

    @app_commands.command(name="withdraw", description="立即撤回自己的全部后续处理同意")
    async def withdraw(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self._controller.withdraw_all(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
        )
        await interaction.followup.send(
            "已撤回本 Discord 服务器内的全部后续处理同意；其它服务器不受影响。",
            ephemeral=True,
        )

    @app_commands.command(name="retry", description="由主持人重试失败的会后处理 Job")
    async def retry(self, interaction: discord.Interaction[discord.Client]) -> None:
        guild_id = _guild_id(interaction)
        await interaction.response.defer(ephemeral=True, thinking=True)
        session = await self._controller.retry_session(
            guild_id=guild_id,
            actor_user_id=interaction.user.id,
        )
        await self._voice.claim_capacity(session=session)
        try:
            changed = await self._controller.retry(
                guild_id=guild_id,
                actor_user_id=interaction.user.id,
            )
        except BaseException:
            await self._voice.release_capacity(session=session, reason="RETRY_FAILED")
            raise
        if not changed:
            await self._voice.release_capacity(session=session, reason="NOTHING_TO_RETRY")
        await interaction.edit_original_response(
            content="失败 Job 已重新排队。" if changed else "没有可重试的失败 Job。"
        )

    @app_commands.command(name="delete", description="由主持人删除最新会议及其全部数据")
    async def delete(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        session = await self._controller.delete(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await self._voice.release_capacity(session=session, reason="MEETING_DELETED")
        await interaction.followup.send("会议及其数据已删除。", ephemeral=True)

    @intent.command(name="resolve", description="用规则和本地模型解析一条会议指令")
    async def intent_resolve(
        self,
        interaction: discord.Interaction[discord.Client],
        text: app_commands.Range[str, 1, 1000],
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await self._intents.resolve_and_execute(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            text=text,
        )
        await interaction.followup.send(self._intent_message(result), ephemeral=True)

    @intent.command(name="confirm", description="确认自己的低置信度候选意图")
    async def intent_confirm(
        self,
        interaction: discord.Interaction[discord.Client],
        candidate_id: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await self._intents.confirm(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            candidate_id=candidate_id,
        )
        await interaction.followup.send(
            self._intent_message(result),
            ephemeral=True,
        )

    @intent.command(name="reject", description="拒绝自己的低置信度候选意图")
    async def intent_reject(
        self,
        interaction: discord.Interaction[discord.Client],
        candidate_id: str,
    ) -> None:
        result = await self._intents.reject(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            candidate_id=candidate_id,
        )
        await interaction.response.send_message(
            self._intent_message(result),
            ephemeral=True,
        )

    @agenda.command(name="add", description="添加议程项")
    async def agenda_add(
        self,
        interaction: discord.Interaction[discord.Client],
        title: str,
        duration: app_commands.Range[int, 1, 480] | None = None,
        requires_vote: bool = False,
        language: Literal["auto", "zh", "yue", "en", "ja", "ko"] = "auto",
    ) -> None:
        guild_id = _guild_id(interaction)
        if requires_vote:
            await interaction.response.send_modal(
                AgendaVoteModal(
                    controller=self._controller,
                    guild_id=guild_id,
                    actor_user_id=interaction.user.id,
                    agenda_title=title,
                    duration_minutes=duration,
                    language=language,
                )
            )
            return
        item = await self._controller.agenda_add(
            guild_id=guild_id,
            actor_user_id=interaction.user.id,
            title=title,
            duration_minutes=duration,
            requires_vote=False,
            language=language,
        )
        await interaction.response.send_message(
            f"已添加议程 {item.position}：{item.title}",
            ephemeral=True,
        )

    @agenda.command(name="list", description="列出当前议程")
    async def agenda_list(self, interaction: discord.Interaction[discord.Client]) -> None:
        status = await self._controller.status(_guild_id(interaction))
        lines = [f"{item.position}. [{item.status.value}] {item.title}" for item in status.agenda]
        await interaction.response.send_message("\n".join(lines) or "议程为空。", ephemeral=True)

    @agenda.command(name="template", description="从内置模板添加受控发言或演讲议程")
    async def agenda_template(
        self,
        interaction: discord.Interaction[discord.Client],
        kind: Literal[
            "round_robin",
            "standup",
            "pre_vote_statements",
            "alternating_sides",
            "presentation",
            "presentation_q_and_a",
        ],
        title: str | None = None,
        duration_minutes: app_commands.Range[int, 1, 480] | None = None,
        turn_seconds: app_commands.Range[int, 1, 14_400] | None = None,
        primary: discord.Member | None = None,
        secondary: discord.Member | None = None,
        followup: Literal["default", "none", "supplement", "q_and_a"] = "default",
        enforce_mute: bool = False,
        host_participation: Literal["auto", "moderator_only"] = "auto",
        prep_seconds: app_commands.Range[int, 0, 300] = 10,
        vote_content: str | None = None,
    ) -> None:
        explicit_ids = tuple(
            member.id for member in (primary, secondary) if member is not None and not member.bot
        )
        if any(member is not None and member.bot for member in (primary, secondary)):
            raise ValueError("议程发言者不能是机器人。")
        selected_kind = AgendaTemplateKind(kind.upper())
        is_presentation = selected_kind in {
            AgendaTemplateKind.PRESENTATION,
            AgendaTemplateKind.PRESENTATION_Q_AND_A,
        }
        selected_followup = {
            "default": None,
            "none": AgendaFollowupMode.NONE,
            "supplement": AgendaFollowupMode.SUPPLEMENT_TURNS,
            "q_and_a": AgendaFollowupMode.Q_AND_A,
        }[followup]
        draft = build_agenda_template(
            selected_kind,
            title=title,
            duration_minutes=duration_minutes,
            turn_seconds=turn_seconds,
            members=() if is_presentation else explicit_ids,
            presenters=explicit_ids if is_presentation else (),
            followup_mode=selected_followup,
            host_participation=(
                HostParticipationMode.AUTO
                if host_participation == "auto"
                else HostParticipationMode.MODERATOR_ONLY
            ),
            mute_enforcement_requested=enforce_mute,
            prep_seconds=prep_seconds,
            vote_content=vote_content,
        )
        item = await self._controller.agenda_add(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            title=draft.title,
            duration_minutes=draft.duration_minutes,
            requires_vote=draft.requires_vote,
            vote_content=draft.vote_content,
            vote_duration_seconds=draft.vote_duration_seconds,
            language=draft.language,
            speaking_policy=draft.speaking_policy,
        )
        permission_note = "；开始会议前请授予聆序 `Mute Members` 权限" if enforce_mute else ""
        await interaction.response.send_message(
            f"已从 `{draft.kind.value}` 添加议程 {item.position}{permission_note}。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @agenda.command(name="remove", description="删除尚未开始的议程项")
    async def agenda_remove(
        self,
        interaction: discord.Interaction[discord.Client],
        position: app_commands.Range[int, 1],
    ) -> None:
        changed = await self._controller.agenda_remove(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            position=position,
        )
        await interaction.response.send_message(
            "议程项已删除。" if changed else "找不到可删除的待处理议程项。",
            ephemeral=True,
        )

    @agenda.command(name="reorder", description="调整议程项顺序")
    async def agenda_reorder(
        self,
        interaction: discord.Interaction[discord.Client],
        from_position: app_commands.Range[int, 1],
        to_position: app_commands.Range[int, 1],
    ) -> None:
        await self._controller.agenda_reorder(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            from_position=from_position,
            to_position=to_position,
        )
        await interaction.response.send_message("议程顺序已更新。", ephemeral=True)

    @speaker.command(name="next", description="由主持人选择下一位在线申请者")
    async def speaker_next(self, interaction: discord.Interaction[discord.Client]) -> None:
        selected = await self._controller.speaker_next(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            f"下一位发言者：<@{selected}>" if selected else "发言队列为空。"
        )

    @pronunciation.command(name="set", description="设置自己在本服务器的会议 TTS 读音")
    async def pronunciation_set(
        self,
        interaction: discord.Interaction[discord.Client],
        alias: app_commands.Range[str, 1, 80],
    ) -> None:
        record = await self._pronunciations_required().set(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
            alias=alias,
        )
        await interaction.response.send_message(
            f"已设置您的会议读音：{discord.utils.escape_markdown(record.alias)}。",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @pronunciation.command(name="show", description="查看自己在本服务器的会议 TTS 读音")
    async def pronunciation_show(self, interaction: discord.Interaction[discord.Client]) -> None:
        record = await self._pronunciations_required().get(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
        )
        message = (
            f"当前会议读音：{discord.utils.escape_markdown(record.alias)}。"
            if record is not None
            else "您尚未在本服务器设置会议读音。"
        )
        await interaction.response.send_message(
            message,
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )

    @pronunciation.command(name="clear", description="清除自己在本服务器的会议 TTS 读音")
    async def pronunciation_clear(self, interaction: discord.Interaction[discord.Client]) -> None:
        changed = await self._pronunciations_required().clear(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            "已清除会议读音。" if changed else "您尚未设置会议读音。",
            ephemeral=True,
        )

    @speaker.command(name="finish", description="当前发言者或主持人结束本轮发言")
    async def speaker_finish(self, interaction: discord.Interaction[discord.Client]) -> None:
        finished = await self._controller.speaker_finish(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            f"<@{finished}> 的发言已结束。" if finished else "当前没有发言者。"
        )

    @speaker.command(name="remove", description="由主持人将成员移出发言队列")
    async def speaker_remove(
        self,
        interaction: discord.Interaction[discord.Client],
        member: discord.Member,
    ) -> None:
        changed = await self._controller.speaker_remove(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            user_id=member.id,
        )
        await interaction.response.send_message(
            "成员已移出发言队列。" if changed else "该成员不在发言队列。",
            ephemeral=True,
        )

    @speaker.command(name="extend", description="主持人延长当前受控发言或演讲")
    async def speaker_extend(
        self,
        interaction: discord.Interaction[discord.Client],
        seconds: app_commands.Range[int, 1, 3600] = 30,
    ) -> None:
        changed = await self._controller.controlled_speaking_extend(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            additional_seconds=seconds,
        )
        await interaction.response.send_message(
            f"当前流程已延长 {seconds} 秒。" if changed else "当前没有可延长的发言流程。",
            ephemeral=True,
        )

    @speaker.command(name="skip", description="主持人跳过当前受控发言者")
    async def speaker_skip(self, interaction: discord.Interaction[discord.Client]) -> None:
        changed = await self._controller.controlled_speaking_finish(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            skip=True,
        )
        await interaction.response.send_message(
            "当前发言已跳过。" if changed else "当前没有可跳过的发言流程。",
            ephemeral=True,
        )

    @speaker.command(name="end", description="主持人提前结束当前受控发言或演讲")
    async def speaker_end(self, interaction: discord.Interaction[discord.Client]) -> None:
        changed = await self._controller.controlled_speaking_finish(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            "当前流程已提前结束。" if changed else "当前没有可结束的发言流程。",
            ephemeral=True,
        )

    @speaker.command(name="abandon", description="当前按序发言者放弃自己的本轮发言")
    async def speaker_abandon(self, interaction: discord.Interaction[discord.Client]) -> None:
        changed = await self._controller.controlled_speaking_abandon(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            "您已放弃本轮发言。" if changed else "当前没有可放弃的发言。",
            ephemeral=True,
        )

    @speaker.command(name="followup-end", description="主持人结束补充发言或统一答疑")
    async def speaker_followup_end(
        self,
        interaction: discord.Interaction[discord.Client],
    ) -> None:
        changed = await self._controller.controlled_followup_finish(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.response.send_message(
            "补充环节已结束。" if changed else "当前不在补充或答疑环节。",
            ephemeral=True,
        )

    @vote.command(name="open", description="由主持人开启公开赞成/反对/弃权投票")
    async def vote_open(
        self,
        interaction: discord.Interaction[discord.Client],
        title: str,
        duration_seconds: app_commands.Range[int, 10, 3600] = 60,
    ) -> None:
        vote = await self._controller.vote_open(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
            title=title,
            duration_seconds=duration_seconds,
        )
        await interaction.response.send_message(
            f"公开投票「{vote.title}」已开启，截止 <t:{int(vote.deadline_at.timestamp())}:R>。",
            view=VoteView(self._controller),
        )

    @vote.command(name="cast", description="投赞成、反对或弃权；也可作为私信投票失败时的入口")
    async def vote_cast(
        self,
        interaction: discord.Interaction[discord.Client],
        option: Literal["yes", "no"],
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        await self._controller.vote_cast(
            guild_id=_guild_id(interaction),
            user_id=interaction.user.id,
            option_id=option,
        )
        await interaction.followup.send(f"已记录：{option}", ephemeral=True)

    @vote.command(name="close", description="由主持人关闭投票并公开准确结果")
    async def vote_close(self, interaction: discord.Interaction[discord.Client]) -> None:
        await interaction.response.defer(thinking=True)
        outcome = await self._controller.vote_close(
            guild_id=_guild_id(interaction),
            actor_user_id=interaction.user.id,
        )
        await interaction.followup.send(outcome.broadcast_text())

    @vote.command(name="status", description="查看当前投票")
    async def vote_status(self, interaction: discord.Interaction[discord.Client]) -> None:
        status = await self._controller.status(_guild_id(interaction))
        message = (
            f"进行中：{status.vote.title}，截止 <t:{int(status.vote.deadline_at.timestamp())}:R>"
            if status.vote
            else "当前没有进行中的投票。"
        )
        await interaction.response.send_message(message, ephemeral=True)
