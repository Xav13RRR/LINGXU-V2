"""Persistent Discord buttons for deterministic Meeting controls and votes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

import discord

from lingxu.domain.enums import AgendaSpeakingMode, MeetingState, SpeakingFlowStage
from lingxu.domain.errors import LingxuError
from lingxu.meeting.controller import MeetingController, MeetingStatus


class HostPanelRuntime(Protocol):
    async def resolve_host_guild(self, *, message_id: int, user_id: int) -> int: ...

    async def refresh_guild(self, guild_id: int) -> None: ...


async def _host_button_action(
    interaction: discord.Interaction[discord.Client],
    panels: HostPanelRuntime,
    fallback_guild_id: int | None,
    operation: Callable[[int, int], Awaitable[str]],
) -> None:
    guild_id = interaction.guild_id or fallback_guild_id
    if guild_id is None:
        if interaction.message is None:
            await interaction.response.send_message("无法识别这张主持控制面板。")
            return
        try:
            guild_id = await panels.resolve_host_guild(
                message_id=interaction.message.id,
                user_id=interaction.user.id,
            )
        except (LingxuError, ValueError) as exc:
            await interaction.response.send_message(str(exc))
            return
    ephemeral = interaction.guild_id is not None
    await interaction.response.defer(ephemeral=ephemeral, thinking=True)
    try:
        message = await operation(guild_id, interaction.user.id)
        await panels.refresh_guild(guild_id)
    except (LingxuError, ValueError) as exc:
        await interaction.followup.send(str(exc), ephemeral=ephemeral)
        return
    await interaction.followup.send(message, ephemeral=ephemeral)


class HostControlView(discord.ui.View):
    """Restart-safe controls delivered only by DM or ephemeral fallback."""

    def __init__(
        self,
        controller: MeetingController,
        panels: HostPanelRuntime,
        *,
        status: MeetingStatus | None = None,
        fallback_guild_id: int | None = None,
    ) -> None:
        super().__init__(timeout=None)
        self._controller = controller
        self._panels = panels
        self._fallback_guild_id = fallback_guild_id
        if status is not None:
            self._apply_context(status)

    def _apply_context(self, status: MeetingStatus) -> None:
        state = status.session.state
        floor_stage = status.floor.stage if status.floor is not None else None
        active = state not in {
            MeetingState.PUBLISHED,
            MeetingState.STOPPED,
            MeetingState.FAILED_CAPTURE,
            MeetingState.FAILED_REALTIME_ASR,
            MeetingState.FAILED_TTS,
            MeetingState.FAILED_POST_PROCESSING,
            MeetingState.DELETED,
        }
        controlled = state is MeetingState.DISCUSSION_READY
        host_can_abandon = (
            controlled
            and status.floor is not None
            and status.floor.mode is AgendaSpeakingMode.ORDERED
            and floor_stage is SpeakingFlowStage.ACTIVE
            and status.floor.current_user_ids == (status.session.creator_user_id,)
        )
        enforcement = status.enforcement
        enforcing = controlled and enforcement is not None and enforcement.enabled
        enforcement_paused = enforcement.paused if enforcement is not None else False
        intervention = (
            enforcing and enforcement is not None and enforcement.host_intervention_active
        )
        disabled = {
            "lingxu:host:pause:v1": not active or state is MeetingState.PAUSED,
            "lingxu:host:resume:v1": state is not MeetingState.PAUSED,
            "lingxu:host:next:v1": state is not MeetingState.DISCUSSION_READY,
            "lingxu:host:extend:v1": not controlled or floor_stage is not SpeakingFlowStage.ACTIVE,
            "lingxu:host:skip-turn:v1": not controlled
            or floor_stage not in {SpeakingFlowStage.PREPARING, SpeakingFlowStage.ACTIVE},
            "lingxu:host:end-turn:v1": not controlled
            or floor_stage not in {SpeakingFlowStage.PREPARING, SpeakingFlowStage.ACTIVE},
            "lingxu:host:end-followup:v1": not controlled
            or floor_stage is not SpeakingFlowStage.FOLLOWUP,
            "lingxu:host:abandon:v1": not host_can_abandon,
            "lingxu:host:stop:v1": not active,
            "lingxu:host:intervene-start:v1": not enforcing or enforcement_paused or intervention,
            "lingxu:host:intervene-end:v1": not intervention,
            "lingxu:host:enforcement-pause:v1": not enforcing or enforcement_paused or intervention,
            "lingxu:host:enforcement-resume:v1": not enforcing or not enforcement_paused,
            "lingxu:host:emergency-release:v1": not enforcing,
            "lingxu:host:exception-grant:v1": not enforcing or enforcement_paused,
            "lingxu:host:exception-revoke:v1": not enforcing or enforcement_paused,
        }
        for item in self.children:
            if isinstance(item, discord.ui.Button) and item.custom_id in disabled:
                item.disabled = disabled[item.custom_id]

    @discord.ui.button(
        label="暂停",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:pause:v1",
    )
    async def pause(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            result = await self._controller.pause(
                guild_id=guild_id,
                actor_user_id=user_id,
                source="BUTTON",
            )
            return f"会议状态：{result.session.state.value}"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="继续",
        style=discord.ButtonStyle.success,
        custom_id="lingxu:host:resume:v1",
    )
    async def resume(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            result = await self._controller.resume(
                guild_id=guild_id,
                actor_user_id=user_id,
                source="BUTTON",
            )
            return f"会议状态：{result.session.state.value}"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="下一项",
        style=discord.ButtonStyle.primary,
        custom_id="lingxu:host:next:v1",
    )
    async def next(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        guild_id = interaction.guild_id or self._fallback_guild_id
        if guild_id is None:
            if interaction.message is None:
                await interaction.response.send_message("无法识别这张主持控制面板。")
                return
            try:
                guild_id = await self._panels.resolve_host_guild(
                    message_id=interaction.message.id,
                    user_id=interaction.user.id,
                )
            except (LingxuError, ValueError) as exc:
                await interaction.response.send_message(str(exc))
                return
        await interaction.response.send_message(
            "确认进入下一项？这会结束当前固定流程。",
            view=ConfirmNextView(
                self._controller,
                self._panels,
                guild_id=guild_id,
                host_user_id=interaction.user.id,
            ),
            ephemeral=interaction.guild_id is not None,
        )

    @discord.ui.button(
        label="延长 30 秒",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:extend:v1",
        row=1,
    )
    async def extend(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.controlled_speaking_extend(
                guild_id=guild_id,
                actor_user_id=user_id,
            )
            return "当前流程已延长 30 秒。" if changed else "当前没有可延长的流程。"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="跳过发言",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:skip-turn:v1",
        row=1,
    )
    async def skip_turn(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.controlled_speaking_finish(
                guild_id=guild_id,
                actor_user_id=user_id,
                skip=True,
            )
            return "当前发言已跳过。" if changed else "当前没有可跳过的发言。"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="提前结束",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:end-turn:v1",
        row=1,
    )
    async def end_turn(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.controlled_speaking_finish(
                guild_id=guild_id,
                actor_user_id=user_id,
            )
            return "当前流程已提前结束。" if changed else "当前没有可结束的流程。"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="结束补充环节",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:end-followup:v1",
        row=1,
    )
    async def end_followup(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.controlled_followup_finish(
                guild_id=guild_id,
                actor_user_id=user_id,
            )
            return "补充环节已结束。" if changed else "当前不在补充环节。"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="本轮放弃",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:abandon:v1",
        row=2,
    )
    async def abandon(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.controlled_speaking_abandon(
                guild_id=guild_id,
                actor_user_id=user_id,
            )
            return "您已放弃本轮发言。" if changed else "当前没有可放弃的发言。"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )

    @discord.ui.button(
        label="主持介入",
        style=discord.ButtonStyle.primary,
        custom_id="lingxu:host:intervene-start:v1",
        row=2,
    )
    async def intervene_start(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.host_intervention(
                guild_id=guild_id,
                actor_user_id=user_id,
                active=True,
                source="BUTTON",
            )
            return "已暂停当前轮次计时，主持人取得发言权。" if changed else "当前无法开始主持介入。"

        await _host_button_action(interaction, self._panels, self._fallback_guild_id, operation)

    @discord.ui.button(
        label="结束介入",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:intervene-end:v1",
        row=2,
    )
    async def intervene_end(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.host_intervention(
                guild_id=guild_id,
                actor_user_id=user_id,
                active=False,
                source="BUTTON",
            )
            return "主持介入已结束，原发言权和计时已恢复。" if changed else "当前没有主持介入。"

        await _host_button_action(interaction, self._panels, self._fallback_guild_id, operation)

    @discord.ui.button(
        label="暂停强制静音",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:enforcement-pause:v1",
        row=3,
    )
    async def enforcement_pause(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.mute_enforcement_pause(
                guild_id=guild_id,
                actor_user_id=user_id,
                source="BUTTON",
            )
            return "强制静音和当前轮次计时已暂停。" if changed else "当前无法暂停强制静音。"

        await _host_button_action(interaction, self._panels, self._fallback_guild_id, operation)

    @discord.ui.button(
        label="恢复强制静音",
        style=discord.ButtonStyle.success,
        custom_id="lingxu:host:enforcement-resume:v1",
        row=3,
    )
    async def enforcement_resume(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.mute_enforcement_resume(
                guild_id=guild_id,
                actor_user_id=user_id,
                source="BUTTON",
            )
            return "已重新检查权限并恢复强制静音。" if changed else "当前没有暂停的强制静音。"

        await _host_button_action(interaction, self._panels, self._fallback_guild_id, operation)

    @discord.ui.button(
        label="紧急解除静音",
        style=discord.ButtonStyle.danger,
        custom_id="lingxu:host:emergency-release:v1",
        row=3,
    )
    async def emergency_release(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.mute_enforcement_downgrade(
                guild_id=guild_id,
                actor_user_id=user_id,
                emergency=True,
                source="BUTTON",
            )
            return (
                "已将本议程降级为提示模式，并释放聆序拥有的静音。"
                if changed
                else "当前没有可释放的强制静音。"
            )

        await _host_button_action(interaction, self._panels, self._fallback_guild_id, operation)

    async def _show_exception_modal(
        self,
        interaction: discord.Interaction[discord.Client],
        *,
        granted: bool,
    ) -> None:
        guild_id = interaction.guild_id or self._fallback_guild_id
        if guild_id is None:
            if interaction.message is None:
                await interaction.response.send_message("无法识别这张主持控制面板。")
                return
            try:
                guild_id = await self._panels.resolve_host_guild(
                    message_id=interaction.message.id,
                    user_id=interaction.user.id,
                )
            except (LingxuError, ValueError) as exc:
                await interaction.response.send_message(str(exc))
                return
        await interaction.response.send_modal(
            FloorExceptionModal(
                self._controller,
                self._panels,
                guild_id=guild_id,
                granted=granted,
            )
        )

    @discord.ui.button(
        label="准许临时发言",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:exception-grant:v1",
        row=4,
    )
    async def exception_grant(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        await self._show_exception_modal(interaction, granted=True)

    @discord.ui.button(
        label="收回临时发言",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:host:exception-revoke:v1",
        row=4,
    )
    async def exception_revoke(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        await self._show_exception_modal(interaction, granted=False)

    @discord.ui.button(
        label="停止",
        style=discord.ButtonStyle.danger,
        custom_id="lingxu:host:stop:v1",
        row=2,
    )
    async def stop_meeting(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[HostControlView],
    ) -> None:
        async def operation(guild_id: int, user_id: int) -> str:
            result = await self._controller.stop(
                guild_id=guild_id,
                actor_user_id=user_id,
                source="BUTTON",
            )
            return f"会议状态：{result.session.state.value}"

        await _host_button_action(
            interaction,
            self._panels,
            self._fallback_guild_id,
            operation,
        )


class FloorExceptionModal(discord.ui.Modal):
    """Resolve one immutable Discord user ID without display-name guessing."""

    member_id: discord.ui.TextInput[FloorExceptionModal] = discord.ui.TextInput(
        label="成员 Discord 用户 ID 或 mention",
        placeholder="例如 123456789012345678 或 @成员",
        min_length=1,
        max_length=32,
    )

    def __init__(
        self,
        controller: MeetingController,
        panels: HostPanelRuntime,
        *,
        guild_id: int,
        granted: bool,
    ) -> None:
        super().__init__(title="准许临时发言" if granted else "收回临时发言")
        self._controller = controller
        self._panels = panels
        self._guild_id = guild_id
        self._granted = granted

    async def on_submit(self, interaction: discord.Interaction[discord.Client]) -> None:
        raw = str(self.member_id).strip().removeprefix("<@").removesuffix(">").lstrip("!")
        if not raw.isdecimal():
            await interaction.response.send_message("请输入有效的 Discord 用户 ID 或 mention。")
            return
        target_user_id = int(raw)

        async def operation(guild_id: int, user_id: int) -> str:
            changed = await self._controller.mute_floor_exception(
                guild_id=guild_id,
                actor_user_id=user_id,
                user_id=target_user_id,
                granted=self._granted,
                source="BUTTON",
            )
            if not changed:
                return "临时发言权限没有变化。"
            return "已准许该成员临时发言。" if self._granted else "已收回该成员的临时发言权限。"

        await _host_button_action(interaction, self._panels, self._guild_id, operation)


class ConfirmNextView(discord.ui.View):
    """Short-lived private confirmation for the destructive next-item control."""

    def __init__(
        self,
        controller: MeetingController,
        panels: HostPanelRuntime,
        *,
        guild_id: int,
        host_user_id: int,
    ) -> None:
        super().__init__(timeout=60)
        self._controller = controller
        self._panels = panels
        self._guild_id = guild_id
        self._host_user_id = host_user_id

    @discord.ui.button(label="确认下一项", style=discord.ButtonStyle.danger)
    async def confirm(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[ConfirmNextView],
    ) -> None:
        if interaction.user.id != self._host_user_id:
            await interaction.response.send_message("只有主持人可以确认。", ephemeral=True)
            return
        await interaction.response.defer()
        try:
            result = await self._controller.next(
                guild_id=self._guild_id,
                actor_user_id=interaction.user.id,
                source="BUTTON",
            )
            await self._panels.refresh_guild(self._guild_id)
        except (LingxuError, ValueError) as exc:
            await interaction.edit_original_response(content=str(exc), view=None)
            return
        await interaction.edit_original_response(
            content=f"已确认；会议状态：{result.session.state.value}",
            view=None,
        )

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[ConfirmNextView],
    ) -> None:
        if interaction.user.id != self._host_user_id:
            await interaction.response.send_message("只有主持人可以取消。", ephemeral=True)
            return
        await interaction.response.edit_message(content="已取消。", view=None)


class VoteView(discord.ui.View):
    """Persistent public voting buttons resolved against the guild's open vote."""

    def __init__(self, controller: MeetingController) -> None:
        super().__init__(timeout=None)
        self._controller = controller

    async def _cast(
        self,
        interaction: discord.Interaction[discord.Client],
        option_id: str,
    ) -> None:
        if interaction.guild_id is None:
            await interaction.response.send_message("该按钮只能在服务器中使用。", ephemeral=True)
            return
        try:
            await self._controller.vote_cast(
                guild_id=interaction.guild_id,
                user_id=interaction.user.id,
                option_id=option_id,
                source="BUTTON",
            )
        except (LingxuError, ValueError) as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(f"已记录：{option_id}", ephemeral=True)

    @discord.ui.button(
        label="赞成",
        style=discord.ButtonStyle.success,
        custom_id="lingxu:vote:yes:v1",
    )
    async def yes(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[VoteView],
    ) -> None:
        await self._cast(interaction, "yes")

    @discord.ui.button(
        label="反对",
        style=discord.ButtonStyle.danger,
        custom_id="lingxu:vote:no:v1",
    )
    async def no(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[VoteView],
    ) -> None:
        await self._cast(interaction, "no")

    @discord.ui.button(
        label="弃权",
        style=discord.ButtonStyle.secondary,
        custom_id="lingxu:vote:abstain:v1",
    )
    async def abstain(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[VoteView],
    ) -> None:
        await self._cast(interaction, "abstain")


class PrivateVoteView(discord.ui.View):
    """Restart-safe Yes/No buttons bound to the recipient's DM message id."""

    def __init__(self, controller: MeetingController) -> None:
        super().__init__(timeout=None)
        self._controller = controller

    async def _cast(
        self,
        interaction: discord.Interaction[discord.Client],
        option_id: str,
    ) -> None:
        if interaction.message is None:
            await interaction.response.send_message("无法识别这张投票卡。")
            return
        await interaction.response.defer(thinking=True)
        try:
            await self._controller.vote_cast_private(
                message_id=interaction.message.id,
                user_id=interaction.user.id,
                option_id=option_id,
            )
        except (LingxuError, ValueError) as exc:
            await interaction.followup.send(str(exc))
            return
        label = "是" if option_id == "yes" else "否"
        await interaction.followup.send(f"已记录：{label}。截止前再次点击可修改选择。")

    @discord.ui.button(
        label="是",
        style=discord.ButtonStyle.success,
        custom_id="lingxu:private-vote:yes:v1",
    )
    async def yes(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[PrivateVoteView],
    ) -> None:
        await self._cast(interaction, "yes")

    @discord.ui.button(
        label="否",
        style=discord.ButtonStyle.danger,
        custom_id="lingxu:private-vote:no:v1",
    )
    async def no(
        self,
        interaction: discord.Interaction[discord.Client],
        _: discord.ui.Button[PrivateVoteView],
    ) -> None:
        await self._cast(interaction, "no")


class AgendaVoteModal(discord.ui.Modal, title="配置议程投票"):
    """Collect fields that only apply when an agenda requires a vote."""

    vote_content: discord.ui.TextInput[AgendaVoteModal] = discord.ui.TextInput(
        label="投票内容",
        placeholder="例如：是否批准本季度预算？",
        min_length=1,
        max_length=300,
    )
    vote_duration: discord.ui.TextInput[AgendaVoteModal] = discord.ui.TextInput(
        label="投票时长（秒）",
        placeholder="10 至 3600",
        default="60",
        min_length=1,
        max_length=4,
    )

    def __init__(
        self,
        *,
        controller: MeetingController,
        guild_id: int,
        actor_user_id: int,
        agenda_title: str,
        duration_minutes: int | None,
        language: str,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._guild_id = guild_id
        self._actor_user_id = actor_user_id
        self._agenda_title = agenda_title
        self._duration_minutes = duration_minutes
        self._language = language

    async def on_submit(self, interaction: discord.Interaction[discord.Client]) -> None:
        try:
            vote_duration_seconds = int(self.vote_duration.value.strip())
        except ValueError:
            await interaction.response.send_message("投票时长必须是整数秒。", ephemeral=True)
            return
        if not 10 <= vote_duration_seconds <= 3600:
            await interaction.response.send_message(
                "投票时长必须在 10 至 3600 秒之间。",
                ephemeral=True,
            )
            return
        try:
            item = await self._controller.agenda_add(
                guild_id=self._guild_id,
                actor_user_id=self._actor_user_id,
                title=self._agenda_title,
                duration_minutes=self._duration_minutes,
                requires_vote=True,
                vote_content=self.vote_content.value,
                vote_duration_seconds=vote_duration_seconds,
                language=self._language,
            )
        except (LingxuError, ValueError) as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"已添加议程 {item.position}：{item.title}（含私信投票）",
            ephemeral=True,
        )
