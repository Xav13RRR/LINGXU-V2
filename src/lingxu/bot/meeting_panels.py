"""Discord delivery and rendering for public status and private host panels."""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any, cast

import discord
import structlog

from lingxu.domain.enums import AgendaSpeakingMode, AgendaStatus, SpeakingFlowStage, TimerStatus
from lingxu.domain.models import MeetingSessionRecord, TimerRecord
from lingxu.meeting.controller import MeetingStatus
from lingxu.meeting.factory import MeetingApplication


def _remaining_label(timer: TimerRecord, now: datetime) -> str:
    if timer.status is TimerStatus.PAUSED:
        seconds = max(0, math.ceil((timer.remaining_ms or 0) / 1000))
        return f"已暂停, 剩余 {seconds} 秒"
    if timer.deadline_at <= now:
        return "已到时"
    return f"<t:{math.floor(timer.deadline_at.timestamp())}:R>"


def _display_name(status: MeetingStatus, user_id: int) -> str:
    floor = status.floor
    value = floor.display_name(user_id) if floor is not None else f"成员{user_id}"
    return discord.utils.escape_markdown(value)


def _flow_timer(status: MeetingStatus) -> TimerRecord | None:
    floor = status.floor
    if floor is not None and floor.timer_type is not None:
        return next(
            (timer for timer in status.timers if timer.timer_type == floor.timer_type), None
        )
    active_item = next(
        (item for item in status.agenda if item.status is AgendaStatus.ACTIVE),
        None,
    )
    if active_item is not None:
        agenda_type = f"FLOW:AGENDA:{active_item.id}"
        selected = next(
            (timer for timer in status.timers if timer.timer_type == agenda_type),
            None,
        )
        if selected is not None:
            return selected
    return next(
        (timer for timer in status.timers if timer.timer_type.startswith("VOTE:")),
        None,
    )


def _floor_lines(status: MeetingStatus, now: datetime) -> list[str]:
    floor = status.floor
    if floor is None:
        current = status.speaking.current_user_id
        return [
            "发言模式: 自由发言",
            f"当前发言: {_display_name(status, current) if current is not None else '无'}",
        ]
    timer = _flow_timer(status)
    remaining = _remaining_label(timer, now) if timer is not None else "未计时"
    if floor.stage is SpeakingFlowStage.FOLLOWUP:
        current = status.speaking.current_user_id
        return [
            f"发言模式: {floor.followup_mode.value}",
            "当前发言: "
            + (_display_name(status, current) if current is not None else "等待主持人选择"),
            f"排队人数: {len(status.speaking.queued_user_ids)}",
        ]
    if floor.stage is SpeakingFlowStage.COMPLETED:
        return ["发言流程: 已完成"]
    current_names = (
        "、".join(_display_name(status, value) for value in floor.current_user_ids) or "无"
    )
    following = "、".join(_display_name(status, value) for value in floor.next_user_ids) or "无"
    lines = [
        f"发言模式: {floor.mode.value} / {floor.stage.value}",
        f"当前: {current_names}",
        f"下一位: {following}",
        f"剩余: {remaining}",
    ]
    if floor.mode is AgendaSpeakingMode.ORDERED:
        order = " → ".join(_display_name(status, turn.user_id) for turn in floor.turns)
        lines.append(f"完整顺序: {order or '无'}")
    return lines


def render_meeting_panel(
    status: MeetingStatus,
    *,
    private: bool,
    now: datetime | None = None,
) -> str:
    """Render bounded content without controls or user mentions."""

    selected_now = now or datetime.now(UTC)
    completed = sum(item.status is AgendaStatus.COMPLETED for item in status.agenda)
    active = next(
        (item for item in status.agenda if item.status is AgendaStatus.ACTIVE),
        None,
    )
    heading = "聆序主持控制面板" if private else "聆序会议状态"
    lines = [
        f"**{heading}**",
        f"会议状态: `{status.session.state.value}`",
        f"议程进度: {completed}/{len(status.agenda)}",
        f"当前议题: {discord.utils.escape_markdown(active.title) if active else '暂无'}",
        *_floor_lines(status, selected_now),
    ]
    if status.enforcement is not None:
        enforcement = status.enforcement
        enforcement_label = (
            "已降级为提示模式"
            if enforcement.downgraded
            else "已结束"
            if not enforcement.enabled
            else f"已暂停 ({enforcement.pause_reason or '未说明原因'})"
            if enforcement.paused
            else "主持介入中"
            if enforcement.host_intervention_active
            else "强制静音运行中"
        )
        lines.append(f"静音执行: {enforcement_label}")
        if enforcement.exception_user_ids:
            lines.append(
                "临时发言: "
                + "、".join(
                    _display_name(status, user_id) for user_id in enforcement.exception_user_ids
                )
            )
    if status.vote is not None:
        lines.append(f"进行中投票: {discord.utils.escape_markdown(status.vote.title)}")
    if status.agenda:
        lines.append("议程:")
        for item in status.agenda[:20]:
            marker = (
                "▶"
                if item.status is AgendaStatus.ACTIVE
                else "✓"
                if item.status is AgendaStatus.COMPLETED
                else "·"
            )
            lines.append(f"{marker} {item.position}. {discord.utils.escape_markdown(item.title)}")
    if private:
        lines.extend(
            (
                "",
                "下方按钮仅在本私信中供主持人使用; 公开状态消息不含控制组件。",
            )
        )
    content = "\n".join(lines)
    return content if len(content) <= 1_950 else f"{content[:1_940]}\n…"


class MeetingPanelCoordinator:
    """Deliver, reissue, and refresh panels from durable binding facts."""

    def __init__(self, *, bot: discord.Client, application: MeetingApplication) -> None:
        self._bot = bot
        self._application = application
        self._digests: dict[tuple[str, str], str] = {}
        self._log = structlog.get_logger(__name__)

    async def status(self, session_id: str) -> MeetingStatus:
        return await self._application.controller.status_for_session(session_id)

    async def content(self, session_id: str, *, private: bool) -> str:
        return render_meeting_panel(await self.status(session_id), private=private)

    async def bind_public(
        self,
        *,
        session: MeetingSessionRecord,
        channel_id: int,
        message_id: int,
        content: str,
    ) -> None:
        await self._application.panel_bindings.bind_public(
            session_id=session.id,
            channel_id=channel_id,
            message_id=message_id,
            actor_user_id=session.creator_user_id,
        )
        self._digests[(session.id, "public")] = hashlib.sha256(content.encode()).hexdigest()

    async def deliver_host(
        self,
        *,
        session: MeetingSessionRecord,
        user: discord.User | discord.Member,
    ) -> None:
        from lingxu.bot.meeting_ui import HostControlView

        status = await self.status(session.id)
        content = render_meeting_panel(status, private=True)
        channel = await user.create_dm()
        message = await channel.send(
            content,
            view=HostControlView(self._application.controller, self, status=status),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        await self._application.panel_bindings.bind_host(
            session_id=session.id,
            host_user_id=session.creator_user_id,
            channel_id=channel.id,
            message_id=message.id,
        )
        self._digests[(session.id, "host")] = hashlib.sha256(content.encode()).hexdigest()

    async def resolve_host_guild(self, *, message_id: int, user_id: int) -> int:
        session = await self._application.panel_bindings.resolve_host_panel(
            message_id=message_id,
            user_id=user_id,
        )
        return session.guild_id

    async def refresh_guild(self, guild_id: int) -> None:
        session = await self._application.sessions.active_for_guild(guild_id)
        if session is None:
            session = await self._application.sessions.latest_for_guild(guild_id)
        if session is not None:
            await self.refresh(session.id)

    async def refresh_recoverable(self) -> None:
        for session in await self._application.sessions.list_recoverable():
            await self.refresh(session.id)

    async def refresh(self, session_id: str) -> None:
        from lingxu.bot.meeting_ui import HostControlView

        binding = await self._application.panel_bindings.binding(session_id)
        status = await self.status(session_id)
        public_content = render_meeting_panel(status, private=False)
        host_content = render_meeting_panel(status, private=True)
        if binding.public_channel_id is not None and binding.public_message_id is not None:
            await self._edit_public(
                session_id=session_id,
                channel_id=binding.public_channel_id,
                message_id=binding.public_message_id,
                content=public_content,
            )
        if (
            binding.host_user_id is not None
            and binding.host_dm_message_id is not None
            and binding.host_user_id == status.session.creator_user_id
        ):
            try:
                user = self._bot.get_user(binding.host_user_id)
                if user is None:
                    user = await self._bot.fetch_user(binding.host_user_id)
                channel = await user.create_dm()
                digest = hashlib.sha256(host_content.encode()).hexdigest()
                if self._digests.get((session_id, "host")) != digest:
                    message = await channel.fetch_message(binding.host_dm_message_id)
                    await message.edit(
                        content=host_content,
                        view=HostControlView(
                            self._application.controller,
                            self,
                            status=status,
                        ),
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                    self._digests[(session_id, "host")] = digest
            except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
                self._log.warning(
                    "meeting_host_panel_refresh_failed",
                    session_id=session_id,
                    host_user_id=binding.host_user_id,
                    error_type=type(exc).__name__,
                )

    async def _edit_public(
        self,
        *,
        session_id: str,
        channel_id: int,
        message_id: int,
        content: str,
    ) -> None:
        digest = hashlib.sha256(content.encode()).hexdigest()
        if self._digests.get((session_id, "public")) == digest:
            return
        try:
            channel = self._bot.get_channel(channel_id)
            if channel is None:
                channel = await self._bot.fetch_channel(channel_id)
            message = await cast(Any, channel).fetch_message(message_id)
            await message.edit(
                content=content,
                view=None,
                allowed_mentions=discord.AllowedMentions.none(),
            )
            self._digests[(session_id, "public")] = digest
        except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
            self._log.warning(
                "meeting_public_panel_refresh_failed",
                session_id=session_id,
                channel_id=channel_id,
                error_type=type(exc).__name__,
            )
