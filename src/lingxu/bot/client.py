"""Official Discord client for Lingxu meetings and resident voice assistance."""

from __future__ import annotations

import contextlib
import os
from asyncio import CancelledError, Lock, Task, create_task, sleep
from collections.abc import Iterable
from typing import cast

import discord
import structlog
from discord import app_commands
from discord.ext import commands

from lingxu.asr.factory import create_realtime_asr_provider
from lingxu.bot.assistant_commands import AssistantCommands
from lingxu.bot.meeting_commands import MeetingCommands
from lingxu.bot.meeting_panels import MeetingPanelCoordinator
from lingxu.bot.meeting_ui import HostControlView, PrivateVoteView, VoteView
from lingxu.bot.meeting_voice import MeetingVoiceCoordinator
from lingxu.bot.resident_assistant import ResidentAssistantCoordinator
from lingxu.bot.voice_modes import GuildVoiceModeRegistry
from lingxu.bot.wakeup_commands import (
    GuildWakePhraseCoordinator,
    WakePhraseRuntime,
    WakeupCommands,
)
from lingxu.config import Settings
from lingxu.console.auth import ConsoleSessionStore
from lingxu.console.server import HostConsoleServer
from lingxu.console.snapshot import ConsoleSnapshotStore
from lingxu.domain.enums import (
    AgendaImportStatus,
    JobStatus,
    MeetingEventType,
    MeetingState,
    ServiceCapacityMode,
)
from lingxu.domain.errors import (
    AgendaImportError,
    ServiceCapacityFullError,
    SessionNotFoundError,
)
from lingxu.domain.models import ServiceCapacityLeaseRecord
from lingxu.meeting.agenda_import import (
    AgendaImportResult,
    decode_markdown,
    structured_agenda_reference_ids,
)
from lingxu.meeting.factory import MeetingApplication, create_meeting_application
from lingxu.storage.guild_wake_phrase_repository import GuildWakePhraseRepository
from lingxu.storage.repositories import ProcessingJobRepository
from lingxu.storage.resident_assistant_repository import ResidentAssistantRepository
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository
from lingxu.tts.factory import create_tts_provider
from lingxu.voice_assistant.factory import (
    create_resident_voice_assistant,
    create_wake_phrase_validator,
)
from lingxu.voice_assistant.service import MeetingVoiceAssistant, SpeakCallback

_VOICE_RECOVERY_STATES = frozenset(
    {
        MeetingState.CONSENT_CHECK,
        MeetingState.OPENING,
        MeetingState.AGENDA_OVERVIEW,
        MeetingState.AGENDA_ITEM_INTRO,
        MeetingState.DISCUSSION_READY,
        MeetingState.SPEAKER_ACTIVE,
        MeetingState.DISCUSSION_COMPLETE,
        MeetingState.VOTE_OPEN,
        MeetingState.VOTE_RESULT,
        MeetingState.NEXT_AGENDA_ITEM,
        MeetingState.CLOSING,
        MeetingState.COMPLETED,
    }
)


def _requires_voice_recovery(
    state: MeetingState,
    resume_state: MeetingState | None,
) -> bool:
    return state in _VOICE_RECOVERY_STATES or (
        state is MeetingState.PAUSED and resume_state in _VOICE_RECOVERY_STATES
    )


def _mentions_discord_user(
    mentions: Iterable[discord.abc.Snowflake],
    user_id: int,
) -> bool:
    """Match a real Discord mention by immutable user ID, never display name."""

    return any(mention.id == user_id for mention in mentions)


async def _read_live_attachment(attachment: discord.Attachment) -> bytes:
    """Prefer the live CDN URL and fall back to Discord's best-effort proxy."""

    try:
        return await attachment.read(use_cached=False)
    except discord.HTTPException as original_error:
        if attachment.proxy_url == attachment.url:
            raise
        try:
            return await attachment.read(use_cached=True)
        except discord.HTTPException as cached_error:
            raise cached_error from original_error


class LingxuBot(commands.Bot):
    """Independent Lingxu meeting and resident-assistant Bot."""

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.messages = True
        application_id = settings.discord_application_id
        if application_id is None:
            raise ValueError("DISCORD_APPLICATION_ID is required")
        super().__init__(
            command_prefix=commands.when_mentioned,
            intents=intents,
            application_id=application_id,
            allowed_contexts=app_commands.AppCommandContext(
                guild=True,
                dm_channel=False,
                private_channel=False,
            ),
            allowed_installs=app_commands.AppInstallationType(guild=True, user=False),
        )
        self.voice_modes = GuildVoiceModeRegistry()
        self.meetings: MeetingApplication = create_meeting_application(settings)
        self._wake_phrases = GuildWakePhraseRepository(self.meetings.database.sessions)
        self.capacity = ServiceCapacityRepository(self.meetings.database.sessions)
        self._meeting_panels = MeetingPanelCoordinator(bot=self, application=self.meetings)
        self._settings = settings
        self._commands_synced = False
        self._flows_recovered = False
        self._imports_recovered = False
        self._resident_recovered = False
        self._capacity_reconciled = False
        self._agenda_import_tasks: dict[str, Task[None]] = {}
        self._timer_task: Task[None] | None = None
        self._console: HostConsoleServer | None = None
        self._manual_tts_lock = Lock()
        self._wake_recovery_locks: dict[int, Lock] = {}
        self._tts_provider = create_tts_provider(settings)
        self._meeting_voice = MeetingVoiceCoordinator(
            bot=self,
            settings=settings,
            application=self.meetings,
            tts_provider=self._tts_provider,
            voice_modes=self.voice_modes,
            capacity=self.capacity,
            wake_phrases=self._wake_phrases,
        )
        self._resident_assistant: ResidentAssistantCoordinator | None = None
        if settings.resident_assistant_enabled:

            def resident_factory(
                *,
                session_id: str,
                speak: SpeakCallback,
                wake_phrase: str,
            ) -> MeetingVoiceAssistant:
                return create_resident_voice_assistant(
                    settings=settings,
                    session_id=session_id,
                    asr_provider=create_realtime_asr_provider(settings),
                    speak=speak,
                    wake_phrase=wake_phrase,
                )

            self._resident_assistant = ResidentAssistantCoordinator(
                bot=self,
                settings=settings,
                repository=ResidentAssistantRepository(self.meetings.database.sessions),
                voice_modes=self.voice_modes,
                assistant_factory=resident_factory,
                tts_provider=self._tts_provider,
                wake_phrases=self._wake_phrases,
                capacity=self.capacity,
            )
        wake_runtimes: list[WakePhraseRuntime] = [self._meeting_voice]
        if self._resident_assistant is not None:
            wake_runtimes.append(self._resident_assistant)
        self._wake_phrase_validator = create_wake_phrase_validator(settings)
        self._wakeup = GuildWakePhraseCoordinator(
            repository=self._wake_phrases,
            runtimes=wake_runtimes,
            preflight=self._wake_phrase_validator.preflight,
            retry_recovery=self._retry_failed_wake_recoveries,
        )
        self.meetings.controller.set_runtime_action_handler(self._meeting_voice.handle_action)
        self.meetings.controller.set_vote_outcome_handler(self._meeting_voice.handle_vote_outcome)
        self.meetings.controller.set_speaking_flow_handler(
            self._meeting_voice.handle_speaking_flow_notice
        )
        self.meetings.controller.set_enforcement_handler(self._meeting_voice.sync_mute_enforcement)
        self.meetings.controller.set_consent_runtime(self._meeting_voice)
        self._log = structlog.get_logger(__name__)

    async def setup_hook(self) -> None:
        """Register the maintained production command groups."""

        await self.add_cog(
            MeetingCommands(
                self.meetings.controller,
                self.meetings.intents,
                self._meeting_voice,
                self._meeting_panels,
                self.meetings.pronunciation_aliases,
            )
        )
        if self._resident_assistant is not None:
            await self.add_cog(AssistantCommands(self._resident_assistant))
        await self.add_cog(WakeupCommands(self._wakeup))
        self.add_view(HostControlView(self.meetings.controller, self._meeting_panels))
        self.add_view(VoteView(self.meetings.controller))
        self.add_view(PrivateVoteView(self.meetings.controller))
        self._timer_task = create_task(self._timer_loop(), name="meeting-timer-loop")
        if self._settings.host_console_enabled:
            guild_id = self._settings.effective_host_console_guild_id
            token = self._settings.effective_host_console_token
            assert guild_id is not None
            assert token is not None
            self._console = HostConsoleServer(
                bind_host=self._settings.host_console_bind_host,
                port=self._settings.host_console_port,
                session_store=ConsoleSessionStore(token),
                snapshots=ConsoleSnapshotStore(
                    database=self.meetings.database,
                    controller=self.meetings.controller,
                    sessions=self.meetings.sessions,
                    guild_id=guild_id,
                ),
                controller=self.meetings.controller,
                manual_tts=self._manual_tts,
            )
            await self._console.start()

    async def _manual_tts(self, text: str) -> str:
        """Speak in the configured Guild, or safely fall back to its publication channel."""

        guild_id = self._settings.effective_host_console_guild_id
        if guild_id is None:
            raise ValueError("Host console Guild is not configured")
        async with self._manual_tts_lock:
            await self.capacity.ensure_available_to_guild(guild_id)
            return await self._meeting_voice.speak(guild_id=guild_id, text=text)

    async def _retry_failed_wake_recoveries(self, guild_id: int) -> None:
        """Retry each retained failed voice recovery once after a valid setting change."""

        recovery_lock = self._wake_recovery_locks.setdefault(guild_id, Lock())
        async with recovery_lock:
            session = await self._meeting_voice.retry_failed_wake_phrase_recovery(guild_id)
            if session is not None:
                try:
                    await self.meetings.controller.recover_automatic_flows(
                        session_ids={session.id},
                    )
                except Exception as exc:
                    self._log.error(
                        "meeting_wake_phrase_flow_recovery_retry_failed",
                        guild_id=guild_id,
                        session_id=session.id,
                        error_type=type(exc).__name__,
                    )
                else:
                    self._meeting_voice.complete_failed_wake_phrase_recovery(
                        guild_id=guild_id,
                        session_id=session.id,
                    )
                    try:
                        await self._meeting_panels.refresh(session.id)
                    except Exception as exc:
                        self._log.warning(
                            "meeting_wake_phrase_panel_refresh_failed",
                            guild_id=guild_id,
                            session_id=session.id,
                            error_type=type(exc).__name__,
                        )
            if self._resident_assistant is not None:
                await self._resident_assistant.retry_failed_wake_phrase_recovery(guild_id)

    async def _timer_loop(self) -> None:
        """Recover active deadlines and publish deterministic notices."""

        while True:
            try:
                if not self._flows_recovered:
                    await sleep(1)
                    continue
                await self.meetings.controller.reconcile_timer_handoffs()
                notices = await self.meetings.timers.tick()
                for notice in notices:
                    if (
                        notice.event_type is MeetingEventType.TIMER_REMINDER_60
                        and not notice.timer_type.startswith("FLOW:OPENING:")
                        and not notice.timer_type.startswith("FLOW:SPEAKING_")
                    ):
                        session = await self.meetings.sessions.get(notice.session_id)
                        await self._meeting_voice.speak(
                            guild_id=session.guild_id,
                            text="当前流程剩余约六十秒。",
                        )
                    await self.meetings.controller.handle_timer_notice(notice)
                results = await self.meetings.controller.monitor_post_processing_jobs()
                for result in results:
                    if result.session.state in {
                        MeetingState.PUBLISHED,
                        MeetingState.FAILED_POST_PROCESSING,
                    }:
                        await self._meeting_voice.release_capacity(
                            session=result.session,
                            reason=result.session.state.value,
                        )
                    await self._meeting_panels.refresh(result.session.id)
                await self._meeting_panels.refresh_recoverable()
                await self._release_terminal_job_capacity()
            except CancelledError:
                raise
            except Exception as exc:
                self._log.error("meeting_timer_tick_failed", error_type=type(exc).__name__)
            await sleep(1)

    async def on_ready(self) -> None:
        """Synchronise commands and publish an atomic readiness marker."""

        if not self._commands_synced:
            guild_id = self._settings.discord_test_guild_id
            if guild_id is None:
                synced = await self.tree.sync()
                scope = "global"
            else:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                scope = "test_guild"
            self._commands_synced = True
            self._log.info(
                "application_commands_synced",
                count=len(synced),
                scope=scope,
            )
        if not self._capacity_reconciled:
            await self._reconcile_capacity()
            self._capacity_reconciled = True
        if not self._flows_recovered:
            await self._meeting_voice.reconcile_mutes()
            recoverable_ids: set[str] = set()
            recoverable_sessions = await self.meetings.sessions.list_recoverable()
            paused_ids = {
                session.id
                for session in recoverable_sessions
                if session.state is MeetingState.PAUSED
            }
            if paused_ids:
                await self.meetings.controller.reconcile_timer_handoffs(
                    session_ids=paused_ids,
                )
            for session in recoverable_sessions:
                if session.state is MeetingState.POST_PROCESSING:
                    try:
                        await self.capacity.claim(
                            guild_id=session.guild_id,
                            mode=ServiceCapacityMode.MEETING,
                            owner_id=session.id,
                        )
                    except ServiceCapacityFullError:
                        self._log.warning(
                            "meeting_recovery_capacity_deferred",
                            guild_id=session.guild_id,
                            session_id=session.id,
                        )
                    else:
                        recoverable_ids.add(session.id)
                    continue
                if not _requires_voice_recovery(session.state, session.resume_state):
                    continue
                channel = (
                    self.get_channel(session.publication_channel_id)
                    if session.publication_channel_id is not None
                    else None
                )
                if not isinstance(channel, discord.VoiceChannel):
                    if session.publication_channel_id is not None:
                        self._meeting_voice.mark_failed_wake_phrase_recovery(
                            session=session,
                            channel_id=session.publication_channel_id,
                        )
                    self._log.error(
                        "meeting_flow_recovery_channel_unavailable",
                        guild_id=session.guild_id,
                        session_id=session.id,
                    )
                    continue
                try:
                    await self._meeting_voice.start(session=session, channel=channel)
                except Exception as exc:
                    self._meeting_voice.mark_failed_wake_phrase_recovery(
                        session=session,
                        channel_id=channel.id,
                    )
                    self._log.error(
                        "meeting_flow_voice_recovery_failed",
                        guild_id=session.guild_id,
                        session_id=session.id,
                        error_type=type(exc).__name__,
                    )
                    continue
                recoverable_ids.add(session.id)
            await self.meetings.controller.recover_automatic_flows(
                session_ids=recoverable_ids,
            )
            await self._meeting_panels.refresh_recoverable()
            self._flows_recovered = True
        if not self._imports_recovered:
            for agenda_import in await self.meetings.agenda_imports.list_processing():
                self._schedule_agenda_import(agenda_import.id, agenda_import.channel_id)
            self._imports_recovered = True
        if not self._resident_recovered and self._resident_assistant is not None:
            await self._resident_assistant.recover_all()
            self._resident_recovered = True
        marker = self._settings.runtime_tmp / "bot.ready"
        marker.parent.mkdir(parents=True, exist_ok=True)
        part = marker.with_name(f"{marker.name}.part")
        with part.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("ready\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, marker)
        if self.user is not None:
            self._log.info("bot_ready", bot_user_id=self.user.id)

    async def _reconcile_capacity(self) -> None:
        """Keep durable owners across restart and clear only provably stale work."""

        current = await self.capacity.current()
        if current is None:
            return
        if current.mode in {ServiceCapacityMode.PROBE, ServiceCapacityMode.AGENDA_IMPORT}:
            await self.capacity.release(current, reason="STALE_AFTER_RESTART")
            return
        if current.mode is ServiceCapacityMode.MEETING:
            recoverable = {
                session.id: session for session in await self.meetings.sessions.list_recoverable()
            }
            session = recoverable.get(current.owner_id)
            if (
                session is not None
                and session.guild_id == current.guild_id
                and (
                    _requires_voice_recovery(session.state, session.resume_state)
                    or session.state is MeetingState.POST_PROCESSING
                )
            ):
                await self.capacity.record_recovered(current)
                return
            try:
                latest = await self.meetings.sessions.get(current.owner_id)
            except SessionNotFoundError:
                await self.capacity.release(current, reason="STALE_AFTER_RESTART")
                return
            job = await ProcessingJobRepository(self.meetings.database.sessions).latest_for_session(
                current.owner_id, job_type="POST_MEETING"
            )
            stopped_actions_pending = False
            if latest.state is MeetingState.STOPPED and job is None:
                stopped_actions_pending = any(
                    candidate.id == latest.id
                    for candidate in await self.meetings.sessions.list_terminal_reconcilable()
                )
            if (
                latest.guild_id == current.guild_id
                and latest.state in {MeetingState.FAILED_POST_PROCESSING, MeetingState.STOPPED}
                and (
                    stopped_actions_pending
                    or (job is not None and job.status in {JobStatus.PENDING, JobStatus.RUNNING})
                )
            ):
                await self.capacity.record_recovered(current)
                return
            await self.capacity.release(current, reason="STALE_AFTER_RESTART")
            return
        active_residents = {
            record.id: record
            for record in await ResidentAssistantRepository(
                self.meetings.database.sessions
            ).list_active()
        }
        resident = active_residents.get(current.owner_id)
        if resident is None or resident.guild_id != current.guild_id:
            await self.capacity.release(current, reason="STALE_AFTER_RESTART")
            return
        await self.capacity.record_recovered(current)

    async def _release_terminal_job_capacity(self) -> None:
        """Release a terminal Meeting's lease after its independent Job finishes."""

        current = await self.capacity.current()
        if current is None or current.mode is not ServiceCapacityMode.MEETING:
            return
        try:
            session = await self.meetings.sessions.get(current.owner_id)
        except SessionNotFoundError:
            await self.capacity.release(current, reason="MEETING_OWNER_MISSING")
            return
        if session.state not in {
            MeetingState.FAILED_POST_PROCESSING,
            MeetingState.STOPPED,
        }:
            return
        job = await ProcessingJobRepository(self.meetings.database.sessions).latest_for_session(
            session.id, job_type="POST_MEETING"
        )
        if job is None or job.status in {JobStatus.PENDING, JobStatus.RUNNING}:
            return
        reason_prefix = (
            "RETRY"
            if session.state is MeetingState.FAILED_POST_PROCESSING
            else "STOPPED_POST_MEETING"
        )
        await self.capacity.release(current, reason=f"{reason_prefix}_{job.status.value}")

    def _schedule_agenda_import(
        self,
        import_id: str,
        channel_id: int,
        capacity_lease: ServiceCapacityLeaseRecord | None = None,
    ) -> None:
        existing = self._agenda_import_tasks.get(import_id)
        if existing is not None and not existing.done():
            return
        task = create_task(
            self._run_agenda_import(import_id, channel_id, capacity_lease),
            name=f"agenda-import-{import_id}",
        )
        self._agenda_import_tasks[import_id] = task

        def discard(completed: Task[None]) -> None:
            self._agenda_import_tasks.pop(import_id, None)
            with contextlib.suppress(CancelledError, Exception):
                completed.result()

        task.add_done_callback(discard)

    async def _send_agenda_import_message(self, channel_id: int, text: str) -> None:
        channel = self.get_channel(channel_id)
        if channel is None:
            channel = await self.fetch_channel(channel_id)
        await cast(discord.abc.Messageable, channel).send(text)

    async def _run_agenda_import(
        self,
        import_id: str,
        channel_id: int,
        capacity_lease: ServiceCapacityLeaseRecord | None,
    ) -> None:
        selected_lease = capacity_lease
        try:
            if selected_lease is None:
                agenda_import = await self.meetings.agenda_imports.get(import_id)
                session = await self.meetings.sessions.get(agenda_import.session_id)
                while selected_lease is None:
                    try:
                        selected_lease = await self.capacity.claim(
                            guild_id=session.guild_id,
                            mode=ServiceCapacityMode.AGENDA_IMPORT,
                            owner_id=import_id,
                        )
                    except ServiceCapacityFullError:
                        await sleep(1)
            result = await self.meetings.agenda_imports.process(import_id)
            await self._send_agenda_import_message(channel_id, result.ready_message)
        except CancelledError:
            raise
        except AgendaImportError as exc:
            await self._send_agenda_import_message(channel_id, str(exc))
            self._log.warning(
                "agenda_import_failed",
                import_id=import_id,
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            self._log.error(
                "agenda_import_reporting_failed",
                import_id=import_id,
                error_type=type(exc).__name__,
            )
        finally:
            if selected_lease is not None:
                await self.capacity.release(selected_lease, reason="AGENDA_IMPORT_FINISHED")

    async def on_message(self, message: discord.Message) -> None:
        """Accept one host-authored mentioned Markdown attachment without Message Content."""

        if message.guild is None or message.author.bot or self.user is None:
            return
        if not _mentions_discord_user(message.mentions, self.user.id):
            await self.process_commands(message)
            return
        stage = "validation"
        capacity_lease: ServiceCapacityLeaseRecord | None = None
        capacity_transferred = False
        try:
            session = await self.meetings.controller.active(message.guild.id)
            if message.author.id != session.creator_user_id:
                raise AgendaImportError("只有会议主持人可以导入议程文档。")
            expected_channel = session.setup_channel_id or session.publication_channel_id
            if expected_channel != message.channel.id:
                raise AgendaImportError("请在创建本次会议的频道中发送议程文档。")
            if len(message.attachments) != 1:
                raise AgendaImportError("请在一条消息中只附加一份 `.md` 议程文档。")
            attachment = message.attachments[0]
            if not attachment.filename.casefold().endswith(".md"):
                raise AgendaImportError("附件必须是 `.md` 格式的会议议程。")
            if attachment.size > self._settings.agenda_import_max_attachment_bytes:
                raise AgendaImportError("议程文档过大，请控制在 64 KiB 以内。")
            capacity_lease = await self.capacity.claim(
                guild_id=message.guild.id,
                mode=ServiceCapacityMode.AGENDA_IMPORT,
                owner_id=f"message:{message.id}",
            )
            stage = "attachment_download"
            payload = await _read_live_attachment(attachment)
            source_text = decode_markdown(payload)
            validated_identity_user_ids: list[int] = []
            for user_id in structured_agenda_reference_ids(source_text):
                member = message.guild.get_member(user_id)
                if member is None:
                    try:
                        member = await message.guild.fetch_member(user_id)
                    except discord.NotFound as exc:
                        raise AgendaImportError(
                            "结构化议程引用了不属于当前 Discord 服务器的成员。"
                        ) from exc
                    except (discord.Forbidden, discord.HTTPException) as exc:
                        raise AgendaImportError(
                            "聆序暂时无法验证结构化议程中的 Discord 成员。"
                        ) from exc
                if member.bot:
                    raise AgendaImportError("结构化议程不能把机器人设为发言者或演讲者。")
                validated_identity_user_ids.append(user_id)
            stage = "agenda_submit"
            submission = await self.meetings.agenda_imports.submit(
                guild_id=message.guild.id,
                actor_user_id=message.author.id,
                channel_id=message.channel.id,
                message_id=message.id,
                attachment_id=attachment.id,
                filename=attachment.filename,
                payload=payload,
                validated_identity_user_ids=tuple(validated_identity_user_ids),
            )
            stage = "discord_reply"
            if submission.record.status is AgendaImportStatus.READY:
                await message.channel.send(
                    AgendaImportResult(record=submission.record, agenda=()).ready_message
                )
            elif submission.record.status is AgendaImportStatus.FAILED:
                await message.channel.send("该议程文档此前处理失败，请重新发送一份附件。")
            else:
                if submission.created:
                    await message.channel.send("已收到议程文档，处理中。")
                else:
                    await message.channel.send("这份议程文档正在处理中，请稍候。")
                self._schedule_agenda_import(
                    submission.record.id,
                    submission.record.channel_id,
                    capacity_lease,
                )
                capacity_transferred = True
        except (AgendaImportError, ServiceCapacityFullError, SessionNotFoundError) as exc:
            try:
                await message.channel.send(str(exc))
            except discord.HTTPException as reply_error:
                self._log.warning(
                    "agenda_import_discord_failed",
                    stage="validation_reply",
                    error_type=type(reply_error).__name__,
                    http_status=reply_error.status,
                    discord_code=reply_error.code,
                )
        except discord.HTTPException as exc:
            self._log.warning(
                "agenda_import_discord_failed",
                stage=stage,
                error_type=type(exc).__name__,
                http_status=exc.status,
                discord_code=exc.code,
            )
            if stage == "attachment_download":
                try:
                    await message.channel.send(
                        "无法从 Discord 下载议程附件，请重新上传 `.md` 文件后再试。"
                    )
                except discord.HTTPException as reply_error:
                    self._log.warning(
                        "agenda_import_discord_failed",
                        stage="attachment_failure_reply",
                        error_type=type(reply_error).__name__,
                        http_status=reply_error.status,
                        discord_code=reply_error.code,
                    )
        finally:
            if capacity_lease is not None and not capacity_transferred:
                await self.capacity.release(capacity_lease, reason="AGENDA_IMPORT_NOT_STARTED")
        await self.process_commands(message)

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Track formal Meeting and resident-assistant voice membership."""

        if self.user is not None and member.id == self.user.id:
            if self._resident_assistant is not None:
                await self._resident_assistant.handle_bot_voice_state(
                    guild_id=member.guild.id,
                    before_channel_id=before.channel.id if before.channel is not None else None,
                    after_channel_id=after.channel.id if after.channel is not None else None,
                )
            return
        if member.bot:
            return
        meeting_channel_id = self._meeting_voice.channel_id(member.guild.id)
        if meeting_channel_id is None:
            return
        was_present = before.channel is not None and before.channel.id == meeting_channel_id
        is_present = after.channel is not None and after.channel.id == meeting_channel_id
        if was_present == is_present:
            if before.mute != after.mute:
                await self._meeting_voice.handle_member_voice_state(guild_id=member.guild.id)
            return
        try:
            if is_present:
                await self.meetings.controller.participant_join(
                    guild_id=member.guild.id,
                    user_id=member.id,
                    display_name=member.display_name,
                )
            else:
                await self.meetings.controller.participant_leave(
                    guild_id=member.guild.id,
                    user_id=member.id,
                )
            await self._meeting_voice.handle_member_voice_state(guild_id=member.guild.id)
        except Exception as exc:
            self._log.warning(
                "meeting_participant_presence_failed",
                error_type=type(exc).__name__,
            )

    async def close(self) -> None:
        """Close all maintained runtime resources before Discord shutdown."""

        marker = self._settings.runtime_tmp / "bot.ready"
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()
        if self._timer_task is not None:
            self._timer_task.cancel()
            with contextlib.suppress(CancelledError):
                await self._timer_task
        import_tasks = tuple(self._agenda_import_tasks.values())
        for task in import_tasks:
            task.cancel()
        for task in import_tasks:
            with contextlib.suppress(CancelledError):
                await task
        if self._console is not None:
            await self._console.stop()
        if self._resident_assistant is not None:
            await self._resident_assistant.close_all()
        await self._meeting_voice.close_all()
        await self.meetings.database.close()
        await super().close()


def create_bot(settings: Settings) -> LingxuBot:
    """Create a Bot without enabling privileged Message Content or member intents."""

    return LingxuBot(settings)
