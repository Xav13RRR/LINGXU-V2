"""Resident Discord voice Q&A lifecycle without Meeting capture or artifacts."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import discord
import structlog
from discord.ext import voice_recv

from lingxu.bot.voice_modes import (
    GuildVoiceLease,
    GuildVoiceMode,
    GuildVoiceModeRegistry,
)
from lingxu.config import Settings
from lingxu.domain.enums import ServiceCapacityMode
from lingxu.domain.errors import (
    ResidentAssistantNotActiveError,
    ResidentAssistantPermissionError,
    ServiceCapacityFullError,
    VoiceModeConflictError,
)
from lingxu.domain.models import ResidentAssistantSessionRecord, ServiceCapacityLeaseRecord
from lingxu.storage.guild_wake_phrase_repository import GuildWakePhraseRepository
from lingxu.storage.resident_assistant_repository import ResidentAssistantRepository
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository
from lingxu.tts.discord_playback import DiscordVoicePlaybackTarget
from lingxu.tts.provider import EphemeralTTSProvider, TTSProvider
from lingxu.utils import new_ulid
from lingxu.voice.dave_adapter import inspect_dave_status
from lingxu.voice.sink import AssistantOnlyPcmSink
from lingxu.voice_assistant.service import MeetingVoiceAssistant


class ResidentAssistantFactory(Protocol):
    """Build one approved, fully local assistant without starting it."""

    def __call__(
        self,
        *,
        session_id: str,
        speak: Callable[[str], Awaitable[None]],
        wake_phrase: str,
    ) -> MeetingVoiceAssistant: ...


@dataclass(slots=True)
class _ResidentRuntime:
    record: ResidentAssistantSessionRecord
    channel: discord.VoiceChannel
    client: Any
    assistant: MeetingVoiceAssistant
    sink: AssistantOnlyPcmSink
    lease: GuildVoiceLease
    capacity_lease: ServiceCapacityLeaseRecord | None
    playback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stopping: bool = False


@dataclass(slots=True)
class _ResidentStartResources:
    """Every independently releasable resource acquired by one start attempt."""

    session_id: str
    guild_id: int
    capacity_lease: ServiceCapacityLeaseRecord | None = None
    voice_lease: GuildVoiceLease | None = None
    assistant: MeetingVoiceAssistant | None = None
    client: Any | None = None
    created_record: ResidentAssistantSessionRecord | None = None
    runtime: _ResidentRuntime | None = None
    capacity_claim_attempted: bool = False
    voice_claim_attempted: bool = False
    assistant_start_attempted: bool = False
    record_create_attempted: bool = False


class ResidentAssistantCoordinator:
    """Own a durable, content-ephemeral resident voice assistant."""

    def __init__(
        self,
        *,
        bot: discord.Client,
        settings: Settings,
        repository: ResidentAssistantRepository,
        voice_modes: GuildVoiceModeRegistry,
        assistant_factory: ResidentAssistantFactory,
        tts_provider: TTSProvider,
        wake_phrases: GuildWakePhraseRepository,
        capacity: ServiceCapacityRepository | None = None,
    ) -> None:
        if not isinstance(tts_provider, EphemeralTTSProvider):
            raise TypeError("resident assistant requires an ephemeral TTS provider")
        self._bot = bot
        self._settings = settings
        self._repository = repository
        self._voice_modes = voice_modes
        self._assistant_factory = assistant_factory
        self._tts = cast(EphemeralTTSProvider, tts_provider)
        self._capacity = capacity
        self._wake_phrases = wake_phrases
        self._sessions: dict[int, _ResidentRuntime] = {}
        self._failed_wake_recoveries: dict[int, ResidentAssistantSessionRecord] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._global_start_lock = asyncio.Lock()
        self._log = structlog.get_logger(__name__)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        return self._locks.setdefault(guild_id, asyncio.Lock())

    def channel_id(self, guild_id: int) -> int | None:
        selected = self._sessions.get(guild_id)
        return selected.channel.id if selected is not None else None

    def runtime_active(self, guild_id: int) -> bool:
        selected = self._sessions.get(guild_id)
        return bool(selected is not None and selected.client.is_connected())

    async def update_wake_phrase(self, *, guild_id: int, phrase: str) -> bool:
        """Apply a validated phrase to this Guild's active resident runtime."""

        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if selected is None:
                return False
            await selected.assistant.update_wake_phrase(phrase)
            return True

    async def status(self, guild_id: int) -> ResidentAssistantSessionRecord:
        selected = self._sessions.get(guild_id)
        if selected is not None:
            return selected.record
        record = await self._repository.active_for_guild(guild_id)
        if record is None:
            raise ResidentAssistantNotActiveError("该服务器没有常驻语音助手。")
        return record

    async def _wait_for_dave(self, client: Any) -> None:
        deadline = asyncio.get_running_loop().time() + (
            self._settings.voice_duplex_dave_ready_timeout_seconds
        )
        while asyncio.get_running_loop().time() < deadline:
            if inspect_dave_status(client).ready:
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("resident assistant inbound DAVE path did not become ready")

    async def _remove_failed_start_runtime(self, resources: _ResidentStartResources) -> None:
        runtime = resources.runtime
        if runtime is not None and self._sessions.get(resources.guild_id) is runtime:
            self._sessions.pop(resources.guild_id, None)

    @staticmethod
    async def _stop_failed_start_input(resources: _ResidentStartResources) -> None:
        if resources.client is not None and resources.client.is_listening():
            resources.client.stop_listening()

    @staticmethod
    async def _stop_failed_start_assistant(resources: _ResidentStartResources) -> None:
        if resources.assistant_start_attempted and resources.assistant is not None:
            await resources.assistant.stop()

    @staticmethod
    async def _disconnect_failed_start_client(resources: _ResidentStartResources) -> None:
        if resources.client is not None:
            await resources.client.disconnect(force=True)

    async def _stop_failed_start_record(self, resources: _ResidentStartResources) -> None:
        if not resources.record_create_attempted:
            return
        created_record = resources.created_record
        if created_record is None:
            candidate = await self._repository.active_for_guild(resources.guild_id)
            if candidate is None or candidate.id != resources.session_id:
                return
            created_record = candidate
        await self._repository.stop_active(
            session_id=created_record.id,
            actor_user_id=None,
            reason="STARTUP_FAILED",
        )

    async def _release_failed_start_voice(self, resources: _ResidentStartResources) -> None:
        if not resources.voice_claim_attempted:
            return
        voice_lease = resources.voice_lease
        if voice_lease is None:
            candidate = self._voice_modes.current(resources.guild_id)
            if (
                candidate is not None
                and candidate.mode is GuildVoiceMode.RESIDENT_ASSISTANT
                and candidate.owner_id == resources.session_id
            ):
                voice_lease = candidate
        if voice_lease is not None:
            await self._voice_modes.release(voice_lease)

    async def _release_failed_start_capacity(
        self,
        resources: _ResidentStartResources,
        *,
        reason: str,
    ) -> None:
        if not resources.capacity_claim_attempted or self._capacity is None:
            return
        if resources.capacity_lease is not None:
            await self._capacity.release(resources.capacity_lease, reason=reason)
            return
        await self._capacity.release_owner(
            guild_id=resources.guild_id,
            mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
            owner_id=resources.session_id,
            reason=reason,
        )

    async def _erase_failed_start_tts(self, resources: _ResidentStartResources) -> None:
        self._erase_ephemeral_tts(resources.session_id)

    async def _rollback_failed_start(
        self,
        resources: _ResidentStartResources,
        *,
        reason: str,
    ) -> tuple[BaseException, ...]:
        """Attempt every exact-owner cleanup step after any partial start."""

        errors: list[BaseException] = []
        cleanup_steps = (
            self._remove_failed_start_runtime(resources),
            self._stop_failed_start_input(resources),
            self._stop_failed_start_assistant(resources),
            self._disconnect_failed_start_client(resources),
            self._stop_failed_start_record(resources),
            self._release_failed_start_voice(resources),
            self._release_failed_start_capacity(resources, reason=reason),
            self._erase_failed_start_tts(resources),
        )
        for cleanup_step in cleanup_steps:
            try:
                await cleanup_step
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)

    async def _finish_failed_start_cleanup(
        self,
        resources: _ResidentStartResources,
        *,
        reason: str,
    ) -> tuple[BaseException, ...]:
        """Finish compensation in an independent task despite repeated cancellation."""

        cleanup = asyncio.create_task(
            self._rollback_failed_start(resources, reason=reason),
            name=f"resident-start-cleanup-{resources.session_id}",
        )
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                if cleanup.done():
                    break
        return cleanup.result()

    def _log_start_cleanup_errors(
        self,
        resources: _ResidentStartResources,
        *,
        stage: str,
        errors: tuple[BaseException, ...],
    ) -> None:
        if not errors:
            return
        self._log.error(
            "resident_assistant_start_cleanup_incomplete",
            guild_id=resources.guild_id,
            session_id=resources.session_id,
            stage=stage,
            error_count=len(errors),
        )

    async def start(
        self,
        *,
        channel: discord.VoiceChannel,
        creator_user_id: int,
    ) -> ResidentAssistantSessionRecord:
        """Prepare local models, connect, listen, then commit the durable ACTIVE lease."""

        async with self._global_start_lock, self._lock(channel.guild.id):
            existing = self._sessions.get(channel.guild.id)
            if existing is not None:
                if existing.channel.id == channel.id and existing.client.is_connected():
                    return existing.record
                raise VoiceModeConflictError("该服务器已有常驻语音助手正在运行。")
            if (
                self._capacity is None
                and len(self._sessions) >= self._settings.resident_assistant_max_sessions
            ):
                raise VoiceModeConflictError("当前部署已达到常驻语音助手会话上限。")
            durable = await self._repository.active_for_guild(channel.guild.id)
            if durable is not None:
                raise VoiceModeConflictError("该服务器的常驻语音助手正在等待重启恢复。")
            session_id = new_ulid()
            resources = _ResidentStartResources(
                session_id=session_id,
                guild_id=channel.guild.id,
            )
            stage = "capacity_claim"
            try:
                resources.capacity_claim_attempted = self._capacity is not None
                resources.capacity_lease = (
                    await self._capacity.claim(
                        guild_id=channel.guild.id,
                        mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
                        owner_id=session_id,
                    )
                    if self._capacity is not None
                    else None
                )
                stage = "voice_claim"
                resources.voice_claim_attempted = True
                resources.voice_lease = await self._voice_modes.claim(
                    guild_id=channel.guild.id,
                    mode=GuildVoiceMode.RESIDENT_ASSISTANT,
                    owner_id=session_id,
                )
                voice_lease = resources.voice_lease
                assert voice_lease is not None
                stage = "connect"
                resources.runtime = await self._connect(
                    session_id=session_id,
                    channel=channel,
                    creator_user_id=creator_user_id,
                    lease=voice_lease,
                    capacity_lease=resources.capacity_lease,
                    recovered_record=None,
                    resources=resources,
                )
            except BaseException:
                cleanup_reason = {
                    "capacity_claim": "CAPACITY_CLAIM_FAILED",
                    "voice_claim": "VOICE_CLAIM_FAILED",
                }.get(stage, "STARTUP_FAILED")
                errors = await self._finish_failed_start_cleanup(
                    resources,
                    reason=cleanup_reason,
                )
                self._log_start_cleanup_errors(resources, stage=stage, errors=errors)
                raise
            runtime = resources.runtime
            assert runtime is not None
            self._sessions[channel.guild.id] = runtime
            self._log.info(
                "resident_assistant_started",
                guild_id=channel.guild.id,
                channel_id=channel.id,
                session_id=session_id,
            )
            return runtime.record

    async def _connect(
        self,
        *,
        session_id: str,
        channel: discord.VoiceChannel,
        creator_user_id: int,
        lease: GuildVoiceLease,
        capacity_lease: ServiceCapacityLeaseRecord | None,
        recovered_record: ResidentAssistantSessionRecord | None,
        resources: _ResidentStartResources,
    ) -> _ResidentRuntime:
        guild_voice = cast(Any, channel.guild.voice_client)
        if guild_voice is not None and guild_voice.is_connected():
            raise VoiceModeConflictError("Bot 已在该服务器的其它语音任务中。")

        wake_phrase = (await self._wake_phrases.get(channel.guild.id)).phrase
        resources.assistant = self._assistant_factory(
            session_id=session_id,
            speak=lambda text: self._speak(session_id, channel.guild.id, text),
            wake_phrase=wake_phrase,
        )
        assistant = resources.assistant
        resources.assistant_start_attempted = True
        await assistant.start()
        resources.client = await channel.connect(
            cls=voice_recv.VoiceRecvClient,
            reconnect=True,
            self_deaf=False,
        )
        await self._wait_for_dave(resources.client)
        sink = AssistantOnlyPcmSink(assistant=assistant)
        if recovered_record is None:
            resources.record_create_attempted = True
            record = await self._repository.create_active(
                session_id=session_id,
                guild_id=channel.guild.id,
                channel_id=channel.id,
                creator_user_id=creator_user_id,
            )
            resources.created_record = record
        else:
            record = recovered_record
        runtime = _ResidentRuntime(
            record=record,
            channel=channel,
            client=resources.client,
            assistant=assistant,
            sink=sink,
            lease=lease,
            capacity_lease=capacity_lease,
        )
        resources.runtime = runtime
        resources.client.listen(sink, after=self._receive_after(channel.guild.id, session_id))
        return runtime

    def _receive_after(self, guild_id: int, session_id: str) -> Any:
        loop = asyncio.get_running_loop()

        def after(error: BaseException | None) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._receiver_ended(guild_id, session_id, error),
                    name=f"resident-assistant-receiver-ended-{session_id}",
                )
            )

        return after

    async def _receiver_ended(
        self,
        guild_id: int,
        session_id: str,
        error: BaseException | None,
    ) -> None:
        selected = self._sessions.get(guild_id)
        if selected is None or selected.record.id != session_id or selected.stopping:
            return
        self._log.warning(
            "resident_assistant_receiver_ended",
            guild_id=guild_id,
            session_id=session_id,
            error_type=type(error).__name__ if error is not None else None,
        )
        await self._stop(
            guild_id=guild_id,
            actor_user_id=None,
            reason="RECEIVER_ENDED",
            persist=True,
        )

    def _ephemeral_tts_dir(self, session_id: str) -> Path:
        return self._settings.runtime_tmp / "resident-tts" / session_id

    async def _speak(self, session_id: str, guild_id: int, text: str) -> None:
        selected = self._sessions.get(guild_id)
        if selected is None or selected.record.id != session_id or selected.stopping:
            raise RuntimeError("resident assistant voice session is unavailable")
        audio = None
        async with selected.playback_lock:
            try:
                audio = await self._tts.synthesize_ephemeral(
                    text,
                    output_dir=self._ephemeral_tts_dir(session_id),
                )
                if audio.duration_ms > 15_000:
                    audio.path.unlink(missing_ok=True)
                    audio = await self._tts.synthesize_ephemeral(
                        "回答较长。请缩小问题范围后再次使用唤醒词。",
                        output_dir=self._ephemeral_tts_dir(session_id),
                    )
                if audio.duration_ms > 15_000:
                    raise RuntimeError("resident assistant TTS exceeds the hard duration limit")
                await DiscordVoicePlaybackTarget(
                    selected.client,
                    volume=self._settings.tts_playback_volume,
                ).play(audio)
            finally:
                if audio is not None:
                    audio.path.unlink(missing_ok=True)

    async def stop(
        self,
        *,
        guild_id: int,
        caller: discord.Member,
    ) -> None:
        selected = self._sessions.get(guild_id)
        record = (
            selected.record
            if selected is not None
            else await self._repository.active_for_guild(guild_id)
        )
        if record is None:
            raise ResidentAssistantNotActiveError("该服务器没有常驻语音助手。")
        if caller.id != record.creator_user_id and not caller.guild_permissions.manage_guild:
            raise ResidentAssistantPermissionError("只有启动者或服务器管理者可以停止常驻助手。")
        await self._stop(
            guild_id=guild_id,
            actor_user_id=caller.id,
            reason="COMMAND",
            persist=True,
        )

    async def handle_bot_voice_state(
        self,
        *,
        guild_id: int,
        before_channel_id: int | None,
        after_channel_id: int | None,
    ) -> None:
        selected = self._sessions.get(guild_id)
        if selected is None or selected.stopping:
            return
        if before_channel_id == selected.channel.id and after_channel_id != selected.channel.id:
            await self._stop(
                guild_id=guild_id,
                actor_user_id=None,
                reason="BOT_REMOVED",
                persist=True,
            )

    async def _stop(
        self,
        *,
        guild_id: int,
        actor_user_id: int | None,
        reason: str,
        persist: bool,
    ) -> None:
        async with self._lock(guild_id):
            selected = self._sessions.pop(guild_id, None)
            if selected is None:
                record = await self._repository.active_for_guild(guild_id)
                if record is None:
                    return
                if persist:
                    stopped = await self._repository.stop_active(
                        session_id=record.id,
                        actor_user_id=actor_user_id,
                        reason=reason,
                    )
                    if stopped is not None:
                        await self._release_capacity_owner(record, reason=reason)
                return
            selected.stopping = True
            errors: list[BaseException] = []
            durable_stopped = False
            if persist:
                try:
                    stopped = await self._repository.stop_active(
                        session_id=selected.record.id,
                        actor_user_id=actor_user_id,
                        reason=reason,
                    )
                    durable_stopped = stopped is not None
                except BaseException as exc:
                    errors.append(exc)
            try:
                if selected.client.is_listening():
                    selected.client.stop_listening()
            except BaseException as exc:
                errors.append(exc)
            try:
                await selected.assistant.stop()
            except BaseException as exc:
                errors.append(exc)
            try:
                await selected.client.disconnect(force=True)
            except BaseException as exc:
                errors.append(exc)
            await self._voice_modes.release(selected.lease)
            if (
                persist
                and durable_stopped
                and selected.capacity_lease is not None
                and self._capacity is not None
            ):
                try:
                    await self._capacity.release(selected.capacity_lease, reason=reason)
                except BaseException as exc:
                    errors.append(exc)
            self._erase_ephemeral_tts(selected.record.id)
            self._log.info(
                "resident_assistant_stopped",
                guild_id=guild_id,
                channel_id=selected.channel.id,
                session_id=selected.record.id,
                reason=reason,
                error_count=len(errors),
            )
            if errors:
                raise RuntimeError("resident assistant finalisation failed") from errors[0]

    def _erase_ephemeral_tts(self, session_id: str) -> None:
        selected_dir = self._ephemeral_tts_dir(session_id).expanduser().resolve()
        if not selected_dir.is_dir():
            return
        for path in selected_dir.iterdir():
            resolved = path.resolve()
            if resolved.parent == selected_dir and resolved.is_file():
                resolved.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            selected_dir.rmdir()

    async def recover_all(self) -> None:
        """Reconnect durable ACTIVE leases after formal Meeting recovery has run."""

        if not self._settings.resident_assistant_recover_on_restart:
            return
        for record in await self._repository.list_active():
            await self._recover_record(record)

    async def retry_failed_wake_phrase_recovery(self, guild_id: int) -> bool:
        """Retry exactly one retained failed recovery after its Guild setting changes."""

        record = self._failed_wake_recoveries.get(guild_id)
        if record is None:
            return False
        return await self._recover_record(record)

    async def _recover_record(self, record: ResidentAssistantSessionRecord) -> bool:
        """Recover one durable owner under the same lock order as a fresh start."""

        async with self._global_start_lock, self._lock(record.guild_id):
            existing = self._sessions.get(record.guild_id)
            if existing is not None:
                if existing.record.id == record.id and existing.client.is_connected():
                    self._failed_wake_recoveries.pop(record.guild_id, None)
                    return True
                return False
            durable = await self._repository.active_for_guild(record.guild_id)
            if durable is None or durable.id != record.id:
                self._failed_wake_recoveries.pop(record.guild_id, None)
                return False
            channel = self._bot.get_channel(record.channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                self._failed_wake_recoveries[record.guild_id] = record
                self._log.warning(
                    "resident_assistant_recovery_channel_deferred",
                    guild_id=record.guild_id,
                    session_id=record.id,
                    channel_id=record.channel_id,
                )
                return False
            permissions = channel.permissions_for(channel.guild.me)
            if not permissions.view_channel or not permissions.connect or not permissions.speak:
                self._failed_wake_recoveries[record.guild_id] = record
                self._log.warning(
                    "resident_assistant_recovery_permission_deferred",
                    guild_id=record.guild_id,
                    session_id=record.id,
                    channel_id=record.channel_id,
                )
                return False
            if len(self._sessions) >= self._settings.resident_assistant_max_sessions:
                self._failed_wake_recoveries[record.guild_id] = record
                self._log.warning(
                    "resident_assistant_recovery_capacity_deferred",
                    guild_id=record.guild_id,
                    session_id=record.id,
                )
                return False
            resources = _ResidentStartResources(
                session_id=record.id,
                guild_id=record.guild_id,
            )
            stage = "capacity_claim"
            try:
                resources.capacity_claim_attempted = self._capacity is not None
                resources.capacity_lease = (
                    await self._capacity.claim(
                        guild_id=record.guild_id,
                        mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
                        owner_id=record.id,
                    )
                    if self._capacity is not None
                    else None
                )
                stage = "voice_claim"
                resources.voice_claim_attempted = True
                resources.voice_lease = await self._voice_modes.claim(
                    guild_id=record.guild_id,
                    mode=GuildVoiceMode.RESIDENT_ASSISTANT,
                    owner_id=record.id,
                )
                voice_lease = resources.voice_lease
                assert voice_lease is not None
                stage = "connect"
                resources.runtime = await self._connect(
                    session_id=record.id,
                    channel=channel,
                    creator_user_id=record.creator_user_id,
                    lease=voice_lease,
                    capacity_lease=resources.capacity_lease,
                    recovered_record=record,
                    resources=resources,
                )
                runtime = resources.runtime
                assert runtime is not None
                self._sessions[record.guild_id] = runtime
                stage = "record_recovered"
                await self._repository.record_recovered(record.id)
                refreshed = await self._repository.active_for_guild(record.guild_id)
                if refreshed is not None:
                    runtime.record = refreshed
                self._failed_wake_recoveries.pop(record.guild_id, None)
                return True
            except ServiceCapacityFullError:
                self._failed_wake_recoveries[record.guild_id] = record
                self._log.warning(
                    "resident_assistant_recovery_capacity_deferred",
                    guild_id=record.guild_id,
                    session_id=record.id,
                )
                return False
            except VoiceModeConflictError as error:
                errors = await self._finish_failed_start_cleanup(
                    resources,
                    reason="RECOVERY_VOICE_CONFLICT",
                )
                self._log_start_cleanup_errors(resources, stage=stage, errors=errors)
                try:
                    await self._repository.stop_active(
                        session_id=record.id,
                        actor_user_id=None,
                        reason="RECOVERY_VOICE_CONFLICT",
                    )
                except BaseException as cleanup_error:
                    self._failed_wake_recoveries[record.guild_id] = record
                    self._log.error(
                        "resident_assistant_recovery_failed",
                        guild_id=record.guild_id,
                        session_id=record.id,
                        error_type=type(cleanup_error).__name__,
                    )
                    raise cleanup_error from error
                self._failed_wake_recoveries.pop(record.guild_id, None)
                return False
            except asyncio.CancelledError:
                errors = await self._finish_failed_start_cleanup(
                    resources,
                    reason="RECOVERY_CANCELLED",
                )
                self._log_start_cleanup_errors(resources, stage=stage, errors=errors)
                self._failed_wake_recoveries[record.guild_id] = record
                raise
            except Exception as exc:
                errors = await self._finish_failed_start_cleanup(
                    resources,
                    reason="RECOVERY_FAILED",
                )
                self._log_start_cleanup_errors(resources, stage=stage, errors=errors)
                self._failed_wake_recoveries[record.guild_id] = record
                self._log.error(
                    "resident_assistant_recovery_failed",
                    guild_id=record.guild_id,
                    session_id=record.id,
                    error_type=type(exc).__name__,
                )
                return False

    async def _release_capacity_owner(
        self,
        record: ResidentAssistantSessionRecord,
        *,
        reason: str,
    ) -> bool:
        if self._capacity is None:
            return False
        return await self._capacity.release_owner(
            guild_id=record.guild_id,
            mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
            owner_id=record.id,
            reason=reason,
        )

    async def close_all(self) -> None:
        """Drop runtime resources on shutdown but retain ACTIVE leases for recovery."""

        for guild_id in tuple(self._sessions):
            with contextlib.suppress(Exception):
                await self._stop(
                    guild_id=guild_id,
                    actor_user_id=None,
                    reason="PROCESS_SHUTDOWN",
                    persist=False,
                )
