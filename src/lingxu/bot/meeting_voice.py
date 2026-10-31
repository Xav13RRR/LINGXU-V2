"""Formal Meeting Discord voice lifecycle and process-prompt execution."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast

import discord
import structlog
from discord.ext import voice_recv

from lingxu.asr.factory import create_realtime_asr_provider, create_realtime_asr_service
from lingxu.asr.service import RealtimeASRService
from lingxu.bot.meeting_ui import PrivateVoteView
from lingxu.bot.mute_enforcement import MeetingMuteCoordinator
from lingxu.bot.voice_modes import (
    GuildVoiceLease,
    GuildVoiceMode,
    GuildVoiceModeRegistry,
)
from lingxu.config import Settings
from lingxu.domain.enums import ConsentKind, MeetingState, ServiceCapacityMode
from lingxu.domain.errors import LingxuError, MeetingVoiceStartupError
from lingxu.domain.models import (
    MeetingSessionRecord,
    ServiceCapacityLeaseRecord,
    VoteRecord,
)
from lingxu.meeting.components import VoteOutcome
from lingxu.meeting.factory import MeetingApplication
from lingxu.meeting.prompts import agenda_intro_prompt
from lingxu.meeting.speaking_flow import SpeakingFlowNotice, SpeakingFlowNoticeKind
from lingxu.privacy.consent_timeline import ConsentTimeline
from lingxu.storage.encryption import FileCipher
from lingxu.storage.guild_wake_phrase_repository import GuildWakePhraseRepository
from lingxu.storage.recording_service import (
    RecordingCoordinator,
    reconcile_recording_artifacts,
)
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    RealtimeUtteranceRepository,
)
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import TranscriptRepository
from lingxu.tts.discord_playback import DiscordVoicePlaybackTarget
from lingxu.tts.provider import TTSProvider
from lingxu.utils import new_ulid
from lingxu.voice.dave_adapter import inspect_dave_status
from lingxu.voice.meeting_consent import MeetingConsentRegistry
from lingxu.voice.segment_recorder import (
    EncryptedSegmentRecorder,
    FormalRecordingPipeline,
    MeetingRecordingQueue,
    RecordingMetrics,
)
from lingxu.voice.sink import MeetingPcmSink
from lingxu.voice_assistant.factory import create_meeting_voice_assistant
from lingxu.voice_assistant.service import MeetingVoiceAssistant

_PROCESS_PROMPTS = {
    "speak_welcome": "本次会议将于十秒后开始。",
    "select_next_speaker": "当前流程现已开始。",
    "speak_closing": "会议结束, 感谢参加。会议总结将自动发送至本语音频道的文字聊天。",
    "freeze_timers": "会议已暂停。",
    "resume_timers": "会议继续。",
}

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


def _requires_voice_recovery(session: MeetingSessionRecord) -> bool:
    return session.state in _VOICE_RECOVERY_STATES or (
        session.state is MeetingState.PAUSED and session.resume_state in _VOICE_RECOVERY_STATES
    )


def _spoken_duration(seconds: int | None) -> str:
    if seconds is None:
        return "未设置"
    minutes, remainder = divmod(seconds, 60)
    if minutes and remainder:
        return f"{minutes}分{remainder}秒"
    if minutes:
        return f"{minutes}分钟"
    return f"{remainder}秒"


def _spoken_names(names: tuple[str, ...]) -> str:
    if not names:
        return "暂无"
    return "、".join(names)


@dataclass(slots=True)
class _MeetingVoiceSession:
    session: MeetingSessionRecord
    channel: discord.VoiceChannel
    client: Any
    consents: MeetingConsentRegistry
    recording: RecordingCoordinator
    realtime_asr: RealtimeASRService
    assistant: MeetingVoiceAssistant | None
    sink: MeetingPcmSink
    started_monotonic_ns: int
    voice_lease: GuildVoiceLease
    playback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _MeetingVoiceStartResources:
    """Every independently releasable resource acquired during one start attempt."""

    session: MeetingSessionRecord
    capacity_lease: ServiceCapacityLeaseRecord | None = None
    voice_lease: GuildVoiceLease | None = None
    consents: MeetingConsentRegistry | None = None
    recording: RecordingCoordinator | None = None
    realtime_asr: RealtimeASRService | None = None
    assistant: MeetingVoiceAssistant | None = None
    client: Any | None = None
    capacity_claim_attempted: bool = False
    voice_claim_attempted: bool = False
    recording_start_attempted: bool = False
    asr_start_attempted: bool = False
    assistant_start_attempted: bool = False


class MeetingVoiceCoordinator:
    """Own one consent-aware formal voice client per Guild."""

    def __init__(
        self,
        *,
        bot: discord.Client,
        settings: Settings,
        application: MeetingApplication,
        tts_provider: TTSProvider,
        voice_modes: GuildVoiceModeRegistry | None = None,
        capacity: ServiceCapacityRepository | None = None,
        wake_phrases: GuildWakePhraseRepository | None = None,
    ) -> None:
        self._bot = bot
        self._settings = settings
        self._application = application
        self._tts_provider = tts_provider
        self._voice_modes = voice_modes or GuildVoiceModeRegistry()
        self._capacity = capacity
        self._wake_phrases = wake_phrases or GuildWakePhraseRepository(
            application.database.sessions
        )
        self._sessions: dict[int, _MeetingVoiceSession] = {}
        self._failed_wake_recoveries: dict[int, tuple[MeetingSessionRecord, int]] = {}
        self._mute = MeetingMuteCoordinator(bot=bot, application=application)
        self._locks: dict[int, asyncio.Lock] = {}
        self._log = structlog.get_logger(__name__)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        return self._locks.setdefault(guild_id, asyncio.Lock())

    def channel_id(self, guild_id: int) -> int | None:
        selected = self._sessions.get(guild_id)
        return selected.channel.id if selected is not None else None

    async def update_wake_phrase(self, *, guild_id: int, phrase: str) -> bool:
        """Apply a validated Guild phrase to an active formal Meeting, if any."""

        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if selected is None or selected.assistant is None:
                return False
            await selected.assistant.update_wake_phrase(phrase)
            return True

    def mark_failed_wake_phrase_recovery(
        self,
        *,
        session: MeetingSessionRecord,
        channel_id: int,
    ) -> None:
        """Remember one failed startup so a repaired Guild setting can retry it once."""

        self._failed_wake_recoveries[session.guild_id] = (session, channel_id)

    def _discard_failed_wake_phrase_recovery(
        self,
        *,
        guild_id: int,
        failed: tuple[MeetingSessionRecord, int],
    ) -> None:
        current = self._failed_wake_recoveries.get(guild_id)
        if current == failed:
            self._failed_wake_recoveries.pop(guild_id, None)

    async def retry_failed_wake_phrase_recovery(
        self,
        guild_id: int,
    ) -> MeetingSessionRecord | None:
        """Reconnect one retained failed recovery without completing its flow marker."""

        try:
            async with self._lock(guild_id):
                failed = self._failed_wake_recoveries.get(guild_id)
                if failed is None:
                    return None
                marked_session, channel_id = failed
                durable = await self._application.sessions.active_for_guild(guild_id)
                if (
                    durable is None
                    or durable.id != marked_session.id
                    or durable.guild_id != guild_id
                    or not _requires_voice_recovery(durable)
                ):
                    self._discard_failed_wake_phrase_recovery(
                        guild_id=guild_id,
                        failed=failed,
                    )
                    return None

                selected = self._sessions.get(guild_id)
                if selected is not None and selected.session.id != durable.id:
                    self._discard_failed_wake_phrase_recovery(
                        guild_id=guild_id,
                        failed=failed,
                    )
                    return None
                voice_owner = self._voice_modes.current(guild_id)
                if voice_owner is not None and (
                    voice_owner.mode is not GuildVoiceMode.MEETING
                    or voice_owner.owner_id != durable.id
                ):
                    return None
                capacity_owner = (
                    await self._capacity.current() if self._capacity is not None else None
                )
                if capacity_owner is not None and (
                    capacity_owner.guild_id != guild_id
                    or capacity_owner.mode is not ServiceCapacityMode.MEETING
                    or capacity_owner.owner_id != durable.id
                ):
                    return None
                if selected is not None and selected.client.is_connected():
                    return durable

                channel = self._bot.get_channel(channel_id)
                if not isinstance(channel, discord.VoiceChannel):
                    return None
                started = await self._start_locked(
                    session=durable,
                    channel=channel,
                    protected_owner_id=durable.id,
                )
                selected = self._sessions.get(guild_id)
                recovered = started or (
                    selected is not None
                    and selected.session.id == durable.id
                    and selected.client.is_connected()
                )
        except Exception as exc:
            marked = self._failed_wake_recoveries.get(guild_id)
            self._log.error(
                "meeting_wake_phrase_recovery_retry_failed",
                guild_id=guild_id,
                session_id=marked[0].id if marked is not None else None,
                error_type=type(exc).__name__,
            )
            return None
        return durable if recovered else None

    def complete_failed_wake_phrase_recovery(
        self,
        *,
        guild_id: int,
        session_id: str,
    ) -> None:
        """Clear a retry marker only after voice and automatic flow recovery both succeed."""

        current = self._failed_wake_recoveries.get(guild_id)
        if current is not None and current[0].id == session_id:
            self._failed_wake_recoveries.pop(guild_id, None)

    async def sync_mute_enforcement(self, session: MeetingSessionRecord) -> bool:
        """Reconcile one Session after any persisted floor or Runtime transition."""

        result = await self._mute.sync_session(session)
        return not result.paused

    async def reconcile_mutes(self) -> None:
        """Reconcile all retained leases before startup readiness."""

        await self._mute.reconcile_all()

    async def handle_member_voice_state(self, *, guild_id: int) -> None:
        session = await self._application.sessions.active_for_guild(guild_id)
        if session is not None:
            await self._mute.sync_session(session)

    async def _wait_for_dave(self, client: Any) -> None:
        deadline = asyncio.get_running_loop().time() + (
            self._settings.voice_duplex_dave_ready_timeout_seconds
        )
        while asyncio.get_running_loop().time() < deadline:
            if inspect_dave_status(client).ready:
                return
            if not client.is_connected():
                break
            await asyncio.sleep(0.1)
        raise MeetingVoiceStartupError("dave_ready")

    @staticmethod
    def _voice_start_diagnostics(client: Any | None) -> dict[str, Any]:
        """Return only bounded protocol booleans and opcode presence for startup logs."""

        if client is None:
            return {
                "voice_connected": False,
                "dave_status_available": False,
            }
        try:
            voice_connected = bool(client.is_connected())
        except Exception:
            voice_connected = False
        try:
            status = inspect_dave_status(client)
        except Exception:
            return {
                "voice_connected": voice_connected,
                "dave_status_available": False,
            }
        raw_opcodes = getattr(client, "_dave_ws_recent_ops", ())
        opcodes = (
            tuple(value for value in raw_opcodes if isinstance(value, int))[-256:]
            if isinstance(raw_opcodes, (list, tuple))
            else ()
        )
        return {
            "voice_connected": voice_connected,
            "dave_status_available": True,
            "dave_supported": status.supported_by_receive_fork,
            "dave_enabled": status.enabled,
            "dave_ready": status.ready,
            "dave_protocol_version": status.protocol_version,
            "dave_event_count": len(opcodes),
            "dave_prepare_epoch_seen": 24 in opcodes,
            "dave_commit_seen": 29 in opcodes,
            "dave_welcome_seen": 30 in opcodes,
        }

    @staticmethod
    def _raise_voice_startup_error(error: BaseException, stage: str) -> NoReturn:
        if isinstance(error, (asyncio.CancelledError, LingxuError, ValueError)):
            raise error
        raise MeetingVoiceStartupError(stage) from error

    async def _seed_consents(
        self,
        *,
        session_id: str,
        consents: MeetingConsentRegistry,
        timeline: ConsentTimeline,
    ) -> None:
        open_grants = {
            (interval.user_id, interval.kind)
            for interval in timeline.intervals
            if interval.ended_ms is None
        }
        for user_id in await self._application.participants.present_user_ids(session_id):
            participant_grants = await self._application.participants.consents_for(
                session_id=session_id,
                user_id=user_id,
            )
            for kind in ConsentKind:
                if kind in participant_grants and (user_id, kind) in open_grants:
                    consents.grant(user_id, kind)

    @staticmethod
    def _allows_recording_segment(
        timeline: ConsentTimeline,
        user_id: int,
        started_ms: int,
        ended_ms: int,
    ) -> bool:
        """Apply raw Segment's exclusive end bound without changing transcript semantics."""

        if started_ms < 0 or ended_ms < started_ms:
            return False
        return any(
            interval.user_id == user_id
            and interval.kind is ConsentKind.RECORDING
            and interval.started_ms <= started_ms
            and (interval.ended_ms is None or ended_ms <= interval.ended_ms)
            for interval in timeline.intervals
        )

    async def _recover_consent_timeline(self, session_id: str) -> ConsentTimeline:
        """Project durable consent and purge crash-window post-cutoff artifacts."""

        events = await MeetingEventStore(self._application.database.sessions).list(session_id)
        timeline = ConsentTimeline.from_events(events)
        files = SessionFiles(self._settings.data_root)
        segments = AudioSegmentRepository(self._application.database.sessions)
        utterances = RealtimeUtteranceRepository(self._application.database.sessions)
        await reconcile_recording_artifacts(
            session_id=session_id,
            segments=segments,
            files=files,
            allows_interval=lambda user_id, started_ms, ended_ms: self._allows_recording_segment(
                timeline,
                user_id,
                started_ms,
                ended_ms,
            ),
        )

        invalid_utterance_ids = [
            utterance.id
            for utterance in await utterances.list_session(session_id)
            if not timeline.allows_interval(
                utterance.user_id,
                ConsentKind.REALTIME_ASR,
                utterance.started_ms,
                utterance.ended_ms,
            )
        ]
        await utterances.delete_ids(
            session_id=session_id,
            utterance_ids=invalid_utterance_ids,
        )
        return timeline

    async def _session_timeline_watermark_ms(self, session_id: str) -> int:
        segments = await AudioSegmentRepository(self._application.database.sessions).list_session(
            session_id
        )
        utterances = await RealtimeUtteranceRepository(
            self._application.database.sessions
        ).list_session(session_id)
        transcripts = await TranscriptRepository(self._application.database.sessions).list(
            session_id
        )
        consent_event_watermarks = [
            int(event.payload["effective_at_ms"])
            for event in await MeetingEventStore(self._application.database.sessions).list(
                session_id
            )
            if event.event_type in {"CONSENT_GRANTED", "CONSENT_WITHDRAWN"}
        ]
        return max(
            [value.ended_ms for value in segments]
            + [value.ended_ms for value in utterances]
            + [value.ended_ms for value in transcripts]
            + consent_event_watermarks,
            default=-1,
        )

    async def _durable_timeline_watermark_ms(self, *, guild_id: int) -> int:
        session = await self._application.sessions.active_for_guild(guild_id)
        if session is None:
            return 0
        return max(0, await self._session_timeline_watermark_ms(session.id))

    async def timeline_now_ms(self, *, guild_id: int, user_id: int) -> int:
        """Return one Meeting-relative event cutoff without changing live state."""

        del user_id
        selected = self._sessions.get(guild_id)
        if selected is not None:
            return max(
                0,
                (time.monotonic_ns() - selected.started_monotonic_ns) // 1_000_000,
            )
        return await self._durable_timeline_watermark_ms(guild_id=guild_id)

    async def _build_capture(
        self,
        session: MeetingSessionRecord,
    ) -> tuple[
        MeetingConsentRegistry,
        RecordingCoordinator,
        RealtimeASRService,
        MeetingVoiceAssistant | None,
        MeetingPcmSink,
        int,
    ]:
        recovered_offset_ms = await self._session_timeline_watermark_ms(session.id) + 1
        started_ns = time.monotonic_ns() - recovered_offset_ms * 1_000_000
        consents = MeetingConsentRegistry()
        recording_queue = MeetingRecordingQueue(
            max_frames=self._settings.recording_audio_queue_max_frames
        )
        metrics = RecordingMetrics(session.id)
        files = SessionFiles(self._settings.data_root)
        recorder = EncryptedSegmentRecorder(
            session_id=session.id,
            session_started_monotonic_ns=started_ns,
            recording_queue=recording_queue,
            session_files=files,
            runtime_tmp=self._settings.runtime_tmp,
            cipher=FileCipher.from_secret(self._settings.file_encryption_key),
            metrics=metrics,
            silence_split_ms=self._settings.recording_silence_split_ms,
            silence_threshold=self._settings.recording_silence_threshold,
            max_segment_seconds=self._settings.recording_max_segment_seconds,
        )
        pipeline = FormalRecordingPipeline(
            session_id=session.id,
            consents=consents,
            recording_queue=recording_queue,
            recorder=recorder,
            metrics=metrics,
        )
        recording = RecordingCoordinator(
            session_id=session.id,
            pipeline=pipeline,
            segments=AudioSegmentRepository(self._application.database.sessions),
            files=files,
        )
        asr_provider = create_realtime_asr_provider(self._settings)
        realtime_asr = create_realtime_asr_service(
            settings=self._settings,
            session_id=session.id,
            session_started_monotonic_ns=started_ns,
            consents=consents,
            utterances=RealtimeUtteranceRepository(self._application.database.sessions),
            provider=asr_provider,
        )
        assistant = (
            create_meeting_voice_assistant(
                settings=self._settings,
                session_id=session.id,
                asr_provider=asr_provider,
                context_provider=lambda: self._assistant_context(session.guild_id),
                speak=lambda text: self._speak_assistant(session.guild_id, text),
                wake_phrase=(await self._wake_phrases.get(session.guild_id)).phrase,
            )
            if self._settings.meeting_assistant_enabled
            else None
        )
        return (
            consents,
            recording,
            realtime_asr,
            assistant,
            MeetingPcmSink(
                pipeline=pipeline,
                realtime_asr=realtime_asr,
                ephemeral_assistant=assistant,
            ),
            started_ns,
        )

    @staticmethod
    async def _stop_failed_start_input(resources: _MeetingVoiceStartResources) -> None:
        if resources.client is not None and resources.client.is_listening():
            resources.client.stop_listening()

    @staticmethod
    async def _stop_failed_start_assistant(resources: _MeetingVoiceStartResources) -> None:
        if resources.assistant_start_attempted and resources.assistant is not None:
            await resources.assistant.stop()

    @staticmethod
    async def _stop_failed_start_asr(resources: _MeetingVoiceStartResources) -> None:
        if resources.asr_start_attempted and resources.realtime_asr is not None:
            await resources.realtime_asr.stop()

    @staticmethod
    async def _stop_failed_start_recording(resources: _MeetingVoiceStartResources) -> None:
        if resources.recording_start_attempted and resources.recording is not None:
            await resources.recording.stop()

    @staticmethod
    async def _clear_failed_start_consents(resources: _MeetingVoiceStartResources) -> None:
        if resources.consents is not None:
            resources.consents.clear()

    @staticmethod
    async def _disconnect_failed_start_client(resources: _MeetingVoiceStartResources) -> None:
        if resources.client is not None:
            await resources.client.disconnect(force=True)

    async def _release_failed_start_voice(
        self,
        resources: _MeetingVoiceStartResources,
    ) -> None:
        if not resources.voice_claim_attempted:
            return
        voice_lease = resources.voice_lease
        if voice_lease is None:
            candidate = self._voice_modes.current(resources.session.guild_id)
            if (
                candidate is not None
                and candidate.mode is GuildVoiceMode.MEETING
                and candidate.owner_id == resources.session.id
            ):
                voice_lease = candidate
        if voice_lease is not None:
            await self._voice_modes.release(voice_lease)

    async def _release_failed_start_capacity(
        self,
        resources: _MeetingVoiceStartResources,
        *,
        reason: str,
    ) -> None:
        if not resources.capacity_claim_attempted or self._capacity is None:
            return
        if resources.capacity_lease is not None:
            await self._capacity.release(resources.capacity_lease, reason=reason)
            return
        await self._capacity.release_owner(
            guild_id=resources.session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=resources.session.id,
            reason=reason,
        )

    async def _rollback_failed_start(
        self,
        resources: _MeetingVoiceStartResources,
        *,
        reason: str,
    ) -> tuple[BaseException, ...]:
        """Attempt every exact-owner cleanup step even when an earlier one fails."""

        errors: list[BaseException] = []
        cleanup_steps = (
            self._stop_failed_start_input(resources),
            self._stop_failed_start_assistant(resources),
            self._stop_failed_start_asr(resources),
            self._stop_failed_start_recording(resources),
            self._clear_failed_start_consents(resources),
            self._disconnect_failed_start_client(resources),
            self._release_failed_start_voice(resources),
            self._release_failed_start_capacity(resources, reason=reason),
        )
        for cleanup_step in cleanup_steps:
            try:
                await cleanup_step
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)

    async def _finish_failed_start_cleanup(
        self,
        resources: _MeetingVoiceStartResources,
        *,
        reason: str,
    ) -> tuple[BaseException, ...]:
        """Finish rollback in an independent task despite repeated caller cancellation."""

        cleanup = asyncio.create_task(
            self._rollback_failed_start(resources, reason=reason),
            name=f"meeting-voice-start-cleanup-{resources.session.id}",
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
        resources: _MeetingVoiceStartResources,
        *,
        stage: str,
        errors: tuple[BaseException, ...],
    ) -> None:
        if not errors:
            return
        self._log.error(
            "meeting_voice_start_cleanup_incomplete",
            guild_id=resources.session.guild_id,
            session_id=resources.session.id,
            stage=stage,
            error_count=len(errors),
        )

    async def start(
        self,
        *,
        session: MeetingSessionRecord,
        channel: discord.VoiceChannel,
    ) -> bool:
        """Connect before Runtime start; return whether a new client was created."""

        if channel.guild.id != session.guild_id:
            raise ValueError("Meeting and voice channel Guild do not match")
        async with self._lock(session.guild_id):
            return await self._start_locked(session=session, channel=channel)

    async def _prepare_start_owner(
        self,
        *,
        session: MeetingSessionRecord,
        channel: discord.VoiceChannel,
        protected_owner_id: str | None,
    ) -> bool:
        existing = self._sessions.get(session.guild_id)
        if existing is not None:
            if (
                existing.session.id == session.id
                and existing.channel.id == channel.id
                and existing.client.is_connected()
            ):
                return False
            if protected_owner_id is not None and existing.session.id != protected_owner_id:
                return False
            await self._stop_locked(session.guild_id)
        guild_voice = cast(Any, channel.guild.voice_client)
        if guild_voice is not None and guild_voice.is_connected():
            raise ValueError("Bot 已在该服务器的其它语音任务中。请先结束常驻助手或旧会议连接。")
        return True

    async def _start_locked(
        self,
        *,
        session: MeetingSessionRecord,
        channel: discord.VoiceChannel,
        protected_owner_id: str | None = None,
    ) -> bool:
        """Start with the Guild lock held, optionally refusing to replace another owner."""

        if not await self._prepare_start_owner(
            session=session,
            channel=channel,
            protected_owner_id=protected_owner_id,
        ):
            return False

        resources = _MeetingVoiceStartResources(session=session)
        resources.capacity_claim_attempted = self._capacity is not None
        try:
            resources.capacity_lease = (
                await self._capacity.claim(
                    guild_id=session.guild_id,
                    mode=ServiceCapacityMode.MEETING,
                    owner_id=session.id,
                )
                if self._capacity is not None
                else None
            )
        except BaseException:
            errors = await self._finish_failed_start_cleanup(
                resources,
                reason="CAPACITY_CLAIM_FAILED",
            )
            self._log_start_cleanup_errors(
                resources,
                stage="capacity_claim",
                errors=errors,
            )
            raise

        resources.voice_claim_attempted = True
        try:
            resources.voice_lease = await self._voice_modes.claim(
                guild_id=session.guild_id,
                mode=GuildVoiceMode.MEETING,
                owner_id=session.id,
            )
        except BaseException:
            errors = await self._finish_failed_start_cleanup(
                resources,
                reason="VOICE_CLAIM_FAILED",
            )
            self._log_start_cleanup_errors(
                resources,
                stage="voice_claim",
                errors=errors,
            )
            raise

        try:
            timeline = await self._recover_consent_timeline(session.id)
            (
                resources.consents,
                resources.recording,
                resources.realtime_asr,
                resources.assistant,
                sink,
                started_ns,
            ) = await self._build_capture(session)
        except BaseException as error:
            errors = await self._finish_failed_start_cleanup(
                resources,
                reason="CAPTURE_BUILD_FAILED",
            )
            self._log_start_cleanup_errors(
                resources,
                stage="capture_build",
                errors=errors,
            )
            self._log.error(
                "meeting_voice_start_failed",
                stage="capture_build",
                error_type=type(error).__name__,
                voice_connected=False,
                dave_status_available=False,
            )
            self._raise_voice_startup_error(error, "capture_build")

        consents = resources.consents
        recording = resources.recording
        realtime_asr = resources.realtime_asr
        voice_lease = resources.voice_lease
        assert consents is not None
        assert recording is not None
        assert realtime_asr is not None
        assert voice_lease is not None
        stage = "consent_seed"
        try:
            await self._seed_consents(
                session_id=session.id,
                consents=consents,
                timeline=timeline,
            )
            stage = "realtime_asr_prepare"
            await realtime_asr.prepare()
            if resources.assistant is not None:
                stage = "meeting_assistant_prepare"
                await resources.assistant.prepare()
            stage = "voice_connect"
            resources.client = await channel.connect(
                cls=voice_recv.VoiceRecvClient,
                reconnect=True,
                self_deaf=False,
            )
            stage = "dave_ready"
            await self._wait_for_dave(resources.client)
            stage = "voice_runtime_start"
            resources.recording_start_attempted = True
            recording.start()
            resources.asr_start_attempted = True
            await realtime_asr.start()
            if resources.assistant is not None:
                resources.assistant_start_attempted = True
                await resources.assistant.start()
            resources.client.listen(
                sink,
                after=self._receive_after(session.guild_id, session.id),
            )
            self._sessions[session.guild_id] = _MeetingVoiceSession(
                session=session,
                channel=channel,
                client=resources.client,
                consents=consents,
                recording=recording,
                realtime_asr=realtime_asr,
                assistant=resources.assistant,
                sink=sink,
                started_monotonic_ns=started_ns,
                voice_lease=voice_lease,
            )
        except BaseException as error:
            diagnostics = self._voice_start_diagnostics(resources.client)
            errors = await self._finish_failed_start_cleanup(
                resources,
                reason="VOICE_START_FAILED",
            )
            self._log_start_cleanup_errors(resources, stage=stage, errors=errors)
            self._log.error(
                "meeting_voice_start_failed",
                stage=stage,
                error_type=type(error).__name__,
                **diagnostics,
            )
            self._raise_voice_startup_error(error, stage)
        self._log.info(
            "meeting_voice_connected",
            guild_id=session.guild_id,
            session_id=session.id,
            channel_id=channel.id,
        )
        return True

    def _receive_after(self, guild_id: int, session_id: str) -> Any:
        def after(error: BaseException | None) -> None:
            if error is not None:
                self._log.error(
                    "meeting_voice_receive_stopped",
                    guild_id=guild_id,
                    session_id=session_id,
                    error_type=type(error).__name__,
                )

        return after

    async def sync_user(self, *, guild_id: int, user_id: int) -> None:
        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if selected is None:
                return
            grants = await self._application.participants.consents_for(
                session_id=selected.session.id,
                user_id=user_id,
            )
            snapshot = selected.consents.snapshot()
            if ConsentKind.REALTIME_ASR in grants:
                selected.realtime_asr.grant(user_id)
            elif snapshot.allows(user_id, ConsentKind.REALTIME_ASR):
                selected.realtime_asr.withdraw(user_id)
            if ConsentKind.RECORDING in grants:
                selected.recording.pipeline.grant(user_id, ConsentKind.RECORDING)
            elif snapshot.allows(user_id, ConsentKind.RECORDING):
                await selected.recording.withdraw_recording(user_id)
            if ConsentKind.MINUTES in grants:
                selected.consents.grant(user_id, ConsentKind.MINUTES)
            else:
                selected.consents.withdraw(user_id, ConsentKind.MINUTES)

    async def apply_withdrawal(
        self,
        *,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
    ) -> None:
        """Convert one durable cutoff once and finish live cleanup before returning."""

        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if selected is None:
                session = await self._application.sessions.active_for_guild(guild_id)
                if session is not None:
                    await self._recover_consent_timeline(session.id)
                return
            cutoff_ns = selected.started_monotonic_ns + effective_at_ms * 1_000_000
            try:
                await selected.realtime_asr.withdraw_at(
                    user_id=user_id,
                    effective_monotonic_ns=cutoff_ns,
                )
                await selected.recording.withdraw_recording_at(user_id, cutoff_ns)
            except BaseException as exc:
                selected.consents.withdraw_all_at_monotonic_ns(user_id, cutoff_ns)
                self._log.error(
                    "meeting_consent_withdrawal_cleanup_failed",
                    guild_id=guild_id,
                    session_id=selected.session.id,
                    error_type=type(exc).__name__,
                )
                try:
                    await self._stop_locked(guild_id)
                except BaseException as stop_exc:
                    raise RuntimeError(
                        "formal Meeting capture could not be safely finalised"
                    ) from stop_exc
                raise

    async def suspend_user(self, *, guild_id: int, user_id: int) -> None:
        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if selected is None:
                return
            selected.realtime_asr.withdraw(user_id)
            await selected.recording.withdraw_recording(user_id)
            selected.consents.withdraw(user_id, ConsentKind.MINUTES)

    def _fallback_channel(
        self,
        session: MeetingSessionRecord,
    ) -> discord.abc.Messageable:
        guild = self._bot.get_guild(session.guild_id)
        if guild is None:
            raise ValueError("Meeting Guild is unavailable")
        channel = (
            guild.get_channel(session.publication_channel_id)
            if session.publication_channel_id is not None
            else None
        )
        if not isinstance(channel, discord.abc.Messageable):
            channel = guild.system_channel
        if channel is None:
            channel = next(
                (
                    candidate
                    for candidate in guild.text_channels
                    if candidate.permissions_for(guild.me).send_messages
                ),
                None,
            )
        if channel is None:
            raise ValueError("no authorized Meeting text fallback channel is available")
        return channel

    async def _assistant_context(self, guild_id: int) -> dict[str, Any]:
        """Expose bounded public Meeting status, never transcript or private audio."""

        status = await self._application.controller.status(guild_id)
        return {
            "meeting_state": status.session.state.value,
            "agenda": [
                {
                    "position": item.position,
                    "title": item.title[:160],
                    "status": item.status.value,
                    "duration_minutes": item.duration_minutes,
                }
                for item in status.agenda[:20]
            ],
            "speaker_active": status.speaking.current_user_id is not None,
            "speaking_queue_length": len(status.speaking.queued_user_ids),
            "vote_open": status.vote is not None,
            "vote_title": status.vote.title[:160] if status.vote is not None else None,
        }

    async def _speak_assistant(self, guild_id: int, text: str) -> None:
        result = await self.speak(guild_id=guild_id, text=text, max_duration_ms=15_000)
        if result != "voice":
            raise RuntimeError("Meeting assistant voice playback was unavailable")

    async def _wait_for_playback_slot(self, client: Any) -> None:
        deadline = asyncio.get_running_loop().time() + max(
            5.0,
            self._settings.tts_timeout_seconds,
        )
        while client.is_playing():
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("existing Meeting voice playback did not finish")
            await asyncio.sleep(0.02)

    async def _synthesize_bounded(
        self,
        text: str,
        max_duration_ms: int | None,
    ) -> tuple[str, Any]:
        spoken_text = text
        audio = await self._tts_provider.synthesize(spoken_text)
        if max_duration_ms is None or audio.duration_ms <= max_duration_ms:
            return spoken_text, audio
        ratio = max_duration_ms / max(1, audio.duration_ms)
        keep = max(8, int(len(spoken_text) * ratio * 0.85))
        spoken_text = spoken_text[:keep].rstrip(" \u3001\uff0c,\uff1b;\uff1a:")
        if spoken_text and spoken_text[-1] not in "。\uff01\uff1f.!?":
            spoken_text += "。"
        audio = await self._tts_provider.synthesize(spoken_text)
        if audio.duration_ms > max_duration_ms:
            spoken_text = "回答较长, 请缩小问题范围后再次使用唤醒词。"
            audio = await self._tts_provider.synthesize(spoken_text)
        if audio.duration_ms > max_duration_ms:
            raise RuntimeError("Meeting assistant TTS exceeds the hard duration limit")
        return spoken_text, audio

    async def speak(
        self,
        *,
        guild_id: int,
        text: str,
        tts_text: str | None = None,
        max_duration_ms: int | None = None,
    ) -> str:
        selected = self._sessions.get(guild_id)
        if selected is not None and selected.client.is_connected():
            assistant = selected.assistant
            if assistant is not None:
                assistant.begin_bot_output()
            try:
                async with selected.playback_lock:
                    try:
                        await self._wait_for_playback_slot(selected.client)
                        spoken_text, audio = await self._synthesize_bounded(
                            tts_text if tts_text is not None else text,
                            max_duration_ms,
                        )
                        playback_started_ns = time.monotonic_ns()
                        await DiscordVoicePlaybackTarget(
                            selected.client,
                            volume=self._settings.tts_playback_volume,
                        ).play(audio)
                        playback_ended_ns = time.monotonic_ns()
                        try:
                            await TranscriptRepository(self._application.database.sessions).upsert(
                                session_id=selected.session.id,
                                source_key=f"BOT_TTS:{new_ulid()}",
                                user_id=None,
                                raw_text=text if tts_text is not None else spoken_text,
                                normalised_text=text if tts_text is not None else spoken_text,
                                started_ms=max(
                                    0,
                                    (playback_started_ns - selected.started_monotonic_ns)
                                    // 1_000_000,
                                ),
                                ended_ms=max(
                                    0,
                                    (playback_ended_ns - selected.started_monotonic_ns)
                                    // 1_000_000,
                                ),
                            )
                        except Exception as exc:
                            self._log.error(
                                "meeting_tts_transcript_failed",
                                guild_id=guild_id,
                                session_id=selected.session.id,
                                error_type=type(exc).__name__,
                            )
                        return "voice"
                    except Exception as exc:
                        self._log.warning(
                            "meeting_tts_voice_failed",
                            guild_id=guild_id,
                            session_id=selected.session.id,
                            error_type=type(exc).__name__,
                        )
            finally:
                if assistant is not None:
                    assistant.end_bot_output()
        session = (
            selected.session
            if selected is not None
            else await self._application.controller.active(guild_id)
        )
        await self._fallback_channel(session).send(
            text,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return "text"

    async def handle_action(self, session: MeetingSessionRecord, action: str) -> None:
        if session.state is MeetingState.STOPPED and action == "speak_closing":
            async with self._lock(session.guild_id):
                selected = self._sessions.get(session.guild_id)
                if selected is None or selected.session.id != session.id:
                    self._log.warning(
                        "meeting_terminal_voice_action_owner_mismatch",
                        guild_id=session.guild_id,
                        session_id=session.id,
                        action=action,
                    )
                    return
                try:
                    await self.speak(
                        guild_id=session.guild_id,
                        text=_PROCESS_PROMPTS[action],
                    )
                except Exception as exc:
                    self._log.error(
                        "meeting_process_prompt_failed",
                        guild_id=session.guild_id,
                        session_id=session.id,
                        action=action,
                        error_type=type(exc).__name__,
                    )
            return
        if session.state is MeetingState.STOPPED and action == "finalise_capture":
            await self.stop(
                guild_id=session.guild_id,
                expected_session_id=session.id,
            )
            return
        prompt = _PROCESS_PROMPTS.get(action)
        if action == "speak_agenda":
            status = await self._application.controller.status(session.guild_id)
            total_minutes = sum(item.duration_minutes or 0 for item in status.agenda)
            prompt = (
                f"下面介绍本次会议议程, 共{len(status.agenda)}项"
                "\N{FULLWIDTH COMMA}"
                f"总流程约{total_minutes}分钟。"
            )
        elif action == "speak_current_agenda_item":
            status = await self._application.controller.status(session.guild_id)
            active_item = next(
                (item for item in status.agenda if item.status.value == "ACTIVE"),
                None,
            )
            prompt = agenda_intro_prompt(active_item) if active_item is not None else None
        elif action == "open_vote":
            status = await self._application.controller.status(session.guild_id)
            vote = status.vote
            if vote is not None and vote.agenda_item_id is not None:
                duration_seconds = max(
                    1,
                    int((vote.deadline_at - vote.opened_at).total_seconds()),
                )
                prompt = (
                    "接下来进入投票环节, "
                    f"投票内容为: {vote.title}。"
                    f"有效投票时长{duration_seconds}秒。"
                    "请通过私信中的按钮选择是或否。"
                )
        if prompt is not None:
            try:
                await self.speak(guild_id=session.guild_id, text=prompt)
            except Exception as exc:
                self._log.error(
                    "meeting_process_prompt_failed",
                    guild_id=session.guild_id,
                    session_id=session.id,
                    action=action,
                    error_type=type(exc).__name__,
                )
        if action == "open_vote":
            status = await self._application.controller.status(session.guild_id)
            if status.vote is not None and status.vote.agenda_item_id is not None:
                await self._dispatch_private_ballots(status.vote)
        if action == "finalise_capture":
            await self.stop(
                guild_id=session.guild_id,
                expected_session_id=session.id,
            )

    async def handle_speaking_flow_notice(
        self,
        session: MeetingSessionRecord,
        notice: SpeakingFlowNotice,
    ) -> None:
        """Deliver private preparation nudges and deterministic process TTS."""

        if notice.kind is SpeakingFlowNoticeKind.PREPARING:
            duration = _spoken_duration(notice.planned_seconds)
            for user_id in notice.current_user_ids:
                text = (
                    "聆序会议提醒: 请准备, "
                    f"您将在{notice.prep_seconds or 0}秒后开始"
                    f"{'演讲' if notice.mode.value == 'PRESENTATION' else '发言'}, "
                    f"计划时长{duration}。"
                )
                try:
                    user = self._bot.get_user(user_id)
                    if user is None:
                        user = await self._bot.fetch_user(user_id)
                    await user.send(text, allowed_mentions=discord.AllowedMentions.none())
                except (discord.Forbidden, discord.HTTPException):
                    try:
                        await self._fallback_channel(session).send(
                            f"<@{user_id}> {text}",
                            allowed_mentions=discord.AllowedMentions(
                                everyone=False,
                                users=True,
                                roles=False,
                                replied_user=False,
                            ),
                        )
                    except (discord.Forbidden, discord.HTTPException) as exc:
                        self._log.warning(
                            "meeting_speaking_prep_delivery_failed",
                            guild_id=session.guild_id,
                            session_id=session.id,
                            recipient_user_id=user_id,
                            error_type=type(exc).__name__,
                        )
            return

        prompt: str | None = None
        canonical_prompt: str | None = None
        try:
            aliases = await self._application.pronunciation_aliases.names_for(
                guild_id=session.guild_id,
                user_ids=notice.current_user_ids + notice.next_user_ids,
            )
        except Exception as exc:
            aliases = {}
            self._log.warning(
                "meeting_pronunciation_lookup_failed",
                guild_id=session.guild_id,
                session_id=session.id,
                error_type=type(exc).__name__,
            )
        current_names = tuple(
            aliases.get(user_id, display_name)
            for user_id, display_name in zip(
                notice.current_user_ids,
                notice.current_display_names,
                strict=True,
            )
        )
        next_names = tuple(
            aliases.get(user_id, display_name)
            for user_id, display_name in zip(
                notice.next_user_ids,
                notice.next_display_names,
                strict=True,
            )
        )
        current = _spoken_names(current_names)
        canonical_current = _spoken_names(notice.current_display_names)
        if notice.kind is SpeakingFlowNoticeKind.STARTED:
            duration = _spoken_duration(notice.planned_seconds)
            if notice.mode.value == "PRESENTATION":
                prompt = f"现在请{current}开始演讲, 演讲时长{duration}。"
                canonical_prompt = f"现在请{canonical_current}开始演讲, 演讲时长{duration}。"
            else:
                following = _spoken_names(next_names)
                canonical_following = _spoken_names(notice.next_display_names)
                prompt = f"现在请{current}发言, 发言时长{duration}, 下一位发言者为{following}。"
                canonical_prompt = (
                    f"现在请{canonical_current}发言, 发言时长{duration}, "
                    f"下一位发言者为{canonical_following}。"
                )
        elif notice.kind is SpeakingFlowNoticeKind.REMINDER:
            prompt = f"当前流程剩余{_spoken_duration(notice.remaining_seconds)}。"
        elif notice.kind is SpeakingFlowNoticeKind.EXTENDED:
            prompt = f"当前流程已延长{_spoken_duration(notice.additional_seconds)}。"
        elif notice.kind is SpeakingFlowNoticeKind.FINISHED:
            if notice.reason == "ABSENT":
                prompt = f"{current}当前不在语音频道, 本轮已跳过。"
                canonical_prompt = f"{canonical_current}当前不在语音频道, 本轮已跳过。"
            elif notice.reason == "HOST_SKIPPED":
                prompt = f"主持人已跳过{current}的本轮发言。"
                canonical_prompt = f"主持人已跳过{canonical_current}的本轮发言。"
            elif notice.reason == "HOST_ENDED_EARLY":
                prompt = "主持人已提前结束当前流程。"
            elif notice.reason == "PARTICIPANT_ABANDONED":
                prompt = f"{current}已放弃本轮发言。"
                canonical_prompt = f"{canonical_current}已放弃本轮发言。"
        elif notice.kind is SpeakingFlowNoticeKind.FOLLOWUP_STARTED:
            prompt = (
                "现在进入统一答疑环节。"
                if notice.followup_mode.value == "Q_AND_A"
                else "现在进入补充发言环节。"
            )
        if prompt is not None:
            try:
                selected_canonical = canonical_prompt or prompt
                await self.speak(
                    guild_id=session.guild_id,
                    text=selected_canonical,
                    tts_text=prompt if prompt != selected_canonical else None,
                )
            except Exception as exc:
                self._log.error(
                    "meeting_speaking_prompt_failed",
                    guild_id=session.guild_id,
                    session_id=session.id,
                    notice_kind=notice.kind.value,
                    error_type=type(exc).__name__,
                )

    async def _dispatch_private_ballots(self, vote: VoteRecord) -> None:
        """Send one native private ballot to each persisted eligible recipient."""

        delivered = 0
        failed = 0
        for recipient in await self._application.controller.vote_recipients(vote.id):
            if recipient.delivery_status != "PENDING":
                continue
            try:
                user = self._bot.get_user(recipient.user_id)
                if user is None:
                    user = await self._bot.fetch_user(recipient.user_id)
                message = await user.send(
                    "聆序会议投票\n"
                    f"投票内容: {vote.title}\n"
                    "请点击下方的“是”或“否”。截止前再次点击可修改选择; "
                    "若私信按钮不可用, 可在会议服务器使用 `/meeting vote cast`。",
                    view=PrivateVoteView(self._application.controller),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                await self._application.controller.mark_vote_delivery(
                    vote_id=vote.id,
                    user_id=recipient.user_id,
                    delivery_status="DELIVERED",
                    dm_message_id=message.id,
                )
                delivered += 1
            except (discord.Forbidden, discord.HTTPException):
                await self._application.controller.mark_vote_delivery(
                    vote_id=vote.id,
                    user_id=recipient.user_id,
                    delivery_status="FAILED",
                )
                failed += 1
        self._log.info(
            "meeting_private_ballots_dispatched",
            session_id=vote.session_id,
            delivered_count=delivered,
            failed_count=failed,
        )

    async def handle_vote_outcome(
        self,
        session: MeetingSessionRecord,
        outcome: VoteOutcome,
    ) -> None:
        try:
            await self.speak(
                guild_id=session.guild_id,
                text=outcome.broadcast_text(),
            )
        except Exception as exc:
            self._log.error(
                "meeting_vote_result_prompt_failed",
                guild_id=session.guild_id,
                session_id=session.id,
                error_type=type(exc).__name__,
            )
            raise

    async def _stop_locked(self, guild_id: int) -> None:
        selected = self._sessions.pop(guild_id, None)
        if selected is None:
            session = await self._application.sessions.active_for_guild(guild_id)
            if session is None:
                session = await self._application.sessions.latest_for_guild(guild_id)
            if session is not None:
                mute_result = await self._mute.release_session(
                    session,
                    reason="VOICE_ALREADY_DISCONNECTED",
                )
                if mute_result.paused:
                    raise RuntimeError(
                        "Discord mute cleanup is incomplete; retain the Session for recovery"
                    )
            return
        errors: list[BaseException] = []
        try:
            mute_result = await self._mute.release_session(
                selected.session,
                reason="VOICE_DISCONNECT",
            )
            if mute_result.paused:
                self._log.warning(
                    "meeting_mute_cleanup_incomplete",
                    guild_id=guild_id,
                    session_id=selected.session.id,
                )
                errors.append(
                    RuntimeError(
                        "Discord mute cleanup is incomplete; retain the Session for recovery"
                    )
                )
        except BaseException as exc:
            errors.append(exc)
        try:
            if selected.client.is_listening():
                selected.client.stop_listening()
        except BaseException as exc:
            errors.append(exc)
        if selected.assistant is not None:
            try:
                await selected.assistant.stop()
            except BaseException as exc:
                errors.append(exc)
        try:
            await selected.realtime_asr.stop()
        except BaseException as exc:
            errors.append(exc)
        try:
            await selected.recording.stop()
        except BaseException as exc:
            errors.append(exc)
        selected.consents.clear()
        try:
            await selected.client.disconnect(force=True)
        except BaseException as exc:
            errors.append(exc)
        await self._voice_modes.release(selected.voice_lease)
        self._log.info(
            "meeting_voice_disconnected",
            guild_id=guild_id,
            session_id=selected.session.id,
            error_count=len(errors),
        )
        if errors:
            raise RuntimeError("formal Meeting voice finalisation failed") from errors[0]

    async def stop(
        self,
        *,
        guild_id: int,
        expected_session_id: str | None = None,
    ) -> bool:
        async with self._lock(guild_id):
            selected = self._sessions.get(guild_id)
            if expected_session_id is not None and (
                selected is None or selected.session.id != expected_session_id
            ):
                self._log.warning(
                    "meeting_voice_stop_owner_mismatch",
                    guild_id=guild_id,
                    session_id=expected_session_id,
                )
                return False
            await self._stop_locked(guild_id)
            return True

    async def release_capacity(
        self,
        *,
        session: MeetingSessionRecord,
        reason: str,
    ) -> bool:
        """Release only the exact Meeting owner after terminal processing or failed start."""

        if self._capacity is None:
            return False
        return await self._capacity.release_owner(
            guild_id=session.guild_id,
            mode=ServiceCapacityMode.MEETING,
            owner_id=session.id,
            reason=reason,
        )

    async def claim_capacity(
        self,
        *,
        session: MeetingSessionRecord,
    ) -> None:
        """Reserve the deployment before requeueing an independent Worker job."""

        if self._capacity is not None:
            await self._capacity.claim(
                guild_id=session.guild_id,
                mode=ServiceCapacityMode.MEETING,
                owner_id=session.id,
            )

    async def close_all(self) -> None:
        for guild_id in tuple(self._sessions):
            with contextlib.suppress(Exception):
                await self.stop(guild_id=guild_id)
