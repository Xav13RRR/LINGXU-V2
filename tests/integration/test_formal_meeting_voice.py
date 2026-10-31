from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from lingxu.asr.provider import SenseVoiceProvider
from lingxu.bot.meeting_voice import MeetingVoiceCoordinator
from lingxu.config import Settings
from lingxu.domain.enums import (
    AgendaSpeakingMode,
    ConsentKind,
    MeetingState,
    ServiceCapacityMode,
)
from lingxu.domain.errors import MeetingVoiceStartupError
from lingxu.domain.models import AudioSegmentRecord, RealtimeUtteranceRecord
from lingxu.meeting.factory import create_meeting_application
from lingxu.meeting.speaking_flow import SpeakingFlowNotice, SpeakingFlowNoticeKind
from lingxu.privacy.consent_timeline import ConsentTimeline
from lingxu.storage.component_repositories import ConsentChangeRepository
from lingxu.storage.encryption import FileCipher
from lingxu.storage.repositories import (
    AudioSegmentRepository,
    MeetingEventStore,
    ProcessingJobRepository,
    RealtimeUtteranceRepository,
)
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.worker_repositories import TranscriptRepository
from lingxu.tts.models import TTSAudio
from lingxu.tts.provider import TTSProvider
from lingxu.voice.dave_adapter import DaveStatus
from lingxu.voice.meeting_consent import MeetingConsentRegistry


@pytest.fixture(autouse=True)
def _isolate_real_asr_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    async def prepared(_provider: SenseVoiceProvider) -> None:
        return None

    monkeypatch.setattr(SenseVoiceProvider, "prepare", prepared)


class FakeVoiceClient:
    def __init__(self) -> None:
        self.connected = True
        self.listening = False
        self.playing = False
        self.disconnect_count = 0
        self.stop_count = 0
        self.sink: Any = None

    def listen(self, sink: Any, *, after: Any) -> None:
        self.sink = sink
        self.listening = True

    def is_connected(self) -> bool:
        return self.connected

    def is_listening(self) -> bool:
        return self.listening

    def is_playing(self) -> bool:
        return self.playing

    def stop_listening(self) -> None:
        self.listening = False

    def stop(self) -> None:
        self.stop_count += 1
        self.playing = False

    async def disconnect(self, *, force: bool) -> None:
        self.disconnect_count += 1
        self.connected = False
        self.listening = False


class FakeVoiceChannel:
    def __init__(self, guild_id: int) -> None:
        self.id = 7_001
        self.guild = SimpleNamespace(id=guild_id, voice_client=None)
        self.client = FakeVoiceClient()
        self.connect_count = 0

    async def connect(self, **kwargs: Any) -> FakeVoiceClient:
        self.connect_count += 1
        self.guild.voice_client = self.client
        return self.client


class FakeBot:
    def get_guild(self, guild_id: int) -> None:
        return None


class FakeDmUser:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, **_kwargs: Any) -> None:
        self.messages.append(text)


class FakeDeliveryBot(FakeBot):
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.user = FakeDmUser()

    def get_user(self, user_id: int) -> FakeDmUser | None:
        return self.user if user_id == self.user_id else None


class FakeTTSProvider:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def synthesize(self, text: str) -> TTSAudio:
        return TTSAudio(
            cache_key="fake",
            path=self.path,
            sha256="0" * 64,
            duration_ms=10,
            sample_rate=24_000,
            channels=1,
            from_cache=True,
        )

    async def pre_generate(self, prompts: tuple[str, ...]) -> tuple[TTSAudio, ...]:
        return tuple(await self.synthesize(prompt) for prompt in prompts)


class _PreparedRecording:
    def start(self) -> None:
        raise AssertionError("recording must not start before runtime preparation passes")

    async def stop(self) -> None:
        return None


class _PreparedRealtimeASR:
    async def prepare(self) -> None:
        return None

    async def start(self) -> None:
        raise AssertionError("ASR tasks must not start before voice connection")

    async def stop(self) -> None:
        return None


class _FailingPreparedAssistant:
    async def prepare(self) -> None:
        raise RuntimeError("synthetic model preparation failure")

    async def start(self) -> None:
        raise AssertionError("assistant tasks must not start after failed preparation")

    async def stop(self) -> None:
        return None


class _RollbackRecording:
    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    async def stop(self) -> None:
        self.stop_count += 1


class _RollbackRealtimeASR:
    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    async def prepare(self) -> None:
        return None

    async def start(self) -> None:
        self.start_count += 1

    async def stop(self) -> None:
        self.stop_count += 1


class _RollbackAssistant:
    def __init__(
        self,
        *,
        start_entered: asyncio.Event | None = None,
        cleanup_entered: asyncio.Event | None = None,
        release_cleanup: asyncio.Event | None = None,
        start_error: BaseException | None = None,
    ) -> None:
        self._start_entered = start_entered
        self._cleanup_entered = cleanup_entered
        self._release_cleanup = release_cleanup
        self._start_error = start_error
        self.start_count = 0
        self.stop_count = 0

    async def prepare(self) -> None:
        return None

    async def start(self) -> None:
        self.start_count += 1
        if self._start_entered is not None:
            self._start_entered.set()
        if self._start_error is not None:
            raise self._start_error
        await asyncio.Future()

    async def stop(self) -> None:
        self.stop_count += 1
        if self._cleanup_entered is not None:
            self._cleanup_entered.set()
        if self._release_cleanup is not None:
            await self._release_cleanup.wait()


class _RollbackVoiceModes:
    def __init__(self, *, release_error: BaseException | None = None) -> None:
        self._release_error = release_error
        self.lease = SimpleNamespace(token="voice", guild_id=7_097)
        self.release_count = 0

    async def claim(self, **_kwargs: Any) -> Any:
        return self.lease

    async def release(self, _lease: Any) -> bool:
        self.release_count += 1
        if self._release_error is not None:
            raise self._release_error
        return True


class _RollbackCapacity:
    def __init__(self) -> None:
        self.lease = SimpleNamespace(id="capacity")
        self.release_count = 0

    async def claim(self, **_kwargs: Any) -> Any:
        return self.lease

    async def release(self, _lease: Any, *, reason: str) -> bool:
        assert reason in {
            "CAPACITY_CLAIM_FAILED",
            "VOICE_CLAIM_FAILED",
            "CAPTURE_BUILD_FAILED",
            "VOICE_START_FAILED",
        }
        self.release_count += 1
        return True


class _RollbackLog:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, event: str, **_kwargs: Any) -> None:
        self.errors.append(event)

    def info(self, _event: str, **_kwargs: Any) -> None:
        return None


def _ready_status(_: Any) -> DaveStatus:
    return DaveStatus(
        supported_by_receive_fork=True,
        enabled=True,
        ready=True,
        protocol_version=1,
        privacy_code_available=True,
    )


async def _advance_runtime_to_closing(application: Any, session_id: str) -> None:
    current = await application.runtime.start(
        session_id,
        actor_user_id=None,
        source="SYSTEM",
    )
    while current.session.state is not MeetingState.CLOSING:
        current = await application.runtime.advance(session_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_preparation_fails_before_voice_connect_with_expected_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_090
    host_id = 7_091
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'prepare.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    channel = FakeVoiceChannel(guild_id)
    capacity = ServiceCapacityRepository(application.database.sessions)
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
        capacity=capacity,
    )

    async def build_capture(_session: Any) -> tuple[Any, ...]:
        return (
            MeetingConsentRegistry(),
            _PreparedRecording(),
            _PreparedRealtimeASR(),
            _FailingPreparedAssistant(),
            object(),
            0,
        )

    monkeypatch.setattr(coordinator, "_build_capture", build_capture)

    with pytest.raises(MeetingVoiceStartupError) as raised:
        await coordinator.start(session=session, channel=cast(Any, channel))

    assert raised.value.stage == "meeting_assistant_prepare"
    assert channel.connect_count == 0
    assert await capacity.current() is None
    await application.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dave_timeout_disconnects_and_returns_expected_startup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_095
    host_id = 7_096
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'dave-timeout.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
        voice_duplex_dave_ready_timeout_seconds=0.01,
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    channel = FakeVoiceChannel(guild_id)
    capacity = ServiceCapacityRepository(application.database.sessions)
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
        capacity=capacity,
    )
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.inspect_dave_status",
        lambda _client: DaveStatus(
            supported_by_receive_fork=True,
            enabled=True,
            ready=False,
            protocol_version=1,
            privacy_code_available=True,
        ),
    )

    with pytest.raises(MeetingVoiceStartupError) as raised:
        await coordinator.start(session=session, channel=cast(Any, channel))

    assert raised.value.stage == "dave_ready"
    assert "RuntimeError" not in str(raised.value)
    assert not channel.client.is_connected()
    assert await capacity.current() is None
    await application.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_every_startup_rollback_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_097
    session = SimpleNamespace(id="meeting-double-cancel", guild_id=guild_id)
    channel = FakeVoiceChannel(guild_id)
    capacity = _RollbackCapacity()
    voice_modes = _RollbackVoiceModes()
    recording = _RollbackRecording()
    realtime_asr = _RollbackRealtimeASR()
    start_entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    assistant = _RollbackAssistant(
        start_entered=start_entered,
        cleanup_entered=cleanup_entered,
        release_cleanup=release_cleanup,
    )
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._capacity = capacity  # type: ignore[attr-defined]
    coordinator._voice_modes = voice_modes  # type: ignore[attr-defined]
    coordinator._log = _RollbackLog()  # type: ignore[attr-defined]

    async def recover_timeline(_session_id: str) -> object:
        return object()

    async def build_capture(_session: Any) -> tuple[Any, ...]:
        return (
            MeetingConsentRegistry(),
            recording,
            realtime_asr,
            assistant,
            object(),
            0,
        )

    async def seed_consents(**_kwargs: Any) -> None:
        return None

    async def dave_ready(_client: Any) -> None:
        return None

    monkeypatch.setattr(coordinator, "_recover_consent_timeline", recover_timeline)
    monkeypatch.setattr(coordinator, "_build_capture", build_capture)
    monkeypatch.setattr(coordinator, "_seed_consents", seed_consents)
    monkeypatch.setattr(coordinator, "_wait_for_dave", dave_ready)

    startup = asyncio.create_task(
        coordinator.start(session=cast(Any, session), channel=cast(Any, channel))
    )
    await asyncio.wait_for(start_entered.wait(), timeout=1)
    startup.cancel("initial-start-cancel")
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
    startup.cancel("repeated-cleanup-cancel")
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(startup, timeout=1)

    assert raised.value.args == ("initial-start-cancel",)
    assert assistant.stop_count == 1
    assert realtime_asr.stop_count == 1
    assert recording.stop_count == 1
    assert channel.client.disconnect_count == 1
    assert voice_modes.release_count == 1
    assert capacity.release_count == 1
    assert coordinator._sessions == {}  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_one_startup_cleanup_failure_does_not_mask_error_or_skip_later_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_097
    session = SimpleNamespace(id="meeting-cleanup-failure", guild_id=guild_id)
    channel = FakeVoiceChannel(guild_id)
    capacity = _RollbackCapacity()
    voice_release_error = OSError("synthetic voice release failure")
    voice_modes = _RollbackVoiceModes(release_error=voice_release_error)
    recording = _RollbackRecording()
    realtime_asr = _RollbackRealtimeASR()
    start_error = RuntimeError("synthetic assistant startup failure")
    assistant = _RollbackAssistant(start_error=start_error)
    coordinator = cast(MeetingVoiceCoordinator, object.__new__(MeetingVoiceCoordinator))
    coordinator._sessions = {}  # type: ignore[attr-defined]
    coordinator._locks = {}  # type: ignore[attr-defined]
    coordinator._capacity = capacity  # type: ignore[attr-defined]
    coordinator._voice_modes = voice_modes  # type: ignore[attr-defined]
    log = _RollbackLog()
    coordinator._log = log  # type: ignore[attr-defined]

    async def recover_timeline(_session_id: str) -> object:
        return object()

    async def build_capture(_session: Any) -> tuple[Any, ...]:
        return (
            MeetingConsentRegistry(),
            recording,
            realtime_asr,
            assistant,
            object(),
            0,
        )

    async def seed_consents(**_kwargs: Any) -> None:
        return None

    async def dave_ready(_client: Any) -> None:
        return None

    monkeypatch.setattr(coordinator, "_recover_consent_timeline", recover_timeline)
    monkeypatch.setattr(coordinator, "_build_capture", build_capture)
    monkeypatch.setattr(coordinator, "_seed_consents", seed_consents)
    monkeypatch.setattr(coordinator, "_wait_for_dave", dave_ready)

    with pytest.raises(MeetingVoiceStartupError) as raised:
        await coordinator.start(session=cast(Any, session), channel=cast(Any, channel))

    assert raised.value.stage == "voice_runtime_start"
    assert raised.value.__cause__ is start_error
    assert assistant.stop_count == 1
    assert realtime_asr.stop_count == 1
    assert recording.stop_count == 1
    assert channel.client.disconnect_count == 1
    assert voice_modes.release_count == 1
    assert capacity.release_count == 1
    assert coordinator._sessions == {}  # type: ignore[attr-defined]
    assert log.errors.count("meeting_voice_start_cleanup_incomplete") == 1
    assert log.errors.count("meeting_voice_start_failed") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agenda_overview_announces_total_scheduled_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_100
    host_id = 7_101
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'voice.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    for title, duration_minutes in (("议题一", 15), ("议题二", 20), ("开放讨论", None)):
        await application.controller.agenda_add(
            guild_id=guild_id,
            actor_user_id=host_id,
            title=title,
            duration_minutes=duration_minutes,
        )
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    spoken: list[str] = []

    async def capture_prompt(
        *,
        guild_id: int,
        text: str,
        max_duration_ms: int | None = None,
    ) -> str:
        assert guild_id == 7_100
        assert max_duration_ms is None
        spoken.append(text)
        return "voice"

    monkeypatch.setattr(coordinator, "speak", capture_prompt)

    await coordinator.handle_action(session, "speak_agenda")

    assert spoken == ["下面介绍本次会议议程, 共3项\N{FULLWIDTH COMMA}总流程约35分钟。"]
    await application.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terminal_voice_actions_cannot_touch_a_new_session_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_105
    host_id = 7_106
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'terminal-owner.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    old_session = await application.controller.create(
        guild_id=guild_id,
        host_user_id=host_id,
    )
    stopped = await application.controller.stop(
        guild_id=guild_id,
        actor_user_id=host_id,
    )
    assert stopped.session.id == old_session.id
    assert stopped.session.state is MeetingState.STOPPED
    new_session = await application.controller.create(
        guild_id=guild_id,
        host_user_id=host_id,
    )
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    new_client = FakeVoiceClient()
    new_client.listening = True
    new_recording = _RollbackRecording()
    new_asr = _RollbackRealtimeASR()
    selected = SimpleNamespace(
        session=new_session,
        client=new_client,
        assistant=None,
        realtime_asr=new_asr,
        recording=new_recording,
        consents=MeetingConsentRegistry(),
        voice_lease=object(),
    )
    coordinator._sessions[guild_id] = selected  # type: ignore[assignment]
    spoken: list[str] = []
    mute_releases: list[str] = []
    voice_releases: list[object] = []

    async def capture_prompt(**kwargs: Any) -> str:
        spoken.append(str(kwargs["text"]))
        return "voice"

    async def release_mute(session: Any, **_kwargs: Any) -> Any:
        mute_releases.append(session.id)
        return SimpleNamespace(paused=False)

    async def release_voice(lease: object) -> bool:
        voice_releases.append(lease)
        return True

    monkeypatch.setattr(coordinator, "speak", capture_prompt)
    monkeypatch.setattr(
        coordinator,
        "_mute",
        SimpleNamespace(release_session=release_mute),
    )
    monkeypatch.setattr(
        coordinator,
        "_voice_modes",
        SimpleNamespace(release=release_voice),
    )

    await coordinator.handle_action(stopped.session, "speak_closing")
    await coordinator.handle_action(stopped.session, "finalise_capture")

    assert spoken == []
    assert coordinator._sessions[guild_id] is selected
    assert coordinator._sessions[guild_id].session.id == new_session.id
    assert new_client.is_connected()
    assert new_client.is_listening()
    assert new_client.disconnect_count == 0
    assert new_client.stop_count == 0
    assert new_recording.stop_count == 0
    assert new_asr.stop_count == 0
    assert mute_releases == []
    assert voice_releases == []
    assert (await application.sessions.get(new_session.id)).state is new_session.state
    await application.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_controlled_speaking_delivers_private_prep_and_exact_start_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 7_110
    host_id = 7_111
    speaker_id = 7_112
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'speaking.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    bot = FakeDeliveryBot(speaker_id)
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, bot),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    spoken: list[str] = []
    canonical: list[str] = []

    async def capture_prompt(
        *,
        guild_id: int,
        text: str,
        tts_text: str | None = None,
        max_duration_ms: int | None = None,
    ) -> str:
        assert guild_id == 7_110
        assert max_duration_ms is None
        canonical.append(text)
        spoken.append(tts_text or text)
        return "voice"

    monkeypatch.setattr(coordinator, "speak", capture_prompt)
    await application.pronunciation_aliases.set(
        guild_id=guild_id,
        user_id=speaker_id,
        alias="艾丽丝",
    )
    await application.pronunciation_aliases.set(
        guild_id=guild_id,
        user_id=7_113,
        alias="鲍勃",
    )
    preparation = SpeakingFlowNotice(
        kind=SpeakingFlowNoticeKind.PREPARING,
        agenda_item_id="agenda-1",
        mode=AgendaSpeakingMode.ORDERED,
        current_user_ids=(speaker_id,),
        current_display_names=("Alice",),
        next_user_ids=(7_113,),
        next_display_names=("Bob",),
        planned_seconds=90,
        prep_seconds=10,
    )
    await coordinator.handle_speaking_flow_notice(session, preparation)
    await coordinator.handle_speaking_flow_notice(
        session,
        replace(preparation, kind=SpeakingFlowNoticeKind.STARTED),
    )

    assert bot.user.messages == ["聆序会议提醒: 请准备, 您将在10秒后开始发言, 计划时长1分30秒。"]
    assert spoken == ["现在请艾丽丝发言, 发言时长1分30秒, 下一位发言者为鲍勃。"]
    assert canonical == ["现在请Alice发言, 发言时长1分30秒, 下一位发言者为Bob。"]
    assert preparation.current_display_names == ("Alice",)
    assert preparation.next_display_names == ("Bob",)
    await application.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_formal_meeting_voice_connects_syncs_consent_and_finalises_before_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 700
    host_id = 7_001
    waiting_user_id = 7_002
    stale_user_id = 7_003
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'voice.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
        voice_duplex_dave_ready_timeout_seconds=0.1,
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(
        guild_id=guild_id,
        host_user_id=host_id,
    )
    await application.controller.participant_join(
        guild_id=guild_id,
        user_id=stale_user_id,
    )
    await application.controller.sync_participants(
        guild_id=guild_id,
        user_ids={host_id, waiting_user_id},
    )
    assert set(await application.participants.present_user_ids(session.id)) == {
        host_id,
        waiting_user_id,
    }

    channel = FakeVoiceChannel(guild_id)
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.inspect_dave_status",
        _ready_status,
    )

    async def complete_playback(_target: Any, _audio: TTSAudio) -> None:
        return None

    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.DiscordVoicePlaybackTarget.play",
        complete_playback,
    )
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
        capacity=ServiceCapacityRepository(application.database.sessions),
    )
    jobs = ProcessingJobRepository(application.database.sessions)
    finalise_saw_no_job = False

    async def handle_action(session_record: Any, action: str) -> None:
        nonlocal finalise_saw_no_job
        if action == "finalise_capture":
            finalise_saw_no_job = (
                await jobs.claim(worker_id="pre-finalise", lease_seconds=60) is None
            )
        await coordinator.handle_action(session_record, action)

    application.controller.set_runtime_action_handler(handle_action)
    application.controller.set_consent_runtime(coordinator)

    assert await coordinator.start(
        session=session,
        channel=cast(Any, channel),
    )
    assert not await coordinator.start(
        session=session,
        channel=cast(Any, channel),
    )
    assert channel.connect_count == 1
    assert channel.client.is_listening()
    assert coordinator.channel_id(guild_id) == channel.id

    started = await application.controller.start(
        guild_id=guild_id,
        actor_user_id=host_id,
    )
    assert started.session.state is MeetingState.CONSENT_CHECK
    selected = coordinator._sessions[guild_id]
    consent_at_ms = 1_234
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.time.monotonic_ns",
        lambda: selected.started_monotonic_ns + consent_at_ms * 1_000_000,
    )
    assert (
        await application.controller.consent_all(
            guild_id=guild_id,
            user_id=host_id,
        )
        is MeetingState.CONSENT_CHECK
    )
    consent_events = await MeetingEventStore(application.database.sessions).list(session.id)
    granted_event = consent_events[-1]
    consent_fact_count = sum(
        event.event_type in {"CONSENT_GRANTED", "CONSENT_WITHDRAWN"} for event in consent_events
    )
    for kind in ConsentKind:
        assert selected.consents.snapshot().allows(host_id, kind)
    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.time.monotonic_ns",
        lambda: selected.started_monotonic_ns + 1_500_000_000,
    )
    await application.controller.consent_all(guild_id=guild_id, user_id=host_id)
    events_after_repeated_consent = await MeetingEventStore(application.database.sessions).list(
        session.id
    )

    withdrawal_at_ms = 2_345
    cutoff_ns = selected.started_monotonic_ns + withdrawal_at_ms * 1_000_000
    voice_frame = b"\x00\x20\x00\x20" * 960
    silence_frame = b"\x00\x00\x00\x00" * 960
    assert selected.recording.pipeline.receive(
        user_id=host_id,
        received_monotonic_ns=cutoff_ns - 820_000_000,
        pcm_s16le_48k_stereo=voice_frame,
    )
    for frame_index in range(1, 41):
        assert selected.recording.pipeline.receive(
            user_id=host_id,
            received_monotonic_ns=cutoff_ns - 820_000_000 + frame_index * 20_000_000,
            pcm_s16le_48k_stereo=silence_frame,
        )
    assert selected.recording.pipeline.receive(
        user_id=host_id,
        received_monotonic_ns=cutoff_ns,
        pcm_s16le_48k_stereo=voice_frame,
    )
    for frame_index in range(1, 41):
        assert selected.recording.pipeline.receive(
            user_id=host_id,
            received_monotonic_ns=cutoff_ns + frame_index * 20_000_000,
            pcm_s16le_48k_stereo=silence_frame,
        )
    for _ in range(200):
        completed = selected.recording.pipeline.metrics.snapshot()["counters"].get(
            "segments_completed",
            0,
        )
        if completed == 2:
            break
        await asyncio.sleep(0.01)
    assert completed == 2

    monkeypatch.setattr(
        "lingxu.bot.meeting_voice.time.monotonic_ns",
        lambda: cutoff_ns,
    )
    original_apply_withdrawal = coordinator.apply_withdrawal
    original_stop_locked = coordinator._stop_locked
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    stop_capture_entered = asyncio.Event()

    async def delayed_apply_withdrawal(**kwargs: int) -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        await original_apply_withdrawal(**kwargs)

    async def observed_stop_locked(selected_guild_id: int) -> None:
        stop_capture_entered.set()
        await original_stop_locked(selected_guild_id)

    monkeypatch.setattr(coordinator, "apply_withdrawal", delayed_apply_withdrawal)
    monkeypatch.setattr(coordinator, "_stop_locked", observed_stop_locked)
    withdrawal_task = asyncio.create_task(
        application.controller.withdraw_all(guild_id=guild_id, user_id=host_id)
    )
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
    stop_task = asyncio.create_task(
        application.controller.stop(guild_id=guild_id, actor_user_id=host_id)
    )
    try:
        await asyncio.wait_for(stop_capture_entered.wait(), timeout=0.2)
        stop_entered_before_cleanup = True
    except TimeoutError:
        stop_entered_before_cleanup = False
    segments_before_cleanup = await AudioSegmentRepository(
        application.database.sessions
    ).list_session(session.id)
    premature_job = await jobs.claim(worker_id="before-cleanup", lease_seconds=60)
    release_cleanup.set()
    await withdrawal_task
    stopped = await stop_task

    consent_events = await MeetingEventStore(application.database.sessions).list(session.id)
    withdrawn_event = next(
        event for event in reversed(consent_events) if event.event_type == "CONSENT_WITHDRAWN"
    )
    segments_after_withdrawal = await AudioSegmentRepository(
        application.database.sessions
    ).list_session(session.id)
    files_after_withdrawal = tuple(
        (settings.data_root / session.id / "audio" / str(host_id)).glob("*.flac.enc")
    )
    assert stopped.session.state is MeetingState.STOPPED
    assert finalise_saw_no_job
    assert not channel.client.is_connected()
    assert coordinator.channel_id(guild_id) is None
    capacity = await coordinator._capacity.current()  # type: ignore[union-attr]
    assert capacity is not None
    assert capacity.mode is ServiceCapacityMode.MEETING
    assert capacity.owner_id == session.id
    job = premature_job or await jobs.claim(worker_id="after-finalise", lease_seconds=60)
    assert job is not None
    assert job.session_id == session.id
    bot_transcript = await TranscriptRepository(application.database.sessions).list(session.id)
    assert any(entry.source_key.startswith("BOT_TTS:") for entry in bot_transcript)
    assert any("会议总结将自动发送" in entry.raw_text for entry in bot_transcript)
    assert channel.client.stop_count == 0
    assert await coordinator.release_capacity(session=stopped.session, reason="TEST_PUBLISHED")
    assert await coordinator._capacity.current() is None  # type: ignore[union-attr]
    await application.database.close()
    assert granted_event.event_type == "CONSENT_GRANTED"
    assert granted_event.payload["effective_at_ms"] == consent_at_ms
    assert withdrawn_event.event_type == "CONSENT_WITHDRAWN"
    assert withdrawn_event.payload["effective_at_ms"] == withdrawal_at_ms
    assert (
        sum(
            event.event_type in {"CONSENT_GRANTED", "CONSENT_WITHDRAWN"}
            for event in events_after_repeated_consent
        )
        == consent_fact_count
    )
    assert not stop_entered_before_cleanup
    assert segments_before_cleanup == ()
    assert premature_job is None
    ConsentTimeline.from_events(consent_events)
    assert len(segments_after_withdrawal) == 1
    assert segments_after_withdrawal[0].ended_ms == withdrawal_at_ms
    assert len(files_after_withdrawal) == 1
    manifest_values = [
        json.loads(line)
        for line in (settings.data_root / session.id / "audio" / str(host_id) / "segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [value["id"] for value in manifest_values] == [segments_after_withdrawal[0].id]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_durable_withdrawal_prevents_restart_from_seeding_stale_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 701
    host_id = 7_011
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'restart.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
        voice_duplex_dave_ready_timeout_seconds=0.1,
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(
        guild_id=guild_id,
        host_user_id=host_id,
    )
    changes = ConsentChangeRepository(application.database.sessions)
    await changes.grant_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=1_000,
        trigger="test",
    )
    await changes.withdraw_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=2_345,
        trigger="test",
    )
    # Simulate a stale materialised participant snapshot after a crash. Events stay authoritative.
    await application.participants.set_consents(
        session_id=session.id,
        user_id=host_id,
        grants=frozenset(ConsentKind),
    )
    files = SessionFiles(settings.data_root)
    segment_repository = AudioSegmentRepository(application.database.sessions)
    utterance_repository = RealtimeUtteranceRepository(application.database.sessions)
    invalid_path = Path("audio") / str(host_id) / "000001.flac.enc"
    manifest_only_path = Path("audio") / str(host_id) / "000002.flac.enc"
    exact_boundary_path = Path("audio") / str(host_id) / "000003.flac.enc"
    missing_file_path = Path("audio") / str(host_id) / "000004.flac.enc"
    files.atomic_write(session.id, invalid_path, b"encrypted-invalid")
    files.atomic_write(session.id, manifest_only_path, b"encrypted-manifest-only")
    exact_boundary_payload = b"encrypted-exact-boundary"
    exact_boundary_digest = hashlib.sha256(exact_boundary_payload).hexdigest()
    files.atomic_write(session.id, exact_boundary_path, exact_boundary_payload)
    substituted_path = Path("audio") / str(host_id) / "000005.flac.enc"
    expected_payload = b"ciphertext-good"
    substituted_payload = b"ciphertext-evil"
    assert len(expected_payload) == len(substituted_payload)
    substituted_digest = hashlib.sha256(expected_payload).hexdigest()
    files.atomic_write(session.id, substituted_path, substituted_payload)
    for segment_number, path, digest, started_ms, ended_ms in (
        (1, invalid_path, "1" * 64, 0, 20),
        (2, manifest_only_path, "2" * 64, 1_200, 1_300),
        (3, exact_boundary_path, exact_boundary_digest, 1_200, 2_345),
        (4, missing_file_path, "4" * 64, 1_300, 1_400),
        (5, substituted_path, substituted_digest, 1_400, 1_500),
    ):
        files.append_jsonl(
            session.id,
            Path("audio") / str(host_id) / "segments.jsonl",
            {
                "id": f"restart-segment-{segment_number}",
                "segment_number": segment_number,
                "path": path.as_posix(),
                "sha256": digest,
                "started_ms": started_ms,
                "ended_ms": ended_ms,
                "bytes_encrypted": (
                    files.resolve(session.id, path).stat().st_size
                    if files.resolve(session.id, path).exists()
                    else len(b"encrypted-4")
                ),
            },
        )
    await segment_repository.add(
        AudioSegmentRecord(
            id="restart-segment-1",
            session_id=session.id,
            user_id=host_id,
            segment_number=1,
            encrypted_path=invalid_path,
            sha256="1" * 64,
            started_ms=0,
            ended_ms=20,
            bytes_encrypted=17,
        )
    )
    await segment_repository.add(
        AudioSegmentRecord(
            id="restart-segment-3",
            session_id=session.id,
            user_id=host_id,
            segment_number=3,
            encrypted_path=exact_boundary_path,
            sha256=exact_boundary_digest,
            started_ms=1_200,
            ended_ms=2_345,
            bytes_encrypted=len(b"encrypted-exact-boundary"),
        )
    )
    await segment_repository.add(
        AudioSegmentRecord(
            id="restart-segment-4",
            session_id=session.id,
            user_id=host_id,
            segment_number=4,
            encrypted_path=missing_file_path,
            sha256="4" * 64,
            started_ms=1_300,
            ended_ms=1_400,
            bytes_encrypted=len(b"encrypted-4"),
        )
    )
    await segment_repository.add(
        AudioSegmentRecord(
            id="restart-segment-5",
            session_id=session.id,
            user_id=host_id,
            segment_number=5,
            encrypted_path=substituted_path,
            sha256=substituted_digest,
            started_ms=1_400,
            ended_ms=1_500,
            bytes_encrypted=len(substituted_payload),
        )
    )
    await utterance_repository.add(
        RealtimeUtteranceRecord(
            id="restart-utterance-1",
            session_id=session.id,
            user_id=host_id,
            started_ms=2_300,
            ended_ms=2_345,
            raw_text="synthetic post-cutoff utterance",
            normalised_text="synthetic post-cutoff utterance",
        )
    )

    other_session = await application.controller.create(
        guild_id=guild_id + 1,
        host_user_id=host_id,
    )
    other_path = Path("audio") / str(host_id) / "000001.flac.enc"
    files.atomic_write(other_session.id, other_path, b"encrypted-other-guild")
    await segment_repository.add(
        AudioSegmentRecord(
            id="other-guild-segment-1",
            session_id=other_session.id,
            user_id=host_id,
            segment_number=1,
            encrypted_path=other_path,
            sha256="3" * 64,
            started_ms=0,
            ended_ms=20,
            bytes_encrypted=21,
        )
    )

    channel = FakeVoiceChannel(guild_id)
    monkeypatch.setattr("lingxu.bot.meeting_voice.inspect_dave_status", _ready_status)
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )

    assert await coordinator.start(session=session, channel=cast(Any, channel))
    selected = coordinator._sessions[guild_id]
    restart_snapshot = selected.consents.snapshot()
    recovered_segments = await segment_repository.list_user(
        session_id=session.id,
        user_id=host_id,
    )
    recovered_utterances = await utterance_repository.list_user(
        session_id=session.id,
        user_id=host_id,
    )
    other_guild_segments = await segment_repository.list_user(
        session_id=other_session.id,
        user_id=host_id,
    )

    await coordinator.stop(guild_id=guild_id)
    await application.database.close()
    for kind in ConsentKind:
        assert not restart_snapshot.allows(host_id, kind)
    assert [segment.id for segment in recovered_segments] == ["restart-segment-3"]
    assert recovered_utterances == ()
    assert not files.resolve(session.id, invalid_path).exists()
    assert not files.resolve(session.id, manifest_only_path).exists()
    assert files.resolve(session.id, exact_boundary_path).is_file()
    assert not files.resolve(session.id, missing_file_path).exists()
    assert not files.resolve(session.id, substituted_path).exists()
    manifest_values = [
        json.loads(line)
        for line in (files.directory(session.id) / "audio" / str(host_id) / "segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [value["id"] for value in manifest_values] == ["restart-segment-3"]
    assert len(other_guild_segments) == 1
    assert files.resolve(other_session.id, other_path).is_file()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_restart_timeline_origin_includes_nonzero_silent_withdrawal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 702
    host_id = 7_021
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'origin.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
        voice_duplex_dave_ready_timeout_seconds=0.1,
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    changes = ConsentChangeRepository(application.database.sessions)
    await changes.grant_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=1_000,
        trigger="test",
    )
    await changes.withdraw_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=2_345,
        trigger="test",
    )

    monkeypatch.setattr("lingxu.bot.meeting_voice.inspect_dave_status", _ready_status)
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    application.controller.set_consent_runtime(coordinator)
    assert await coordinator.start(
        session=session,
        channel=cast(Any, FakeVoiceChannel(guild_id)),
    )

    await application.controller.consent_all(guild_id=guild_id, user_id=host_id)
    events = await MeetingEventStore(application.database.sessions).list(session.id)
    regrant = next(event for event in reversed(events) if event.event_type == "CONSENT_GRANTED")

    await coordinator.stop(guild_id=guild_id)
    await application.database.close()
    assert int(regrant.payload["effective_at_ms"]) > 2_345
    ConsentTimeline.from_events(events)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_automatic_recovery_waits_for_withdrawal_cleanup_before_job_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 703
    host_id = 7_031
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'automatic-race.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    changes = ConsentChangeRepository(application.database.sessions)
    await changes.grant_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=1_000,
        trigger="test",
    )
    await _advance_runtime_to_closing(application, session.id)

    files = SessionFiles(settings.data_root)
    segments = AudioSegmentRepository(application.database.sessions)
    encrypted_path = Path("audio") / str(host_id) / "000001.flac.enc"
    encrypted_payload = b"post-cutoff-ciphertext"
    encrypted_digest = hashlib.sha256(encrypted_payload).hexdigest()
    files.atomic_write(session.id, encrypted_path, encrypted_payload)
    files.append_jsonl(
        session.id,
        Path("audio") / str(host_id) / "segments.jsonl",
        {
            "id": "automatic-race-segment",
            "segment_number": 1,
            "path": encrypted_path.as_posix(),
            "sha256": encrypted_digest,
            "started_ms": 1_990,
            "ended_ms": 2_020,
            "bytes_encrypted": len(encrypted_payload),
        },
    )
    await segments.add(
        AudioSegmentRecord(
            id="automatic-race-segment",
            session_id=session.id,
            user_id=host_id,
            segment_number=1,
            encrypted_path=encrypted_path,
            sha256=encrypted_digest,
            started_ms=1_990,
            ended_ms=2_020,
            bytes_encrypted=len(encrypted_payload),
        )
    )

    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    application.controller.set_consent_runtime(coordinator)
    finalise_entered = asyncio.Event()

    async def handle_action(_session: Any, action: str) -> None:
        if action == "finalise_capture":
            finalise_entered.set()

    application.controller.set_runtime_action_handler(handle_action)

    async def fixed_timeline(**_kwargs: int) -> int:
        return 2_000

    original_apply_withdrawal = coordinator.apply_withdrawal
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def delayed_apply_withdrawal(**kwargs: int) -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        await original_apply_withdrawal(**kwargs)

    monkeypatch.setattr(coordinator, "timeline_now_ms", fixed_timeline)
    monkeypatch.setattr(coordinator, "apply_withdrawal", delayed_apply_withdrawal)
    withdrawal_task = asyncio.create_task(
        application.controller.withdraw_all(guild_id=guild_id, user_id=host_id)
    )
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
    automatic_task = asyncio.create_task(
        application.controller.recover_automatic_flows(session_ids={session.id})
    )
    await asyncio.sleep(0.1)
    automatic_completed_before_cleanup = automatic_task.done()
    finalise_entered_before_cleanup = finalise_entered.is_set()
    artifacts_before_cleanup = await segments.list_session(session.id)
    jobs = ProcessingJobRepository(application.database.sessions)
    premature_job = await jobs.latest_for_session(session.id, job_type="POST_MEETING")

    release_cleanup.set()
    await withdrawal_task
    await automatic_task
    artifacts_after_cleanup = await segments.list_session(session.id)
    visible_job = await jobs.latest_for_session(session.id, job_type="POST_MEETING")
    await application.database.close()

    assert not automatic_completed_before_cleanup
    assert not finalise_entered_before_cleanup
    assert [value.id for value in artifacts_before_cleanup] == ["automatic-race-segment"]
    assert premature_job is None
    assert artifacts_after_cleanup == ()
    assert not files.resolve(session.id, encrypted_path).exists()
    assert visible_job is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_waits_for_inflight_withdrawal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 704
    host_id = 7_041
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'delete-race.db').as_posix()}",
        data_root=tmp_path / "sessions",
        runtime_tmp=tmp_path / "runtime",
        model_cache_dir=tmp_path / "models",
        file_encryption_key=FileCipher.generate_key(),
    )
    application = create_meeting_application(settings)
    await application.database.create_schema()
    session = await application.controller.create(guild_id=guild_id, host_user_id=host_id)
    changes = ConsentChangeRepository(application.database.sessions)
    await changes.grant_all(
        session_id=session.id,
        guild_id=guild_id,
        user_id=host_id,
        effective_at_ms=1_000,
        trigger="test",
    )
    coordinator = MeetingVoiceCoordinator(
        bot=cast(Any, FakeBot()),
        settings=settings,
        application=application,
        tts_provider=cast(TTSProvider, FakeTTSProvider(tmp_path / "fake.wav")),
    )
    application.controller.set_consent_runtime(coordinator)
    finalise_entered = asyncio.Event()

    async def handle_action(_session: Any, action: str) -> None:
        if action == "finalise_capture":
            finalise_entered.set()

    application.controller.set_runtime_action_handler(handle_action)

    async def fixed_timeline(**_kwargs: int) -> int:
        return 2_000

    original_apply_withdrawal = coordinator.apply_withdrawal
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def delayed_apply_withdrawal(**kwargs: int) -> None:
        cleanup_entered.set()
        await release_cleanup.wait()
        await original_apply_withdrawal(**kwargs)

    monkeypatch.setattr(coordinator, "timeline_now_ms", fixed_timeline)
    monkeypatch.setattr(coordinator, "apply_withdrawal", delayed_apply_withdrawal)
    withdrawal_task = asyncio.create_task(
        application.controller.withdraw_all(guild_id=guild_id, user_id=host_id)
    )
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
    delete_task = asyncio.create_task(
        application.controller.delete(guild_id=guild_id, actor_user_id=host_id)
    )
    await asyncio.sleep(0.1)
    delete_completed_before_cleanup = delete_task.done()
    finalise_entered_before_cleanup = finalise_entered.is_set()

    release_cleanup.set()
    await withdrawal_task
    deleted = await delete_task
    latest = await application.sessions.latest_for_guild(guild_id)
    await application.database.close()

    assert not delete_completed_before_cleanup
    assert not finalise_entered_before_cleanup
    assert deleted.id == session.id
    assert latest is None
