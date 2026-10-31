# ruff: noqa: RUF001
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy import text

from lingxu.bot.resident_assistant import ResidentAssistantCoordinator
from lingxu.bot.voice_modes import GuildVoiceModeRegistry
from lingxu.config import Settings
from lingxu.domain.enums import ServiceCapacityMode
from lingxu.domain.errors import ServiceCapacityFullError
from lingxu.storage.database import Database
from lingxu.storage.guild_wake_phrase_repository import GuildWakePhraseRepository
from lingxu.storage.resident_assistant_repository import ResidentAssistantRepository
from lingxu.storage.service_capacity_repository import ServiceCapacityRepository
from lingxu.tts.models import TTSAudio
from lingxu.tts.provider import TTSProvider
from lingxu.voice.dave_adapter import DaveStatus


class FakeAssistant:
    def __init__(self, wake_phrase: str, *, fail_start: bool = False) -> None:
        self.started = False
        self.stopped = False
        self.fail_start = fail_start
        self.received: list[int] = []
        self.wake_phrase = wake_phrase
        self.wake_phrase_updates: list[str] = []

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("synthetic unsupported wake phrase")
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def update_wake_phrase(self, wake_phrase: str) -> None:
        self.wake_phrase = wake_phrase
        self.wake_phrase_updates.append(wake_phrase)

    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool:
        self.received.append(user_id)
        return True


class FakeAssistantFactory:
    def __init__(self, *, fail_phrase: str | None = None) -> None:
        self.instances: list[FakeAssistant] = []
        self.fail_phrase = fail_phrase

    def __call__(self, *, session_id: str, speak: Any, wake_phrase: str) -> FakeAssistant:
        assistant = FakeAssistant(wake_phrase, fail_start=wake_phrase == self.fail_phrase)
        self.instances.append(assistant)
        return assistant


class BlockingStartAssistant(FakeAssistant):
    def __init__(self, wake_phrase: str) -> None:
        super().__init__(wake_phrase)
        self.entered = asyncio.Event()

    async def start(self) -> None:
        self.entered.set()
        await asyncio.Event().wait()


class BlockingAssistantFactory:
    def __init__(self) -> None:
        self.instances: list[BlockingStartAssistant] = []
        self.created = asyncio.Event()

    def __call__(
        self,
        *,
        session_id: str,
        speak: Any,
        wake_phrase: str,
    ) -> BlockingStartAssistant:
        assistant = BlockingStartAssistant(wake_phrase)
        self.instances.append(assistant)
        self.created.set()
        return assistant


class CleanupBlockingAssistant(FakeAssistant):
    def __init__(self, wake_phrase: str) -> None:
        super().__init__(wake_phrase)
        self.cleanup_entered = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.stop_count = 0

    async def stop(self) -> None:
        self.stop_count += 1
        self.cleanup_entered.set()
        await self.release_cleanup.wait()
        self.stopped = True


class CleanupBlockingAssistantFactory:
    def __init__(self) -> None:
        self.instances: list[CleanupBlockingAssistant] = []

    def __call__(
        self,
        *,
        session_id: str,
        speak: Any,
        wake_phrase: str,
    ) -> CleanupBlockingAssistant:
        assistant = CleanupBlockingAssistant(wake_phrase)
        self.instances.append(assistant)
        return assistant


class FakeVoiceClient:
    def __init__(self, guild: Any) -> None:
        self._guild = guild
        self.connected = True
        self.listening = False
        self.sink: Any = None
        self.after: Any = None

    def listen(self, sink: Any, *, after: Any) -> None:
        self.sink = sink
        self.after = after
        self.listening = True

    def is_connected(self) -> bool:
        return self.connected

    def is_listening(self) -> bool:
        return self.listening

    def is_playing(self) -> bool:
        return False

    def stop_listening(self) -> None:
        self.listening = False

    async def disconnect(self, *, force: bool) -> None:
        self.connected = False
        self.listening = False
        self._guild.voice_client = None


class FakeVoiceChannel:
    def __init__(self, guild_id: int) -> None:
        self.id = 9_001
        self.mention = "<#9001>"
        self.guild = SimpleNamespace(id=guild_id, voice_client=None, me=object())
        self.clients: list[FakeVoiceClient] = []

    async def connect(self, **kwargs: Any) -> FakeVoiceClient:
        client = FakeVoiceClient(self.guild)
        self.clients.append(client)
        self.guild.voice_client = client
        return client

    def permissions_for(self, _: Any) -> Any:
        return SimpleNamespace(view_channel=True, connect=True, speak=True)


class CleanupFaultVoiceClient(FakeVoiceClient):
    def __init__(self, guild: Any) -> None:
        super().__init__(guild)
        self.disconnect_count = 0

    def is_listening(self) -> bool:
        raise OSError("synthetic listening-state failure")

    async def disconnect(self, *, force: bool) -> None:
        self.disconnect_count += 1
        await super().disconnect(force=force)


class CleanupFaultVoiceChannel(FakeVoiceChannel):
    async def connect(self, **kwargs: Any) -> CleanupFaultVoiceClient:
        client = CleanupFaultVoiceClient(self.guild)
        self.clients.append(client)
        self.guild.voice_client = client
        return client


class FakeBot:
    def __init__(self, channel: FakeVoiceChannel) -> None:
        self.channel = channel

    def get_channel(self, channel_id: int) -> FakeVoiceChannel | None:
        return self.channel if self.channel.id == channel_id else None


class FakeEphemeralTTS:
    async def synthesize(self, text: str) -> TTSAudio:
        raise AssertionError("persistent synthesis must not be used")

    async def pre_generate(self, prompts: tuple[str, ...]) -> tuple[TTSAudio, ...]:
        return ()

    async def synthesize_ephemeral(self, text: str, *, output_dir: Path) -> TTSAudio:
        path = await asyncio.to_thread(self._write_audio, output_dir)
        return TTSAudio(
            cache_key="ephemeral",
            path=path,
            sha256="0" * 64,
            duration_ms=100,
            sample_rate=24_000,
            channels=1,
            from_cache=False,
        )

    @staticmethod
    def _write_audio(output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "answer.wav"
        path.write_bytes(b"unused")
        return path


def _ready_status(_: Any) -> DaveStatus:
    return DaveStatus(
        supported_by_receive_fork=True,
        enabled=True,
        ready=True,
        protocol_version=1,
        privacy_code_available=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resident_voice_stays_recoverable_and_creates_no_meeting_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 900
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'resident-voice.db').as_posix()}")
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)
    wake_phrases = GuildWakePhraseRepository(database.sessions)
    capacity = ServiceCapacityRepository(database.sessions)
    channel = FakeVoiceChannel(guild_id)
    bot = FakeBot(channel)
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'resident-voice.db').as_posix()}",
        runtime_tmp=tmp_path / "runtime",
        resident_assistant_enabled=False,
        voice_duplex_dave_ready_timeout_seconds=0.1,
    )
    monkeypatch.setattr(
        "lingxu.bot.resident_assistant.inspect_dave_status",
        _ready_status,
    )
    monkeypatch.setattr(
        "lingxu.bot.resident_assistant.discord.VoiceChannel",
        FakeVoiceChannel,
    )
    first_factory = FakeAssistantFactory()
    first = ResidentAssistantCoordinator(
        bot=cast(Any, bot),
        settings=settings,
        repository=repository,
        voice_modes=GuildVoiceModeRegistry(),
        assistant_factory=cast(Any, first_factory),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )

    await wake_phrases.set(
        guild_id=guild_id,
        phrase="灵序语音助手",
        updated_by_user_id=901,
    )
    record = await first.start(channel=cast(Any, channel), creator_user_id=901)
    assert first.runtime_active(guild_id)
    assert first_factory.instances[0].started
    assert first_factory.instances[0].wake_phrase == "灵序语音助手"
    assert await first.update_wake_phrase(guild_id=guild_id, phrase="呼叫会议助手")
    assert first_factory.instances[0].wake_phrase_updates == ["呼叫会议助手"]
    assert not await first.update_wake_phrase(guild_id=guild_id + 1, phrase="呼叫会议助手")
    with pytest.raises(ServiceCapacityFullError) as busy:
        await first.start(
            channel=cast(Any, FakeVoiceChannel(guild_id + 1)),
            creator_user_id=902,
        )
    assert str(busy.value) == (
        "服务容量已满。聆序目前正在其它 Discord 服务器提供服务，请稍后再试。"
    )
    assert channel.clients[0].sink.__class__.__name__ == "AssistantOnlyPcmSink"
    async with database.sessions() as db:
        counts = {
            table: (await db.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar_one()
            for table in (
                "meeting_sessions",
                "meeting_events",
                "meeting_realtime_utterances",
                "audio_segments",
                "transcript_entries",
                "processing_jobs",
                "artifacts",
            )
        }
    assert counts == {table: 0 for table in counts}

    await first.close_all()
    assert not first.runtime_active(guild_id)
    assert (await repository.active_for_guild(guild_id)).id == record.id  # type: ignore[union-attr]
    assert (await capacity.current()).owner_id == record.id  # type: ignore[union-attr]

    second_factory = FakeAssistantFactory(fail_phrase="灵序语音助手")
    second_voice_modes = GuildVoiceModeRegistry()
    second = ResidentAssistantCoordinator(
        bot=cast(Any, bot),
        settings=settings,
        repository=repository,
        voice_modes=second_voice_modes,
        assistant_factory=cast(Any, second_factory),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )
    await second.recover_all()
    assert not second.runtime_active(guild_id)
    assert await capacity.current() is None
    assert second_voice_modes.current(guild_id) is None

    await wake_phrases.set(
        guild_id=guild_id,
        phrase="呼叫会议助手",
        updated_by_user_id=901,
    )
    assert await second.retry_failed_wake_phrase_recovery(guild_id)
    assert second.runtime_active(guild_id)
    assert len(channel.clients) == 2
    assert not second_factory.instances[0].started
    assert second_factory.instances[1].started
    assert second_factory.instances[1].wake_phrase == "呼叫会议助手"
    assert not await second.retry_failed_wake_phrase_recovery(guild_id)

    await second.handle_bot_voice_state(
        guild_id=guild_id,
        before_channel_id=channel.id,
        after_channel_id=None,
    )
    assert not second.runtime_active(guild_id)
    assert await repository.active_for_guild(guild_id) is None
    assert await capacity.current() is None
    assert [event.event_type for event in await repository.events(record.id)] == [
        "STARTED",
        "RECOVERED",
        "STOPPED",
    ]
    assert not tuple((tmp_path / "runtime").rglob("*.wav"))
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_resident_recovery_releases_voice_and_capacity_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 910
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'cancel-recovery.db').as_posix()}")
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)
    wake_phrases = GuildWakePhraseRepository(database.sessions)
    capacity = ServiceCapacityRepository(database.sessions)
    channel = FakeVoiceChannel(guild_id)
    bot = FakeBot(channel)
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'cancel-recovery.db').as_posix()}",
        runtime_tmp=tmp_path / "runtime",
        resident_assistant_enabled=False,
    )
    monkeypatch.setattr(
        "lingxu.bot.resident_assistant.discord.VoiceChannel",
        FakeVoiceChannel,
    )
    record = await repository.create_active(
        session_id="resident-cancel-recovery",
        guild_id=guild_id,
        channel_id=channel.id,
        creator_user_id=911,
    )
    await capacity.claim(
        guild_id=guild_id,
        mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
        owner_id=record.id,
    )
    voice_modes = GuildVoiceModeRegistry()
    factory = BlockingAssistantFactory()
    coordinator = ResidentAssistantCoordinator(
        bot=cast(Any, bot),
        settings=settings,
        repository=repository,
        voice_modes=voice_modes,
        assistant_factory=cast(Any, factory),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )

    recovery = asyncio.create_task(coordinator.recover_all())
    async with asyncio.timeout(1):
        await factory.created.wait()
        await factory.instances[0].entered.wait()
    assert voice_modes.current(guild_id) is not None
    recovery.cancel()
    with pytest.raises(asyncio.CancelledError):
        await recovery

    assert not coordinator.runtime_active(guild_id)
    assert voice_modes.current(guild_id) is None
    assert await capacity.current() is None
    assert (await repository.active_for_guild(guild_id)).id == record.id  # type: ignore[union-attr]
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_start_cancellation_waits_for_every_resident_cleanup_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 915
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'double-cancel.db').as_posix()}")
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)
    capacity = ServiceCapacityRepository(database.sessions)
    voice_modes = GuildVoiceModeRegistry()
    wake_phrases = GuildWakePhraseRepository(database.sessions)
    channel = FakeVoiceChannel(guild_id)
    factory = CleanupBlockingAssistantFactory()
    coordinator = ResidentAssistantCoordinator(
        bot=cast(Any, FakeBot(channel)),
        settings=Settings(
            _env_file=None,
            database_url=f"sqlite+aiosqlite:///{(tmp_path / 'double-cancel.db').as_posix()}",
            runtime_tmp=tmp_path / "runtime",
            resident_assistant_enabled=False,
        ),
        repository=repository,
        voice_modes=voice_modes,
        assistant_factory=cast(Any, factory),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )
    dave_entered = asyncio.Event()

    async def block_dave(_client: Any) -> None:
        dave_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(coordinator, "_wait_for_dave", block_dave)
    startup = asyncio.create_task(
        coordinator.start(channel=cast(Any, channel), creator_user_id=916)
    )
    await asyncio.wait_for(dave_entered.wait(), timeout=1)
    assistant = factory.instances[0]

    startup.cancel("initial-resident-cancel")
    await asyncio.wait_for(assistant.cleanup_entered.wait(), timeout=1)
    startup.cancel("repeated-resident-cleanup-cancel")
    assistant.release_cleanup.set()

    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(startup, timeout=1)

    assert raised.value.args == ("initial-resident-cancel",)
    assert assistant.stop_count == 1
    assert assistant.stopped
    assert len(channel.clients) == 1
    assert not channel.clients[0].is_connected()
    assert voice_modes.current(guild_id) is None
    assert await capacity.current() is None
    assert await repository.active_for_guild(guild_id) is None
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resident_cleanup_failure_preserves_error_and_releases_later_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild_id = 917
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'cleanup-failure.db').as_posix()}")
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)
    capacity = ServiceCapacityRepository(database.sessions)
    voice_modes = GuildVoiceModeRegistry()
    wake_phrases = GuildWakePhraseRepository(database.sessions)
    channel = CleanupFaultVoiceChannel(guild_id)
    factory = FakeAssistantFactory()
    coordinator = ResidentAssistantCoordinator(
        bot=cast(Any, FakeBot(channel)),
        settings=Settings(
            _env_file=None,
            database_url=f"sqlite+aiosqlite:///{(tmp_path / 'cleanup-failure.db').as_posix()}",
            runtime_tmp=tmp_path / "runtime",
            resident_assistant_enabled=False,
        ),
        repository=repository,
        voice_modes=voice_modes,
        assistant_factory=cast(Any, factory),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )

    async def fail_dave(_client: Any) -> None:
        raise RuntimeError("synthetic resident DAVE failure")

    monkeypatch.setattr(coordinator, "_wait_for_dave", fail_dave)

    with pytest.raises(RuntimeError, match="synthetic resident DAVE failure"):
        await coordinator.start(channel=cast(Any, channel), creator_user_id=918)

    assert factory.instances[0].stopped
    assert len(channel.clients) == 1
    client = cast(CleanupFaultVoiceClient, channel.clients[0])
    assert client.disconnect_count == 1
    assert not client.is_connected()
    assert voice_modes.current(guild_id) is None
    assert await capacity.current() is None
    assert await repository.active_for_guild(guild_id) is None
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_resident_recovery_channel_retains_durable_pending_owner(
    tmp_path: Path,
) -> None:
    guild_id = 920
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'missing-channel.db').as_posix()}")
    await database.create_schema()
    repository = ResidentAssistantRepository(database.sessions)
    wake_phrases = GuildWakePhraseRepository(database.sessions)
    capacity = ServiceCapacityRepository(database.sessions)
    record = await repository.create_active(
        session_id="resident-missing-channel",
        guild_id=guild_id,
        channel_id=9_201,
        creator_user_id=921,
    )
    capacity_lease = await capacity.claim(
        guild_id=guild_id,
        mode=ServiceCapacityMode.RESIDENT_ASSISTANT,
        owner_id=record.id,
    )
    coordinator = ResidentAssistantCoordinator(
        bot=cast(Any, SimpleNamespace(get_channel=lambda _channel_id: None)),
        settings=Settings(
            _env_file=None,
            database_url=f"sqlite+aiosqlite:///{(tmp_path / 'missing-channel.db').as_posix()}",
            runtime_tmp=tmp_path / "runtime",
            resident_assistant_enabled=False,
        ),
        repository=repository,
        voice_modes=GuildVoiceModeRegistry(),
        assistant_factory=cast(Any, FakeAssistantFactory()),
        tts_provider=cast(TTSProvider, FakeEphemeralTTS()),
        wake_phrases=wake_phrases,
        capacity=capacity,
    )

    await coordinator.recover_all()

    assert (await repository.active_for_guild(guild_id)).id == record.id  # type: ignore[union-attr]
    assert await capacity.current() == capacity_lease
    assert not await coordinator.retry_failed_wake_phrase_recovery(guild_id)
    assert (await repository.active_for_guild(guild_id)).id == record.id  # type: ignore[union-attr]
    await database.close()
