from pathlib import Path
from typing import Any

import pytest

from lingxu.tts.discord_playback import DiscordVoicePlaybackTarget
from lingxu.tts.models import TTSAudio


class FakeSource:
    def __init__(self, path: str, **kwargs: Any) -> None:
        self.path = path
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class FakeVoiceClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.play_calls = 0
        self.stop_calls = 0
        self.playing = False

    def play(self, source: Any, *, after: Any) -> None:
        self.play_calls += 1
        if self.fail:
            raise RuntimeError("fixture play failure")
        self.playing = True
        after(None)
        self.playing = False

    def is_playing(self) -> bool:
        return self.playing

    def stop(self) -> None:
        self.stop_calls += 1
        self.playing = False


def _audio() -> TTSAudio:
    return TTSAudio("key", Path("voice.wav"), "0" * 64, 100, 22_050, 1, False)


@pytest.mark.asyncio
async def test_discord_target_plays_and_completes_from_after_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    def ffmpeg(path: str) -> FakeSource:
        source = FakeSource(path)
        created.append(source)
        return source

    monkeypatch.setattr("lingxu.tts.discord_playback.discord.FFmpegPCMAudio", ffmpeg)
    monkeypatch.setattr(
        "lingxu.tts.discord_playback.discord.PCMVolumeTransformer",
        lambda source, volume: source,
    )
    voice = FakeVoiceClient()
    target = DiscordVoicePlaybackTarget(voice, volume=0.8)  # type: ignore[arg-type]

    await target.play(_audio())

    assert voice.play_calls == 1
    assert created[0].path == "voice.wav"


@pytest.mark.asyncio
async def test_discord_target_cleans_source_when_play_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FakeSource("voice.wav")
    monkeypatch.setattr(
        "lingxu.tts.discord_playback.discord.FFmpegPCMAudio",
        lambda path: source,
    )
    monkeypatch.setattr(
        "lingxu.tts.discord_playback.discord.PCMVolumeTransformer",
        lambda candidate, volume: candidate,
    )
    target = DiscordVoicePlaybackTarget(FakeVoiceClient(fail=True))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="fixture"):
        await target.play(_audio())

    assert source.cleaned
