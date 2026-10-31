"""Discord voice-send adapter for cached TTS artifacts."""

from __future__ import annotations

import asyncio

import discord

from lingxu.tts.models import TTSAudio


class DiscordVoicePlaybackTarget:
    """Play through the same voice client that remains subscribed to receive."""

    def __init__(self, voice_client: discord.VoiceClient, *, volume: float = 0.8) -> None:
        if not 0.0 <= volume <= 2.0:
            raise ValueError("playback volume must be between 0 and 2")
        self._voice_client = voice_client
        self._volume = volume

    async def play(self, audio: TTSAudio) -> None:
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()
        source = discord.PCMVolumeTransformer(
            discord.FFmpegPCMAudio(str(audio.path)),
            volume=self._volume,
        )

        def after(error: Exception | None) -> None:
            def resolve() -> None:
                if completed.done():
                    return
                if error is None:
                    completed.set_result(None)
                else:
                    completed.set_exception(error)

            loop.call_soon_threadsafe(resolve)

        try:
            self._voice_client.play(source, after=after)
        except BaseException:
            source.cleanup()
            raise
        await completed

    def stop(self) -> None:
        if self._voice_client.is_playing():
            self._voice_client.stop()


class DiscordTextFallback:
    """Send failed spoken prompts as plain text without mention expansion."""

    def __init__(self, channel: discord.abc.Messageable) -> None:
        self._channel = channel

    async def send(self, text: str) -> None:
        await self._channel.send(
            text,
            allowed_mentions=discord.AllowedMentions.none(),
        )
