"""Minimal consent-gated discord-ext-voice-recv AudioSink."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import discord
from discord.ext import voice_recv

from lingxu.asr.service import RealtimeASRService
from lingxu.voice.segment_recorder import FormalRecordingPipeline


class EphemeralPcmReceiver(Protocol):
    def receive(
        self,
        *,
        user_id: int,
        received_monotonic_ns: int,
        pcm_s16le_48k_stereo: bytes,
    ) -> bool: ...


class MeetingPcmSink(voice_recv.AudioSink):
    """Minimal formal Meeting sink backed by the consent-gated recording pipeline."""

    def __init__(
        self,
        *,
        pipeline: FormalRecordingPipeline,
        realtime_asr: RealtimeASRService | None = None,
        ephemeral_assistant: EphemeralPcmReceiver | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._realtime_asr = realtime_asr
        self._ephemeral_assistant = ephemeral_assistant
        self._clock_ns = clock_ns

    def wants_opus(self) -> bool:
        return False

    def write(
        self,
        user: discord.User | discord.Member | None,
        data: voice_recv.VoiceData,
    ) -> None:
        if user is None:
            self._pipeline.metrics.increment("unknown_user_frames")
            return
        user_id = int(user.id)
        received_ns = self._clock_ns()
        pcm = bytes(data.pcm)
        self._pipeline.receive(
            user_id=user_id,
            received_monotonic_ns=received_ns,
            pcm_s16le_48k_stereo=pcm,
        )
        if self._realtime_asr is not None:
            self._realtime_asr.receive(
                user_id=user_id,
                received_monotonic_ns=received_ns,
                pcm_s16le_48k_stereo=pcm,
            )
        if self._ephemeral_assistant is not None and not bool(getattr(user, "bot", False)):
            self._ephemeral_assistant.receive(
                user_id=user_id,
                received_monotonic_ns=received_ns,
                pcm_s16le_48k_stereo=pcm,
            )

    def cleanup(self) -> None:
        """The owning Meeting controller performs graceful finalisation."""


class AssistantOnlyPcmSink(voice_recv.AudioSink):
    """Forward human PCM only to the ephemeral assistant without persistence."""

    def __init__(
        self,
        *,
        assistant: EphemeralPcmReceiver,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__()
        self._assistant = assistant
        self._clock_ns = clock_ns

    def wants_opus(self) -> bool:
        return False

    def write(
        self,
        user: discord.User | discord.Member | None,
        data: voice_recv.VoiceData,
    ) -> None:
        """Perform one bounded copy and non-blocking queue insertion only."""

        if user is None or bool(getattr(user, "bot", False)):
            return
        self._assistant.receive(
            user_id=int(user.id),
            received_monotonic_ns=self._clock_ns(),
            pcm_s16le_48k_stereo=bytes(data.pcm),
        )

    def cleanup(self) -> None:
        """The owning resident coordinator clears all ephemeral state."""
