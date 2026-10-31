"""TTS and playback value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


class PlaybackPriority(IntEnum):
    REPEAT = 20
    NORMAL_NOTICE = 40
    VOTE_PROMPT = 50
    TIMER_REMINDER = 60
    REQUIRED_PROCESS_PROMPT = 80
    HOST_OVERRIDE = 90
    EMERGENCY = 100


class InterruptPolicy(StrEnum):
    NEVER = "NEVER"
    LOWER_PRIORITY = "LOWER_PRIORITY"
    ALWAYS = "ALWAYS"


@dataclass(frozen=True, slots=True)
class TTSAudio:
    """One validated cached or ephemeral WAV without retaining source text in metadata."""

    cache_key: str
    path: Path
    sha256: str
    duration_ms: int
    sample_rate: int
    channels: int
    from_cache: bool


@dataclass(frozen=True, slots=True)
class PlaybackRequest:
    """One deduplicated text-to-speech request."""

    request_id: str
    dedupe_key: str
    text: str
    priority: PlaybackPriority = PlaybackPriority.NORMAL_NOTICE
    interrupt_policy: InterruptPolicy = InterruptPolicy.LOWER_PRIORITY
