"""Value objects crossing the realtime ASR provider boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UtteranceAudio:
    """One per-user sentence candidate in Discord's decoded PCM format."""

    session_id: str
    user_id: int
    started_monotonic_ns: int
    ended_monotonic_ns: int
    pcm_s16le_48k_stereo: bytes
    consent_generation: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, (self.ended_monotonic_ns - self.started_monotonic_ns) // 1_000_000)


@dataclass(frozen=True, slots=True)
class ASRResult:
    """Provider output with immutable raw text and a separate display form."""

    raw_text: str
    normalised_text: str
    language: str | None
    processing_ms: int
