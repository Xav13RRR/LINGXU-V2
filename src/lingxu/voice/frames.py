"""Audio frame value objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionAudioFrame:
    """One formal Meeting PCM frame on a Session-relative monotonic timeline."""

    session_id: str
    user_id: int
    received_monotonic_ns: int
    pcm_s16le_48k_stereo: bytes
    consent_generation: int = 0
