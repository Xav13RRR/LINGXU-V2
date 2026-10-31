"""Offline encrypted-Segment transcription adapter for the post-Meeting Worker."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
import soundfile

from lingxu.asr.models import UtteranceAudio
from lingxu.asr.provider import RealtimeASRProvider
from lingxu.storage.encryption import FileCipher


class SegmentTranscriber:
    """Decrypt FLAC only in memory and reuse the approved non-LLM ASR provider."""

    def __init__(
        self,
        *,
        cipher: FileCipher,
        provider: RealtimeASRProvider,
        language: str,
        hotwords: tuple[str, ...],
    ) -> None:
        self._cipher = cipher
        self._provider = provider
        self._language = language
        self._hotwords = hotwords

    async def transcribe(
        self,
        *,
        session_id: str,
        user_id: int,
        encrypted_path: Path,
        started_ms: int,
        ended_ms: int,
    ) -> tuple[str, str]:
        plaintext = self._cipher.decrypt(await asyncio.to_thread(encrypted_path.read_bytes))
        samples, sample_rate = await asyncio.to_thread(
            soundfile.read,
            io.BytesIO(plaintext),
            dtype="float32",
            always_2d=True,
        )
        if sample_rate != 48_000 or samples.shape[1] != 2:
            raise ValueError("recorded Segment must be 48 kHz stereo")
        pcm = np.clip(samples * 32_767.0, -32_768, 32_767).astype("<i2").tobytes()
        result = await self._provider.transcribe(
            UtteranceAudio(
                session_id=session_id,
                user_id=user_id,
                started_monotonic_ns=started_ms * 1_000_000,
                ended_monotonic_ns=ended_ms * 1_000_000,
                pcm_s16le_48k_stereo=pcm,
            ),
            language=self._language,
            hotwords=self._hotwords,
        )
        return result.raw_text, result.normalised_text
