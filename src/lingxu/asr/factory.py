"""Configuration-to-runtime assembly for realtime ASR."""

from __future__ import annotations

from lingxu.asr.provider import RealtimeASRProvider, SenseVoiceProvider
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.service import (
    RealtimeASRMetrics,
    RealtimeASRService,
    RealtimeUtteranceStore,
)
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.config import Settings
from lingxu.voice.meeting_consent import MeetingConsentRegistry


def create_realtime_asr_provider(
    settings: Settings,
    *,
    allow_download: bool = False,
) -> RealtimeASRProvider:
    """Create one lazy provider that consented ASR and ephemeral wake turns can share."""

    if settings.realtime_asr_provider != "sensevoice":
        raise ValueError(f"unsupported realtime ASR provider: {settings.realtime_asr_provider}")
    return SenseVoiceProvider(
        model=settings.realtime_asr_model,
        model_revision=settings.realtime_asr_model_revision,
        vad_model=settings.realtime_asr_vad_model,
        vad_model_revision=settings.realtime_asr_vad_model_revision,
        model_cache_dir=settings.model_cache_dir,
        expected_model_sha256=settings.realtime_asr_model_sha256,
        expected_vad_model_sha256=settings.realtime_asr_vad_model_sha256,
        allow_download=allow_download,
    )


def create_realtime_asr_service(
    *,
    settings: Settings,
    session_id: str,
    session_started_monotonic_ns: int,
    consents: MeetingConsentRegistry,
    utterances: RealtimeUtteranceStore,
    provider: RealtimeASRProvider | None = None,
) -> RealtimeASRService:
    """Build the configured Phase 4 runtime without loading weights eagerly."""

    selected_provider = provider or create_realtime_asr_provider(settings)
    return RealtimeASRService(
        session_id=session_id,
        session_started_monotonic_ns=session_started_monotonic_ns,
        consents=consents,
        frame_queue=PerUserRealtimeQueue(
            max_frames_per_user=settings.realtime_asr_queue_max_frames_per_user
        ),
        utterance_builder=UtteranceBuilder(
            session_id=session_id,
            vad_threshold=settings.realtime_asr_vad_threshold,
            vad_start_ms=settings.realtime_asr_vad_start_ms,
            vad_end_ms=settings.realtime_asr_vad_end_ms,
            min_utterance_ms=settings.realtime_asr_min_utterance_ms,
            max_utterance_seconds=settings.realtime_asr_max_utterance_seconds,
        ),
        provider=selected_provider,
        utterances=utterances,
        metrics=RealtimeASRMetrics(session_id),
        language=settings.realtime_asr_language_mode,
        hotwords=settings.realtime_asr_hotword_values,
        timeout_seconds=settings.realtime_asr_timeout_seconds,
        max_concurrency=settings.realtime_asr_max_concurrency,
        max_pending_utterances=settings.realtime_asr_queue_max_utterances,
    )
