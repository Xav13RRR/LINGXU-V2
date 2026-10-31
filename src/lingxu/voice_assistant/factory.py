"""Configuration assembly for approved ephemeral wake-word assistants."""

from __future__ import annotations

from typing import Literal

from lingxu.asr.provider import RealtimeASRProvider
from lingxu.asr.queue import PerUserRealtimeQueue
from lingxu.asr.utterance import UtteranceBuilder
from lingxu.config import Settings
from lingxu.voice_assistant.response import OllamaShortAnswerProvider
from lingxu.voice_assistant.service import ContextProvider, MeetingVoiceAssistant, SpeakCallback
from lingxu.voice_assistant.wakeword import SherpaOnnxKeywordSpotter
from lingxu.voice_assistant.window import WakeWindowBuilder


def create_wake_phrase_validator(settings: Settings) -> SherpaOnnxKeywordSpotter:
    """Build a model-bound validator that never activates a voice runtime."""

    return SherpaOnnxKeywordSpotter(
        model_dir=settings.meeting_assistant_wakeword_model_dir,
        expected_model_digest=settings.meeting_assistant_wakeword_model_digest,
        keywords_root=settings.runtime_tmp / "wakewords",
        keywords_score=settings.meeting_assistant_wakeword_score,
        keywords_threshold=settings.meeting_assistant_wakeword_threshold,
    )


def create_meeting_voice_assistant(
    *,
    settings: Settings,
    session_id: str,
    asr_provider: RealtimeASRProvider,
    context_provider: ContextProvider,
    speak: SpeakCallback,
    wake_phrase: str | None = None,
) -> MeetingVoiceAssistant:
    """Build the approved local-only, tool-free assistant role."""

    return _create_voice_assistant(
        settings=settings,
        session_id=session_id,
        asr_provider=asr_provider,
        context_provider=context_provider,
        speak=speak,
        role="meeting",
        wake_phrase=wake_phrase,
    )


async def _resident_context() -> dict[str, str]:
    """Expose only a fixed mode marker, never Discord or Meeting content."""

    return {"mode": "resident_voice_qa"}


def create_resident_voice_assistant(
    *,
    settings: Settings,
    session_id: str,
    asr_provider: RealtimeASRProvider,
    speak: SpeakCallback,
    wake_phrase: str | None = None,
) -> MeetingVoiceAssistant:
    """Build the owner-approved resident role with no Meeting context."""

    return _create_voice_assistant(
        settings=settings,
        session_id=session_id,
        asr_provider=asr_provider,
        context_provider=_resident_context,
        speak=speak,
        role="resident",
        wake_phrase=wake_phrase,
    )


def _create_voice_assistant(
    *,
    settings: Settings,
    session_id: str,
    asr_provider: RealtimeASRProvider,
    context_provider: ContextProvider,
    speak: SpeakCallback,
    role: Literal["meeting", "resident"],
    wake_phrase: str | None,
) -> MeetingVoiceAssistant:
    """Assemble the shared acoustic and bounded local-answer pipeline."""

    api_key = settings.meeting_assistant_llm_api_key
    selected_wake_phrase = wake_phrase or settings.meeting_assistant_wake_phrase
    return MeetingVoiceAssistant(
        session_id=session_id,
        wake_phrase=selected_wake_phrase,
        frame_queue=PerUserRealtimeQueue(
            max_frames_per_user=settings.meeting_assistant_queue_max_frames_per_user
        ),
        utterance_builder=UtteranceBuilder(
            session_id=session_id,
            vad_threshold=settings.meeting_assistant_vad_threshold,
            vad_start_ms=settings.meeting_assistant_vad_start_ms,
            vad_end_ms=settings.meeting_assistant_vad_end_ms,
            min_utterance_ms=settings.meeting_assistant_min_utterance_ms,
            max_utterance_seconds=settings.meeting_assistant_max_utterance_seconds,
        ),
        wake_window_builder=WakeWindowBuilder(
            session_id=session_id,
            activity_threshold=settings.meeting_assistant_vad_threshold,
            window_ms=settings.meeting_assistant_kws_window_ms,
            hop_ms=settings.meeting_assistant_kws_hop_ms,
            min_window_ms=settings.meeting_assistant_kws_min_window_ms,
            preroll_ms=settings.meeting_assistant_kws_preroll_ms,
        ),
        keyword_spotter=SherpaOnnxKeywordSpotter(
            model_dir=settings.meeting_assistant_wakeword_model_dir,
            expected_model_digest=settings.meeting_assistant_wakeword_model_digest,
            wake_phrase=selected_wake_phrase,
            keywords_root=settings.runtime_tmp / "wakewords",
            keywords_score=settings.meeting_assistant_wakeword_score,
            keywords_threshold=settings.meeting_assistant_wakeword_threshold,
        ),
        asr_provider=asr_provider,
        answer_provider=OllamaShortAnswerProvider(
            base_url=settings.meeting_assistant_llm_base_url,
            model=settings.meeting_assistant_llm_model,
            expected_model_digest=settings.meeting_assistant_llm_model_digest,
            api_key=(
                api_key.get_secret_value()
                if api_key is not None and api_key.get_secret_value()
                else None
            ),
            timeout_seconds=settings.meeting_assistant_llm_timeout_seconds,
            max_output_tokens=settings.meeting_assistant_llm_max_output_tokens,
            max_question_chars=settings.meeting_assistant_max_question_chars,
            role=role,
        ),
        context_provider=context_provider,
        speak=speak,
        language=settings.realtime_asr_language_mode,
        asr_timeout_seconds=settings.meeting_assistant_asr_timeout_seconds,
        followup_seconds=settings.meeting_assistant_followup_seconds,
        quarantine_ms=settings.meeting_assistant_quarantine_ms,
        max_pending_utterances=settings.meeting_assistant_queue_max_utterances,
    )
