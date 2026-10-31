"""Content-safe real-model smoke for wake detection and local short answers."""

from __future__ import annotations

import asyncio
import hashlib
import time

import numpy as np
import soundfile as sf

from lingxu.config import Settings
from lingxu.tts.factory import create_tts_provider
from lingxu.voice.frames import SessionAudioFrame
from lingxu.voice_assistant.response import OllamaShortAnswerProvider
from lingxu.voice_assistant.wakeword import SherpaOnnxKeywordSpotter
from lingxu.voice_assistant.window import WakeWindowBuilder


def _discord_pcm(path: str) -> bytes:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    output_count = max(1, round(len(mono) * 48_000 / sample_rate))
    positions = np.linspace(0, max(0, len(mono) - 1), output_count)
    resampled = np.interp(positions, np.arange(len(mono)), mono)
    int16 = np.clip(resampled * 32_767, -32_768, 32_767).astype("<i2")
    return np.column_stack((int16, int16)).astype("<i2").tobytes()


async def _streaming_detect(
    *,
    settings: Settings,
    spotter: SherpaOnnxKeywordSpotter,
    pcm: bytes,
    user_id: int,
) -> bool:
    builder = WakeWindowBuilder(
        session_id="assistant-smoke",
        activity_threshold=settings.meeting_assistant_vad_threshold,
        window_ms=settings.meeting_assistant_kws_window_ms,
        hop_ms=settings.meeting_assistant_kws_hop_ms,
        min_window_ms=settings.meeting_assistant_kws_min_window_ms,
        preroll_ms=settings.meeting_assistant_kws_preroll_ms,
    )
    frame_bytes = 48_000 * 4 // 50
    padding = b"\0" * (48_000 * 4 * 2)
    payload = pcm + padding
    received_ns = 0
    for offset in range(0, len(payload), frame_bytes):
        frame_pcm = payload[offset : offset + frame_bytes]
        window = builder.accept(
            SessionAudioFrame(
                session_id="assistant-smoke",
                user_id=user_id,
                received_monotonic_ns=received_ns,
                pcm_s16le_48k_stereo=frame_pcm,
            )
        )
        received_ns += max(1, len(frame_pcm) * 1_000_000_000 // (48_000 * 4))
        if window is not None and await spotter.detect(window):
            return True
    return False


async def main() -> None:
    settings = Settings()
    if not settings.meeting_assistant_enabled:
        raise RuntimeError("Meeting assistant is disabled")
    spotter = SherpaOnnxKeywordSpotter(
        model_dir=settings.meeting_assistant_wakeword_model_dir,
        expected_model_digest=settings.meeting_assistant_wakeword_model_digest,
        keywords_score=settings.meeting_assistant_wakeword_score,
        keywords_threshold=settings.meeting_assistant_wakeword_threshold,
    )
    await spotter.prepare()
    tts = create_tts_provider(settings)
    phrases = (
        ("呼叫指挥中心", True),
        ("呼叫智慧中心", True),
        ("呼叫指中心", True),
        ("我们继续讨论，然后呼叫指挥中心，请说明当前流程", True),
        ("指挥中心", False),
    )
    detections: list[bool] = []
    for index, (phrase, expected) in enumerate(phrases):
        audio = await tts.synthesize(phrase)
        pcm = _discord_pcm(str(audio.path))
        detected = await _streaming_detect(
            settings=settings,
            spotter=spotter,
            pcm=pcm,
            user_id=index + 1,
        )
        detections.append(detected)
        if detected is not expected:
            raise RuntimeError(
                f"Meeting wake-word acceptance smoke failed at case {index}: "
                f"expected={expected}, detected={detected}"
            )

    api_key = settings.meeting_assistant_llm_api_key
    answers = OllamaShortAnswerProvider(
        base_url=settings.meeting_assistant_llm_base_url,
        model=settings.meeting_assistant_llm_model,
        expected_model_digest=settings.meeting_assistant_llm_model_digest,
        api_key=(api_key.get_secret_value() if api_key is not None else None),
        timeout_seconds=settings.meeting_assistant_llm_timeout_seconds,
        max_output_tokens=settings.meeting_assistant_llm_max_output_tokens,
        max_question_chars=settings.meeting_assistant_max_question_chars,
    )
    started = time.perf_counter()
    answer = await answers.answer(
        question="请用一句话说明当前会议状态。",
        context={
            "meeting_state": "SPEAKER_ACTIVE",
            "agenda": [
                {
                    "position": 1,
                    "title": "Synthetic assistant smoke",
                    "status": "ACTIVE",
                    "duration_minutes": 5,
                }
            ],
            "speaker_active": True,
            "speaking_queue_length": 0,
            "vote_open": False,
            "vote_title": None,
        },
    )
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    print(
        {
            "wake_cases": len(detections),
            "wake_expected_matches": sum(
                detected is expected
                for detected, (_, expected) in zip(detections, phrases, strict=True)
            ),
            "answer_nonempty": bool(answer),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "elapsed_ms": elapsed_ms,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
