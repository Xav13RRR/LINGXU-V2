"""Configured offline TTS assembly."""

from __future__ import annotations

from lingxu.config import Settings
from lingxu.tts.provider import CosyVoiceHttpProvider, EspeakNgProvider, TTSProvider

COMMON_PROCESS_PROMPTS = (
    "本次会议将于十秒后开始。",
    "下面介绍本次会议议程。",
    "现在进入当前议题。",
    "请下一位发言人准备。",
    "当前流程剩余约六十秒。",
    "投票现在开始。",
    "投票已经结束。",
    "会议已暂停。",
    "会议继续。",
    "会议结束, 感谢参加。会议总结将自动发送至本语音频道的文字聊天。",
)


def create_tts_provider(settings: Settings) -> TTSProvider:
    if settings.tts_provider == "espeak-ng":
        return EspeakNgProvider(
            cache_dir=settings.tts_cache_dir,
            binary=settings.tts_binary,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            timeout_seconds=settings.tts_timeout_seconds,
        )
    if settings.tts_provider == "cosyvoice-sft":
        return CosyVoiceHttpProvider(
            cache_dir=settings.tts_cache_dir,
            base_url=settings.tts_base_url,
            model=settings.tts_model,
            model_revision=settings.tts_model_revision,
            model_digest=settings.tts_model_digest,
            voice=settings.tts_voice,
            speed=settings.tts_speed,
            timeout_seconds=settings.tts_timeout_seconds,
        )
    raise ValueError(f"unsupported TTS provider: {settings.tts_provider}")
