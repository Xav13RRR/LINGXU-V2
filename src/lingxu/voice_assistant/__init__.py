"""Ephemeral wake-word Meeting assistant."""

from lingxu.voice_assistant.response import (
    OllamaShortAnswerProvider,
    ShortAnswerProvider,
)
from lingxu.voice_assistant.service import MeetingVoiceAssistant
from lingxu.voice_assistant.wakeword import (
    KeywordSpotter,
    SherpaOnnxKeywordSpotter,
)
from lingxu.voice_assistant.window import WakeWindowBuilder

__all__ = [
    "KeywordSpotter",
    "MeetingVoiceAssistant",
    "OllamaShortAnswerProvider",
    "SherpaOnnxKeywordSpotter",
    "ShortAnswerProvider",
    "WakeWindowBuilder",
]
