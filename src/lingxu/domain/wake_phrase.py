"""Guild wake-phrase value object and runtime-compatible validation."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime

DEFAULT_WAKE_PHRASE = "呼叫指挥中心"
_MIN_HAN_CHARACTERS = 4
_MAX_HAN_CHARACTERS = 16


@dataclass(frozen=True, slots=True)
class GuildWakePhraseRecord:
    """The effective wake phrase for exactly one Discord Guild."""

    guild_id: int
    phrase: str
    is_override: bool
    updated_by_user_id: int | None
    updated_at: datetime | None


def _is_cjk_unified_ideograph(character: str) -> bool:
    return unicodedata.name(character, "").startswith("CJK UNIFIED IDEOGRAPH-")


def normalise_wake_phrase(value: str) -> str:
    """Normalize a phrase accepted by the pinned Mandarin keyword spotter."""

    phrase = unicodedata.normalize("NFKC", value)
    if not all(_is_cjk_unified_ideograph(character) for character in phrase):
        raise ValueError("唤醒词只能包含连续汉字, 不能包含空格、标点或不可见字符。")
    if not _MIN_HAN_CHARACTERS <= len(phrase) <= _MAX_HAN_CHARACTERS:
        raise ValueError("唤醒词必须包含 4 到 16 个汉字。")
    return phrase
