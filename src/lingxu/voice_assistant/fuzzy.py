"""Phonetic prefix confirmation and query extraction for the wake phrase."""

from __future__ import annotations

import re

from pypinyin import Style, lazy_pinyin

_MEANINGFUL = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_LEADING_QUERY_PUNCTUATION = " ，,。.!！?？:：;；、\t\r\n"


def _phonemes(text: str) -> tuple[str, ...]:
    return tuple(
        value.lower()
        for value in lazy_pinyin(
            text,
            style=Style.NORMAL,
            errors=lambda value: value,
            strict=False,
        )
    )


def _distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    prior = list(range(len(right) + 1))
    for left_value in left:
        current = [prior[0] + 1]
        for index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    prior[index] + 1,
                    prior[index - 1] + (left_value != right_value),
                )
            )
        prior = current
    return prior[-1]


def split_wake_query(
    text: str,
    *,
    wake_phrase: str = "呼叫指挥中心",
    max_leading_characters: int | None = None,
    max_syllable_distance: int = 1,
) -> str | None:
    """Return text after a near-phonetic wake occurrence, or ``None`` when rejected."""

    meaningful = [(match.group(), match.start()) for match in _MEANINGFUL.finditer(text)]
    target = _phonemes(wake_phrase)
    if len(meaningful) < len(target) - max_syllable_distance:
        return None

    best: tuple[int, int, int] | None = None
    shortest = max(1, len(target) - max_syllable_distance)
    longest = len(target) + max_syllable_distance
    last_start = len(meaningful) - shortest
    if max_leading_characters is not None:
        last_start = min(max_leading_characters, last_start)
    for start in range(last_start + 1):
        for length in range(shortest, longest + 1):
            end = start + length
            if end > len(meaningful):
                continue
            candidate = "".join(value for value, _ in meaningful[start:end])
            distance = _distance(_phonemes(candidate), target)
            if distance > max_syllable_distance:
                continue
            score = (distance, start, abs(length - len(target)))
            if best is None or score < best:
                best = score
                best_end = end
    if best is None:
        return None
    original_end = meaningful[best_end - 1][1] + len(meaningful[best_end - 1][0])
    return text[original_end:].lstrip(_LEADING_QUERY_PUNCTUATION)
