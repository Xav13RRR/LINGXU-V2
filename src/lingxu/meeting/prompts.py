"""Deterministic multilingual Meeting process prompts."""

from __future__ import annotations

import re

from lingxu.domain.models import AgendaItemRecord

_HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def agenda_language(item: AgendaItemRecord) -> str:
    """Return the configured language or a conservative title-based detection."""

    if item.language != "auto":
        return item.language
    title = item.title
    if _HIRAGANA_KATAKANA.search(title):
        return "ja"
    if _HANGUL.search(title):
        return "ko"
    if _CJK.search(title):
        return "zh"
    if _LATIN.search(title):
        return "en"
    return "zh"


def agenda_intro_prompt(item: AgendaItemRecord) -> str:
    """Announce one agenda title and its automatic/manual timing contract."""

    language = agenda_language(item)
    minutes = item.duration_minutes
    if language == "en":
        timing = (
            f"This stage is scheduled for {minutes} minutes."
            if minutes is not None
            else (
                "This stage has no fixed duration. "
                "The host should advance it manually when complete."
            )
        )
        return f"Now entering agenda item: {item.title}. {timing}"
    if language == "ja":
        timing = (
            f"この進行の予定時間は{minutes}分です。"
            if minutes is not None
            else "この進行には固定時間がありません。完了後、主催者が手動で次へ進んでください。"
        )
        return f"議題「{item.title}」に入ります。{timing}"
    if language == "ko":
        timing = (
            f"이 순서의 예정 시간은 {minutes}분입니다."
            if minutes is not None
            else (
                "이 순서에는 정해진 시간이 없습니다. "
                "완료 후 진행자가 수동으로 다음 순서로 이동해 주세요."
            )
        )
        return f"안건 '{item.title}'을 시작합니다. {timing}"
    if language == "yue":
        timing = (
            f"呢个流程预计需时{minutes}分钟。"
            if minutes is not None
            else "呢个流程冇固定时长, 请主持人喺流程完成之后手动进入下一流程。"
        )
        return f"而家进入议程: {item.title}。{timing}"
    timing = (
        f"该流程预计持续{minutes}分钟。"
        if minutes is not None
        else "该流程无固定时长, 请主持人在流程完成后手动进入下一流程。"
    )
    return f"现在进入议程: {item.title}。{timing}"
