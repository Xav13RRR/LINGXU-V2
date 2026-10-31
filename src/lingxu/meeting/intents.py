"""Local rule-only intent resolver; this module contains no LLM path."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from lingxu.domain.errors import MeetingPermissionError


class EchoDetector(Protocol):
    def is_echo(self, transcript: str) -> bool: ...


class RuleIntent(StrEnum):
    REQUEST_SPEAK = "REQUEST_SPEAK"
    SPEAKER_FINISH = "SPEAKER_FINISH"
    VOTE_YES = "VOTE_YES"
    VOTE_NO = "VOTE_NO"
    VOTE_ABSTAIN = "VOTE_ABSTAIN"
    MEETING_PAUSE = "MEETING_PAUSE"
    MEETING_RESUME = "MEETING_RESUME"
    MEETING_NEXT = "MEETING_NEXT"
    MEETING_BACK = "MEETING_BACK"
    MEETING_SKIP = "MEETING_SKIP"
    MEETING_STOP = "MEETING_STOP"


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    intent: RuleIntent
    actor_user_id: int
    source: str = "VOICE_RULE"
    confidence: float = 1.0


_RULES: tuple[tuple[RuleIntent, re.Pattern[str]], ...] = (
    (
        RuleIntent.REQUEST_SPEAK,
        re.compile(r"^(申请发言|请求发言|我要发言|举手|request (?:to )?speak|raise hand)$", re.I),
    ),
    (
        RuleIntent.SPEAKER_FINISH,
        re.compile(r"^(发言完毕|我说完了|结束发言|finished speaking|i am done)$", re.I),
    ),
    (RuleIntent.VOTE_YES, re.compile(r"^(赞成|同意|支持|yes|approve)$", re.I)),
    (RuleIntent.VOTE_NO, re.compile(r"^(反对|不同意|不支持|no|reject)$", re.I)),
    (RuleIntent.VOTE_ABSTAIN, re.compile(r"^(弃权|保留|abstain)$", re.I)),
    (RuleIntent.MEETING_PAUSE, re.compile(r"^(暂停会议|会议暂停|pause meeting)$", re.I)),
    (RuleIntent.MEETING_RESUME, re.compile(r"^(继续会议|恢复会议|resume meeting)$", re.I)),
    (RuleIntent.MEETING_NEXT, re.compile(r"^(下一步|下一个环节|next)$", re.I)),
    (RuleIntent.MEETING_BACK, re.compile(r"^(返回上一步|上一个环节|go back)$", re.I)),
    (RuleIntent.MEETING_SKIP, re.compile(r"^(跳过|跳过本环节|skip)$", re.I)),
    (RuleIntent.MEETING_STOP, re.compile(r"^(停止会议|结束会议|stop meeting)$", re.I)),
)

_HOST_ONLY = frozenset(
    {
        RuleIntent.MEETING_PAUSE,
        RuleIntent.MEETING_RESUME,
        RuleIntent.MEETING_NEXT,
        RuleIntent.MEETING_BACK,
        RuleIntent.MEETING_SKIP,
        RuleIntent.MEETING_STOP,
    }
)


class RuleIntentResolver:
    """Resolve an exact, auditable phrase list and reject Bot playback echoes."""

    def __init__(self, echo_detector: EchoDetector) -> None:
        self._echo_detector = echo_detector

    def resolve(
        self,
        text: str,
        *,
        actor_user_id: int,
        host_user_id: int,
        source: str = "VOICE_RULE",
    ) -> IntentCandidate | None:
        if self._echo_detector.is_echo(text):
            return None
        normalised = " ".join(text.strip().split())
        for intent, pattern in _RULES:
            if not pattern.fullmatch(normalised):
                continue
            if intent in _HOST_ONLY and actor_user_id != host_user_id:
                raise MeetingPermissionError("only the Meeting host may use this voice control")
            return IntentCandidate(
                intent=intent,
                actor_user_id=actor_user_id,
                source=source,
            )
        return None
