import pytest

from lingxu.domain.errors import MeetingPermissionError
from lingxu.meeting.intents import RuleIntent, RuleIntentResolver


class FakeEchoDetector:
    def __init__(self, echo: bool = False) -> None:
        self.echo = echo

    def is_echo(self, transcript: str) -> bool:
        return self.echo


def test_rule_intents_are_exact_and_do_not_need_a_model() -> None:
    resolver = RuleIntentResolver(FakeEchoDetector())

    candidate = resolver.resolve("  申请发言  ", actor_user_id=2, host_user_id=1)
    assert candidate is not None
    assert candidate.intent is RuleIntent.REQUEST_SPEAK
    assert candidate.confidence == 1.0
    assert resolver.resolve("请问我稍后可以发言吗", actor_user_id=2, host_user_id=1) is None


def test_high_risk_voice_control_is_host_only_and_echo_is_ignored() -> None:
    resolver = RuleIntentResolver(FakeEchoDetector())

    with pytest.raises(MeetingPermissionError):
        resolver.resolve("暂停会议", actor_user_id=2, host_user_id=1)
    candidate = resolver.resolve("暂停会议", actor_user_id=1, host_user_id=1)
    assert candidate is not None
    assert candidate.intent is RuleIntent.MEETING_PAUSE

    echo_resolver = RuleIntentResolver(FakeEchoDetector(echo=True))
    assert echo_resolver.resolve("停止会议", actor_user_id=1, host_user_id=1) is None
