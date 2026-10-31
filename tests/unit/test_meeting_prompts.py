from lingxu.domain.enums import AgendaStatus
from lingxu.domain.models import AgendaItemRecord
from lingxu.meeting.prompts import agenda_intro_prompt, agenda_language
from lingxu.tts.factory import COMMON_PROCESS_PROMPTS


def _item(*, title: str, language: str = "auto", duration: int | None = 5) -> AgendaItemRecord:
    return AgendaItemRecord(
        id="agenda-1",
        session_id="session-1",
        position=1,
        title=title,
        status=AgendaStatus.ACTIVE,
        duration_minutes=duration,
        requires_vote=False,
        language=language,
    )


def test_agenda_prompt_reads_title_and_duration_in_detected_language() -> None:
    english = _item(title="Release plan")
    japanese = _item(title="次の四半期")

    assert agenda_language(english) == "en"
    assert "Release plan" in agenda_intro_prompt(english)
    assert "5 minutes" in agenda_intro_prompt(english)
    assert agenda_language(japanese) == "ja"
    assert "次の四半期" in agenda_intro_prompt(japanese)


def test_untimed_agenda_explains_manual_host_advance() -> None:
    prompt = agenda_intro_prompt(_item(title="开放讨论", duration=None))

    assert "开放讨论" in prompt
    assert "无固定时长" in prompt
    assert "主持人" in prompt


def test_fixed_process_prompts_keep_only_the_sixty_second_reminder() -> None:
    assert "本次会议将于十秒后开始。" in COMMON_PROCESS_PROMPTS
    assert "会议将于十秒后开始。" not in COMMON_PROCESS_PROMPTS
    assert "当前流程剩余约六十秒。" in COMMON_PROCESS_PROMPTS
    assert "发言时间还剩十秒。" not in COMMON_PROCESS_PROMPTS
    assert "当前流程剩余约十秒。" not in COMMON_PROCESS_PROMPTS
