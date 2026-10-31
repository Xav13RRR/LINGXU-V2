from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaStatus,
    HostParticipationMode,
)
from lingxu.domain.models import AgendaItemRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan
from lingxu.worker.speaking_plans import project_authoritative_speaking_plans


def _agenda(
    position: int,
    title: str,
    policy: AgendaSpeakingPolicy,
    *,
    duration_minutes: int | None = 10,
) -> AgendaItemRecord:
    return AgendaItemRecord(
        id=f"agenda-{position}",
        session_id="session-1",
        position=position,
        title=title,
        status=AgendaStatus.PENDING,
        duration_minutes=duration_minutes,
        requires_vote=False,
        speaking_policy=policy,
    )


def test_all_agenda_modes_project_closed_privacy_filtered_plan_snapshots() -> None:
    plans = project_authoritative_speaking_plans(
        (
            _agenda(1, "自由", AgendaSpeakingPolicy.free()),
            _agenda(
                2,
                "指定顺序",
                AgendaSpeakingPolicy.ordered_explicit(
                    turns=(
                        SpeakerTurnPlan(user_id=10, planned_seconds=120),
                        SpeakerTurnPlan(user_id=20, planned_seconds=90),
                    ),
                    followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
                    mute_enforcement_requested=True,
                    prep_seconds=5,
                ),
            ),
            _agenda(
                3,
                "首字母顺序",
                AgendaSpeakingPolicy.ordered_alphabetical(
                    default_turn_seconds=60,
                    host_participation=HostParticipationMode.MODERATOR_ONLY,
                    followup_mode=AgendaFollowupMode.Q_AND_A,
                    prep_seconds=0,
                ),
            ),
            _agenda(
                4,
                "演讲",
                AgendaSpeakingPolicy.presentation(
                    presenter_user_ids=(10, 20),
                    mute_enforcement_requested=True,
                    prep_seconds=30,
                ),
                duration_minutes=15,
            ),
        ),
        labels={10: "主持人甲"},
    )

    assert [value.mode for value in plans] == ["FREE", "ORDERED", "ORDERED", "PRESENTATION"]
    assert plans[1].turns[0].user_id == 10
    assert plans[1].turns[0].name == "主持人甲 (user:10)"
    assert plans[1].turns[1].user_id is None
    assert plans[1].turns[1].name == "未获纪要授权成员"
    assert plans[1].followup_mode == "SUPPLEMENT_TURNS"
    assert plans[1].mute_enforcement_requested is True
    assert plans[2].order_strategy == "DISCORD_ALPHABETICAL"
    assert plans[2].host_participation == "MODERATOR_ONLY"
    assert plans[2].default_turn_seconds == 60
    assert plans[2].prep_seconds == 0
    assert plans[3].presentation_seconds == 900
    assert [value.user_id for value in plans[3].presenters] == [10, None]
