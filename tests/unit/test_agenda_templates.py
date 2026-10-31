# ruff: noqa: RUF001
from __future__ import annotations

import pytest

from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    HostParticipationMode,
    SpeakingOrderStrategy,
)
from lingxu.meeting.agenda_templates import AgendaTemplateKind, build_agenda_template


def test_built_in_templates_compile_to_closed_speaking_policies() -> None:
    standup = build_agenda_template(
        AgendaTemplateKind.STANDUP,
        host_participation=HostParticipationMode.MODERATOR_ONLY,
        mute_enforcement_requested=True,
    )
    assert standup.title == "每日站会"
    assert standup.speaking_policy.mode is AgendaSpeakingMode.ORDERED
    assert standup.speaking_policy.order_strategy is SpeakingOrderStrategy.DISCORD_ALPHABETICAL
    assert standup.speaking_policy.default_turn_seconds == 120
    assert standup.speaking_policy.host_participation is HostParticipationMode.MODERATOR_ONLY
    assert standup.speaking_policy.mute_enforcement_requested
    assert standup.speaking_policy.followup_mode is AgendaFollowupMode.NONE

    alternating = build_agenda_template(
        AgendaTemplateKind.ALTERNATING_SIDES,
        members=(101, 202),
        turn_seconds=90,
        followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
    )
    assert [turn.user_id for turn in alternating.speaking_policy.turns] == [101, 202]
    assert [turn.planned_seconds for turn in alternating.speaking_policy.turns] == [90, 90]
    assert alternating.speaking_policy.followup_mode is AgendaFollowupMode.SUPPLEMENT_TURNS

    presentation = build_agenda_template(
        AgendaTemplateKind.PRESENTATION_Q_AND_A,
        presenters=(303, 404),
    )
    assert presentation.speaking_policy.mode is AgendaSpeakingMode.PRESENTATION
    assert presentation.speaking_policy.presenter_user_ids == (303, 404)
    assert presentation.speaking_policy.followup_mode is AgendaFollowupMode.Q_AND_A


def test_pre_vote_template_and_invalid_template_roles_are_deterministic() -> None:
    draft = build_agenda_template(AgendaTemplateKind.PRE_VOTE_STATEMENTS, title="发布表决")
    assert draft.requires_vote
    assert draft.vote_content == "是否通过「发布表决」？"
    assert draft.vote_duration_seconds == 60

    with pytest.raises(ValueError, match="exactly two distinct"):
        build_agenda_template(AgendaTemplateKind.ALTERNATING_SIDES, members=(101,))
    with pytest.raises(ValueError, match="require one or more presenters"):
        build_agenda_template(AgendaTemplateKind.PRESENTATION)
    with pytest.raises(ValueError, match="do not accept explicit members"):
        build_agenda_template(AgendaTemplateKind.ROUND_ROBIN, members=(101,))
