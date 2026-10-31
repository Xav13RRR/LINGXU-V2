# ruff: noqa: RUF001
"""Built-in agenda templates that compile to the closed speaking-policy schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lingxu.domain.enums import AgendaFollowupMode, HostParticipationMode
from lingxu.domain.speaking import AgendaSpeakingPolicy, SpeakerTurnPlan

_SUPPORTED_LANGUAGES = frozenset({"auto", "zh", "yue", "en", "ja", "ko"})


class AgendaTemplateKind(StrEnum):
    ROUND_ROBIN = "ROUND_ROBIN"
    STANDUP = "STANDUP"
    PRE_VOTE_STATEMENTS = "PRE_VOTE_STATEMENTS"
    ALTERNATING_SIDES = "ALTERNATING_SIDES"
    PRESENTATION = "PRESENTATION"
    PRESENTATION_Q_AND_A = "PRESENTATION_Q_AND_A"


@dataclass(frozen=True, slots=True)
class AgendaTemplateDraft:
    """One fully validated agenda item ready for deterministic persistence."""

    kind: AgendaTemplateKind
    title: str
    duration_minutes: int
    requires_vote: bool
    vote_content: str | None
    vote_duration_seconds: int | None
    language: str
    speaking_policy: AgendaSpeakingPolicy


_DEFAULTS: dict[AgendaTemplateKind, tuple[str, int, int]] = {
    AgendaTemplateKind.ROUND_ROBIN: ("圆桌汇报", 15, 180),
    AgendaTemplateKind.STANDUP: ("每日站会", 15, 120),
    AgendaTemplateKind.PRE_VOTE_STATEMENTS: ("表决前陈述", 20, 120),
    AgendaTemplateKind.ALTERNATING_SIDES: ("正反方轮流陈述", 10, 300),
    AgendaTemplateKind.PRESENTATION: ("演讲", 15, 0),
    AgendaTemplateKind.PRESENTATION_Q_AND_A: ("演讲与统一答疑", 20, 0),
}


def _bounded(value: int, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def build_agenda_template(
    kind: AgendaTemplateKind | str,
    *,
    title: str | None = None,
    duration_minutes: int | None = None,
    turn_seconds: int | None = None,
    members: tuple[int, ...] = (),
    presenters: tuple[int, ...] = (),
    followup_mode: AgendaFollowupMode | None = None,
    host_participation: HostParticipationMode = HostParticipationMode.AUTO,
    mute_enforcement_requested: bool = False,
    prep_seconds: int = 10,
    vote_content: str | None = None,
    vote_duration_seconds: int | None = None,
    language: str = "auto",
) -> AgendaTemplateDraft:
    """Compile one named template without consulting Discord names or mutable state."""

    selected_kind = AgendaTemplateKind(kind)
    default_title, default_duration, default_turn = _DEFAULTS[selected_kind]
    selected_title = " ".join((title or default_title).split())
    if not 1 <= len(selected_title) <= 200:
        raise ValueError("template title must contain between 1 and 200 characters")
    selected_duration = _bounded(
        duration_minutes if duration_minutes is not None else default_duration,
        field="duration_minutes",
        minimum=1,
        maximum=480,
    )
    selected_prep = _bounded(prep_seconds, field="prep_seconds", minimum=0, maximum=300)
    if not isinstance(mute_enforcement_requested, bool):
        raise ValueError("mute_enforcement_requested must be boolean")
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError("template language is unsupported")

    requires_vote = selected_kind is AgendaTemplateKind.PRE_VOTE_STATEMENTS
    selected_vote_content: str | None = None
    selected_vote_seconds: int | None = None
    if requires_vote:
        selected_vote_content = " ".join(
            (vote_content or f"是否通过「{selected_title}」？").split()
        )
        if not 1 <= len(selected_vote_content) <= 300:
            raise ValueError("vote_content must contain between 1 and 300 characters")
        selected_vote_seconds = _bounded(
            vote_duration_seconds if vote_duration_seconds is not None else 60,
            field="vote_duration_seconds",
            minimum=10,
            maximum=3600,
        )
    elif vote_content is not None or vote_duration_seconds is not None:
        raise ValueError("vote settings are only valid for PRE_VOTE_STATEMENTS")

    if selected_kind in {
        AgendaTemplateKind.ROUND_ROBIN,
        AgendaTemplateKind.STANDUP,
        AgendaTemplateKind.PRE_VOTE_STATEMENTS,
    }:
        if members or presenters:
            raise ValueError("alphabetical templates do not accept explicit members")
        selected_turn = _bounded(
            turn_seconds if turn_seconds is not None else default_turn,
            field="turn_seconds",
            minimum=1,
            maximum=14_400,
        )
        policy = AgendaSpeakingPolicy.ordered_alphabetical(
            default_turn_seconds=selected_turn,
            host_participation=host_participation,
            followup_mode=followup_mode or AgendaFollowupMode.NONE,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=selected_prep,
        )
    elif selected_kind is AgendaTemplateKind.ALTERNATING_SIDES:
        if presenters or len(members) != 2 or len(set(members)) != 2:
            raise ValueError("ALTERNATING_SIDES requires exactly two distinct members")
        selected_turn = _bounded(
            turn_seconds if turn_seconds is not None else default_turn,
            field="turn_seconds",
            minimum=1,
            maximum=14_400,
        )
        policy = AgendaSpeakingPolicy.ordered_explicit(
            turns=tuple(
                SpeakerTurnPlan(user_id=user_id, planned_seconds=selected_turn)
                for user_id in members
            ),
            followup_mode=followup_mode or AgendaFollowupMode.NONE,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=selected_prep,
        )
    else:
        if members or turn_seconds is not None or not presenters:
            raise ValueError("presentation templates require one or more presenters")
        default_followup = (
            AgendaFollowupMode.Q_AND_A
            if selected_kind is AgendaTemplateKind.PRESENTATION_Q_AND_A
            else AgendaFollowupMode.NONE
        )
        policy = AgendaSpeakingPolicy.presentation(
            presenter_user_ids=presenters,
            followup_mode=followup_mode or default_followup,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=selected_prep,
        )

    return AgendaTemplateDraft(
        kind=selected_kind,
        title=selected_title,
        duration_minutes=selected_duration,
        requires_vote=requires_vote,
        vote_content=selected_vote_content,
        vote_duration_seconds=selected_vote_seconds,
        language=language,
        speaking_policy=policy,
    )
