"""Project privacy-filtered agenda speaking-policy snapshots for Meeting minutes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from lingxu.domain.enums import AgendaSpeakingMode
from lingxu.domain.models import AgendaItemRecord
from lingxu.worker.minutes_models import (
    MinutesSpeakingPlan,
    MinutesSpeakingPlanParticipant,
    MinutesSpeakingPlanTurn,
)

_WITHHELD_NAME = "未获纪要授权成员"


def _identity(user_id: int, labels: Mapping[int, str]) -> tuple[int | None, str]:
    label = labels.get(user_id)
    if label is None:
        return None, _WITHHELD_NAME
    if not label or len(label) > 256 or label == "聆序":
        raise ValueError("speaking-plan display identity is invalid")
    return user_id, f"{label} (user:{user_id})"


def project_authoritative_speaking_plans(
    agendas: Sequence[AgendaItemRecord],
    *,
    labels: Mapping[int, str],
) -> tuple[MinutesSpeakingPlan, ...]:
    """Copy every closed policy without inferring execution from transcript content."""

    results: list[MinutesSpeakingPlan] = []
    seen_agenda_ids: set[str] = set()
    for agenda in sorted(agendas, key=lambda item: (item.position, item.id)):
        if agenda.id in seen_agenda_ids:
            raise ValueError("speaking-plan agenda identity is duplicated")
        seen_agenda_ids.add(agenda.id)
        policy = agenda.speaking_policy
        turns = tuple(
            MinutesSpeakingPlanTurn(
                user_id=public_user_id,
                name=name,
                planned_seconds=turn.planned_seconds,
            )
            for turn in policy.turns
            for public_user_id, name in (_identity(turn.user_id, labels),)
        )
        presenters = tuple(
            MinutesSpeakingPlanParticipant(user_id=public_user_id, name=name)
            for user_id in policy.presenter_user_ids
            for public_user_id, name in (_identity(user_id, labels),)
        )
        presentation_seconds = (
            agenda.duration_minutes * 60
            if policy.mode is AgendaSpeakingMode.PRESENTATION
            and agenda.duration_minutes is not None
            else None
        )
        results.append(
            MinutesSpeakingPlan(
                agenda_item_id=agenda.id,
                mode=policy.mode.value,
                order_strategy=policy.order_strategy.value,
                host_participation=policy.host_participation.value,
                turns=turns,
                default_turn_seconds=policy.default_turn_seconds,
                presenters=presenters,
                presentation_seconds=presentation_seconds,
                followup_mode=policy.followup_mode.value,
                mute_enforcement_requested=policy.mute_enforcement_requested,
                prep_seconds=policy.prep_seconds,
            )
        )
    return tuple(results)
