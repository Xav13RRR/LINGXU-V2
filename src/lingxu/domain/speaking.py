"""Pure agenda speaking-policy and floor-authority contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from unicodedata import normalize

from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    FloorPhase,
    HostParticipationMode,
    SpeakingOrderStrategy,
)

_POLICY_SCHEMA_VERSION = 1
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "mode",
        "order_strategy",
        "host_participation",
        "turns",
        "default_turn_seconds",
        "presenter_user_ids",
        "followup_mode",
        "mute_enforcement_requested",
        "prep_seconds",
    }
)


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _bounded_seconds(value: object, *, field_name: str) -> int:
    seconds = _positive_integer(value, field_name=field_name)
    if seconds > 14_400:
        raise ValueError(f"{field_name} must not exceed 14400 seconds")
    return seconds


@dataclass(frozen=True, slots=True)
class SpeakerTurnPlan:
    """One immutable ordered turn and its planned floor occupancy."""

    user_id: int
    planned_seconds: int

    def __post_init__(self) -> None:
        _positive_integer(self.user_id, field_name="turn user_id")
        _bounded_seconds(self.planned_seconds, field_name="planned_seconds")

    def to_payload(self) -> dict[str, int]:
        return {"user_id": self.user_id, "planned_seconds": self.planned_seconds}

    @classmethod
    def from_payload(cls, payload: object) -> SpeakerTurnPlan:
        if not isinstance(payload, Mapping) or set(payload) != {"user_id", "planned_seconds"}:
            raise ValueError("turn payload has an invalid shape")
        return cls(
            user_id=_positive_integer(payload["user_id"], field_name="turn user_id"),
            planned_seconds=_bounded_seconds(
                payload["planned_seconds"], field_name="planned_seconds"
            ),
        )


@dataclass(frozen=True, slots=True)
class AgendaSpeakingPolicy:
    """Closed schema persisted with one agenda item."""

    mode: AgendaSpeakingMode
    order_strategy: SpeakingOrderStrategy = SpeakingOrderStrategy.NONE
    host_participation: HostParticipationMode = HostParticipationMode.AUTO
    turns: tuple[SpeakerTurnPlan, ...] = ()
    default_turn_seconds: int | None = None
    presenter_user_ids: tuple[int, ...] = ()
    followup_mode: AgendaFollowupMode = AgendaFollowupMode.NONE
    mute_enforcement_requested: bool = False
    prep_seconds: int = 10

    def __post_init__(self) -> None:
        if (
            isinstance(self.prep_seconds, bool)
            or not isinstance(self.prep_seconds, int)
            or not 0 <= self.prep_seconds <= 300
        ):
            raise ValueError("prep_seconds must be between 0 and 300")
        if not isinstance(self.mute_enforcement_requested, bool):
            raise ValueError("mute_enforcement_requested must be boolean")

        turn_user_ids = tuple(turn.user_id for turn in self.turns)
        if len(turn_user_ids) != len(set(turn_user_ids)):
            raise ValueError("ordered turn user IDs must be unique")
        if len(self.presenter_user_ids) != len(set(self.presenter_user_ids)):
            raise ValueError("presenter user IDs must be unique")
        for user_id in self.presenter_user_ids:
            _positive_integer(user_id, field_name="presenter user_id")

        if self.mode is AgendaSpeakingMode.FREE:
            if (
                self.order_strategy is not SpeakingOrderStrategy.NONE
                or self.turns
                or self.default_turn_seconds is not None
                or self.presenter_user_ids
                or self.followup_mode is not AgendaFollowupMode.NONE
                or self.mute_enforcement_requested
            ):
                raise ValueError("free speaking policy cannot contain controlled-floor settings")
            return

        if self.mode is AgendaSpeakingMode.ORDERED:
            if self.presenter_user_ids:
                raise ValueError("ordered speaking policy cannot contain presenters")
            if self.order_strategy is SpeakingOrderStrategy.EXPLICIT:
                if not self.turns or self.default_turn_seconds is not None:
                    raise ValueError("explicit order requires turns and no default duration")
                return
            if self.order_strategy is SpeakingOrderStrategy.DISCORD_ALPHABETICAL:
                if self.turns or self.default_turn_seconds is None:
                    raise ValueError("alphabetical order requires only a default duration")
                _bounded_seconds(
                    self.default_turn_seconds,
                    field_name="default_turn_seconds",
                )
                return
            raise ValueError("ordered speaking policy requires an order strategy")

        if self.mode is AgendaSpeakingMode.PRESENTATION:
            if (
                self.order_strategy is not SpeakingOrderStrategy.NONE
                or self.turns
                or self.default_turn_seconds is not None
                or not self.presenter_user_ids
            ):
                raise ValueError("presentation policy requires presenters only")
            return

        raise ValueError("unknown agenda speaking mode")

    @classmethod
    def free(cls) -> AgendaSpeakingPolicy:
        return cls(mode=AgendaSpeakingMode.FREE)

    @classmethod
    def ordered_explicit(
        cls,
        *,
        turns: tuple[SpeakerTurnPlan, ...],
        followup_mode: AgendaFollowupMode = AgendaFollowupMode.NONE,
        mute_enforcement_requested: bool = False,
        prep_seconds: int = 10,
    ) -> AgendaSpeakingPolicy:
        return cls(
            mode=AgendaSpeakingMode.ORDERED,
            order_strategy=SpeakingOrderStrategy.EXPLICIT,
            turns=turns,
            followup_mode=followup_mode,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=prep_seconds,
        )

    @classmethod
    def ordered_alphabetical(
        cls,
        *,
        default_turn_seconds: int,
        host_participation: HostParticipationMode = HostParticipationMode.AUTO,
        followup_mode: AgendaFollowupMode = AgendaFollowupMode.NONE,
        mute_enforcement_requested: bool = False,
        prep_seconds: int = 10,
    ) -> AgendaSpeakingPolicy:
        return cls(
            mode=AgendaSpeakingMode.ORDERED,
            order_strategy=SpeakingOrderStrategy.DISCORD_ALPHABETICAL,
            host_participation=host_participation,
            default_turn_seconds=default_turn_seconds,
            followup_mode=followup_mode,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=prep_seconds,
        )

    @classmethod
    def presentation(
        cls,
        *,
        presenter_user_ids: tuple[int, ...],
        followup_mode: AgendaFollowupMode = AgendaFollowupMode.NONE,
        mute_enforcement_requested: bool = False,
        prep_seconds: int = 10,
    ) -> AgendaSpeakingPolicy:
        return cls(
            mode=AgendaSpeakingMode.PRESENTATION,
            presenter_user_ids=presenter_user_ids,
            followup_mode=followup_mode,
            mute_enforcement_requested=mute_enforcement_requested,
            prep_seconds=prep_seconds,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _POLICY_SCHEMA_VERSION,
            "mode": self.mode.value,
            "order_strategy": self.order_strategy.value,
            "host_participation": self.host_participation.value,
            "turns": [turn.to_payload() for turn in self.turns],
            "default_turn_seconds": self.default_turn_seconds,
            "presenter_user_ids": list(self.presenter_user_ids),
            "followup_mode": self.followup_mode.value,
            "mute_enforcement_requested": self.mute_enforcement_requested,
            "prep_seconds": self.prep_seconds,
        }

    @classmethod
    def from_payload(cls, payload: object | None) -> AgendaSpeakingPolicy:
        if payload is None:
            return cls.free()
        if not isinstance(payload, Mapping) or set(payload) != _POLICY_KEYS:
            raise ValueError("speaking policy payload has an invalid shape")
        schema_version = payload["schema_version"]
        if isinstance(schema_version, bool) or schema_version != _POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported speaking policy schema version")
        try:
            mode = AgendaSpeakingMode(payload["mode"])
            order_strategy = SpeakingOrderStrategy(payload["order_strategy"])
            host_participation = HostParticipationMode(payload["host_participation"])
            followup_mode = AgendaFollowupMode(payload["followup_mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError("speaking policy contains an unknown enum") from exc
        raw_turns = payload["turns"]
        raw_presenters = payload["presenter_user_ids"]
        if not isinstance(raw_turns, list) or not isinstance(raw_presenters, list):
            raise ValueError("speaking policy lists have an invalid shape")
        raw_default = payload["default_turn_seconds"]
        default_turn_seconds = (
            None
            if raw_default is None
            else _bounded_seconds(raw_default, field_name="default_turn_seconds")
        )
        mute_requested = payload["mute_enforcement_requested"]
        if not isinstance(mute_requested, bool):
            raise ValueError("mute_enforcement_requested must be boolean")
        prep_seconds = payload["prep_seconds"]
        if isinstance(prep_seconds, bool) or not isinstance(prep_seconds, int):
            raise ValueError("prep_seconds must be an integer")
        return cls(
            mode=mode,
            order_strategy=order_strategy,
            host_participation=host_participation,
            turns=tuple(SpeakerTurnPlan.from_payload(item) for item in raw_turns),
            default_turn_seconds=default_turn_seconds,
            presenter_user_ids=tuple(
                _positive_integer(item, field_name="presenter user_id") for item in raw_presenters
            ),
            followup_mode=followup_mode,
            mute_enforcement_requested=mute_requested,
            prep_seconds=prep_seconds,
        )


@dataclass(frozen=True, slots=True)
class FloorParticipant:
    """Identity fields used only to snapshot deterministic alphabetical turns."""

    user_id: int
    display_name: str | None

    def __post_init__(self) -> None:
        _positive_integer(self.user_id, field_name="participant user_id")


@dataclass(frozen=True, slots=True)
class FloorContext:
    """Complete deterministic input for one floor decision."""

    policy: AgendaSpeakingPolicy
    present_user_ids: tuple[int, ...]
    host_user_id: int
    current_speaker_user_id: int | None = None
    current_questioner_user_id: int | None = None
    exception_user_ids: tuple[int, ...] = ()
    phase: FloorPhase = FloorPhase.MAIN
    host_intervention_active: bool = False


def can_control_session(*, actor_user_id: int, host_user_id: int) -> bool:
    """Return host control authority without consulting the floor policy."""

    return actor_user_id == host_user_id


def planned_turns(
    policy: AgendaSpeakingPolicy,
    participants: tuple[FloorParticipant, ...],
    host_user_id: int,
) -> tuple[SpeakerTurnPlan, ...]:
    """Build the ordered activation snapshot without relying on Discord cache order."""

    if policy.mode is not AgendaSpeakingMode.ORDERED:
        return ()
    if policy.order_strategy is SpeakingOrderStrategy.EXPLICIT:
        return policy.turns
    participant_user_ids = tuple(participant.user_id for participant in participants)
    if len(participant_user_ids) != len(set(participant_user_ids)):
        raise ValueError("alphabetical participant user IDs must be unique")
    duration = policy.default_turn_seconds
    if duration is None:  # protected by policy validation
        raise ValueError("alphabetical policy has no default duration")
    eligible = (
        participant
        for participant in participants
        if not (
            participant.user_id == host_user_id
            and policy.host_participation is HostParticipationMode.MODERATOR_ONLY
        )
    )
    ordered = sorted(
        eligible,
        key=lambda participant: (
            normalize("NFKC", participant.display_name or "").casefold(),
            participant.user_id,
        ),
    )
    return tuple(
        SpeakerTurnPlan(user_id=participant.user_id, planned_seconds=duration)
        for participant in ordered
    )


def _ordered_present(values: tuple[int, ...], present: frozenset[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(value for value in values if value in present))


def floor_user_ids(context: FloorContext) -> tuple[int, ...]:
    """Compute the current floor; host identity grants no implicit speaking bypass."""

    present = frozenset(context.present_user_ids)
    if context.host_intervention_active:
        return (context.host_user_id,) if context.host_user_id in present else ()

    if context.phase is FloorPhase.FOLLOWUP:
        if context.policy.followup_mode is AgendaFollowupMode.Q_AND_A:
            base = context.policy.presenter_user_ids
            if context.current_questioner_user_id is not None:
                base += (context.current_questioner_user_id,)
        elif context.policy.followup_mode is AgendaFollowupMode.SUPPLEMENT_TURNS:
            base = (
                (context.current_speaker_user_id,)
                if context.current_speaker_user_id is not None
                else ()
            )
        else:
            base = ()
    elif context.policy.mode is AgendaSpeakingMode.FREE:
        base = context.present_user_ids
    elif context.policy.mode is AgendaSpeakingMode.ORDERED:
        base = (
            (context.current_speaker_user_id,)
            if context.current_speaker_user_id is not None
            else ()
        )
    else:
        base = context.policy.presenter_user_ids

    return _ordered_present(base + context.exception_user_ids, present)
