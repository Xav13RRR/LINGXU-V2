import pytest

from lingxu.domain.enums import (
    AgendaFollowupMode,
    AgendaSpeakingMode,
    FloorPhase,
    HostParticipationMode,
)
from lingxu.domain.speaking import (
    AgendaSpeakingPolicy,
    FloorContext,
    FloorParticipant,
    SpeakerTurnPlan,
    can_control_session,
    floor_user_ids,
    planned_turns,
)


def test_legacy_policy_is_free_and_schema_one_round_trips() -> None:
    legacy = AgendaSpeakingPolicy.from_payload(None)

    assert legacy == AgendaSpeakingPolicy.free()
    assert legacy.mode is AgendaSpeakingMode.FREE
    assert AgendaSpeakingPolicy.from_payload(legacy.to_payload()) == legacy
    assert legacy.to_payload() == {
        "schema_version": 1,
        "mode": "FREE",
        "order_strategy": "NONE",
        "host_participation": "AUTO",
        "turns": [],
        "default_turn_seconds": None,
        "presenter_user_ids": [],
        "followup_mode": "NONE",
        "mute_enforcement_requested": False,
        "prep_seconds": 10,
    }


def test_explicit_order_keeps_per_user_durations_and_rejects_duplicates() -> None:
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(
            SpeakerTurnPlan(user_id=30, planned_seconds=90),
            SpeakerTurnPlan(user_id=10, planned_seconds=120),
        ),
        followup_mode=AgendaFollowupMode.SUPPLEMENT_TURNS,
        mute_enforcement_requested=True,
    )

    assert planned_turns(policy, participants=(), host_user_id=10) == policy.turns
    assert AgendaSpeakingPolicy.from_payload(policy.to_payload()) == policy
    with pytest.raises(ValueError, match="unique"):
        AgendaSpeakingPolicy.ordered_explicit(
            turns=(
                SpeakerTurnPlan(user_id=10, planned_seconds=60),
                SpeakerTurnPlan(user_id=10, planned_seconds=90),
            )
        )


def test_alphabetical_order_is_nfkc_casefolded_and_host_policy_is_explicit() -> None:
    participants = (
        FloorParticipant(user_id=30, display_name="\uff22eta"),
        FloorParticipant(user_id=20, display_name="alpha"),
        FloorParticipant(user_id=10, display_name="Alpha"),
        FloorParticipant(user_id=40, display_name=None),
    )
    automatic = AgendaSpeakingPolicy.ordered_alphabetical(default_turn_seconds=75)
    moderator_only = AgendaSpeakingPolicy.ordered_alphabetical(
        default_turn_seconds=75,
        host_participation=HostParticipationMode.MODERATOR_ONLY,
    )

    assert [turn.user_id for turn in planned_turns(automatic, participants, host_user_id=10)] == [
        40,
        10,
        20,
        30,
    ]
    assert [
        turn.user_id for turn in planned_turns(moderator_only, participants, host_user_id=10)
    ] == [40, 20, 30]
    assert {turn.planned_seconds for turn in planned_turns(automatic, participants, 10)} == {75}
    with pytest.raises(ValueError, match="unique"):
        planned_turns(
            automatic,
            (*participants, FloorParticipant(user_id=20, display_name="renamed")),
            host_user_id=10,
        )


def test_host_control_never_implies_presentation_floor_access() -> None:
    policy = AgendaSpeakingPolicy.presentation(
        presenter_user_ids=(200, 300),
        mute_enforcement_requested=True,
    )
    context = FloorContext(
        policy=policy,
        present_user_ids=(100, 200, 300),
        host_user_id=100,
    )

    assert can_control_session(actor_user_id=100, host_user_id=100)
    assert not can_control_session(actor_user_id=200, host_user_id=100)
    assert floor_user_ids(context) == (200, 300)
    assert 100 not in floor_user_ids(context)

    host_presenter = AgendaSpeakingPolicy.presentation(presenter_user_ids=(100, 200))
    assert floor_user_ids(
        FloorContext(
            policy=host_presenter,
            present_user_ids=(100, 200, 300),
            host_user_id=100,
        )
    ) == (100, 200)


def test_host_intervention_overrides_floor_while_exception_is_parallel() -> None:
    policy = AgendaSpeakingPolicy.ordered_explicit(
        turns=(SpeakerTurnPlan(user_id=200, planned_seconds=60),),
        followup_mode=AgendaFollowupMode.Q_AND_A,
    )
    normal = FloorContext(
        policy=policy,
        present_user_ids=(100, 200, 300),
        host_user_id=100,
        current_speaker_user_id=200,
        exception_user_ids=(300,),
    )
    intervention = FloorContext(
        policy=policy,
        present_user_ids=(100, 200, 300),
        host_user_id=100,
        current_speaker_user_id=200,
        exception_user_ids=(300,),
        host_intervention_active=True,
    )

    assert floor_user_ids(normal) == (200, 300)
    assert floor_user_ids(intervention) == (100,)


def test_q_and_a_floor_contains_presenters_and_current_questioner_only() -> None:
    policy = AgendaSpeakingPolicy.presentation(
        presenter_user_ids=(100, 200),
        followup_mode=AgendaFollowupMode.Q_AND_A,
    )

    assert floor_user_ids(
        FloorContext(
            policy=policy,
            present_user_ids=(100, 200, 300, 400),
            host_user_id=400,
            phase=FloorPhase.FOLLOWUP,
            current_questioner_user_id=300,
        )
    ) == (100, 200, 300)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2, "mode": "FREE"},
        {"schema_version": 1, "mode": "UNKNOWN"},
    ],
)
def test_malformed_or_unknown_persisted_policy_fails_closed(payload: object) -> None:
    with pytest.raises(ValueError):
        AgendaSpeakingPolicy.from_payload(payload)
