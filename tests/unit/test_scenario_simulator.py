from datetime import UTC, datetime
from pathlib import Path

import pytest

from lingxu.domain.errors import MeetingPermissionError
from lingxu.scenario.loader import load_scenario
from lingxu.scenario.permissions import (
    DiscordRoleMapper,
    DiscordRoleMapping,
    ScenarioPermission,
    ScenarioRole,
)
from lingxu.scenario.simulator import FakeClock, ScenarioSimulator, SimulatorAPI


def test_fake_clock_simulator_reaches_terminal_and_records_actions() -> None:
    scenario = load_scenario(Path("scenarios/fixed_meeting_v1.yaml")).scenario
    clock = FakeClock(datetime(2026, 7, 27, tzinfo=UTC))
    simulator = ScenarioSimulator(scenario, clock=clock)

    first = simulator.dispatch("MEETING_INITIALISED", role=ScenarioRole.HOST)
    assert first.actions == ("publish_host_panel", "request_consent")
    clock.advance(seconds=5)
    while not simulator.snapshot().terminal:
        transition = next(
            item for item in scenario.transitions if item.from_state == simulator.snapshot().state
        )
        simulator.dispatch(transition.event, role=ScenarioRole.HOST)
    snapshot = simulator.snapshot()
    assert snapshot.state == "PUBLISHED"
    assert snapshot.sequence == 15
    assert snapshot.events[1].occurred_at == datetime(2026, 7, 27, 0, 0, 5, tzinfo=UTC)

    api = SimulatorAPI()
    assert api.create("one", scenario).state == "CREATED"
    assert (
        api.dispatch(
            "one",
            event="MEETING_INITIALISED",
            role=ScenarioRole.HOST,
        ).state
        == "AWAITING_PARTICIPANTS"
    )
    assert api.snapshot("one")["sequence"] == 1


def test_role_permissions_and_discord_id_mapping_are_fail_closed() -> None:
    mapper = DiscordRoleMapper(
        (
            DiscordRoleMapping(10, ScenarioRole.PARTICIPANT),
            DiscordRoleMapping(20, ScenarioRole.FACILITATOR),
        )
    )
    assert (
        mapper.resolve(
            user_id=1,
            session_host_user_id=1,
            discord_role_ids=frozenset(),
        )
        is ScenarioRole.HOST
    )
    assert (
        mapper.resolve(
            user_id=2,
            session_host_user_id=1,
            discord_role_ids=frozenset({10, 20}),
        )
        is ScenarioRole.FACILITATOR
    )
    assert (
        mapper.resolve(
            user_id=3,
            session_host_user_id=1,
            discord_role_ids=frozenset({999}),
        )
        is ScenarioRole.OBSERVER
    )

    scenario = load_scenario(Path("scenarios/fixed_meeting_v1.yaml")).scenario
    simulator = ScenarioSimulator(scenario)
    with pytest.raises(MeetingPermissionError):
        simulator.dispatch("MEETING_INITIALISED", role=ScenarioRole.PARTICIPANT)
    with pytest.raises(MeetingPermissionError) as error:
        simulator.dispatch("MEETING_INITIALISED", role=ScenarioRole.OBSERVER)
    assert ScenarioPermission.CONTROL_SCENARIO.value in str(error.value)
