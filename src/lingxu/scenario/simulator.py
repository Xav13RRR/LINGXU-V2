"""Deterministic in-memory Scenario simulator with an injectable clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from lingxu.scenario.permissions import (
    RolePolicy,
    ScenarioPermission,
    ScenarioRole,
)
from lingxu.scenario.schema import ScenarioDefinition


class FakeClock:
    def __init__(self, now: datetime | None = None) -> None:
        selected = now or datetime(2026, 1, 1, tzinfo=UTC)
        if selected.tzinfo is None:
            raise ValueError("FakeClock requires a timezone-aware timestamp")
        self._now = selected

    def now(self) -> datetime:
        return self._now

    def advance(self, *, seconds: float) -> datetime:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._now += timedelta(seconds=seconds)
        return self._now


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    sequence: int
    event: str
    state_before: str
    state_after: str
    role: ScenarioRole
    actions: tuple[str, ...]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class SimulatorSnapshot:
    scenario_key: str
    state: str
    sequence: int
    terminal: bool
    events: tuple[SimulationEvent, ...]


class ScenarioSimulator:
    def __init__(
        self,
        scenario: ScenarioDefinition,
        *,
        clock: FakeClock | None = None,
        policy: RolePolicy | None = None,
    ) -> None:
        self._scenario = scenario
        self._clock = clock or FakeClock()
        self._policy = policy or RolePolicy()
        self._state = scenario.initial_state
        self._events: list[SimulationEvent] = []
        self._transitions = {
            (transition.from_state, transition.event): transition.to_state
            for transition in scenario.transitions
        }
        self._states = {state.id: state for state in scenario.states}

    @property
    def scenario(self) -> ScenarioDefinition:
        """Expose the immutable Scenario definition for an authorized console projection."""

        return self._scenario

    def dispatch(self, event: str, *, role: ScenarioRole) -> SimulationEvent:
        permission = (
            ScenarioPermission.VOTE
            if event == "VOTE_CAST"
            else ScenarioPermission.SPEAK
            if event == "SPEAK_REQUESTED"
            else ScenarioPermission.CONTROL_SCENARIO
        )
        self._policy.require(role, permission)
        target = self._transitions.get((self._state, event))
        if target is None:
            raise ValueError(f"no Scenario transition from {self._state} for {event}")
        state_before = self._state
        self._state = target
        record = SimulationEvent(
            sequence=len(self._events) + 1,
            event=event,
            state_before=state_before,
            state_after=target,
            role=role,
            actions=self._states[target].actions,
            occurred_at=self._clock.now(),
        )
        self._events.append(record)
        return record

    def snapshot(self) -> SimulatorSnapshot:
        state = self._states[self._state]
        return SimulatorSnapshot(
            scenario_key=self._scenario.key,
            state=self._state,
            sequence=len(self._events),
            terminal=state.terminal,
            events=tuple(self._events),
        )


class SimulatorAPI:
    """Small programmatic API; HTTP exposure remains a later host-console concern."""

    def __init__(self) -> None:
        self._simulators: dict[str, ScenarioSimulator] = {}

    def create(self, simulation_id: str, scenario: ScenarioDefinition) -> SimulatorSnapshot:
        if simulation_id in self._simulators:
            raise ValueError("simulation id already exists")
        simulator = ScenarioSimulator(scenario)
        self._simulators[simulation_id] = simulator
        return simulator.snapshot()

    def dispatch(
        self,
        simulation_id: str,
        *,
        event: str,
        role: ScenarioRole,
    ) -> SimulatorSnapshot:
        simulator = self._simulators[simulation_id]
        simulator.dispatch(event, role=role)
        return simulator.snapshot()

    def snapshot(self, simulation_id: str) -> dict[str, Any]:
        snapshot = self._simulators[simulation_id].snapshot()
        return {
            "scenario_key": snapshot.scenario_key,
            "state": snapshot.state,
            "sequence": snapshot.sequence,
            "terminal": snapshot.terminal,
        }
