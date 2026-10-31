"""Strict Pydantic schema and reachability checks for Scenario DSL v1."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lingxu.domain.enums import MeetingEventType


class ScenarioStatus(StrEnum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


ACTION_WHITELIST = frozenset(
    {
        "publish_host_panel",
        "request_consent",
        "validate_consents",
        "speak_welcome",
        "speak_agenda",
        "speak_current_agenda_item",
        "select_next_speaker",
        "start_speaker_timer",
        "cancel_speaker_timer",
        "open_vote",
        "start_vote_timer",
        "close_vote",
        "speak_vote_result",
        "advance_agenda",
        "speak_closing",
        "finalise_capture",
        "trigger_post_meeting_job",
        "monitor_post_meeting_job",
        "publish_completion",
        "freeze_timers",
        "resume_timers",
        "cancel_timers",
    }
)
EVENT_WHITELIST = frozenset(event.value for event in MeetingEventType)


class ScenarioState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    label: str = Field(min_length=1, max_length=120)
    actions: tuple[str, ...] = ()
    terminal: bool = False


class ScenarioTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_state: str
    event: str
    to_state: str


class ScenarioDefinition(BaseModel):
    """Immutable canonical Scenario snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    key: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    name: str = Field(min_length=1, max_length=120)
    initial_state: str
    states: tuple[ScenarioState, ...]
    transitions: tuple[ScenarioTransition, ...]

    @model_validator(mode="after")
    def validate_graph(self) -> ScenarioDefinition:
        if not self.states:
            raise ValueError("Scenario requires at least one state")
        state_ids = [state.id for state in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("Scenario state ids must be unique")
        known = set(state_ids)
        if self.initial_state not in known:
            raise ValueError("Scenario initial_state is not declared")
        for state in self.states:
            unknown_actions = set(state.actions) - ACTION_WHITELIST
            if unknown_actions:
                raise ValueError(f"Scenario uses unknown actions: {sorted(unknown_actions)}")
        edges: dict[str, set[str]] = {state_id: set() for state_id in state_ids}
        seen_edges: set[tuple[str, str]] = set()
        for transition in self.transitions:
            if transition.from_state not in known or transition.to_state not in known:
                raise ValueError("Scenario transition references an unknown state")
            if transition.event not in EVENT_WHITELIST:
                raise ValueError(f"Scenario uses unknown event: {transition.event}")
            key = (transition.from_state, transition.event)
            if key in seen_edges:
                raise ValueError("Scenario transition is non-deterministic")
            seen_edges.add(key)
            edges[transition.from_state].add(transition.to_state)
        reachable = {self.initial_state}
        frontier = [self.initial_state]
        while frontier:
            current = frontier.pop()
            for target in edges[current] - reachable:
                reachable.add(target)
                frontier.append(target)
        unreachable = known - reachable
        if unreachable:
            raise ValueError(f"Scenario has unreachable states: {sorted(unreachable)}")
        terminal_ids = {state.id for state in self.states if state.terminal}
        if not terminal_ids:
            raise ValueError("Scenario requires at least one terminal state")
        if any(edges[state_id] for state_id in terminal_ids):
            raise ValueError("Scenario terminal states cannot have outgoing transitions")
        return self
