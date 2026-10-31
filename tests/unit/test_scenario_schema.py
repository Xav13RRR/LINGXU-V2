from pathlib import Path

import pytest
from pydantic import ValidationError

from lingxu.scenario.loader import load_scenario, load_scenario_bytes


def test_fixed_meeting_yaml_is_strict_reachable_and_canonical() -> None:
    loaded = load_scenario(Path("scenarios/fixed_meeting_v1.yaml"))

    assert loaded.scenario.key == "fixed-meeting"
    assert loaded.scenario.initial_state == "CREATED"
    assert len(loaded.scenario.states) == 16
    assert len(loaded.scenario.transitions) == 15
    assert len(loaded.source_sha256) == 64
    assert b'"schema_version":1' in loaded.canonical_json


@pytest.mark.parametrize(
    "source,match",
    [
        (
            b"""
schema_version: 1
key: invalid-action
name: Invalid
initial_state: START
states:
  - {id: START, label: Start, actions: [run_shell]}
  - {id: END, label: End, terminal: true}
transitions:
  - {from_state: START, event: STATE_ADVANCED, to_state: END}
""",
            "unknown actions",
        ),
        (
            b"""
schema_version: 1
key: unreachable-state
name: Invalid
initial_state: START
states:
  - {id: START, label: Start}
  - {id: END, label: End, terminal: true}
transitions: []
""",
            "unreachable states",
        ),
        (
            b"""
schema_version: 1
key: invalid-event
name: Invalid
initial_state: START
states:
  - {id: START, label: Start}
  - {id: END, label: End, terminal: true}
transitions:
  - {from_state: START, event: RUN_ARBITRARY_CODE, to_state: END}
""",
            "unknown event",
        ),
    ],
)
def test_scenario_rejects_unknown_or_unreachable_graph(source: bytes, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        load_scenario_bytes(source)
