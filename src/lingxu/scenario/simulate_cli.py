"""Run the fixed Scenario to its terminal state without Discord."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingxu.scenario.loader import load_scenario
from lingxu.scenario.permissions import ScenarioRole
from lingxu.scenario.simulator import ScenarioSimulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    scenario = load_scenario(args.path).scenario
    simulator = ScenarioSimulator(scenario)
    while not simulator.snapshot().terminal:
        transition = next(
            item for item in scenario.transitions if item.from_state == simulator.snapshot().state
        )
        simulator.dispatch(transition.event, role=ScenarioRole.HOST)
    snapshot = simulator.snapshot()
    print(
        json.dumps(
            {
                "scenario_key": snapshot.scenario_key,
                "state": snapshot.state,
                "sequence": snapshot.sequence,
                "terminal": snapshot.terminal,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
