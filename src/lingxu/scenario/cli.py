"""Scenario validation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingxu.scenario.loader import load_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    loaded = load_scenario(args.path)
    print(
        json.dumps(
            {
                "key": loaded.scenario.key,
                "name": loaded.scenario.name,
                "states": len(loaded.scenario.states),
                "transitions": len(loaded.scenario.transitions),
                "source_sha256": loaded.source_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
