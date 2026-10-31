"""Safe YAML/JSON loading and canonical source hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lingxu.scenario.schema import ScenarioDefinition


@dataclass(frozen=True, slots=True)
class LoadedScenario:
    scenario: ScenarioDefinition
    source_sha256: str
    canonical_json: bytes


def _load_mapping(payload: bytes, *, suffix: str) -> dict[str, Any]:
    text = payload.decode("utf-8")
    value = json.loads(text) if suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Scenario source must contain one mapping")
    return value


def load_scenario_bytes(payload: bytes, *, suffix: str = ".yaml") -> LoadedScenario:
    mapping = _load_mapping(payload, suffix=suffix)
    scenario = ScenarioDefinition.model_validate(mapping)
    canonical = json.dumps(
        scenario.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return LoadedScenario(
        scenario=scenario,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        canonical_json=canonical,
    )


def load_scenario(path: Path) -> LoadedScenario:
    return load_scenario_bytes(path.read_bytes(), suffix=path.suffix)
