"""Idempotently publish the approved fixed Meeting Scenario snapshot."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lingxu.config import Settings
from lingxu.scenario.loader import load_scenario
from lingxu.scenario.repository import ScenarioRepository
from lingxu.storage.database import Database


async def deploy() -> dict[str, object]:
    settings = Settings()
    database = Database.open(settings.database_url)
    try:
        loaded = load_scenario(Path("scenarios/fixed_meeting_v1.yaml"))
        repository = ScenarioRepository(database.sessions)
        active = await repository.active(loaded.scenario.key)
        if active is not None and active.source_sha256 == loaded.source_sha256:
            return {
                "changed": False,
                "key": active.schema.key,
                "version": active.version,
                "status": active.status.value,
                "source_sha256": active.source_sha256,
            }
        draft = await repository.create_draft(loaded)
        published = await repository.publish(draft.id)
        return {
            "changed": True,
            "key": published.schema.key,
            "version": published.version,
            "status": published.status.value,
            "source_sha256": published.source_sha256,
        }
    finally:
        await database.close()


def main() -> None:
    print(json.dumps(asyncio.run(deploy()), sort_keys=True))


if __name__ == "__main__":
    main()
