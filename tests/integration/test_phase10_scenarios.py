from pathlib import Path

import pytest

from lingxu.scenario.loader import load_scenario
from lingxu.scenario.repository import ScenarioRepository
from lingxu.scenario.schema import ScenarioStatus
from lingxu.storage.database import Database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scenario_versions_publish_archive_and_preserve_snapshots(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'scenario.db').as_posix()}")
    await database.create_schema()
    repository = ScenarioRepository(database.sessions)
    loaded = load_scenario(Path("scenarios/fixed_meeting_v1.yaml"))

    first = await repository.create_draft(loaded)
    assert first.version == 1
    assert first.status is ScenarioStatus.DRAFT
    published_first = await repository.publish(first.id)
    assert published_first.status is ScenarioStatus.PUBLISHED
    assert (await repository.active("fixed-meeting")).id == first.id

    second = await repository.create_draft(loaded)
    assert second.version == 2
    assert second.id != first.id
    published_second = await repository.publish(second.id)
    assert published_second.status is ScenarioStatus.PUBLISHED
    assert (await repository.active("fixed-meeting")).id == second.id
    archived_first = await repository.archive(first.id)
    assert archived_first.status is ScenarioStatus.ARCHIVED
    assert first.schema == second.schema
    assert first.source_sha256 == second.source_sha256

    await database.close()
