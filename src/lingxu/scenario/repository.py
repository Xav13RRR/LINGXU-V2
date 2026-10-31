"""Immutable Scenario snapshots with draft/published/archived lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.scenario.loader import LoadedScenario
from lingxu.scenario.schema import ScenarioDefinition, ScenarioStatus
from lingxu.storage.tables import ScenarioDefinitionRow, ScenarioVersionRow
from lingxu.utils import new_ulid


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ScenarioVersionRecord:
    id: str
    scenario_id: str
    version: int
    status: ScenarioStatus
    schema: ScenarioDefinition
    source_sha256: str
    created_at: datetime
    published_at: datetime | None


def _record(row: ScenarioVersionRow) -> ScenarioVersionRecord:
    return ScenarioVersionRecord(
        id=row.id,
        scenario_id=row.scenario_id,
        version=row.version,
        status=ScenarioStatus(row.status),
        schema=ScenarioDefinition.model_validate(row.schema_json),
        source_sha256=row.source_sha256,
        created_at=_aware(row.created_at),
        published_at=_aware(row.published_at) if row.published_at else None,
    )


class ScenarioRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_draft(self, loaded: LoadedScenario) -> ScenarioVersionRecord:
        async with self._sessions() as db, db.begin():
            definition = await db.scalar(
                select(ScenarioDefinitionRow).where(
                    ScenarioDefinitionRow.name == loaded.scenario.key
                )
            )
            if definition is None:
                definition = ScenarioDefinitionRow(
                    id=new_ulid(),
                    name=loaded.scenario.key,
                    active_version_id=None,
                )
                db.add(definition)
                await db.flush()
            highest = await db.scalar(
                select(func.max(ScenarioVersionRow.version)).where(
                    ScenarioVersionRow.scenario_id == definition.id
                )
            )
            row = ScenarioVersionRow(
                id=new_ulid(),
                scenario_id=definition.id,
                version=int(highest or 0) + 1,
                status=ScenarioStatus.DRAFT.value,
                schema_json=loaded.scenario.model_dump(mode="json"),
                source_sha256=loaded.source_sha256,
                compiled_blob=loaded.canonical_json,
                published_at=None,
            )
            db.add(row)
            await db.flush()
            await db.refresh(row)
            return _record(row)

    async def publish(self, version_id: str) -> ScenarioVersionRecord:
        async with self._sessions() as db, db.begin():
            row = await db.get(ScenarioVersionRow, version_id)
            if row is None:
                raise ValueError(f"Scenario version not found: {version_id}")
            if row.status != ScenarioStatus.DRAFT.value:
                if row.status == ScenarioStatus.PUBLISHED.value:
                    return _record(row)
                raise ValueError("only a draft Scenario version may be published")
            definition = await db.get(ScenarioDefinitionRow, row.scenario_id)
            if definition is None:
                raise RuntimeError("Scenario definition is missing")
            previous = await db.scalar(
                select(ScenarioVersionRow).where(
                    ScenarioVersionRow.scenario_id == row.scenario_id,
                    ScenarioVersionRow.status == ScenarioStatus.PUBLISHED.value,
                )
            )
            if previous is not None:
                previous.status = ScenarioStatus.ARCHIVED.value
            row.status = ScenarioStatus.PUBLISHED.value
            row.published_at = datetime.now(UTC)
            definition.active_version_id = row.id
            await db.flush()
            await db.refresh(row)
            return _record(row)

    async def archive(self, version_id: str) -> ScenarioVersionRecord:
        async with self._sessions() as db, db.begin():
            row = await db.get(ScenarioVersionRow, version_id)
            if row is None:
                raise ValueError(f"Scenario version not found: {version_id}")
            row.status = ScenarioStatus.ARCHIVED.value
            definition = await db.get(ScenarioDefinitionRow, row.scenario_id)
            if definition is not None and definition.active_version_id == row.id:
                definition.active_version_id = None
            await db.flush()
            return _record(row)

    async def active(self, key: str) -> ScenarioVersionRecord | None:
        async with self._sessions() as db:
            definition = await db.scalar(
                select(ScenarioDefinitionRow).where(ScenarioDefinitionRow.name == key)
            )
            if definition is None or definition.active_version_id is None:
                return None
            row = await db.get(ScenarioVersionRow, definition.active_version_id)
            return _record(row) if row else None
