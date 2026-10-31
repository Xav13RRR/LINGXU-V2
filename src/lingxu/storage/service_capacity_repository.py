"""Persistent, append-only coordination for one deployment-wide service slot."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.enums import ServiceCapacityMode, ServiceCapacityStatus
from lingxu.domain.errors import AppendOnlyViolationError, ServiceCapacityFullError
from lingxu.domain.models import ServiceCapacityEventRecord, ServiceCapacityLeaseRecord
from lingxu.storage.tables import ServiceCapacityEventRow, ServiceCapacityLeaseRow
from lingxu.utils import new_ulid

_SLOT_NAME = "PRIMARY"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _lease_record(row: ServiceCapacityLeaseRow) -> ServiceCapacityLeaseRecord:
    return ServiceCapacityLeaseRecord(
        id=row.id,
        guild_id=row.guild_id,
        mode=ServiceCapacityMode(row.mode),
        owner_id=row.owner_id,
        status=ServiceCapacityStatus(row.status),
        next_sequence=row.next_sequence,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        released_at=_aware(row.released_at) if row.released_at else None,
    )


def _event_record(row: ServiceCapacityEventRow) -> ServiceCapacityEventRecord:
    return ServiceCapacityEventRecord(
        id=row.id,
        lease_id=row.lease_id,
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        reason=row.reason,
        occurred_at=_aware(row.occurred_at),
    )


async def _append_event(
    db: AsyncSession,
    *,
    lease_id: str,
    event_type: str,
    reason: str,
) -> ServiceCapacityEventRow:
    result = await db.execute(
        update(ServiceCapacityLeaseRow)
        .where(ServiceCapacityLeaseRow.id == lease_id)
        .values(next_sequence=ServiceCapacityLeaseRow.next_sequence + 1)
        .returning(ServiceCapacityLeaseRow.next_sequence)
    )
    sequence_number = result.scalar_one_or_none()
    if sequence_number is None:
        raise ValueError("service capacity lease does not exist")
    event = ServiceCapacityEventRow(
        id=new_ulid(),
        lease_id=lease_id,
        sequence_number=sequence_number,
        event_type=event_type,
        reason=reason[:80],
        occurred_at=_utc_now(),
    )
    db.add(event)
    await db.flush()
    return event


class ServiceCapacityRepository:
    """Atomically grant one workload exclusive use of the local deployment."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def claim(
        self,
        *,
        guild_id: int,
        mode: ServiceCapacityMode,
        owner_id: str,
    ) -> ServiceCapacityLeaseRecord:
        if not owner_id or len(owner_id) > 80:
            raise ValueError("service capacity owner_id must contain 1 to 80 characters")
        current = await self.current()
        if current is not None:
            if (
                current.guild_id == guild_id
                and current.mode is mode
                and current.owner_id == owner_id
            ):
                return current
            raise ServiceCapacityFullError()

        now = _utc_now()
        row = ServiceCapacityLeaseRow(
            id=new_ulid(),
            slot_name=_SLOT_NAME,
            guild_id=guild_id,
            mode=mode.value,
            owner_id=owner_id,
            status=ServiceCapacityStatus.ACTIVE.value,
            next_sequence=0,
            active_marker=1,
            created_at=now,
            updated_at=now,
        )
        try:
            async with self._sessions() as db, db.begin():
                db.add(row)
                await db.flush()
                await _append_event(
                    db,
                    lease_id=row.id,
                    event_type="ACQUIRED",
                    reason="WORKLOAD_START",
                )
                await db.refresh(row)
                return _lease_record(row)
        except IntegrityError as exc:
            current = await self.current()
            if (
                current is not None
                and current.guild_id == guild_id
                and current.mode is mode
                and current.owner_id == owner_id
            ):
                return current
            raise ServiceCapacityFullError() from exc

    async def current(self) -> ServiceCapacityLeaseRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(ServiceCapacityLeaseRow).where(
                    ServiceCapacityLeaseRow.slot_name == _SLOT_NAME,
                    ServiceCapacityLeaseRow.status == ServiceCapacityStatus.ACTIVE.value,
                )
            )
            if row is None:
                return None
            await self._validate_stream(db, row)
            return _lease_record(row)

    async def ensure_available_to_guild(self, guild_id: int) -> None:
        """Permit adjunct work only when the slot is free or owned by this Guild."""

        current = await self.current()
        if current is not None and current.guild_id != guild_id:
            raise ServiceCapacityFullError()

    async def release(
        self,
        lease: ServiceCapacityLeaseRecord,
        *,
        reason: str,
    ) -> bool:
        now = _utc_now()
        async with self._sessions() as db, db.begin():
            row = await db.get(ServiceCapacityLeaseRow, lease.id)
            if (
                row is None
                or row.status != ServiceCapacityStatus.ACTIVE.value
                or row.guild_id != lease.guild_id
                or row.mode != lease.mode.value
                or row.owner_id != lease.owner_id
            ):
                return False
            await _append_event(
                db,
                lease_id=row.id,
                event_type="RELEASED",
                reason=reason,
            )
            row.status = ServiceCapacityStatus.RELEASED.value
            row.active_marker = None
            row.released_at = now
            row.updated_at = now
            await db.flush()
            return True

    async def release_owner(
        self,
        *,
        guild_id: int,
        mode: ServiceCapacityMode,
        owner_id: str,
        reason: str,
    ) -> bool:
        current = await self.current()
        if (
            current is None
            or current.guild_id != guild_id
            or current.mode is not mode
            or current.owner_id != owner_id
        ):
            return False
        return await self.release(current, reason=reason)

    async def record_recovered(
        self,
        lease: ServiceCapacityLeaseRecord,
    ) -> ServiceCapacityEventRecord:
        async with self._sessions() as db, db.begin():
            row = await db.get(ServiceCapacityLeaseRow, lease.id)
            if row is None or row.status != ServiceCapacityStatus.ACTIVE.value:
                raise ValueError("service capacity lease is not active")
            event = await _append_event(
                db,
                lease_id=row.id,
                event_type="RECOVERED",
                reason="BOT_RESTART",
            )
        return _event_record(event)

    async def events(self, lease_id: str) -> tuple[ServiceCapacityEventRecord, ...]:
        async with self._sessions() as db:
            rows = (
                await db.scalars(
                    select(ServiceCapacityEventRow)
                    .where(ServiceCapacityEventRow.lease_id == lease_id)
                    .order_by(ServiceCapacityEventRow.sequence_number)
                )
            ).all()
            return tuple(_event_record(row) for row in rows)

    @staticmethod
    async def _validate_stream(db: AsyncSession, row: ServiceCapacityLeaseRow) -> None:
        events = (
            await db.scalars(
                select(ServiceCapacityEventRow)
                .where(ServiceCapacityEventRow.lease_id == row.id)
                .order_by(ServiceCapacityEventRow.sequence_number)
            )
        ).all()
        sequences = [event.sequence_number for event in events]
        if (
            sequences != list(range(1, row.next_sequence + 1))
            or not events
            or events[0].event_type != "ACQUIRED"
            or (
                row.status == ServiceCapacityStatus.ACTIVE.value
                and events[-1].event_type == "RELEASED"
            )
        ):
            raise AppendOnlyViolationError(
                "service capacity event stream does not match materialised state"
            )
