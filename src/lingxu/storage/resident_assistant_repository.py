"""Append-only operational persistence for resident voice-assistant leases."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.enums import ResidentAssistantStatus
from lingxu.domain.errors import AppendOnlyViolationError, VoiceModeConflictError
from lingxu.domain.models import (
    ResidentAssistantEventRecord,
    ResidentAssistantSessionRecord,
)
from lingxu.storage.tables import ResidentAssistantEventRow, ResidentAssistantSessionRow
from lingxu.utils import new_ulid


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _session_record(row: ResidentAssistantSessionRow) -> ResidentAssistantSessionRecord:
    return ResidentAssistantSessionRecord(
        id=row.id,
        guild_id=row.guild_id,
        channel_id=row.channel_id,
        creator_user_id=row.creator_user_id,
        status=ResidentAssistantStatus(row.status),
        next_sequence=row.next_sequence,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        stopped_at=_aware(row.stopped_at) if row.stopped_at else None,
    )


def _event_record(row: ResidentAssistantEventRow) -> ResidentAssistantEventRecord:
    return ResidentAssistantEventRecord(
        id=row.id,
        session_id=row.session_id,
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        actor_user_id=row.actor_user_id,
        reason=row.reason,
        occurred_at=_aware(row.occurred_at),
    )


async def _append_event(
    db: AsyncSession,
    *,
    session_id: str,
    event_type: str,
    actor_user_id: int | None,
    reason: str,
) -> ResidentAssistantEventRow:
    result = await db.execute(
        update(ResidentAssistantSessionRow)
        .where(ResidentAssistantSessionRow.id == session_id)
        .values(next_sequence=ResidentAssistantSessionRow.next_sequence + 1)
        .returning(ResidentAssistantSessionRow.next_sequence)
    )
    sequence_number = result.scalar_one_or_none()
    if sequence_number is None:
        raise ValueError("resident assistant session does not exist")
    event = ResidentAssistantEventRow(
        id=new_ulid(),
        session_id=session_id,
        sequence_number=sequence_number,
        event_type=event_type,
        actor_user_id=actor_user_id,
        reason=reason,
        occurred_at=_utc_now(),
    )
    db.add(event)
    await db.flush()
    return event


class ResidentAssistantRepository:
    """Keep the event stream authoritative and materialised ACTIVE rows recoverable."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_active(
        self,
        *,
        session_id: str,
        guild_id: int,
        channel_id: int,
        creator_user_id: int,
    ) -> ResidentAssistantSessionRecord:
        now = _utc_now()
        row = ResidentAssistantSessionRow(
            id=session_id,
            guild_id=guild_id,
            channel_id=channel_id,
            creator_user_id=creator_user_id,
            status=ResidentAssistantStatus.ACTIVE.value,
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
                    session_id=session_id,
                    event_type="STARTED",
                    actor_user_id=creator_user_id,
                    reason="COMMAND",
                )
                await db.refresh(row)
                record = _session_record(row)
            return record
        except IntegrityError as exc:
            raise VoiceModeConflictError(
                "该服务器已有持久化的常驻语音助手。请先停止或等待恢复。"
            ) from exc

    async def active_for_guild(self, guild_id: int) -> ResidentAssistantSessionRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(ResidentAssistantSessionRow).where(
                    ResidentAssistantSessionRow.guild_id == guild_id,
                    ResidentAssistantSessionRow.status == ResidentAssistantStatus.ACTIVE.value,
                )
            )
            if row is None:
                return None
            await self._validate_stream(db, row)
            return _session_record(row)

    async def list_active(self) -> tuple[ResidentAssistantSessionRecord, ...]:
        async with self._sessions() as db:
            rows = tuple(
                (
                    await db.scalars(
                        select(ResidentAssistantSessionRow)
                        .where(
                            ResidentAssistantSessionRow.status
                            == ResidentAssistantStatus.ACTIVE.value
                        )
                        .order_by(ResidentAssistantSessionRow.created_at)
                    )
                ).all()
            )
            for row in rows:
                await self._validate_stream(db, row)
            return tuple(_session_record(row) for row in rows)

    async def record_recovered(self, session_id: str) -> ResidentAssistantEventRecord:
        async with self._sessions() as db, db.begin():
            event = await _append_event(
                db,
                session_id=session_id,
                event_type="RECOVERED",
                actor_user_id=None,
                reason="BOT_RESTART",
            )
        return _event_record(event)

    async def stop_active(
        self,
        *,
        session_id: str,
        actor_user_id: int | None,
        reason: str,
    ) -> ResidentAssistantSessionRecord | None:
        now = _utc_now()
        async with self._sessions() as db, db.begin():
            row = await db.get(ResidentAssistantSessionRow, session_id)
            if row is None or row.status != ResidentAssistantStatus.ACTIVE.value:
                return None
            await _append_event(
                db,
                session_id=session_id,
                event_type="STOPPED",
                actor_user_id=actor_user_id,
                reason=reason,
            )
            row.status = ResidentAssistantStatus.STOPPED.value
            row.active_marker = None
            row.stopped_at = now
            row.updated_at = now
            await db.flush()
        return _session_record(row)

    async def events(self, session_id: str) -> tuple[ResidentAssistantEventRecord, ...]:
        async with self._sessions() as db:
            rows = (
                await db.scalars(
                    select(ResidentAssistantEventRow)
                    .where(ResidentAssistantEventRow.session_id == session_id)
                    .order_by(ResidentAssistantEventRow.sequence_number)
                )
            ).all()
            return tuple(_event_record(row) for row in rows)

    @staticmethod
    async def _validate_stream(
        db: AsyncSession,
        row: ResidentAssistantSessionRow,
    ) -> None:
        events = (
            await db.scalars(
                select(ResidentAssistantEventRow)
                .where(ResidentAssistantEventRow.session_id == row.id)
                .order_by(ResidentAssistantEventRow.sequence_number)
            )
        ).all()
        sequences = [event.sequence_number for event in events]
        if (
            sequences != list(range(1, row.next_sequence + 1))
            or not events
            or events[0].event_type != "STARTED"
            or (
                row.status == ResidentAssistantStatus.ACTIVE.value
                and events[-1].event_type == "STOPPED"
            )
        ):
            raise AppendOnlyViolationError(
                "resident assistant event stream does not match materialised state"
            )
