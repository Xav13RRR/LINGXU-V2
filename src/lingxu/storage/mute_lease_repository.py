"""Persistent Discord server-mute ownership leases."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.enums import MeetingEventType, MeetingState, MuteLeaseOwnership
from lingxu.domain.models import MeetingMuteLeaseRecord
from lingxu.storage.repositories import append_meeting_event_row
from lingxu.storage.tables import MeetingMuteLeaseRow, MeetingSessionRow
from lingxu.utils import new_ulid


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _record(row: MeetingMuteLeaseRow) -> MeetingMuteLeaseRecord:
    return MeetingMuteLeaseRecord(
        id=row.id,
        session_id=row.session_id,
        agenda_item_id=row.agenda_item_id,
        user_id=row.user_id,
        original_server_mute=row.original_server_mute,
        ownership=MuteLeaseOwnership(row.ownership),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        released_at=_aware(row.released_at) if row.released_at is not None else None,
    )


async def _append_event(
    db: AsyncSession,
    *,
    row: MeetingMuteLeaseRow,
    event_type: MeetingEventType,
    ownership: MuteLeaseOwnership,
    reason: str,
) -> None:
    state_value = await db.scalar(
        select(MeetingSessionRow.state).where(MeetingSessionRow.id == row.session_id)
    )
    if state_value is None:
        raise ValueError("mute lease Session does not exist")
    state = MeetingState(state_value)
    await append_meeting_event_row(
        db,
        session_id=row.session_id,
        event_type=event_type.value,
        actor_user_id=None,
        source="SYSTEM",
        payload={
            "agenda_item_id": row.agenda_item_id,
            "lease_id": row.id,
            "user_id": row.user_id,
            "original_server_mute": row.original_server_mute,
            "ownership": ownership.value,
            "reason": reason,
        },
        evidence_utterance_ids=(),
        state_before=state,
        state_after=state,
        reversible=False,
        correlation_id=None,
    )


class MuteLeaseRepository:
    """Own the materialised half of the event-audited mute protocol."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def active_for_member(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> MeetingMuteLeaseRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingMuteLeaseRow).where(
                    MeetingMuteLeaseRow.session_id == session_id,
                    MeetingMuteLeaseRow.user_id == user_id,
                    MeetingMuteLeaseRow.active_marker == 1,
                )
            )
            return _record(row) if row is not None else None

    async def list_active(
        self,
        *,
        session_id: str | None = None,
    ) -> tuple[MeetingMuteLeaseRecord, ...]:
        query = select(MeetingMuteLeaseRow).where(MeetingMuteLeaseRow.active_marker == 1)
        if session_id is not None:
            query = query.where(MeetingMuteLeaseRow.session_id == session_id)
        query = query.order_by(
            MeetingMuteLeaseRow.session_id,
            MeetingMuteLeaseRow.user_id,
        )
        async with self._sessions() as db:
            rows = await db.scalars(query)
            return tuple(_record(row) for row in rows)

    async def acquire(
        self,
        *,
        session_id: str,
        agenda_item_id: str,
        user_id: int,
        original_server_mute: bool,
        ownership: MuteLeaseOwnership,
        reason: str,
    ) -> tuple[MeetingMuteLeaseRecord, bool]:
        expected_ownership = (
            MuteLeaseOwnership.OBSERVED if original_server_mute else MuteLeaseOwnership.APPLYING
        )
        if ownership is not expected_ownership:
            raise ValueError("initial mute lease ownership does not match observed server mute")
        existing = await self.active_for_member(session_id=session_id, user_id=user_id)
        if existing is not None:
            return existing, False
        now = _now()
        row = MeetingMuteLeaseRow(
            id=new_ulid(),
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            user_id=user_id,
            original_server_mute=original_server_mute,
            ownership=ownership.value,
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
                    row=row,
                    event_type=MeetingEventType.MUTE_LEASE_ACQUIRED,
                    ownership=ownership,
                    reason=reason,
                )
                await db.refresh(row)
                return _record(row), True
        except IntegrityError:
            existing = await self.active_for_member(session_id=session_id, user_id=user_id)
            if existing is None:
                raise
            return existing, False

    async def set_ownership(
        self,
        *,
        lease_id: str,
        ownership: MuteLeaseOwnership,
        reason: str,
    ) -> MeetingMuteLeaseRecord:
        if ownership not in {
            MuteLeaseOwnership.OWNED,
            MuteLeaseOwnership.UNCERTAIN,
        }:
            raise ValueError("mute lease can only transition to OWNED or UNCERTAIN")
        async with self._sessions() as db, db.begin():
            row = await db.get(MeetingMuteLeaseRow, lease_id)
            if row is None or row.active_marker != 1:
                raise ValueError("mute lease is not active")
            row.ownership = ownership.value
            row.updated_at = _now()
            await db.flush()
            await _append_event(
                db,
                row=row,
                event_type=(
                    MeetingEventType.MUTE_LEASE_OWNED
                    if ownership is MuteLeaseOwnership.OWNED
                    else MeetingEventType.MUTE_LEASE_UNCERTAIN
                ),
                ownership=ownership,
                reason=reason,
            )
            return _record(row)

    async def release(
        self,
        lease_id: str,
        *,
        reason: str,
    ) -> MeetingMuteLeaseRecord | None:
        now = _now()
        async with self._sessions() as db, db.begin():
            row = await db.get(MeetingMuteLeaseRow, lease_id)
            if row is None or row.active_marker != 1:
                return None
            previous_ownership = MuteLeaseOwnership(row.ownership)
            await _append_event(
                db,
                row=row,
                event_type=MeetingEventType.MUTE_LEASE_RELEASED,
                ownership=previous_ownership,
                reason=reason,
            )
            row.ownership = MuteLeaseOwnership.RELEASED.value
            row.active_marker = None
            row.released_at = now
            row.updated_at = now
            await db.flush()
            return _record(row)
