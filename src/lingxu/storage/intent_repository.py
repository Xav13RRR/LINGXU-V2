"""Persistence for content-minimal structured-intent confirmation decisions."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.errors import IntentConfirmationError
from lingxu.domain.models import IntentCandidateRecord
from lingxu.storage.tables import MeetingIntentCandidateRow
from lingxu.utils import new_ulid


def _record(row: MeetingIntentCandidateRow) -> IntentCandidateRecord:
    payload = dict(row.payload_json)
    actor_user_id = payload.get("actor_user_id")
    source = payload.get("source")
    if not isinstance(actor_user_id, int) or not isinstance(source, str):
        raise IntentConfirmationError("persisted intent candidate metadata is invalid")
    return IntentCandidateRecord(
        id=row.id,
        session_id=row.session_id,
        utterance_id=row.utterance_id,
        intent_type=row.intent_type,
        actor_user_id=actor_user_id,
        source=source,
        confidence=row.confidence,
        accepted=row.accepted,
    )


class IntentCandidateRepository:
    """Store only the candidate enum, actor, confidence, source, and decision."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(
        self,
        *,
        session_id: str,
        intent_type: str,
        actor_user_id: int,
        source: str,
        confidence: float | None,
        utterance_id: str | None = None,
    ) -> IntentCandidateRecord:
        row = MeetingIntentCandidateRow(
            id=new_ulid(),
            session_id=session_id,
            utterance_id=utterance_id,
            intent_type=intent_type,
            payload_json={
                "actor_user_id": actor_user_id,
                "source": source,
            },
            confidence=confidence,
            accepted=None,
        )
        async with self._sessions() as db, db.begin():
            db.add(row)
            await db.flush()
            return _record(row)

    async def get(self, candidate_id: str) -> IntentCandidateRecord | None:
        async with self._sessions() as db:
            row = await db.get(MeetingIntentCandidateRow, candidate_id)
            return _record(row) if row is not None else None

    async def decide(
        self,
        *,
        candidate_id: str,
        session_id: str,
        actor_user_id: int,
        accepted: bool,
    ) -> IntentCandidateRecord:
        """Atomically decide one pending candidate exactly once."""

        async with self._sessions() as db, db.begin():
            existing = await db.get(MeetingIntentCandidateRow, candidate_id)
            if existing is None:
                raise IntentConfirmationError("intent candidate was not found")
            current = _record(existing)
            if current.session_id != session_id or current.actor_user_id != actor_user_id:
                raise IntentConfirmationError(
                    "intent candidate does not belong to this participant"
                )
            if current.accepted is not None:
                raise IntentConfirmationError("intent candidate was already decided")
            row = await db.scalar(
                update(MeetingIntentCandidateRow)
                .where(
                    MeetingIntentCandidateRow.id == candidate_id,
                    MeetingIntentCandidateRow.accepted.is_(None),
                )
                .values(accepted=accepted)
                .returning(MeetingIntentCandidateRow)
            )
            if row is None:
                raise IntentConfirmationError("intent candidate was decided concurrently")
            return _record(row)

    async def mark_execution_failed(self, candidate_id: str) -> None:
        """Record that an accepted proposal could not pass deterministic guards."""

        async with self._sessions() as db, db.begin():
            await db.execute(
                update(MeetingIntentCandidateRow)
                .where(MeetingIntentCandidateRow.id == candidate_id)
                .values(accepted=False)
            )

    async def list_session(self, session_id: str) -> tuple[IntentCandidateRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingIntentCandidateRow)
                .where(MeetingIntentCandidateRow.session_id == session_id)
                .order_by(MeetingIntentCandidateRow.id)
            )
            return tuple(_record(row) for row in rows)

    async def content_scan_values(self, session_id: str) -> tuple[dict[str, Any], ...]:
        """Expose candidate metadata for privacy-focused tests without raw SQL."""

        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingIntentCandidateRow).where(
                    MeetingIntentCandidateRow.session_id == session_id
                )
            )
            return tuple(dict(row.payload_json) for row in rows)
