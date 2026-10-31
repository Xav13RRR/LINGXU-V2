"""Async repositories for Sessions, immutable events, snapshots, and Worker jobs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from lingxu.domain.enums import JobStatus, MeetingEventType, MeetingState
from lingxu.domain.errors import (
    InvalidStateTransitionError,
    JobClaimError,
    SessionAlreadyActiveError,
    SessionNotFoundError,
)
from lingxu.domain.meeting_state import TERMINAL_STATES, validate_transition
from lingxu.domain.models import (
    AudioSegmentRecord,
    MeetingEventRecord,
    MeetingSessionRecord,
    MeetingSnapshotRecord,
    ProcessingJobRecord,
    RealtimeUtteranceRecord,
)
from lingxu.storage.tables import (
    AudioSegmentRow,
    MeetingEventRow,
    MeetingRealtimeUtteranceRow,
    MeetingSessionRow,
    MeetingStateSnapshotRow,
    ProcessingJobRow,
)
from lingxu.utils import new_ulid

_TRANSITION_CAS_ATTEMPTS = 8


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _session_record(row: MeetingSessionRow) -> MeetingSessionRecord:
    return MeetingSessionRecord(
        id=row.id,
        guild_id=row.guild_id,
        creator_user_id=row.creator_user_id,
        state=MeetingState(row.state),
        resume_state=MeetingState(row.resume_state) if row.resume_state else None,
        next_sequence=row.next_sequence,
        manifest_path=Path(row.manifest_path),
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
        publication_channel_id=row.publication_channel_id,
        setup_channel_id=row.setup_channel_id,
    )


def _event_record(row: MeetingEventRow) -> MeetingEventRecord:
    return MeetingEventRecord(
        id=row.id,
        session_id=row.session_id,
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        actor_user_id=row.actor_user_id,
        source=row.source,
        payload=dict(row.payload_json),
        evidence_utterance_ids=tuple(row.evidence_utterance_ids_json),
        state_before=MeetingState(row.state_before),
        state_after=MeetingState(row.state_after),
        reversible=row.reversible,
        correlation_id=row.correlation_id,
        occurred_at=_aware(row.occurred_at),
    )


@dataclass(frozen=True, slots=True)
class MeetingTransitionResult:
    """Committed state transition and the exact append-only event it created."""

    session: MeetingSessionRecord
    changed: bool
    event: MeetingEventRecord | None


def _snapshot_record(row: MeetingStateSnapshotRow) -> MeetingSnapshotRecord:
    return MeetingSnapshotRecord(
        id=row.id,
        session_id=row.session_id,
        sequence_number=row.sequence_number,
        state=MeetingState(row.state),
        payload=dict(row.snapshot_json),
        created_at=_aware(row.created_at),
    )


def _job_record(row: ProcessingJobRow) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        id=row.id,
        session_id=row.session_id,
        job_type=row.job_type,
        status=JobStatus(row.status),
        attempt_count=row.attempt_count,
        available_at=_aware(row.available_at),
        locked_at=_aware(row.locked_at) if row.locked_at else None,
        locked_by=row.locked_by,
        lock_expires_at=_aware(row.lock_expires_at) if row.lock_expires_at else None,
        current_stage=row.current_stage,
        last_error_type=row.last_error,
        checkpoint=dict(row.checkpoint_json),
        idempotency_key=row.idempotency_key,
        created_at=_aware(row.created_at),
        updated_at=_aware(row.updated_at),
    )


def _audio_segment_record(row: AudioSegmentRow) -> AudioSegmentRecord:
    return AudioSegmentRecord(
        id=row.id,
        session_id=row.session_id,
        user_id=row.user_id,
        segment_number=row.segment_number,
        encrypted_path=Path(row.encrypted_path),
        sha256=row.sha256,
        started_ms=row.started_ms,
        ended_ms=row.ended_ms,
        bytes_encrypted=row.bytes_encrypted,
    )


def _realtime_utterance_record(row: MeetingRealtimeUtteranceRow) -> RealtimeUtteranceRecord:
    return RealtimeUtteranceRecord(
        id=row.id,
        session_id=row.session_id,
        user_id=row.user_id,
        started_ms=row.started_ms,
        ended_ms=row.ended_ms,
        raw_text=row.raw_text,
        normalised_text=row.normalised_text,
        created_at=_aware(row.created_at),
    )


async def _session_row(db: AsyncSession, session_id: str) -> MeetingSessionRow:
    row = await db.get(MeetingSessionRow, session_id)
    if row is None:
        raise SessionNotFoundError(f"Meeting Session not found: {session_id}")
    return row


async def append_meeting_event_row(
    db: AsyncSession,
    *,
    session_id: str,
    event_type: str,
    actor_user_id: int | None,
    source: str,
    payload: dict[str, Any],
    evidence_utterance_ids: Iterable[str],
    state_before: MeetingState,
    state_after: MeetingState,
    reversible: bool,
    correlation_id: str | None,
) -> MeetingEventRow:
    sequence_result = await db.execute(
        update(MeetingSessionRow)
        .where(MeetingSessionRow.id == session_id)
        .values(next_sequence=MeetingSessionRow.next_sequence + 1)
        .returning(MeetingSessionRow.next_sequence)
    )
    sequence_number = sequence_result.scalar_one_or_none()
    if sequence_number is None:
        raise SessionNotFoundError(f"Meeting Session not found: {session_id}")
    row = MeetingEventRow(
        id=new_ulid(),
        session_id=session_id,
        sequence_number=sequence_number,
        event_type=event_type,
        actor_user_id=actor_user_id,
        source=source,
        payload_json=dict(payload),
        evidence_utterance_ids_json=list(evidence_utterance_ids),
        state_before=state_before.value,
        state_after=state_after.value,
        reversible=reversible,
        correlation_id=correlation_id or new_ulid(),
        occurred_at=utc_now(),
    )
    db.add(row)
    await db.flush()
    return row


class MeetingSessionRepository:
    """Persist and deterministically transition Meeting Sessions."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        *,
        guild_id: int,
        creator_user_id: int,
        manifest_path: Path,
        session_id: str | None = None,
    ) -> MeetingSessionRecord:
        row = MeetingSessionRow(
            id=session_id or new_ulid(),
            guild_id=guild_id,
            creator_user_id=creator_user_id,
            state=MeetingState.CREATED.value,
            resume_state=None,
            next_sequence=0,
            active_marker=1,
            manifest_path=str(manifest_path),
        )
        try:
            async with self._sessions() as db, db.begin():
                db.add(row)
                await db.flush()
        except IntegrityError as exc:
            raise SessionAlreadyActiveError(
                f"guild {guild_id} already has an active Meeting Session"
            ) from exc
        return _session_record(row)

    async def get(self, session_id: str) -> MeetingSessionRecord:
        async with self._sessions() as db:
            return _session_record(await _session_row(db, session_id))

    async def active_for_guild(self, guild_id: int) -> MeetingSessionRecord | None:
        """Return the guild's single non-terminal Meeting, if one exists."""

        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingSessionRow).where(
                    MeetingSessionRow.guild_id == guild_id,
                    MeetingSessionRow.active_marker == 1,
                )
            )
            return _session_record(row) if row else None

    async def latest_for_guild(self, guild_id: int) -> MeetingSessionRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingSessionRow)
                .where(MeetingSessionRow.guild_id == guild_id)
                .order_by(MeetingSessionRow.created_at.desc())
                .limit(1)
            )
            return _session_record(row) if row else None

    async def set_publication_channel(
        self,
        *,
        session_id: str,
        channel_id: int,
    ) -> MeetingSessionRecord:
        """Persist the exact Discord channel selected when the Meeting was created."""

        async with self._sessions() as db, db.begin():
            row = await db.scalar(
                update(MeetingSessionRow)
                .where(MeetingSessionRow.id == session_id)
                .values(publication_channel_id=channel_id)
                .returning(MeetingSessionRow)
            )
            if row is None:
                raise SessionNotFoundError(f"Meeting Session not found: {session_id}")
            return _session_record(row)

    async def set_setup_channel(
        self,
        *,
        session_id: str,
        channel_id: int,
    ) -> MeetingSessionRecord:
        """Persist the exact Discord channel in which the host created the Meeting."""

        async with self._sessions() as db, db.begin():
            row = await db.scalar(
                update(MeetingSessionRow)
                .where(MeetingSessionRow.id == session_id)
                .values(setup_channel_id=channel_id)
                .returning(MeetingSessionRow)
            )
            if row is None:
                raise SessionNotFoundError(f"Meeting Session not found: {session_id}")
            return _session_record(row)

    async def list_recoverable(self) -> tuple[MeetingSessionRecord, ...]:
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingSessionRow)
                .where(MeetingSessionRow.active_marker == 1)
                .order_by(MeetingSessionRow.created_at)
            )
            return tuple(_session_record(row) for row in result)

    async def list_terminal_reconcilable(self) -> tuple[MeetingSessionRecord, ...]:
        """Return terminal Sessions with exact actions or timer handoffs still pending."""

        transition = aliased(MeetingEventRow)
        completion = aliased(MeetingEventRow)
        expiry = aliased(MeetingEventRow)
        handoff = aliased(MeetingEventRow)

        def action_is_missing(action: str) -> Any:
            return ~exists(
                select(completion.id).where(
                    completion.session_id == MeetingSessionRow.id,
                    completion.event_type == MeetingEventType.RUNTIME_ACTION_COMPLETED.value,
                    completion.payload_json["transition_event_id"].as_string() == transition.id,
                    completion.payload_json["action"].as_string() == action,
                )
            )

        pending_handoff = exists(
            select(expiry.id).where(
                expiry.session_id == MeetingSessionRow.id,
                expiry.event_type == MeetingEventType.TIMER_EXPIRED.value,
                ~exists(
                    select(handoff.id).where(
                        handoff.session_id == MeetingSessionRow.id,
                        handoff.event_type == MeetingEventType.TIMER_HANDOFF_COMPLETED.value,
                        handoff.payload_json["timer_id"].as_string()
                        == expiry.payload_json["timer_id"].as_string(),
                    )
                ),
            )
        )
        stopped_actions = (
            "cancel_timers",
            "speak_closing",
            "finalise_capture",
            "trigger_post_meeting_job",
        )

        def is_exact_persisted_v2_action_batch(actions: tuple[str, ...]) -> Any:
            return and_(
                transition.payload_json["runtime_version"].as_integer() == 2,
                func.json_type(transition.payload_json, "$.runtime_actions") == "array",
                func.json_array_length(transition.payload_json, "$.runtime_actions")
                == len(actions),
                *(
                    func.json_extract(
                        transition.payload_json,
                        f"$.runtime_actions[{index}]",
                    )
                    == action
                    for index, action in enumerate(actions)
                ),
            )

        stopped_v2_action_batch = is_exact_persisted_v2_action_batch(stopped_actions)
        published_v2_action_batch = is_exact_persisted_v2_action_batch(("publish_completion",))
        terminal_states = (MeetingState.STOPPED.value, MeetingState.PUBLISHED.value)
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingSessionRow)
                .join(
                    transition,
                    and_(
                        transition.session_id == MeetingSessionRow.id,
                        transition.state_after == MeetingSessionRow.state,
                        transition.state_before != transition.state_after,
                    ),
                )
                .where(
                    MeetingSessionRow.state.in_(terminal_states),
                    or_(
                        and_(
                            stopped_v2_action_batch,
                            MeetingSessionRow.state == MeetingState.STOPPED.value,
                            or_(*(action_is_missing(action) for action in stopped_actions)),
                        ),
                        and_(
                            published_v2_action_batch,
                            MeetingSessionRow.state == MeetingState.PUBLISHED.value,
                            action_is_missing("publish_completion"),
                        ),
                        pending_handoff,
                    ),
                )
                .distinct()
                .order_by(MeetingSessionRow.created_at)
            )
            return tuple(_session_record(row) for row in result)

    async def list_post_processing_reconcilable(self) -> tuple[MeetingSessionRecord, ...]:
        """Return Sessions whose latest Worker result may still change the terminal projection."""

        latest_job_status = (
            select(ProcessingJobRow.status)
            .where(
                ProcessingJobRow.session_id == MeetingSessionRow.id,
                ProcessingJobRow.job_type == "POST_MEETING",
            )
            .order_by(ProcessingJobRow.created_at.desc())
            .limit(1)
            .scalar_subquery()
        )
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingSessionRow)
                .where(
                    or_(
                        MeetingSessionRow.state == MeetingState.POST_PROCESSING.value,
                        (MeetingSessionRow.state == MeetingState.FAILED_POST_PROCESSING.value)
                        & (latest_job_status == JobStatus.COMPLETED.value),
                    )
                )
                .order_by(MeetingSessionRow.created_at)
            )
            return tuple(_session_record(row) for row in result)

    async def list_deleted(self) -> tuple[MeetingSessionRecord, ...]:
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingSessionRow)
                .where(MeetingSessionRow.state == MeetingState.DELETED.value)
                .order_by(MeetingSessionRow.created_at)
            )
            return tuple(_session_record(row) for row in result)

    async def transition(
        self,
        *,
        session_id: str,
        target: MeetingState,
        actor_user_id: int | None,
        source: str,
        event_type: str = "SESSION_STATE_CHANGED",
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        allow_back: bool = False,
        allow_auto: bool = False,
        allow_post_processing_recovery: bool = False,
    ) -> MeetingSessionRecord:
        """Apply a transition while preserving the existing session-only API."""

        expected = await self.get(session_id)
        result = await self.transition_with_event(
            session_id=session_id,
            expected_state=expected.state,
            expected_resume_state=expected.resume_state,
            expected_sequence=expected.next_sequence,
            target=target,
            actor_user_id=actor_user_id,
            source=source,
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
            allow_back=allow_back,
            allow_auto=allow_auto,
            allow_post_processing_recovery=allow_post_processing_recovery,
        )
        return result.session

    async def transition_with_event(
        self,
        *,
        session_id: str,
        expected_state: MeetingState,
        expected_resume_state: MeetingState | None,
        expected_sequence: int,
        target: MeetingState,
        actor_user_id: int | None,
        source: str,
        event_type: str = "SESSION_STATE_CHANGED",
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        allow_back: bool = False,
        allow_auto: bool = False,
        allow_post_processing_recovery: bool = False,
    ) -> MeetingTransitionResult:
        """Return the transaction's explicit winner fact and exact committed event."""

        candidate_state = expected_state
        candidate_resume_state = expected_resume_state
        candidate_sequence = expected_sequence
        for _ in range(_TRANSITION_CAS_ATTEMPTS):
            validate_transition(
                candidate_state,
                target,
                resume_state=candidate_resume_state,
                allow_back=allow_back,
                allow_auto=allow_auto,
                allow_post_processing_recovery=allow_post_processing_recovery,
            )
            if candidate_state is target:
                current = await self.get(session_id)
                if current.state is target:
                    return MeetingTransitionResult(
                        session=current,
                        changed=False,
                        event=None,
                    )
                candidate_state = current.state
                candidate_resume_state = current.resume_state
                candidate_sequence = current.next_sequence
                continue

            resume_condition = (
                MeetingSessionRow.resume_state.is_(None)
                if candidate_resume_state is None
                else MeetingSessionRow.resume_state == candidate_resume_state.value
            )
            next_resume_state = candidate_resume_state.value if candidate_resume_state else None
            if target in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
                next_resume_state = candidate_state.value
            elif candidate_state in {MeetingState.PAUSED, MeetingState.INTERRUPTED}:
                next_resume_state = None
            values: dict[str, Any] = {
                "state": target.value,
                "resume_state": next_resume_state,
            }
            if target in TERMINAL_STATES:
                values["active_marker"] = None

            async with self._sessions() as db, db.begin():
                row = await db.scalar(
                    update(MeetingSessionRow)
                    .where(
                        MeetingSessionRow.id == session_id,
                        MeetingSessionRow.state == candidate_state.value,
                        resume_condition,
                        MeetingSessionRow.next_sequence == candidate_sequence,
                    )
                    .values(**values)
                    .returning(MeetingSessionRow)
                )
                if row is None:
                    current = _session_record(await _session_row(db, session_id))
                else:
                    event_row = await append_meeting_event_row(
                        db,
                        session_id=session_id,
                        event_type=event_type,
                        actor_user_id=actor_user_id,
                        source=source,
                        payload=payload or {},
                        evidence_utterance_ids=(),
                        state_before=candidate_state,
                        state_after=target,
                        reversible=False,
                        correlation_id=correlation_id,
                    )
                    await db.flush()
                    await db.refresh(row)
                    return MeetingTransitionResult(
                        session=_session_record(row),
                        changed=True,
                        event=_event_record(event_row),
                    )

            if current.state is target:
                return MeetingTransitionResult(
                    session=current,
                    changed=False,
                    event=None,
                )
            candidate_state = current.state
            candidate_resume_state = current.resume_state
            candidate_sequence = current.next_sequence

        raise InvalidStateTransitionError(
            "Meeting Session transition did not stabilize after concurrent changes"
        )

    async def mark_deleted(self, session_id: str, *, actor_user_id: int | None) -> None:
        await self.transition(
            session_id=session_id,
            target=MeetingState.DELETED,
            actor_user_id=actor_user_id,
            source="SYSTEM",
            event_type="SESSION_DELETED",
        )
        async with self._sessions() as db, db.begin():
            row = await _session_row(db, session_id)
            row.deleted_at = utc_now()

    async def hard_delete(self, session_id: str) -> None:
        async with self._sessions() as db, db.begin():
            # SQLite removes the parent row before executing cascaded child
            # deletes. Delete events explicitly while the DELETED parent is
            # still visible so the append-only trigger can authorize cleanup.
            await db.execute(
                delete(MeetingEventRow).where(MeetingEventRow.session_id == session_id)
            )
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(MeetingSessionRow).where(
                        MeetingSessionRow.id == session_id,
                        MeetingSessionRow.state == MeetingState.DELETED.value,
                    )
                ),
            )
            if result.rowcount != 1:
                raise SessionNotFoundError(
                    f"deleted Meeting Session not found for cleanup: {session_id}"
                )


class MeetingEventStore:
    """Append and read immutable Meeting Session events."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def append(
        self,
        *,
        session_id: str,
        event_type: str,
        actor_user_id: int | None = None,
        source: str = "SYSTEM",
        payload: dict[str, Any] | None = None,
        evidence_utterance_ids: Iterable[str] = (),
        reversible: bool = False,
        correlation_id: str | None = None,
    ) -> MeetingEventRecord:
        async with self._sessions() as db, db.begin():
            session = await _session_row(db, session_id)
            state = MeetingState(session.state)
            row = await append_meeting_event_row(
                db,
                session_id=session_id,
                event_type=event_type,
                actor_user_id=actor_user_id,
                source=source,
                payload=payload or {},
                evidence_utterance_ids=evidence_utterance_ids,
                state_before=state,
                state_after=state,
                reversible=reversible,
                correlation_id=correlation_id,
            )
            return _event_record(row)

    async def list(self, session_id: str) -> tuple[MeetingEventRecord, ...]:
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingEventRow)
                .where(MeetingEventRow.session_id == session_id)
                .order_by(MeetingEventRow.sequence_number)
            )
            return tuple(_event_record(row) for row in result)

    async def list_types(
        self,
        session_id: str,
        event_types: Iterable[str],
    ) -> tuple[MeetingEventRecord, ...]:
        """Read only selected fact kinds while preserving event-stream order."""

        selected_types = tuple(dict.fromkeys(event_types))
        if not selected_types:
            return ()
        async with self._sessions() as db:
            result = await db.scalars(
                select(MeetingEventRow)
                .where(
                    MeetingEventRow.session_id == session_id,
                    MeetingEventRow.event_type.in_(selected_types),
                )
                .order_by(MeetingEventRow.sequence_number)
            )
            return tuple(_event_record(row) for row in result)


class MeetingSnapshotStore:
    """Persist and recover deterministic Meeting state snapshots."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(
        self,
        *,
        session_id: str,
        state: MeetingState,
        payload: dict[str, Any],
        sequence_number: int | None = None,
    ) -> MeetingSnapshotRecord:
        async with self._sessions() as db, db.begin():
            session = await _session_row(db, session_id)
            selected_sequence = (
                session.next_sequence if sequence_number is None else sequence_number
            )
            if selected_sequence > session.next_sequence:
                raise ValueError("snapshot sequence cannot exceed the event stream")
            row = MeetingStateSnapshotRow(
                id=new_ulid(),
                session_id=session_id,
                sequence_number=selected_sequence,
                state=state.value,
                snapshot_json=dict(payload),
            )
            db.add(row)
            await db.flush()
            return _snapshot_record(row)

    async def latest(self, session_id: str) -> MeetingSnapshotRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingStateSnapshotRow)
                .where(MeetingStateSnapshotRow.session_id == session_id)
                .order_by(MeetingStateSnapshotRow.sequence_number.desc())
                .limit(1)
            )
            return _snapshot_record(row) if row else None


class ProcessingJobRepository:
    """Create idempotent jobs and atomically lease them to an independent Worker."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        *,
        session_id: str,
        job_type: str,
        idempotency_key: str,
        available_at: datetime | None = None,
    ) -> ProcessingJobRecord:
        job_id = new_ulid()
        values = {
            "id": job_id,
            "session_id": session_id,
            "job_type": job_type,
            "status": JobStatus.PENDING.value,
            "attempt_count": 0,
            "available_at": available_at or utc_now(),
            "idempotency_key": idempotency_key,
            "checkpoint_json": {},
        }
        async with self._sessions() as db, db.begin():
            await _session_row(db, session_id)
            result = await db.execute(
                sqlite_insert(ProcessingJobRow)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=("session_id", "job_type", "idempotency_key")
                )
                .returning(ProcessingJobRow.id)
            )
            inserted_id = result.scalar_one_or_none()
            if inserted_id is None:
                row = await db.scalar(
                    select(ProcessingJobRow).where(
                        ProcessingJobRow.session_id == session_id,
                        ProcessingJobRow.job_type == job_type,
                        ProcessingJobRow.idempotency_key == idempotency_key,
                    )
                )
            else:
                row = await db.get(ProcessingJobRow, inserted_id)
            if row is None:
                raise RuntimeError("idempotent processing job insert returned no row")
            return _job_record(row)

    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        job_types: Iterable[str] | None = None,
        now: datetime | None = None,
    ) -> ProcessingJobRecord | None:
        selected_now = now or utc_now()
        eligible = or_(
            (
                (ProcessingJobRow.status == JobStatus.PENDING.value)
                & (ProcessingJobRow.available_at <= selected_now)
            ),
            (
                (ProcessingJobRow.status == JobStatus.RUNNING.value)
                & (ProcessingJobRow.lock_expires_at <= selected_now)
            ),
        )
        candidate = select(ProcessingJobRow.id).where(eligible)
        selected_types = tuple(job_types or ())
        if selected_types:
            candidate = candidate.where(ProcessingJobRow.job_type.in_(selected_types))
        candidate = candidate.order_by(
            ProcessingJobRow.available_at, ProcessingJobRow.created_at
        ).limit(1)
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(ProcessingJobRow)
                .where(ProcessingJobRow.id == candidate.scalar_subquery(), eligible)
                .values(
                    status=JobStatus.RUNNING.value,
                    attempt_count=ProcessingJobRow.attempt_count + 1,
                    locked_at=selected_now,
                    locked_by=worker_id,
                    lock_expires_at=selected_now + timedelta(seconds=lease_seconds),
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            row = result.scalar_one_or_none()
            return _job_record(row) if row else None

    async def checkpoint(
        self,
        *,
        job_id: str,
        worker_id: str,
        stage: str,
        checkpoint: dict[str, Any],
        lease_seconds: float,
    ) -> ProcessingJobRecord:
        selected_now = utc_now()
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(ProcessingJobRow)
                .where(
                    ProcessingJobRow.id == job_id,
                    ProcessingJobRow.status == JobStatus.RUNNING.value,
                    ProcessingJobRow.locked_by == worker_id,
                )
                .values(
                    current_stage=stage,
                    checkpoint_json=dict(checkpoint),
                    lock_expires_at=selected_now + timedelta(seconds=lease_seconds),
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise JobClaimError(f"Worker {worker_id} does not own job {job_id}")
            return _job_record(row)

    async def renew_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> ProcessingJobRecord:
        """Extend an exact Worker's in-flight lease without changing its checkpoint."""

        if isinstance(lease_seconds, bool) or lease_seconds <= 0:
            raise ValueError("processing job lease duration must be positive")
        selected_now = now or utc_now()
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(ProcessingJobRow)
                .where(
                    ProcessingJobRow.id == job_id,
                    ProcessingJobRow.status == JobStatus.RUNNING.value,
                    ProcessingJobRow.locked_by == worker_id,
                )
                .values(
                    lock_expires_at=selected_now + timedelta(seconds=lease_seconds),
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise JobClaimError(f"Worker {worker_id} does not own job {job_id}")
            return _job_record(row)

    async def complete(self, *, job_id: str, worker_id: str) -> ProcessingJobRecord:
        selected_now = utc_now()
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(ProcessingJobRow)
                .where(
                    ProcessingJobRow.id == job_id,
                    ProcessingJobRow.status == JobStatus.RUNNING.value,
                    ProcessingJobRow.locked_by == worker_id,
                )
                .values(
                    status=JobStatus.COMPLETED.value,
                    locked_at=None,
                    locked_by=None,
                    lock_expires_at=None,
                    last_error=None,
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise JobClaimError(f"Worker {worker_id} does not own job {job_id}")
            return _job_record(row)

    async def fail(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_type: str,
        retry_at: datetime | None = None,
    ) -> ProcessingJobRecord:
        selected_now = utc_now()
        retry = retry_at is not None
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(ProcessingJobRow)
                .where(
                    ProcessingJobRow.id == job_id,
                    ProcessingJobRow.status == JobStatus.RUNNING.value,
                    ProcessingJobRow.locked_by == worker_id,
                )
                .values(
                    status=JobStatus.PENDING.value if retry else JobStatus.FAILED.value,
                    available_at=retry_at or selected_now,
                    locked_at=None,
                    locked_by=None,
                    lock_expires_at=None,
                    last_error=error_type[:160],
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise JobClaimError(f"Worker {worker_id} does not own job {job_id}")
            return _job_record(row)

    async def recover_expired(self, *, now: datetime | None = None) -> int:
        selected_now = now or utc_now()
        async with self._sessions() as db, db.begin():
            result = cast(
                CursorResult[Any],
                await db.execute(
                    update(ProcessingJobRow)
                    .where(
                        ProcessingJobRow.status == JobStatus.RUNNING.value,
                        ProcessingJobRow.lock_expires_at <= selected_now,
                    )
                    .values(
                        status=JobStatus.PENDING.value,
                        locked_at=None,
                        locked_by=None,
                        lock_expires_at=None,
                        available_at=selected_now,
                        updated_at=selected_now,
                    )
                ),
            )
            return int(result.rowcount)

    async def retry_failed(
        self,
        *,
        session_id: str,
        job_type: str,
        authorize_content_generation_retry: bool = False,
    ) -> ProcessingJobRecord | None:
        """Requeue one failed job while preserving its durable checkpoint."""

        selected_now = utc_now()
        async with self._sessions() as db, db.begin():
            failed = await db.scalar(
                select(ProcessingJobRow).where(
                    ProcessingJobRow.session_id == session_id,
                    ProcessingJobRow.job_type == job_type,
                    ProcessingJobRow.status == JobStatus.FAILED.value,
                )
            )
            if failed is None:
                return None
            checkpoint = dict(failed.checkpoint_json)
            if authorize_content_generation_retry:
                current_generation = checkpoint.get("full_transcript_minutes_attempt_generation", 0)
                if (
                    isinstance(current_generation, bool)
                    or not isinstance(current_generation, int)
                    or not 0 <= current_generation < 1_000
                ):
                    raise ValueError("content generation retry generation is invalid")
                checkpoint["full_transcript_minutes_attempt_generation"] = current_generation + 1
            row = await db.scalar(
                update(ProcessingJobRow)
                .where(
                    ProcessingJobRow.id == failed.id,
                    ProcessingJobRow.status == JobStatus.FAILED.value,
                )
                .values(
                    status=JobStatus.PENDING.value,
                    available_at=selected_now,
                    locked_at=None,
                    locked_by=None,
                    lock_expires_at=None,
                    last_error=None,
                    checkpoint_json=checkpoint,
                    updated_at=selected_now,
                )
                .returning(ProcessingJobRow)
            )
            return _job_record(row) if row else None

    async def get(self, job_id: str) -> ProcessingJobRecord | None:
        async with self._sessions() as db:
            row = await db.get(ProcessingJobRow, job_id)
            return _job_record(row) if row else None

    async def latest_for_session(
        self,
        session_id: str,
        *,
        job_type: str | None = None,
    ) -> ProcessingJobRecord | None:
        """Return the newest durable job status observed by the Bot controller."""

        query = select(ProcessingJobRow).where(ProcessingJobRow.session_id == session_id)
        if job_type is not None:
            query = query.where(ProcessingJobRow.job_type == job_type)
        async with self._sessions() as db:
            row = await db.scalar(query.order_by(ProcessingJobRow.created_at.desc()).limit(1))
            return _job_record(row) if row else None


class AudioSegmentRepository:
    """Persist and delete content-free encrypted Segment metadata."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, segment: AudioSegmentRecord) -> AudioSegmentRecord:
        """Idempotently persist one finalised encrypted Segment."""

        values = {
            "id": segment.id,
            "session_id": segment.session_id,
            "user_id": segment.user_id,
            "segment_number": segment.segment_number,
            "encrypted_path": segment.encrypted_path.as_posix(),
            "sha256": segment.sha256,
            "started_ms": segment.started_ms,
            "ended_ms": segment.ended_ms,
            "bytes_encrypted": segment.bytes_encrypted,
        }
        async with self._sessions() as db, db.begin():
            await _session_row(db, segment.session_id)
            result = await db.execute(
                sqlite_insert(AudioSegmentRow)
                .values(**values)
                .on_conflict_do_nothing(index_elements=("session_id", "user_id", "segment_number"))
                .returning(AudioSegmentRow.id)
            )
            inserted_id = result.scalar_one_or_none()
            if inserted_id is None:
                row = await db.scalar(
                    select(AudioSegmentRow).where(
                        AudioSegmentRow.session_id == segment.session_id,
                        AudioSegmentRow.user_id == segment.user_id,
                        AudioSegmentRow.segment_number == segment.segment_number,
                    )
                )
            else:
                row = await db.get(AudioSegmentRow, inserted_id)
            if row is None:
                raise RuntimeError("idempotent audio Segment insert returned no row")
            persisted = _audio_segment_record(row)
            if persisted.sha256 != segment.sha256:
                raise ValueError("audio Segment number already exists with a different hash")
            return persisted

    async def list_user(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> tuple[AudioSegmentRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(AudioSegmentRow)
                .where(
                    AudioSegmentRow.session_id == session_id,
                    AudioSegmentRow.user_id == user_id,
                )
                .order_by(AudioSegmentRow.segment_number)
            )
            return tuple(_audio_segment_record(row) for row in rows)

    async def list_session(self, session_id: str) -> tuple[AudioSegmentRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(AudioSegmentRow)
                .where(AudioSegmentRow.session_id == session_id)
                .order_by(
                    AudioSegmentRow.started_ms,
                    AudioSegmentRow.user_id,
                    AudioSegmentRow.segment_number,
                )
            )
            return tuple(_audio_segment_record(row) for row in rows)

    async def delete_user(self, *, session_id: str, user_id: int) -> int:
        async with self._sessions() as db, db.begin():
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(AudioSegmentRow).where(
                        AudioSegmentRow.session_id == session_id,
                        AudioSegmentRow.user_id == user_id,
                    )
                ),
            )
            return int(result.rowcount)

    async def delete_ids(self, *, session_id: str, segment_ids: Iterable[str]) -> int:
        """Delete exact recovered Segment rows without crossing the Session boundary."""

        selected_ids = tuple(segment_ids)
        if not selected_ids:
            return 0
        async with self._sessions() as db, db.begin():
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(AudioSegmentRow).where(
                        AudioSegmentRow.session_id == session_id,
                        AudioSegmentRow.id.in_(selected_ids),
                    )
                ),
            )
            return int(result.rowcount)


class RealtimeUtteranceRepository:
    """Persist raw ASR output separately from its normalised display form."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, utterance: RealtimeUtteranceRecord) -> RealtimeUtteranceRecord:
        async with self._sessions() as db, db.begin():
            await _session_row(db, utterance.session_id)
            row = MeetingRealtimeUtteranceRow(
                id=utterance.id,
                session_id=utterance.session_id,
                user_id=utterance.user_id,
                started_ms=utterance.started_ms,
                ended_ms=utterance.ended_ms,
                raw_text=utterance.raw_text,
                normalised_text=utterance.normalised_text,
            )
            db.add(row)
            await db.flush()
            await db.refresh(row)
            return _realtime_utterance_record(row)

    async def list_user(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> tuple[RealtimeUtteranceRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingRealtimeUtteranceRow)
                .where(
                    MeetingRealtimeUtteranceRow.session_id == session_id,
                    MeetingRealtimeUtteranceRow.user_id == user_id,
                )
                .order_by(
                    MeetingRealtimeUtteranceRow.started_ms,
                    MeetingRealtimeUtteranceRow.id,
                )
            )
            return tuple(_realtime_utterance_record(row) for row in rows)

    async def delete_at_or_after(
        self,
        *,
        session_id: str,
        user_id: int,
        cutoff_ms: int,
    ) -> int:
        """Delete one user's closed-interval utterances touching/following a cutoff."""

        async with self._sessions() as db, db.begin():
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(MeetingRealtimeUtteranceRow).where(
                        MeetingRealtimeUtteranceRow.session_id == session_id,
                        MeetingRealtimeUtteranceRow.user_id == user_id,
                        MeetingRealtimeUtteranceRow.ended_ms >= cutoff_ms,
                    )
                ),
            )
            return int(result.rowcount)

    async def list_session(self, session_id: str) -> tuple[RealtimeUtteranceRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingRealtimeUtteranceRow)
                .where(MeetingRealtimeUtteranceRow.session_id == session_id)
                .order_by(
                    MeetingRealtimeUtteranceRow.started_ms,
                    MeetingRealtimeUtteranceRow.user_id,
                    MeetingRealtimeUtteranceRow.id,
                )
            )
            return tuple(_realtime_utterance_record(row) for row in rows)

    async def delete_ids(self, *, session_id: str, utterance_ids: Iterable[str]) -> int:
        """Delete exact recovered utterances without crossing the Session boundary."""

        selected_ids = tuple(utterance_ids)
        if not selected_ids:
            return 0
        async with self._sessions() as db, db.begin():
            result = cast(
                CursorResult[Any],
                await db.execute(
                    delete(MeetingRealtimeUtteranceRow).where(
                        MeetingRealtimeUtteranceRow.session_id == session_id,
                        MeetingRealtimeUtteranceRow.id.in_(selected_ids),
                    )
                ),
            )
            return int(result.rowcount)
