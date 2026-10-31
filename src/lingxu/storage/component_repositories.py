"""Persistent repositories for deterministic Phase 7 Meeting components."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from lingxu.domain.enums import (
    AgendaSpeakingMode,
    AgendaStatus,
    ConsentKind,
    MeetingEventType,
    MeetingState,
    TimerStatus,
    VoteStatus,
)
from lingxu.domain.errors import (
    ParticipantNotPresentError,
    SessionNotFoundError,
    VoteNotOpenError,
)
from lingxu.domain.models import (
    AgendaItemRecord,
    MeetingEventRecord,
    ParticipantRecord,
    TimerRecord,
    VoteRecipientRecord,
    VoteRecord,
)
from lingxu.domain.speaking import AgendaSpeakingPolicy
from lingxu.storage.repositories import _event_record, append_meeting_event_row
from lingxu.storage.tables import (
    GuildMemberConsentRow,
    MeetingAgendaItemRow,
    MeetingEventRow,
    MeetingParticipantRow,
    MeetingSessionRow,
    MeetingTimerRow,
    MeetingVoteEntryRow,
    MeetingVoteRecipientRow,
    MeetingVoteRow,
)
from lingxu.utils import new_ulid


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _participant_record(row: MeetingParticipantRow) -> ParticipantRecord:
    return ParticipantRecord(
        id=row.id,
        session_id=row.session_id,
        user_id=row.user_id,
        present=row.left_at is None,
        display_name=row.display_name,
        is_host=row.is_host,
        joined_at=_aware(row.joined_at),
        left_at=_aware(row.left_at) if row.left_at else None,
    )


def _agenda_record(row: MeetingAgendaItemRow) -> AgendaItemRecord:
    payload = dict(row.payload_json)
    duration = payload.get("duration_minutes", 15)
    return AgendaItemRecord(
        id=row.id,
        session_id=row.session_id,
        position=row.position,
        title=row.title,
        status=AgendaStatus(row.status),
        duration_minutes=int(duration) if duration is not None else None,
        requires_vote=bool(payload.get("requires_vote", False)),
        vote_content=(
            str(payload["vote_content"]).strip() if payload.get("vote_content") else None
        ),
        vote_duration_seconds=(
            int(payload["vote_duration_seconds"])
            if payload.get("vote_duration_seconds") is not None
            else None
        ),
        language=str(payload.get("language", "auto")),
        speaking_policy=AgendaSpeakingPolicy.from_payload(payload.get("speaking_policy")),
    )


def _timer_record(row: MeetingTimerRow) -> TimerRecord:
    return TimerRecord(
        id=row.id,
        session_id=row.session_id,
        timer_type=row.timer_type,
        deadline_at=_aware(row.deadline_at),
        remaining_ms=row.remaining_ms,
        status=TimerStatus(row.status),
    )


def _vote_record(row: MeetingVoteRow) -> VoteRecord:
    if row.opened_at is None:
        raise ValueError(f"vote {row.id} has no opened_at timestamp")
    return VoteRecord(
        id=row.id,
        session_id=row.session_id,
        title=row.title,
        status=VoteStatus(row.status),
        options=tuple(dict(option) for option in row.options_json),
        deadline_at=_aware(row.deadline_at),
        opened_at=_aware(row.opened_at),
        closed_at=_aware(row.closed_at) if row.closed_at else None,
        anonymous=row.anonymous,
        agenda_item_id=row.agenda_item_id,
    )


def _vote_recipient_record(row: MeetingVoteRecipientRow) -> VoteRecipientRecord:
    return VoteRecipientRecord(
        vote_id=row.vote_id,
        user_id=row.user_id,
        delivery_status=row.delivery_status,
        dm_message_id=row.dm_message_id,
    )


async def _vote_deadline_has_elapsed(
    db: AsyncSession,
    *,
    vote: MeetingVoteRow,
    now: datetime,
) -> bool:
    """Treat the vote deadline as frozen while its persistent timer is paused."""

    paused_timer = await db.scalar(
        select(MeetingTimerRow)
        .where(
            MeetingTimerRow.session_id == vote.session_id,
            MeetingTimerRow.timer_type == f"VOTE:{vote.id}",
            MeetingTimerRow.status == TimerStatus.PAUSED.value,
        )
        .limit(1)
    )
    deadline_is_frozen = (
        paused_timer is not None
        and paused_timer.remaining_ms is not None
        and paused_timer.remaining_ms > 0
    )
    return not deadline_is_frozen and _aware(vote.deadline_at) <= now


@dataclass(slots=True)
class MeetingComponentTransaction:
    """One SQLAlchemy transaction shared by a component projection and its facts."""

    db: AsyncSession

    async def append_event(
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
        state_value = await self.db.scalar(
            select(MeetingSessionRow.state).where(MeetingSessionRow.id == session_id)
        )
        if state_value is None:
            raise SessionNotFoundError(f"Meeting Session not found: {session_id}")
        state = MeetingState(state_value)
        row = await append_meeting_event_row(
            self.db,
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


class MeetingComponentUnitOfWork:
    """Open the shared transaction used by Agenda, Vote, and Timer commands."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[MeetingComponentTransaction]:
        async with self._sessions() as db, db.begin():
            yield MeetingComponentTransaction(db)


class GuildConsentRepository:
    """Persist explicit consent within one Guild until the user withdraws it."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def grants_for(self, *, guild_id: int, user_id: int) -> frozenset[ConsentKind]:
        async with self._sessions() as db:
            row = (
                await db.execute(
                    select(
                        GuildMemberConsentRow.consent_realtime_asr,
                        GuildMemberConsentRow.consent_recording,
                        GuildMemberConsentRow.consent_minutes,
                    ).where(
                        GuildMemberConsentRow.guild_id == guild_id,
                        GuildMemberConsentRow.user_id == user_id,
                    )
                )
            ).one_or_none()
        if row is None:
            return frozenset()
        values = {
            ConsentKind.REALTIME_ASR: bool(row.consent_realtime_asr),
            ConsentKind.RECORDING: bool(row.consent_recording),
            ConsentKind.MINUTES: bool(row.consent_minutes),
        }
        return frozenset(kind for kind, granted in values.items() if granted)

    async def grant_all(self, *, guild_id: int, user_id: int) -> None:
        now = utc_now()
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(GuildMemberConsentRow)
                .values(
                    guild_id=guild_id,
                    user_id=user_id,
                    consent_realtime_asr=True,
                    consent_recording=True,
                    consent_minutes=True,
                    granted_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=("guild_id", "user_id"),
                    set_={
                        "consent_realtime_asr": True,
                        "consent_recording": True,
                        "consent_minutes": True,
                        "granted_at": now,
                        "updated_at": now,
                    },
                )
            )

    async def withdraw_all(self, *, guild_id: int, user_id: int) -> bool:
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                delete(GuildMemberConsentRow)
                .where(
                    GuildMemberConsentRow.guild_id == guild_id,
                    GuildMemberConsentRow.user_id == user_id,
                )
                .returning(GuildMemberConsentRow.user_id)
            )
            return result.scalar_one_or_none() is not None


class ConsentChangeRepository:
    """Atomically persist one active-Session consent change and its event."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    @staticmethod
    def _payload(*, user_id: int, effective_at_ms: int, trigger: str) -> dict[str, object]:
        if effective_at_ms < 0:
            raise ValueError("consent cutoff must be non-negative")
        return {
            "subject_user_id": user_id,
            "consent_kinds": [kind.value for kind in ConsentKind],
            "effective_at_ms": effective_at_ms,
            "trigger": trigger,
            "scope": "participant",
        }

    async def _active_session(
        self,
        db: AsyncSession,
        *,
        session_id: str,
        guild_id: int,
    ) -> MeetingSessionRow:
        row = await db.scalar(
            select(MeetingSessionRow)
            .where(
                MeetingSessionRow.id == session_id,
                MeetingSessionRow.guild_id == guild_id,
                MeetingSessionRow.active_marker == 1,
            )
            .with_for_update()
        )
        if row is None:
            raise SessionNotFoundError("active Meeting Session not found for Guild")
        return row

    @staticmethod
    async def _latest_all_consent_event(
        db: AsyncSession,
        *,
        session_id: str,
        user_id: int,
    ) -> MeetingEventRow | None:
        rows = await db.scalars(
            select(MeetingEventRow)
            .where(
                MeetingEventRow.session_id == session_id,
                MeetingEventRow.event_type.in_(
                    (
                        MeetingEventType.CONSENT_GRANTED.value,
                        MeetingEventType.CONSENT_WITHDRAWN.value,
                    )
                ),
            )
            .order_by(MeetingEventRow.sequence_number.desc())
        )
        expected_kinds = {kind.value for kind in ConsentKind}
        for row in rows:
            payload = dict(row.payload_json)
            if int(payload.get("subject_user_id", -1)) != user_id:
                continue
            if set(payload.get("consent_kinds", ())) == expected_kinds:
                return row
        return None

    @staticmethod
    async def _set_participant_consents(
        db: AsyncSession,
        *,
        session_id: str,
        user_id: int,
        granted: bool,
    ) -> None:
        result = await db.execute(
            update(MeetingParticipantRow)
            .where(
                MeetingParticipantRow.session_id == session_id,
                MeetingParticipantRow.user_id == user_id,
            )
            .values(
                consent_realtime_asr=granted,
                consent_recording=granted,
                consent_minutes=granted,
            )
            .returning(MeetingParticipantRow.id)
        )
        if result.scalar_one_or_none() is None:
            raise ParticipantNotPresentError("participant is not registered for this Meeting")

    async def grant_all(
        self,
        *,
        session_id: str,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
        trigger: str,
    ) -> MeetingEventRecord:
        """Grant all purposes and append a fact only for an actual transition."""

        payload = self._payload(
            user_id=user_id,
            effective_at_ms=effective_at_ms,
            trigger=trigger,
        )
        now = utc_now()
        async with self._sessions() as db, db.begin():
            session = await self._active_session(
                db,
                session_id=session_id,
                guild_id=guild_id,
            )
            latest = await self._latest_all_consent_event(
                db,
                session_id=session_id,
                user_id=user_id,
            )
            await db.execute(
                sqlite_insert(GuildMemberConsentRow)
                .values(
                    guild_id=guild_id,
                    user_id=user_id,
                    consent_realtime_asr=True,
                    consent_recording=True,
                    consent_minutes=True,
                    granted_at=now,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=("guild_id", "user_id"),
                    set_={
                        "consent_realtime_asr": True,
                        "consent_recording": True,
                        "consent_minutes": True,
                        "granted_at": now,
                        "updated_at": now,
                    },
                )
            )
            await self._set_participant_consents(
                db,
                session_id=session_id,
                user_id=user_id,
                granted=True,
            )
            if latest is not None and latest.event_type == MeetingEventType.CONSENT_GRANTED.value:
                return _event_record(latest)
            state = MeetingState(session.state)
            event = await append_meeting_event_row(
                db,
                session_id=session_id,
                event_type=MeetingEventType.CONSENT_GRANTED.value,
                actor_user_id=user_id,
                source="CONSENT",
                payload=payload,
                evidence_utterance_ids=(),
                state_before=state,
                state_after=state,
                reversible=False,
                correlation_id=None,
            )
            return _event_record(event)

    async def withdraw_all(
        self,
        *,
        session_id: str,
        guild_id: int,
        user_id: int,
        effective_at_ms: int,
        trigger: str,
    ) -> MeetingEventRecord | None:
        """Withdraw all purposes and append a fact only for an actual transition."""

        payload = self._payload(
            user_id=user_id,
            effective_at_ms=effective_at_ms,
            trigger=trigger,
        )
        async with self._sessions() as db, db.begin():
            session = await self._active_session(
                db,
                session_id=session_id,
                guild_id=guild_id,
            )
            latest = await self._latest_all_consent_event(
                db,
                session_id=session_id,
                user_id=user_id,
            )
            await db.execute(
                delete(GuildMemberConsentRow).where(
                    GuildMemberConsentRow.guild_id == guild_id,
                    GuildMemberConsentRow.user_id == user_id,
                )
            )
            await self._set_participant_consents(
                db,
                session_id=session_id,
                user_id=user_id,
                granted=False,
            )
            if latest is not None and latest.event_type == MeetingEventType.CONSENT_WITHDRAWN.value:
                return _event_record(latest)
            if latest is None:
                return None
            state = MeetingState(session.state)
            event = await append_meeting_event_row(
                db,
                session_id=session_id,
                event_type=MeetingEventType.CONSENT_WITHDRAWN.value,
                actor_user_id=user_id,
                source="CONSENT",
                payload=payload,
                evidence_utterance_ids=(),
                state_before=state,
                state_after=state,
                reversible=False,
                correlation_id=None,
            )
            return _event_record(event)


class ParticipantRepository:
    """Track current voice presence without requiring privileged member intents."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def join(
        self,
        *,
        session_id: str,
        user_id: int,
        inherited_consents: frozenset[ConsentKind] | None = None,
        display_name: str | None = None,
        is_host: bool = False,
    ) -> ParticipantRecord:
        participant_id = new_ulid()
        consent_values = {
            "consent_realtime_asr": (
                ConsentKind.REALTIME_ASR in inherited_consents
                if inherited_consents is not None
                else False
            ),
            "consent_recording": (
                ConsentKind.RECORDING in inherited_consents
                if inherited_consents is not None
                else False
            ),
            "consent_minutes": (
                ConsentKind.MINUTES in inherited_consents
                if inherited_consents is not None
                else False
            ),
        }
        clean_display_name = display_name.strip()[:100] if display_name else None
        update_values: dict[str, Any] = {
            "left_at": None,
            "is_host": is_host,
        }
        if clean_display_name is not None:
            update_values["display_name"] = clean_display_name
        if inherited_consents is not None:
            update_values.update(consent_values)
        async with self._sessions() as db, db.begin():
            await db.execute(
                sqlite_insert(MeetingParticipantRow)
                .values(
                    id=participant_id,
                    session_id=session_id,
                    user_id=user_id,
                    display_name=clean_display_name,
                    is_host=is_host,
                    left_at=None,
                    **consent_values,
                )
                .on_conflict_do_update(
                    index_elements=("session_id", "user_id"),
                    set_=update_values,
                )
            )
            row = await db.scalar(
                select(MeetingParticipantRow).where(
                    MeetingParticipantRow.session_id == session_id,
                    MeetingParticipantRow.user_id == user_id,
                )
            )
            if row is None:
                raise RuntimeError("participant upsert returned no row")
            return _participant_record(row)

    async def leave(self, *, session_id: str, user_id: int) -> bool:
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(MeetingParticipantRow)
                .where(
                    MeetingParticipantRow.session_id == session_id,
                    MeetingParticipantRow.user_id == user_id,
                    MeetingParticipantRow.left_at.is_(None),
                )
                .values(left_at=utc_now())
                .returning(MeetingParticipantRow.id)
            )
            return result.scalar_one_or_none() is not None

    async def present_user_ids(self, session_id: str) -> tuple[int, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingParticipantRow.user_id)
                .where(
                    MeetingParticipantRow.session_id == session_id,
                    MeetingParticipantRow.left_at.is_(None),
                )
                .order_by(MeetingParticipantRow.joined_at, MeetingParticipantRow.user_id)
            )
            return tuple(rows)

    async def list_session(self, session_id: str) -> tuple[ParticipantRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingParticipantRow)
                .where(MeetingParticipantRow.session_id == session_id)
                .order_by(MeetingParticipantRow.joined_at, MeetingParticipantRow.user_id)
            )
            return tuple(_participant_record(row) for row in rows)

    async def require_present(
        self,
        *,
        session_id: str,
        user_id: int,
        transaction: MeetingComponentTransaction | None = None,
    ) -> None:
        query = select(MeetingParticipantRow.id).where(
            MeetingParticipantRow.session_id == session_id,
            MeetingParticipantRow.user_id == user_id,
            MeetingParticipantRow.left_at.is_(None),
        )
        if transaction is None:
            async with self._sessions() as db:
                participant_id = await db.scalar(query)
        else:
            participant_id = await transaction.db.scalar(query)
        if participant_id is None:
            raise ParticipantNotPresentError("participant is not currently in voice")

    async def set_consent(
        self,
        *,
        session_id: str,
        user_id: int,
        kind: ConsentKind,
        granted: bool,
    ) -> None:
        columns = {
            ConsentKind.REALTIME_ASR: "consent_realtime_asr",
            ConsentKind.RECORDING: "consent_recording",
            ConsentKind.MINUTES: "consent_minutes",
        }
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(MeetingParticipantRow)
                .where(
                    MeetingParticipantRow.session_id == session_id,
                    MeetingParticipantRow.user_id == user_id,
                )
                .values({columns[kind]: granted})
                .returning(MeetingParticipantRow.id)
            )
            if result.scalar_one_or_none() is None:
                raise ParticipantNotPresentError("participant is not registered for this Meeting")

    async def set_consents(
        self,
        *,
        session_id: str,
        user_id: int,
        grants: frozenset[ConsentKind],
    ) -> None:
        async with self._sessions() as db, db.begin():
            result = await db.execute(
                update(MeetingParticipantRow)
                .where(
                    MeetingParticipantRow.session_id == session_id,
                    MeetingParticipantRow.user_id == user_id,
                )
                .values(
                    consent_realtime_asr=ConsentKind.REALTIME_ASR in grants,
                    consent_recording=ConsentKind.RECORDING in grants,
                    consent_minutes=ConsentKind.MINUTES in grants,
                )
                .returning(MeetingParticipantRow.id)
            )
            if result.scalar_one_or_none() is None:
                raise ParticipantNotPresentError("participant is not registered for this Meeting")

    async def all_present_fully_consented(self, session_id: str) -> bool:
        async with self._sessions() as db:
            rows = (
                await db.execute(
                    select(
                        MeetingParticipantRow.consent_realtime_asr,
                        MeetingParticipantRow.consent_recording,
                        MeetingParticipantRow.consent_minutes,
                    ).where(
                        MeetingParticipantRow.session_id == session_id,
                        MeetingParticipantRow.left_at.is_(None),
                    )
                )
            ).all()
        return bool(rows) and all(
            bool(realtime_asr and recording and minutes)
            for realtime_asr, recording, minutes in rows
        )

    async def consents_for(
        self,
        *,
        session_id: str,
        user_id: int,
    ) -> frozenset[ConsentKind]:
        async with self._sessions() as db:
            row = (
                await db.execute(
                    select(
                        MeetingParticipantRow.consent_realtime_asr,
                        MeetingParticipantRow.consent_recording,
                        MeetingParticipantRow.consent_minutes,
                    ).where(
                        MeetingParticipantRow.session_id == session_id,
                        MeetingParticipantRow.user_id == user_id,
                        MeetingParticipantRow.left_at.is_(None),
                    )
                )
            ).one_or_none()
        if row is None:
            return frozenset()
        values = {
            ConsentKind.REALTIME_ASR: bool(row.consent_realtime_asr),
            ConsentKind.RECORDING: bool(row.consent_recording),
            ConsentKind.MINUTES: bool(row.consent_minutes),
        }
        return frozenset(kind for kind, granted in values.items() if granted)

    async def consented_user_ids(
        self,
        *,
        session_id: str,
        kind: ConsentKind,
    ) -> tuple[int, ...]:
        columns = {
            ConsentKind.REALTIME_ASR: MeetingParticipantRow.consent_realtime_asr,
            ConsentKind.RECORDING: MeetingParticipantRow.consent_recording,
            ConsentKind.MINUTES: MeetingParticipantRow.consent_minutes,
        }
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingParticipantRow.user_id)
                .where(
                    MeetingParticipantRow.session_id == session_id,
                    columns[kind].is_(True),
                )
                .order_by(MeetingParticipantRow.user_id)
            )
            return tuple(rows)


class AgendaRepository:
    """Persist ordered, fixed-template agenda items."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions
        self.uow = MeetingComponentUnitOfWork(sessions)

    async def add(
        self,
        *,
        session_id: str,
        title: str,
        duration_minutes: int | None,
        requires_vote: bool,
        vote_content: str | None = None,
        vote_duration_seconds: int | None = None,
        language: str = "auto",
        speaking_policy: AgendaSpeakingPolicy | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> AgendaItemRecord:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("agenda title cannot be empty")
        if duration_minutes is not None and duration_minutes < 1:
            raise ValueError("agenda duration must be positive")
        clean_vote_content = vote_content.strip() if vote_content is not None else None
        if requires_vote:
            if not clean_vote_content:
                raise ValueError("vote content is required when requires_vote is true")
            if vote_duration_seconds is None or not 10 <= vote_duration_seconds <= 3600:
                raise ValueError("vote duration must be between 10 and 3600 seconds")
        elif clean_vote_content is not None or vote_duration_seconds is not None:
            raise ValueError("vote details require requires_vote to be true")
        if language not in {"auto", "zh", "yue", "en", "ja", "ko"}:
            raise ValueError("unsupported agenda language")
        selected_speaking_policy = speaking_policy or AgendaSpeakingPolicy.free()
        if (
            selected_speaking_policy.mode is AgendaSpeakingMode.PRESENTATION
            and duration_minutes is None
        ):
            raise ValueError("presentation agenda requires an overall duration")
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.add(
                    session_id=session_id,
                    title=title,
                    duration_minutes=duration_minutes,
                    requires_vote=requires_vote,
                    vote_content=vote_content,
                    vote_duration_seconds=vote_duration_seconds,
                    language=language,
                    speaking_policy=speaking_policy,
                    transaction=active_transaction,
                )
        db = transaction.db
        highest = await db.scalar(
            select(func.max(MeetingAgendaItemRow.position)).where(
                MeetingAgendaItemRow.session_id == session_id
            )
        )
        row = MeetingAgendaItemRow(
            id=new_ulid(),
            session_id=session_id,
            position=int(highest or 0) + 1,
            title=clean_title,
            status=AgendaStatus.PENDING.value,
            payload_json={
                "duration_minutes": duration_minutes,
                "requires_vote": requires_vote,
                "vote_content": clean_vote_content,
                "vote_duration_seconds": vote_duration_seconds,
                "language": language,
                "speaking_policy": selected_speaking_policy.to_payload(),
            },
        )
        db.add(row)
        await db.flush()
        return _agenda_record(row)

    async def list(
        self,
        session_id: str,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[AgendaItemRecord, ...]:
        query = (
            select(MeetingAgendaItemRow)
            .where(MeetingAgendaItemRow.session_id == session_id)
            .order_by(MeetingAgendaItemRow.position)
        )
        if transaction is None:
            async with self._sessions() as db:
                rows = await db.scalars(query)
                return tuple(_agenda_record(row) for row in rows)
        rows = await transaction.db.scalars(query)
        return tuple(_agenda_record(row) for row in rows)

    async def set_status(
        self,
        item_id: str,
        status: AgendaStatus,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> AgendaItemRecord:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.set_status(
                    item_id,
                    status,
                    transaction=active_transaction,
                )
        row = await transaction.db.scalar(
            update(MeetingAgendaItemRow)
            .where(MeetingAgendaItemRow.id == item_id)
            .values(status=status.value)
            .returning(MeetingAgendaItemRow)
        )
        if row is None:
            raise ValueError(f"agenda item not found: {item_id}")
        return _agenda_record(row)

    async def remove(
        self,
        *,
        session_id: str,
        position: int,
        transaction: MeetingComponentTransaction | None = None,
    ) -> bool:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.remove(
                    session_id=session_id,
                    position=position,
                    transaction=active_transaction,
                )
        db = transaction.db
        removed = await db.scalar(
            delete(MeetingAgendaItemRow)
            .where(
                MeetingAgendaItemRow.session_id == session_id,
                MeetingAgendaItemRow.position == position,
                MeetingAgendaItemRow.status == AgendaStatus.PENDING.value,
            )
            .returning(MeetingAgendaItemRow.id)
        )
        if removed is None:
            return False
        following = list(
            await db.scalars(
                select(MeetingAgendaItemRow)
                .where(
                    MeetingAgendaItemRow.session_id == session_id,
                    MeetingAgendaItemRow.position > position,
                )
                .order_by(MeetingAgendaItemRow.position)
            )
        )
        for row in following:
            row.position -= 1
        return True

    async def reorder(
        self,
        *,
        session_id: str,
        from_position: int,
        to_position: int,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[AgendaItemRecord, ...]:
        if from_position < 1 or to_position < 1:
            raise ValueError("agenda positions start at 1")
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.reorder(
                    session_id=session_id,
                    from_position=from_position,
                    to_position=to_position,
                    transaction=active_transaction,
                )
        db = transaction.db
        rows = list(
            await db.scalars(
                select(MeetingAgendaItemRow)
                .where(MeetingAgendaItemRow.session_id == session_id)
                .order_by(MeetingAgendaItemRow.position)
            )
        )
        if from_position > len(rows) or to_position > len(rows):
            raise ValueError("agenda position is outside the current agenda")
        selected = rows.pop(from_position - 1)
        rows.insert(to_position - 1, selected)
        # Move through temporary negative positions to preserve the unique key.
        for index, row in enumerate(rows, start=1):
            row.position = -index
        await db.flush()
        for index, row in enumerate(rows, start=1):
            row.position = index
        await db.flush()
        return tuple(_agenda_record(row) for row in rows)


class TimerRepository:
    """Persist running and paused deadlines so Bot restarts do not reset time."""

    _ACTIVE = (TimerStatus.RUNNING.value, TimerStatus.PAUSED.value)

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions
        self.uow = MeetingComponentUnitOfWork(sessions)

    async def start(
        self,
        *,
        session_id: str,
        timer_type: str,
        duration_seconds: float,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> TimerRecord:
        if duration_seconds <= 0:
            raise ValueError("timer duration must be positive")
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.start(
                    session_id=session_id,
                    timer_type=timer_type,
                    duration_seconds=duration_seconds,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        existing = await db.scalar(
            select(MeetingTimerRow).where(
                MeetingTimerRow.session_id == session_id,
                MeetingTimerRow.timer_type == timer_type,
                MeetingTimerRow.status.in_(self._ACTIVE),
            )
        )
        if existing is not None:
            return _timer_record(existing)
        row = MeetingTimerRow(
            id=new_ulid(),
            session_id=session_id,
            timer_type=timer_type,
            deadline_at=selected_now + timedelta(seconds=duration_seconds),
            remaining_ms=None,
            status=TimerStatus.RUNNING.value,
        )
        db.add(row)
        await db.flush()
        return _timer_record(row)

    async def list_active(
        self,
        session_id: str | None = None,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[TimerRecord, ...]:
        query = select(MeetingTimerRow).where(MeetingTimerRow.status.in_(self._ACTIVE))
        if session_id is not None:
            query = query.where(MeetingTimerRow.session_id == session_id)
        query = query.order_by(MeetingTimerRow.deadline_at, MeetingTimerRow.id)
        if transaction is None:
            async with self._sessions() as db:
                rows = await db.scalars(query)
                return tuple(_timer_record(row) for row in rows)
        rows = await transaction.db.scalars(query)
        return tuple(_timer_record(row) for row in rows)

    async def pause_all(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[TimerRecord, ...]:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.pause_all(
                    session_id,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        rows = list(
            await db.scalars(
                select(MeetingTimerRow).where(
                    MeetingTimerRow.session_id == session_id,
                    MeetingTimerRow.status == TimerStatus.RUNNING.value,
                )
            )
        )
        for row in rows:
            row.remaining_ms = max(
                0,
                int((_aware(row.deadline_at) - selected_now).total_seconds() * 1000),
            )
            row.status = TimerStatus.PAUSED.value
        await db.flush()
        return tuple(_timer_record(row) for row in rows)

    async def pause_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> TimerRecord | None:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.pause_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        row = await db.scalar(
            select(MeetingTimerRow).where(
                MeetingTimerRow.session_id == session_id,
                MeetingTimerRow.timer_type == timer_type,
                MeetingTimerRow.status == TimerStatus.RUNNING.value,
            )
        )
        if row is None:
            return None
        row.remaining_ms = max(
            0,
            int((_aware(row.deadline_at) - selected_now).total_seconds() * 1000),
        )
        row.status = TimerStatus.PAUSED.value
        await db.flush()
        return _timer_record(row)

    async def resume_all(
        self,
        session_id: str,
        *,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[TimerRecord, ...]:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.resume_all(
                    session_id,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        rows = list(
            await db.scalars(
                select(MeetingTimerRow).where(
                    MeetingTimerRow.session_id == session_id,
                    MeetingTimerRow.status == TimerStatus.PAUSED.value,
                )
            )
        )
        for row in rows:
            row.deadline_at = selected_now + timedelta(milliseconds=row.remaining_ms or 0)
            row.remaining_ms = None
            row.status = TimerStatus.RUNNING.value
        await db.flush()
        return tuple(_timer_record(row) for row in rows)

    async def resume_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> TimerRecord | None:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.resume_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        row = await db.scalar(
            select(MeetingTimerRow).where(
                MeetingTimerRow.session_id == session_id,
                MeetingTimerRow.timer_type == timer_type,
                MeetingTimerRow.status == TimerStatus.PAUSED.value,
            )
        )
        if row is None:
            return None
        row.deadline_at = selected_now + timedelta(milliseconds=row.remaining_ms or 0)
        row.remaining_ms = None
        row.status = TimerStatus.RUNNING.value
        await db.flush()
        return _timer_record(row)

    async def cancel_all(
        self,
        session_id: str,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> int:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.cancel_all(
                    session_id,
                    transaction=active_transaction,
                )
        rows = list(
            await transaction.db.scalars(
                select(MeetingTimerRow).where(
                    MeetingTimerRow.session_id == session_id,
                    MeetingTimerRow.status.in_(self._ACTIVE),
                )
            )
        )
        for row in rows:
            row.status = TimerStatus.CANCELLED.value
        return len(rows)

    async def cancel_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        transaction: MeetingComponentTransaction | None = None,
    ) -> int:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.cancel_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    transaction=active_transaction,
                )
        rows = list(
            await transaction.db.scalars(
                select(MeetingTimerRow).where(
                    MeetingTimerRow.session_id == session_id,
                    MeetingTimerRow.timer_type == timer_type,
                    MeetingTimerRow.status.in_(self._ACTIVE),
                )
            )
        )
        for row in rows:
            row.status = TimerStatus.CANCELLED.value
        return len(rows)

    async def extend_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        additional_seconds: int,
        transaction: MeetingComponentTransaction | None = None,
    ) -> TimerRecord | None:
        """Atomically extend one active deadline, including a paused timer."""

        if isinstance(additional_seconds, bool) or additional_seconds <= 0:
            raise ValueError("additional_seconds must be positive")
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.extend_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    additional_seconds=additional_seconds,
                    transaction=active_transaction,
                )
        db = transaction.db
        row = await db.scalar(
            select(MeetingTimerRow).where(
                MeetingTimerRow.session_id == session_id,
                MeetingTimerRow.timer_type == timer_type,
                MeetingTimerRow.status.in_(self._ACTIVE),
            )
        )
        if row is None:
            return None
        if row.status == TimerStatus.PAUSED.value:
            row.remaining_ms = (row.remaining_ms or 0) + additional_seconds * 1000
        else:
            row.deadline_at = _aware(row.deadline_at) + timedelta(seconds=additional_seconds)
        await db.flush()
        return _timer_record(row)

    async def expire(
        self,
        timer_id: str,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> tuple[TimerRecord, bool]:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.expire(timer_id, transaction=active_transaction)
        row = await transaction.db.get(MeetingTimerRow, timer_id)
        if row is None:
            raise ValueError(f"timer not found: {timer_id}")
        changed = row.status == TimerStatus.RUNNING.value
        if changed:
            row.status = TimerStatus.EXPIRED.value
            row.remaining_ms = 0
            await transaction.db.flush()
        return _timer_record(row), changed


@dataclass(frozen=True, slots=True)
class VoteCloseResult:
    vote: VoteRecord
    counts: dict[str, int]
    eligible_count: int
    changed: bool


class VoteRepository:
    """Persist public fixed-option votes and one mutable row per voter."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions
        self.uow = MeetingComponentUnitOfWork(sessions)

    async def open(
        self,
        *,
        session_id: str,
        title: str,
        options: tuple[dict[str, Any], ...],
        duration_seconds: float,
        anonymous: bool = False,
        agenda_item_id: str | None = None,
        eligible_user_ids: tuple[int, ...] = (),
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> VoteRecord:
        if duration_seconds <= 0:
            raise ValueError("vote duration must be positive")
        option_ids = tuple(str(option.get("id", "")) for option in options)
        if not option_ids or any(not value for value in option_ids):
            raise ValueError("each vote option requires a non-empty id")
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("vote option ids must be unique")
        selected_now = now or utc_now()
        selected_users = tuple(sorted(set(eligible_user_ids)))
        if agenda_item_id is None and selected_users:
            raise ValueError("eligible recipients require an agenda-linked vote")
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.open(
                    session_id=session_id,
                    title=title,
                    options=options,
                    duration_seconds=duration_seconds,
                    anonymous=anonymous,
                    agenda_item_id=agenda_item_id,
                    eligible_user_ids=selected_users,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        existing = await db.scalar(
            select(MeetingVoteRow).where(
                MeetingVoteRow.session_id == session_id,
                MeetingVoteRow.status == VoteStatus.OPEN.value,
            )
        )
        if existing is not None:
            return _vote_record(existing)
        row = MeetingVoteRow(
            id=new_ulid(),
            session_id=session_id,
            agenda_item_id=agenda_item_id,
            title=title.strip() or "Public vote",
            status=VoteStatus.OPEN.value,
            anonymous=anonymous,
            options_json=[dict(option) for option in options],
            opened_at=selected_now,
            deadline_at=selected_now + timedelta(seconds=duration_seconds),
            closed_at=None,
        )
        db.add(row)
        await db.flush()
        for user_id in selected_users:
            db.add(
                MeetingVoteRecipientRow(
                    id=new_ulid(),
                    vote_id=row.id,
                    user_id=user_id,
                    delivery_status="PENDING",
                )
            )
        await db.flush()
        return _vote_record(row)

    async def get(
        self,
        vote_id: str,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> VoteRecord:
        if transaction is None:
            async with self._sessions() as db:
                row = await db.get(MeetingVoteRow, vote_id)
        else:
            row = await transaction.db.get(MeetingVoteRow, vote_id)
        if row is None:
            raise VoteNotOpenError(f"vote not found: {vote_id}")
        return _vote_record(row)

    async def active_for_session(
        self,
        session_id: str,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> VoteRecord | None:
        query = select(MeetingVoteRow).where(
            MeetingVoteRow.session_id == session_id,
            MeetingVoteRow.status == VoteStatus.OPEN.value,
        )
        if transaction is None:
            async with self._sessions() as db:
                row = await db.scalar(query)
        else:
            row = await transaction.db.scalar(query)
        return _vote_record(row) if row else None

    async def latest_for_session(self, session_id: str) -> VoteRecord | None:
        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingVoteRow)
                .where(MeetingVoteRow.session_id == session_id)
                .order_by(MeetingVoteRow.opened_at.desc(), MeetingVoteRow.id.desc())
            )
            return _vote_record(row) if row else None

    async def recipients(self, vote_id: str) -> tuple[VoteRecipientRecord, ...]:
        async with self._sessions() as db:
            rows = await db.scalars(
                select(MeetingVoteRecipientRow)
                .where(MeetingVoteRecipientRow.vote_id == vote_id)
                .order_by(MeetingVoteRecipientRow.user_id)
            )
            return tuple(_vote_recipient_record(row) for row in rows)

    async def mark_delivery(
        self,
        *,
        vote_id: str,
        user_id: int,
        delivery_status: str,
        dm_message_id: int | None = None,
    ) -> VoteRecipientRecord:
        if delivery_status not in {"PENDING", "DELIVERED", "FAILED"}:
            raise ValueError("unknown vote delivery status")
        async with self._sessions() as db, db.begin():
            row = await db.scalar(
                select(MeetingVoteRecipientRow).where(
                    MeetingVoteRecipientRow.vote_id == vote_id,
                    MeetingVoteRecipientRow.user_id == user_id,
                )
            )
            if row is None:
                raise ParticipantNotPresentError("user is not eligible for this vote")
            row.delivery_status = delivery_status
            row.dm_message_id = dm_message_id
            await db.flush()
            return _vote_recipient_record(row)

    async def vote_for_dm_message(self, *, message_id: int, user_id: int) -> VoteRecord:
        async with self._sessions() as db:
            row = await db.scalar(
                select(MeetingVoteRow)
                .join(MeetingVoteRecipientRow, MeetingVoteRecipientRow.vote_id == MeetingVoteRow.id)
                .where(
                    MeetingVoteRecipientRow.dm_message_id == message_id,
                    MeetingVoteRecipientRow.user_id == user_id,
                )
            )
            if row is None:
                raise ParticipantNotPresentError("this private ballot is not assigned to you")
            return _vote_record(row)

    async def eligible_progress(self, vote_id: str) -> tuple[int, int]:
        async with self._sessions() as db:
            eligible = int(
                await db.scalar(
                    select(func.count(MeetingVoteRecipientRow.id)).where(
                        MeetingVoteRecipientRow.vote_id == vote_id
                    )
                )
                or 0
            )
            responded = int(
                await db.scalar(
                    select(func.count(MeetingVoteEntryRow.id))
                    .join(
                        MeetingVoteRecipientRow,
                        (MeetingVoteRecipientRow.vote_id == MeetingVoteEntryRow.vote_id)
                        & (MeetingVoteRecipientRow.user_id == MeetingVoteEntryRow.user_id),
                    )
                    .where(MeetingVoteEntryRow.vote_id == vote_id)
                )
                or 0
            )
            return eligible, responded

    async def set_deadline(
        self,
        vote_id: str,
        deadline_at: datetime,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> None:
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                await self.set_deadline(
                    vote_id,
                    deadline_at,
                    transaction=active_transaction,
                )
                return
        row = await transaction.db.get(MeetingVoteRow, vote_id)
        if row is not None and row.status == VoteStatus.OPEN.value:
            row.deadline_at = deadline_at

    async def cast(
        self,
        *,
        vote_id: str,
        user_id: int,
        option_id: str,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> str:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.cast(
                    vote_id=vote_id,
                    user_id=user_id,
                    option_id=option_id,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        vote = await db.get(MeetingVoteRow, vote_id)
        if vote is None or vote.status != VoteStatus.OPEN.value:
            raise VoteNotOpenError("vote is not open")
        if await _vote_deadline_has_elapsed(db, vote=vote, now=selected_now):
            raise VoteNotOpenError("vote is not open")
        if vote.anonymous:
            raise VoteNotOpenError("anonymous vote requires a session-scoped token")
        if vote.agenda_item_id is not None:
            recipient = await db.scalar(
                select(MeetingVoteRecipientRow.id).where(
                    MeetingVoteRecipientRow.vote_id == vote_id,
                    MeetingVoteRecipientRow.user_id == user_id,
                )
            )
            if recipient is None:
                raise ParticipantNotPresentError("user is not eligible for this vote")
        valid_ids = {str(option["id"]) for option in vote.options_json}
        if option_id not in valid_ids:
            raise ValueError("unknown vote option")
        await db.execute(
            sqlite_insert(MeetingVoteEntryRow)
            .values(
                id=new_ulid(),
                vote_id=vote_id,
                user_id=user_id,
                option_id=option_id,
                created_at=selected_now,
            )
            .on_conflict_do_update(
                index_elements=("vote_id", "user_id"),
                set_={"option_id": option_id, "created_at": selected_now},
            )
        )
        return option_id

    async def cast_anonymous(
        self,
        *,
        vote_id: str,
        pseudonymous_user_id: int,
        voter_token_hash: str,
        option_id: str,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> str:
        if pseudonymous_user_id >= 0:
            raise ValueError("anonymous voter pseudonym must be negative")
        if len(voter_token_hash) != 64:
            raise ValueError("anonymous voter token hash must be SHA-256")
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.cast_anonymous(
                    vote_id=vote_id,
                    pseudonymous_user_id=pseudonymous_user_id,
                    voter_token_hash=voter_token_hash,
                    option_id=option_id,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        vote = await db.get(MeetingVoteRow, vote_id)
        if vote is None or vote.status != VoteStatus.OPEN.value or not vote.anonymous:
            raise VoteNotOpenError("anonymous vote is not open")
        if await _vote_deadline_has_elapsed(db, vote=vote, now=selected_now):
            raise VoteNotOpenError("anonymous vote is not open")
        valid_ids = {str(option["id"]) for option in vote.options_json}
        if option_id not in valid_ids:
            raise ValueError("unknown vote option")
        await db.execute(
            sqlite_insert(MeetingVoteEntryRow)
            .values(
                id=new_ulid(),
                vote_id=vote_id,
                user_id=pseudonymous_user_id,
                voter_token_hash=voter_token_hash,
                option_id=option_id,
                created_at=selected_now,
            )
            .on_conflict_do_update(
                index_elements=("vote_id", "voter_token_hash"),
                set_={
                    "option_id": option_id,
                    "created_at": selected_now,
                },
            )
        )
        return option_id

    async def close(
        self,
        vote_id: str,
        *,
        now: datetime | None = None,
        transaction: MeetingComponentTransaction | None = None,
    ) -> VoteCloseResult:
        selected_now = now or utc_now()
        if transaction is None:
            async with self.uow.begin() as active_transaction:
                return await self.close(
                    vote_id,
                    now=selected_now,
                    transaction=active_transaction,
                )
        db = transaction.db
        vote = await db.get(MeetingVoteRow, vote_id)
        if vote is None:
            raise VoteNotOpenError(f"vote not found: {vote_id}")
        changed = vote.status == VoteStatus.OPEN.value
        if changed:
            vote.status = VoteStatus.CLOSED.value
            vote.closed_at = selected_now
            await db.flush()
        grouped = await db.execute(
            select(MeetingVoteEntryRow.option_id, func.count(MeetingVoteEntryRow.id))
            .where(MeetingVoteEntryRow.vote_id == vote_id)
            .group_by(MeetingVoteEntryRow.option_id)
        )
        observed = {str(option_id): int(count) for option_id, count in grouped}
        counts = {
            str(option["id"]): observed.get(str(option["id"]), 0) for option in vote.options_json
        }
        eligible_count = int(
            await db.scalar(
                select(func.count(MeetingVoteRecipientRow.id)).where(
                    MeetingVoteRecipientRow.vote_id == vote_id
                )
            )
            or 0
        )
        if vote.agenda_item_id is not None:
            counts["abstain"] = max(0, eligible_count - sum(counts.values()))
        return VoteCloseResult(
            vote=_vote_record(vote),
            counts=counts,
            eligible_count=eligible_count,
            changed=changed,
        )
