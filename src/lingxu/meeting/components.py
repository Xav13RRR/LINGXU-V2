"""Deterministic agenda, speaking queue, timer, and public vote services."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from lingxu.domain.enums import AgendaStatus, MeetingEventType, TimerStatus
from lingxu.domain.errors import VoteNotOpenError
from lingxu.domain.models import AgendaItemRecord, TimerRecord, VoteRecipientRecord, VoteRecord
from lingxu.domain.speaking import AgendaSpeakingPolicy
from lingxu.storage.component_repositories import (
    AgendaRepository,
    MeetingComponentTransaction,
    ParticipantRepository,
    TimerRepository,
    VoteCloseResult,
    VoteRepository,
)
from lingxu.storage.repositories import MeetingEventStore

_TransitionResult = TypeVar("_TransitionResult")

_TIMER_TICK_EVENT_TYPES = (
    MeetingEventType.TIMER_STARTED.value,
    MeetingEventType.TIMER_REMINDER_60.value,
    MeetingEventType.TIMER_REMINDER_10.value,
    MeetingEventType.TIMER_EXPIRED.value,
)
_TIMER_HANDOFF_EVENT_TYPES = (
    MeetingEventType.TIMER_STARTED.value,
    MeetingEventType.TIMER_EXPIRED.value,
    MeetingEventType.TIMER_HANDOFF_COMPLETED.value,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgendaService:
    """Manage a persistent ordered agenda and its append-only audit events."""

    def __init__(self, repository: AgendaRepository, events: MeetingEventStore) -> None:
        self._repository = repository
        self._events = events
        self._lock = asyncio.Lock()

    async def add(
        self,
        *,
        session_id: str,
        title: str,
        duration_minutes: int | None = None,
        requires_vote: bool = False,
        vote_content: str | None = None,
        vote_duration_seconds: int | None = None,
        language: str = "auto",
        speaking_policy: AgendaSpeakingPolicy | None = None,
        actor_user_id: int | None,
        source: str,
    ) -> AgendaItemRecord:
        async with self._lock:
            async with self._repository.uow.begin() as transaction:
                item = await self._repository.add(
                    session_id=session_id,
                    title=title,
                    duration_minutes=duration_minutes,
                    requires_vote=requires_vote,
                    vote_content=vote_content,
                    vote_duration_seconds=vote_duration_seconds,
                    language=language,
                    speaking_policy=speaking_policy,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.AGENDA_ITEM_ADDED.value,
                    actor_user_id=actor_user_id,
                    source=source,
                    payload={
                        "agenda_item_id": item.id,
                        "position": item.position,
                        "duration_minutes": item.duration_minutes,
                        "requires_vote": item.requires_vote,
                        "vote_content": item.vote_content,
                        "vote_duration_seconds": item.vote_duration_seconds,
                        "language": item.language,
                        "speaking_policy": item.speaking_policy.to_payload(),
                    },
                )
                return item

    async def list(self, session_id: str) -> tuple[AgendaItemRecord, ...]:
        return await self._repository.list(session_id)

    async def remove(
        self,
        *,
        session_id: str,
        position: int,
        actor_user_id: int,
        source: str,
    ) -> bool:
        async with self._lock:
            async with self._repository.uow.begin() as transaction:
                changed = await self._repository.remove(
                    session_id=session_id,
                    position=position,
                    transaction=transaction,
                )
                if changed:
                    await transaction.append_event(
                        session_id=session_id,
                        event_type=MeetingEventType.AGENDA_ITEM_REMOVED.value,
                        actor_user_id=actor_user_id,
                        source=source,
                        payload={"position": position},
                    )
                return changed

    async def reorder(
        self,
        *,
        session_id: str,
        from_position: int,
        to_position: int,
        actor_user_id: int,
        source: str,
    ) -> tuple[AgendaItemRecord, ...]:
        async with self._lock:
            async with self._repository.uow.begin() as transaction:
                items = await self._repository.reorder(
                    session_id=session_id,
                    from_position=from_position,
                    to_position=to_position,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.AGENDA_REORDERED.value,
                    actor_user_id=actor_user_id,
                    source=source,
                    payload={"from_position": from_position, "to_position": to_position},
                )
                return items

    async def activate_next(self, *, session_id: str) -> AgendaItemRecord | None:
        async with self._lock:
            async with self._repository.uow.begin() as transaction:
                items = await self._repository.list(session_id, transaction=transaction)
                active = next((item for item in items if item.status is AgendaStatus.ACTIVE), None)
                if active is not None:
                    return active
                pending = next(
                    (item for item in items if item.status is AgendaStatus.PENDING),
                    None,
                )
                if pending is None:
                    return None
                item = await self._repository.set_status(
                    pending.id,
                    AgendaStatus.ACTIVE,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.AGENDA_ITEM_ACTIVATED.value,
                    payload={"agenda_item_id": item.id, "position": item.position},
                )
                return item

    async def complete_active(self, *, session_id: str) -> AgendaItemRecord | None:
        async with self._lock:
            async with self._repository.uow.begin() as transaction:
                items = await self._repository.list(session_id, transaction=transaction)
                active = next((item for item in items if item.status is AgendaStatus.ACTIVE), None)
                if active is None:
                    return None
                item = await self._repository.set_status(
                    active.id,
                    AgendaStatus.COMPLETED,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.AGENDA_ITEM_COMPLETED.value,
                    payload={"agenda_item_id": item.id, "position": item.position},
                )
                return item


@dataclass(frozen=True, slots=True)
class SpeakingQueueState:
    queued_user_ids: tuple[int, ...]
    current_user_id: int | None


class SpeakingQueueService:
    """Recover the deduplicated speaking queue entirely from immutable events."""

    def __init__(
        self,
        *,
        participants: ParticipantRepository,
        events: MeetingEventStore,
    ) -> None:
        self._participants = participants
        self._events = events
        self._lock = asyncio.Lock()

    async def _state(self, session_id: str) -> SpeakingQueueState:
        queued: list[int] = []
        current: int | None = None
        for event in await self._events.list(session_id):
            value = event.payload.get("user_id")
            if not isinstance(value, int):
                continue
            if event.event_type == MeetingEventType.SPEAK_REQUESTED.value:
                if value != current and value not in queued:
                    queued.append(value)
            elif event.event_type == MeetingEventType.SPEAKER_SELECTED.value:
                if value in queued:
                    queued.remove(value)
                current = value
            elif event.event_type == MeetingEventType.SPEAKER_FINISHED.value:
                if current == value:
                    current = None
            elif event.event_type == MeetingEventType.SPEAKER_REMOVED.value:
                if value in queued:
                    queued.remove(value)
                if current == value:
                    current = None
        return SpeakingQueueState(tuple(queued), current)

    async def state(self, session_id: str) -> SpeakingQueueState:
        return await self._state(session_id)

    async def request(
        self,
        *,
        session_id: str,
        user_id: int,
        source: str,
    ) -> bool:
        async with self._lock:
            await self._participants.require_present(session_id=session_id, user_id=user_id)
            state = await self._state(session_id)
            if user_id == state.current_user_id or user_id in state.queued_user_ids:
                return False
            await self._events.append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAK_REQUESTED.value,
                actor_user_id=user_id,
                source=source,
                payload={"user_id": user_id},
            )
            return True

    async def _remove_offline_locked(self, session_id: str) -> tuple[int, ...]:
        state = await self._state(session_id)
        present = set(await self._participants.present_user_ids(session_id))
        candidates = state.queued_user_ids + (
            (state.current_user_id,) if state.current_user_id is not None else ()
        )
        removed = tuple(user_id for user_id in candidates if user_id not in present)
        for user_id in removed:
            await self._events.append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKER_REMOVED.value,
                source="VOICE_STATE",
                payload={"user_id": user_id, "reason": "offline"},
            )
        return removed

    async def remove_offline(self, session_id: str) -> tuple[int, ...]:
        async with self._lock:
            return await self._remove_offline_locked(session_id)

    async def next(
        self,
        *,
        session_id: str,
        actor_user_id: int,
        source: str,
    ) -> int | None:
        async with self._lock:
            await self._remove_offline_locked(session_id)
            state = await self._state(session_id)
            if state.current_user_id is not None:
                return state.current_user_id
            if not state.queued_user_ids:
                return None
            selected = state.queued_user_ids[0]
            await self._events.append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKER_SELECTED.value,
                actor_user_id=actor_user_id,
                source=source,
                payload={"user_id": selected},
            )
            return selected

    async def finish(
        self,
        *,
        session_id: str,
        actor_user_id: int,
        source: str,
    ) -> int | None:
        async with self._lock:
            state = await self._state(session_id)
            if state.current_user_id is None:
                return None
            finished = state.current_user_id
            await self._events.append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKER_FINISHED.value,
                actor_user_id=actor_user_id,
                source=source,
                payload={"user_id": finished},
            )
            return finished

    async def remove(
        self,
        *,
        session_id: str,
        user_id: int,
        actor_user_id: int,
        source: str,
    ) -> bool:
        async with self._lock:
            state = await self._state(session_id)
            if user_id != state.current_user_id and user_id not in state.queued_user_ids:
                return False
            await self._events.append(
                session_id=session_id,
                event_type=MeetingEventType.SPEAKER_REMOVED.value,
                actor_user_id=actor_user_id,
                source=source,
                payload={"user_id": user_id, "reason": "host_removed"},
            )
            return True


@dataclass(frozen=True, slots=True)
class VoteOutcome:
    vote: VoteRecord
    counts: dict[str, int]
    eligible_count: int
    changed: bool

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def result(self) -> str:
        yes = self.counts.get("yes", 0)
        no = self.counts.get("no", 0)
        if yes > no:
            return "是"
        if no > yes:
            return "否"
        return "平票"

    def broadcast_text(self) -> str:
        if self.vote.agenda_item_id is not None:
            return (
                f"在上一轮投票中，针对{self.vote.title}问题，"
                f"{self.counts.get('yes', 0)}人选择是，"
                f"{self.counts.get('no', 0)}人选择否，"
                f"{self.counts.get('abstain', 0)}人弃权。"
                f"投票的最终结果为：{self.result}。"
            )
        labels = {
            str(option["id"]): str(option.get("label", option["id"]))
            for option in self.vote.options
        }
        result = "，".join(
            f"{labels.get(option_id, option_id)} {count}"
            for option_id, count in self.counts.items()
        )
        return f"投票「{self.vote.title}」已结束：{result}；共 {self.total} 票。"


class VotingService:
    """Open, cast, close, tally, and audit a public yes/no/abstain vote."""

    OPTIONS = (
        {"id": "yes", "label": "赞成"},
        {"id": "no", "label": "反对"},
        {"id": "abstain", "label": "弃权"},
    )
    AGENDA_OPTIONS = (
        {"id": "yes", "label": "是"},
        {"id": "no", "label": "否"},
    )

    def __init__(
        self,
        *,
        votes: VoteRepository,
        timers: TimerRepository,
        participants: ParticipantRepository,
        events: MeetingEventStore,
    ) -> None:
        self._votes = votes
        self._timers = timers
        self._participants = participants
        self._events = events
        self._lock = asyncio.Lock()

    async def open(
        self,
        *,
        session_id: str,
        title: str,
        duration_seconds: float,
        actor_user_id: int,
        source: str,
        agenda_item_id: str | None = None,
        eligible_user_ids: tuple[int, ...] = (),
        now: datetime | None = None,
    ) -> VoteRecord:
        async with self._lock:
            async with self._votes.uow.begin() as transaction:
                existing = await self._votes.active_for_session(
                    session_id,
                    transaction=transaction,
                )
                if existing is not None:
                    return existing
                options = self.AGENDA_OPTIONS if agenda_item_id is not None else self.OPTIONS
                vote = await self._votes.open(
                    session_id=session_id,
                    title=title,
                    options=options,
                    duration_seconds=duration_seconds,
                    agenda_item_id=agenda_item_id,
                    eligible_user_ids=eligible_user_ids,
                    now=now,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.VOTE_OPENED.value,
                    actor_user_id=actor_user_id,
                    source=source,
                    payload={
                        "vote_id": vote.id,
                        "agenda_item_id": vote.agenda_item_id,
                        "deadline_at": vote.deadline_at.isoformat(),
                        "option_ids": [option["id"] for option in options],
                        "eligible_count": len(set(eligible_user_ids)),
                    },
                )
                timer = await self._timers.start(
                    session_id=session_id,
                    timer_type=f"VOTE:{vote.id}",
                    duration_seconds=duration_seconds,
                    now=now,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.TIMER_STARTED.value,
                    source="SYSTEM",
                    payload={
                        "timer_id": timer.id,
                        "timer_type": timer.timer_type,
                        "deadline_at": timer.deadline_at.isoformat(),
                        "duration_seconds": duration_seconds,
                    },
                )
                return vote

    async def cast(
        self,
        *,
        vote_id: str,
        session_id: str,
        user_id: int,
        option_id: str,
        source: str,
        now: datetime | None = None,
    ) -> str:
        async with self._lock:
            async with self._votes.uow.begin() as transaction:
                vote = await self._votes.get(vote_id, transaction=transaction)
                if vote.session_id != session_id:
                    raise VoteNotOpenError("vote is not open for this Meeting Session")
                if vote.agenda_item_id is None:
                    await self._participants.require_present(
                        session_id=vote.session_id,
                        user_id=user_id,
                        transaction=transaction,
                    )
                selected = await self._votes.cast(
                    vote_id=vote_id,
                    user_id=user_id,
                    option_id=option_id,
                    now=now,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=vote.session_id,
                    event_type=MeetingEventType.VOTE_CAST.value,
                    actor_user_id=user_id,
                    source=source,
                    payload={"vote_id": vote_id, "option_id": selected},
                )
                return selected

    async def close(
        self,
        vote_id: str,
        *,
        actor_user_id: int | None,
        source: str,
        now: datetime | None = None,
    ) -> VoteOutcome:
        async with self._lock:
            async with self._votes.uow.begin() as transaction:
                return await self._close_in_transaction(
                    vote_id,
                    actor_user_id=actor_user_id,
                    source=source,
                    now=now,
                    transaction=transaction,
                )

    async def expire_timer_and_close_vote(
        self,
        *,
        timer: TimerRecord,
        now: datetime,
    ) -> VoteOutcome | None:
        """Expire a vote timer and close its vote under one lock and transaction."""

        if not timer.timer_type.startswith("VOTE:"):
            raise ValueError("timer is not associated with a vote")
        vote_id = timer.timer_type.removeprefix("VOTE:")
        async with self._lock:
            async with self._votes.uow.begin() as transaction:
                vote = await self._votes.get(vote_id, transaction=transaction)
                if vote.session_id != timer.session_id:
                    raise VoteNotOpenError("vote timer does not belong to its Meeting Session")
                _, changed = await self._timers.expire(
                    timer.id,
                    transaction=transaction,
                )
                if not changed:
                    return None
                outcome = await self._close_in_transaction(
                    vote_id,
                    actor_user_id=None,
                    source="TIMER",
                    now=now,
                    transaction=transaction,
                )
                await transaction.append_event(
                    session_id=timer.session_id,
                    event_type=MeetingEventType.TIMER_EXPIRED.value,
                    payload={"timer_id": timer.id, "timer_type": timer.timer_type},
                )
                return outcome

    async def _close_in_transaction(
        self,
        vote_id: str,
        *,
        actor_user_id: int | None,
        source: str,
        now: datetime | None,
        transaction: MeetingComponentTransaction,
    ) -> VoteOutcome:
        result: VoteCloseResult = await self._votes.close(
            vote_id,
            now=now,
            transaction=transaction,
        )
        outcome = VoteOutcome(
            result.vote,
            result.counts,
            result.eligible_count,
            result.changed,
        )
        if result.changed:
            await transaction.append_event(
                session_id=result.vote.session_id,
                event_type=MeetingEventType.VOTE_CLOSED.value,
                actor_user_id=actor_user_id,
                source=source,
                payload={
                    "vote_id": vote_id,
                    "agenda_item_id": result.vote.agenda_item_id,
                    "vote_content": result.vote.title,
                    "counts": dict(result.counts),
                    "total": outcome.total,
                    "eligible_count": outcome.eligible_count,
                    "responded_count": outcome.total - outcome.counts.get("abstain", 0),
                    "result": outcome.result,
                    "opened_at": result.vote.opened_at.isoformat(),
                    "deadline_at": result.vote.deadline_at.isoformat(),
                    "closed_at": (
                        result.vote.closed_at.isoformat()
                        if result.vote.closed_at is not None
                        else None
                    ),
                },
            )
        return outcome

    async def active_for_session(self, session_id: str) -> VoteRecord | None:
        return await self._votes.active_for_session(session_id)

    async def get(self, vote_id: str) -> VoteRecord:
        return await self._votes.get(vote_id)

    async def latest_for_session(self, session_id: str) -> VoteRecord | None:
        return await self._votes.latest_for_session(session_id)

    async def recipients(self, vote_id: str) -> tuple[VoteRecipientRecord, ...]:
        return await self._votes.recipients(vote_id)

    async def mark_delivery(
        self,
        *,
        vote_id: str,
        user_id: int,
        delivery_status: str,
        dm_message_id: int | None = None,
    ) -> VoteRecipientRecord:
        return await self._votes.mark_delivery(
            vote_id=vote_id,
            user_id=user_id,
            delivery_status=delivery_status,
            dm_message_id=dm_message_id,
        )

    async def vote_for_dm_message(self, *, message_id: int, user_id: int) -> VoteRecord:
        return await self._votes.vote_for_dm_message(message_id=message_id, user_id=user_id)

    async def all_eligible_voted(self, vote_id: str) -> bool:
        vote = await self._votes.get(vote_id)
        if vote.agenda_item_id is None:
            return False
        eligible, responded = await self._votes.eligible_progress(vote_id)
        return responded >= eligible

    async def sync_deadline(
        self,
        vote_id: str,
        deadline_at: datetime,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> None:
        await self._votes.set_deadline(
            vote_id,
            deadline_at,
            transaction=transaction,
        )


class AutoVoteCloser(Protocol):
    async def close(
        self,
        vote_id: str,
        *,
        actor_user_id: int | None,
        source: str,
        now: datetime | None = None,
    ) -> VoteOutcome: ...

    async def expire_timer_and_close_vote(
        self,
        *,
        timer: TimerRecord,
        now: datetime,
    ) -> VoteOutcome | None: ...

    async def sync_deadline(
        self,
        vote_id: str,
        deadline_at: datetime,
        *,
        transaction: MeetingComponentTransaction | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TimerNotice:
    session_id: str
    timer_id: str
    timer_type: str
    event_type: MeetingEventType
    remaining_seconds: int
    vote_outcome: VoteOutcome | None = None


class TimerService:
    """Recover persistent timers and durably hand expiry facts to the Controller."""

    def __init__(
        self,
        *,
        timers: TimerRepository,
        events: MeetingEventStore,
        vote_closer: AutoVoteCloser | None = None,
    ) -> None:
        self._timers = timers
        self._events = events
        self._vote_closer = vote_closer
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        session_id: str,
        timer_type: str,
        duration_seconds: float,
        now: datetime | None = None,
    ) -> TimerRecord:
        async with self._lock:
            return await self._start_locked(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=duration_seconds,
                now=now,
            )

    async def _start_locked(
        self,
        *,
        session_id: str,
        timer_type: str,
        duration_seconds: float,
        now: datetime | None,
    ) -> TimerRecord:
        async with self._timers.uow.begin() as transaction:
            active = await self._timers.list_active(
                session_id,
                transaction=transaction,
            )
            existing = next((timer for timer in active if timer.timer_type == timer_type), None)
            if existing is not None:
                return existing
            timer = await self._timers.start(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=duration_seconds,
                now=now,
                transaction=transaction,
            )
            await transaction.append_event(
                session_id=session_id,
                event_type=MeetingEventType.TIMER_STARTED.value,
                payload={
                    "timer_id": timer.id,
                    "timer_type": timer.timer_type,
                    "deadline_at": timer.deadline_at.isoformat(),
                    "duration_seconds": duration_seconds,
                },
            )
            return timer

    async def start_after_handoff_barrier(
        self,
        *,
        session_id: str,
        timer_type: str,
        duration_seconds: float,
        now: datetime | None = None,
    ) -> TimerRecord | None:
        """Start a flow timer only when an older same-type expiry is drained.

        The pending-expiry check and timer start share the tick lock.  If tick wins the
        race, the Controller sees ``None`` and drains the durable expiry instead of
        replacing it.  If this method wins, it returns the still-active timer and a later
        tick keeps that timer's identity authoritative.
        """

        async with self._lock:
            pending = await self._durable_expiry_notices_locked(session_id)
            if any(notice.timer_type == timer_type for notice in pending):
                return None
            return await self._start_locked(
                session_id=session_id,
                timer_type=timer_type,
                duration_seconds=duration_seconds,
                now=now,
            )

    async def active(self, session_id: str) -> tuple[TimerRecord, ...]:
        """Return persistent running/paused timers for read-only status projection."""

        return await self._timers.list_active(session_id)

    async def _pause_locked(self, session_id: str, *, now: datetime | None = None) -> int:
        async with self._timers.uow.begin() as transaction:
            timers = await self._timers.pause_all(
                session_id,
                now=now,
                transaction=transaction,
            )
            for timer in timers:
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.TIMER_PAUSED.value,
                    payload={"timer_id": timer.id, "remaining_ms": timer.remaining_ms},
                )
            return len(timers)

    async def pause(self, session_id: str, *, now: datetime | None = None) -> int:
        async with self._lock:
            return await self._pause_locked(session_id, now=now)

    async def freeze_for_transition(
        self,
        session_id: str,
        transition: Callable[[], Awaitable[_TransitionResult]],
        *,
        now: datetime | None = None,
    ) -> _TransitionResult:
        """Serialize a persisted pause transition with timer freezing.

        The Meeting transition and timer projection use separate transactions, so a process
        crash can still occur between them. ``reconcile_timer_handoffs`` repairs that durable
        window from the append-only Meeting state before the Bot permits another timer tick.
        Holding this lock prevents an in-process tick from entering the same window.
        """

        async with self._lock:
            result = await transition()
            await self._pause_locked(session_id, now=now)
            return result

    async def durable_expiry_notices(self, session_id: str) -> tuple[TimerNotice, ...]:
        """Rebuild committed expiry handoffs without relying on process memory."""

        async with self._lock:
            return await self._durable_expiry_notices_locked(session_id)

    async def uncompleted_expiry_notices(self, session_id: str) -> tuple[TimerNotice, ...]:
        """Return every exact unacknowledged expiry for terminal cleanup.

        Normal recovery ignores an expiry superseded by a newer timer of the same type.
        A terminal Session has no future flow to protect, so obsolete expiry facts also
        need an acknowledgement or terminal polling never converges.
        """

        async with self._lock:
            events = await self._events.list_types(
                session_id,
                (
                    MeetingEventType.TIMER_EXPIRED.value,
                    MeetingEventType.TIMER_HANDOFF_COMPLETED.value,
                ),
            )
            completed_timer_ids = {
                str(event.payload["timer_id"])
                for event in events
                if event.event_type == MeetingEventType.TIMER_HANDOFF_COMPLETED.value
                and isinstance(event.payload.get("timer_id"), str)
                and event.payload["timer_id"]
            }
            notices: list[TimerNotice] = []
            seen_timer_ids: set[str] = set()
            for event in events:
                if event.event_type != MeetingEventType.TIMER_EXPIRED.value:
                    continue
                timer_id = event.payload.get("timer_id")
                timer_type = event.payload.get("timer_type")
                if (
                    not isinstance(timer_id, str)
                    or not timer_id
                    or timer_id in seen_timer_ids
                    or timer_id in completed_timer_ids
                    or not isinstance(timer_type, str)
                    or not timer_type
                ):
                    continue
                seen_timer_ids.add(timer_id)
                notices.append(
                    TimerNotice(
                        session_id=session_id,
                        timer_id=timer_id,
                        timer_type=timer_type,
                        event_type=MeetingEventType.TIMER_EXPIRED,
                        remaining_seconds=0,
                    )
                )
            return tuple(notices)

    async def _durable_expiry_notices_locked(
        self,
        session_id: str,
    ) -> tuple[TimerNotice, ...]:
        """Project pending expiry facts while the caller owns the timer lock."""

        events = await self._events.list_types(session_id, _TIMER_HANDOFF_EVENT_TYPES)
        latest_timer_ids: dict[str, str] = {}
        completed_timer_ids: set[str] = set()
        for event in events:
            if event.event_type == MeetingEventType.TIMER_HANDOFF_COMPLETED.value:
                timer_id = event.payload.get("timer_id")
                if isinstance(timer_id, str) and timer_id:
                    completed_timer_ids.add(timer_id)
                continue
            if event.event_type != MeetingEventType.TIMER_STARTED.value:
                continue
            timer_id = event.payload.get("timer_id")
            timer_type = event.payload.get("timer_type")
            if (
                isinstance(timer_id, str)
                and timer_id
                and isinstance(timer_type, str)
                and timer_type
            ):
                latest_timer_ids[timer_type] = timer_id
        notices: list[TimerNotice] = []
        seen_timer_ids: set[str] = set()
        for event in events:
            if event.event_type != MeetingEventType.TIMER_EXPIRED.value:
                continue
            timer_id = event.payload.get("timer_id")
            timer_type = event.payload.get("timer_type")
            if (
                not isinstance(timer_id, str)
                or not timer_id
                or timer_id in seen_timer_ids
                or timer_id in completed_timer_ids
                or not isinstance(timer_type, str)
                or not timer_type
                or latest_timer_ids.get(timer_type) != timer_id
            ):
                continue
            seen_timer_ids.add(timer_id)
            notices.append(
                TimerNotice(
                    session_id=session_id,
                    timer_id=timer_id,
                    timer_type=timer_type,
                    event_type=MeetingEventType.TIMER_EXPIRED,
                    remaining_seconds=0,
                )
            )
        return tuple(notices)

    async def is_latest_timer_identity(self, notice: TimerNotice) -> bool:
        """Accept a notice only for the latest persisted instance of its timer type."""

        latest_timer_id: str | None = None
        for event in await self._events.list_types(
            notice.session_id,
            (MeetingEventType.TIMER_STARTED.value,),
        ):
            if event.event_type != MeetingEventType.TIMER_STARTED.value:
                continue
            if event.payload.get("timer_type") != notice.timer_type:
                continue
            timer_id = event.payload.get("timer_id")
            if isinstance(timer_id, str) and timer_id:
                latest_timer_id = timer_id
        return latest_timer_id == notice.timer_id

    async def complete_handoff(self, notice: TimerNotice) -> bool:
        """Persist successful consumption and suppress later retries of this handoff."""

        if notice.event_type is not MeetingEventType.TIMER_EXPIRED:
            return False
        async with self._lock:
            completed = await self._events.list_types(
                notice.session_id,
                (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
            )
            if any(event.payload.get("timer_id") == notice.timer_id for event in completed):
                return False
            await self._events.append(
                session_id=notice.session_id,
                event_type=MeetingEventType.TIMER_HANDOFF_COMPLETED.value,
                payload={"timer_id": notice.timer_id, "timer_type": notice.timer_type},
            )
            return True

    async def handoff_is_completed(self, notice: TimerNotice) -> bool:
        """Return whether this exact expiry already completed its Controller handoff."""

        if notice.event_type is not MeetingEventType.TIMER_EXPIRED:
            return False
        completed = await self._events.list_types(
            notice.session_id,
            (MeetingEventType.TIMER_HANDOFF_COMPLETED.value,),
        )
        return any(event.payload.get("timer_id") == notice.timer_id for event in completed)

    async def resume(self, session_id: str, *, now: datetime | None = None) -> int:
        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                timers = await self._timers.resume_all(
                    session_id,
                    now=now,
                    transaction=transaction,
                )
                for timer in timers:
                    if (
                        timer.status is TimerStatus.RUNNING
                        and timer.timer_type.startswith("VOTE:")
                        and self._vote_closer is not None
                    ):
                        await self._vote_closer.sync_deadline(
                            timer.timer_type.removeprefix("VOTE:"),
                            timer.deadline_at,
                            transaction=transaction,
                        )
                    await transaction.append_event(
                        session_id=session_id,
                        event_type=MeetingEventType.TIMER_RESUMED.value,
                        payload={
                            "timer_id": timer.id,
                            "deadline_at": timer.deadline_at.isoformat(),
                        },
                    )
                return len(timers)

    async def pause_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        """Pause only the controlled-floor timer owned by one operation."""

        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                timer = await self._timers.pause_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    now=now,
                    transaction=transaction,
                )
                if timer is None:
                    return False
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.TIMER_PAUSED.value,
                    payload={
                        "timer_id": timer.id,
                        "timer_type": timer.timer_type,
                        "remaining_ms": timer.remaining_ms,
                        "reason": reason,
                    },
                )
                return True

    async def resume_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        """Resume only a timer whose separate operation recorded ownership."""

        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                timer = await self._timers.resume_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    now=now,
                    transaction=transaction,
                )
                if timer is None:
                    return False
                if timer.timer_type.startswith("VOTE:") and self._vote_closer is not None:
                    await self._vote_closer.sync_deadline(
                        timer.timer_type.removeprefix("VOTE:"),
                        timer.deadline_at,
                        transaction=transaction,
                    )
                await transaction.append_event(
                    session_id=session_id,
                    event_type=MeetingEventType.TIMER_RESUMED.value,
                    payload={
                        "timer_id": timer.id,
                        "timer_type": timer.timer_type,
                        "deadline_at": timer.deadline_at.isoformat(),
                        "reason": reason,
                    },
                )
                return True

    async def cancel(self, session_id: str) -> int:
        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                count = await self._timers.cancel_all(
                    session_id,
                    transaction=transaction,
                )
                if count:
                    await transaction.append_event(
                        session_id=session_id,
                        event_type=MeetingEventType.TIMER_CANCELLED.value,
                        payload={"count": count},
                    )
                return count

    async def cancel_type(self, *, session_id: str, timer_type: str) -> int:
        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                count = await self._timers.cancel_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    transaction=transaction,
                )
                if count:
                    await transaction.append_event(
                        session_id=session_id,
                        event_type=MeetingEventType.TIMER_CANCELLED.value,
                        payload={"count": count, "timer_type": timer_type},
                    )
                return count

    async def extend_type(
        self,
        *,
        session_id: str,
        timer_type: str,
        additional_seconds: int,
    ) -> TimerRecord | None:
        """Extend one active timer and record the exact deadline as a fact."""

        async with self._lock:
            async with self._timers.uow.begin() as transaction:
                timer = await self._timers.extend_type(
                    session_id=session_id,
                    timer_type=timer_type,
                    additional_seconds=additional_seconds,
                    transaction=transaction,
                )
                if timer is not None:
                    if (
                        timer.status is TimerStatus.RUNNING
                        and timer.timer_type.startswith("VOTE:")
                        and self._vote_closer is not None
                    ):
                        await self._vote_closer.sync_deadline(
                            timer.timer_type.removeprefix("VOTE:"),
                            timer.deadline_at,
                            transaction=transaction,
                        )
                    await transaction.append_event(
                        session_id=session_id,
                        event_type=MeetingEventType.TIMER_EXTENDED.value,
                        payload={
                            "timer_id": timer.id,
                            "timer_type": timer.timer_type,
                            "additional_seconds": additional_seconds,
                            "deadline_at": timer.deadline_at.isoformat(),
                            "remaining_ms": timer.remaining_ms,
                        },
                    )
                return timer

    async def tick(self, *, now: datetime | None = None) -> tuple[TimerNotice, ...]:
        selected_now = now or utc_now()
        notices: list[TimerNotice] = []
        async with self._lock:
            active = [
                timer
                for timer in await self._timers.list_active()
                if timer.status is TimerStatus.RUNNING
            ]
            histories: dict[
                str,
                tuple[set[tuple[str, str]], dict[str, float]],
            ] = {}
            for timer in active:
                if timer.session_id not in histories:
                    events = await self._events.list_types(
                        timer.session_id,
                        _TIMER_TICK_EVENT_TYPES,
                    )
                    emitted = {
                        (event.event_type, str(event.payload.get("timer_id", "")))
                        for event in events
                    }
                    configured_durations: dict[str, float] = {}
                    for event in events:
                        if event.event_type != MeetingEventType.TIMER_STARTED.value:
                            continue
                        timer_id = str(event.payload.get("timer_id", ""))
                        duration = event.payload.get("duration_seconds")
                        if (
                            timer_id
                            and isinstance(duration, (int, float))
                            and not isinstance(duration, bool)
                            and duration > 0
                        ):
                            configured_durations[timer_id] = float(duration)
                    histories[timer.session_id] = (emitted, configured_durations)
                emitted, configured_durations = histories[timer.session_id]
                remaining = max(
                    0,
                    int((timer.deadline_at - selected_now).total_seconds() + 0.999),
                )
                if configured_durations.get(timer.id, 0) > 120:
                    threshold = 60
                    event_type = MeetingEventType.TIMER_REMINDER_60
                    marker = (event_type.value, timer.id)
                    if 0 < remaining <= threshold and marker not in emitted:
                        await self._events.append(
                            session_id=timer.session_id,
                            event_type=event_type.value,
                            payload={
                                "timer_id": timer.id,
                                "timer_type": timer.timer_type,
                                "remaining_seconds": remaining,
                            },
                        )
                        emitted.add(marker)
                        notices.append(
                            TimerNotice(
                                session_id=timer.session_id,
                                timer_id=timer.id,
                                timer_type=timer.timer_type,
                                event_type=event_type,
                                remaining_seconds=remaining,
                            )
                        )
                expired_marker = (MeetingEventType.TIMER_EXPIRED.value, timer.id)
                if remaining == 0 and expired_marker not in emitted:
                    vote_outcome: VoteOutcome | None = None
                    if timer.timer_type.startswith("VOTE:") and self._vote_closer is not None:
                        vote_outcome = await self._vote_closer.expire_timer_and_close_vote(
                            timer=timer,
                            now=selected_now,
                        )
                        if vote_outcome is None:
                            continue
                    else:
                        async with self._timers.uow.begin() as transaction:
                            _, changed = await self._timers.expire(
                                timer.id,
                                transaction=transaction,
                            )
                            if not changed:
                                continue
                            await transaction.append_event(
                                session_id=timer.session_id,
                                event_type=MeetingEventType.TIMER_EXPIRED.value,
                                payload={"timer_id": timer.id, "timer_type": timer.timer_type},
                            )
                    emitted.add(expired_marker)
                    notices.append(
                        TimerNotice(
                            session_id=timer.session_id,
                            timer_id=timer.id,
                            timer_type=timer.timer_type,
                            event_type=MeetingEventType.TIMER_EXPIRED,
                            remaining_seconds=0,
                            vote_outcome=vote_outcome,
                        )
                    )
        return tuple(notices)
