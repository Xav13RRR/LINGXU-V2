from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from lingxu.domain.enums import AgendaStatus, TimerStatus, VoteStatus
from lingxu.domain.errors import VoteNotOpenError
from lingxu.meeting.components import AgendaService, TimerService, VotingService
from lingxu.scenario.anonymous_vote import AnonymousVotingService
from lingxu.scenario.privacy import AnonymousVoteTokenService
from lingxu.storage import component_repositories as component_repository_module
from lingxu.storage.component_repositories import (
    AgendaRepository,
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    append_meeting_event_row,
)
from lingxu.storage.tables import (
    MeetingAgendaItemRow,
    MeetingEventRow,
    MeetingSessionRow,
    MeetingTimerRow,
    MeetingVoteEntryRow,
    MeetingVoteRow,
)


class InjectedEventFailure(RuntimeError):
    pass


@contextmanager
def _fail_event_after_flush(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on: int = 1,
) -> Iterator[None]:
    original = append_meeting_event_row
    calls = 0

    async def failing_append(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        row = await original(*args, **kwargs)
        if calls == fail_on:
            raise InjectedEventFailure("injected after event flush")
        return row

    with monkeypatch.context() as scoped:
        scoped.setattr(component_repository_module, "append_meeting_event_row", failing_append)
        yield


@dataclass(slots=True)
class AtomicityStack:
    database: Database
    session_id: str
    events: MeetingEventStore
    participants: ParticipantRepository
    agenda_repository: AgendaRepository
    agenda: AgendaService
    timer_repository: TimerRepository
    timers: TimerService
    vote_repository: VoteRepository
    voting: VotingService


async def _stack(tmp_path: Path, *, guild_id: int) -> AtomicityStack:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'atomicity.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session = await sessions.create(
        guild_id=guild_id,
        creator_user_id=100,
        manifest_path=tmp_path / "manifest.json",
    )
    events = MeetingEventStore(database.sessions)
    participants = ParticipantRepository(database.sessions)
    agenda_repository = AgendaRepository(database.sessions)
    timer_repository = TimerRepository(database.sessions)
    vote_repository = VoteRepository(database.sessions)
    voting = VotingService(
        votes=vote_repository,
        timers=timer_repository,
        participants=participants,
        events=events,
    )
    timers = TimerService(
        timers=timer_repository,
        events=events,
        vote_closer=voting,
    )
    return AtomicityStack(
        database=database,
        session_id=session.id,
        events=events,
        participants=participants,
        agenda_repository=agenda_repository,
        agenda=AgendaService(agenda_repository, events),
        timer_repository=timer_repository,
        timers=timers,
        vote_repository=vote_repository,
        voting=voting,
    )


async def _database_snapshot(stack: AtomicityStack) -> tuple[int, int, int, int, int, int]:
    async with stack.database.sessions() as db:
        session = await db.get(MeetingSessionRow, stack.session_id)
        assert session is not None
        return (
            int(
                await db.scalar(
                    select(func.count(MeetingAgendaItemRow.id)).where(
                        MeetingAgendaItemRow.session_id == stack.session_id
                    )
                )
                or 0
            ),
            int(
                await db.scalar(
                    select(func.count(MeetingTimerRow.id)).where(
                        MeetingTimerRow.session_id == stack.session_id
                    )
                )
                or 0
            ),
            int(
                await db.scalar(
                    select(func.count(MeetingVoteRow.id)).where(
                        MeetingVoteRow.session_id == stack.session_id
                    )
                )
                or 0
            ),
            int(await db.scalar(select(func.count(MeetingVoteEntryRow.id))) or 0),
            int(
                await db.scalar(
                    select(func.count(MeetingEventRow.id)).where(
                        MeetingEventRow.session_id == stack.session_id
                    )
                )
                or 0
            ),
            session.next_sequence,
        )


async def _timer_status(stack: AtomicityStack, timer_id: str) -> TimerStatus:
    async with stack.database.sessions() as db:
        row = await db.get(MeetingTimerRow, timer_id)
        assert row is not None
        return TimerStatus(row.status)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agenda_projection_and_each_event_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1701)
    try:
        baseline = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.agenda.add(
                    session_id=stack.session_id,
                    title="will roll back",
                    actor_user_id=100,
                    source="TEST",
                )
        assert await _database_snapshot(stack) == baseline
        assert await stack.agenda.list(stack.session_id) == ()

        first = await stack.agenda.add(
            session_id=stack.session_id,
            title="first",
            actor_user_id=100,
            source="TEST",
        )
        await stack.agenda.add(
            session_id=stack.session_id,
            title="second",
            actor_user_id=100,
            source="TEST",
        )
        await stack.agenda.add(
            session_id=stack.session_id,
            title="third",
            actor_user_id=100,
            source="TEST",
        )

        before_remove = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.agenda.remove(
                    session_id=stack.session_id,
                    position=2,
                    actor_user_id=100,
                    source="TEST",
                )
        assert await _database_snapshot(stack) == before_remove
        assert [item.title for item in await stack.agenda.list(stack.session_id)] == [
            "first",
            "second",
            "third",
        ]

        before_reorder = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.agenda.reorder(
                    session_id=stack.session_id,
                    from_position=1,
                    to_position=3,
                    actor_user_id=100,
                    source="TEST",
                )
        assert await _database_snapshot(stack) == before_reorder
        assert next(item.id for item in await stack.agenda.list(stack.session_id)) == first.id

        before_activate = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.agenda.activate_next(session_id=stack.session_id)
        assert await _database_snapshot(stack) == before_activate
        assert (await stack.agenda.list(stack.session_id))[0].status is AgendaStatus.PENDING

        activated = await stack.agenda.activate_next(session_id=stack.session_id)
        assert activated is not None
        before_complete = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.agenda.complete_active(session_id=stack.session_id)
        assert await _database_snapshot(stack) == before_complete
        assert (await stack.agenda.list(stack.session_id))[0].status is AgendaStatus.ACTIVE
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_rows_timer_and_events_share_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1702)
    await stack.participants.join(session_id=stack.session_id, user_id=200)
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    try:
        baseline = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch, fail_on=2):
            with pytest.raises(InjectedEventFailure):
                await stack.voting.open(
                    session_id=stack.session_id,
                    title="middle failure",
                    duration_seconds=60,
                    actor_user_id=100,
                    source="TEST",
                    now=now,
                )
        assert await _database_snapshot(stack) == baseline

        vote = await stack.voting.open(
            session_id=stack.session_id,
            title="atomic vote",
            duration_seconds=60,
            actor_user_id=100,
            source="TEST",
            now=now,
        )
        before_retry = await _database_snapshot(stack)
        retried = await stack.voting.open(
            session_id=stack.session_id,
            title="ignored retry",
            duration_seconds=60,
            actor_user_id=100,
            source="TEST",
            now=now,
        )
        assert retried.id == vote.id
        assert await _database_snapshot(stack) == before_retry

        before_cast = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.voting.cast(
                    vote_id=vote.id,
                    session_id=stack.session_id,
                    user_id=200,
                    option_id="yes",
                    source="TEST",
                    now=now + timedelta(seconds=1),
                )
        assert await _database_snapshot(stack) == before_cast

        await stack.voting.cast(
            vote_id=vote.id,
            session_id=stack.session_id,
            user_id=200,
            option_id="yes",
            source="TEST",
            now=now + timedelta(seconds=1),
        )
        before_close = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.voting.close(
                    vote.id,
                    actor_user_id=100,
                    source="TEST",
                    now=now + timedelta(seconds=2),
                )
        assert await _database_snapshot(stack) == before_close
        assert (await stack.vote_repository.get(vote.id)).status is VoteStatus.OPEN

        await stack.voting.close(
            vote.id,
            actor_user_id=100,
            source="TEST",
            now=now + timedelta(seconds=2),
        )
        anonymous = AnonymousVotingService(
            votes=stack.vote_repository,
            timers=stack.timer_repository,
            participants=stack.participants,
            events=stack.events,
            tokens=AnonymousVoteTokenService(b"a" * 32),
        )
        before_anonymous_open = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch, fail_on=2):
            with pytest.raises(InjectedEventFailure):
                await anonymous.open(
                    session_id=stack.session_id,
                    title="anonymous middle failure",
                    duration_seconds=60,
                    actor_user_id=100,
                    now=now,
                )
        assert await _database_snapshot(stack) == before_anonymous_open

        anonymous_vote = await anonymous.open(
            session_id=stack.session_id,
            title="anonymous atomic vote",
            duration_seconds=60,
            actor_user_id=100,
            now=now,
        )
        before_anonymous_cast = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await anonymous.cast(
                    session_id=stack.session_id,
                    vote_id=anonymous_vote.id,
                    user_id=200,
                    option_id="yes",
                    now=now + timedelta(seconds=1),
                )
        assert await _database_snapshot(stack) == before_anonymous_cast

        before_anonymous_close = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await anonymous.close(anonymous_vote.id)
        assert await _database_snapshot(stack) == before_anonymous_close
        assert (await stack.vote_repository.get(anonymous_vote.id)).status is VoteStatus.OPEN
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timer_mutations_and_completion_roll_back_when_event_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1703)
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    try:
        baseline = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.start(
                    session_id=stack.session_id,
                    timer_type="TEST",
                    duration_seconds=60,
                    now=now,
                )
        assert await _database_snapshot(stack) == baseline

        timer = await stack.timers.start(
            session_id=stack.session_id,
            timer_type="TEST",
            duration_seconds=60,
            now=now,
        )
        before_retry = await _database_snapshot(stack)
        assert (
            await stack.timers.start(
                session_id=stack.session_id,
                timer_type="TEST",
                duration_seconds=60,
                now=now,
            )
        ).id == timer.id
        assert await _database_snapshot(stack) == before_retry

        before_pause = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.pause(stack.session_id, now=now + timedelta(seconds=10))
        assert await _database_snapshot(stack) == before_pause
        assert await _timer_status(stack, timer.id) is TimerStatus.RUNNING

        await stack.timers.pause(stack.session_id, now=now + timedelta(seconds=10))
        paused = (await stack.timer_repository.list_active(stack.session_id))[0]
        assert paused.status is TimerStatus.PAUSED
        before_resume = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.resume(stack.session_id, now=now + timedelta(seconds=20))
        assert await _database_snapshot(stack) == before_resume
        assert await _timer_status(stack, timer.id) is TimerStatus.PAUSED

        await stack.timers.resume(stack.session_id, now=now + timedelta(seconds=20))
        running_before_extend = (await stack.timer_repository.list_active(stack.session_id))[0]
        before_extend = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.extend_type(
                    session_id=stack.session_id,
                    timer_type="TEST",
                    additional_seconds=30,
                )
        assert await _database_snapshot(stack) == before_extend
        running_after_extend = (await stack.timer_repository.list_active(stack.session_id))[0]
        assert running_after_extend.deadline_at == running_before_extend.deadline_at

        before_cancel = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.cancel_type(
                    session_id=stack.session_id,
                    timer_type="TEST",
                )
        assert await _database_snapshot(stack) == before_cancel
        assert await _timer_status(stack, timer.id) is TimerStatus.RUNNING

        before_expire = await _database_snapshot(stack)
        with _fail_event_after_flush(monkeypatch):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.tick(now=now + timedelta(minutes=10))
        assert await _database_snapshot(stack) == before_expire
        assert await _timer_status(stack, timer.id) is TimerStatus.RUNNING

        notices = await stack.timers.tick(now=now + timedelta(minutes=10))
        assert notices[-1].timer_id == timer.id
        assert await _timer_status(stack, timer.id) is TimerStatus.EXPIRED
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_timer_expiry_and_close_share_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1704)
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    try:
        vote = await stack.voting.open(
            session_id=stack.session_id,
            title="atomic timed vote",
            duration_seconds=60,
            actor_user_id=100,
            source="TEST",
            now=now,
        )
        timer = (await stack.timer_repository.list_active(stack.session_id))[0]
        before_expiry = await _database_snapshot(stack)

        with _fail_event_after_flush(monkeypatch, fail_on=2):
            with pytest.raises(InjectedEventFailure):
                await stack.timers.tick(now=now + timedelta(seconds=60))

        assert await _database_snapshot(stack) == before_expiry
        assert await _timer_status(stack, timer.id) is TimerStatus.RUNNING
        assert (await stack.vote_repository.get(vote.id)).status is VoteStatus.OPEN

        notices = await stack.timers.tick(now=now + timedelta(seconds=60))
        assert len(notices) == 1
        assert notices[0].vote_outcome is not None
        assert await _timer_status(stack, timer.id) is TimerStatus.EXPIRED
        assert (await stack.vote_repository.get(vote.id)).status is VoteStatus.CLOSED
        event_types = [event.event_type for event in await stack.events.list(stack.session_id)]
        assert event_types[-2:] == ["VOTE_CLOSED", "TIMER_EXPIRED"]
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vote_timer_expiry_waits_for_voting_lock_before_database_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1705)
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    release_cast = asyncio.Event()
    cast_lookup_started = asyncio.Event()
    expiry_requested = asyncio.Event()
    expire_before_cast_release = False
    cast_task: asyncio.Task[str] | None = None
    tick_task: asyncio.Task[tuple[Any, ...]] | None = None
    try:
        await stack.participants.join(session_id=stack.session_id, user_id=200)
        vote = await stack.voting.open(
            session_id=stack.session_id,
            title="concurrent timed vote",
            duration_seconds=1,
            actor_user_id=100,
            source="TEST",
            now=now,
        )

        original_get = stack.vote_repository.get
        block_first_get = True

        async def observed_get(*args: Any, **kwargs: Any) -> Any:
            nonlocal block_first_get
            if block_first_get:
                block_first_get = False
                cast_lookup_started.set()
                await release_cast.wait()
            return await original_get(*args, **kwargs)

        original_expire = stack.timer_repository.expire

        async def observed_expire(*args: Any, **kwargs: Any) -> Any:
            nonlocal expire_before_cast_release
            if not release_cast.is_set():
                expire_before_cast_release = True
            return await original_expire(*args, **kwargs)

        original_vote_expiry = stack.voting.expire_timer_and_close_vote

        async def observed_vote_expiry(*args: Any, **kwargs: Any) -> Any:
            expiry_requested.set()
            return await original_vote_expiry(*args, **kwargs)

        monkeypatch.setattr(stack.vote_repository, "get", observed_get)
        monkeypatch.setattr(stack.timer_repository, "expire", observed_expire)
        monkeypatch.setattr(
            stack.voting,
            "expire_timer_and_close_vote",
            observed_vote_expiry,
        )

        cast_task = asyncio.create_task(
            stack.voting.cast(
                vote_id=vote.id,
                session_id=stack.session_id,
                user_id=200,
                option_id="yes",
                source="TEST",
                now=now + timedelta(milliseconds=500),
            )
        )
        await asyncio.wait_for(cast_lookup_started.wait(), timeout=2)
        tick_task = asyncio.create_task(stack.timers.tick(now=now + timedelta(seconds=2)))
        await asyncio.wait_for(expiry_requested.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not expire_before_cast_release

        release_cast.set()
        selected, notices = await asyncio.wait_for(
            asyncio.gather(cast_task, tick_task),
            timeout=5,
        )
        assert selected == "yes"
        assert len(notices) == 1
        assert notices[0].vote_outcome is not None
        assert notices[0].vote_outcome.counts["yes"] == 1
        assert not expire_before_cast_release
    finally:
        release_cast.set()
        pending = [task for task in (cast_task, tick_task) if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_vote_cast_rejects_cross_session_without_writes(tmp_path: Path) -> None:
    stack = await _stack(tmp_path, guild_id=1706)
    sessions = MeetingSessionRepository(stack.database.sessions)
    other = await sessions.create(
        guild_id=2706,
        creator_user_id=200,
        manifest_path=tmp_path / "other-public.json",
    )
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    try:
        await stack.participants.join(session_id=other.id, user_id=200)
        vote = await stack.voting.open(
            session_id=stack.session_id,
            title="session-bound public vote",
            duration_seconds=60,
            actor_user_id=100,
            source="TEST",
            now=now,
        )
        before = await _database_snapshot(stack)
        other_sequence = (await sessions.get(other.id)).next_sequence

        with pytest.raises(VoteNotOpenError):
            await stack.voting.cast(
                vote_id=vote.id,
                session_id=other.id,
                user_id=200,
                option_id="yes",
                source="TEST",
                now=now + timedelta(seconds=1),
            )

        assert await _database_snapshot(stack) == before
        assert (await sessions.get(other.id)).next_sequence == other_sequence
        assert await stack.events.list(other.id) == ()
        assert (await stack.vote_repository.get(vote.id)).status is VoteStatus.OPEN
    finally:
        await stack.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anonymous_vote_cast_rejects_cross_session_before_token_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = await _stack(tmp_path, guild_id=1707)
    sessions = MeetingSessionRepository(stack.database.sessions)
    other = await sessions.create(
        guild_id=2707,
        creator_user_id=200,
        manifest_path=tmp_path / "other-anonymous.json",
    )
    tokens = AnonymousVoteTokenService(b"b" * 32)
    anonymous = AnonymousVotingService(
        votes=stack.vote_repository,
        timers=stack.timer_repository,
        participants=stack.participants,
        events=stack.events,
        tokens=tokens,
    )
    token_issued = False
    original_issue = tokens.issue

    def observed_issue(*args: Any, **kwargs: Any) -> Any:
        nonlocal token_issued
        token_issued = True
        return original_issue(*args, **kwargs)

    monkeypatch.setattr(tokens, "issue", observed_issue)
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
    try:
        await stack.participants.join(session_id=other.id, user_id=200)
        vote = await anonymous.open(
            session_id=stack.session_id,
            title="session-bound anonymous vote",
            duration_seconds=60,
            actor_user_id=100,
            now=now,
        )
        before = await _database_snapshot(stack)
        other_sequence = (await sessions.get(other.id)).next_sequence

        with pytest.raises(VoteNotOpenError):
            await anonymous.cast(
                session_id=other.id,
                vote_id=vote.id,
                user_id=200,
                option_id="yes",
                now=now + timedelta(seconds=1),
            )

        assert not token_issued
        assert await _database_snapshot(stack) == before
        assert (await sessions.get(other.id)).next_sequence == other_sequence
        assert await stack.events.list(other.id) == ()
        assert (await stack.vote_repository.get(vote.id)).status is VoteStatus.OPEN
    finally:
        await stack.database.close()
