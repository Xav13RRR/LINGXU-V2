import asyncio
from pathlib import Path

import pytest
from sqlalchemy import update

from lingxu.domain.enums import MeetingState
from lingxu.domain.errors import (
    InvalidStateTransitionError,
    MeetingPermissionError,
    MeetingRecoveryError,
)
from lingxu.domain.models import MeetingSessionRecord
from lingxu.meeting.events import MeetingEventType
from lingxu.meeting.runtime import MeetingRuntime, RuntimeResult
from lingxu.storage.database import Database
from lingxu.storage.repositories import (
    MeetingEventStore,
    MeetingSessionRepository,
    MeetingSnapshotStore,
)
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import MeetingSessionRow


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


async def _runtime(
    tmp_path: Path,
    *,
    guild_id: int = 1,
    host_id: int = 100,
    snapshot_interval: int = 3,
) -> tuple[
    Database,
    MeetingRuntime,
    MeetingSessionRepository,
    MeetingEventStore,
    MeetingSnapshotStore,
    str,
]:
    database = Database.open(_database_url(tmp_path / f"runtime-{guild_id}.db"))
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    events = MeetingEventStore(database.sessions)
    snapshots = MeetingSnapshotStore(database.sessions)
    session = await MeetingSessionService(
        sessions,
        SessionFiles(tmp_path / f"sessions-{guild_id}"),
    ).create(guild_id=guild_id, creator_user_id=host_id)
    return (
        database,
        MeetingRuntime(
            sessions=sessions,
            events=events,
            snapshots=snapshots,
            snapshot_interval=snapshot_interval,
        ),
        sessions,
        events,
        snapshots,
        session.id,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixed_template_runs_without_discord_and_recovers(tmp_path: Path) -> None:
    database, runtime, _, events, snapshots, session_id = await _runtime(tmp_path)

    initialised = await runtime.initialise(session_id, actor_user_id=100, source="SLASH_COMMAND")
    assert initialised.session.state is MeetingState.AWAITING_PARTICIPANTS
    started = await runtime.start(session_id, actor_user_id=100, source="SLASH_COMMAND")
    assert started.session.state is MeetingState.CONSENT_CHECK
    repeated_start = await runtime.start(session_id, actor_user_id=100, source="SLASH_COMMAND")
    assert repeated_start.event is None

    current = started
    while current.session.state is not MeetingState.PUBLISHED:
        current = await runtime.advance(session_id)

    assert current.session.state is MeetingState.PUBLISHED
    history = await events.list(session_id)
    assert [event.sequence_number for event in history] == list(range(1, 16))
    assert history[0].event_type == MeetingEventType.MEETING_INITIALISED
    assert history[-1].state_after is MeetingState.PUBLISHED
    latest = await snapshots.latest(session_id)
    assert latest is not None
    assert latest.sequence_number == 15
    recovered = await runtime.recover(session_id)
    assert recovered.session.state is MeetingState.PUBLISHED
    assert recovered.actions == ()

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pause_resume_back_and_stop_are_event_sourced_and_idempotent(
    tmp_path: Path,
) -> None:
    database, runtime, _, events, _, session_id = await _runtime(tmp_path, guild_id=2)
    await runtime.initialise(session_id, actor_user_id=100)
    await runtime.start(session_id, actor_user_id=100, source="SLASH_COMMAND")
    for _ in range(4):
        current = await runtime.advance(session_id)
    assert current.session.state is MeetingState.DISCUSSION_READY

    paused = await runtime.pause(session_id, actor_user_id=100, source="BUTTON")
    assert paused.session.state is MeetingState.PAUSED
    assert paused.actions == ("freeze_timers",)
    assert (await runtime.pause(session_id, actor_user_id=100, source="BUTTON")).event is None
    resumed = await runtime.resume(session_id, actor_user_id=100, source="BUTTON")
    assert resumed.session.state is MeetingState.DISCUSSION_READY
    assert resumed.actions == ("resume_timers",)

    before_back = await events.list(session_id)
    backed = await runtime.back(session_id, actor_user_id=100, source="SLASH_COMMAND")
    assert backed.session.state is MeetingState.AGENDA_ITEM_INTRO
    after_back = await events.list(session_id)
    assert len(after_back) == len(before_back) + 1
    assert after_back[-1].event_type == MeetingEventType.MEETING_BACK
    assert after_back[:-1] == before_back

    stopped = await runtime.stop(session_id, actor_user_id=100, source="SLASH_COMMAND")
    assert stopped.session.state is MeetingState.STOPPED
    assert (await runtime.stop(session_id, actor_user_id=100, source="SLASH_COMMAND")).event is None
    assert (await runtime.recover(session_id)).session.state is MeetingState.STOPPED

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_same_transition_returns_actions_only_to_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, runtime, sessions, events, _, session_id = await _runtime(
        tmp_path,
        guild_id=20,
    )
    original_get = sessions.get
    first_read = asyncio.Event()
    both_read = asyncio.Event()
    first_transition_finished = asyncio.Event()
    read_count = 0

    async def get_stale_before_first_commit(
        selected_session_id: str,
    ) -> MeetingSessionRecord:
        nonlocal read_count
        record = await original_get(selected_session_id)
        read_count += 1
        if read_count == 1:
            first_read.set()
            await both_read.wait()
        else:
            both_read.set()
            await first_transition_finished.wait()
        return record

    monkeypatch.setattr(sessions, "get", get_stale_before_first_commit)

    async def run_first() -> RuntimeResult:
        try:
            return await runtime.initialise(
                session_id,
                actor_user_id=100,
                source="SLASH_COMMAND",
            )
        finally:
            first_transition_finished.set()

    first = asyncio.create_task(run_first())
    await first_read.wait()
    second = asyncio.create_task(
        runtime.initialise(
            session_id,
            actor_user_id=100,
            source="SLASH_COMMAND",
        )
    )
    results = await asyncio.gather(first, second)

    history = await events.list(session_id)
    assert len(history) == 1
    assert sum(result.event is not None for result in results) == 1
    assert sum(bool(result.actions) for result in results) == 1
    assert [result.event.id for result in results if result.event is not None] == [history[0].id]

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_snapshot_failure_does_not_hide_committed_winner_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, runtime, sessions, events, snapshots, session_id = await _runtime(
        tmp_path,
        guild_id=23,
        snapshot_interval=1,
    )

    async def fail_snapshot(**_kwargs: object) -> object:
        raise OSError("synthetic snapshot write failure")

    monkeypatch.setattr(snapshots, "save", fail_snapshot)

    result = await runtime.initialise(
        session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )

    assert result.session.state is MeetingState.AWAITING_PARTICIPANTS
    assert result.event is not None
    assert result.actions == ("publish_host_panel", "request_consent")
    assert (await sessions.get(session_id)).state is MeetingState.AWAITING_PARTICIPANTS
    assert len(await events.list(session_id)) == 1
    repeated = await runtime.initialise(
        session_id,
        actor_user_id=100,
        source="SLASH_COMMAND",
    )
    assert repeated.event is None
    assert repeated.actions == ()

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transition_retries_when_an_event_advances_the_expected_sequence(
    tmp_path: Path,
) -> None:
    database, _, sessions, events, _, session_id = await _runtime(tmp_path, guild_id=21)
    expected = await sessions.get(session_id)
    concurrent_event = await events.append(
        session_id=session_id,
        event_type="CONCURRENT_FACT",
    )

    transition = await sessions.transition_with_event(
        session_id=session_id,
        expected_state=expected.state,
        expected_resume_state=expected.resume_state,
        expected_sequence=expected.next_sequence,
        target=MeetingState.AWAITING_PARTICIPANTS,
        actor_user_id=100,
        source="SLASH_COMMAND",
        event_type=MeetingEventType.MEETING_INITIALISED.value,
    )

    assert transition.changed is True
    assert transition.event is not None
    assert transition.event.sequence_number == 2
    assert transition.event.state_before is MeetingState.CREATED
    assert transition.event.state_after is MeetingState.AWAITING_PARTICIPANTS
    assert transition.session.next_sequence == 2
    assert [event.id for event in await events.list(session_id)] == [
        concurrent_event.id,
        transition.event.id,
    ]

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_only_transition_api_retries_a_stale_different_target(
    tmp_path: Path,
) -> None:
    database, _, sessions, events, _, session_id = await _runtime(tmp_path, guild_id=22)
    stale = await sessions.get(session_id)
    transitioned = await sessions.transition(
        session_id=session_id,
        target=MeetingState.AWAITING_PARTICIPANTS,
        actor_user_id=100,
        source="SYSTEM",
        event_type=MeetingEventType.MEETING_INITIALISED.value,
    )
    assert isinstance(transitioned, MeetingSessionRecord)
    assert transitioned.state is MeetingState.AWAITING_PARTICIPANTS

    stopped = await sessions.transition_with_event(
        session_id=session_id,
        expected_state=stale.state,
        expected_resume_state=stale.resume_state,
        expected_sequence=stale.next_sequence,
        target=MeetingState.STOPPED,
        actor_user_id=100,
        source="SLASH_COMMAND",
        event_type=MeetingEventType.MEETING_STOPPED.value,
    )

    assert stopped.changed is True
    assert stopped.event is not None
    assert stopped.event.state_before is MeetingState.AWAITING_PARTICIPANTS
    assert stopped.event.state_after is MeetingState.STOPPED
    assert stopped.session.state is MeetingState.STOPPED
    assert len(await events.list(session_id)) == 2

    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_guards_illegal_transition_and_recovery_mismatch(tmp_path: Path) -> None:
    database, runtime, sessions, _, _, session_id = await _runtime(tmp_path, guild_id=3)

    with pytest.raises(MeetingPermissionError):
        await runtime.initialise(session_id, actor_user_id=999, source="SLASH_COMMAND")
    with pytest.raises(InvalidStateTransitionError):
        await runtime.back(session_id, actor_user_id=100, source="SLASH_COMMAND")

    await runtime.initialise(session_id, actor_user_id=100)
    await sessions.transition(
        session_id=session_id,
        target=MeetingState.CONSENT_CHECK,
        actor_user_id=None,
        source="SYSTEM",
    )
    async with database.sessions() as db, db.begin():
        await db.execute(
            update(MeetingSessionRow)
            .where(MeetingSessionRow.id == session_id)
            .values(state=MeetingState.OPENING.value)
        )
    with pytest.raises(MeetingRecoveryError, match="materialised state"):
        await runtime.recover(session_id)

    await database.close()
