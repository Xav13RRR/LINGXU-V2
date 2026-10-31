from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lingxu.domain.errors import VoteNotOpenError
from lingxu.meeting.components import TimerService, VotingService
from lingxu.storage.component_repositories import (
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository


@pytest.mark.integration
@pytest.mark.asyncio
async def test_paused_vote_freezes_cast_deadline_until_timer_resumes(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'paused-vote.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session = await sessions.create(
        guild_id=1,
        creator_user_id=100,
        manifest_path=tmp_path / "session.json",
    )
    events = MeetingEventStore(database.sessions)
    participants = ParticipantRepository(database.sessions)
    await participants.join(session_id=session.id, user_id=200)
    timer_repository = TimerRepository(database.sessions)
    voting = VotingService(
        votes=VoteRepository(database.sessions),
        timers=timer_repository,
        participants=participants,
        events=events,
    )
    timers = TimerService(
        timers=timer_repository,
        events=events,
        vote_closer=voting,
    )
    started_at = datetime(2026, 9, 2, tzinfo=UTC)
    vote = await voting.open(
        session_id=session.id,
        title="Pause semantics",
        duration_seconds=60,
        actor_user_id=100,
        source="TEST",
        now=started_at,
    )

    assert await timers.pause(session.id, now=started_at + timedelta(seconds=10)) == 1

    selected = await voting.cast(
        vote_id=vote.id,
        session_id=session.id,
        user_id=200,
        option_id="yes",
        source="TEST",
        now=started_at + timedelta(seconds=70),
    )
    assert selected == "yes"

    assert await timers.resume(session.id, now=started_at + timedelta(seconds=100)) == 1
    selected = await voting.cast(
        vote_id=vote.id,
        session_id=session.id,
        user_id=200,
        option_id="no",
        source="TEST",
        now=started_at + timedelta(seconds=149),
    )
    assert selected == "no"
    with pytest.raises(VoteNotOpenError, match="vote is not open"):
        await voting.cast(
            vote_id=vote.id,
            session_id=session.id,
            user_id=200,
            option_id="yes",
            source="TEST",
            now=started_at + timedelta(seconds=151),
        )
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True], ids=["public", "anonymous"])
async def test_elapsed_vote_cannot_be_revived_by_pausing_at_zero_remaining(
    tmp_path: Path,
    *,
    anonymous: bool,
) -> None:
    database = Database.open(
        f"sqlite+aiosqlite:///{(tmp_path / f'elapsed-{anonymous}.db').as_posix()}"
    )
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    session = await sessions.create(
        guild_id=2,
        creator_user_id=100,
        manifest_path=tmp_path / "session.json",
    )
    events = MeetingEventStore(database.sessions)
    timer_repository = TimerRepository(database.sessions)
    votes = VoteRepository(database.sessions)
    timers = TimerService(timers=timer_repository, events=events)
    started_at = datetime(2026, 9, 2, tzinfo=UTC)
    vote = await votes.open(
        session_id=session.id,
        title="Elapsed pause boundary",
        options=({"id": "yes", "label": "Yes"},),
        duration_seconds=60,
        anonymous=anonymous,
        now=started_at,
    )
    await timers.start(
        session_id=session.id,
        timer_type=f"VOTE:{vote.id}",
        duration_seconds=60,
        now=started_at,
    )

    assert await timers.pause(session.id, now=started_at + timedelta(seconds=61)) == 1
    paused = await timer_repository.list_active(session.id)
    assert len(paused) == 1
    assert paused[0].remaining_ms == 0

    if anonymous:
        with pytest.raises(VoteNotOpenError, match="anonymous vote is not open"):
            await votes.cast_anonymous(
                vote_id=vote.id,
                pseudonymous_user_id=-1,
                voter_token_hash="a" * 64,
                option_id="yes",
                now=started_at + timedelta(seconds=600),
            )
    else:
        with pytest.raises(VoteNotOpenError, match="vote is not open"):
            await votes.cast(
                vote_id=vote.id,
                user_id=200,
                option_id="yes",
                now=started_at + timedelta(seconds=600),
            )
    await database.close()
