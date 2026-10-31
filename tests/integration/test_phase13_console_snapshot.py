from pathlib import Path

import pytest

from lingxu.config import Settings
from lingxu.console.snapshot import ConsoleSnapshotStore
from lingxu.domain.models import RealtimeUtteranceRecord
from lingxu.meeting.factory import create_meeting_application
from lingxu.scenario.anonymous_vote import AnonymousVotingService
from lingxu.scenario.privacy import AnonymousVoteTokenService
from lingxu.storage.component_repositories import (
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.repositories import MeetingEventStore, RealtimeUtteranceRepository
from lingxu.utils import new_ulid


@pytest.mark.integration
@pytest.mark.asyncio
async def test_console_projection_is_guild_scoped_consent_filtered_and_vote_anonymous(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'console.db').as_posix()}",
        data_root=tmp_path / "sessions",
    )
    app = create_meeting_application(settings)
    await app.database.create_schema()
    guild_one = 101
    guild_two = 202
    first = await app.controller.create(guild_id=guild_one, host_user_id=1_001)
    await app.controller.participant_join(guild_id=guild_one, user_id=1_002)
    await app.controller.consent_all(guild_id=guild_one, user_id=1_001)
    await app.controller.agenda_add(
        guild_id=guild_one,
        actor_user_id=1_001,
        title="Console agenda",
    )
    await app.controller.speak(guild_id=guild_one, user_id=1_002)

    utterances = RealtimeUtteranceRepository(app.database.sessions)
    await utterances.add(
        RealtimeUtteranceRecord(
            id=new_ulid(),
            session_id=first.id,
            user_id=1_001,
            started_ms=0,
            ended_ms=1_000,
            raw_text="consented final",
            normalised_text=None,
        )
    )
    await utterances.add(
        RealtimeUtteranceRecord(
            id=new_ulid(),
            session_id=first.id,
            user_id=1_002,
            started_ms=1_000,
            ended_ms=2_000,
            raw_text="must not be projected",
            normalised_text=None,
        )
    )

    participants = ParticipantRepository(app.database.sessions)
    anonymous = AnonymousVotingService(
        votes=VoteRepository(app.database.sessions),
        timers=TimerRepository(app.database.sessions),
        participants=participants,
        events=MeetingEventStore(app.database.sessions),
        tokens=AnonymousVoteTokenService(b"a" * 32),
    )
    vote = await anonymous.open(
        session_id=first.id,
        title="Private vote",
        duration_seconds=60,
        actor_user_id=1_001,
    )
    await anonymous.cast(
        session_id=first.id,
        vote_id=vote.id,
        user_id=1_001,
        option_id="yes",
    )

    second = await app.controller.create(guild_id=guild_two, host_user_id=2_001)
    store_one = ConsoleSnapshotStore(
        database=app.database,
        controller=app.controller,
        sessions=app.sessions,
        guild_id=guild_one,
    )
    store_two = ConsoleSnapshotStore(
        database=app.database,
        controller=app.controller,
        sessions=app.sessions,
        guild_id=guild_two,
    )

    snapshot_one = await store_one.snapshot()
    assert snapshot_one["session_id"] == first.id
    assert snapshot_one["guild_id"] == str(guild_one)
    assert {person["user_id"] for person in snapshot_one["participants"]} == {
        "1001",
        "1002",
    }
    assert [item["text"] for item in snapshot_one["final_utterances"]] == ["consented final"]
    assert snapshot_one["vote"]["anonymous"] is True
    assert snapshot_one["vote"]["options"][0]["count"] == 1
    assert all("user_id" not in option for option in snapshot_one["vote"]["options"])
    assert all("actor_user_id" not in event for event in snapshot_one["recent_events"])
    assert all("payload" not in event for event in snapshot_one["recent_events"])

    snapshot_two = await store_two.snapshot()
    assert snapshot_two["session_id"] == second.id
    assert snapshot_two["guild_id"] == str(guild_two)
    assert [person["user_id"] for person in snapshot_two["participants"]] == ["2001"]
    assert snapshot_two["vote"] is None
    await app.database.close()
