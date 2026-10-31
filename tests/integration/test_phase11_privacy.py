from pathlib import Path

import pytest
from sqlalchemy import select

from lingxu.domain.errors import MeetingPermissionError
from lingxu.scenario.anonymous_vote import AnonymousVotingService
from lingxu.scenario.permissions import ScenarioRole
from lingxu.scenario.privacy import AnonymousVoteTokenService, PrivateContentVault
from lingxu.storage.component_repositories import (
    ParticipantRepository,
    TimerRepository,
    VoteRepository,
)
from lingxu.storage.database import Database
from lingxu.storage.encryption import FileCipher
from lingxu.storage.repositories import MeetingEventStore, MeetingSessionRepository
from lingxu.storage.services import MeetingSessionService
from lingxu.storage.session_files import SessionFiles
from lingxu.storage.tables import MeetingVoteEntryRow


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anonymous_vote_is_unlinkable_in_rows_events_and_across_votes(
    tmp_path: Path,
) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'privacy.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    files = SessionFiles(tmp_path / "sessions")
    session = await MeetingSessionService(sessions, files).create(
        guild_id=11,
        creator_user_id=100,
    )
    participants = ParticipantRepository(database.sessions)
    for user_id in (100, 200):
        await participants.join(session_id=session.id, user_id=user_id)
    token_service = AnonymousVoteTokenService(b"a" * 32)
    events = MeetingEventStore(database.sessions)
    service = AnonymousVotingService(
        votes=VoteRepository(database.sessions),
        timers=TimerRepository(database.sessions),
        participants=participants,
        events=events,
        tokens=token_service,
    )
    vote = await service.open(
        session_id=session.id,
        title="Secret ballot",
        duration_seconds=60,
        actor_user_id=100,
    )
    assert vote.anonymous
    await service.cast(
        session_id=session.id,
        vote_id=vote.id,
        user_id=100,
        option_id="yes",
    )
    await service.cast(
        session_id=session.id,
        vote_id=vote.id,
        user_id=100,
        option_id="no",
    )
    await service.cast(
        session_id=session.id,
        vote_id=vote.id,
        user_id=200,
        option_id="yes",
    )
    outcome = await service.close(vote.id)
    assert outcome.counts == {"yes": 1, "no": 1, "abstain": 0}

    async with database.sessions() as db:
        rows = list(
            await db.scalars(
                select(MeetingVoteEntryRow).where(MeetingVoteEntryRow.vote_id == vote.id)
            )
        )
    assert len(rows) == 2
    assert all(row.user_id < 0 for row in rows)
    assert all(row.user_id not in {100, 200} for row in rows)
    assert all(row.voter_token_hash and len(row.voter_token_hash) == 64 for row in rows)
    vote_events = [
        event for event in await events.list(session.id) if event.event_type == "VOTE_CAST"
    ]
    assert all(event.actor_user_id is None for event in vote_events)
    assert all("option_id" not in event.payload for event in vote_events)

    first = token_service.issue(session_id=session.id, vote_id=vote.id, user_id=100)
    second = token_service.issue(session_id=session.id, vote_id="other-vote", user_id=100)
    other_session = token_service.issue(session_id="other-session", vote_id=vote.id, user_id=100)
    assert len({first.token_hash, second.token_hash, other_session.token_hash}) == 3
    await database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_private_content_is_encrypted_and_role_scoped(tmp_path: Path) -> None:
    database = Database.open(f"sqlite+aiosqlite:///{(tmp_path / 'vault.db').as_posix()}")
    await database.create_schema()
    sessions = MeetingSessionRepository(database.sessions)
    files = SessionFiles(tmp_path / "sessions")
    session = await MeetingSessionService(sessions, files).create(
        guild_id=12,
        creator_user_id=100,
    )
    vault = PrivateContentVault(files=files, cipher=FileCipher(b"b" * 32))
    secret = "facilitator-only secret"
    reference = vault.put(
        session_id=session.id,
        audience=frozenset({ScenarioRole.FACILITATOR}),
        content=secret,
    )
    assert secret.encode() not in reference.encrypted_path.read_bytes()
    with pytest.raises(MeetingPermissionError):
        vault.read(reference, role=ScenarioRole.PARTICIPANT)
    assert vault.read(reference, role=ScenarioRole.FACILITATOR) == secret
    assert vault.read(reference, role=ScenarioRole.HOST) == secret
    await database.close()
