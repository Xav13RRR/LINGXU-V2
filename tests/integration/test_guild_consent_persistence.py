from pathlib import Path

import pytest

from lingxu.config import Settings
from lingxu.domain.enums import ConsentKind, MeetingState
from lingxu.meeting.factory import create_meeting_application
from lingxu.storage.component_repositories import (
    ConsentChangeRepository,
    GuildConsentRepository,
    ParticipantRepository,
)
from lingxu.storage.repositories import MeetingEventStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'guild-consent.db').as_posix()}",
        data_root=tmp_path / "sessions",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consent_auto_advances_and_is_reused_only_in_the_same_guild(
    tmp_path: Path,
) -> None:
    app = create_meeting_application(_settings(tmp_path))
    await app.database.create_schema()
    user_id = 1_001
    guild_one = 101
    guild_two = 202

    await app.controller.create(guild_id=guild_one, host_user_id=user_id)
    first_start = await app.controller.start(guild_id=guild_one, actor_user_id=user_id)
    assert first_start.session.state is MeetingState.CONSENT_CHECK

    consent_state = await app.controller.consent_all(guild_id=guild_one, user_id=user_id)
    assert consent_state is MeetingState.OPENING
    await app.controller.stop(guild_id=guild_one, actor_user_id=user_id)

    await app.controller.create(guild_id=guild_one, host_user_id=user_id)
    reused_start = await app.controller.start(guild_id=guild_one, actor_user_id=user_id)
    assert reused_start.session.state is MeetingState.OPENING
    await app.controller.stop(guild_id=guild_one, actor_user_id=user_id)

    await app.controller.create(guild_id=guild_two, host_user_id=user_id)
    isolated_start = await app.controller.start(guild_id=guild_two, actor_user_id=user_id)
    assert isolated_start.session.state is MeetingState.CONSENT_CHECK

    grants = GuildConsentRepository(app.database.sessions)
    assert await grants.grants_for(guild_id=guild_one, user_id=user_id)
    assert not await grants.grants_for(guild_id=guild_two, user_id=user_id)

    assert (
        await app.controller.consent_all(guild_id=guild_two, user_id=user_id)
        is MeetingState.OPENING
    )
    await app.controller.stop(guild_id=guild_two, actor_user_id=user_id)
    await app.controller.withdraw_all(guild_id=guild_one, user_id=user_id)
    assert not await grants.grants_for(guild_id=guild_one, user_id=user_id)
    assert await grants.grants_for(guild_id=guild_two, user_id=user_id)
    await app.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_withdraw_without_active_meeting_revokes_future_guild_consent(
    tmp_path: Path,
) -> None:
    app = create_meeting_application(_settings(tmp_path))
    await app.database.create_schema()
    guild_id = 303
    user_id = 3_003

    first = await app.controller.create(guild_id=guild_id, host_user_id=user_id)
    await app.controller.consent_all(guild_id=guild_id, user_id=user_id)
    await app.controller.stop(guild_id=guild_id, actor_user_id=user_id)

    await app.controller.withdraw_all(guild_id=guild_id, user_id=user_id)
    participants = ParticipantRepository(app.database.sessions)
    for kind in ConsentKind:
        assert user_id in await participants.consented_user_ids(
            session_id=first.id,
            kind=kind,
        )
    await app.controller.create(guild_id=guild_id, host_user_id=user_id)
    started = await app.controller.start(guild_id=guild_id, actor_user_id=user_id)

    assert started.session.state is MeetingState.CONSENT_CHECK
    grants = GuildConsentRepository(app.database.sessions)
    assert not await grants.grants_for(guild_id=guild_id, user_id=user_id)
    await app.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_temporal_consent_changes_append_events_and_only_update_active_session(
    tmp_path: Path,
) -> None:
    app = create_meeting_application(_settings(tmp_path))
    await app.database.create_schema()
    guild_id = 606
    user_id = 6_006
    participants = ParticipantRepository(app.database.sessions)
    guild_consents = GuildConsentRepository(app.database.sessions)
    changes = ConsentChangeRepository(app.database.sessions)
    event_store = MeetingEventStore(app.database.sessions)

    historical = await app.controller.create(guild_id=guild_id, host_user_id=user_id)
    await guild_consents.grant_all(guild_id=guild_id, user_id=user_id)
    await participants.set_consents(
        session_id=historical.id,
        user_id=user_id,
        grants=frozenset(ConsentKind),
    )
    await app.controller.stop(guild_id=guild_id, actor_user_id=user_id)

    active = await app.controller.create(guild_id=guild_id, host_user_id=user_id)
    granted = await changes.grant_all(
        session_id=active.id,
        guild_id=guild_id,
        user_id=user_id,
        effective_at_ms=0,
        trigger="inherited_guild_grant",
    )
    withdrawn = await changes.withdraw_all(
        session_id=active.id,
        guild_id=guild_id,
        user_id=user_id,
        effective_at_ms=12_345,
        trigger="slash_command",
    )

    events = await event_store.list(active.id)
    assert [event.event_type for event in events[-2:]] == [
        "CONSENT_GRANTED",
        "CONSENT_WITHDRAWN",
    ]
    assert granted.payload == {
        "subject_user_id": user_id,
        "consent_kinds": [kind.value for kind in ConsentKind],
        "effective_at_ms": 0,
        "trigger": "inherited_guild_grant",
        "scope": "participant",
    }
    assert withdrawn.payload == {
        "subject_user_id": user_id,
        "consent_kinds": [kind.value for kind in ConsentKind],
        "effective_at_ms": 12_345,
        "trigger": "slash_command",
        "scope": "participant",
    }
    assert withdrawn.actor_user_id == user_id
    assert not await guild_consents.grants_for(guild_id=guild_id, user_id=user_id)
    for kind in ConsentKind:
        assert user_id not in await participants.consented_user_ids(
            session_id=active.id,
            kind=kind,
        )
        assert user_id in await participants.consented_user_ids(
            session_id=historical.id,
            kind=kind,
        )
    await app.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consent_check_waits_for_every_present_participant(tmp_path: Path) -> None:
    app = create_meeting_application(_settings(tmp_path))
    await app.database.create_schema()
    guild_id = 404
    host_id = 4_001
    participant_id = 4_002

    await app.controller.create(guild_id=guild_id, host_user_id=host_id)
    await app.controller.participant_join(guild_id=guild_id, user_id=participant_id)
    await app.controller.consent_all(guild_id=guild_id, user_id=host_id)
    started = await app.controller.start(guild_id=guild_id, actor_user_id=host_id)
    assert started.session.state is MeetingState.CONSENT_CHECK

    final_state = await app.controller.consent_all(
        guild_id=guild_id,
        user_id=participant_id,
    )
    assert final_state is MeetingState.OPENING
    await app.database.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consent_check_advances_when_the_only_unconsented_participant_leaves(
    tmp_path: Path,
) -> None:
    app = create_meeting_application(_settings(tmp_path))
    await app.database.create_schema()
    guild_id = 505
    host_id = 5_001
    leaving_id = 5_002

    await app.controller.create(guild_id=guild_id, host_user_id=host_id)
    await app.controller.participant_join(guild_id=guild_id, user_id=leaving_id)
    await app.controller.consent_all(guild_id=guild_id, user_id=host_id)
    started = await app.controller.start(guild_id=guild_id, actor_user_id=host_id)
    assert started.session.state is MeetingState.CONSENT_CHECK

    await app.controller.participant_leave(guild_id=guild_id, user_id=leaving_id)
    assert (await app.controller.active(guild_id)).state is MeetingState.OPENING
    await app.database.close()
