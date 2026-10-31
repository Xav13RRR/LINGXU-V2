import sqlite3
from pathlib import Path

import pytest

from lingxu.storage.migrate import (
    DatabaseRevisionError,
    downgrade_database,
    require_database_current,
    upgrade_database,
)


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def test_alembic_upgrade_creates_phase7_schema_and_guards(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.db"
    database_url = _database_url(database_path)

    with pytest.raises(DatabaseRevisionError, match="database schema is not current"):
        require_database_current(database_url)

    upgrade_database(database_url)
    require_database_current(database_url)

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        triggers = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        participant_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(meeting_participants)")
        }
        vote_columns = {row[1] for row in connection.execute("PRAGMA table_info(meeting_votes)")}
        vote_indexes = {row[1] for row in connection.execute("PRAGMA index_list(meeting_votes)")}
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(meeting_sessions)")
        }
        transcript_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(transcript_entries)")
        }
        scenario_version_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(scenario_versions)")
        }
        vote_entry_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(meeting_vote_entries)")
        }
        vote_recipient_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(meeting_vote_recipients)")
        }
        agenda_import_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(agenda_imports)")
        }
        agenda_import_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(agenda_imports)")
        }
        resident_session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(resident_assistant_sessions)")
        }
        capacity_lease_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(service_capacity_leases)")
        }
        mute_lease_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(meeting_mute_leases)")
        }
        pronunciation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(guild_member_pronunciations)")
        }
        wakeup_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(guild_wakeup_settings)")
        }

    assert {
        "guild_settings",
        "guild_member_consents",
        "meeting_sessions",
        "meeting_participants",
        "meeting_agenda_items",
        "meeting_events",
        "meeting_state_snapshots",
        "meeting_timers",
        "meeting_votes",
        "meeting_vote_entries",
        "meeting_vote_recipients",
        "agenda_imports",
        "meeting_realtime_utterances",
        "meeting_intent_candidates",
        "meeting_playback_jobs",
        "processing_jobs",
        "audio_segments",
        "transcript_entries",
        "artifacts",
        "scenario_definitions",
        "scenario_versions",
        "resident_assistant_sessions",
        "resident_assistant_events",
        "service_capacity_leases",
        "service_capacity_events",
        "meeting_mute_leases",
        "guild_member_pronunciations",
        "guild_wakeup_settings",
    }.issubset(tables)
    assert {
        "meeting_events_reject_update",
        "meeting_events_reject_delete",
        "resident_assistant_events_reject_update",
        "resident_assistant_events_reject_delete",
        "service_capacity_events_reject_update",
        "service_capacity_events_reject_delete",
    }.issubset(triggers)
    assert {"consent_minutes", "display_name", "is_host"}.issubset(participant_columns)
    assert {"title", "deadline_at", "agenda_item_id"}.issubset(vote_columns)
    assert {"publication_channel_id", "setup_channel_id"}.issubset(session_columns)
    assert {"source_key"}.issubset(transcript_columns)
    assert {"status", "published_at"}.issubset(scenario_version_columns)
    assert {"voter_token_hash"}.issubset(vote_entry_columns)
    assert {"vote_id", "user_id", "delivery_status", "dm_message_id"}.issubset(
        vote_recipient_columns
    )
    assert "ix_meeting_votes_agenda_item_id" in vote_indexes
    assert {
        "session_id",
        "channel_id",
        "message_id",
        "attachment_id",
        "source_text",
        "status",
        "model_digest",
        "result_json",
        "validated_identity_user_ids_json",
    }.issubset(agenda_import_columns)
    assert "uq_agenda_imports_one_processing" in agenda_import_indexes
    assert {
        "guild_id",
        "channel_id",
        "creator_user_id",
        "status",
        "next_sequence",
        "active_marker",
    }.issubset(resident_session_columns)
    assert {
        "slot_name",
        "guild_id",
        "mode",
        "owner_id",
        "status",
        "next_sequence",
        "active_marker",
        "released_at",
    }.issubset(capacity_lease_columns)
    assert {
        "session_id",
        "agenda_item_id",
        "user_id",
        "original_server_mute",
        "ownership",
        "active_marker",
        "released_at",
    }.issubset(mute_lease_columns)
    assert {"guild_id", "user_id", "alias", "created_at", "updated_at"}.issubset(
        pronunciation_columns
    )
    assert {
        "guild_id",
        "wake_phrase",
        "updated_by_user_id",
        "created_at",
        "updated_at",
    }.issubset(wakeup_columns)
    assert version == ("0015_guild_wakeup",)

    downgrade_database(database_url, "base")
    with pytest.raises(DatabaseRevisionError, match="database schema is not current"):
        require_database_current(database_url)


def test_0009_repairs_missing_agenda_vote_index_after_partial_0008(tmp_path: Path) -> None:
    database_path = tmp_path / "partial-0008.db"
    database_url = _database_url(database_path)
    upgrade_database(database_url, "0008_agenda_private_votes")
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX ix_meeting_votes_agenda_item_id")
        connection.commit()

    upgrade_database(database_url)

    with sqlite3.connect(database_path) as connection:
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(meeting_votes)")}
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert "ix_meeting_votes_agenda_item_id" in indexes
    assert version == ("0015_guild_wakeup",)
