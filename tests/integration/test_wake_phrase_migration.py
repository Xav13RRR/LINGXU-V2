import sqlite3
from pathlib import Path

from lingxu.storage.migrate import downgrade_database, upgrade_database


def _database_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


def test_0015_wakeup_settings_upgrade_and_downgrade(tmp_path: Path) -> None:
    database_path = tmp_path / "wakeup-migration.db"
    database_url = _database_url(database_path)
    upgrade_database(database_url, "0014_pronunciations")

    with sqlite3.connect(database_path) as connection:
        # Revision 0001 calls current Base.metadata.create_all(), so simulate a database
        # originally created by the pre-0015 application before testing the incremental DDL.
        connection.execute("DROP TABLE guild_wakeup_settings")
        connection.commit()
        tables_before = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "guild_wakeup_settings" not in tables_before

    upgrade_database(database_url)
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(guild_wakeup_settings)")
        }
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        row_count = connection.execute("SELECT COUNT(*) FROM guild_wakeup_settings").fetchone()
    assert set(columns) == {
        "guild_id",
        "wake_phrase",
        "updated_by_user_id",
        "created_at",
        "updated_at",
    }
    assert columns["guild_id"][5] == 1
    assert columns["wake_phrase"][3] == 1
    assert columns["updated_by_user_id"][3] == 0
    assert version == ("0015_guild_wakeup",)
    assert row_count == (0,)

    downgrade_database(database_url, "0014_pronunciations")
    with sqlite3.connect(database_path) as connection:
        tables_after = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert "guild_wakeup_settings" not in tables_after
    assert version == ("0014_pronunciations",)
