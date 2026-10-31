"""Programmatic and CLI Alembic migration entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from lingxu.config import Settings
from lingxu.storage.database import ensure_sqlite_parent, sync_database_url


class DatabaseRevisionError(RuntimeError):
    """Raised when an application process starts against a non-current schema."""


def migration_config(database_url: str) -> Config:
    """Build an Alembic configuration that also works in the packaged image."""

    ensure_sqlite_parent(database_url)
    config = Config()
    config.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parent / "migrations"),
    )
    config.set_main_option("sqlalchemy.url", sync_database_url(database_url))
    return config


def upgrade_database(database_url: str, revision: str = "head") -> None:
    """Upgrade the independent Lingxu database."""

    command.upgrade(migration_config(database_url), revision)


def downgrade_database(database_url: str, revision: str) -> None:
    """Downgrade the independent Lingxu database explicitly."""

    command.downgrade(migration_config(database_url), revision)


def require_database_current(database_url: str) -> None:
    """Fail closed unless every configured Alembic head is active.

    Schema mutation belongs to the dedicated migration process. Bot and Worker call this
    read-only guard so concurrent service startup cannot race through Alembic DDL.
    """

    config = migration_config(database_url)
    expected_heads = tuple(sorted(ScriptDirectory.from_config(config).get_heads()))
    engine = create_engine(sync_database_url(database_url))
    try:
        with engine.connect() as connection:
            current_heads = tuple(
                sorted(MigrationContext.configure(connection).get_current_heads())
            )
    finally:
        engine.dispose()
    if current_heads != expected_heads:
        raise DatabaseRevisionError(
            "database schema is not current; run the Lingxu migration command before "
            f"starting services (current={current_heads!r}, expected={expected_heads!r})"
        )


def main() -> None:
    """Run migrations against the configured independent database."""

    parser = argparse.ArgumentParser()
    parser.add_argument("revision", nargs="?", default="head")
    parser.add_argument("--downgrade", action="store_true")
    args = parser.parse_args()
    database_url = Settings().database_url
    if args.downgrade:
        downgrade_database(database_url, args.revision)
    else:
        upgrade_database(database_url, args.revision)


if __name__ == "__main__":
    main()
