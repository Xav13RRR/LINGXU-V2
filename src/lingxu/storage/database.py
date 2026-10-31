"""Async SQLAlchemy engine configuration and SQLite safety settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from lingxu.storage.tables import Base

_APPEND_ONLY_TRIGGER_NAMES = (
    "meeting_events_reject_update",
    "meeting_events_reject_delete",
    "resident_assistant_events_reject_update",
    "resident_assistant_events_reject_delete",
    "service_capacity_events_reject_update",
    "service_capacity_events_reject_delete",
)


def sync_database_url(database_url: str) -> str:
    """Convert the supported async SQLite URL into an Alembic sync URL."""

    url = make_url(database_url)
    if url.drivername == "sqlite+aiosqlite":
        url = url.set(drivername="sqlite")
    return url.render_as_string(hide_password=False)


def ensure_sqlite_parent(database_url: str) -> None:
    """Create the parent of a file-backed SQLite database."""

    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database or url.database == ":memory:":
        return
    Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def install_sqlite_append_only_guards(connection: Connection) -> None:
    """Install database-level guards for the meeting event log."""

    execute = connection.exec_driver_sql
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS meeting_events_reject_update
        BEFORE UPDATE ON meeting_events
        BEGIN
            SELECT RAISE(ABORT, 'meeting_events is append-only');
        END
        """
    )
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS resident_assistant_events_reject_update
        BEFORE UPDATE ON resident_assistant_events
        BEGIN
            SELECT RAISE(ABORT, 'resident_assistant_events is append-only');
        END
        """
    )
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS resident_assistant_events_reject_delete
        BEFORE DELETE ON resident_assistant_events
        BEGIN
            SELECT RAISE(ABORT, 'resident_assistant_events is append-only');
        END
        """
    )
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS service_capacity_events_reject_update
        BEFORE UPDATE ON service_capacity_events
        BEGIN
            SELECT RAISE(ABORT, 'service_capacity_events is append-only');
        END
        """
    )
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS service_capacity_events_reject_delete
        BEFORE DELETE ON service_capacity_events
        BEGIN
            SELECT RAISE(ABORT, 'service_capacity_events is append-only');
        END
        """
    )
    execute(
        """
        CREATE TRIGGER IF NOT EXISTS meeting_events_reject_delete
        BEFORE DELETE ON meeting_events
        WHEN COALESCE(
            (SELECT state FROM meeting_sessions WHERE id = OLD.session_id),
            ''
        ) != 'DELETED'
        BEGIN
            SELECT RAISE(ABORT, 'meeting_events is append-only');
        END
        """
    )


@dataclass(slots=True)
class Database:
    """Owned async engine and session factory."""

    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]

    @classmethod
    def open(cls, database_url: str) -> Database:
        """Create an engine with SQLite WAL, foreign keys, and bounded lock waits."""

        ensure_sqlite_parent(database_url)
        engine = create_async_engine(database_url, pool_pre_ping=True)
        if make_url(database_url).drivername.startswith("sqlite"):

            @event.listens_for(engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_connection: Any, _: Any) -> None:
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA busy_timeout=5000")
                finally:
                    cursor.close()

        return cls(
            engine=engine,
            sessions=async_sessionmaker(engine, expire_on_commit=False),
        )

    async def create_schema(self) -> None:
        """Create tables for isolated tests and install append-only guards."""

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            if self.engine.url.drivername.startswith("sqlite"):
                await connection.run_sync(install_sqlite_append_only_guards)

    async def drop_schema(self) -> None:
        """Drop the isolated schema used by tests."""

        async with self.engine.begin() as connection:
            for trigger in _APPEND_ONLY_TRIGGER_NAMES:
                await connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger}")
            await connection.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Dispose all pooled connections."""

        await self.engine.dispose()
