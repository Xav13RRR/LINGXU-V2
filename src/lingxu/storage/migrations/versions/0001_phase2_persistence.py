"""Create the Phase 2 persistence schema.

Revision ID: 0001_phase2
Revises:
Create Date: 2026-07-27
"""

from collections.abc import Sequence

from alembic import op

from lingxu.storage.database import install_sqlite_append_only_guards
from lingxu.storage.tables import Base

revision: str = "0001_phase2"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create every independent Lingxu table and append-only guard."""

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    if bind.dialect.name == "sqlite":
        install_sqlite_append_only_guards(bind)


def downgrade() -> None:
    """Drop the Phase 2 schema."""

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        bind.exec_driver_sql("DROP TRIGGER IF EXISTS meeting_events_reject_update")
        bind.exec_driver_sql("DROP TRIGGER IF EXISTS meeting_events_reject_delete")
    Base.metadata.drop_all(bind=bind)
