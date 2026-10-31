"""Add append-only resident voice-assistant operational leases.

Revision ID: 0011_resident_assistant
Revises: 0010_agenda_import
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_resident_assistant"
down_revision: str | None = "0010_agenda_import"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "resident_assistant_sessions" not in tables:
        op.create_table(
            "resident_assistant_sessions",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("guild_id", sa.BigInteger(), nullable=False),
            sa.Column("channel_id", sa.BigInteger(), nullable=False),
            sa.Column("creator_user_id", sa.BigInteger(), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("next_sequence", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("active_marker", sa.Integer(), nullable=True, server_default="1"),
            sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("guild_id", "active_marker"),
        )
        op.create_index(
            "ix_resident_assistant_sessions_guild_id",
            "resident_assistant_sessions",
            ["guild_id"],
        )
    if "resident_assistant_events" not in tables:
        op.create_table(
            "resident_assistant_events",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("session_id", sa.String(length=26), nullable=False),
            sa.Column("sequence_number", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(length=40), nullable=False),
            sa.Column("actor_user_id", sa.BigInteger(), nullable=True),
            sa.Column("reason", sa.String(length=80), nullable=False),
            sa.Column(
                "occurred_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.ForeignKeyConstraint(
                ["session_id"], ["resident_assistant_sessions.id"], ondelete="RESTRICT"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("session_id", "sequence_number"),
        )
        op.create_index(
            "ix_resident_assistant_events_session_id",
            "resident_assistant_events",
            ["session_id"],
        )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS resident_assistant_events_reject_update
        BEFORE UPDATE ON resident_assistant_events
        BEGIN
            SELECT RAISE(ABORT, 'resident_assistant_events is append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS resident_assistant_events_reject_delete
        BEFORE DELETE ON resident_assistant_events
        BEGIN
            SELECT RAISE(ABORT, 'resident_assistant_events is append-only');
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS resident_assistant_events_reject_delete")
    op.execute("DROP TRIGGER IF EXISTS resident_assistant_events_reject_update")
    op.drop_index("ix_resident_assistant_events_session_id", table_name="resident_assistant_events")
    op.drop_table("resident_assistant_events")
    op.drop_index(
        "ix_resident_assistant_sessions_guild_id", table_name="resident_assistant_sessions"
    )
    op.drop_table("resident_assistant_sessions")
