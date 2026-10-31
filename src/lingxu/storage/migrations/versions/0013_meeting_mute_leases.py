"""Add recoverable Discord server-mute leases.

Revision ID: 0013_meeting_mute_leases
Revises: 0012_service_capacity
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_meeting_mute_leases"
down_revision: str | None = "0012_service_capacity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if "meeting_mute_leases" in set(sa.inspect(bind).get_table_names()):
        return
    op.create_table(
        "meeting_mute_leases",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("session_id", sa.String(length=26), nullable=False),
        sa.Column("agenda_item_id", sa.String(length=26), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("original_server_mute", sa.Boolean(), nullable=False),
        sa.Column("ownership", sa.String(length=24), nullable=False),
        sa.Column("active_marker", sa.Integer(), nullable=True, server_default="1"),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["session_id"], ["meeting_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "user_id", "active_marker"),
    )
    op.create_index(
        "ix_meeting_mute_leases_session_id",
        "meeting_mute_leases",
        ["session_id"],
    )
    op.create_index(
        "ix_meeting_mute_leases_agenda_item_id",
        "meeting_mute_leases",
        ["agenda_item_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_meeting_mute_leases_agenda_item_id", table_name="meeting_mute_leases")
    op.drop_index("ix_meeting_mute_leases_session_id", table_name="meeting_mute_leases")
    op.drop_table("meeting_mute_leases")
