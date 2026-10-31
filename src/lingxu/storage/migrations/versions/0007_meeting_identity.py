"""Add meeting-time participant identity snapshots.

Revision ID: 0007_meeting_identity
Revises: 0006_guild_consents
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_meeting_identity"
down_revision: str | None = "0006_guild_consents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("meeting_participants")}
    if "display_name" not in columns:
        op.add_column(
            "meeting_participants",
            sa.Column("display_name", sa.Text(), nullable=True),
        )
    if "is_host" not in columns:
        op.add_column(
            "meeting_participants",
            sa.Column("is_host", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.execute(
            sa.text(
                "UPDATE meeting_participants SET is_host = 1 "
                "WHERE EXISTS (SELECT 1 FROM meeting_sessions "
                "WHERE meeting_sessions.id = meeting_participants.session_id "
                "AND meeting_sessions.creator_user_id = meeting_participants.user_id)"
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("meeting_participants")}
    if "is_host" in columns:
        op.drop_column("meeting_participants", "is_host")
    if "display_name" in columns:
        op.drop_column("meeting_participants", "display_name")
