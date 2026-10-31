"""Add Phase 7 component fields.

Revision ID: 0002_phase7
Revises: 0001_phase2
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_phase7"
down_revision: str | None = "0001_phase2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    participant_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("meeting_participants")
    }
    if "consent_minutes" not in participant_columns:
        op.add_column(
            "meeting_participants",
            sa.Column("consent_minutes", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    vote_columns = {column["name"] for column in sa.inspect(bind).get_columns("meeting_votes")}
    if "title" not in vote_columns:
        op.add_column(
            "meeting_votes",
            sa.Column("title", sa.Text(), nullable=False, server_default=""),
        )
    if "deadline_at" not in vote_columns:
        op.add_column(
            "meeting_votes",
            sa.Column(
                "deadline_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    vote_columns = {column["name"] for column in sa.inspect(bind).get_columns("meeting_votes")}
    if "deadline_at" in vote_columns:
        op.drop_column("meeting_votes", "deadline_at")
    if "title" in vote_columns:
        op.drop_column("meeting_votes", "title")
    participant_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("meeting_participants")
    }
    if "consent_minutes" in participant_columns:
        op.drop_column("meeting_participants", "consent_minutes")
