"""Repair the agenda-vote index after an interrupted SQLite 0008 upgrade.

Revision ID: 0009_vote_agenda_index
Revises: 0008_agenda_private_votes
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_vote_agenda_index"
down_revision: str | None = "0008_agenda_private_votes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("meeting_votes")}
    if "agenda_item_id" not in columns:
        op.add_column(
            "meeting_votes",
            sa.Column("agenda_item_id", sa.String(length=26), nullable=True),
        )
    indexes = {index["name"] for index in inspector.get_indexes("meeting_votes")}
    if "ix_meeting_votes_agenda_item_id" not in indexes:
        op.create_index(
            "ix_meeting_votes_agenda_item_id",
            "meeting_votes",
            ["agenda_item_id"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("meeting_votes")}
    if "ix_meeting_votes_agenda_item_id" in indexes:
        op.drop_index("ix_meeting_votes_agenda_item_id", table_name="meeting_votes")
