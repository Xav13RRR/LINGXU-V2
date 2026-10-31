"""Add agenda-linked private-vote persistence.

Revision ID: 0008_agenda_private_votes
Revises: 0007_meeting_identity
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_agenda_private_votes"
down_revision: str | None = "0007_meeting_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    vote_columns = {column["name"] for column in inspector.get_columns("meeting_votes")}
    if "agenda_item_id" not in vote_columns:
        op.add_column(
            "meeting_votes",
            sa.Column(
                "agenda_item_id",
                sa.String(length=26),
                nullable=True,
            ),
        )
    vote_indexes = {index["name"] for index in inspector.get_indexes("meeting_votes")}
    if "ix_meeting_votes_agenda_item_id" not in vote_indexes:
        op.create_index(
            "ix_meeting_votes_agenda_item_id",
            "meeting_votes",
            ["agenda_item_id"],
        )
    if "meeting_vote_recipients" not in inspector.get_table_names():
        op.create_table(
            "meeting_vote_recipients",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("vote_id", sa.String(length=26), nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column(
                "delivery_status",
                sa.String(length=24),
                nullable=False,
                server_default="PENDING",
            ),
            sa.Column("dm_message_id", sa.BigInteger(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["vote_id"], ["meeting_votes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("vote_id", "user_id"),
            sa.UniqueConstraint("dm_message_id"),
        )
        op.create_index(
            "ix_meeting_vote_recipients_vote_id",
            "meeting_vote_recipients",
            ["vote_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "meeting_vote_recipients" in inspector.get_table_names():
        op.drop_index(
            "ix_meeting_vote_recipients_vote_id",
            table_name="meeting_vote_recipients",
        )
        op.drop_table("meeting_vote_recipients")
    vote_columns = {column["name"] for column in inspector.get_columns("meeting_votes")}
    if "agenda_item_id" in vote_columns and bind.dialect.name != "sqlite":
        op.drop_index("ix_meeting_votes_agenda_item_id", table_name="meeting_votes")
        op.drop_column("meeting_votes", "agenda_item_id")
