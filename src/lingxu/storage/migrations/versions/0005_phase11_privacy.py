"""Add unlinkable anonymous vote token hashes.

Revision ID: 0005_phase11
Revises: 0004_phase10
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_phase11"
down_revision: str | None = "0004_phase10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("meeting_vote_entries")}
    if "voter_token_hash" not in columns:
        op.add_column(
            "meeting_vote_entries",
            sa.Column("voter_token_hash", sa.String(length=64), nullable=True),
        )
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("meeting_vote_entries")}
    unique_sets = {
        tuple(constraint["column_names"])
        for constraint in sa.inspect(bind).get_unique_constraints("meeting_vote_entries")
    }
    if (
        "uq_vote_anonymous_token" not in indexes
        and ("vote_id", "voter_token_hash") not in unique_sets
    ):
        op.create_index(
            "uq_vote_anonymous_token",
            "meeting_vote_entries",
            ["vote_id", "voter_token_hash"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("meeting_vote_entries")}
    if "uq_vote_anonymous_token" in indexes:
        op.drop_index("uq_vote_anonymous_token", table_name="meeting_vote_entries")
    if bind.dialect.name != "sqlite":
        columns = {
            column["name"] for column in sa.inspect(bind).get_columns("meeting_vote_entries")
        }
        if "voter_token_hash" in columns:
            op.drop_column("meeting_vote_entries", "voter_token_hash")
