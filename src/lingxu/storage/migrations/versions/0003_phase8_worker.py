"""Add idempotent Worker source and publication fields.

Revision ID: 0003_phase8
Revises: 0002_phase7
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_phase8"
down_revision: str | None = "0002_phase7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    session_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("meeting_sessions")
    }
    if "publication_channel_id" not in session_columns:
        op.add_column(
            "meeting_sessions",
            sa.Column("publication_channel_id", sa.BigInteger(), nullable=True),
        )
    transcript_columns = {
        column["name"] for column in sa.inspect(bind).get_columns("transcript_entries")
    }
    if "source_key" not in transcript_columns:
        op.add_column(
            "transcript_entries",
            sa.Column("source_key", sa.String(length=96), nullable=False, server_default="legacy"),
        )
        op.execute(
            sa.text(
                "UPDATE transcript_entries "
                "SET source_key = 'legacy:' || id "
                "WHERE source_key = 'legacy'"
            )
        )
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("transcript_entries")}
    unique_sets = {
        tuple(constraint["column_names"])
        for constraint in sa.inspect(bind).get_unique_constraints("transcript_entries")
    }
    if (
        "uq_transcript_session_source" not in indexes
        and ("session_id", "source_key") not in unique_sets
    ):
        op.create_index(
            "uq_transcript_session_source",
            "transcript_entries",
            ["session_id", "source_key"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    indexes = {index["name"] for index in sa.inspect(bind).get_indexes("transcript_entries")}
    if "uq_transcript_session_source" in indexes:
        op.drop_index("uq_transcript_session_source", table_name="transcript_entries")
    # SQLite cannot drop these columns while the current package metadata's
    # unnamed table constraints reference them. They remain inert at revision
    # 0002 and are removed when revision 0001 drops the owned tables.
    if bind.dialect.name != "sqlite":
        transcript_columns = {
            column["name"] for column in sa.inspect(bind).get_columns("transcript_entries")
        }
        if "source_key" in transcript_columns:
            op.drop_column("transcript_entries", "source_key")
        session_columns = {
            column["name"] for column in sa.inspect(bind).get_columns("meeting_sessions")
        }
        if "publication_channel_id" in session_columns:
            op.drop_column("meeting_sessions", "publication_channel_id")
