"""Add restart-safe Markdown agenda imports and the create-channel identity.

Revision ID: 0010_agenda_import
Revises: 0009_vote_agenda_index
Create Date: 2026-07-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_agenda_import"
down_revision: str | None = "0009_vote_agenda_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    session_columns = {column["name"] for column in inspector.get_columns("meeting_sessions")}
    if "setup_channel_id" not in session_columns:
        op.add_column(
            "meeting_sessions",
            sa.Column("setup_channel_id", sa.BigInteger(), nullable=True),
        )
        op.execute(
            sa.text(
                "UPDATE meeting_sessions SET setup_channel_id = publication_channel_id "
                "WHERE setup_channel_id IS NULL"
            )
        )

    if "agenda_imports" not in sa.inspect(bind).get_table_names():
        op.create_table(
            "agenda_imports",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("session_id", sa.String(length=26), nullable=False),
            sa.Column("channel_id", sa.BigInteger(), nullable=False),
            sa.Column("message_id", sa.BigInteger(), nullable=False),
            sa.Column("attachment_id", sa.BigInteger(), nullable=False),
            sa.Column("actor_user_id", sa.BigInteger(), nullable=False),
            sa.Column("content_sha256", sa.String(length=64), nullable=False),
            sa.Column("source_text", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("policy_version", sa.String(length=32), nullable=False),
            sa.Column("model_name", sa.String(length=80), nullable=False),
            sa.Column("model_digest", sa.String(length=64), nullable=False),
            sa.Column("result_json", sa.JSON(), nullable=False, server_default="{}"),
            sa.Column("last_error_type", sa.String(length=80), nullable=True),
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
            sa.UniqueConstraint("session_id", "message_id", "attachment_id"),
        )
        op.create_index("ix_agenda_imports_session_id", "agenda_imports", ["session_id"])
        op.create_index(
            "uq_agenda_imports_one_processing",
            "agenda_imports",
            ["session_id"],
            unique=True,
            sqlite_where=sa.text("status = 'PROCESSING'"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "agenda_imports" in sa.inspect(bind).get_table_names():
        op.drop_index("uq_agenda_imports_one_processing", table_name="agenda_imports")
        op.drop_index("ix_agenda_imports_session_id", table_name="agenda_imports")
        op.drop_table("agenda_imports")
    if bind.dialect.name != "sqlite":
        session_columns = {
            column["name"] for column in sa.inspect(bind).get_columns("meeting_sessions")
        }
        if "setup_channel_id" in session_columns:
            op.drop_column("meeting_sessions", "setup_channel_id")
