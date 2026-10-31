"""Add Guild pronunciation aliases and validated agenda-import identities.

Revision ID: 0014_pronunciations
Revises: 0013_meeting_mute_leases
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_pronunciations"
down_revision: str | None = "0013_meeting_mute_leases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "guild_member_pronunciations" not in inspector.get_table_names():
        op.create_table(
            "guild_member_pronunciations",
            sa.Column("guild_id", sa.BigInteger(), nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column("alias", sa.Text(), nullable=False),
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
            sa.PrimaryKeyConstraint("guild_id", "user_id"),
        )
    columns = {column["name"] for column in sa.inspect(bind).get_columns("agenda_imports")}
    if "validated_identity_user_ids_json" not in columns:
        op.add_column(
            "agenda_imports",
            sa.Column(
                "validated_identity_user_ids_json",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        columns = {column["name"] for column in sa.inspect(bind).get_columns("agenda_imports")}
        if "validated_identity_user_ids_json" in columns:
            op.drop_column("agenda_imports", "validated_identity_user_ids_json")
    if "guild_member_pronunciations" in sa.inspect(bind).get_table_names():
        op.drop_table("guild_member_pronunciations")
