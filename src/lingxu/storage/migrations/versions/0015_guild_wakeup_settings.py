"""Add optional per-Guild wake-phrase overrides.

Revision ID: 0015_guild_wakeup
Revises: 0014_pronunciations
Create Date: 2026-09-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_guild_wakeup"
down_revision: str | None = "0014_pronunciations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if "guild_wakeup_settings" not in sa.inspect(bind).get_table_names():
        op.create_table(
            "guild_wakeup_settings",
            sa.Column("guild_id", sa.BigInteger(), nullable=False),
            sa.Column("wake_phrase", sa.Text(), nullable=False),
            sa.Column("updated_by_user_id", sa.BigInteger(), nullable=True),
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
            sa.PrimaryKeyConstraint("guild_id"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "guild_wakeup_settings" in sa.inspect(bind).get_table_names():
        op.drop_table("guild_wakeup_settings")
