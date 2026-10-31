"""Add durable Guild-scoped member consent.

Revision ID: 0006_guild_consents
Revises: 0005_phase11
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_guild_consents"
down_revision: str | None = "0005_phase11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if "guild_member_consents" in sa.inspect(bind).get_table_names():
        return
    op.create_table(
        "guild_member_consents",
        sa.Column("guild_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "consent_realtime_asr",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "consent_recording",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "consent_minutes",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "granted_at",
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


def downgrade() -> None:
    bind = op.get_bind()
    if "guild_member_consents" in sa.inspect(bind).get_table_names():
        op.drop_table("guild_member_consents")
