"""Add one append-only deployment-wide service-capacity slot.

Revision ID: 0012_service_capacity
Revises: 0011_resident_assistant
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_service_capacity"
down_revision: str | None = "0011_resident_assistant"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "service_capacity_leases" not in tables:
        op.create_table(
            "service_capacity_leases",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("slot_name", sa.String(length=24), nullable=False),
            sa.Column("guild_id", sa.BigInteger(), nullable=False),
            sa.Column("mode", sa.String(length=32), nullable=False),
            sa.Column("owner_id", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("next_sequence", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("active_marker", sa.Integer(), nullable=True, server_default="1"),
            sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
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
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("slot_name", "active_marker"),
        )
        op.create_index(
            "ix_service_capacity_leases_guild_id",
            "service_capacity_leases",
            ["guild_id"],
        )
    if "service_capacity_events" not in tables:
        op.create_table(
            "service_capacity_events",
            sa.Column("id", sa.String(length=26), nullable=False),
            sa.Column("lease_id", sa.String(length=26), nullable=False),
            sa.Column("sequence_number", sa.Integer(), nullable=False),
            sa.Column("event_type", sa.String(length=40), nullable=False),
            sa.Column("reason", sa.String(length=80), nullable=False),
            sa.Column(
                "occurred_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.ForeignKeyConstraint(
                ["lease_id"], ["service_capacity_leases.id"], ondelete="RESTRICT"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("lease_id", "sequence_number"),
        )
        op.create_index(
            "ix_service_capacity_events_lease_id",
            "service_capacity_events",
            ["lease_id"],
        )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS service_capacity_events_reject_update
        BEFORE UPDATE ON service_capacity_events
        BEGIN
            SELECT RAISE(ABORT, 'service_capacity_events is append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS service_capacity_events_reject_delete
        BEFORE DELETE ON service_capacity_events
        BEGIN
            SELECT RAISE(ABORT, 'service_capacity_events is append-only');
        END
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS service_capacity_events_reject_delete")
    op.execute("DROP TRIGGER IF EXISTS service_capacity_events_reject_update")
    op.drop_index("ix_service_capacity_events_lease_id", table_name="service_capacity_events")
    op.drop_table("service_capacity_events")
    op.drop_index("ix_service_capacity_leases_guild_id", table_name="service_capacity_leases")
    op.drop_table("service_capacity_leases")
