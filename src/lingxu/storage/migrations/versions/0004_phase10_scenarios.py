"""Add Scenario lifecycle fields.

Revision ID: 0004_phase10
Revises: 0003_phase8
Create Date: 2026-07-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_phase10"
down_revision: str | None = "0003_phase8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("scenario_versions")}
    if "status" not in columns:
        op.add_column(
            "scenario_versions",
            sa.Column("status", sa.String(length=24), nullable=False, server_default="DRAFT"),
        )
    if "published_at" not in columns:
        op.add_column(
            "scenario_versions",
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    columns = {column["name"] for column in sa.inspect(bind).get_columns("scenario_versions")}
    if "published_at" in columns:
        op.drop_column("scenario_versions", "published_at")
    if "status" in columns:
        op.drop_column("scenario_versions", "status")
