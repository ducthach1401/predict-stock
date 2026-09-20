"""predictions.details: per-signal quantiles, holding-time estimate, raw/calibrated probability and top feature contributions

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-20 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('predictions', sa.Column('details', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('predictions', 'details')
