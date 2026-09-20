"""recommendations: validity date, the full structured card and its Vietnamese text

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-20 23:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('recommendations', sa.Column('valid_until', sa.Date(), nullable=True))
    op.add_column('recommendations', sa.Column('card', sa.JSON(), nullable=True))
    op.add_column('recommendations', sa.Column('card_text', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('recommendations', 'card_text')
    op.drop_column('recommendations', 'card')
    op.drop_column('recommendations', 'valid_until')
