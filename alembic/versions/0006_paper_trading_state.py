"""paper trading: sleeve target book, idempotency key of paper orders, flags on recommendations

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-21 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0006'
down_revision: Union[str, Sequence[str], None] = '0005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'sleeve_targets',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('sleeve', sa.String(length=16), nullable=False),
        sa.Column('as_of_date', sa.Date(), nullable=False),
        sa.Column('kind', sa.String(length=12), nullable=False),
        sa.Column('tranche_n', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('tranches_total', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('target', sa.JSON(), nullable=True),
        sa.Column('current', sa.JSON(), nullable=False),
        sa.Column('next_review', sa.Date(), nullable=True),
        sa.Column('run_id', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['job_runs.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sleeve', 'as_of_date', 'kind', 'tranche_n', name='uq_sleeve_target'),
        mysql_charset='utf8mb4', mysql_engine='InnoDB',
    )
    op.add_column('paper_orders', sa.Column('order_key', sa.String(length=160), nullable=True))
    op.create_unique_constraint('uq_paper_order_key', 'paper_orders', ['order_key'])
    op.add_column('recommendations', sa.Column('flags', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('recommendations', 'flags')
    op.drop_constraint('uq_paper_order_key', 'paper_orders', type_='unique')
    op.drop_column('paper_orders', 'order_key')
    op.drop_table('sleeve_targets')
