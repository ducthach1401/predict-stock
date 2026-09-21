"""model lifecycle: strategy and training cut-off on models, status history, provenance of recommendations

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CHAMPIONS = {"swing": "swing_lgbm_final", "invest_b1": "invest_b1_factor_final", "invest_b2": "invest_b2_factor_final"}


def upgrade() -> None:
    op.add_column('models', sa.Column('strategy', sa.String(length=16), nullable=True))
    op.add_column('models', sa.Column('trained_until', sa.Date(), nullable=True))
    op.create_index('ix_models_strategy', 'models', ['strategy', 'status'])
    op.create_table(
        'model_status_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.BigInteger(), nullable=False),
        sa.Column('from_status', sa.String(length=16), nullable=True),
        sa.Column('to_status', sa.String(length=16), nullable=False),
        sa.Column('changed_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
        sa.Column('actor', sa.String(length=64), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['models.id']),
        sa.PrimaryKeyConstraint('id'),
        mysql_charset='utf8mb4', mysql_engine='InnoDB',
    )
    op.create_index('ix_model_status_log_model', 'model_status_log', ['model_id', 'changed_at'])
    op.add_column('recommendations', sa.Column('provenance', sa.JSON(), nullable=True))

    # existing rows -------------------------------------------------------------------------------------------------------------------------------
    op.execute("UPDATE models SET status='retired' WHERE status='superseded'")
    op.execute("UPDATE models SET strategy='swing' WHERE name LIKE 'swing\\_%'")
    op.execute("UPDATE models SET strategy='invest_b1' WHERE name LIKE 'invest\\_b1\\_%'")
    op.execute("UPDATE models SET strategy='invest_b2' WHERE name LIKE 'invest\\_b2\\_%'")
    op.execute("UPDATE models SET trained_until=STR_TO_DATE(JSON_UNQUOTE(JSON_EXTRACT(params, '$.fold.train[1]')), '%Y-%m-%d') WHERE JSON_EXTRACT(params, '$.fold.train[1]') IS NOT NULL")
    for strategy, name in CHAMPIONS.items():          # the models in service today (behind every card and paper order so far) are the champions
        op.execute(sa.text(
            "UPDATE models m JOIN (SELECT name, MAX(version) v FROM models WHERE name = :n GROUP BY name) x ON x.name = m.name AND x.v = m.version "
            "SET m.status = 'champion' WHERE m.name = :n").bindparams(n=name))
    op.execute("INSERT INTO model_status_log (model_id, from_status, to_status, actor, reason) SELECT id, NULL, status, 'migration 0007', "
               "'status recorded when the lifecycle registry was introduced' FROM models")
    op.create_check_constraint('ck_model_status', 'models', "status IN ('candidate','shadow','champion','retired')")


def downgrade() -> None:
    op.drop_constraint('ck_model_status', 'models', type_='check')
    op.drop_column('recommendations', 'provenance')
    op.drop_table('model_status_log')
    op.drop_index('ix_models_strategy', table_name='models')
    op.drop_column('models', 'trained_until')
    op.drop_column('models', 'strategy')
    op.execute("UPDATE models SET status='superseded' WHERE status='retired'")
    op.execute("UPDATE models SET status='candidate' WHERE status IN ('shadow','champion')")
