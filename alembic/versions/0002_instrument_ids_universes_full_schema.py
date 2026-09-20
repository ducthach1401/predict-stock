"""instrument ids, symbol/status history, universes, price bars + revisions, versioned adjustments,
feature/model/recommendation/paper-trading tables, config snapshots

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-20

Data is preserved in both directions, with these limits:
* upgrade: legacy symbols become instruments (one symbol_history row valid from 2000-01-01, exchange
  unknown); bars become price_bar rows with price_basis='vendor_adjusted'; each membership row gets an
  'add' (and, if closed, 'remove') universe_change_log entry; job_runs.config_snapshot moves to config_snapshots.
* downgrade: only what the old schema can hold is copied back (latest symbol per instrument, bars,
  memberships, calendar, config snapshots). Everything else (weights, notes, revisions, history of
  symbols/status, groups C-F) is discarded with its tables.
"""
import hashlib
import json
from datetime import date
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FLOOR = date(2000, 1, 1)
CALENDAR_CODE = "VN_CONSENSUS"
NEW_TABLES_REVERSED = ['recommendation_outcomes', 'paper_positions', 'paper_orders', 'recommendations', 'predictions', 'monitoring_metrics', 'model_metrics', 'models', 'universe_change_log', 'price_bar_revisions', 'price_bar', 'portfolio_snapshots', 'experiments', 'data_ingest_runs', 'alerts', 'adjustment_factors', 'universe_membership', 'instrument_symbol_history', 'instrument_status_history', 'datasets', 'corporate_actions', 'universes', 'trading_calendar', 'label_specs', 'instruments', 'feature_sets', 'config_snapshots']  # created tables in reverse dependency order

VIEW_ADJUSTED = """
CREATE VIEW v_price_adjusted AS
SELECT b.instrument_id, b.trade_date,
       CAST(b.open  * x.cf AS DECIMAL(18,4)) AS open,
       CAST(b.high  * x.cf AS DECIMAL(18,4)) AS high,
       CAST(b.low   * x.cf AS DECIMAL(18,4)) AS low,
       CAST(b.close * x.cf AS DECIMAL(18,4)) AS close,
       b.volume, b.price_basis, x.cf AS adj_factor
FROM price_bar b
JOIN LATERAL (
    SELECT CASE WHEN b.price_basis = 'raw' THEN COALESCE((
               SELECT CAST(EXP(SUM(LN(af.factor))) AS DECIMAL(24,12))
               FROM adjustment_factors af
               WHERE af.instrument_id = b.instrument_id
                 AND af.effective_date > b.trade_date
                 AND af.version = (SELECT MAX(v.version) FROM adjustment_factors v
                                   WHERE v.instrument_id = b.instrument_id)), 1)
           ELSE 1 END AS cf
) AS x ON TRUE
"""


def _canonical_sha(content) -> str:
    canon = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def upgrade() -> None:
    conn = op.get_bind()

    # 1. park the legacy tables (their foreign keys follow the rename)
    for t in ("ohlcv_daily", "universe_membership", "trading_days", "instruments"):
        op.rename_table(t, f"legacy_{t}")

    # 2. new tables (job_runs is altered in place below)
    op.create_table('config_snapshots',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('content', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('sha256'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('feature_sets',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('spec', sa.JSON(), nullable=False),
    sa.Column('code_ref', sa.String(length=255), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_feature_set'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('instruments',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('kind', sa.String(length=8), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('label_specs',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('horizon_days', sa.Integer(), nullable=False),
    sa.Column('spec', sa.JSON(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_label_spec'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('trading_calendar',
    sa.Column('calendar_code', sa.String(length=24), nullable=False),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.PrimaryKeyConstraint('calendar_code', 'trade_date'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('universes',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('code', sa.String(length=32), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('code'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('corporate_actions',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('action_type', sa.String(length=24), nullable=False),
    sa.Column('ex_date', sa.Date(), nullable=False),
    sa.Column('record_date', sa.Date(), nullable=True),
    sa.Column('ratio', sa.Numeric(precision=24, scale=12), nullable=True),
    sa.Column('cash_per_share', sa.Numeric(precision=18, scale=4), nullable=True),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.Column('source', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_corp_action_instrument', 'corporate_actions', ['instrument_id', 'ex_date'], unique=False)
    op.create_table('datasets',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('universe_id', sa.BigInteger(), nullable=False),
    sa.Column('start_date', sa.Date(), nullable=False),
    sa.Column('end_date', sa.Date(), nullable=False),
    sa.Column('feature_set_id', sa.BigInteger(), nullable=False),
    sa.Column('label_spec_id', sa.BigInteger(), nullable=False),
    sa.Column('path', sa.String(length=512), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('row_count', sa.BigInteger(), nullable=False),
    sa.Column('manifest', sa.JSON(), nullable=True),
    sa.Column('config_snapshot_id', sa.BigInteger(), nullable=True),
    sa.Column('git_commit', sa.String(length=40), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['config_snapshot_id'], ['config_snapshots.id'], ),
    sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id'], ),
    sa.ForeignKeyConstraint(['label_spec_id'], ['label_specs.id'], ),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_dataset'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('instrument_status_history',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('status', sa.String(length=12), nullable=False),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('note', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.CheckConstraint("status IN ('suspended', 'delisted')", name='ck_status_hist_status'),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name='ck_status_hist_interval'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'valid_from', name='uq_status_hist_instrument_from'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('instrument_symbol_history',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('exchange', sa.String(length=8), nullable=True),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name='ck_symbol_hist_interval'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'valid_from', name='uq_symbol_hist_instrument_from'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_symbol_hist_symbol', 'instrument_symbol_history', ['symbol', 'valid_from'], unique=False)
    op.create_table('universe_membership',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('universe_id', sa.BigInteger(), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('valid_from', sa.Date(), nullable=False),
    sa.Column('valid_to', sa.Date(), nullable=True),
    sa.Column('weight', sa.Numeric(precision=12, scale=8), nullable=True),
    sa.Column('source', sa.String(length=255), nullable=False),
    sa.Column('note', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name='ck_membership_interval'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('universe_id', 'instrument_id', 'valid_from', name='uq_membership'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_membership_lookup', 'universe_membership', ['universe_id', 'valid_from', 'valid_to'], unique=False)
    op.create_table('adjustment_factors',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('effective_date', sa.Date(), nullable=False),
    sa.Column('factor', sa.Numeric(precision=24, scale=12), nullable=False),
    sa.Column('corporate_action_id', sa.BigInteger(), nullable=True),
    sa.Column('source', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.CheckConstraint('factor > 0', name='ck_adjustment_factor_positive'),
    sa.ForeignKeyConstraint(['corporate_action_id'], ['corporate_actions.id'], ),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('instrument_id', 'version', 'effective_date', name='uq_adjustment_factor'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('alerts',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('severity', sa.String(length=16), nullable=False),
    sa.Column('category', sa.String(length=48), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_alerts_open', 'alerts', ['acknowledged_at', 'severity'], unique=False)
    op.create_table('data_ingest_runs',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('run_id', sa.BigInteger(), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('mode', sa.String(length=24), nullable=False),
    sa.Column('requested_start', sa.Date(), nullable=False),
    sa.Column('requested_end', sa.Date(), nullable=False),
    sa.Column('fetched', sa.Integer(), nullable=False),
    sa.Column('inserted', sa.Integer(), nullable=False),
    sa.Column('updated', sa.Integer(), nullable=False),
    sa.Column('unchanged', sa.Integer(), nullable=False),
    sa.Column('drift', sa.Boolean(), nullable=False),
    sa.Column('first_date', sa.Date(), nullable=True),
    sa.Column('last_date', sa.Date(), nullable=True),
    sa.Column('warnings', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_ingest_run_job', 'data_ingest_runs', ['run_id'], unique=False)
    op.create_table('experiments',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('config_snapshot_id', sa.BigInteger(), nullable=True),
    sa.Column('params', sa.JSON(), nullable=True),
    sa.Column('summary', sa.JSON(), nullable=True),
    sa.Column('status', sa.String(length=16), server_default='running', nullable=False),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.Column('finished_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['config_snapshot_id'], ['config_snapshots.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('portfolio_snapshots',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('portfolio_code', sa.String(length=32), nullable=False),
    sa.Column('snapshot_date', sa.Date(), nullable=False),
    sa.Column('cash', sa.Numeric(precision=20, scale=2), nullable=False),
    sa.Column('market_value', sa.Numeric(precision=20, scale=2), nullable=False),
    sa.Column('equity', sa.Numeric(precision=20, scale=2), nullable=False),
    sa.Column('positions', sa.JSON(), nullable=True),
    sa.Column('metrics', sa.JSON(), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('portfolio_code', 'snapshot_date', name='uq_portfolio_snapshot'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('price_bar',
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('open', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('high', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('low', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('close', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('volume', sa.BigInteger(), nullable=False),
    sa.Column('price_basis', sa.String(length=16), server_default='vendor_adjusted', nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('fetched_at', sa.DateTime(), nullable=False),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('instrument_id', 'trade_date'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('price_bar_revisions',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('trade_date', sa.Date(), nullable=False),
    sa.Column('open', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('high', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('low', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('close', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('volume', sa.BigInteger(), nullable=False),
    sa.Column('price_basis', sa.String(length=16), nullable=False),
    sa.Column('source', sa.String(length=32), nullable=False),
    sa.Column('fetched_at', sa.DateTime(), nullable=False),
    sa.Column('superseded_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.Column('superseded_by_run_id', sa.BigInteger(), nullable=True),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['superseded_by_run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_revision_bar', 'price_bar_revisions', ['instrument_id', 'trade_date'], unique=False)
    op.create_table('universe_change_log',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('universe_id', sa.BigInteger(), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=True),
    sa.Column('symbol', sa.String(length=20), nullable=True),
    sa.Column('action', sa.String(length=24), nullable=False),
    sa.Column('effective_date', sa.Date(), nullable=False),
    sa.Column('old_value', sa.JSON(), nullable=True),
    sa.Column('new_value', sa.JSON(), nullable=True),
    sa.Column('source', sa.String(length=255), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_change_log_universe', 'universe_change_log', ['universe_id', 'effective_date'], unique=False)
    op.create_table('models',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('algo', sa.String(length=32), nullable=False),
    sa.Column('feature_set_id', sa.BigInteger(), nullable=False),
    sa.Column('label_spec_id', sa.BigInteger(), nullable=False),
    sa.Column('dataset_id', sa.BigInteger(), nullable=True),
    sa.Column('experiment_id', sa.BigInteger(), nullable=True),
    sa.Column('artifact_path', sa.String(length=512), nullable=False),
    sa.Column('artifact_sha256', sa.String(length=64), nullable=False),
    sa.Column('params', sa.JSON(), nullable=True),
    sa.Column('seed', sa.Integer(), nullable=True),
    sa.Column('git_commit', sa.String(length=40), nullable=True),
    sa.Column('status', sa.String(length=16), server_default='candidate', nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id'], ),
    sa.ForeignKeyConstraint(['label_spec_id'], ['label_specs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name', 'version', name='uq_model'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('model_metrics',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('model_id', sa.BigInteger(), nullable=False),
    sa.Column('experiment_id', sa.BigInteger(), nullable=True),
    sa.Column('split', sa.String(length=32), nullable=False),
    sa.Column('metric_name', sa.String(length=64), nullable=False),
    sa.Column('value', sa.Double(), nullable=False),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_model_metrics', 'model_metrics', ['model_id', 'metric_name'], unique=False)
    op.create_table('monitoring_metrics',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('model_id', sa.BigInteger(), nullable=True),
    sa.Column('metric_name', sa.String(length=64), nullable=False),
    sa.Column('metric_date', sa.Date(), nullable=False),
    sa.Column('value', sa.Double(), nullable=False),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_monitoring', 'monitoring_metrics', ['metric_name', 'metric_date'], unique=False)
    op.create_table('predictions',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('model_id', sa.BigInteger(), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('universe_id', sa.BigInteger(), nullable=True),
    sa.Column('as_of_date', sa.Date(), nullable=False),
    sa.Column('horizon_days', sa.Integer(), nullable=False),
    sa.Column('score', sa.Double(), nullable=False),
    sa.Column('proba', sa.Double(), nullable=True),
    sa.Column('rank_in_universe', sa.Integer(), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('model_id', 'instrument_id', 'as_of_date', 'horizon_days', name='uq_prediction'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_prediction_asof', 'predictions', ['as_of_date', 'model_id'], unique=False)
    op.create_table('recommendations',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('strategy', sa.String(length=16), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('universe_id', sa.BigInteger(), nullable=True),
    sa.Column('as_of_date', sa.Date(), nullable=False),
    sa.Column('action', sa.String(length=8), nullable=False),
    sa.Column('entry_price', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('target_price', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('stop_loss', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('hold_days_min', sa.Integer(), nullable=False),
    sa.Column('hold_days_max', sa.Integer(), nullable=False),
    sa.Column('exit_conditions', sa.JSON(), nullable=False),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('rationale_data', sa.JSON(), nullable=True),
    sa.Column('confidence', sa.Double(), nullable=True),
    sa.Column('model_id', sa.BigInteger(), nullable=True),
    sa.Column('prediction_id', sa.BigInteger(), nullable=True),
    sa.Column('status', sa.String(length=12), server_default='open', nullable=False),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.CheckConstraint('entry_price > 0 AND target_price > 0 AND stop_loss > 0', name='ck_reco_prices'),
    sa.CheckConstraint('hold_days_min > 0 AND hold_days_max >= hold_days_min', name='ck_reco_horizon'),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], ),
    sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.ForeignKeyConstraint(['universe_id'], ['universes.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('strategy', 'instrument_id', 'as_of_date', 'action', name='uq_recommendation'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('paper_orders',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('portfolio_code', sa.String(length=32), nullable=False),
    sa.Column('recommendation_id', sa.BigInteger(), nullable=True),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('side', sa.String(length=4), nullable=False),
    sa.Column('order_type', sa.String(length=24), nullable=False),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('limit_price', sa.Numeric(precision=18, scale=4), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('placed_date', sa.Date(), nullable=False),
    sa.Column('executed_date', sa.Date(), nullable=True),
    sa.Column('fill_price', sa.Numeric(precision=18, scale=4), nullable=True),
    sa.Column('fee', sa.Numeric(precision=20, scale=2), nullable=True),
    sa.Column('tax', sa.Numeric(precision=20, scale=2), nullable=True),
    sa.Column('slippage_cost', sa.Numeric(precision=20, scale=2), nullable=True),
    sa.Column('reject_reason', sa.String(length=255), nullable=True),
    sa.Column('run_id', sa.BigInteger(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], ),
    sa.ForeignKeyConstraint(['run_id'], ['job_runs.id'], ),
    sa.PrimaryKeyConstraint('id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_index('ix_paper_orders_date', 'paper_orders', ['placed_date', 'status'], unique=False)
    op.create_table('paper_positions',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('portfolio_code', sa.String(length=32), nullable=False),
    sa.Column('instrument_id', sa.BigInteger(), nullable=False),
    sa.Column('recommendation_id', sa.BigInteger(), nullable=True),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('avg_cost', sa.Numeric(precision=18, scale=4), nullable=False),
    sa.Column('opened_date', sa.Date(), nullable=False),
    sa.Column('closed_date', sa.Date(), nullable=True),
    sa.Column('close_price', sa.Numeric(precision=18, scale=4), nullable=True),
    sa.Column('status', sa.String(length=8), server_default='open', nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id'], ),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('portfolio_code', 'instrument_id', 'opened_date', name='uq_paper_position'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('recommendation_outcomes',
    sa.Column('recommendation_id', sa.BigInteger(), nullable=False),
    sa.Column('evaluated_at', sa.DateTime(), server_default=sa.text('(UTC_TIMESTAMP())'), nullable=False),
    sa.Column('exit_date', sa.Date(), nullable=True),
    sa.Column('exit_price', sa.Numeric(precision=18, scale=4), nullable=True),
    sa.Column('exit_reason', sa.String(length=24), nullable=True),
    sa.Column('holding_days', sa.Integer(), nullable=True),
    sa.Column('gross_return', sa.Double(), nullable=True),
    sa.Column('net_return', sa.Double(), nullable=True),
    sa.Column('max_favorable', sa.Double(), nullable=True),
    sa.Column('max_adverse', sa.Double(), nullable=True),
    sa.Column('benchmark_return', sa.Double(), nullable=True),
    sa.Column('details', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['recommendation_id'], ['recommendations.id'], ),
    sa.PrimaryKeyConstraint('recommendation_id'),
    mysql_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    # ### end Alembic commands ###

    # 3. job_runs: JSON config snapshot -> deduplicated config_snapshots
    op.add_column("job_runs", sa.Column("config_snapshot_id", sa.BigInteger(), nullable=True))
    op.create_foreign_key(None, "job_runs", "config_snapshots", ["config_snapshot_id"], ["id"])
    for run_id, snap in conn.execute(sa.text("SELECT id, config_snapshot FROM job_runs")).all():
        content = json.loads(snap) if isinstance(snap, (str, bytes)) else snap
        sha = _canonical_sha(content)
        conn.execute(
            sa.text("INSERT INTO config_snapshots (sha256, content) VALUES (:s, :c) ON DUPLICATE KEY UPDATE id = id"),
            {"s": sha, "c": json.dumps(content, ensure_ascii=False)},
        )
        sid = conn.execute(sa.text("SELECT id FROM config_snapshots WHERE sha256 = :s"), {"s": sha}).scalar_one()
        conn.execute(sa.text("UPDATE job_runs SET config_snapshot_id = :i WHERE id = :r"), {"i": sid, "r": run_id})
    op.drop_column("job_runs", "config_snapshot")
    op.alter_column("job_runs", "started_at", existing_type=sa.DateTime(), existing_nullable=False,
                    server_default=sa.text("(UTC_TIMESTAMP())"))

    # 4. instruments: symbol -> stable id + symbol history (exchange unknown, valid since FLOOR)
    for symbol, kind, created_at in conn.execute(
        sa.text("SELECT symbol, kind, created_at FROM legacy_instruments ORDER BY symbol")
    ).all():
        iid = conn.execute(
            sa.text("INSERT INTO instruments (kind, created_at) VALUES (:k, :c)"), {"k": kind, "c": created_at}
        ).lastrowid
        conn.execute(
            sa.text("INSERT INTO instrument_symbol_history (instrument_id, symbol, exchange, valid_from, valid_to) "
                    "VALUES (:i, :s, NULL, :f, NULL)"),
            {"i": iid, "s": symbol, "f": FLOOR},
        )

    # 5. bars, calendar, universes, memberships (+ change log)
    conn.execute(sa.text(
        "INSERT INTO price_bar (instrument_id, trade_date, open, high, low, close, volume, price_basis, source, fetched_at, run_id) "
        "SELECT h.instrument_id, o.trade_date, o.open, o.high, o.low, o.close, o.volume, 'vendor_adjusted', o.source, o.fetched_at, o.run_id "
        "FROM legacy_ohlcv_daily o JOIN instrument_symbol_history h ON h.symbol = o.symbol AND h.valid_to IS NULL"))
    conn.execute(sa.text(
        "INSERT INTO trading_calendar (calendar_code, trade_date, source) "
        "SELECT :c, trade_date, source FROM legacy_trading_days"), {"c": CALENDAR_CODE})
    conn.execute(sa.text(
        "INSERT INTO universes (code, name, description) "
        "SELECT DISTINCT universe_code, universe_code, 'migrated from legacy universe_membership (0002)' "
        "FROM legacy_universe_membership"))
    conn.execute(sa.text(
        "INSERT INTO universe_membership (universe_id, instrument_id, valid_from, valid_to, weight, source, note, created_at) "
        "SELECT u.id, h.instrument_id, m.effective_from, m.effective_to, NULL, m.source, NULL, m.loaded_at "
        "FROM legacy_universe_membership m JOIN universes u ON u.code = m.universe_code "
        "JOIN instrument_symbol_history h ON h.symbol = m.symbol AND h.valid_to IS NULL"))
    conn.execute(sa.text(
        "INSERT INTO universe_change_log (universe_id, instrument_id, symbol, action, effective_date, new_value, source) "
        "SELECT m.universe_id, m.instrument_id, h.symbol, 'add', m.valid_from, "
        "       JSON_OBJECT('valid_from', CAST(m.valid_from AS CHAR)), 'migration 0002' "
        "FROM universe_membership m JOIN instrument_symbol_history h ON h.instrument_id = m.instrument_id AND h.valid_to IS NULL"))
    conn.execute(sa.text(
        "INSERT INTO universe_change_log (universe_id, instrument_id, symbol, action, effective_date, new_value, source) "
        "SELECT m.universe_id, m.instrument_id, h.symbol, 'remove', m.valid_to, "
        "       JSON_OBJECT('valid_to', CAST(m.valid_to AS CHAR)), 'migration 0002' "
        "FROM universe_membership m JOIN instrument_symbol_history h ON h.instrument_id = m.instrument_id AND h.valid_to IS NULL "
        "WHERE m.valid_to IS NOT NULL"))

    # 6. adjusted-price view, then drop the legacy tables
    conn.execute(sa.text(VIEW_ADJUSTED))
    for t in ("ohlcv_daily", "universe_membership", "trading_days", "instruments"):
        op.drop_table(f"legacy_{t}")


def _fk_names(conn, table: str, column: str) -> list[str]:
    return [r[0] for r in conn.execute(sa.text(
        "SELECT CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t AND COLUMN_NAME = :c AND REFERENCED_TABLE_NAME IS NOT NULL"),
        {"t": table, "c": column}).all()]


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP VIEW IF EXISTS v_price_adjusted"))

    # 1. park the new instruments table, recreate the 0001 tables
    op.rename_table("instruments", "tmp_instruments")
    op.rename_table("universe_membership", "tmp_universe_membership")
    op.create_table("instruments",
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("symbol"), mysql_charset="utf8mb4", mysql_engine="InnoDB")
    op.create_table("trading_days",
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("trade_date"), mysql_charset="utf8mb4", mysql_engine="InnoDB")
    op.create_table("ohlcv_daily",
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("high", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("low", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("close", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("fetched_at", sa.DateTime(), nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["job_runs.id"]),
        sa.ForeignKeyConstraint(["symbol"], ["instruments.symbol"]),
        sa.PrimaryKeyConstraint("symbol", "trade_date"), mysql_charset="utf8mb4", mysql_engine="InnoDB")
    op.create_table("universe_membership",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("universe_code", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("loaded_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["symbol"], ["instruments.symbol"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("universe_code", "symbol", "effective_from", name="uq_membership"),
        mysql_charset="utf8mb4", mysql_engine="InnoDB")
    op.create_index("ix_membership_lookup", "universe_membership", ["universe_code", "effective_from", "effective_to"])

    # 2. copy back what the old schema can hold (latest symbol of each instrument)
    latest = ("JOIN instrument_symbol_history h ON h.instrument_id = {i} AND h.valid_from = "
              "(SELECT MAX(h2.valid_from) FROM instrument_symbol_history h2 WHERE h2.instrument_id = h.instrument_id)")
    conn.execute(sa.text("INSERT INTO instruments (symbol, kind, created_at) SELECT h.symbol, i.kind, i.created_at "
                         "FROM tmp_instruments i " + latest.format(i="i.id")))
    conn.execute(sa.text(
        "INSERT INTO ohlcv_daily (symbol, trade_date, open, high, low, close, volume, source, fetched_at, run_id) "
        "SELECT h.symbol, b.trade_date, b.open, b.high, b.low, b.close, b.volume, b.source, b.fetched_at, b.run_id "
        "FROM price_bar b " + latest.format(i="b.instrument_id")))
    conn.execute(sa.text(
        "INSERT INTO universe_membership (universe_code, symbol, effective_from, effective_to, source, loaded_at) "
        "SELECT u.code, h.symbol, m.valid_from, m.valid_to, m.source, m.created_at "
        "FROM tmp_universe_membership m JOIN universes u ON u.id = m.universe_id " + latest.format(i="m.instrument_id")))
    conn.execute(sa.text("INSERT INTO trading_days (trade_date, source) SELECT trade_date, source "
                         "FROM trading_calendar WHERE calendar_code = :c"), {"c": CALENDAR_CODE})

    # 3. job_runs: config_snapshots -> JSON column
    op.add_column("job_runs", sa.Column("config_snapshot", sa.JSON(), nullable=True))
    conn.execute(sa.text("UPDATE job_runs r JOIN config_snapshots c ON c.id = r.config_snapshot_id "
                         "SET r.config_snapshot = c.content"))
    conn.execute(sa.text("UPDATE job_runs SET config_snapshot = JSON_OBJECT() WHERE config_snapshot IS NULL"))
    op.alter_column("job_runs", "config_snapshot", existing_type=sa.JSON(), nullable=False)
    for fk in _fk_names(conn, "job_runs", "config_snapshot_id"):
        op.drop_constraint(fk, "job_runs", type_="foreignkey")
    op.drop_column("job_runs", "config_snapshot_id")
    op.alter_column("job_runs", "started_at", existing_type=sa.DateTime(), existing_nullable=False,
                    server_default=sa.text("now()"))

    # 4. drop the new tables (reverse dependency order)
    op.drop_table("tmp_universe_membership")  # references universes and tmp_instruments: drop it first
    for t in NEW_TABLES_REVERSED:
        if t not in ("instruments", "universe_membership"):
            op.drop_table(t)
    op.drop_table("tmp_instruments")
