"""ORM models (MySQL 8 / InnoDB / utf8mb4). Every schema change goes through Alembic.

Conventions
* Prices are DECIMAL; trade dates are DATE; every timestamp is UTC (default UTC_TIMESTAMP()).
* Stock prices are stored in VND (DNSE quotes thousand VND; converted once in data/dnse_client.py).
* Binary artifacts (models, datasets) are never stored in the DB: only path + sha256.
* Intervals are half-open: [valid_from, valid_to), valid_to NULL = still valid.
  Point-in-time test:  valid_from <= d AND (valid_to IS NULL OR d < valid_to)

Groups: A instruments/universes · B prices/calendar/ingest · C feature/label/dataset ·
        D models/experiments/predictions · E recommendations/paper trading · F config/runs/alerts
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PRICE = Numeric(18, 4)
MONEY = Numeric(20, 2)
FACTOR = Numeric(24, 12)
WEIGHT = Numeric(12, 8)
UTC_NOW = text("(UTC_TIMESTAMP())")
TABLE_OPTS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}
BigPK = BigInteger


class Base(DeclarativeBase):
    __table_args__ = TABLE_OPTS


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime, server_default=UTC_NOW)


# =============================================================================
# F. config / runs / alerts  (defined first: many tables reference job_runs)
# =============================================================================
class ConfigSnapshot(Base):
    """Deduplicated config snapshots, keyed by sha256 of the canonical JSON."""

    __tablename__ = "config_snapshots"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    content: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = _created()


class JobRun(Base):
    """One row per job execution: reproducibility record (principle 6)."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # running | success | failed
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    git_commit: Mapped[str | None] = mapped_column(String(40))
    git_dirty: Mapped[bool | None] = mapped_column()
    seed: Mapped[int | None] = mapped_column(Integer)
    config_snapshot_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("config_snapshots.id"))
    params: Mapped[dict | None] = mapped_column(JSON)
    stats: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_open", "acknowledged_at", "severity"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    severity: Mapped[str] = mapped_column(String(16))  # info | warn | error | critical
    category: Mapped[str] = mapped_column(String(48))
    message: Mapped[str] = mapped_column(Text)
    instrument_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("instruments.id"))
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _created()
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime)


# =============================================================================
# A. instruments / universes
# =============================================================================
class Instrument(Base):
    """Stable internal identity. Symbols, exchanges and statuses change over time and
    live in the *_history tables; nothing else should key on a ticker."""

    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(8))  # stock | index
    name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class InstrumentSymbolHistory(Base):
    """Which ticker (and exchange) an instrument had over [valid_from, valid_to).
    A rename or an exchange move closes one row and opens the next."""

    __tablename__ = "instrument_symbol_history"
    __table_args__ = (
        UniqueConstraint("instrument_id", "valid_from", name="uq_symbol_hist_instrument_from"),
        Index("ix_symbol_hist_symbol", "symbol", "valid_from"),
        CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_symbol_hist_interval"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    symbol: Mapped[str] = mapped_column(String(20))
    exchange: Mapped[str | None] = mapped_column(String(8))  # HOSE | HNX | UPCOM | NULL = unknown
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = _created()


class InstrumentStatusHistory(Base):
    """Suspension / delisting periods. No row = normal trading."""

    __tablename__ = "instrument_status_history"
    __table_args__ = (
        UniqueConstraint("instrument_id", "valid_from", name="uq_status_hist_instrument_from"),
        CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_status_hist_interval"),
        CheckConstraint("status IN ('suspended', 'delisted')", name="ck_status_hist_status"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    status: Mapped[str] = mapped_column(String(12))
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    note: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class Universe(Base):
    __tablename__ = "universes"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class UniverseMembership(Base):
    __tablename__ = "universe_membership"
    __table_args__ = (
        UniqueConstraint("universe_id", "instrument_id", "valid_from", name="uq_membership"),
        Index("ix_membership_lookup", "universe_id", "valid_from", "valid_to"),
        CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="ck_membership_interval"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    universe_id: Mapped[int] = mapped_column(BigPK, ForeignKey("universes.id"))
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    valid_from: Mapped[date] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    weight: Mapped[float | None] = mapped_column(WEIGHT)
    source: Mapped[str] = mapped_column(String(255))
    note: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class UniverseChangeLog(Base):
    __tablename__ = "universe_change_log"
    __table_args__ = (Index("ix_change_log_universe", "universe_id", "effective_date"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    universe_id: Mapped[int] = mapped_column(BigPK, ForeignKey("universes.id"))
    instrument_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("instruments.id"))
    symbol: Mapped[str | None] = mapped_column(String(20))  # ticker at the time of the change
    action: Mapped[str] = mapped_column(String(24))
    effective_date: Mapped[date] = mapped_column(Date)
    old_value: Mapped[dict | None] = mapped_column(JSON)
    new_value: Mapped[dict | None] = mapped_column(JSON)
    source: Mapped[str | None] = mapped_column(String(255))
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    created_at: Mapped[datetime] = _created()


# =============================================================================
# B. prices / corporate actions / calendar / ingest
# =============================================================================
class PriceBar(Base):
    """Daily bar exactly as received. ``price_basis`` says what the numbers are:
    'raw' (unadjusted) or 'vendor_adjusted' (already adjusted by the data vendor).
    DNSE only serves vendor-adjusted prices, so that is what is stored today. A bar is
    only ever rewritten when the vendor re-adjusts, and the old values are then kept in
    price_bar_revisions."""

    __tablename__ = "price_bar"

    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(PRICE)
    high: Mapped[float] = mapped_column(PRICE)
    low: Mapped[float] = mapped_column(PRICE)
    close: Mapped[float] = mapped_column(PRICE)
    volume: Mapped[int] = mapped_column(BigInteger)
    price_basis: Mapped[str] = mapped_column(String(16), server_default="vendor_adjusted")
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))


class PriceBarRevision(Base):
    """Values a price_bar row had before the vendor changed them (audit trail)."""

    __tablename__ = "price_bar_revisions"
    __table_args__ = (Index("ix_revision_bar", "instrument_id", "trade_date"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    trade_date: Mapped[date] = mapped_column(Date)
    open: Mapped[float] = mapped_column(PRICE)
    high: Mapped[float] = mapped_column(PRICE)
    low: Mapped[float] = mapped_column(PRICE)
    close: Mapped[float] = mapped_column(PRICE)
    volume: Mapped[int] = mapped_column(BigInteger)
    price_basis: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    superseded_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    superseded_by_run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))


class CorporateAction(Base):
    """DNSE offers no corporate-action data, so rows come from (a) gap detection, stored as
    status='candidate' and never applied on their own, or (b) a confirmed file loaded by an
    operator (status='confirmed'), which is what adjustment_factors are built from."""

    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint("instrument_id", "ex_date", "action_type", name="uq_corporate_action"),
        Index("ix_corp_action_instrument", "instrument_id", "ex_date"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    action_type: Mapped[str] = mapped_column(String(24))  # cash_dividend | stock_dividend | bonus | split | rights ...
    ex_date: Mapped[date] = mapped_column(Date)
    record_date: Mapped[date | None] = mapped_column(Date)
    ratio: Mapped[float | None] = mapped_column(FACTOR)  # new shares per old share, where applicable
    cash_per_share: Mapped[float | None] = mapped_column(PRICE)
    details: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(12), server_default="confirmed")  # candidate | confirmed | rejected | stale
    source: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class AdjustmentFactor(Base):
    """Versioned back-adjustment factors. Per instrument, the highest ``version`` is the
    active set; a bar dated before ``effective_date`` (the ex-date) is multiplied by the
    factor. Bars with price_basis='vendor_adjusted' are never re-adjusted."""

    __tablename__ = "adjustment_factors"
    __table_args__ = (
        UniqueConstraint("instrument_id", "version", "effective_date", name="uq_adjustment_factor"),
        CheckConstraint("factor > 0", name="ck_adjustment_factor_positive"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    version: Mapped[int] = mapped_column(Integer)
    effective_date: Mapped[date] = mapped_column(Date)
    factor: Mapped[float] = mapped_column(FACTOR)
    corporate_action_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("corporate_actions.id"))
    source: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class TradingCalendar(Base):
    """Observed trading days per calendar (currently the stock-consensus calendar)."""

    __tablename__ = "trading_calendar"

    calendar_code: Mapped[str] = mapped_column(String(24), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))


class DataIngestRun(Base):
    """Per-instrument result of one ingest job (job_runs holds the job itself)."""

    __tablename__ = "data_ingest_runs"
    __table_args__ = (Index("ix_ingest_run_job", "run_id"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    source: Mapped[str] = mapped_column(String(32))
    resolution: Mapped[str] = mapped_column(String(4), server_default="1D")
    mode: Mapped[str] = mapped_column(String(24))  # full | incremental | full-after-drift | noop | failed
    status: Mapped[str] = mapped_column(String(10), server_default="ok")  # ok | blocked | failed
    error: Mapped[str | None] = mapped_column(Text)  # failure or the reason an instrument is blocked
    params: Mapped[dict | None] = mapped_column(JSON)
    requested_start: Mapped[date] = mapped_column(Date)
    requested_end: Mapped[date] = mapped_column(Date)
    fetched: Mapped[int] = mapped_column(Integer)
    inserted: Mapped[int] = mapped_column(Integer)
    updated: Mapped[int] = mapped_column(Integer)
    unchanged: Mapped[int] = mapped_column(Integer)
    drift: Mapped[bool] = mapped_column()
    first_date: Mapped[date | None] = mapped_column(Date)
    last_date: Mapped[date | None] = mapped_column(Date)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    warnings: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _created()


class PriceBarIntraday(Base):
    """Optional intraday bars (config ingest.intraday.enabled, default off). ``bar_time`` is the
    bar's start in UTC. Same price_basis semantics as price_bar; no revision history."""

    __tablename__ = "price_bar_intraday"

    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"), primary_key=True)
    resolution: Mapped[str] = mapped_column(String(4), primary_key=True)  # 1H
    bar_time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    open: Mapped[float] = mapped_column(PRICE)
    high: Mapped[float] = mapped_column(PRICE)
    low: Mapped[float] = mapped_column(PRICE)
    close: Mapped[float] = mapped_column(PRICE)
    volume: Mapped[int] = mapped_column(BigInteger)
    price_basis: Mapped[str] = mapped_column(String(16), server_default="vendor_adjusted")
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))


class DataQualityKnownIssue(Base):
    """A characterised, understood data defect. A finding matching one (same check, instrument
    if given, date within [date_from, date_to]) is reported as 'explained' instead of 'open'.
    ``explanation`` states what is known and what is not (root cause may be unverified)."""

    __tablename__ = "data_quality_known_issues"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    issue_key: Mapped[str] = mapped_column(String(64), unique=True)
    check_name: Mapped[str] = mapped_column(String(32))
    instrument_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("instruments.id"))  # NULL = any instrument
    date_from: Mapped[date | None] = mapped_column(Date)
    date_to: Mapped[date | None] = mapped_column(Date)  # inclusive
    treatment: Mapped[str] = mapped_column(String(48))  # e.g. keep_flagged, clip_envelope_downstream, exclude_bar
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = _created()


class DataQualityFinding(Base):
    """Current state of each detected problem, one row per (instrument, check, date). Re-running the
    checks updates rows in place; a problem that no longer occurs becomes 'resolved'."""

    __tablename__ = "data_quality_findings"
    __table_args__ = (
        UniqueConstraint("instrument_id", "check_name", "subject_key", name="uq_dq_finding"),
        Index("ix_dq_status", "status", "severity"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    check_name: Mapped[str] = mapped_column(String(32))
    severity: Mapped[str] = mapped_column(String(8))  # error | warn | info
    trade_date: Mapped[date | None] = mapped_column(Date)  # NULL for series-level findings
    subject_key: Mapped[str] = mapped_column(String(10))  # ISO date, or 'series'
    detail: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(10))  # open | explained | resolved
    known_issue_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("data_quality_known_issues.id"))
    first_seen_run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    last_seen_run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


# =============================================================================
# C. feature sets / label specs / datasets
# =============================================================================
class FeatureSet(Base):
    __tablename__ = "feature_sets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_feature_set"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict] = mapped_column(JSON)  # feature names, params, lookbacks
    code_ref: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class LabelSpec(Base):
    __tablename__ = "label_specs"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_label_spec"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    horizon_days: Mapped[int] = mapped_column(Integer)  # also the purge/embargo width
    spec: Mapped[dict] = mapped_column(JSON)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class Dataset(Base):
    """Manifest of a materialised training set. The data itself lives on disk (path + sha256)."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    universe_id: Mapped[int] = mapped_column(BigPK, ForeignKey("universes.id"))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    feature_set_id: Mapped[int] = mapped_column(BigPK, ForeignKey("feature_sets.id"))
    label_spec_id: Mapped[int] = mapped_column(BigPK, ForeignKey("label_specs.id"))
    path: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64))
    row_count: Mapped[int] = mapped_column(BigInteger)
    manifest: Mapped[dict | None] = mapped_column(JSON)
    config_snapshot_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("config_snapshots.id"))
    git_commit: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = _created()


# =============================================================================
# D. experiments / models / predictions / monitoring
# =============================================================================
class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    config_snapshot_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("config_snapshots.id"))
    params: Mapped[dict | None] = mapped_column(JSON)
    summary: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), server_default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class Model(Base):
    """A trained model. The binary artifact is on disk: only path + sha256 are stored."""

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    algo: Mapped[str] = mapped_column(String(32))
    feature_set_id: Mapped[int] = mapped_column(BigPK, ForeignKey("feature_sets.id"))
    label_spec_id: Mapped[int] = mapped_column(BigPK, ForeignKey("label_specs.id"))
    dataset_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("datasets.id"))
    experiment_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("experiments.id"))
    artifact_path: Mapped[str] = mapped_column(String(512))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict | None] = mapped_column(JSON)
    seed: Mapped[int | None] = mapped_column(Integer)
    git_commit: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), server_default="candidate")
    created_at: Mapped[datetime] = _created()


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    __table_args__ = (Index("ix_model_metrics", "model_id", "metric_name"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(BigPK, ForeignKey("models.id"))
    experiment_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("experiments.id"))
    split: Mapped[str] = mapped_column(String(32))  # e.g. fold_3, oos, walk_forward_all
    metric_name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Double)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _created()


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("model_id", "instrument_id", "as_of_date", "horizon_days", name="uq_prediction"),
        Index("ix_prediction_asof", "as_of_date", "model_id"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(BigPK, ForeignKey("models.id"))
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    universe_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("universes.id"))
    as_of_date: Mapped[date] = mapped_column(Date)
    horizon_days: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Double)
    proba: Mapped[float | None] = mapped_column(Double)
    rank_in_universe: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[dict | None] = mapped_column(JSON)  # q10/q50/q90, expected holding time (median, p75, n), raw proba, top SHAP contributions
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    created_at: Mapped[datetime] = _created()


class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"
    __table_args__ = (Index("ix_monitoring", "metric_name", "metric_date"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    model_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("models.id"))
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_date: Mapped[date] = mapped_column(Date)
    value: Mapped[float] = mapped_column(Double)
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = _created()


# =============================================================================
# E. recommendations / paper trading
# =============================================================================
class Recommendation(Base):
    """A recommendation always carries entry, target, stop, holding period, exit conditions
    and a rationale (all NOT NULL): a row without them cannot be written."""

    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint("strategy", "instrument_id", "as_of_date", "action", name="uq_recommendation"),
        CheckConstraint("entry_price > 0 AND target_price > 0 AND stop_loss > 0", name="ck_reco_prices"),
        CheckConstraint("hold_days_min > 0 AND hold_days_max >= hold_days_min", name="ck_reco_horizon"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(16))  # SWING | INVEST_B1 | INVEST_B2
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    universe_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("universes.id"))
    as_of_date: Mapped[date] = mapped_column(Date)
    action: Mapped[str] = mapped_column(String(8))  # BUY | EXIT (long-only)
    entry_price: Mapped[float] = mapped_column(PRICE)
    target_price: Mapped[float] = mapped_column(PRICE)
    stop_loss: Mapped[float] = mapped_column(PRICE)
    hold_days_min: Mapped[int] = mapped_column(Integer)
    hold_days_max: Mapped[int] = mapped_column(Integer)
    exit_conditions: Mapped[dict] = mapped_column(JSON)
    rationale: Mapped[str] = mapped_column(Text)
    rationale_data: Mapped[dict | None] = mapped_column(JSON)  # e.g. top features / SHAP
    confidence: Mapped[float | None] = mapped_column(Double)
    model_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("models.id"))
    prediction_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("predictions.id"))
    status: Mapped[str] = mapped_column(String(12), server_default="open")
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    created_at: Mapped[datetime] = _created()


class RecommendationOutcome(Base):
    __tablename__ = "recommendation_outcomes"

    recommendation_id: Mapped[int] = mapped_column(BigPK, ForeignKey("recommendations.id"), primary_key=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, server_default=UTC_NOW)
    exit_date: Mapped[date | None] = mapped_column(Date)
    exit_price: Mapped[float | None] = mapped_column(PRICE)
    exit_reason: Mapped[str | None] = mapped_column(String(24))  # target | stop | time | rebalance | open
    holding_days: Mapped[int | None] = mapped_column(Integer)
    gross_return: Mapped[float | None] = mapped_column(Double)
    net_return: Mapped[float | None] = mapped_column(Double)  # after fees, tax and slippage
    max_favorable: Mapped[float | None] = mapped_column(Double)
    max_adverse: Mapped[float | None] = mapped_column(Double)
    benchmark_return: Mapped[float | None] = mapped_column(Double)
    details: Mapped[dict | None] = mapped_column(JSON)


class PaperOrder(Base):
    __tablename__ = "paper_orders"
    __table_args__ = (Index("ix_paper_orders_date", "placed_date", "status"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    portfolio_code: Mapped[str] = mapped_column(String(32))
    recommendation_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("recommendations.id"))
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    side: Mapped[str] = mapped_column(String(4))  # BUY | SELL
    order_type: Mapped[str] = mapped_column(String(24))
    quantity: Mapped[int] = mapped_column(Integer)  # shares; lot size enforced in code from config
    limit_price: Mapped[float | None] = mapped_column(PRICE)
    status: Mapped[str] = mapped_column(String(16))  # pending | filled | rejected | cancelled | expired
    placed_date: Mapped[date] = mapped_column(Date)
    executed_date: Mapped[date | None] = mapped_column(Date)
    fill_price: Mapped[float | None] = mapped_column(PRICE)
    fee: Mapped[float | None] = mapped_column(MONEY)
    tax: Mapped[float | None] = mapped_column(MONEY)
    slippage_cost: Mapped[float | None] = mapped_column(MONEY)
    reject_reason: Mapped[str | None] = mapped_column(String(255))
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    created_at: Mapped[datetime] = _created()


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    __table_args__ = (
        UniqueConstraint("portfolio_code", "instrument_id", "opened_date", name="uq_paper_position"),
        TABLE_OPTS,
    )

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    portfolio_code: Mapped[str] = mapped_column(String(32))
    instrument_id: Mapped[int] = mapped_column(BigPK, ForeignKey("instruments.id"))
    recommendation_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("recommendations.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    avg_cost: Mapped[float] = mapped_column(PRICE)
    opened_date: Mapped[date] = mapped_column(Date)
    closed_date: Mapped[date | None] = mapped_column(Date)
    close_price: Mapped[float | None] = mapped_column(PRICE)
    status: Mapped[str] = mapped_column(String(8), server_default="open")
    created_at: Mapped[datetime] = _created()


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (UniqueConstraint("portfolio_code", "snapshot_date", name="uq_portfolio_snapshot"), TABLE_OPTS)

    id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    portfolio_code: Mapped[str] = mapped_column(String(32))
    snapshot_date: Mapped[date] = mapped_column(Date)
    cash: Mapped[float] = mapped_column(MONEY)
    market_value: Mapped[float] = mapped_column(MONEY)
    equity: Mapped[float] = mapped_column(MONEY)
    positions: Mapped[list | None] = mapped_column(JSON)
    metrics: Mapped[dict | None] = mapped_column(JSON)
    run_id: Mapped[int | None] = mapped_column(BigPK, ForeignKey("job_runs.id"))
    created_at: Mapped[datetime] = _created()
