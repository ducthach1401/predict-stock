"""ORM models (MySQL 8 / InnoDB / utf8mb4). Schema changes go through Alembic.

Units: stock prices are stored in VND (DNSE quotes thousand VND; the conversion
happens once, in data/dnse_client.py). Index rows are index points.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

PRICE = Numeric(18, 4)


class Base(DeclarativeBase):
    __table_args__ = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


class JobRun(Base):
    """One row per job execution: reproducibility record (principle 6)."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # running | success | failed
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    git_commit: Mapped[str | None] = mapped_column(String(40))
    git_dirty: Mapped[bool | None] = mapped_column(Boolean)
    seed: Mapped[int | None] = mapped_column(Integer)
    config_snapshot: Mapped[dict] = mapped_column(JSON)
    params: Mapped[dict | None] = mapped_column(JSON)
    stats: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


class Instrument(Base):
    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    kind: Mapped[str] = mapped_column(String(8))  # stock | index
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class UniverseMembership(Base):
    """Point-in-time membership. A symbol is a member on day d iff
    effective_from <= d < effective_to (effective_to NULL = still a member)."""

    __tablename__ = "universe_membership"
    __table_args__ = (
        UniqueConstraint("universe_code", "symbol", "effective_from", name="uq_membership"),
        Index("ix_membership_lookup", "universe_code", "effective_from", "effective_to"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    universe_code: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("instruments.symbol"))
    effective_from: Mapped[date] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(255))
    loaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class OhlcvDaily(Base):
    __tablename__ = "ohlcv_daily"

    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("instruments.symbol"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(PRICE)
    high: Mapped[float] = mapped_column(PRICE)
    low: Mapped[float] = mapped_column(PRICE)
    close: Mapped[float] = mapped_column(PRICE)
    volume: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    run_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("job_runs.id"))


class TradingDay(Base):
    """Observed trading days, derived from the calendar symbol's bars."""

    __tablename__ = "trading_days"

    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
