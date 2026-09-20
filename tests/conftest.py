from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import text

from predict_stock.config import PROJECT_ROOT, load_config
from predict_stock.data.dnse_client import DailyBar, DnseInvalidSymbol
from predict_stock.db.session import make_engine

TABLES = ["ohlcv_daily", "universe_membership", "trading_days", "instruments", "job_runs"]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def _schema():
    """Bring the *test* database to Alembic head (exercises the real migrations)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "alembic", "-x", "db=test", "upgrade", "head"],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"test database unavailable: {exc.stderr[-300:]}")


@pytest.fixture
def engine(_schema):
    """Runtime-role (DML-only) engine on the test DB, tables emptied before each test."""
    admin = make_engine("migrator", test=True)
    with admin.begin() as c:
        c.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in TABLES:
            c.execute(text(f"TRUNCATE TABLE {t}"))
        c.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    admin.dispose()
    eng = make_engine("app", test=True)
    yield eng
    eng.dispose()


def make_bars(start: date, n: int, base: float = 50000.0, step: float = 100.0) -> list[DailyBar]:
    """n weekday bars from ``start`` (skips Sat/Sun) with a deterministic price path."""
    bars, d, i = [], start, 0
    while len(bars) < n:
        if d.weekday() < 5:
            px = Decimal(str(base + step * i))
            bars.append(DailyBar(d, px, px + 200, px - 200, px + 100, 1_000_000 + i))
            i += 1
        d += timedelta(days=1)
    return bars


class FakeClient:
    """Stands in for DnseClient: serves canned series, records calls."""

    def __init__(self, series: dict[str, list[DailyBar]]):
        self.series = series
        self.calls: list[tuple[str, date, date]] = []
        self.warnings: list[str] = []

    def fetch_daily(self, symbol, kind, start, end):
        self.calls.append((symbol, start, end))
        if symbol not in self.series:
            raise DnseInvalidSymbol(symbol)
        return [b for b in self.series[symbol] if start <= b.trade_date <= end]


@pytest.fixture
def after_close():
    """A 'now' safely after the session close on a Friday (2026-09-18, 16:00 ICT)."""
    return datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)
