from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import text

from predict_stock.config import PROJECT_ROOT, load_config
from predict_stock.data.dnse_client import DailyBar, DnseInvalidSymbol, IntradayBar
from predict_stock.db.session import make_engine, session_scope
from predict_stock.universe_sync import SnapshotRow, apply_snapshot


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
        tables = [r[0] for r in c.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() "
            "AND table_type = 'BASE TABLE' AND table_name <> 'alembic_version'"))]
        c.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        for t in tables:
            c.execute(text(f"DELETE FROM `{t}`"))
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


def make_intraday(start: date, n_days: int, base: float = 50000.0) -> list[IntradayBar]:
    """Five 1H bars per weekday (09,10,11,13,14 ICT = 02,03,04,06,07 UTC), n_days weekdays from ``start``."""
    bars, d, k = [], start, 0
    while d.weekday() >= 5:
        d += timedelta(days=1)
    days = 0
    while days < n_days:
        if d.weekday() < 5:
            for hour in (2, 3, 4, 6, 7):
                px = Decimal(str(base + 10 * k))
                bars.append(IntradayBar(datetime(d.year, d.month, d.day, hour), px, px + 50, px - 50, px + 20, 10_000 + k))
                k += 1
            days += 1
        d += timedelta(days=1)
    return bars


class FakeClient:
    """Stands in for DnseClient: serves canned series, records calls."""

    def __init__(self, series: dict[str, list[DailyBar]], intraday: dict[str, list[IntradayBar]] | None = None,
                 warn_on: dict[str, str] | None = None):
        self.series = series
        self.intraday = intraday or {}
        self.warn_on = warn_on or {}
        self.calls: list[tuple[str, date, date]] = []
        self.intraday_calls: list[tuple[str, date, date]] = []
        self.warnings: list[str] = []

    def fetch_daily(self, symbol, kind, start, end):
        self.calls.append((symbol, start, end))
        if symbol not in self.series:
            raise DnseInvalidSymbol(symbol)
        if symbol in self.warn_on:
            self.warnings.append(self.warn_on[symbol])
        return [b for b in self.series[symbol] if start <= b.trade_date <= end]

    def fetch_intraday(self, symbol, kind, start, end, resolution="1H"):
        from datetime import timezone
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Ho_Chi_Minh")
        self.intraday_calls.append((symbol, start, end))
        if symbol not in self.intraday:
            raise DnseInvalidSymbol(symbol)
        return [b for b in self.intraday[symbol] if start <= b.bar_time.replace(tzinfo=timezone.utc).astimezone(tz).date() <= end]


@pytest.fixture
def after_close():
    """A 'now' safely after the session close on a Friday (2026-09-18, 16:00 ICT)."""
    return datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)


FLOOR = date(2000, 1, 1)


def rows(*items) -> list[SnapshotRow]:
    """rows("AAA", ("BBB", {"exchange": "HNX"})) -> snapshot rows."""
    out = []
    for it in items:
        sym, kw = (it, {}) if isinstance(it, str) else it
        out.append(SnapshotRow(sym, **kw))
    return out


def apply(engine, code, items, eff, *, create=True, **kw):
    """Apply a snapshot in its own transaction; returns the plan items."""
    with session_scope(engine) as s:
        return apply_snapshot(s, code, rows(*items), eff, source="test", floor=FLOOR, create=create, **kw)
