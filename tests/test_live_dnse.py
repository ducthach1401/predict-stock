"""Hits the real DNSE endpoint. Excluded by default; run with:  pytest -m live"""
from datetime import date

from sqlalchemy import func, select

import pytest

from predict_stock.data.dnse_client import DnseClient, DnseInvalidSymbol

pytestmark = pytest.mark.live


def test_live_stock_bars(cfg):
    bars = DnseClient(cfg.dnse).fetch_daily("VCB", "stock", date(2025, 1, 2), date(2025, 1, 15))
    assert 5 <= len(bars) <= 10
    assert bars[0].trade_date == date(2025, 1, 2)
    assert all(b.trade_date.weekday() < 5 for b in bars)
    assert all(10_000 < b.close < 300_000 for b in bars)          # VND, not thousand VND
    assert all(b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close) for b in bars)


def test_live_index_and_invalid_symbol(cfg):
    (first, *_), = [DnseClient(cfg.dnse).fetch_daily("VNINDEX", "index", date(2025, 1, 2), date(2025, 1, 3))]
    assert 800 < first.close < 3000                               # points
    with pytest.raises(DnseInvalidSymbol):
        DnseClient(cfg.dnse).fetch_daily("ZZZZ", "stock", date(2025, 1, 2), date(2025, 1, 3))


# ---- backfill timing against the real endpoint (test database) -----------------------------------------------
def test_live_backfill_of_new_members_is_timed(engine, cfg, capsys):
    """Measures how long a brand-new member takes end to end (fetch + write + checks). Run with `-s` to see it."""
    from datetime import date
    from sqlalchemy import select
    from conftest import apply
    from predict_stock.data.backfill import backfill, ready_members
    from predict_stock.db.models import DataIngestRun, PriceBar
    from predict_stock.db.session import session_scope

    live = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []}),
                                  "universe": cfg.universe.model_copy(update={"training_code": "LIVE", "trading_code": "LIVE"})})
    symbols = ["SAB", "BCM", "BVH"]                                          # not part of LARGE50
    apply(engine, "LIVE", symbols, date.today())
    stats = backfill(engine, DnseClient(cfg.dnse), live, ["LIVE"], date.fromisoformat(cfg.ingest.history_start), date.today())
    assert set(stats["new_instruments"]) == set(symbols) and stats["blocked"] == {}
    with session_scope(engine) as s:
        rows = list(s.scalars(select(DataIngestRun)))
        assert all(r.status == "ok" and r.mode == "full" and r.duration_ms is not None and r.inserted > 2000 for r in rows)
        assert len(ready_members(s, live, date.today())) == 3
        n = s.scalar(select(func.count()).select_from(PriceBar))
    per = {sym: (stats["duration_ms"][sym]) for sym in symbols}
    with capsys.disabled():
        print(f"\nLIVE backfill of a new member, ms per symbol: {per}  (bars stored: {n})")


def test_live_intraday_backfill_is_timed(engine, cfg, capsys):
    from datetime import date
    from conftest import apply
    from predict_stock.data.ingest import ingest_universe
    from predict_stock.db.models import DataIngestRun, PriceBarIntraday
    from predict_stock.db.session import session_scope

    c = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": [], "intraday": cfg.ingest.intraday.model_copy(update={"enabled": True})})})
    apply(engine, "LIVE", ["SAB"], date.today())
    stats = ingest_universe(engine, DnseClient(cfg.dnse), c, ["LIVE"], date(2026, 8, 1), date.today())
    with session_scope(engine) as s:
        rows = {r.resolution: r for r in s.scalars(select(DataIngestRun))}
        n = s.scalar(select(func.count()).select_from(PriceBarIntraday))
        lo, hi = s.execute(select(func.min(PriceBarIntraday.bar_time), func.max(PriceBarIntraday.bar_time))).one()
    assert n > 100 and set(rows) == {"1D", "1H"} and lo.hour in (2, 3, 4, 6, 7)          # UTC hours of 09,10,11,13,14 ICT
    with capsys.disabled():
        print(f"\nLIVE 1H: {n} bars {lo} → {hi} UTC; ms 1D={rows['1D'].duration_ms} 1H={rows['1H'].duration_ms}")
