from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select

from conftest import FakeClient, apply, make_bars
from predict_stock.data.calendar import calendar_gaps, sync_trading_calendar
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import TradingCalendar
from predict_stock.db.session import session_scope

START, END = date(2026, 1, 5), date(2026, 9, 18)


def load(engine, cfg, now, series):
    cfg0 = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})
    apply(engine, "U1", list(series), date(2020, 1, 1))
    ingest_universe(engine, FakeClient(series), cfg0, ["U1"], START, END, now=now)
    return cfg0


def test_gaps_separate_holiday_like_days_from_partial_days(engine, cfg, after_close):
    a, b, c = (make_bars(START, 30, base=x) for x in (1000, 2000, 3000))
    holiday = a[10].trade_date
    partial = a[15].trade_date
    a, b, c = ([x for x in s if x.trade_date != holiday] for s in (a, b, c))                    # nobody traded: holiday-like
    b, c = ([x for x in s if x.trade_date != partial] for s in (b, c))                          # only A traded: partial
    cfg0 = load(engine, cfg, after_close, {"A": a, "B": b, "C": c})
    with session_scope(engine) as s:
        gaps = calendar_gaps(s, cfg0)
    assert holiday in gaps["holiday_like"] and partial not in gaps["holiday_like"]
    assert (partial, 1) in gaps["partial"]
    assert gaps["trading_days"] == 28 and gaps["by_year"][2026]["weekday_gaps"] >= 1
    assert gaps["first"] == a[0].trade_date and gaps["last"] == a[-1].trade_date


def test_empty_calendar_report(engine, cfg):
    with session_scope(engine) as s:
        assert calendar_gaps(s, cfg)["trading_days"] == 0


def test_sync_is_idempotent_and_repairs_the_table(engine, cfg, after_close):
    cfg0 = load(engine, cfg, after_close, {"A": make_bars(START, 20), "B": make_bars(START, 20, base=500)})
    with session_scope(engine) as s:
        assert sync_trading_calendar(s, cfg0) == {"added": 0, "removed": 0, "total": 20}
        s.add(TradingCalendar(calendar_code=cfg0.ingest.calendar_code, trade_date=date(2026, 1, 3), source="bogus"))   # a Saturday
        s.execute(TradingCalendar.__table__.delete().where(TradingCalendar.trade_date == date(2026, 1, 6)))
    with session_scope(engine) as s:
        assert sync_trading_calendar(s, cfg0) == {"added": 1, "removed": 1, "total": 20}
        days = set(s.scalars(select(TradingCalendar.trade_date)))
        assert date(2026, 1, 3) not in days and date(2026, 1, 6) in days
        assert sync_trading_calendar(s, cfg0) == {"added": 0, "removed": 0, "total": 20}
