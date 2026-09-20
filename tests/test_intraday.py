"""Optional 1H bars: off by default, incremental and idempotent when enabled."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select

from conftest import FakeClient, apply, make_bars, make_intraday
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import DataIngestRun, PriceBarIntraday
from predict_stock.db.session import session_scope

START, END = date(2026, 8, 3), date(2026, 9, 18)


def cfg_with(cfg, enabled, benchmarks=()):
    icfg = cfg.ingest.intraday.model_copy(update={"enabled": enabled, "history_start": "2026-08-03"})
    return cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": list(benchmarks), "intraday": icfg})})


def n_rows(engine):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(PriceBarIntraday))


def test_intraday_is_off_by_default(cfg):
    assert cfg.ingest.intraday.enabled is False


def test_disabled_flag_fetches_and_stores_no_intraday(engine, cfg, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    client = FakeClient({"A": make_bars(START, 10)}, intraday={"A": make_intraday(START, 10)})
    ingest_universe(engine, client, cfg_with(cfg, False), ["U1"], START, END, now=after_close)
    assert client.intraday_calls == [] and n_rows(engine) == 0
    with session_scope(engine) as s:
        assert {r.resolution for r in s.scalars(select(DataIngestRun))} == {"1D"}


def test_enabled_flag_stores_1h_bars_for_stocks_only(engine, cfg, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    c = cfg_with(cfg, True, benchmarks=["IDX"])
    client = FakeClient({"A": make_bars(START, 10), "IDX": make_bars(START, 10, base=1200, step=1)}, intraday={"A": make_intraday(START, 10)})
    stats = ingest_universe(engine, client, c, ["U1"], START, END, now=after_close)
    assert n_rows(engine) == 50 and stats["totals_intraday"]["inserted"] == 50               # 10 days x 5 bars
    assert [x[0] for x in client.intraday_calls] == ["A"]                                     # the index is skipped
    with session_scope(engine) as s:
        first = s.scalars(select(PriceBarIntraday).order_by(PriceBarIntraday.bar_time)).first()
        assert first.resolution == "1H" and first.bar_time.hour == 2 and first.price_basis == "vendor_adjusted"      # UTC
        runs = {(r.resolution, r.status) for r in s.scalars(select(DataIngestRun))}
        assert ("1H", "ok") in runs and ("1D", "ok") in runs


def test_intraday_rerun_is_idempotent_and_incremental(engine, cfg, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    c = cfg_with(cfg, True)
    bars = make_intraday(START, 12)
    ingest_universe(engine, FakeClient({"A": make_bars(START, 12)}, intraday={"A": bars[:50]}), c, ["U1"], START, END, now=after_close)   # first 10 days
    assert n_rows(engine) == 50
    with session_scope(engine) as s:
        before = {(r.bar_time): (r.close, r.fetched_at, r.run_id) for r in s.scalars(select(PriceBarIntraday))}
    client = FakeClient({"A": make_bars(START, 12)}, intraday={"A": bars})
    stats = ingest_universe(engine, client, c, ["U1"], START, END, now=after_close + timedelta(hours=2))
    assert n_rows(engine) == 60 and stats["totals_intraday"]["inserted"] == 10                 # only the two new days
    assert client.intraday_calls[0][1] == (bars[49].bar_time.date() - timedelta(days=c.ingest.intraday.overlap_calendar_days))
    with session_scope(engine) as s:
        after = {(r.bar_time): (r.close, r.fetched_at, r.run_id) for r in s.scalars(select(PriceBarIntraday))}
    assert all(after[k] == v for k, v in before.items())                                       # existing rows untouched
    again = ingest_universe(engine, FakeClient({"A": make_bars(START, 12)}, intraday={"A": bars}), c, ["U1"], START, END, now=after_close)
    assert again["totals_intraday"]["inserted"] == 0 and again["totals_intraday"]["updated"] == 0 and n_rows(engine) == 60


def test_intraday_changed_bar_is_updated(engine, cfg, after_close):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    c = cfg_with(cfg, True)
    bars = make_intraday(START, 6)
    ingest_universe(engine, FakeClient({"A": make_bars(START, 6)}, intraday={"A": bars}), c, ["U1"], START, END, now=after_close)
    changed = list(bars)
    changed[-1] = replace(changed[-1], close=changed[-1].close + 500)
    stats = ingest_universe(engine, FakeClient({"A": make_bars(START, 6)}, intraday={"A": changed}), c, ["U1"], START, END, now=after_close)
    assert stats["totals_intraday"]["updated"] == 1
    with session_scope(engine) as s:
        last = s.scalars(select(PriceBarIntraday).order_by(PriceBarIntraday.bar_time.desc())).first()
        assert Decimal(last.close) == changed[-1].close


def test_todays_intraday_bars_wait_for_the_close(engine, cfg):
    apply(engine, "U1", ["A"], date(2020, 1, 1))
    c = cfg_with(cfg, True)
    midday = datetime(2026, 9, 18, 4, 30, tzinfo=timezone.utc)                                  # 11:30 ICT
    bars = make_intraday(date(2026, 9, 14), 5)                                                   # Mon-Fri, ends 2026-09-18
    ingest_universe(engine, FakeClient({"A": make_bars(date(2026, 9, 14), 5)}, intraday={"A": bars}), c, ["U1"], date(2026, 9, 14), END, now=midday)
    with session_scope(engine) as s:
        assert s.scalar(select(func.max(PriceBarIntraday.bar_time))).date() == date(2026, 9, 17)
