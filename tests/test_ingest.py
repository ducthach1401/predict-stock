from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import func, select

from conftest import FakeClient, make_bars
from predict_stock.data.dnse_client import DnseError
from predict_stock.data.ingest import _drifted, compute_trading_days, ingest_universe, is_final
from predict_stock.db.models import Instrument, JobRun, OhlcvDaily, TradingDay
from predict_stock.db.session import session_scope
from predict_stock.universe import load_membership_csv

START, END = date(2026, 8, 3), date(2026, 9, 18)


@pytest.fixture
def cfg0(cfg):
    """Config without benchmark indices, so FakeClient only needs the stocks."""
    return cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})


def load_universe(engine, tmp_path, rows="U1,AAA,2020-01-01,\nU1,BBB,2020-01-01,\n"):
    p = tmp_path / "m.csv"
    p.write_text("universe_code,symbol,effective_from,effective_to\n" + rows)
    with session_scope(engine) as s:
        load_membership_csv(s, p)


def snapshot(engine):
    with session_scope(engine) as s:
        return {
            (r.symbol, r.trade_date): (r.open, r.high, r.low, r.close, r.volume, r.fetched_at, r.run_id)
            for r in s.scalars(select(OhlcvDaily))
        }


def test_first_ingest_stores_bars_and_run_record(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path)
    series = {"AAA": make_bars(START, 30), "BBB": make_bars(START, 30, base=20000)}
    stats = ingest_universe(engine, FakeClient(series), cfg0, "U1", START, END, now=after_close)
    assert stats["totals"] == {"fetched": 60, "inserted": 60, "updated": 0, "unchanged": 0}
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(OhlcvDaily)) == 60
        run = s.scalars(select(JobRun)).one()
        assert run.status == "success" and run.seed == cfg0.seed
        assert run.config_snapshot == cfg0.snapshot()          # principle 6: config snapshot
        assert run.params["universe"] == "U1" and run.finished_at is not None
        # git commit/dirty are recorded when git can provide them (None before the first commit)
        assert (run.git_commit is None) == (run.git_dirty is None)


def test_rerun_on_unchanged_data_writes_nothing(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path)
    series = {"AAA": make_bars(START, 30), "BBB": make_bars(START, 30, base=20000)}
    ingest_universe(engine, FakeClient(series), cfg0, "U1", START, END, now=after_close)
    before = snapshot(engine)
    later = after_close + timedelta(hours=3)
    stats = ingest_universe(engine, FakeClient(series), cfg0, "U1", START, END, now=later)
    assert stats["totals"]["inserted"] == 0 and stats["totals"]["updated"] == 0
    assert stats["totals"]["unchanged"] > 0
    assert snapshot(engine) == before        # values, fetched_at and run_id all untouched
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(OhlcvDaily)) == 60   # no duplicates
        assert s.scalar(select(func.count()).select_from(JobRun)) == 2        # but each run is recorded


def test_incremental_run_appends_only_new_bars(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\n")
    full = make_bars(START, 30)
    ingest_universe(engine, FakeClient({"AAA": full[:20]}), cfg0, "U1", START, END, now=after_close)
    client = FakeClient({"AAA": full})
    stats = ingest_universe(engine, client, cfg0, "U1", START, END, now=after_close)
    (sym,) = stats["symbols"]
    assert sym["mode"] == "incremental" and sym["inserted"] == 10 and sym["updated"] == 0
    # it re-fetched only an overlap window, not the full history
    assert client.calls[0][1] == full[19].trade_date - timedelta(days=cfg0.ingest.overlap_calendar_days)
    assert snapshot(engine).keys() == {("AAA", b.trade_date) for b in full}


def test_readjustment_triggers_full_refetch_without_mixing_series(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\n")
    old = make_bars(START, 20)
    ingest_universe(engine, FakeClient({"AAA": old}), cfg0, "U1", START, END, now=after_close)

    # DNSE re-adjusts the whole history by x0.8 (e.g. a stock dividend) and adds 5 new bars.
    f = Decimal("0.8")
    adjusted = [replace(b, open=b.open * f, high=b.high * f, low=b.low * f, close=b.close * f) for b in make_bars(START, 25)]
    stats = ingest_universe(engine, FakeClient({"AAA": adjusted}), cfg0, "U1", START, END, now=after_close)

    assert stats["drifted"] == ["AAA"]
    assert stats["symbols"][0]["mode"] == "full-after-drift"
    with session_scope(engine) as s:
        rows = {r.trade_date: Decimal(r.close) for r in s.scalars(select(OhlcvDaily))}
    assert rows == {b.trade_date: b.close for b in adjusted}     # every row, including old ones, is adjusted


def test_small_close_noise_below_tolerance_is_not_drift(cfg):
    class Row:  # minimal stand-in for OhlcvDaily
        close = Decimal("50000.0000")
    bar = make_bars(START, 1)[0]
    d = bar.trade_date
    tiny = replace(bar, close=Decimal("50000") * (1 + Decimal("0.0001")))
    big = replace(bar, close=Decimal("50000") * (1 + Decimal("0.01")))
    assert not _drifted({d: Row()}, [tiny], cfg.ingest.adjust_rel_tolerance)
    assert _drifted({d: Row()}, [big], cfg.ingest.adjust_rel_tolerance)
    assert not _drifted({}, [big], cfg.ingest.adjust_rel_tolerance)     # no overlap -> nothing to compare


def test_is_final_only_after_close_plus_buffer(cfg):
    tz = timezone(timedelta(hours=7))
    d = date(2026, 9, 18)
    at = lambda h, m: datetime(2026, 9, 18, h, m, tzinfo=tz)
    assert is_final(date(2026, 9, 17), at(9, 0), cfg)
    assert not is_final(d, at(9, 0), cfg)
    assert not is_final(d, at(15, 14), cfg)          # close 15:00 + 15 min buffer
    assert is_final(d, at(15, 15), cfg)
    assert not is_final(date(2026, 9, 19), at(23, 0), cfg)   # never a future bar


def test_todays_partial_bar_is_not_stored(engine, cfg0, tmp_path):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\n")
    bars = make_bars(date(2026, 9, 1), 14)            # ends 2026-09-18
    assert bars[-1].trade_date == date(2026, 9, 18)
    midday = datetime(2026, 9, 18, 4, 0, tzinfo=timezone.utc)      # 11:00 ICT, market still open
    ingest_universe(engine, FakeClient({"AAA": bars}), cfg0, "U1", date(2026, 9, 1), END, now=midday)
    with session_scope(engine) as s:
        assert s.scalar(select(func.max(OhlcvDaily.trade_date))) == date(2026, 9, 17)


def test_failure_is_isolated_and_recorded(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\nU1,ZZZ,2020-01-01,\n")
    with pytest.raises(DnseError, match="ZZZ"):
        ingest_universe(engine, FakeClient({"AAA": make_bars(START, 10)}), cfg0, "U1", START, END, now=after_close)
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(OhlcvDaily).where(OhlcvDaily.symbol == "AAA")) == 10
        run = s.scalars(select(JobRun)).one()
        assert run.status == "failed" and "ZZZ" in run.stats["failures"]
        assert run.error and run.finished_at is not None


def test_empty_universe_is_an_error(engine, cfg0, after_close):
    with pytest.raises(DnseError, match="no members"):
        ingest_universe(engine, FakeClient({}), cfg0, "NOPE", START, END, now=after_close)


def test_only_symbols_that_were_members_in_range_are_ingested(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\nU1,OLD,2015-01-01,2020-01-01\nU1,NEW,2027-01-01,\n")
    series = {s: make_bars(START, 5) for s in ("AAA", "OLD", "NEW")}
    ingest_universe(engine, FakeClient(series), cfg0, "U1", START, END, now=after_close)
    with session_scope(engine) as s:
        assert set(s.scalars(select(OhlcvDaily.symbol).distinct())) == {"AAA"}


def test_benchmarks_are_ingested_as_index_kind_and_kept_out_of_calendar(engine, cfg, tmp_path, after_close):
    cfg_b = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": ["IDX"]})})
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\n")
    stocks = make_bars(START, 10)
    idx = make_bars(START, 12, base=1200, step=1)          # index has two extra days
    ingest_universe(engine, FakeClient({"AAA": stocks, "IDX": idx}), cfg_b, "U1", START, END, now=after_close)
    with session_scope(engine) as s:
        assert s.get(Instrument, "IDX").kind == "index" and s.get(Instrument, "AAA").kind == "stock"
        assert set(s.scalars(select(TradingDay.trade_date))) == {b.trade_date for b in stocks}


def test_calendar_is_recomputed_and_stale_days_removed(engine, cfg0, tmp_path, after_close):
    load_universe(engine, tmp_path, "U1,AAA,2020-01-01,\n")
    with session_scope(engine) as s:
        s.add(TradingDay(trade_date=date(2026, 8, 1), source="bogus"))   # a Saturday nobody traded
    stats = ingest_universe(engine, FakeClient({"AAA": make_bars(START, 5)}), cfg0, "U1", START, END, now=after_close)
    assert stats["trading_days"] == {"added": 5, "removed": 1, "total": 5}


def test_compute_trading_days_consensus():
    d = lambda n: date(2026, 1, n)
    rows = [("A", d(1)), ("B", d(1)), ("C", d(1)),
            ("A", d(2)), ("B", d(2)),               # C missing (suspended)   -> 2/3 present: trading day
            ("A", d(3)),                            # only 1 of 3 present      -> not a trading day
            ("A", d(4)), ("B", d(4)), ("C", d(4)),
            ("D", d(4)), ("D", d(5))]               # day 5: A,B,C have no later bars (not active), D is
    days = compute_trading_days(pd.DataFrame(rows, columns=["symbol", "trade_date"]), 0.5)
    assert days == {d(1), d(2), d(4), d(5)}
    # ...but if A, B, C are still active (they trade on day 6) a day-5 bar from D alone is not enough
    rows += [("A", d(6)), ("B", d(6)), ("C", d(6))]
    days = compute_trading_days(pd.DataFrame(rows, columns=["symbol", "trade_date"]), 0.5)
    assert d(5) not in days and d(6) in days
    assert compute_trading_days(pd.DataFrame(columns=["symbol", "trade_date"]), 0.5) == set()
