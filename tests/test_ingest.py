from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import func, select

from conftest import FakeClient, apply, make_bars
from predict_stock.data.dnse_client import DnseError
from predict_stock.data.ingest import _drifted, compute_trading_days, ingest_universe, is_final
from predict_stock.db.models import (
    ConfigSnapshot, DataIngestRun, Instrument, JobRun, PriceBar, PriceBarRevision, TradingCalendar,
)
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.instruments import rename_symbol
from predict_stock.runs import save_config_snapshot

START, END = date(2026, 8, 3), date(2026, 9, 18)


@pytest.fixture
def cfg0(cfg):
    """Config without benchmark indices, so FakeClient only needs the stocks."""
    return cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})


def ingest(engine, client, cfg0, now, codes=("U1",), **kw):
    return ingest_universe(engine, client, cfg0, list(codes), START, END, now=now, **kw)


def snapshot(engine):
    with session_scope(engine) as s:
        return {
            (r.instrument_id, r.trade_date): (r.open, r.high, r.low, r.close, r.volume, r.fetched_at, r.run_id)
            for r in s.scalars(select(PriceBar))
        }


def n(engine, model):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_first_ingest_stores_bars_and_run_records(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA", "BBB"], date(2020, 1, 1))
    series = {"AAA": make_bars(START, 30), "BBB": make_bars(START, 30, base=20000)}
    stats = ingest(engine, FakeClient(series), cfg0, after_close)
    assert stats["totals"] == {"fetched": 60, "inserted": 60, "updated": 0, "unchanged": 0}
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 60
        assert {b.price_basis for b in s.scalars(select(PriceBar))} == {"vendor_adjusted"}    # DNSE gives adjusted prices only
        run = s.scalars(select(JobRun).where(JobRun.job_name == "ingest_ohlcv")).one()
        assert run.status == "success" and run.seed == cfg0.seed and run.finished_at is not None
        assert s.get(ConfigSnapshot, run.config_snapshot_id).content == cfg0.snapshot()       # principle 6
        assert run.params["universes"] == ["U1"]
        assert (run.git_commit is None) == (run.git_dirty is None)
        per = {d.instrument_id: d for d in s.scalars(select(DataIngestRun))}                   # one row per instrument
        assert len(per) == 2 and all(d.run_id == run.id and d.mode == "full" and d.inserted == 30 for d in per.values())


def test_rerun_on_unchanged_data_writes_nothing(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA", "BBB"], date(2020, 1, 1))
    series = {"AAA": make_bars(START, 30), "BBB": make_bars(START, 30, base=20000)}
    ingest(engine, FakeClient(series), cfg0, after_close)
    before = snapshot(engine)
    stats = ingest(engine, FakeClient(series), cfg0, after_close + timedelta(hours=3))
    assert stats["totals"]["inserted"] == 0 and stats["totals"]["updated"] == 0 and stats["totals"]["unchanged"] > 0
    assert snapshot(engine) == before                                    # values, fetched_at and run_id all untouched
    assert n(engine, PriceBar) == 60 and n(engine, PriceBarRevision) == 0
    assert n(engine, JobRun) == 2                                        # but each run is recorded
    with session_scope(engine) as s:                                     # ...and shares ONE config snapshot
        assert s.scalar(select(func.count()).select_from(ConfigSnapshot)) == 1


def test_config_snapshot_upsert_is_idempotent(engine, cfg):
    with session_scope(engine) as s:
        a = save_config_snapshot(s, cfg.snapshot())
        b = save_config_snapshot(s, cfg.snapshot())
        c = save_config_snapshot(s, {**cfg.snapshot(), "seed": 7})
        assert a == b and c != a
        assert s.scalar(select(func.count()).select_from(ConfigSnapshot)) == 2


def test_incremental_run_appends_only_new_bars(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    full = make_bars(START, 30)
    ingest(engine, FakeClient({"AAA": full[:20]}), cfg0, after_close)
    client = FakeClient({"AAA": full})
    stats = ingest(engine, client, cfg0, after_close)
    (sym,) = stats["symbols"]
    assert sym["mode"] == "incremental" and sym["inserted"] == 10 and sym["updated"] == 0
    assert client.calls[0][1] == full[19].trade_date - timedelta(days=cfg0.ingest.overlap_calendar_days)
    assert {d for _, d in snapshot(engine)} == {b.trade_date for b in full}


def test_readjustment_triggers_full_refetch_and_keeps_old_versions(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    old = make_bars(START, 20)
    ingest(engine, FakeClient({"AAA": old}), cfg0, after_close)
    # DNSE re-adjusts the whole history by x0.8 (e.g. a stock dividend) and adds 5 new bars.
    f = Decimal("0.8")
    adjusted = [replace(b, open=b.open * f, high=b.high * f, low=b.low * f, close=b.close * f) for b in make_bars(START, 25)]
    stats = ingest(engine, FakeClient({"AAA": adjusted}), cfg0, after_close)
    assert stats["drifted"] == ["AAA"] and stats["symbols"][0]["mode"] == "full-after-drift"
    with session_scope(engine) as s:
        rows = {r.trade_date: Decimal(r.close) for r in s.scalars(select(PriceBar))}
        assert rows == {b.trade_date: b.close for b in adjusted}                     # no spliced series
        revs = {r.trade_date: Decimal(r.close) for r in s.scalars(select(PriceBarRevision))}
        assert revs == {b.trade_date: b.close for b in old}                          # the previous version is preserved
        rev = s.scalars(select(PriceBarRevision)).first()
        assert rev.price_basis == "vendor_adjusted" and rev.superseded_by_run_id is not None


def test_small_close_noise_below_tolerance_is_not_drift(cfg):
    class Row:  # minimal stand-in for PriceBar
        close = Decimal("50000.0000")
    bar = make_bars(START, 1)[0]
    d = bar.trade_date
    tiny = replace(bar, close=Decimal("50000") * (1 + Decimal("0.0001")))
    big = replace(bar, close=Decimal("50000") * (1 + Decimal("0.01")))
    assert not _drifted({d: Row()}, [tiny], cfg.ingest.adjust_rel_tolerance)
    assert _drifted({d: Row()}, [big], cfg.ingest.adjust_rel_tolerance)
    assert not _drifted({}, [big], cfg.ingest.adjust_rel_tolerance)


def test_is_final_only_after_close_plus_buffer(cfg):
    tz = timezone(timedelta(hours=7))
    d = date(2026, 9, 18)
    at = lambda h, m: datetime(2026, 9, 18, h, m, tzinfo=tz)
    assert is_final(date(2026, 9, 17), at(9, 0), cfg)
    assert not is_final(d, at(9, 0), cfg) and not is_final(d, at(15, 14), cfg)
    assert is_final(d, at(15, 15), cfg)
    assert not is_final(date(2026, 9, 19), at(23, 0), cfg)


def test_todays_partial_bar_is_not_stored(engine, cfg0):
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    bars = make_bars(date(2026, 9, 1), 14)                                # ends 2026-09-18
    midday = datetime(2026, 9, 18, 4, 0, tzinfo=timezone.utc)             # 11:00 ICT, market open
    ingest_universe(engine, FakeClient({"AAA": bars}), cfg0, ["U1"], date(2026, 9, 1), END, now=midday)
    with session_scope(engine) as s:
        assert s.scalar(select(func.max(PriceBar.trade_date))) == date(2026, 9, 17)


def test_failure_is_isolated_and_recorded(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA", "ZZZ"], date(2020, 1, 1))
    with pytest.raises(DnseError, match="ZZZ"):
        ingest(engine, FakeClient({"AAA": make_bars(START, 10)}), cfg0, after_close)
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 10               # AAA was saved
        run = s.scalars(select(JobRun).where(JobRun.job_name == "ingest_ohlcv")).one()
        assert run.status == "failed" and "ZZZ" in run.stats["failures"] and run.error and run.finished_at
        runs = {(d.status, d.mode): d for d in s.scalars(select(DataIngestRun))}         # success AND failure are recorded
        assert set(runs) == {("ok", "full"), ("failed", "failed")}
        assert "ZZZ" in runs[("failed", "failed")].error and runs[("failed", "failed")].duration_ms is not None
        assert runs[("ok", "full")].params["start"] == START.isoformat() and runs[("ok", "full")].inserted == 10


def test_empty_universe_is_an_error(engine, cfg0, after_close):
    with pytest.raises(DnseError, match="no members"):
        ingest(engine, FakeClient({}), cfg0, after_close, codes=("NOPE",))


def test_only_instruments_that_were_members_in_range_are_ingested(engine, cfg0, after_close):
    apply(engine, "U1", ["OLD"], date(2015, 1, 1))
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))                       # OLD left in 2020
    apply(engine, "U1", ["AAA", "NEW"], date(2027, 1, 1))                # NEW joins after the window
    series = {s: make_bars(START, 5) for s in ("AAA", "OLD", "NEW")}
    ingest(engine, FakeClient(series), cfg0, after_close)
    with session_scope(engine) as s:
        ingested = set(s.scalars(select(PriceBar.instrument_id).distinct()))
        assert ingested == {find_symbol_row(s, "AAA").instrument_id}


def test_several_universes_are_merged_without_duplicates(engine, cfg0, after_close):
    apply(engine, "TRAIN", ["AAA", "BBB", "CCC"], date(2020, 1, 1))
    apply(engine, "TRADE", ["AAA", "BBB"], date(2020, 1, 1))
    series = {s: make_bars(START, 5) for s in ("AAA", "BBB", "CCC")}
    stats = ingest(engine, FakeClient(series), cfg0, after_close, codes=("TRAIN", "TRADE"))
    assert sorted(x["symbol"] for x in stats["symbols"]) == ["AAA", "BBB", "CCC"]


def test_rename_between_runs_keeps_one_history_under_the_same_instrument(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    bars = make_bars(START, 30)
    ingest(engine, FakeClient({"AAA": bars[:20]}), cfg0, after_close)
    with session_scope(engine) as s:
        iid = find_symbol_row(s, "AAA").instrument_id
        rename_symbol(s, "AAA", "AAX", date(2026, 9, 1))
    stats = ingest(engine, FakeClient({"AAX": bars}), cfg0, after_close)         # the vendor now serves it as AAX
    assert stats["symbols"][0]["symbol"] == "AAX" and stats["symbols"][0]["mode"] == "incremental"
    with session_scope(engine) as s:
        assert set(s.scalars(select(PriceBar.instrument_id).distinct())) == {iid}
        assert s.scalar(select(func.count()).select_from(PriceBar)) == 30 and s.scalar(select(func.count()).select_from(Instrument)) == 1


def test_benchmarks_are_index_instruments_and_stay_out_of_the_calendar(engine, cfg, after_close):
    cfg_b = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": ["IDX"]})})
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    stocks, idx = make_bars(START, 10), make_bars(START, 12, base=1200, step=1)     # the index has two extra days
    ingest_universe(engine, FakeClient({"AAA": stocks, "IDX": idx}), cfg_b, ["U1"], START, END, now=after_close)
    with session_scope(engine) as s:
        assert s.get(Instrument, find_symbol_row(s, "IDX").instrument_id).kind == "index"
        assert s.get(Instrument, find_symbol_row(s, "AAA").instrument_id).kind == "stock"
        cal = set(s.scalars(select(TradingCalendar.trade_date).where(TradingCalendar.calendar_code == cfg.ingest.calendar_code)))
        assert cal == {b.trade_date for b in stocks}


def test_calendar_is_recomputed_and_stale_days_removed(engine, cfg0, after_close):
    apply(engine, "U1", ["AAA"], date(2020, 1, 1))
    with session_scope(engine) as s:
        s.add(TradingCalendar(calendar_code=cfg0.ingest.calendar_code, trade_date=date(2026, 8, 1), source="bogus"))
        s.add(TradingCalendar(calendar_code="OTHER", trade_date=date(2026, 8, 1), source="untouched"))
    stats = ingest(engine, FakeClient({"AAA": make_bars(START, 5)}), cfg0, after_close)
    assert stats["trading_days"] == {"added": 5, "removed": 1, "total": 5}
    with session_scope(engine) as s:
        assert s.get(TradingCalendar, ("OTHER", date(2026, 8, 1))) is not None       # other calendars are left alone


def test_compute_trading_days_consensus():
    d = lambda k: date(2026, 1, k)
    rows = [("A", d(1)), ("B", d(1)), ("C", d(1)),
            ("A", d(2)), ("B", d(2)),               # C missing (suspended)   -> 2/3 present: trading day
            ("A", d(3)),                            # only 1 of 3 present      -> not a trading day
            ("A", d(4)), ("B", d(4)), ("C", d(4)),
            ("D", d(4)), ("D", d(5))]               # D lists on day 4; on day 5 only D has a bar
    frame = pd.DataFrame(rows, columns=["symbol", "trade_date"])
    # A, B, C stay "active" for the grace period after their last bar, so D alone cannot make day 5 a trading day
    assert compute_trading_days(frame, 0.5) == {d(1), d(2), d(4)}
    # with no grace they are treated as gone after day 4 and D's bar counts (the old, weaker rule)
    assert compute_trading_days(frame, 0.5, grace_days=0) == {d(1), d(2), d(4), d(5)}
    rows += [("A", d(6)), ("B", d(6)), ("C", d(6))]
    assert d(5) not in compute_trading_days(pd.DataFrame(rows, columns=["symbol", "trade_date"]), 0.5, grace_days=0)
    assert compute_trading_days(pd.DataFrame(columns=["symbol", "trade_date"]), 0.5) == set()


def test_a_stray_bar_ahead_of_the_others_does_not_create_a_trading_day(engine, cfg0, after_close):
    apply(engine, "U1", ["A", "B", "C"], date(2020, 1, 1))
    a = make_bars(START, 20)
    stray = replace(a[-1], trade_date=a[-1].trade_date + timedelta(days=4))       # only A has data for this date
    ingest(engine, FakeClient({"A": a + [stray], "B": make_bars(START, 20, base=2000), "C": make_bars(START, 20, base=3000)}), cfg0, after_close)
    with session_scope(engine) as s:
        cal = set(s.scalars(select(TradingCalendar.trade_date).where(TradingCalendar.calendar_code == cfg0.ingest.calendar_code)))
    assert stray.trade_date not in cal and {b.trade_date for b in a} <= cal
