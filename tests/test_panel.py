"""Price panel loading: repair rule, calendar, cutoff, benchmark, membership from the database."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from conftest import FakeClient, apply, make_bars
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.features.data import load_panel, membership_mask, repair_ohlc, splice_benchmark

START, END = date(2026, 1, 5), date(2026, 9, 18)
NaN = np.nan


def frames(o, h, l, c, v=None):
    idx = pd.bdate_range("2024-01-01", periods=len(c))
    mk = lambda a: pd.DataFrame(np.asarray(a, float).reshape(-1, 1), index=idx, columns=[1])
    return mk(o), mk(h), mk(l), mk(c), mk(v if v is not None else np.ones(len(c)))


# ---- the repair rule ---------------------------------------------------------------------------------------
def test_consistent_bars_are_untouched():
    o, h, l, c, v = frames([10, 11], [12, 12], [9, 10], [11, 11.5])
    o2, h2, l2, c2, v2, n = repair_ohlc(o, h, l, c, v)
    assert n == 0 and o2.equals(o) and h2.equals(h) and l2.equals(l)


def test_open_outside_the_range_is_dropped_and_the_range_is_kept():
    o, h, l, c, v = frames([10, 13.0], [12, 12.5], [9, 11.0], [11, 11.5])     # day 2: open 13 above high 12.5
    o2, h2, l2, c2, _, n = repair_ohlc(o, h, l, c, v)
    assert n == 1 and np.isnan(o2.iloc[1, 0]) and h2.iloc[1, 0] == 12.5 and l2.iloc[1, 0] == 11.0 and o2.iloc[0, 0] == 10


def test_close_outside_the_range_widens_the_range_to_contain_it():
    o, h, l, c, v = frames([10, 11.0], [12, 12.0], [9, 11.0], [11, 10.5])     # day 2: close 10.5 below low 11
    o2, h2, l2, c2, _, n = repair_ohlc(o, h, l, c, v)
    assert n == 1 and l2.iloc[1, 0] == 10.5 and h2.iloc[1, 0] == 12.0 and o2.iloc[1, 0] == 11.0 and c2.iloc[1, 0] == 10.5   # open is inside: kept


def test_high_below_low_is_reordered():
    o, h, l, c, v = frames([10.0], [9.0], [11.0], [10.0])
    o2, h2, l2, c2, _, n = repair_ohlc(o, h, l, c, v)
    assert n == 1 and (l2.iloc[0, 0], h2.iloc[0, 0]) == (9.0, 11.0) and o2.iloc[0, 0] == 10.0


def test_non_positive_prices_empty_the_bar_and_missing_bars_stay_missing():
    o, h, l, c, v = frames([10, 10, NaN], [11, 11, NaN], [9, 0, NaN], [10, 10, NaN], [5, 5, NaN])
    o2, h2, l2, c2, v2, n = repair_ohlc(o, h, l, c, v)
    assert n == 1 and c2.iloc[1, 0] != c2.iloc[1, 0] and v2.iloc[1, 0] != v2.iloc[1, 0] and c2.iloc[0, 0] == 10
    assert np.isnan(c2.iloc[2, 0])


# ---- benchmark splice ------------------------------------------------------------------------------------------------
def test_benchmark_splice_keeps_levels_continuous_and_uses_no_later_value():
    idx = pd.bdate_range("2024-01-01", periods=8)
    primary = pd.Series([NaN, NaN, NaN, 800, 808, 816, 824, 832.0], index=idx)
    fallback = pd.Series([1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070.0], index=idx)
    s = splice_benchmark(primary, fallback)
    assert s.iloc[3:].tolist() == primary.iloc[3:].tolist()                      # the primary wins wherever it exists
    assert s.iloc[3] == 800 and s.iloc[2] == pytest.approx(1020 * 800 / 1030)   # fallback rescaled to agree on the first primary date
    assert (s.iloc[:3] / s.shift().iloc[:3]).dropna().tolist() == pytest.approx([1010 / 1000, 1020 / 1010])   # its own returns
    assert splice_benchmark(primary, None).equals(primary)
    assert splice_benchmark(pd.Series(np.nan, index=idx), fallback).equals(fallback)   # primary has no data at all


# ---- loading from the database --------------------------------------------------------------------------------------------
def setup_db(engine, cfg, now, series, benchmarks=()):
    cfg_b = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": list(benchmarks)})})
    apply(engine, "U1", [s for s in series if s not in benchmarks], date(2020, 1, 1))
    ingest_universe(engine, FakeClient(series), cfg_b, ["U1"], START, END, now=now)
    with session_scope(engine) as s:
        return cfg_b, {sym: find_symbol_row(s, sym).instrument_id for sym in series}


def test_panel_uses_calendar_days_repairs_bars_and_respects_the_cutoff(engine, cfg, after_close):
    a, b, c = make_bars(START, 40), make_bars(START, 40, base=20000), make_bars(START, 40, base=30000)
    ghost = replace(a[-1], trade_date=a[-1].trade_date + timedelta(days=4))                 # only A has a bar that day: not a trading day
    a = a + [ghost]
    a[10] = replace(a[10], open=a[10].high + 90)                                              # open outside the range
    cfg0, ids = setup_db(engine, cfg, after_close, {"A": a, "B": b, "C": c})
    days = [x.trade_date for x in b]
    with session_scope(engine) as s:
        panel = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=list(ids.values()), load_start=START, cutoff=days[29])
    assert panel.calendar[-1] == pd.Timestamp(days[29]) and len(panel.calendar) == 30          # nothing after the cutoff
    assert panel.repaired_bars == 1 and np.isnan(panel.open[ids["A"]].iloc[10]) and panel.close[ids["A"]].notna().all()
    with session_scope(engine) as s:
        full = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=list(ids.values()), load_start=START, cutoff=END)
    assert full.dropped_non_calendar == 1 and pd.Timestamp(ghost.trade_date) not in full.calendar
    assert list(full.close.columns) == sorted(ids.values()) and full.close.shape == (40, 3)


def test_benchmark_is_aligned_to_the_calendar_with_a_limited_forward_fill(engine, cfg, after_close):
    stocks = {s: make_bars(START, 40, base=1000 * (i + 1)) for i, s in enumerate("ABC")}
    idx = make_bars(START, 40, base=1200, step=1)
    idx = [x for k, x in enumerate(idx) if k not in (10, 11, 12, 13, 14, 15, 16, 17)]         # the index misses 8 sessions in a row
    cfg_b, ids = setup_db(engine, cfg, after_close, {**stocks, "IDX": idx}, benchmarks=["IDX"])
    days = [x.trade_date for x in stocks["A"]]
    with session_scope(engine) as s:
        panel = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=[ids[k] for k in "ABC"], load_start=START, cutoff=END,
                           bench_symbols=("IDX", None), bench_ffill_limit=3)
    b = panel.bench
    assert b.iloc[9] == pytest.approx(float(idx[9].close)) and b.iloc[10] == b.iloc[9] and b.iloc[12] == b.iloc[9]     # carried forward for 3 sessions
    assert np.isnan(b.iloc[13]) and np.isnan(b.iloc[17]) and b.iloc[18] == pytest.approx(float(idx[10].close))        # then unknown, never invented
    assert panel.bench_info["primary"] == "IDX" and panel.bench_info["first_valid"] == str(days[0])


def test_missing_benchmark_gives_none(engine, cfg, after_close):
    cfg_b, ids = setup_db(engine, cfg, after_close, {"A": make_bars(START, 30)})
    with session_scope(engine) as s:
        p = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=[ids["A"]], load_start=START, cutoff=END, bench_symbols=("NOPE", None))
    assert p.bench is None and p.bench_info is None


def test_membership_mask_from_database_intervals(engine, cfg, after_close):
    series = {s: make_bars(START, 40, base=1000 * (i + 1)) for i, s in enumerate("ABC")}
    days = [x.trade_date for x in series["A"]]
    apply(engine, "U1", ["A", "B"], days[5])                                       # B and A from day 5
    apply(engine, "U1", ["A", "B", "C"], days[15])                                 # C joins on day 15
    apply(engine, "U1", ["A", "C"], days[25])                                      # B leaves on day 25
    ingest_universe(engine, FakeClient(series), cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})}), ["U1"], START, END, now=after_close)
    with session_scope(engine) as s:
        ids = {k: find_symbol_row(s, k).instrument_id for k in "ABC"}
        panel = load_panel(s, calendar_code=cfg.ingest.calendar_code, instrument_ids=list(ids.values()), load_start=START, cutoff=END)
        mask = membership_mask(s, "U1", panel.calendar, list(ids.values()))
    m = mask[[ids[k] for k in "ABC"]].to_numpy()
    assert m[4].tolist() == [False] * 3 and m[5].tolist() == [True, True, False]
    assert m[15].tolist() == [True, True, True] and m[24][1] and not m[25][1] and m[25].tolist() == [True, False, True]
