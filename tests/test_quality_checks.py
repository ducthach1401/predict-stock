"""One test per kind of data defect: each planted defect must be reported by the intended check, and only by it."""
from __future__ import annotations

import re
from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from conftest import FakeClient, apply, make_bars
from predict_stock.data.ingest import ingest_universe
from predict_stock.data.quality import SEVERITY, run_quality_checks, summarize
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope

START, END = date(2026, 1, 5), date(2026, 9, 18)
N = 130                                                       # long enough for the MAD / rolling-volume checks
DAYS = [b.trade_date for b in make_bars(START, N)]


def build(engine, cfg, now, series, warn_on=None):
    cfg0 = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})
    apply(engine, "U1", list(series), date(2020, 1, 1))
    ingest_universe(engine, FakeClient(series, warn_on=warn_on), cfg0, ["U1"], START, END, now=now)
    with session_scope(engine) as s:
        return cfg0, [(find_symbol_row(s, sym).instrument_id, sym) for sym in series]


def check(engine, cfg, now, defect, *, n=N, base=20000.0, warn_on=None):
    """Ingest one defective series next to a clean one; return the findings for the defective symbol."""
    bars = make_bars(START, n, base=base)
    series = {"BAD": defect(bars) if defect else bars, "OK": make_bars(START, n, base=30000)}
    cfg0, pairs = build(engine, cfg, now, series, warn_on)
    with session_scope(engine) as s:
        issues = run_quality_checks(s, pairs, cfg0)
    assert issues[issues.symbol == "OK"].empty, "the clean series must stay clean"
    return issues[issues.symbol == "BAD"]


def kinds(issues):
    return sorted(set(issues["check"]))


def test_clean_series_reports_nothing(engine, cfg, after_close):
    assert check(engine, cfg, after_close, None).empty


# ---- bar consistency ------------------------------------------------------------------------
@pytest.mark.parametrize("field,expect", [
    ("open_above_high", "open outside"), ("open_below_low", "open outside"),
    ("close_above_high", "close outside"), ("close_below_low", "close outside"), ("high_below_low", "high < low"),
])
def test_ohlc_inconsistent_variants(engine, cfg, after_close, field, expect):
    def defect(b):
        x = b[40]
        b[40] = {"open_above_high": replace(x, open=x.high + 60), "open_below_low": replace(x, open=x.low - 60),
                 "close_above_high": replace(x, close=x.high + 60), "close_below_low": replace(x, close=x.low - 60),
                 "high_below_low": replace(x, high=x.low - 10)}[field]
        return b
    issues = check(engine, cfg, after_close, defect)
    row = issues[issues["check"] == "ohlc_inconsistent"]
    assert list(row.trade_date) == [DAYS[40]] and expect in row.detail.iloc[0] and "violation" in row.detail.iloc[0]
    assert set(row.severity) == {"error"}


def test_nonpositive_price(engine, cfg, after_close):
    def defect(b):
        b[50] = replace(b[50], low=Decimal(0))
        return b
    assert "nonpositive_price" in kinds(check(engine, cfg, after_close, defect))


# ---- price unit ------------------------------------------------------------------------------
def test_price_unit_whole_series_in_thousand_vnd(engine, cfg, after_close):
    """A series quoted in thousand VND instead of VND: every close is far below any plausible price."""
    issues = check(engine, cfg, after_close, None, base=20.0)          # SECOND series would be flagged too: use the BAD one only
    # base 20.0 with step 100 rises to ~13,000: only the early bars are implausibly low
    low = issues[issues["check"] == "price_unit"]
    assert len(low) > 0 and low.trade_date.min() == DAYS[0] and "outside the plausible range" in low.detail.iloc[0]


def test_price_unit_single_bar_x1000(engine, cfg, after_close):
    def defect(b):
        x = b[60]
        b[60] = replace(x, open=x.open * 1000, high=x.high * 1000, low=x.low * 1000, close=x.close * 1000)
        return b
    issues = check(engine, cfg, after_close, defect)
    unit = issues[issues["check"] == "price_unit"]
    assert DAYS[60] in set(unit.trade_date) and DAYS[61] in set(unit.trade_date)      # jump up, then back down
    assert any(re.search(r"x\d{3,4} in one session", d) for d in unit.detail)


def test_price_unit_above_range(engine, cfg, after_close):
    def defect(b):
        b[70] = replace(b[70], close=Decimal(9_000_000), high=Decimal(9_000_000))
        return b
    assert any("outside the plausible range" in d for d in check(engine, cfg, after_close, defect).detail)


# ---- volume, repeated bars, outliers ----------------------------------------------------------------
def test_zero_volume(engine, cfg, after_close):
    def defect(b):
        b[30] = replace(b[30], volume=0)
        return b
    issues = check(engine, cfg, after_close, defect)
    assert list(issues[issues["check"] == "zero_volume"].trade_date) == [DAYS[30]]


def test_repeated_bar(engine, cfg, after_close):
    def defect(b):
        b[45] = replace(b[44], trade_date=b[45].trade_date)
        return b
    assert list(check(engine, cfg, after_close, defect).query("`check` == 'repeated_bar'").trade_date) == [DAYS[45]]


def test_volume_spike(engine, cfg, after_close):
    def defect(b):
        b[100] = replace(b[100], volume=b[100].volume * 80)
        return b
    row = check(engine, cfg, after_close, defect).query("`check` == 'volume_spike'")
    assert list(row.trade_date) == [DAYS[100]] and "x" in row.detail.iloc[0]


def test_volume_spike_needs_history(engine, cfg, after_close):
    def defect(b):
        b[5] = replace(b[5], volume=b[5].volume * 80)                      # too early for a 60-session median
        return b
    assert check(engine, cfg, after_close, defect).query("`check` == 'volume_spike'").empty


def test_return_outlier_and_big_move(engine, cfg, after_close):
    def defect(b):
        for i in range(90, N):                                             # a permanent +25% level shift on day 90
            x = b[i]
            b[i] = replace(x, open=x.open * Decimal("1.25"), high=x.high * Decimal("1.25"), low=x.low * Decimal("1.25"), close=x.close * Decimal("1.25"))
        return b
    issues = check(engine, cfg, after_close, defect)
    assert list(issues.query("`check` == 'big_move'").trade_date) == [DAYS[90]]
    out = issues.query("`check` == 'return_outlier'")
    assert list(out.trade_date) == [DAYS[90]] and "robust z" in out.detail.iloc[0]
    assert "price_unit" not in kinds(issues)


def test_ordinary_limit_move_is_not_a_return_outlier(engine, cfg, after_close):
    def defect(b):
        for i in range(90, N):
            x = b[i]
            b[i] = replace(x, open=x.open * Decimal("1.06"), high=x.high * Decimal("1.06"), low=x.low * Decimal("1.06"), close=x.close * Decimal("1.06"))
        return b
    issues = check(engine, cfg, after_close, defect)
    assert "big_move" not in kinds(issues)                                 # +6% is inside the ±7% band


# ---- calendar ---------------------------------------------------------------------------------------------
def test_missing_extra_and_stale(engine, cfg, after_close):
    """Three stocks so that the calendar consensus is well defined."""
    a, b, c = (make_bars(START, 40, base=x) for x in (10000, 20000, 30000))
    gone = c[10].trade_date
    del c[10]                                                              # C skips a day A and B traded
    ghost = replace(a[-1], trade_date=date(2026, 3, 8))                    # a Sunday: bar outside the calendar
    a = a + [ghost]
    a.sort(key=lambda x: x.trade_date)
    b = b[:-3]                                                             # B stops three sessions early
    last_b = b[-1].trade_date
    cfg0, pairs = build(engine, cfg, after_close, {"A": a, "B": b, "C": c})
    with session_scope(engine) as s:
        issues = run_quality_checks(s, pairs, cfg0)
    found = {(r.symbol, r.check, r.trade_date) for r in issues.itertuples()}
    assert ("C", "missing_trading_day", gone) in found
    assert ("A", "extra_day", date(2026, 3, 8)) in found
    assert any(r.symbol == "B" and r.check == "stale_series" and r.trade_date == last_b for r in issues.itertuples())


def test_instrument_without_bars_is_an_error(engine, cfg, after_close):
    cfg0, pairs = build(engine, cfg, after_close, {"A": make_bars(START, 5)})
    apply(engine, "U2", ["GHOST"], date(2020, 1, 1))
    with session_scope(engine) as s:
        ghost = (find_symbol_row(s, "GHOST").instrument_id, "GHOST")
        issues = run_quality_checks(s, pairs + [ghost], cfg0)
    assert [(r.symbol, r.check, r.severity) for r in issues.itertuples()] == [("GHOST", "no_data", "error")]


# ---- duplicates -----------------------------------------------------------------------------------------
def test_database_refuses_a_second_row_for_the_same_bar(engine, cfg, after_close):
    build(engine, cfg, after_close, {"A": make_bars(START, 5)})
    with pytest.raises(IntegrityError):
        with engine.begin() as c:
            c.execute(text("INSERT INTO price_bar (instrument_id, trade_date, open, high, low, close, volume, source, fetched_at) "
                           "SELECT instrument_id, trade_date, open, high, low, close, volume, source, fetched_at FROM price_bar LIMIT 1"))


def test_vendor_duplicate_rows_are_reported_as_info(engine, cfg, after_close):
    warn = {"BAD": f"BAD: merged 2 same-date rows for {DAYS[20]} (inferred split session)"}
    issues = check(engine, cfg, after_close, None, warn_on=warn)
    row = issues[issues["check"] == "vendor_duplicate"]
    assert list(row.trade_date) == [DAYS[20]] and set(row.severity) == {"info"}


# ---- the catalogue ----------------------------------------------------------------------------------------
def test_every_check_has_a_severity_and_the_summary_orders_errors_first(engine, cfg, after_close):
    assert set(SEVERITY.values()) == {"error", "warn", "info"}
    def defect(b):
        b[30] = replace(b[30], volume=0, high=b[30].low - 5)
        return b
    s = summarize(check(engine, cfg, after_close, defect))
    assert list(s.severity)[0] == "error" and set(s.columns) == {"check", "severity", "count", "symbols"}
