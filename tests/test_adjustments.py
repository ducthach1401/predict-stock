"""Corporate-action detection (report only), confirmed actions, versioned factors and the adjusted-price view."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text

from conftest import FakeClient, apply, make_bars
from predict_stock.data.adjustments import (
    AdjustmentError, apply_confirmed, compute_factor, detect_gap_candidates, record_candidates, set_price_basis,
)
from predict_stock.data.ingest import ingest_universe
from predict_stock.db.models import AdjustmentFactor, Alert, CorporateAction, PriceBar
from predict_stock.db.repo import find_symbol_row
from predict_stock.db.session import session_scope
from predict_stock.instruments import change_exchange

START, END = date(2026, 1, 5), date(2026, 9, 18)
N = 40


def scaled(bars, from_index, factor):
    f = Decimal(str(factor))
    return [replace(b, open=b.open * f, high=b.high * f, low=b.low * f, close=b.close * f) if i >= from_index else b for i, b in enumerate(bars)]


def load(engine, cfg, now, series):
    cfg0 = cfg.model_copy(update={"ingest": cfg.ingest.model_copy(update={"benchmark_symbols": []})})
    apply(engine, "U1", list(series), date(2020, 1, 1))
    ingest_universe(engine, FakeClient(series), cfg0, ["U1"], START, END, now=now)
    with session_scope(engine) as s:
        return cfg0, [(find_symbol_row(s, x).instrument_id, x) for x in series]


def write_csv(tmp_path, body, name="ca.csv"):
    p = tmp_path / name
    p.write_text("symbol,ex_date,action_type,ratio,cash_per_share,factor,note\n" + body)
    return p


def count(engine, model):
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(model))


# ---- detection: report only ---------------------------------------------------------------------------
def test_a_split_that_was_not_adjusted_is_detected(engine, cfg, after_close):
    bars = make_bars(START, N)
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(bars, 20, 0.5), "B": make_bars(START, N, base=9000)})
    ex = bars[20].trade_date
    with session_scope(engine) as s:
        (c,) = detect_gap_candidates(s, cfg0, pairs)
    assert (c.symbol, c.trade_date) == ("A", ex) and c.gap < -0.4 and c.implied_factor == pytest.approx(0.5, abs=0.02)


def test_a_gap_over_missing_sessions_is_not_an_event(engine, cfg, after_close):
    """A -20% open is impossible overnight but plausible after several sessions without bars (found on real data:
    GVR 2020-03-17, -16% after five missing sessions, band 15% per session)."""
    bars = make_bars(START, N)
    gap_down = scaled(bars, 20, 0.80)                                          # -20% overnight
    cfg0, pairs = load(engine, cfg, after_close, {"A": gap_down, "B": make_bars(START, N, base=9000), "C": make_bars(START, N, base=7000)})
    with session_scope(engine) as s:
        assert [c.trade_date for c in detect_gap_candidates(s, cfg0, pairs)] == [bars[20].trade_date]        # consecutive: flagged
    # same series, but B and C traded while A had no bars for the five sessions before the gap (6 sessions in all)
    missing = {b.trade_date for b in bars[15:20]}
    with session_scope(engine) as s:
        s.execute(PriceBar.__table__.delete().where(PriceBar.instrument_id == pairs[0][0], PriceBar.trade_date.in_(missing)))
    with session_scope(engine) as s:
        assert detect_gap_candidates(s, cfg0, pairs[:1]) == []                                              # compounded band: explained
        s.execute(PriceBar.__table__.delete().where(PriceBar.instrument_id == pairs[0][0], PriceBar.trade_date.in_({b.trade_date for b in bars[13:15]})))
    with session_scope(engine) as s:                                                                        # (7 missing sessions -> 8: still fine)
        assert detect_gap_candidates(s, cfg0, pairs[:1]) == []


def test_a_huge_gap_over_missing_sessions_is_still_flagged(engine, cfg, after_close):
    bars = make_bars(START, N)
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(bars, 20, 0.5), "B": make_bars(START, N, base=9000), "C": make_bars(START, N, base=7000)})
    with session_scope(engine) as s:
        s.execute(PriceBar.__table__.delete().where(PriceBar.instrument_id == pairs[0][0], PriceBar.trade_date.in_({b.trade_date for b in bars[18:20]})))
    with session_scope(engine) as s:
        (c,) = detect_gap_candidates(s, cfg0, pairs[:1])
        assert c.sessions == 3 and c.gap < -0.4                                                             # -50% needs more than 3 x 15%


def test_clean_and_ordinary_series_have_no_candidates(engine, cfg, after_close):
    cfg0, pairs = load(engine, cfg, after_close, {"A": make_bars(START, N), "B": make_bars(START, N, base=9000)})
    with session_scope(engine) as s:
        assert detect_gap_candidates(s, cfg0, pairs) == []


def test_the_band_depends_on_the_known_exchange(engine, cfg, after_close):
    bars = make_bars(START, N)
    series = {"A": scaled(bars, 20, 0.88)}                                    # a 12% overnight gap
    cfg0, pairs = load(engine, cfg, after_close, series)
    with session_scope(engine) as s:
        assert detect_gap_candidates(s, cfg0, pairs) == []                     # exchange unknown: widest band (15%) + 1% tolerance
        change_exchange(s, pairs[0][0], "HOSE", date(2020, 6, 1))
    with session_scope(engine) as s:
        (c,) = detect_gap_candidates(s, cfg0, pairs)                           # HOSE: ±7% + 1% tolerance
        assert c.band == 0.07


def test_candidates_are_recorded_once_alerted_and_never_applied(engine, cfg, after_close):
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(make_bars(START, N), 20, 0.5)})
    with session_scope(engine) as s:
        out = record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, run_id=None)
    assert out == {"candidates": 1, "new": 1, "stale": 0}
    with session_scope(engine) as s:
        again = record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, run_id=None)
        assert again["new"] == 0
        ca = s.scalars(select(CorporateAction)).one()
        assert ca.status == "candidate" and ca.action_type == "unknown_gap" and ca.details["implied_factor"] == pytest.approx(0.5, abs=0.02)
        alert = s.scalars(select(Alert)).one()
        assert alert.severity == "warn" and alert.category == "corporate_action_candidate" and "possible unadjusted" in alert.message
    assert count(engine, CorporateAction) == 1 and count(engine, Alert) == 1                  # idempotent
    assert count(engine, AdjustmentFactor) == 0                                               # nothing was adjusted silently
    with session_scope(engine) as s:                                                          # and the prices are exactly as ingested
        raw = {str(d): Decimal(c) for d, c in s.execute(text("SELECT trade_date, close FROM price_bar")).all()}
        view = {str(d): Decimal(c) for d, c in s.execute(text("SELECT trade_date, close FROM v_price_adjusted")).all()}
    assert raw == view


def test_a_candidate_that_disappears_becomes_stale(engine, cfg, after_close):
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(make_bars(START, N), 20, 0.5)})
    with session_scope(engine) as s:
        record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, None)
        record_candidates(s, [], pairs, None)                                                # the vendor fixed its data
        assert s.scalars(select(CorporateAction)).one().status == "stale"
        record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, None)              # ...and it came back
        assert s.scalars(select(CorporateAction)).one().status == "candidate"


def test_alert_is_acknowledged_when_its_candidate_goes_stale(engine, cfg, after_close):
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(make_bars(START, N), 20, 0.5)})
    with session_scope(engine) as s:
        record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, None)
        assert s.scalars(select(Alert)).one().acknowledged_at is None
        record_candidates(s, [], pairs, None)
        assert s.scalars(select(Alert)).one().acknowledged_at is not None


# ---- factors ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw,expected", [
    (dict(action_type="stock_dividend", ratio=Decimal("0.2")), Decimal("0.833333333333")),
    (dict(action_type="bonus", ratio=Decimal("1")), Decimal("0.5")),
    (dict(action_type="split", ratio=Decimal("2")), Decimal("0.333333333333")),
    (dict(action_type="cash_dividend", cash=Decimal("2000"), prev_close=Decimal("50000")), Decimal("0.96")),
    (dict(action_type="rights", factor=Decimal("0.9")), Decimal("0.9")),
    (dict(action_type="stock_dividend", ratio=Decimal("0.2"), factor=Decimal("0.8")), Decimal("0.8")),      # explicit wins
])
def test_compute_factor(kw, expected):
    args = dict(action_type=None, ratio=None, cash=None, factor=None, prev_close=None) | kw
    assert compute_factor(**args) == expected


@pytest.mark.parametrize("kw,msg", [
    (dict(action_type="rights"), "cannot compute"),
    (dict(action_type="cash_dividend", cash=Decimal("5")), "previous close"),
    (dict(action_type="cash_dividend", cash=Decimal("60000"), prev_close=Decimal("50000")), "previous close"),
    (dict(action_type="stock_dividend", ratio=Decimal("-1")), "positive"),
    (dict(action_type="other", factor=Decimal("1")), "not a valid"),
    (dict(action_type="other", factor=Decimal("-0.5")), "not a valid"),
])
def test_compute_factor_errors(kw, msg):
    args = dict(action_type=None, ratio=None, cash=None, factor=None, prev_close=None) | kw
    with pytest.raises(AdjustmentError, match=msg):
        compute_factor(**args)


# ---- confirmed actions -> versioned factors -> adjusted view ------------------------------------------------
def test_confirmed_action_creates_versioned_factors_and_is_idempotent(engine, cfg, after_close, tmp_path):
    bars = make_bars(START, N)
    load(engine, cfg, after_close, {"A": bars})
    ex1, ex2 = bars[10].trade_date, bars[25].trade_date
    f1 = write_csv(tmp_path, f"A,{ex1},stock_dividend,0.25,,,20% bonus shares (test)\n")
    with session_scope(engine) as s:
        out = apply_confirmed(s, f1)
        assert out["actions"] == 1 and len(out["new_versions"]) == 1 and out["new_versions"][0][1] == 1
    with session_scope(engine) as s:
        assert apply_confirmed(s, f1)["new_versions"] == []                                  # same file again: no new version
        ca = s.scalars(select(CorporateAction)).one()
        assert ca.status == "confirmed" and ca.ratio == Decimal("0.25") and Decimal(ca.details["factor"]) == Decimal("0.8")
    f2 = write_csv(tmp_path, f"A,{ex1},stock_dividend,0.25,,,20% bonus shares (test)\nA,{ex2},other,,,0.9,rights (test)\n", "ca2.csv")
    with session_scope(engine) as s:
        assert apply_confirmed(s, f2)["new_versions"][0][1] == 2                              # a second event -> version 2 holds both
    with session_scope(engine) as s:
        v = {ver: sorted((str(f.effective_date), Decimal(f.factor)) for f in s.scalars(select(AdjustmentFactor).where(AdjustmentFactor.version == ver)))
             for ver in (1, 2)}
        assert v[1] == [(str(ex1), Decimal("0.8"))]
        assert v[2] == [(str(ex1), Decimal("0.8")), (str(ex2), Decimal("0.9"))]
        assert s.scalar(select(func.count()).select_from(CorporateAction)) == 2


def test_confirming_an_event_rejects_the_gap_candidate_it_explains(engine, cfg, after_close, tmp_path):
    bars = make_bars(START, N)
    cfg0, pairs = load(engine, cfg, after_close, {"A": scaled(bars, 20, 0.5)})
    ex = bars[20].trade_date
    with session_scope(engine) as s:
        record_candidates(s, detect_gap_candidates(s, cfg0, pairs), pairs, None)
        apply_confirmed(s, write_csv(tmp_path, f"A,{ex},split,1,,,2-for-1 (test)\n"))
    with session_scope(engine) as s:
        rows = {(c.action_type, c.status) for c in s.scalars(select(CorporateAction))}
        assert rows == {("unknown_gap", "rejected"), ("split", "confirmed")}


def test_cash_dividend_uses_the_previous_close(engine, cfg, after_close, tmp_path):
    bars = make_bars(START, N)
    load(engine, cfg, after_close, {"A": bars})
    ex = bars[12].trade_date
    prev = bars[11].close
    with session_scope(engine) as s:
        apply_confirmed(s, write_csv(tmp_path, f"A,{ex},cash_dividend,,1000,,cash (test)\n"))
        f = s.scalars(select(AdjustmentFactor)).one()
        assert Decimal(f.factor) == ((prev - 1000) / prev).quantize(Decimal("0.000000000001"))


def view_closes(engine):
    with engine.connect() as c:
        return {d: (Decimal(p), Decimal(f)) for d, p, f in c.execute(text("SELECT trade_date, close, adj_factor FROM v_price_adjusted"))}


def test_adjusted_view_applies_factors_only_to_raw_bars(engine, cfg, after_close, tmp_path):
    bars = make_bars(START, N)
    load(engine, cfg, after_close, {"A": bars})
    ex = bars[20].trade_date
    with session_scope(engine) as s:
        apply_confirmed(s, write_csv(tmp_path, f"A,{ex},split,1,,,test\n"))
    before = view_closes(engine)
    assert all(f == 1 for _, f in before.values())                                          # vendor_adjusted bars: never re-adjusted
    with session_scope(engine) as s:
        assert set_price_basis(s, "A", "raw") == N
        assert set_price_basis(s, "A", "raw") == 0                                          # idempotent
    after = view_closes(engine)
    for b in bars:
        p, f = after[b.trade_date]
        if b.trade_date < ex:
            assert f == Decimal("0.5") and p == (b.close * Decimal("0.5")).quantize(Decimal("0.0001"))
        else:
            assert f == 1 and p == b.close                                                  # the ex-date itself and later: untouched
    with session_scope(engine) as s:
        set_price_basis(s, "A", "vendor_adjusted")
    assert view_closes(engine) == before


def test_new_bars_of_a_raw_instrument_stay_raw(engine, cfg, after_close):
    bars = make_bars(START, 30)
    cfg0, pairs = load(engine, cfg, after_close, {"A": bars[:20]})
    with session_scope(engine) as s:
        set_price_basis(s, "A", "raw")
    ingest_universe(engine, FakeClient({"A": bars}), cfg0, ["U1"], START, END, now=after_close)
    with session_scope(engine) as s:
        assert {b.price_basis for b in s.scalars(select(PriceBar))} == {"raw"} and s.scalar(select(func.count()).select_from(PriceBar)) == 30


@pytest.mark.parametrize("body,msg", [
    ("NOPE,2026-02-02,split,1,,,x\n", "unknown symbol"),
    ("A,not-a-date,split,1,,,x\n", "bad ex_date"),
    ("A,2026-02-02,dividend,1,,,x\n", "action_type must be"),
    ("A,2026-02-02,rights,,,,x\n", "cannot compute"),
    ("A,2026-02-02,split,abc,,,x\n", "bad ex_date"),
])
def test_apply_confirmed_validation(engine, cfg, after_close, tmp_path, body, msg):
    load(engine, cfg, after_close, {"A": make_bars(START, N)})
    with session_scope(engine) as s:
        with pytest.raises(AdjustmentError, match=msg):
            apply_confirmed(s, write_csv(tmp_path, body))


def test_apply_confirmed_empty_file_and_bad_basis(engine, cfg, after_close, tmp_path):
    load(engine, cfg, after_close, {"A": make_bars(START, N)})
    with session_scope(engine) as s:
        with pytest.raises(AdjustmentError, match="no data rows"):
            apply_confirmed(s, write_csv(tmp_path, ""))
        with pytest.raises(AdjustmentError, match="basis must be"):
            set_price_basis(s, "A", "adjusted")
        with pytest.raises(AdjustmentError, match="unknown symbol"):
            set_price_basis(s, "NOPE", "raw")
