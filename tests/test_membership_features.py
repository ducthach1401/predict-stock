"""ACCEPTANCE: a stock joining the universe mid-period and a stock being removed.

Cross-sectional features and label ranks use only the universe members on that very date; time-series features of a
new member may (legitimately) use history from before it joined, because that data existed at the time."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predict_stock.features.cs  # noqa: F401
import predict_stock.features.labels  # noqa: F401
import predict_stock.features.ts  # noqa: F401
from predict_stock.features.data import Panel, mask_from_intervals
from predict_stock.features.engine import assemble_frame, required_warmup
from predict_stock.features.registry import FeatureSetSpec, LabelSpecSpec
from panel_helpers import random_panel

FIELDS = ("open", "high", "low", "close", "volume")
FSET = FeatureSetSpec.from_dict("t", {"version": 1, "features": [
    {"name": "ret", "version": 1, "params": {"windows": [1, 5]}}, {"name": "rsi", "version": 1, "params": {"window": 14}},
    {"name": "cs_rank", "version": 1, "params": {"inputs": ["ret_5", "rsi_14"], "min_count": 2}}]})
LSPEC = LabelSpecSpec.from_dict("t", {"version": 1, "labels": [
    {"name": "fwd_rank_return", "version": 1, "params": {"horizons": [5], "min_count": 2}},
    {"name": "triple_barrier", "version": 1, "params": {"horizon": 5}}]})
CS_COLS = ["ret_5_csrank", "rsi_14_csrank"]


def without(panel: Panel, iid: int) -> Panel:
    keep = [i for i in panel.instrument_ids if i != iid]
    return Panel(panel.calendar, *(getattr(panel, f)[keep] for f in FIELDS), bench=panel.bench)


def build(panel, mask):
    return assemble_frame(panel, FSET.instantiate(), LSPEC.instantiate(), mask, panel.calendar[0], panel.calendar[-1])


def cell(frame, date, iid, col):
    r = frame[(frame["trade_date"] == date) & (frame["instrument_id"] == iid)]
    return None if r.empty else r[col].iloc[0]


def test_a_stock_that_joins_mid_period_has_no_rows_before_joining():
    panel = random_panel(n_days=260, n_inst=6, seed=21, gaps=0)
    join = 150
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(2, 7)] + [(1, panel.calendar[join], None)])
    frame = build(panel, mask)
    rows1 = frame[frame["instrument_id"] == 1]
    assert rows1["trade_date"].min() == panel.calendar[join] and len(rows1) == 260 - join          # from its valid_from, every session
    first = rows1.iloc[0]
    assert not np.isnan(first["ret_5"]) and not np.isnan(first["rsi_14"])          # time-series features use pre-membership history: that data existed then
    assert not np.isnan(first["ret_5_csrank"])


def test_ranks_before_a_stock_joins_are_exactly_those_of_a_universe_without_it():
    panel = random_panel(n_days=260, n_inst=6, seed=22, gaps=0)
    join = 150
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(2, 7)] + [(1, panel.calendar[join], None)])
    with_new = build(panel, mask)
    p2 = without(panel, 1)
    never_existed = build(p2, mask_from_intervals(p2.calendar, p2.instrument_ids, [(i, p2.calendar[0], None) for i in range(2, 7)]))
    early = lambda f: f[f["trade_date"] < panel.calendar[join]].set_index(["trade_date", "instrument_id"]).sort_index()
    pd.testing.assert_frame_equal(early(with_new)[CS_COLS + ["fwd_rank_5"]], early(never_existed)[CS_COLS + ["fwd_rank_5"]])
    late_a = with_new[with_new["trade_date"] >= panel.calendar[join + 10]].set_index(["trade_date", "instrument_id"]).sort_index()
    late_b = never_existed[never_existed["trade_date"] >= panel.calendar[join + 10]].set_index(["trade_date", "instrument_id"]).sort_index()
    both = late_a.index.intersection(late_b.index)
    assert not np.allclose(late_a.loc[both, "ret_5_csrank"], late_b.loc[both, "ret_5_csrank"], equal_nan=True)   # once it is a member, it counts


def test_a_removed_stock_stops_producing_rows_and_leaves_the_ranks():
    panel = random_panel(n_days=260, n_inst=6, seed=23, gaps=0)
    leave = 200
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in (1, 3, 4, 5, 6)] + [(2, panel.calendar[0], panel.calendar[leave])])
    frame = build(panel, mask)
    rows2 = frame[frame["instrument_id"] == 2]
    assert rows2["trade_date"].max() == panel.calendar[leave - 1]                   # valid_to is exclusive: last row is the day before
    p2 = without(panel, 2)
    m2 = mask_from_intervals(p2.calendar, p2.instrument_ids, [(i, p2.calendar[0], None) for i in (1, 3, 4, 5, 6)])
    never = build(p2, m2)
    key = lambda f, lo=None, hi=None: f[(f["trade_date"] >= (lo or f["trade_date"].min())) & (f["trade_date"] < (hi or f["trade_date"].max() + pd.Timedelta(days=1)))].set_index(["trade_date", "instrument_id"]).sort_index()
    after_a, after_b = key(frame, panel.calendar[leave]), key(never, panel.calendar[leave])
    pd.testing.assert_frame_equal(after_a[CS_COLS + ["fwd_rank_5"]], after_b.loc[after_a.index, CS_COLS + ["fwd_rank_5"]])   # after leaving: as if it never existed
    before_a, before_b = key(frame, None, panel.calendar[leave]), key(never, None, panel.calendar[leave])
    common = before_b.index
    assert not np.allclose(before_a.loc[common, "ret_5_csrank"], before_b.loc[common, "ret_5_csrank"], equal_nan=True)   # while a member it was ranked against


def test_the_last_row_of_a_removed_stock_is_labelled_with_prices_after_its_removal():
    panel = random_panel(n_days=260, n_inst=6, seed=24, gaps=0)
    leave = 200
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in (1, 3, 4, 5, 6)] + [(2, panel.calendar[0], panel.calendar[leave])])
    frame = build(panel, mask)
    last = panel.calendar[leave - 1]
    expected = panel.close[2].iloc[leave - 1 + 5] / panel.close[2].iloc[leave - 1] - 1              # uses closes AFTER the removal date
    assert cell(frame, last, 2, "fwd_ret_5") == pytest.approx(expected)
    assert not np.isnan(cell(frame, last, 2, "fwd_rank_5"))
    assert cell(frame, panel.calendar[leave], 2, "fwd_rank_5") is None                          # no row once removed
    # and on that day the removed stock still takes part in the label ranks of the others (no survivorship in the ranks)
    same_day = frame[frame["trade_date"] == last]
    assert len(same_day) == 6 and same_day["fwd_rank_5"].notna().all()
    assert sorted(same_day["fwd_rank_5"]) == pytest.approx(list(np.arange(1, 7) / 6))


def test_a_stock_that_stops_trading_gets_no_rows_and_unknown_labels():
    panel = random_panel(n_days=260, n_inst=5, seed=25, gaps=0)
    stop = 220                                                                                # delisted: no bars from here on, still "a member"
    for f in FIELDS:
        getattr(panel, f).loc[panel.calendar[stop:], 3] = np.nan
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(1, 6)])
    frame = build(panel, mask)
    rows3 = frame[frame["instrument_id"] == 3]
    assert rows3["trade_date"].max() == panel.calendar[stop - 1]
    assert np.isnan(cell(frame, panel.calendar[stop - 3], 3, "fwd_ret_5")) and np.isnan(cell(frame, panel.calendar[stop - 3], 3, "tb_label"))   # the window runs past its last bar
    assert not np.isnan(cell(frame, panel.calendar[stop - 8], 3, "fwd_ret_5"))
    assert frame[frame["trade_date"] == panel.calendar[stop + 2]]["instrument_id"].tolist() == [1, 2, 4, 5]


def test_a_suspended_session_gives_no_row_even_for_a_member():
    panel = random_panel(n_days=200, n_inst=4, seed=26, gaps=0)
    for f in FIELDS:
        getattr(panel, f).loc[panel.calendar[120], 2] = np.nan
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(1, 5)])
    frame = build(panel, mask)
    assert cell(frame, panel.calendar[120], 2, "ret_1") is None and cell(frame, panel.calendar[121], 2, "ret_1") is not None
    assert (frame[frame["trade_date"] == panel.calendar[120]]["ret_5_csrank"].dropna().sort_values().to_numpy() == np.array([1, 2, 3]) / 3).all()   # ranked among the 3 that traded


def test_a_new_listing_gets_rows_only_after_its_warmup():
    panel = random_panel(n_days=260, n_inst=5, seed=27, gaps=0)
    listed = 100
    for f in FIELDS:
        getattr(panel, f).loc[panel.calendar[:listed], 4] = np.nan
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(1, 6)])
    frame = build(panel, mask)
    warm = required_warmup(FSET.instantiate())
    first = frame[frame["instrument_id"] == 4]["trade_date"].min()
    assert first == panel.calendar[listed + warm - 1]
    assert frame[frame["instrument_id"] == 4]["rsi_14"].notna().all()


def test_membership_intervals_are_half_open():
    cal = pd.bdate_range("2024-01-01", periods=10)
    m = mask_from_intervals(cal, [1, 2, 3], [(1, cal[2], cal[6]), (2, cal[0], None), (1, cal[8], None)])
    assert m[1].tolist() == [False, False, True, True, True, True, False, False, True, True]     # leaves and re-joins
    assert m[2].all() and not m[3].any()
    assert mask_from_intervals(cal, [1], [(99, cal[0], None)])[1].sum() == 0                    # unknown instrument ignored


def test_rows_never_carry_identifiers_as_features():
    panel = random_panel(n_days=120, n_inst=4, seed=28)
    mask = mask_from_intervals(panel.calendar, panel.instrument_ids, [(i, panel.calendar[0], None) for i in range(1, 5)])
    frame = build(panel, mask)
    features = set(FSET.columns()) | set(LSPEC.columns())
    assert {"trade_date", "instrument_id"}.isdisjoint(features)
    assert set(frame.columns) == {"trade_date", "instrument_id"} | features
