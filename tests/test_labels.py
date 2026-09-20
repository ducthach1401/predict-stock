"""Labels: forward-return ranks and triple barrier, checked against hand-worked cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.features import labels as _labels  # noqa: F401
from predict_stock.features.registry import get_label
from panel_helpers import all_members, make_calendar, panel_from_arrays, random_panel

NaN = np.nan


def fwd(panel, mask=None, **params):
    lb = get_label("fwd_rank_return", 1)(min_count=2, **params)
    return lb.compute(panel, all_members(panel) if mask is None else mask)


# ---- forward return rank --------------------------------------------------------------------------
def test_forward_return_and_rank_and_end_date():
    close = np.array([[100, 100, 100], [110, 100, 90], [121, 100, 90], [130, 105, 80], [140, 110, 70]], dtype=float)
    p = panel_from_arrays(close)
    out = fwd(p, horizons=[2])
    r = out["fwd_ret_2"]
    assert r.iloc[0].tolist() == pytest.approx([0.21, 0.0, -0.10])
    assert out["fwd_rank_2"].iloc[0].tolist() == [1.0, 2 / 3, 1 / 3]
    assert r.iloc[3:].isna().all().all() and out["fwd_rank_2"].iloc[3:].isna().all().all()       # the last 2 sessions have no outcome yet
    assert out["fwd_end_2"].iloc[0].tolist() == [p.calendar[2]] * 3 and out["fwd_end_2"].iloc[3:].isna().all().all()


def test_rank_is_among_the_members_of_that_date():
    close = np.array([[100, 100, 100, 100], [110, 100, 90, 300], [120, 100, 80, 400]], dtype=float)
    p = panel_from_arrays(close)
    mask = all_members(p)
    mask.loc[p.calendar[0], 4] = False                          # instrument 4 is not a member on day 0 (a huge winner, but outside)
    out = fwd(p, mask=mask, horizons=[1])
    assert out["fwd_rank_1"].iloc[0, :3].tolist() == [1.0, 2 / 3, 1 / 3] and np.isnan(out["fwd_rank_1"].iloc[0, 3])
    assert out["fwd_ret_1"].iloc[0, 3] == pytest.approx(2.0)     # its return is still computed; it is just not ranked
    # day 1 has all four members: returns 120/110-1, 0, 80/90-1, 400/300-1 -> ranks 0.75, 0.5, 0.25, 1.0
    assert out["fwd_rank_1"].iloc[1].tolist() == [0.75, 0.5, 0.25, 1.0]


def test_a_missing_end_bar_gives_no_label_and_multiple_horizons():
    close = np.array([[100, 100], [101, 101], [NaN, 102], [103, 103], [104, 104], [105, 105]], dtype=float)
    p = panel_from_arrays(close)
    out = fwd(p, horizons=[1, 2])
    assert np.isnan(out["fwd_ret_1"].iloc[1, 0]) and out["fwd_ret_1"].iloc[1, 1] == pytest.approx(102 / 101 - 1)     # no bar on day 2 for #1
    assert np.isnan(out["fwd_ret_2"].iloc[0, 0]) and out["fwd_ret_2"].iloc[0, 1] == pytest.approx(102 / 100 - 1)     # #1 has no bar on day 2
    assert set(out) == {"fwd_ret_1", "fwd_rank_1", "fwd_end_1", "fwd_ret_2", "fwd_rank_2", "fwd_end_2"}


# ---- triple barrier ------------------------------------------------------------------------------------------
def tb_panel(highs, lows, closes=None, opens=None, entry=100.0):
    """One instrument. Day 0 has high-low = 2 around ``entry`` so that ATR(1) = 2; days 1.. are given explicitly."""
    n = len(highs) + 1
    h = np.r_[entry + 1, highs]
    l = np.r_[entry - 1, lows]
    c = np.r_[entry, closes if closes is not None else np.full(len(highs), entry)]
    o = np.r_[entry, opens if opens is not None else np.full(len(highs), entry)]
    return panel_from_arrays(c.reshape(n, 1), h.reshape(n, 1), l.reshape(n, 1), o.reshape(n, 1))


def tb(panel, **params):
    p = {"horizon": 3, "atr_window": 1, "target_mult": 2.0, "stop_mult": 1.0, **params}
    out = get_label("triple_barrier", 1)(**p).compute(panel, all_members(panel))
    return {k: v.iloc[0, 0] for k, v in out.items()}


# entry 100, ATR(day 0) = 2 -> target 104 (2 x ATR), stop 98 (1 x ATR)
def test_target_hit():
    r = tb(tb_panel(highs=[102, 105, 101], lows=[99, 100, 99]))
    assert (r["tb_label"], r["tb_time"]) == (1.0, 2) and r["tb_ret"] == pytest.approx(0.04)      # exits AT the target on day 2
    assert r["tb_end"] == pd.Timestamp("2024-01-03")


def test_stop_hit():
    r = tb(tb_panel(highs=[101, 103, 101], lows=[99, 97, 99]))
    assert (r["tb_label"], r["tb_time"]) == (-1.0, 2) and r["tb_ret"] == pytest.approx(-0.02)


def test_same_session_touching_both_barriers_counts_as_the_stop():
    r = tb(tb_panel(highs=[105, 101, 101], lows=[97, 99, 99]))
    assert (r["tb_label"], r["tb_time"]) == (-1.0, 1)


def test_time_barrier_returns_the_close_of_the_last_session():
    r = tb(tb_panel(highs=[102, 102, 103], lows=[99, 99, 99], closes=[101, 101, 102.5]))
    assert (r["tb_label"], r["tb_time"]) == (0.0, 3) and r["tb_ret"] == pytest.approx(0.025)


def test_gap_through_the_stop_exits_at_the_open_not_at_the_stop():
    r = tb(tb_panel(highs=[101, 96, 96], lows=[99, 93, 93], opens=[100, 95, 95]))     # day 2 opens at 95, below the 98 stop
    assert r["tb_label"] == -1.0 and r["tb_time"] == 2 and r["tb_ret"] == pytest.approx(-0.05)


def test_gap_through_the_target_exits_at_the_open():
    r = tb(tb_panel(highs=[101, 111, 111], lows=[99, 106, 106], opens=[100, 108, 108]))
    assert r["tb_label"] == 1.0 and r["tb_ret"] == pytest.approx(0.08)                # opened at 108 >= target 104


def test_the_barriers_never_use_the_entry_day():
    r = tb(tb_panel(highs=[101, 101, 101], lows=[99, 99, 99]))                          # day 0 itself has high 101 / low 99: no touch
    assert r["tb_label"] == 0.0


def test_a_window_that_does_not_fit_in_the_data_is_unknown():
    p = tb_panel(highs=[102, 102], lows=[99, 99])                                        # 3 sessions in all, horizon 3 needs 4
    assert np.isnan(tb(p)["tb_label"]) and pd.isna(tb(p)["tb_end"])
    p2 = tb_panel(highs=[105, 101], lows=[99, 99])                                       # target hit on day 1 but window incomplete: still unknown
    assert np.isnan(tb(p2)["tb_label"])


def test_suspended_sessions_are_skipped_and_a_missing_final_bar_is_unknown():
    highs = np.array([NaN, 105, 101]); lows = np.array([NaN, 100, 99]); closes = np.array([NaN, 104, 101])
    r = tb(tb_panel(highs, lows, closes=closes))
    assert (r["tb_label"], r["tb_time"]) == (1.0, 2)                                     # day 1 had no bar: skipped
    highs2 = np.array([101, 101, NaN]); lows2 = np.array([99, 99, NaN]); closes2 = np.array([100, 100, NaN])
    assert np.isnan(tb(tb_panel(highs2, lows2, closes=closes2))["tb_label"])             # would be a time-out, but no close on day 3


def test_barrier_distance_scales_with_the_configured_multipliers():
    p = tb_panel(highs=[102, 102, 102], lows=[99, 99, 99])
    assert tb(p, target_mult=1.0)["tb_label"] == 1.0                                    # target 102 is touched
    assert tb(p, target_mult=2.0)["tb_label"] == 0.0
    p2 = tb_panel(highs=[101, 101, 101], lows=[97.5, 99, 99])
    assert tb(p2, stop_mult=1.0)["tb_label"] == -1.0 and tb(p2, stop_mult=2.0)["tb_label"] == 0.0


def test_column_suffix_allows_several_barrier_configs():
    a = get_label("triple_barrier", 1)(suffix="_a")
    b = get_label("triple_barrier", 1)(suffix="_b", horizon=20)
    assert set(a.columns()).isdisjoint(b.columns()) and a.columns()[0] == "tb_label_a" and b.horizon() == 20


def test_labels_are_computed_per_instrument_independently():
    p = random_panel(n_days=200, n_inst=4, seed=3, gaps=0, bench=False)
    full = get_label("triple_barrier", 1)(horizon=5).compute(p, all_members(p))
    single = random_panel(n_days=200, n_inst=4, seed=3, gaps=0, bench=False)
    for c in ("open", "high", "low", "close", "volume"):
        setattr(single, c, getattr(single, c)[[2]])
    alone = get_label("triple_barrier", 1)(horizon=5).compute(single, all_members(single))
    for k in full:
        pd.testing.assert_series_equal(full[k][2], alone[k][2], check_names=False)
