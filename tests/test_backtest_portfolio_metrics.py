"""Portfolio construction and performance metrics, against values worked out by hand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.backtest import metrics as M
from predict_stock.backtest.portfolio import apply_cap, build_weights, select_top_k


# ---- portfolio -------------------------------------------------------------------------------------------------
def test_top_k_picks_the_best_scores_and_breaks_ties_by_id():
    s = pd.Series({5: 0.3, 2: 0.9, 9: 0.9, 1: 0.1, 7: np.nan})
    assert select_top_k(s, 2) == [2, 9]                                    # a tie at 0.9 goes to the lower id
    assert select_top_k(s, 3) == [2, 9, 5]
    assert select_top_k(s, 2, ascending=True) == [1, 5]
    assert select_top_k(s, 10) == [2, 9, 5, 1] and select_top_k(s, 0) == [] and select_top_k(pd.Series(dtype=float), 3) == []
    assert select_top_k(s.iloc[::-1], 3) == select_top_k(s, 3)              # order of the input does not matter


def test_equal_weights_sum_to_gross():
    w = build_weights(pd.Series({1: 3, 2: 2, 3: 1, 4: 0}), 3)
    assert w.to_dict() == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3}) and w.sum() == pytest.approx(1.0)
    assert build_weights(pd.Series({1: 3, 2: 2}), 2, gross=0.8).sum() == pytest.approx(0.8)
    assert build_weights(pd.Series(dtype=float), 3).empty


def test_inverse_volatility_weights():
    scores = pd.Series({1: 3, 2: 2, 3: 1})
    vol = pd.Series({1: 0.2, 2: 0.4, 3: 0.1})
    w = build_weights(scores, 3, scheme="inverse_vol", vol=vol)
    inv = 1 / vol
    assert w.to_dict() == pytest.approx((inv / inv.sum()).to_dict()) and w.sum() == pytest.approx(1.0)
    assert w[3] > w[1] > w[2]                                                 # the calmest stock weighs most
    dropped = build_weights(scores, 3, scheme="inverse_vol", vol=pd.Series({1: 0.2, 2: np.nan, 3: 0.0}))
    assert dropped.index.tolist() == [1] and dropped.sum() == pytest.approx(1.0)      # unusable volatilities are left out
    assert build_weights(scores, 3, scheme="inverse_vol", vol=pd.Series({1: np.nan})).to_dict() == pytest.approx({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
    with pytest.raises(ValueError):
        build_weights(scores, 2, scheme="magic")


def test_cap_redistributes_the_excess_and_can_leave_cash():
    w = apply_cap(pd.Series({1: 0.5, 2: 0.3, 3: 0.2}), 0.4)
    assert w[1] == pytest.approx(0.4) and w.sum() == pytest.approx(1.0) and w.max() <= 0.4 + 1e-9
    assert w[2] / w[3] == pytest.approx(0.3 / 0.2)                            # the excess goes to the others in proportion
    cascade = apply_cap(pd.Series({1: 0.6, 2: 0.3, 3: 0.1}), 0.35)             # capping 1 pushes 2 over the cap too
    assert cascade.max() <= 0.35 + 1e-9 and cascade.sum() == pytest.approx(1.0)
    tight = apply_cap(pd.Series({1: 0.5, 2: 0.5}), 0.3)                        # 2 x 30% < 100%: the rest stays in cash
    assert tight.to_dict() == pytest.approx({1: 0.3, 2: 0.3})
    assert apply_cap(pd.Series({1: 0.5, 2: 0.5}), None).sum() == 1.0 and apply_cap(pd.Series(dtype=float), 0.2).empty


def test_cap_applies_after_weighting_in_build_weights():
    w = build_weights(pd.Series({1: 3, 2: 2, 3: 1, 4: 0}), 4, scheme="inverse_vol", vol=pd.Series({1: 0.01, 2: 0.5, 3: 0.5, 4: 0.5}), cap=0.4)
    assert w.max() <= 0.4 + 1e-9 and w.sum() == pytest.approx(1.0) and w[1] == pytest.approx(0.4)


# ---- metrics ------------------------------------------------------------------------------------------------------
def curve(values, start="2020-01-01"):
    return pd.Series(np.asarray(values, float), index=pd.bdate_range(start, periods=len(values)))


def test_cagr_uses_calendar_years():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])                       # 366 days
    e = pd.Series([100.0, 121.0], index=idx)
    assert M.cagr(e) == pytest.approx(1.21 ** (365.25 / 366) - 1)
    assert np.isnan(M.cagr(pd.Series([100.0], index=idx[:1])))


def test_max_drawdown_and_calmar():
    e = curve([100, 120, 90, 110, 130, 104])
    assert M.max_drawdown(e) == pytest.approx(90 / 120 - 1)                    # the deepest fall is from 120 to 90
    assert M.drawdown_series(e).iloc[-1] == pytest.approx(104 / 130 - 1) and M.drawdown_series(e).max() == 0
    m = M.compute_metrics(e)
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.25) and m["max_drawdown"] == pytest.approx(-0.25)
    assert np.isnan(M.compute_metrics(curve([100, 101, 102]))["calmar"])         # no drawdown: undefined, not infinite


def test_sharpe_and_sortino_formulas():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015, -0.005])
    assert M.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert M.sharpe(r, rf_annual=0.05) == pytest.approx((r.mean() - 0.05 / 252) / r.std(ddof=1) * np.sqrt(252))
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())                              # root mean square of the negative returns over ALL days
    assert M.sortino(r) == pytest.approx(r.mean() / dd * np.sqrt(252))
    assert np.isnan(M.sharpe(pd.Series([0.01, 0.01, 0.01]))) and np.isnan(M.sortino(pd.Series([0.01, 0.02])))


def test_compute_metrics_on_a_known_curve():
    e = curve([100.0] + list(100 * 1.001 ** np.arange(1, 253)))
    m = M.compute_metrics(e)
    assert m["total_return"] == pytest.approx(1.001**252 - 1) and m["max_drawdown"] == 0 and m["sessions"] == 253
    assert m["cagr"] == pytest.approx(1.001 ** (252 * 365.25 / (e.index[-1] - e.index[0]).days) - 1)


def test_turnover_costs_and_exposure_are_reported_with_fills():
    e = curve([100, 100, 100, 100, 100])
    fills = pd.DataFrame({"value": [40.0, 40.0], "fee": [0.1, 0.1], "tax": [0.0, 0.04], "slippage_cost": [0.05, 0.05]})
    m = M.compute_metrics(e, fills=fills, exposure=pd.Series([0, 0.4, 0.4, 0.4, 0]))
    years = (e.index[-1] - e.index[0]).days / 365.25
    assert m["turnover_annual"] == pytest.approx(80 / 2 / 100 / years) and m["costs_paid"] == pytest.approx(0.24) and m["slippage_paid"] == pytest.approx(0.10)
    assert m["avg_exposure"] == pytest.approx(0.24)
    assert M.compute_metrics(e, fills=pd.DataFrame(columns=fills.columns))["turnover_annual"] == 0


def test_trade_statistics():
    trips = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -150.0, 0.0], "net_return": [0.10, -0.05, 0.20, -0.15, 0.0], "sessions": [5, 3, 10, 4, 2]})
    t = M.trade_stats(trips)
    assert t["trades"] == 5 and t["win_rate"] == pytest.approx(2 / 5)          # a flat trade is not a win
    assert t["profit_factor"] == pytest.approx(300 / 200) and t["expectancy"] == pytest.approx(0.02)
    assert t["avg_holding_sessions"] == pytest.approx(4.8) and t["avg_win"] == pytest.approx(0.15) and t["avg_loss"] == pytest.approx(-0.10)
    assert M.trade_stats(pd.DataFrame(columns=trips.columns))["trades"] == 0 and np.isnan(M.trade_stats(None)["win_rate"])
    only_wins = M.trade_stats(pd.DataFrame({"pnl": [1.0], "net_return": [0.1], "sessions": [1]}))
    assert only_wins["profit_factor"] == float("inf")


def test_regime_classification_and_metrics():
    up, flat, down = [100 * 1.002 ** i for i in range(126)], [130.0] * 126, [130 * 0.998 ** i for i in range(126)]
    idx = curve(up + flat + down)
    reg = M.classify_regimes(idx, window=126, threshold=0.10)
    assert reg.iloc[0] == "up" and reg.iloc[150] == "sideways" and reg.iloc[-1] == "down" and reg.notna().all()
    assert set(reg) == {"up", "sideways", "down"}
    m = M.by_regime(curve(list(100 * 1.001 ** np.arange(378))), reg)
    assert set(m) == {"up", "sideways", "down"} and all(v["sessions"] > 100 for v in m.values())
    assert m["up"]["annualised_return"] == pytest.approx(1.001 ** 252 - 1, rel=0.02)
    short_tail = M.classify_regimes(curve(list(range(100, 100 + 130))), window=126)     # 4 leftover sessions join the last window
    assert short_tail.nunique() == 1


def test_yearly_metrics_split_by_calendar_year():
    e = curve(list(100 * 1.0005 ** np.arange(520)), start="2019-06-03")
    y = M.by_year(e)
    assert sorted(y) == [2019, 2020, 2021] and all("return" in v and "sharpe" in v for v in y.values())
    assert y[2020]["return"] == pytest.approx(e[e.index.year == 2020].iloc[-1] / e[e.index.year == 2020].iloc[0] - 1)
