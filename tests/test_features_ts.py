"""Known-value tests for every time-series feature (expected values worked out independently of the code)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import predict_stock.features.cs  # noqa: F401
import predict_stock.features.ts  # noqa: F401
from predict_stock.features.registry import get_feature
from panel_helpers import bars_from_close, make_calendar


def feat(name, **params):
    return get_feature(name, 1)(**params)


def col(name, bars, bench=None, **params):
    f = feat(name, **params)
    out = f.compute(bars, bench)
    assert list(out.columns) == f.columns()
    return out


# ---- returns --------------------------------------------------------------------------------------
def test_returns():
    bars = bars_from_close([100, 110, 121, 133.1])
    out = col("ret", bars, windows=[1, 2])
    assert np.isnan(out["ret_1"].iloc[0]) and out["ret_1"].iloc[1:].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert out["ret_2"].iloc[2] == pytest.approx(0.21) and np.isnan(out["ret_2"].iloc[1])


# ---- RSI ----------------------------------------------------------------------------------------------
def test_rsi_extremes_and_midpoint():
    up = col("rsi", bars_from_close(np.arange(100, 140.0)), window=14)["rsi_14"]
    assert up.iloc[-1] == pytest.approx(100.0) and up.iloc[:13].isna().all()
    down = col("rsi", bars_from_close(np.arange(140, 100.0, -1)), window=14)["rsi_14"]
    assert down.iloc[-1] == pytest.approx(0.0, abs=1e-9)
    flat = col("rsi", bars_from_close([100.0] * 40), window=14)["rsi_14"]
    assert flat.iloc[-1] == pytest.approx(50.0)
    zigzag = col("rsi", bars_from_close([100 + (i % 2) for i in range(80)]), window=14)["rsi_14"]
    assert 45 < zigzag.iloc[-1] < 55


def test_rsi_matches_wilder_reference():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 120))
    got = col("rsi", bars_from_close(c), window=14)["rsi_14"]
    d = np.diff(c)
    gain, loss = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = gain[:14].mean(), loss[:14].mean()                       # Wilder seed = simple mean of the first n changes
    # pandas' EWM(alpha=1/n, adjust=False) seeds with the first value instead: both converge, so compare late values
    for i in range(14, len(d)):
        ag, al = (ag * 13 + gain[i]) / 14, (al * 13 + loss[i]) / 14
    ref = 100 - 100 / (1 + ag / al)
    assert got.iloc[-1] == pytest.approx(ref, abs=0.6)


# ---- MACD ----------------------------------------------------------------------------------------------
def test_macd_flat_is_zero_and_uptrend_positive_and_matches_ewm():
    assert col("macd", bars_from_close([100.0] * 60))["macd_pct"].dropna().abs().max() == pytest.approx(0.0, abs=1e-12)
    c = np.linspace(100, 160, 90)
    out = col("macd", bars_from_close(c))
    assert (out["macd_pct"].dropna() > 0).all()
    s = pd.Series(c)
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sig = macd.iloc[25:].ewm(span=9, adjust=False).mean()            # the signal line starts when the MACD line is defined (26th bar)
    assert out["macd_pct"].iloc[-1] == pytest.approx(macd.iloc[-1] / c[-1])
    assert out["macd_hist_pct"].iloc[-1] == pytest.approx((macd - sig).iloc[-1] / c[-1])
    assert out["macd_pct"].iloc[:24].isna().all()                     # slow=26 -> defined from the 26th observation


# ---- ATR ---------------------------------------------------------------------------------------------------
def test_atr_true_range_including_gaps():
    n = 20
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=make_calendar(n))
    out = col("atr", bars, window=5)["atr_pct_5"]
    assert out.iloc[-1] == pytest.approx(0.02)                          # TR = 2 every day
    gap = bars.copy()
    gap.iloc[10, gap.columns.get_loc("high")], gap.iloc[10, gap.columns.get_loc("low")] = 110.0, 108.0   # gap up: TR = |110-100| = 10
    g = col("atr", gap, window=1)["atr_pct_1"]                          # n=1 -> ATR is the day's own TR
    assert g.iloc[10] == pytest.approx(10 / gap["close"].iloc[10]) and g.iloc[9] == pytest.approx(0.02)


# ---- Bollinger %B ----------------------------------------------------------------------------------------------
def test_bollinger_pctb():
    c = np.r_[np.full(19, 100.0), 100.0]
    assert np.isnan(col("bollinger_pctb", bars_from_close(c), window=20)["bb_pctb_20"].iloc[-1])   # zero width: undefined, not inf
    rng = np.random.default_rng(3)
    x = 100 + rng.normal(0, 2, 60)
    out = col("bollinger_pctb", bars_from_close(x), window=20, k=2.0)["bb_pctb_20"]
    w = x[-20:]
    mid, sd = w.mean(), w.std(ddof=1)
    assert out.iloc[-1] == pytest.approx((x[-1] - (mid - 2 * sd)) / (4 * sd))
    spike = np.r_[np.full(19, 100.0) + np.arange(19) % 2, 140.0]
    assert col("bollinger_pctb", bars_from_close(spike), window=20)["bb_pctb_20"].iloc[-1] > 1     # above the upper band


# ---- Donchian ----------------------------------------------------------------------------------------------------
def test_donchian_uses_the_prior_window_only():
    bars = bars_from_close(np.r_[np.full(25, 100.0), 130.0])
    out = col("donchian", bars, window=20)
    prior_high = bars["high"].iloc[-21:-1].max()
    assert out["don_hi_dist_20"].iloc[-1] == pytest.approx(130.0 / prior_high - 1) and out["don_hi_dist_20"].iloc[-1] > 0.25   # broke out
    assert out["don_hi_dist_20"].iloc[-2] < 0                                        # the day before, close was inside the channel
    assert out["don_lo_dist_20"].iloc[-1] == pytest.approx(130.0 / bars["low"].iloc[-21:-1].min() - 1)
    assert out["don_hi_dist_20"].iloc[:20].isna().all()


# ---- z-score, gap, volume -------------------------------------------------------------------------------------------
def test_zscore():
    rng = np.random.default_rng(4)
    x = 50 + rng.normal(0, 1, 50)
    z = col("zscore", bars_from_close(x), window=20)["zscore_20"].iloc[-1]
    w = x[-20:]
    assert z == pytest.approx((x[-1] - w.mean()) / w.std(ddof=1))
    assert np.isnan(col("zscore", bars_from_close([10.0] * 30), window=20)["zscore_20"].iloc[-1])


def test_gap_uses_open_and_is_nan_when_the_open_was_repaired():
    bars = bars_from_close([100.0, 102.0, 101.0])
    bars.loc[bars.index[1], "open"] = 105.0                  # opened +5% above the previous close
    bars.loc[bars.index[2], "open"] = np.nan                 # open declared unreliable by the repair rule
    g = col("gap", bars)["gap_1"]
    assert np.isnan(g.iloc[0]) and g.iloc[1] == pytest.approx(0.05) and np.isnan(g.iloc[2])


def test_volume_spike_excludes_today_from_the_baseline():
    v = np.r_[np.full(20, 100.0), 300.0, 100.0]
    out = col("volume_spike", bars_from_close(np.full(22, 10.0), volume=v), window=20)["vol_spike_20"]
    assert out.iloc[20] == pytest.approx(3.0)                # 300 / median(prior 20 = 100)
    assert out.iloc[21] == pytest.approx(1.0)                # the spike is in the baseline, but the median ignores one outlier
    zero = col("volume_spike", bars_from_close(np.full(22, 10.0), volume=np.zeros(22)), window=20)["vol_spike_20"]
    assert zero.isna().all()                                 # zero baseline: undefined, not inf


# ---- relative strength -----------------------------------------------------------------------------------------------
def test_relative_strength_vs_benchmark():
    cal = make_calendar(30)
    stock = bars_from_close(100 * 1.02 ** np.arange(30), index=cal)
    bench = pd.Series(1000 * 1.01 ** np.arange(30), index=cal)
    out = col("rel_strength", stock, bench, windows=[5])["rs_5"]
    assert out.iloc[-1] == pytest.approx(1.02**5 - 1.01**5)
    assert col("rel_strength", stock, None, windows=[5])["rs_5"].isna().all()
    late = bench.copy()
    late.iloc[:20] = np.nan                                  # the benchmark starts late: no history -> NaN, no invention
    r = col("rel_strength", stock, late, windows=[5])["rs_5"]
    assert r.iloc[:25].isna().all() and r.iloc[-1] == pytest.approx(1.02**5 - 1.01**5)


# ---- INVEST ---------------------------------------------------------------------------------------------------------------
def test_momentum_skips_the_most_recent_month():
    c = np.arange(1.0, 401.0)
    out = col("momentum_skip", bars_from_close(c))
    i = 399
    for m, name in ((3, "mom_3m"), (6, "mom_6m"), (12, "mom_12m")):
        assert out[name].iloc[i] == pytest.approx(c[i - 21] / c[i - 21 - m * 21] - 1)
    assert out["mom_12m"].iloc[:21 + 252].isna().all() and not np.isnan(out["mom_12m"].iloc[21 + 252])
    # the last month must NOT matter: a crash in the final 21 sessions leaves momentum unchanged
    crash = c.copy(); crash[-20:] *= 0.5
    assert col("momentum_skip", bars_from_close(crash))["mom_6m"].iloc[-21] == pytest.approx(out["mom_6m"].iloc[-21])
    assert col("momentum_skip", bars_from_close(crash))["mom_6m"].iloc[-1] == pytest.approx(crash[-22] / crash[-22 - 126] - 1)


def test_volatility_annualised():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.02, 200)
    c = 100 * np.exp(np.cumsum(r))
    out = col("volatility", bars_from_close(c), windows=[63])["vol_63"].iloc[-1]
    simple = pd.Series(c).pct_change().iloc[-63:]
    assert out == pytest.approx(simple.std(ddof=1) * np.sqrt(252))
    assert col("volatility", bars_from_close([100.0] * 100), windows=[63])["vol_63"].iloc[-1] == pytest.approx(0.0)


def test_downside_beta_is_measured_on_down_days_only():
    rng = np.random.default_rng(6)
    n = 300
    rb = rng.normal(0.0, 0.01, n)
    rs = np.where(rb < 0, 1.5 * rb, rng.normal(0.0, 0.03, n))       # exactly 1.5x on the benchmark's down days, unrelated otherwise
    cal = make_calendar(n)
    bench = pd.Series(1000 * np.cumprod(1 + rb), index=cal)
    stock = bars_from_close(100 * np.cumprod(1 + rs), index=cal)
    out = col("downside_beta", stock, bench, window=126, min_obs=20)["dbeta_126"]
    assert out.iloc[-1] == pytest.approx(1.5, abs=1e-6)
    ordinary_beta = np.cov(rs[-126:], rb[-126:])[0, 1] / np.var(rb[-126:], ddof=1)
    assert abs(ordinary_beta - 1.5) > 0.1                            # so it really is not the ordinary beta
    assert col("downside_beta", stock, None)["dbeta_126"].isna().all()
    few = pd.Series(np.linspace(1000, 1100, n), index=cal)          # a benchmark that never falls: no down days
    assert col("downside_beta", stock, few, window=126, min_obs=20)["dbeta_126"].isna().all()


def test_drawdown_and_sma_ratio():
    c = np.r_[np.linspace(100, 200, 260), np.linspace(200, 150, 20)]
    dd = col("drawdown", bars_from_close(c), window=252)["dd_252"]
    assert dd.iloc[-1] == pytest.approx(150 / c[-252:].max() - 1) and dd.iloc[:251].isna().all()
    assert dd.iloc[259] == pytest.approx(0.0)                        # at the peak
    sma = col("sma_ratio", bars_from_close(c), window=200)["sma_ratio_200"]
    assert sma.iloc[-1] == pytest.approx(c[-1] / c[-200:].mean() - 1)


def test_liquidity_value_and_zero_share():
    n = 80
    vol = np.full(n, 1000.0); vol[-6:] = 0.0
    bars = bars_from_close(np.full(n, 50.0), volume=vol)
    out = col("liquidity", bars, window=60)
    assert out["liq_zero_share_60"].iloc[-1] == pytest.approx(6 / 60)
    assert out["liq_logvalue_60"].iloc[-1] == pytest.approx(np.log(50.0 * 1000.0))       # median ignores the six zero days
    dead = col("liquidity", bars_from_close(np.full(n, 50.0), volume=np.zeros(n)), window=60)
    assert dead["liq_logvalue_60"].isna().all() and dead["liq_zero_share_60"].iloc[-1] == pytest.approx(1.0)


def liq2(bars, **params):
    f = get_feature("liquidity", 2)(**params)
    out = f.compute(bars, None)
    assert list(out.columns) == f.columns()
    return out


def test_scale_free_liquidity_measures_the_trend_of_traded_value():
    n = 400
    vol = np.r_[np.full(n - 60, 1000.0), np.full(60, 3000.0)]                      # traded value triples over the last 60 sessions
    bars = bars_from_close(np.full(n, 50.0), volume=vol)
    out = liq2(bars, short=60, long=252)
    assert out["liq_trend_60_252"].iloc[-1] == pytest.approx(np.log(3000 / 1000))
    assert out["liq_trend_60_252"].iloc[:251].isna().all() and out["liq_zero_share_60"].iloc[-1] == 0.0
    dead = liq2(bars_from_close(np.full(n, 50.0), volume=np.zeros(n)))
    assert dead["liq_trend_60_252"].isna().all() and dead["liq_zero_share_60"].iloc[-1] == 1.0


def test_scale_free_liquidity_ignores_the_price_level_but_the_old_one_does_not():
    """The point of v2: back-adjustment rescales OLD prices by a factor that depends on FUTURE events."""
    rng = np.random.default_rng(12)
    n = 500
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    vol = rng.integers(500, 5000, n).astype(float)
    plain = bars_from_close(close, volume=vol)
    adj = bars_from_close(close * np.where(np.arange(n) < 300, 0.6, 1.0), volume=vol)      # a later event rescales the first 300 sessions
    t = 299                                                                                 # a date before the event; every window is fully before it
    assert liq2(plain)["liq_trend_60_252"].iloc[t] == pytest.approx(liq2(adj)["liq_trend_60_252"].iloc[t])          # unchanged
    v1 = lambda b: get_feature("liquidity", 1)().compute(b, None)["liq_logvalue_60"].iloc[t]
    assert v1(adj) - v1(plain) == pytest.approx(np.log(0.6))                                # the old level feature moved: future information


# ---- generic properties ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,params", [("ret", {}), ("rsi", {}), ("macd", {}), ("atr", {}), ("bollinger_pctb", {}), ("donchian", {}),
                                        ("zscore", {}), ("gap", {}), ("volume_spike", {}), ("rel_strength", {}), ("momentum_skip", {}),
                                        ("volatility", {}), ("downside_beta", {}), ("drawdown", {}), ("sma_ratio", {}), ("liquidity", {})])
def test_output_shape_index_and_warmup(name, params):
    rng = np.random.default_rng(9)
    n = 400
    cal = make_calendar(n)
    bars = bars_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=cal, volume=rng.integers(1, 1000, n).astype(float))
    bench = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.008, n))), index=cal)
    f = feat(name, **params)
    out = f.compute(bars, bench)
    assert out.index.equals(bars.index) and list(out.columns) == f.columns()
    assert out.iloc[-1].notna().all()                                # defined at the end of a long series
    assert np.isfinite(out.iloc[-1].to_numpy(float)).all()
    warm = f.warmup()
    assert 0 <= warm < n
    # warmup() is the history that makes EVERY output column defined: past it nothing may still be NaN
    assert out.iloc[warm:].notna().all().all(), (name, warm, out.iloc[warm:].isna().sum().to_dict())
    if warm > 1 and name not in ("gap", "downside_beta"):
        assert out.iloc[0].isna().all()                              # and the very first row cannot be defined yet


def test_features_do_not_mutate_their_input():
    rng = np.random.default_rng(10)
    bars = bars_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    before = bars.copy()
    for name in ("ret", "rsi", "macd", "atr", "bollinger_pctb", "donchian", "zscore", "gap", "volume_spike", "momentum_skip", "volatility", "drawdown", "sma_ratio", "liquidity"):
        feat(name).compute(bars, None)
    pd.testing.assert_frame_equal(bars, before)


def test_liquidity_v2_output_shape_and_warmup():
    rng = np.random.default_rng(13)
    n = 400
    bars = bars_from_close(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), volume=rng.integers(1, 1000, n).astype(float))
    f = get_feature("liquidity", 2)()
    out = f.compute(bars, None)
    assert out.index.equals(bars.index) and out.iloc[f.warmup():].notna().all().all() and f.warmup() == 252
    with pytest.raises(Exception, match="short must be shorter"):
        get_feature("liquidity", 2)(short=300, long=100)
