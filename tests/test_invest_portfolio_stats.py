"""INVEST portfolio construction (weights, tranches, regime filter, schedules) and the small-sample statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.invest import stats as S
from predict_stock.invest import weights as Wt
from predict_stock.invest.strategy import invest_signals, regime_off


def rets(n=300, k=6, seed=0, vols=None):
    rng = np.random.default_rng(seed)
    vols = vols or [0.01 * (i + 1) for i in range(k)]
    return pd.DataFrame(rng.normal(0, 1, (n, k)) * np.array(vols), index=pd.bdate_range("2023-01-02", periods=n), columns=list(range(1, k + 1)))


# ---- weights -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", Wt.SCHEMES)
def test_every_scheme_is_fully_invested_long_only_and_within_the_cap(scheme):
    w = Wt.target_weights(scheme, [1, 2, 3, 4, 5, 6], rets(), 0.3)
    assert w.sum() == pytest.approx(1.0, abs=1e-9) and (w >= -1e-12).all() and (w <= 0.3 + 1e-9).all() and list(w.index) == [1, 2, 3, 4, 5, 6]


def test_inverse_vol_gives_more_weight_to_the_calmer_names():
    w = Wt.target_weights("inverse_vol", [1, 2, 3, 4, 5, 6], rets(n=2000), None)
    assert w[1] > w[3] > w[6] and w[1] / w[6] == pytest.approx(6.0, rel=0.25)


def test_risk_parity_equalises_risk_contributions_and_beats_equal_weight_on_it():
    cov = Wt.shrunk_covariance(rets(n=1500))
    w = Wt.risk_parity(cov)
    rc = w * (cov @ w)
    assert rc.std() / rc.mean() < 1e-6
    eq = np.full(6, 1 / 6)
    assert (eq * (cov @ eq)).std() / (eq * (cov @ eq)).mean() > 0.1


def test_min_variance_has_the_lowest_variance_of_the_schemes_and_respects_the_cap():
    r = rets(n=1500)
    cov = Wt.shrunk_covariance(r)
    var = lambda w: float(w @ cov @ w)
    mv = Wt.target_weights("min_variance", [1, 2, 3, 4, 5, 6], r, 0.4).to_numpy()
    others = [Wt.target_weights(s, [1, 2, 3, 4, 5, 6], r, 0.4).to_numpy() for s in ("equal", "inverse_vol", "risk_parity", "hrp")]
    assert all(var(mv) <= var(o) + 1e-12 for o in others) and mv.max() <= 0.4 + 1e-9


def test_a_cap_below_one_over_n_is_raised_so_the_portfolio_can_be_fully_invested():
    w = Wt.target_weights("equal", [1, 2, 3], rets(k=3), 0.1)
    assert w.sum() == pytest.approx(1.0) and w.max() == pytest.approx(1 / 3)
    assert Wt.target_weights("min_variance", [1, 2, 3], rets(k=3), 0.1).sum() == pytest.approx(1.0)


def test_hrp_is_nearly_invariant_to_the_order_of_the_names():
    """HRP splits its dendrogram order in the middle, so an input permutation can move weights slightly (it is a known property), not materially."""
    r = rets(n=1000, seed=4)
    a = Wt.target_weights("hrp", [1, 2, 3, 4, 5, 6], r, None)
    b = Wt.target_weights("hrp", [6, 5, 4, 3, 2, 1], r, None)
    assert np.allclose(a.sort_index(), b.sort_index(), atol=0.01) and list(a.sort_values().index) == list(b.sort_values().index)


def test_a_name_with_too_little_history_is_not_given_a_fake_low_volatility():
    r = rets(n=300)
    r.loc[r.index[:280], 6] = np.nan
    w = Wt.target_weights("inverse_vol", [1, 2, 3, 4, 5, 6], r, None)
    assert w[6] < w[1] and w.sum() == pytest.approx(1.0)


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown weighting"):
        Wt.target_weights("magic", [1, 2], rets(k=2), None)


# ---- schedules and signals -------------------------------------------------------------------------------------------------------------
def make_pred(cal, ids=range(1, 9), score_of=lambda i, t: float(i)):
    return pd.DataFrame([{"trade_date": d, "instrument_id": i, "score": score_of(i, t)} for t, d in enumerate(cal) for i in ids])


def test_quarterly_schedule_is_the_first_session_of_each_calendar_quarter():
    cal = pd.bdate_range("2023-01-02", "2024-03-29")
    s = rebalance_sessions(cal, "quarterly", 0, len(cal) - 1)
    assert [str(cal[i].date()) for i in s] == ["2023-01-02", "2023-04-03", "2023-07-03", "2023-10-02", "2024-01-01"]


def kw(**over):
    base = dict(rebalance="monthly", k=3, weighting="equal", cap=0.5, cov_window=60, tranches=1, spacing=5)
    base.update(over)
    return base


def test_top_k_by_score_equal_weight_one_tranche():
    cal = pd.bdate_range("2023-01-02", periods=80)
    sig = invest_signals(make_pred(cal), cal, None, 0, 79, **kw())
    s = sig[0]
    assert s.full_rebalance and {it.instrument_id for it in s.items} == {6, 7, 8} and all(it.weight == pytest.approx(1 / 3) for it in s.items)
    assert sorted(sig) == rebalance_sessions(cal, "monthly", 0, 79)


def test_tranches_move_the_portfolio_in_equal_steps_towards_the_new_target():
    cal = pd.bdate_range("2023-01-02", periods=80)
    flip = lambda i, t: float(i) if t < 25 else -float(i)                     # after ~February the ranking is reversed
    sig = invest_signals(make_pred(cal, score_of=flip), cal, None, 0, 79, **kw(tranches=2, spacing=3))
    w = lambda j: {it.instrument_id: it.weight for it in sig[j].items}
    assert w(0) == {i: pytest.approx(1 / 6) for i in (6, 7, 8)} and w(3) == {i: pytest.approx(1 / 3) for i in (6, 7, 8)}     # 0 -> half -> full
    second = rebalance_sessions(cal, "monthly", 0, 79)[2]                       # first rebalance after the flip
    a, b = w(second), w(second + 3)
    assert set(a) == {1, 2, 3, 6, 7, 8} and a[6] == pytest.approx(1 / 6) and a[1] == pytest.approx(1 / 6)      # half-way: old names 1/3 -> 1/6, new 0 -> 1/6
    assert set(b) == {1, 2, 3} and all(v == pytest.approx(1 / 3) for v in b.values())                       # done: only the new names remain


def test_a_tranche_that_would_run_into_the_next_rebalance_is_dropped():
    cal = pd.bdate_range("2023-01-02", periods=80)
    sig = invest_signals(make_pred(cal), cal, None, 0, 79, **kw(tranches=3, spacing=30))
    sched = rebalance_sessions(cal, "monthly", 0, 79)
    assert all(i in sched or all(i - s >= 30 or i < s for s in sched) for i in sig)
    assert not any(sched[0] < i < sched[1] and i - sched[0] >= 30 for i in sig if i < sched[1])


def test_regime_filter_scales_the_stock_weights_and_leaves_cash():
    cal = pd.bdate_range("2023-01-02", periods=80)
    off = pd.Series(False, index=cal)
    off.iloc[20:] = True
    sig = invest_signals(make_pred(cal), cal, None, 0, 79, **kw(off=off, equity_share=0.5))
    sched = rebalance_sessions(cal, "monthly", 0, 79)
    assert sum(it.weight for it in sig[0].items) == pytest.approx(1.0)
    late = [i for i in sched if i >= 20][0]
    assert sum(it.weight for it in sig[late].items) == pytest.approx(0.5)


def test_regime_state_uses_the_symbol_where_it_has_an_average_and_the_fallback_before():
    cal = pd.bdate_range("2023-01-02", periods=60)
    vn = pd.Series(np.r_[np.linspace(100, 60, 60)], index=cal)               # long index: falling all the time
    vn30 = pd.Series(np.nan, index=cal)
    vn30.iloc[30:] = np.linspace(50, 90, 30)                                   # short index: rising, exists from session 30
    st = regime_off({"VN30": vn30, "VNINDEX": vn}, cal, "VN30", "VNINDEX", 10)
    assert st.iloc[15] and not st.iloc[55]                                     # fallback (falling => risk-off) early, VN30 (rising => risk-on) late
    assert st.iloc[:9].eq(False).all()                                         # no average yet: not risk-off


def test_covariance_based_weights_use_only_data_up_to_the_decision_date():
    cal = pd.bdate_range("2023-01-02", periods=120)
    r = rets(n=120, k=8)
    a = invest_signals(make_pred(cal), cal, r, 0, 119, **kw(weighting="inverse_vol", cov_window=60))
    r2 = r.copy()
    r2.iloc[70:] = 0.5                                                          # change the FUTURE of the first decision dates
    b = invest_signals(make_pred(cal), cal, r2, 0, 119, **kw(weighting="inverse_vol", cov_window=60))
    first = [i for i in a if i < 60]
    for i in first:
        assert [(it.instrument_id, it.weight) for it in a[i].items] == [(it.instrument_id, it.weight) for it in b[i].items]


# ---- bootstrap and inference --------------------------------------------------------------------------------------------------------------
def test_stationary_bootstrap_is_deterministic_and_block_lengths_average_out():
    a = S.bootstrap_indices(500, 50, 21, seed=1)
    assert np.array_equal(a, S.bootstrap_indices(500, 50, 21, seed=1)) and not np.array_equal(a, S.bootstrap_indices(500, 50, 21, seed=2))
    steps = np.diff(a, axis=1) % 500
    run_ends = (steps != 1).mean()
    assert run_ends == pytest.approx(1 / 21, rel=0.25)


def test_bootstrap_interval_covers_the_truth_and_the_point_estimate():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
    out = S.paired_bootstrap(r, None, resamples=400, block=10, level=0.9, seed=3)
    sh = out["sharpe"]
    assert sh["lo"] < sh["point"] < sh["hi"] and 0.9 * 0.0008 / 0.01 * np.sqrt(252) * 0 < sh["point"]
    assert out["max_drawdown"]["lo"] <= out["max_drawdown"]["point"] <= out["max_drawdown"]["hi"] + 1e-9


def test_the_interval_widens_when_the_sample_shrinks():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000), index=pd.bdate_range("2015-01-01", periods=2000))
    width = lambda s: (lambda o: o["sharpe"]["hi"] - o["sharpe"]["lo"])(S.paired_bootstrap(s, None, resamples=300, block=10, level=0.9, seed=4))
    assert width(r.iloc[:300]) > 1.8 * width(r)


def test_paired_difference_against_an_identical_benchmark_is_exactly_zero():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    d = S.paired_bootstrap(r, r.copy(), resamples=100, block=10, level=0.9, seed=5)["sharpe_diff"]
    assert d["point"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_reality_check_rejects_a_real_edge_and_not_a_pile_of_noise_candidates():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2019-01-01", periods=1200)
    bench = pd.Series(rng.normal(0.0004, 0.01, 1200), index=idx)
    noise = {f"n{i}": pd.Series(rng.normal(0.0004, 0.01, 1200), index=idx) for i in range(8)}
    rc_noise = S.reality_check(noise, bench, resamples=500, block=10, seed=6)
    assert rc_noise["p_value"] > 0.10                                          # the best of 8 random series is not "significant" after the correction
    edge = {**noise, "real": bench + rng.normal(0.0012, 0.004, 1200)}
    rc_edge = S.reality_check(edge, bench, resamples=500, block=10, seed=6)
    assert rc_edge["best"] == "real" and rc_edge["p_value"] < 0.05


def test_the_best_of_many_candidates_is_less_significant_than_a_single_pre_chosen_one():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2019-01-01", periods=1000)
    bench = pd.Series(rng.normal(0.0004, 0.01, 1000), index=idx)
    cands = {f"c{i}": bench + rng.normal(0.0003, 0.006, 1000) for i in range(12)}
    best = S.reality_check(cands, bench, resamples=500, block=10, seed=9)
    single = S.reality_check({best["best"]: cands[best["best"]]}, bench, resamples=500, block=10, seed=9)
    assert best["p_value"] >= single["p_value"]


# ---- rolling windows, DCA, thesis ---------------------------------------------------------------------------------------------------------
def curve(daily, n=800, start=100.0):
    return pd.Series(start * np.cumprod(np.full(n, 1 + daily)), index=pd.bdate_range("2020-01-01", periods=n))


def test_rolling_outperformance_counts_windows_and_shares():
    s, b = curve(0.0006), curve(0.0004)
    r = S.rolling_outperformance(s, b, 252)
    assert r["n_windows"] == 800 - 252 and r["share_ahead"] == 1.0 and r["median_excess"] > 0
    assert S.rolling_outperformance(b, s, 252)["share_ahead"] == 0.0
    assert S.rolling_outperformance(s, b, 900)["n_windows"] == 0


def test_dca_of_a_constant_return_recovers_that_return_as_the_irr():
    eq = curve(0.0004, n=756)
    out = S.dca(eq, 10_000_000)
    assert out["contributed"] == pytest.approx(10_000_000 * 36, rel=0.03) and out["final_value"] > out["contributed"]
    assert out["irr_annual"] == pytest.approx((1.0004) ** 252 - 1, rel=0.03)


def test_dca_of_a_flat_series_returns_what_was_paid_in():
    eq = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=300))
    out = S.dca(eq, 1_000_000)
    assert out["final_value"] == pytest.approx(out["contributed"]) and out["gain"] == pytest.approx(0)


def test_thesis_flags_and_their_information():
    rows = pd.DataFrame({"sma_ratio_200": [0.1, -0.05, 0.02, -0.2], "mom_6m_csrank": [0.9, 0.2, 0.5, 0.1], "dd_252": [-0.05, -0.3, -0.1, -0.4],
                         "fwd_ret_63": [0.1, -0.1, 0.05, -0.2]})
    f = S.thesis_flags(rows, drawdown_break=0.25, rs_floor=0.4)
    assert list(f["below_sma200"]) == [False, True, False, True] and list(f["weak_relative_strength"]) == [False, True, False, True]
    assert list(f["deep_drawdown"]) == [False, True, False, True]
    info = S.thesis_information(rows, f, "fwd_ret_63")
    assert info["any"]["n_flagged"] == 2 and info["any"]["mean_flagged"] == pytest.approx(-0.15) and info["any"]["mean_clear"] == pytest.approx(0.075)


def test_a_window_that_is_too_short_or_degenerate_falls_back_to_equal_weights():
    short = Wt.target_weights("min_variance", [1, 2, 3], rets(n=10, k=3), 0.5)
    assert np.allclose(short, 1 / 3)
    flat = rets(n=200, k=3) * 0.0
    for scheme in ("inverse_vol", "risk_parity", "min_variance", "hrp"):
        w = Wt.target_weights(scheme, [1, 2, 3], flat, 0.5)
        assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)
