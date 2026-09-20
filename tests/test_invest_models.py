"""The INVEST candidates and their walk-forward, on synthetic data where the truth is known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from predict_stock.config import InvestConfig
from predict_stock.invest import walkforward as W
from predict_stock.invest.models import FactorModel, InvestModelError, fit_linear, fit_lgbm, fit_scenario, load_candidate
from predict_stock.swing.metrics import daily_rank_ic, ic_summary
from tests.invest_helpers import synthetic_invest


def cfg_small(**kw):
    base = dict(n_estimators=60, early_stopping_rounds=8, lgbm={"learning_rate": 0.1, "num_leaves": 4, "max_depth": 2, "min_child_samples": 60},
                folds={"train_min_sessions": 350, "test_sessions": 100, "step_sessions": 100, "val_sessions": 80}, ridge_alphas=[1.0, 100.0], enet_alphas=[0.001, 0.01],
                enet_l1_ratios=[0.5])
    base.update(kw)
    return InvestConfig(**base)


@pytest.fixture(scope="module")
def frame():
    return synthetic_invest(n_days=900, seed=3)


@pytest.fixture(scope="module")
def cfg():
    return cfg_small()


@pytest.fixture(scope="module")
def preset(cfg):
    return cfg.presets["b1"]


@pytest.fixture(scope="module")
def prepared(frame, preset):
    return W.prepare(frame, preset)


# ---- factor score ----------------------------------------------------------------------------------------------------------------------
def test_factor_score_is_the_mean_of_signed_ranks_and_needs_no_fit(frame, cfg):
    m = FactorModel([(c, int(s)) for c, s in cfg.factors])
    rows = frame.head(200)
    manual = (rows["mom_6m_csrank"] + rows["mom_12m_csrank"] + (1 - rows["vol_126_csrank"]) + rows["sma_ratio_200_csrank"] + rows["dd_252_csrank"]) / 5
    assert np.allclose(m.predict(rows), manual)


def test_factor_contributions_add_up_to_the_score_minus_one_half(frame, cfg):
    m = FactorModel([(c, int(s)) for c, s in cfg.factors])
    rows = frame.head(100)
    contrib, names = m.contributions(rows)
    assert np.allclose(contrib.sum(axis=1) + 0.5, m.predict(rows)) and names[2] == "-vol_126_csrank"


def test_factor_score_is_missing_when_too_many_factors_are_missing(frame, cfg):
    m = FactorModel([(c, int(s)) for c, s in cfg.factors])
    rows = frame.head(3).copy()
    rows.loc[rows.index[0], ["mom_6m_csrank", "mom_12m_csrank", "vol_126_csrank"]] = np.nan
    rows.loc[rows.index[1], ["mom_6m_csrank"]] = np.nan
    s = m.predict(rows)
    assert np.isnan(s[0]) and np.isfinite(s[1]) and np.isfinite(s[2])


# ---- linear models ---------------------------------------------------------------------------------------------------------------------
def test_ridge_finds_the_planted_signal_and_shrinks_with_alpha(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    small = fit_linear("ridge", d.fit, "fwd_rank_63", {"alpha": 1.0})
    big = fit_linear("ridge", d.fit, "fwd_rank_63", {"alpha": 1e6})
    assert np.abs(big.coef).max() < 0.05 * np.abs(small.coef).max()
    coef = dict(zip(small.features, small.coef))
    assert coef["mom_6m"] + coef["mom_6m_csrank"] > 0 and coef["vol_126"] + coef["vol_126_csrank"] < 0          # the planted signs
    assert "liq_zero_share_60" not in small.features                                                                # constants are dropped
    rows = d.val
    assert spearmanr(small.predict(rows), rows["fwd_ret_63"], nan_policy="omit")[0] > 0.1


def test_elasticnet_is_sparse_at_a_strong_penalty(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    m = fit_linear("elasticnet", d.fit, "fwd_rank_63", {"alpha": 0.02, "l1_ratio": 0.9})
    assert 0 < sum(abs(c) > 1e-12 for c in m.coef) < len(m.coef)


def test_linear_contributions_add_up_to_the_prediction(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    m = fit_linear("ridge", d.fit, "fwd_rank_63", {"alpha": 10.0})
    c, names = m.contributions(d.val.head(40))
    assert np.allclose(c.sum(axis=1) + m.intercept, m.predict(d.val.head(40))) and names == m.features


def test_missing_features_are_imputed_with_the_training_median_never_dropped(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    m = fit_linear("ridge", d.fit, "fwd_rank_63", {"alpha": 10.0})
    rows = d.val.head(20).copy()
    rows["dbeta_126"] = np.nan
    assert np.isfinite(m.predict(rows)).all()


def test_too_few_rows_is_refused(prepared, cfg):
    with pytest.raises(InvestModelError, match="too few"):
        fit_linear("ridge", prepared.head(100), "fwd_rank_63", {"alpha": 1.0})


# ---- shallow lgbm + scenario --------------------------------------------------------------------------------------------------------------
def test_shallow_lgbm_learns_the_signal_and_its_shap_is_additive(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    m = fit_lgbm(cfg, d.fit, d.val, "fwd_rank_63")
    rows = d.val.head(50)
    assert spearmanr(m.predict(d.val), d.val["fwd_ret_63"], nan_policy="omit")[0] > 0.1
    c, names = m.contributions(rows)
    raw = m.booster.predict(rows[m.features].to_numpy(float), raw_score=True)
    assert np.allclose(c.sum(axis=1) + m.booster.predict(rows[m.features].to_numpy(float), pred_contrib=True)[:, -1], raw, atol=1e-8) and len(names) == c.shape[1]


def test_scenario_quantiles_are_ordered_and_roughly_calibrated(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    sc = fit_scenario(cfg, d.fit, d.val, "fwd_ret_63")
    q = sc.predict(d.val)
    assert (q["q10"] <= q["q50"]).all() and (q["q50"] <= q["q90"]).all()
    y = d.val["fwd_ret_63"]
    ok = y.notna()
    cov = ((y[ok] >= q.loc[ok, "q10"]) & (y[ok] <= q.loc[ok, "q90"])).mean()
    assert 0.55 < cov < 0.97


def test_artifacts_are_byte_stable_and_round_trip(prepared, cfg, preset):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    hypers = {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}
    a = W.fit_candidates(d, cfg, "b1", preset, hypers)
    b = W.fit_candidates(d, cfg, "b1", preset, hypers)
    for name in cfg.candidates:
        assert a.artifact(name) == b.artifact(name)
        model, scen, doc = load_candidate(a.artifact(name))
        assert np.allclose(np.nan_to_num(model.predict(d.val)), np.nan_to_num(a.models[name].predict(d.val)))
        pd.testing.assert_frame_equal(scen.predict(d.val), a.scenario.predict(d.val))
        assert doc["preset"] == "b1" and doc["horizon"] == 63


# ---- walk-forward ------------------------------------------------------------------------------------------------------------------------
def test_the_embargo_equals_the_preset_horizon(prepared, cfg):
    sessions = pd.DatetimeIndex(sorted(prepared["trade_date"].unique()))
    for key in ("b1", "b2"):
        p = cfg.presets[key]
        for f in W.plan_folds(prepared, cfg, p, pd.Timestamp("2100-01-01")):
            gap = sessions[(sessions > f.train_end) & (sessions < f.test_start)]
            assert len(gap) == p.horizon


def test_training_labels_end_before_the_test_window_and_use_the_presets_own_label(prepared, cfg):
    p = cfg.presets["b1"]
    for f in W.plan_folds(prepared, cfg, p, pd.Timestamp("2100-01-01")):
        d = W.split(prepared, f, cfg)
        for part in (d.fit, d.val):
            assert (pd.to_datetime(part["fwd_end_63"]) < f.test_start).all()
        assert d.purged == 0                                                       # the embargo already equals the horizon
    frame126 = W.prepare(prepared, cfg.presets["b2"])
    assert not frame126["label_end_max"].equals(prepared["label_end_max"])          # B2 purges by its 126-session label, B1 by the 63-session one


def test_nothing_from_the_test_window_or_later_reaches_a_fold_model(frame, cfg, preset):
    hypers = {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}
    base = W.prepare(frame, preset)
    runs = W.run_walk_forward(base, cfg, "b1", preset, hypers, pd.Timestamp("2100-01-01"))
    k = 1
    cut = runs[k].fold.test_end
    mutated = frame.copy()
    later = pd.to_datetime(mutated["trade_date"]) > cut
    rng = np.random.default_rng(2)
    for c in ("mom_6m", "mom_12m", "vol_126", "sma_ratio_200", "fwd_ret_63", "fwd_rank_63", "fwd_ret_126", "fwd_rank_126"):
        mutated.loc[later, c] = rng.normal(size=int(later.sum()))
    for c in ("mom_6m_csrank", "mom_12m_csrank", "vol_126_csrank", "sma_ratio_200_csrank", "dd_252_csrank"):
        mutated.loc[later, c] = rng.random(int(later.sum()))
    other = W.run_walk_forward(W.prepare(mutated, preset), cfg, "b1", preset, hypers, pd.Timestamp("2100-01-01"))
    for a, b in zip(runs[: k + 1], other[: k + 1]):
        for name in cfg.candidates:
            assert a.cands.artifact(name) == b.cands.artifact(name)
            pd.testing.assert_frame_equal(a.preds[name], b.preds[name])


def test_grid_logs_every_point_including_failures_and_freezes_the_best_per_family(prepared, cfg, preset, monkeypatch):
    d = W.split(prepared, W.plan_folds(prepared, cfg, preset, pd.Timestamp("2100-01-01"))[0], cfg)
    seen = []
    real = W.fit_linear

    def flaky(kind, fit, target, hyper, seed=0):
        if kind == "elasticnet" and hyper["alpha"] == 0.01:
            raise RuntimeError("did not converge")
        return real(kind, fit, target, hyper, seed)
    monkeypatch.setattr(W, "fit_linear", flaky)
    best = W.tune(d, cfg, preset, seen.append)
    assert len(seen) == len(W.grid(cfg)) == 4
    assert [r["state"] for r in seen] == ["COMPLETE", "COMPLETE", "COMPLETE", "FAIL"] and "did not converge" in seen[-1]["error"]
    assert set(best) == {"ridge", "elasticnet"} and best["elasticnet"]["alpha"] == 0.001


def test_planted_signal_is_found_and_null_data_gives_no_ic(cfg, preset):
    hypers = {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}
    out = {}
    for name, signal in (("planted", 1.0), ("null", 0.0)):
        f = W.prepare(synthetic_invest(n_days=900, seed=11, signal=signal), preset)
        runs = W.run_walk_forward(f, cfg, "b1", preset, hypers, pd.Timestamp("2100-01-01"))
        for cand in ("factor", "ridge", "lgbm"):
            ev = W.evaluable(W.concat(runs, cand), preset, pd.Timestamp("2100-01-01"))
            out[(name, cand)] = ic_summary(daily_rank_ic(ev, "score", "fwd_ret_63"), 63)
    for cand in ("factor", "ridge", "lgbm"):
        assert out[("planted", cand)]["mean"] > 0.1
        assert abs(out[("null", cand)]["mean"]) < 0.08                                   # noise: a leak would inflate this


def test_labels_reaching_into_the_held_out_period_are_excluded_from_evaluation(prepared, cfg, preset):
    hold = pd.Timestamp(sorted(prepared["trade_date"].unique())[640])
    runs = W.run_walk_forward(prepared, cfg, "b1", preset, {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}, hold)
    ev = W.evaluable(W.concat(runs, "factor"), preset, hold)
    assert (pd.to_datetime(ev["fwd_end_63"]) < hold).all() and len(ev) < len(W.concat(runs, "factor"))


def test_final_model_never_sees_the_held_out_period(frame, cfg):
    p = cfg.presets["b2"]
    f = W.prepare(frame, p)
    dates = sorted(f["trade_date"].unique())
    hs, he = pd.Timestamp(dates[760]), pd.Timestamp(dates[-1])
    hypers = {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}
    cands, d = W.fit_final(f, cfg, "b2", p, hypers, hs, he)
    assert (pd.to_datetime(d.fit["fwd_end_126"]) < hs).all() and (pd.to_datetime(d.val["fwd_end_126"]) < hs).all()
    m = frame.copy()
    inside = pd.to_datetime(m["trade_date"]) >= hs
    m.loc[inside, "mom_6m"] = 99.0
    m.loc[inside, "fwd_ret_126"] = 5.0
    m.loc[inside, "fwd_rank_126"] = 1.0
    cands2, _ = W.fit_final(W.prepare(m, p), cfg, "b2", p, hypers, hs, he)
    for name in cfg.candidates:
        assert cands.artifact(name) == cands2.artifact(name)
