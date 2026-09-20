"""Optuna tuning (every trial logged, failures included), IC / quantile / holding metrics, strategies and the decision rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from predict_stock.backtest.engine import Signal
from predict_stock.config import SwingConfig, SwingDecision
from predict_stock.db.models import Experiment
from predict_stock.swing import metrics as SM
from predict_stock.swing.evaluate import decide
from predict_stock.swing.registry import log_experiment, trial_sink
from predict_stock.swing.strategies import barrier_signals, topk_signals, with_probability_filter
from predict_stock.swing.tuning import run_study, suggest, validation_ic
from tests.swing_helpers import synthetic_frame


def small_cfg(trials=4, **kw):
    return SwingConfig(n_estimators=40, early_stopping_rounds=8, optuna={"trials": trials, "seed": 1, "timeout_s": None},
                       lgbm={"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30}, **kw)


@pytest.fixture(scope="module")
def parts():
    df = synthetic_frame(n_days=300, n_inst=15, seed=21)
    d = sorted(df["trade_date"].unique())
    return df[df["trade_date"] < d[200]], df[(df["trade_date"] >= d[200]) & (df["trade_date"] < d[280])]


# ---- Optuna ------------------------------------------------------------------------------------------------------------------------
def test_the_number_of_trials_is_capped_and_every_one_is_sunk(parts):
    fit, val = parts
    seen = []
    res = run_study(fit, val, small_cfg(trials=5), seen.append)
    assert len(seen) == 5 and [r["number"] for r in seen] == list(range(5))
    assert res.n_complete == 5 and res.n_failed == 0
    assert set(res.best_params) == {"learning_rate", "num_leaves", "min_child_samples", "feature_fraction", "bagging_fraction", "lambda_l2"}
    assert all(r["state"] == "COMPLETE" and r["value"] is not None and r["params"] for r in seen)


def test_failed_trials_are_logged_with_their_error_and_the_study_continues(parts):
    fit, val = parts
    seen = []

    def flaky(trial, fit, val, cfg):
        tree = suggest(trial)
        if trial.number % 2 == 1:
            raise RuntimeError(f"boom {trial.number}")
        return validation_ic(fit, val, cfg, tree)
    res = run_study(fit, val, small_cfg(trials=4), seen.append, objective=flaky)
    assert [r["state"] for r in seen] == ["COMPLETE", "FAIL", "COMPLETE", "FAIL"]
    assert seen[1]["error"] == "RuntimeError: boom 1" and seen[1]["value"] is None
    assert (res.n_complete, res.n_failed) == (2, 2) and res.best_params


def test_a_study_where_everything_fails_returns_no_parameters(parts):
    fit, val = parts
    seen = []
    res = run_study(fit, val, small_cfg(trials=3), seen.append, objective=lambda t, f, v, c: (_ for _ in ()).throw(ValueError("nope")))
    assert res.best_params == {} and res.best_value is None and len(seen) == 3 and res.n_failed == 3


def test_the_study_is_reproducible_with_a_seed(parts):
    fit, val = parts
    a, b = [], []
    run_study(fit, val, small_cfg(trials=3), a.append)
    run_study(fit, val, small_cfg(trials=3), b.append)
    assert [(r["params"], r["value"]) for r in a] == [(r["params"], r["value"]) for r in b]


def test_trials_are_written_to_the_experiments_table_including_failures(engine, cfg, parts):
    fit, val = parts
    sink = trial_sink(engine, cfg, "study1", run_id=None)

    def flaky(trial, fit, val, cfg_):
        tree = suggest(trial)
        if trial.number == 1:
            raise RuntimeError("bad trial")
        return validation_ic(fit, val, cfg_, tree)
    run_study(fit, val, small_cfg(trials=3), sink, objective=flaky)
    with engine.connect() as c:
        rows = c.execute(select(Experiment.name, Experiment.status, Experiment.summary).where(Experiment.name.like("swing:optuna:study1:trial%")).order_by(Experiment.name)).all()
    assert [r.status for r in rows] == ["success", "failed", "success"]
    assert rows[1].summary["error"] == "RuntimeError: bad trial" and rows[0].summary["validation_rank_ic"] is not None
    # logging the same trial again refreshes the row instead of adding one
    sink({"number": 0, "state": "COMPLETE", "value": 0.5, "params": {}, "error": None, "duration_s": 1.0})
    with engine.connect() as c:
        assert c.execute(select(Experiment.id).where(Experiment.name.like("swing:optuna:study1:trial%"))).all().__len__() == 3


def test_log_experiment_with_a_key_is_idempotent(engine, cfg):
    a = log_experiment(engine, cfg, "swing:x", params={"a": 1}, key="k1", summary={"v": 1})
    b = log_experiment(engine, cfg, "swing:x", params={"a": 1}, key="k1", summary={"v": 2})
    c = log_experiment(engine, cfg, "swing:x", params={"a": 2}, key="k2")
    assert a == b and c != a
    with engine.connect() as conn:
        assert conn.execute(select(Experiment.summary).where(Experiment.id == a)).scalar() == {"v": 2}


# ---- IC and the other metrics ------------------------------------------------------------------------------------------------------------
def test_daily_rank_ic_of_a_perfect_and_a_reversed_score():
    d = pd.DataFrame({"trade_date": np.repeat(pd.bdate_range("2024-01-01", periods=3), 10), "score": np.tile(np.arange(10.0), 3)})
    d["ret"] = d["score"] * 2
    assert daily(d, "ret") == pytest.approx([1, 1, 1])
    d["ret"] = -d["score"]
    assert daily(d, "ret") == pytest.approx([-1, -1, -1])


def daily(d, target):
    return list(SM.daily_rank_ic(d, "score", target).values)


def test_ic_skips_days_with_too_few_names_or_a_constant_score():
    d = pd.DataFrame({"trade_date": ["2024-01-01"] * 5 + ["2024-01-02"] * 10, "score": list(range(5)) + [1.0] * 10, "ret": list(range(5)) + list(range(10))})
    assert len(SM.daily_rank_ic(d, "score", "ret")) == 0


def test_a_planted_signal_has_a_significant_ic_and_pure_noise_does_not():
    planted = synthetic_frame(n_days=250, n_inst=20, seed=3, signal=1.0)
    null = synthetic_frame(n_days=250, n_inst=20, seed=3, signal=0.0)
    ic_p = SM.ic_summary(SM.daily_rank_ic(planted.assign(score=planted["f_a"]), "score", "fwd_ret_5"), 5)
    ic_n = SM.ic_summary(SM.daily_rank_ic(null.assign(score=null["f_a"]), "score", "fwd_ret_5"), 5)
    assert ic_p["mean"] > 0.1 and ic_p["t_stat"] > 4
    assert abs(ic_n["t_stat"]) < 3 and abs(ic_n["mean"]) < 0.05


def test_the_overlap_adjustment_shrinks_the_t_statistic():
    ic = pd.Series(np.random.default_rng(0).normal(0.05, 0.2, 500))
    assert SM.ic_summary(ic, 5)["t_stat"] == pytest.approx(SM.ic_summary(ic, 1)["t_stat"] / np.sqrt(5))


def test_pinball_and_quantile_report():
    assert SM.pinball(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5) == pytest.approx(0.75)
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    pred = pd.DataFrame({"y": y, "q10": np.quantile(y, 0.1), "q50": np.quantile(y, 0.5), "q90": np.quantile(y, 0.9)})
    r = SM.quantile_report(pred, "y")
    assert r["coverage"]["q10"] == pytest.approx(0.1, abs=0.01) and r["interval_coverage"] == pytest.approx(0.8, abs=0.01)
    assert r["pinball"]["q50"] == pytest.approx(r["pinball_constant"]["q50"])


def test_holding_report_compares_with_the_constant_guess():
    rng = np.random.default_rng(0)
    med = rng.choice([2.0, 8.0], 2000)
    pred = pd.DataFrame({"hold_median": med, "hold_p75": med + 1, "tb_time": med + rng.integers(-1, 2, 2000)})
    r = SM.holding_report(pred, buckets=2)
    assert r["mae_median"] < r["mae_constant"] and [b["n"] for b in r["buckets"]] == [1000 - (1000 - b["n"]) for b in r["buckets"]]
    assert r["buckets"][0]["realised_median"] < r["buckets"][1]["realised_median"]


# ---- strategies --------------------------------------------------------------------------------------------------------------------------
def make_pred(n_days=12, n_inst=6):
    cal = pd.bdate_range("2024-01-01", periods=n_days)
    rows = [{"trade_date": d, "instrument_id": j + 1, "score": float(j), "proba": 0.2 + 0.1 * j, "atr_pct_14": 0.02 + 0.005 * j} for d in cal for j in range(n_inst)]
    return pd.DataFrame(rows), cal


def test_topk_signals_pick_the_best_scores_on_the_first_session_of_each_week():
    pred, cal = make_pred()
    sig = topk_signals(pred, cal, 0, len(cal) - 1, k=2, max_weight=None, rebalance="weekly")
    assert sorted(sig) == [0, 5, 10]                                            # Mondays
    assert {it.instrument_id for it in sig[0].items} == {6, 5} and all(it.weight == pytest.approx(0.5) for it in sig[0].items) and sig[0].full_rebalance


def test_the_probability_filter_leaves_cash_when_too_few_names_pass():
    pred, cal = make_pred()
    sig = topk_signals(pred, cal, 0, len(cal) - 1, k=4, max_weight=None, base_rate=0.4, min_prob_multiple=1.0)   # proba >= 0.4: names 3..6 -> j = 2..5
    assert len(sig[0].items) == 4
    strict = topk_signals(pred, cal, 0, len(cal) - 1, k=4, max_weight=None, base_rate=0.4, min_prob_multiple=1.25)   # proba >= 0.5: only j = 3, 4, 5
    assert sorted(it.instrument_id for it in strict[0].items) == [4, 5, 6] and sum(it.weight for it in strict[0].items) == pytest.approx(0.75)
    assert with_probability_filter(pred, 0.4, 0.0) is pred
    none = topk_signals(pred, cal, 0, len(cal) - 1, k=4, max_weight=None, base_rate=0.9, min_prob_multiple=1.0)
    assert none[0].items == [] and none[0].full_rebalance                       # nothing passes: an empty full rebalance = sell everything


def test_topk_without_a_filter_matches_the_baseline_signals_exactly():
    from predict_stock.backtest.baselines import BaselineSpec, make_signals
    pred, cal = make_pred()
    mine = topk_signals(pred, cal, 0, len(cal) - 1, k=3, max_weight=0.4, rebalance="weekly")
    spec = BaselineSpec("x", "x", "x", score="score", k=3, rebalance="weekly")
    ref = make_signals(spec, pred, cal, 0, len(cal) - 1, 0.4)
    assert mine.keys() == ref.keys() and all(mine[i].items == ref[i].items for i in mine)


def test_barrier_signals_set_atr_based_levels_and_entry_only_when_flat():
    pred, cal = make_pred()
    sig = barrier_signals(pred, cal, 0, len(cal) - 1, max_positions=2, target_mult=2.0, stop_mult=1.0, max_hold=10, base_rate=0.3, min_prob_multiple=1.0)
    s = sig[0]
    assert not s.full_rebalance and len(s.items) == 2
    top = s.items[0]
    assert top.instrument_id == 6 and top.stop_pct == pytest.approx(0.045) and top.target_pct == pytest.approx(0.09)
    assert top.max_hold == 10 and top.only_if_flat and top.weight == pytest.approx(0.5)
    assert sorted(sig) == list(range(len(cal)))                                 # a signal every session; the engine ignores names already held


def test_barrier_signals_skip_names_without_atr_or_below_the_probability_bar():
    pred, cal = make_pred(3, 4)
    pred.loc[pred.instrument_id == 4, "atr_pct_14"] = np.nan
    sig = barrier_signals(pred, cal, 0, 2, max_positions=3, target_mult=2, stop_mult=1, max_hold=5, base_rate=0.35, min_prob_multiple=1.0)
    assert [it.instrument_id for it in sig[0].items] == [3]                     # 4 has no ATR; 1 and 2 have proba 0.2 / 0.3 < 0.35


# ---- the pre-registered decision -------------------------------------------------------------------------------------------------------
def summ(sharpe, folds=(1, 1, 1, 1), stress=0.5):
    return {"net": {"sharpe": sharpe}, "folds": [{"sharpe": f} for f in folds], "sensitivity": {"2.0": {"sharpe": stress}}}


NOISE = {"schedules": {"weekly": {"net": {"sharpe": {"p95": -0.3}}}}}
RULE = SwingDecision()
IC = {"mean": 0.03, "t_stat": 2.5}


def test_decision_passes_only_when_every_criterion_holds():
    base = {"equal_weight": summ(0.9), "mom_short": summ(0.7), "random_weekly": summ(-0.5), "bh_vnindex": summ(0.5)}
    assert decide(summ(1.2), base, NOISE, IC, RULE)["passed"]


@pytest.mark.parametrize("model,noise,ic,failing", [
    (summ(0.8), NOISE, IC, 0),                                                            # below equal_weight
    (summ(1.2), {"schedules": {"weekly": {"net": {"sharpe": {"p95": 1.5}}}}}, IC, 1),     # inside the noise of random portfolios
    (summ(1.2), NOISE, {"mean": 0.03, "t_stat": 1.0}, 2),                                 # IC not significant
    (summ(1.2), NOISE, {"mean": -0.03, "t_stat": 3.0}, 2),                                # significant but the wrong sign
    (summ(1.2, folds=(1, -1, -1, -1)), NOISE, IC, 3),                                     # works in one window only
    (summ(1.2, stress=-0.1), NOISE, IC, 4),                                               # does not survive 2x costs
])
def test_decision_fails_when_exactly_one_criterion_fails(model, noise, ic, failing):
    base = {"equal_weight": summ(0.9), "mom_short": summ(0.7), "random_weekly": summ(-0.5)}
    out = decide(model, base, noise, ic, RULE)
    assert not out["passed"]
    assert [i for i, c in enumerate(out["criteria"]) if not c["ok"]] == [failing]


def test_random_baselines_are_not_counted_as_the_baseline_to_beat_and_the_best_one_is_named():
    base = {"equal_weight": summ(0.9), "mom_long": summ(1.0), "random_weekly": summ(5.0)}
    out = decide(summ(0.95), base, NOISE, IC, RULE)
    assert not out["passed"] and out["criteria"][0]["detail"] == "best baseline: mom_long" and out["criteria"][0]["threshold"] == 1.0
