"""INVEST decision rule, per-signal details, fundamentals ablation and the registry/held-out plumbing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.config import InvestDecision
from predict_stock.invest import evaluate as E
from predict_stock.invest import walkforward as W
from tests.invest_helpers import synthetic_invest
from tests.test_invest_models import cfg_small

RULE = InvestDecision()


def summ(sharpe, stress=0.5):
    return {"net": {"sharpe": sharpe}, "sensitivity": {"2.0": {"sharpe": stress}}}


def rc(p=0.05, best="factor", obs=0.3):
    return {"candidates": ["factor", "ridge"], "p_value": p, "best": best, "observed": {"factor": obs, "ridge": 0.0}}


BASE = {"equal_weight": summ(0.9), "mom_long": summ(0.8), "mom_short": summ(0.5), "random_monthly": summ(5.0)}
IC = {"factor": {"mean": 0.03, "t_stat": 2.5}, "ridge": {"mean": 0.0, "t_stat": 0.0}}
ROLL = {"252": {"share_ahead": 0.7, "n_windows": 500}}


def test_decision_passes_only_when_every_criterion_holds():
    out = E.decide("factor", {"factor": summ(1.2)}, BASE, None, IC, rc(), ROLL, RULE)
    assert out["passed"] and out["best"] == "factor"


@pytest.mark.parametrize("model,ic,r,roll,failing", [
    (summ(0.85), IC, rc(), ROLL, 0),                                              # below equal-weight
    (summ(1.2), IC, rc(p=0.4), ROLL, 1),                                          # could be luck once the number of candidates is accounted for
    (summ(1.2), {"factor": {"mean": 0.03, "t_stat": 1.0}}, rc(), ROLL, 2),
    (summ(1.2), IC, rc(), {"252": {"share_ahead": 0.4, "n_windows": 500}}, 3),
    (summ(1.2, stress=-0.1), IC, rc(), ROLL, 4),
])
def test_decision_fails_when_exactly_one_criterion_fails(model, ic, r, roll, failing):
    out = E.decide("factor", {"factor": model}, BASE, None, ic, r, roll, RULE)
    assert not out["passed"] and [i for i, c in enumerate(out["criteria"]) if not c["ok"]] == [failing]


def test_random_portfolios_are_never_the_baseline_to_beat_and_the_best_rival_is_named():
    base = {**BASE, "mom_long": summ(1.0)}
    out = E.decide("factor", {"factor": summ(0.95)}, base, None, IC, rc(), ROLL, RULE)
    assert not out["passed"] and out["criteria"][0]["detail"] == "best baseline: mom_long"


def test_a_significant_check_on_a_negative_difference_does_not_pass():
    out = E.decide("factor", {"factor": summ(1.2)}, BASE, None, IC, rc(p=0.01, obs=-0.4), ROLL, RULE)
    assert not out["criteria"][1]["ok"]


# ---- details of a signal ----------------------------------------------------------------------------------------------------------------
def test_details_carry_scenarios_horizon_thesis_flags_and_contributions():
    cfg = cfg_small()
    p = cfg.presets["b2"]
    f = W.prepare(synthetic_invest(n_days=900, seed=5), p)
    fold = W.plan_folds(f, cfg, p, pd.Timestamp("2100-01-01"))[0]
    d = W.split(f, fold, cfg)
    cands = W.fit_candidates(d, cfg, "b2", p, {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}})
    preds = W.predict_frames(cands, d.test, p)
    for name in cfg.candidates:
        rows = d.test.head(20)
        det = E.build_details(rows, preds[name].loc[rows.index], cands.models[name], p, cfg)
        one = det[0]
        assert one["horizon_sessions"] == 126 and one["scenario"]["bear_q10"] <= one["scenario"]["base_q50"] <= one["scenario"]["bull_q90"]
        assert set(one["thesis_break"]["limits"]) == {"rs_rank_below", "drawdown_beyond"} and one["thesis_break"]["limits"]["drawdown_beyond"] == -0.25
        assert 1 <= len(one["contributions"]) <= 5 and all(len(c) == 3 for c in one["contributions"])
        mags = [abs(c[2]) for c in one["contributions"]]
        assert mags == sorted(mags, reverse=True)
    flagged = E.build_details(d.test.head(200), preds["factor"].loc[d.test.head(200).index], cands.models["factor"], p, cfg)
    assert any(x["thesis_break"]["triggered"] for x in flagged) and any(not x["thesis_break"]["triggered"] for x in flagged)


# ---- fundamentals -----------------------------------------------------------------------------------------------------------------------
def test_without_fundamental_columns_the_report_says_so_and_nothing_is_fitted():
    cfg = cfg_small()
    f = synthetic_invest(n_days=500, seed=1)
    out = E.fundamentals_ablation(f, cfg, "b1", cfg.presets["b1"], {}, pd.Timestamp("2100-01-01"))
    assert out["available"] is False and "prices only" in out["note"]


def test_a_planted_fundamental_signal_shows_up_as_an_ic_gain():
    cfg = cfg_small()
    p = cfg.presets["b1"]
    f = synthetic_invest(n_days=800, seed=7, fund=True)
    assert E.fundamental_columns(f, cfg.fundamental_prefixes) == ["fund_quality"]
    out = E.fundamentals_ablation(f, cfg, "b1", p, {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}, pd.Timestamp("2100-01-01"))
    assert out["available"] and set(out["candidates"]) == {"ridge", "elasticnet", "lgbm"}
    assert all(d["ic_gain"] > 0.03 for d in out["candidates"].values())          # the models with the column are clearly better than without it
    null = synthetic_invest(n_days=800, seed=7, fund=True, signal=0.0)
    out0 = E.fundamentals_ablation(null, cfg, "b1", p, {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}}, pd.Timestamp("2100-01-01"))
    assert all(abs(d["ic_gain"]) < 0.05 for d in out0["candidates"].values())


def test_the_factor_score_never_uses_the_fundamental_columns():
    cfg = cfg_small()
    assert not [c for c, _ in cfg.factors if c.startswith("fund_")]


# ---- database ---------------------------------------------------------------------------------------------------------------------------
def test_candidate_artifacts_are_registered_idempotently_with_versions(engine, cfg, tmp_path):
    from sqlalchemy import select

    from predict_stock.config import PROJECT_ROOT
    from predict_stock.db.models import Model
    from predict_stock.db.session import session_scope
    from predict_stock.features.sets import load_definitions, sync_definitions
    from predict_stock.invest.models import sha256
    from predict_stock.swing.registry import register_artifact
    fsets, lspecs = load_definitions(PROJECT_ROOT / cfg.features.definitions_path)
    with session_scope(engine) as s:
        sync_definitions(s, fsets, lspecs)
    c = cfg_small()
    p = c.presets["b1"]
    f = W.prepare(synthetic_invest(n_days=700, seed=2), p)
    d = W.split(f, W.plan_folds(f, c, p, pd.Timestamp("2100-01-01"))[0], c)
    cands = W.fit_candidates(d, c, "b1", p, {"ridge": {"alpha": 10.0}, "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5}})
    kw = dict(algo="invest_candidate", feature_set="invest:2", label_spec="invest:1", dataset_id=None, experiment_id=None, params={"preset": "b1"}, seed=42, artifact_dir=tmp_path)
    a = register_artifact(engine, cfg, cands.artifact("ridge"), "invest_b1_ridge_f00", **kw)
    b = register_artifact(engine, cfg, cands.artifact("ridge"), "invest_b1_ridge_f00", **kw)
    other = register_artifact(engine, cfg, cands.artifact("lgbm"), "invest_b1_ridge_f00", **kw)
    assert a[:2] == (b[0], 1) and b[3] and other[1] == 2 and other[2] == sha256(cands.artifact("lgbm"))
    with engine.connect() as conn:
        status = dict(conn.execute(select(Model.version, Model.status).where(Model.name == "invest_b1_ridge_f00")).all())
    assert status == {1: "retired", 2: "candidate"}
    assert (tmp_path / "invest_b1_ridge_f00" / "v2.json.gz").read_bytes() == cands.artifact("lgbm")
