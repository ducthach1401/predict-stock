"""Model lifecycle: registry transitions, promote / rollback, the champion-vs-challenger comparison (a worse challenger is REJECTED, a better one may be promoted), multiple testing,
the held-out guards, PSI / KS and the drift alert, provenance, and the outcome-based corrections."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from conftest import apply
from panel_helpers import random_panel
from predict_stock.backtest.engine import MarketData
from predict_stock.backtest.market import MarketRules
from predict_stock.backtest.runner import Setup
from predict_stock.backtest.walkforward import Holdout
from predict_stock.config import PROJECT_ROOT
from predict_stock.db.models import Alert, Experiment, Model, ModelStatusLog, MonitoringMetric, Prediction
from predict_stock.db.session import session_scope
from predict_stock.features.sets import load_definitions, sync_definitions
from predict_stock.lifecycle import compare as CMP
from predict_stock.lifecycle import monitor as MON
from predict_stock.lifecycle import outcomes as OUT
from predict_stock.lifecycle import reference as REF
from predict_stock.lifecycle import registry as REG
from predict_stock.lifecycle import retrain as RT
from predict_stock.swing.registry import log_experiment, register_artifact, save_predictions

N_DAYS, N_INST = 330, 12


@pytest.fixture
def db(engine, cfg):
    fsets, lspecs = load_definitions(PROJECT_ROOT / cfg.features.definitions_path)
    with session_scope(engine) as s:
        sync_definitions(s, fsets, lspecs)
    apply(engine, "LU", [f"L{k:02d}" for k in range(N_INST)], date(2020, 1, 1))
    return engine


def lcfg(cfg, **promotion):
    """The test config: a protected period far away, small bootstraps, a short shadow requirement and a top-3 portfolio (12 stocks)."""
    pr = cfg.lifecycle.promotion.model_copy(update={"min_days": {"swing": 20, "invest_b1": 20, "invest_b2": 20}, "shadow_min_weeks": {"swing": 2, "invest_b1": 2, "invest_b2": 2},
                                                    "bootstrap_resamples": 400, **promotion})
    lc = cfg.lifecycle.model_copy(update={"promotion": pr, "protected_period": ("2099-01-01", "2099-12-31")})
    sw = cfg.swing.model_copy(update={"strategy": cfg.swing.strategy.model_copy(update={"top_k": 3})})
    return cfg.model_copy(update={"lifecycle": lc, "swing": sw})


def mk(engine, cfg, tmp_path, name, *, strategy="swing", trained_until=date(2024, 3, 1), payload=None, status=None, params=None):
    mid, ver, sha, reused = register_artifact(engine, cfg, (payload or name).encode(), name, algo="test", feature_set="swing:1", label_spec="swing:1", dataset_id=None, experiment_id=None,
                                              params=params or {}, seed=1, artifact_dir=tmp_path, strategy=strategy, trained_until=trained_until)
    if status:
        REG.set_status(engine, mid, status, actor="test", reason="setup")
    return mid


def world(engine, cfg, seed=3, shift_last=0.0, shift_features=3):
    """A synthetic market for the universe LU with a SWING-like frame: forward returns, event labels and four features (the last 60 sessions shifted by ``shift_last`` on the first
    ``shift_features`` of them)."""
    with engine.connect() as c:
        ids = sorted(c.execute(select(Model.id).where(Model.id < 0)).scalars().all())
    from predict_stock.db.models import InstrumentSymbolHistory
    with engine.connect() as c:
        ids = sorted(c.execute(select(InstrumentSymbolHistory.instrument_id)).scalars().all())
    panel = random_panel(N_DAYS, N_INST, seed=seed, gaps=0, bench=False, first_id=1)
    cal = panel.calendar
    frames = {k: getattr(panel, k).set_axis(ids, axis=1) for k in ("open", "high", "low", "close")}
    rng = np.random.default_rng(seed)
    close = frames["close"]
    rows = []
    for i in range(N_DAYS - 5):
        fwd = close.iloc[i + 5] / close.iloc[i] - 1
        for j, iid in enumerate(ids):
            f = rng.normal(size=4)
            if shift_last and i >= N_DAYS - 65:
                f[:shift_features] += shift_last
            rows.append({"trade_date": cal[i], "instrument_id": iid, "f1": f[0], "f2": f[1], "f3": f[2], "f4": f[3], "fwd_ret_5": float(fwd[iid]), "fwd_end_5": cal[i + 5],
                         "tb_label": float(rng.choice([-1.0, 0.0, 1.0], p=[0.4, 0.3, 0.3])), "tb_end": cal[min(i + 10, N_DAYS - 1)]})
    frame = pd.DataFrame(rows)
    data = MarketData(frames["open"], frames["high"], frames["low"], frames["close"], pd.DataFrame(0.07, index=cal, columns=ids))
    setup = Setup(cfg, MarketRules.from_config(cfg.market), data, cal, {"swing": frame, "invest": frame}, {}, 0, N_DAYS - 1, Holdout(cal[-1] + pd.Timedelta(days=1), cal[-1] + pd.Timedelta(days=30)),
                  {}, {}, "LU", {})
    return setup, ids


def store_scores(engine, model_id, setup, kind, lo, hi, seed, proba=0.3):
    """Stored predictions of a model: 'skilled' = the cross-sectional rank of the FUTURE 5-day return plus noise, 'noise' = random."""
    frame = setup.frames["swing"]
    d = pd.to_datetime(frame["trade_date"])
    part = frame[(d >= setup.calendar[lo]) & (d <= setup.calendar[hi])].copy()
    rng = np.random.default_rng(seed)
    if kind == "skilled":
        part["score"] = part.groupby("trade_date")["fwd_ret_5"].rank(pct=True) + rng.normal(0, 0.25, len(part))
    else:
        part["score"] = rng.normal(size=len(part))
    part["proba"] = proba
    part["rank_in_universe"] = part.groupby("trade_date")["score"].rank(ascending=False, method="first").astype(int)
    part["details"] = [{} for _ in range(len(part))]
    from predict_stock.lifecycle.shadow import _universe_id
    save_predictions(engine, model_id, _universe_id(engine, "LU"), 5, part[["trade_date", "instrument_id", "score", "proba", "rank_in_universe", "details"]])


def pair(db, cfg, tmp_path, setup, champ_kind, chall_kind, protocol=True):
    champ = mk(db, cfg, tmp_path, "swing_lgbm_final", trained_until=setup.calendar[60].date(), payload="champ", status="champion")
    chall = mk(db, cfg, tmp_path, "swing_lgbm_final", trained_until=setup.calendar[80].date(), payload="chall", params={"protocol": CMP.protocol_hash(cfg, "swing")} if protocol else {})
    REG.set_status(db, chall, "shadow", actor="test", reason="retrain")
    store_scores(db, champ, setup, champ_kind, 100, 300, 11)
    store_scores(db, chall, setup, chall_kind, 100, 300, 12)
    return champ, chall


# ---- registry -----------------------------------------------------------------------------------------------------------------------
def test_one_champion_per_strategy_promotion_retires_the_old_one_and_every_change_is_logged(db, cfg, tmp_path):
    a = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="a", status="champion")
    b = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="b")
    REG.set_status(db, b, "shadow", actor="t", reason="challenger")
    assert REG.champion(db, "swing").id == a and [m.id for m in REG.with_status(db, "swing", "shadow")] == [b]
    REG.set_status(db, b, "champion", actor="t", reason="promoted")
    assert REG.champion(db, "swing").id == b and REG.get(db, a).status == "retired"
    with session_scope(db) as s:
        assert s.scalar(select(func.count()).select_from(Model).where(Model.strategy == "swing", Model.status == "champion")) == 1
        log = s.execute(select(ModelStatusLog.model_id, ModelStatusLog.from_status, ModelStatusLog.to_status).order_by(ModelStatusLog.id)).all()
    assert (a, "champion", "retired") in log and (b, "shadow", "champion") in log and (b, "candidate", "shadow") in log
    assert REG.previous_champion(db, "swing").id == a


@pytest.mark.parametrize("frm, to", [("retired", "shadow"), ("retired", "candidate"), ("candidate", "candidate"), ("champion", "shadow"), ("champion", "candidate")])
def test_illegal_transitions_are_refused(db, cfg, tmp_path, frm, to):
    mid = mk(db, cfg, tmp_path, "swing_lgbm_final", payload=f"{frm}{to}")
    if frm == "champion":
        REG.set_status(db, mid, "champion", actor="t", reason="x")
    elif frm == "retired":
        REG.set_status(db, mid, "retired", actor="t", reason="x")
    with pytest.raises(REG.LifecycleError):
        REG.set_status(db, mid, to, actor="t", reason="x")


def test_a_candidate_becomes_champion_directly_only_when_the_strategy_has_none(db, cfg, tmp_path):
    a = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="a", status="champion")                    # the first champion: allowed
    b = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="b")
    with pytest.raises(REG.LifecycleError, match="cannot become champion"):
        REG.set_status(db, b, "champion", actor="t", reason="skip the shadow stage")                # a second one has to pass the shadow stage
    assert REG.champion(db, "swing").id == a


def test_a_model_without_a_strategy_cannot_serve(db, cfg, tmp_path):
    mid = mk(db, cfg, tmp_path, "orphan", strategy=None)
    with pytest.raises(REG.LifecycleError):
        REG.set_status(db, mid, "champion", actor="t", reason="x")


def test_rollback_restores_the_previous_champion_and_retires_the_current_one(db, cfg, tmp_path):
    a = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="a", status="champion")
    b = mk(db, cfg, tmp_path, "swing_lgbm_final", payload="b", status="shadow")
    REG.set_status(db, b, "champion", actor="t", reason="promoted")
    out = CMP.rollback(db, "swing", reason="the new model misbehaves")
    assert REG.champion(db, "swing").id == a and REG.get(db, b).status == "retired" and "swing_lgbm_final" in out["champion"]
    with pytest.raises(REG.LifecycleError):
        CMP.rollback(db, "invest_b1", reason="nothing to go back to")


# ---- champion vs challenger -----------------------------------------------------------------------------------------------------------
def test_a_challenger_that_does_not_beat_the_champion_is_rejected_and_cannot_be_promoted(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    champ, chall = pair(db, c, tmp_path, setup, "skilled", "noise")
    rep = CMP.compare(db, c, setup, chall, setup.calendar[300])
    assert rep["passed"] is False and rep["verdict"].startswith("REJECTED")
    beats = [x for x in rep["criteria"] if x["name"].startswith("rank IC OR net Sharpe")][0]
    assert beats["ok"] is False and rep["ic"]["diff"] < 0 and rep["ic"]["lower_bound"] < 0
    with session_scope(db) as s:
        exp = s.scalars(select(Experiment).where(Experiment.name == f"lifecycle:compare:{chall}")).one()
    assert exp.status == "rejected" and exp.summary["verdict"] == rep["verdict"]
    with pytest.raises(REG.LifecycleError, match="does not meet the promotion rule"):
        CMP.promote(db, c, setup, chall, setup.calendar[300])
    assert REG.champion(db, "swing").id == champ and REG.get(db, chall).status == "shadow"
    CMP.reject(db, chall, reason=rep["verdict"])
    assert REG.get(db, chall).status == "retired"


def test_a_clearly_better_challenger_passes_and_is_promoted_only_by_the_manual_command(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    champ, chall = pair(db, c, tmp_path, setup, "noise", "skilled")
    rep = CMP.compare(db, c, setup, chall, setup.calendar[300])
    assert rep["passed"] is True, rep["criteria"]
    assert REG.champion(db, "swing").id == champ                              # comparing never changes anything
    out = CMP.promote(db, c, setup, chall, setup.calendar[300], note="reviewed")
    assert REG.champion(db, "swing").id == chall and REG.get(db, champ).status == "retired" and out["promoted"].startswith("swing_lgbm_final")
    CMP.rollback(db, "swing", reason="test")
    assert REG.champion(db, "swing").id == champ


def test_a_retrained_model_that_is_merely_a_different_random_draw_is_not_promoted_even_if_its_point_estimate_is_higher(db, cfg, tmp_path):
    """The point of the bootstrap rule: two models with the same skill differ by luck on any finite sample, and the luckier one must not replace the other."""
    c = lcfg(cfg)
    setup, _ = world(db, c)
    champ, chall = pair(db, c, tmp_path, setup, "skilled", "skilled")
    rep = CMP.compare(db, c, setup, chall, setup.calendar[300])
    assert rep["ic"]["lower_bound"] < 0 < abs(rep["ic"]["diff"]) and rep["passed"] is False       # a nonzero difference, but not distinguishable from luck
    assert REG.champion(db, "swing").id == champ


def test_no_promotion_without_enough_shadow_time_or_realised_days(db, cfg, tmp_path):
    c = lcfg(cfg, shadow_min_weeks={"swing": 60, "invest_b1": 60, "invest_b2": 60})
    setup, _ = world(db, c)
    _, chall = pair(db, c, tmp_path, setup, "noise", "skilled")
    rep = CMP.compare(db, c, setup, chall, setup.calendar[300])
    assert rep["passed"] is False and rep["verdict"].startswith("NOT ENOUGH SHADOW DATA")
    early = CMP.compare(db, c, setup, chall, setup.calendar[104])
    assert early["passed"] is False and early["realised_days"] < 20


def test_a_challenger_trained_under_another_protocol_is_not_comparable(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    _, chall = pair(db, c, tmp_path, setup, "noise", "skilled", protocol=False)
    rep = CMP.compare(db, c, setup, chall, setup.calendar[300])
    assert rep["passed"] is False and rep["protocol_identical"] is False
    assert not [x for x in rep["criteria"] if x["name"].startswith("challenger trained under")][0]["ok"]


def test_each_extra_attempt_against_the_same_champion_lowers_the_level_and_the_bound(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    champ, chall = pair(db, c, tmp_path, setup, "noise", "skilled")
    first = CMP.compare(db, c, setup, chall, setup.calendar[300], write=False)
    assert first["attempts"] == 1 and first["alpha_adjusted"] == pytest.approx(0.10)
    for k in range(5):                                                        # five challengers were tried against this champion (this one included)
        log_experiment(db, c, "lifecycle:retrain:swing", params={"strategy": "swing", "n": k}, status="success")
    later = CMP.compare(db, c, setup, chall, setup.calendar[300], write=False)
    assert later["attempts"] == 5 and later["alpha_adjusted"] == pytest.approx(0.02)
    assert later["ic"]["lower_bound"] < first["ic"]["lower_bound"]           # a stricter level: the bootstrap bound moves down


def test_the_protected_period_can_only_be_compared_after_it_was_consumed(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    _, chall = pair(db, c, tmp_path, setup, "noise", "skilled")
    inside = c.model_copy(update={"lifecycle": c.lifecycle.model_copy(update={"protected_period": (str(setup.calendar[90].date()), str(setup.calendar[200].date()))})})
    with pytest.raises(REG.LifecycleError, match="protected held-out period"):
        CMP.compare(db, inside, setup, chall, setup.calendar[300])
    log_experiment(db, inside, "holdout:consumed", key="consumed", params={}, summary={}, status="success")
    assert CMP.compare(db, inside, setup, chall, setup.calendar[300], write=False)["window_start"]


# ---- retrain guards -------------------------------------------------------------------------------------------------------------------
def test_retrain_refuses_data_that_reaches_into_the_held_out_period_unless_consumed_knowingly(db, cfg, tmp_path):
    setup, _ = world(db, cfg)
    mk(db, cfg, tmp_path, "swing_lgbm_final", payload="champ", status="champion")
    with pytest.raises(RT.RetrainRefused, match="protected held-out period"):
        RT.retrain(db, cfg, "swing", "manual", as_of=pd.Timestamp("2026-01-05"), setup=setup)
    with session_scope(db) as s:
        assert s.scalar(select(func.count()).select_from(Experiment).where(Experiment.name == "holdout:consumed")) == 0
        assert s.scalar(select(func.count()).select_from(Model).where(Model.status == "shadow")) == 0


def test_retrain_needs_a_champion_a_known_trigger_and_at_most_one_challenger_at_a_time(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    with pytest.raises(RT.RetrainRefused, match="no champion"):
        RT.retrain(db, c, "swing", "manual", setup=setup)
    mk(db, c, tmp_path, "swing_lgbm_final", payload="champ", status="champion")
    with pytest.raises(RT.RetrainRefused, match="unknown trigger"):
        RT.retrain(db, c, "swing", "whenever", setup=setup)
    with pytest.raises(RT.RetrainRefused, match="unknown strategy"):
        RT.retrain(db, c, "crypto", "manual", setup=setup)
    mk(db, c, tmp_path, "swing_lgbm_final", payload="chall", status="shadow")
    with pytest.raises(RT.RetrainRefused, match="already has a challenger"):
        RT.retrain(db, c, "swing", "manual", setup=setup)


def test_due_reports_schedule_drift_and_a_new_feature_set(db, cfg, tmp_path):
    c = lcfg(cfg)
    mk(db, c, tmp_path, "swing_lgbm_final", payload="champ", status="champion", trained_until=date(2026, 1, 1))
    assert RT.due(db, c, "swing", pd.Timestamp("2026-03-01")) == []                                   # 2 months: not due
    kinds = {x["kind"] for x in RT.due(db, c, "swing", pd.Timestamp("2026-09-01"), {"swing": {"breaches": [{"kind": "drift", "message": "d"}, {"kind": "fill", "message": "f"}]}})}
    assert kinds == {"schedule", "drift"}                                                              # 8 months >= 6; fill rate is an alert, not a retrain reason
    newer = c.model_copy(update={"swing": c.swing.model_copy(update={"feature_set": "swing:2"})})
    assert "new_feature_set" in {x["kind"] for x in RT.due(db, newer, "swing", pd.Timestamp("2026-03-01"))}


def test_retrain_months_is_limited_to_three_to_twelve(cfg):
    with pytest.raises(ValueError):
        type(cfg.lifecycle)(retrain_months=2)
    with pytest.raises(ValueError):
        type(cfg.lifecycle)(retrain_months=13)
    assert type(cfg.lifecycle)(retrain_months=12).retrain_months == 12


# ---- PSI / KS and the drift alert ---------------------------------------------------------------------------------------------------------
def test_psi_and_ks_are_zero_for_the_same_distribution_and_large_for_a_shifted_one():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=5000)})
    prof = REF.profile(train, ["x"])
    same, moved = rng.normal(size=1000), rng.normal(1.5, 1.0, size=1000)
    assert REF.psi(prof["x"], same) < 0.05 and REF.ks(prof["x"], same) < 0.08
    assert REF.psi(prof["x"], moved) > 0.5 and REF.ks(prof["x"], moved) > 0.4
    assert np.isnan(REF.psi(prof["x"], same[:5]))                                                     # too few values: no figure, not a fake zero


def test_the_yardstick_of_the_training_periods_own_slices_keeps_a_normal_trending_feature_quiet():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2022-01-03", periods=400)
    level = np.cumsum(rng.normal(0, 0.15, 400))                                                       # a slowly wandering market level shared by all stocks
    frame = pd.DataFrame({"trade_date": np.repeat(days, 10), "x": np.repeat(level, 10) + rng.normal(0, 0.3, 4000)})
    prof = REF.profile(frame, ["x"])
    plain = REF.psi(prof["x"], frame["x"].to_numpy()[-600:])
    REF.window_baseline(frame, prof, 60, 10, 0.95)
    assert prof["x"]["psi_ref"] is not None and prof["x"]["ref_window"] == 60
    assert plain > 0.10 and prof["x"]["psi_ref"] >= 0.10                                              # even in ordinary times a 60-session slice sits far from the pooled history


def test_a_drifted_feature_set_raises_a_drift_alert_and_stores_the_metrics(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c, shift_last=3.0, shift_features=3)                                        # three of the four features move by 3 standard deviations in the last 60 sessions
    mk(db, c, tmp_path, "swing_lgbm_final", payload="champ", status="champion", trained_until=setup.calendar[200].date())
    res = MON.run_monitor(db, c, setup, setup.calendar[-6], strategies=("swing",))["swing"]
    kinds = {b["kind"] for b in res["breaches"]}
    assert "drift" in kinds and res["drift"]["n_alert"] >= 3 and set(res["drift"]["features_alert"]) >= {"f1", "f2", "f3"}
    with session_scope(db) as s:
        alert = s.scalars(select(Alert).where(Alert.category == "monitor_drift")).first()
        names = set(s.scalars(select(MonitoringMetric.metric_name)).all())
        stored = s.get(Model, REG.champion(db, "swing").id).params["reference"]
    assert alert is not None and alert.severity == "error" and "drifted" in alert.message
    assert {"psi:f1", "ks:f1", "psi_max", "drift_features_alert"} <= names and "psi_ref" in stored["f1"]
    assert "drift" in {x["kind"] for x in RT.due(db, c, "swing", setup.calendar[-6], {"swing": res})}                # ... and the monitor result makes a retrain DUE (not started)


def test_an_undisturbed_market_raises_no_drift_alert(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    mk(db, c, tmp_path, "swing_lgbm_final", payload="champ", status="champion", trained_until=setup.calendar[200].date())
    res = MON.run_monitor(db, c, setup, setup.calendar[-6], strategies=("swing",))["swing"]
    assert "drift" not in {b["kind"] for b in res["breaches"]}
    with session_scope(db) as s:
        assert s.scalar(select(func.count()).select_from(Alert).where(Alert.category == "monitor_drift")) == 0


def test_performance_is_not_computed_without_a_paper_record(db, cfg, tmp_path):
    c = lcfg(cfg)
    setup, _ = world(db, c)
    mk(db, c, tmp_path, "swing_lgbm_final", payload="champ", status="champion", trained_until=setup.calendar[200].date())
    perf = MON.run_monitor(db, c, setup, setup.calendar[-6], strategies=("swing",))["swing"]["performance"]
    assert "ic" not in perf and perf.get("skipped")                                                  # the research's held-out data is never used to judge a champion


# ---- what the closed recommendations say ----------------------------------------------------------------------------------------------
def trades(n, seed, informative):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.7, n)
    win = (rng.random(n) < (0.25 + 0.6 * (p - 0.2) if informative else 0.45)).astype(float)
    return pd.DataFrame({"recommendation_id": range(n), "as_of": pd.bdate_range("2026-01-01", periods=n), "exit_reason": np.where(win > 0, "target", "stop"), "sessions": rng.integers(2, 12, n),
                         "net_return": np.where(win > 0, 0.05, -0.03) + rng.normal(0, 0.005, n), "p_model": p, "p_display": p, "atr_pct": rng.uniform(0.01, 0.04, n), "rr_target2": 2.0,
                         "expected_hold": 6.0, "rsi14": rng.uniform(30, 70, n), "vol_spike": rng.uniform(0.5, 3, n), "q50_5d": rng.normal(0, 0.02, n), "regime_off": 0.0, "style": "pullback",
                         "hit": win, "win": win})


def test_meta_labeling_is_not_tried_on_a_small_sample_and_is_recorded_when_tried(db, cfg):
    small = OUT.meta_label(db, cfg, trades(120, 0, True))
    assert small["tried"] is False and "not tried" in small["verdict"]
    with session_scope(db) as s:
        assert s.scalar(select(func.count()).select_from(Experiment).where(Experiment.name == "lifecycle:meta_label")) == 0
    big = OUT.meta_label(db, cfg, trades(900, 1, True))
    assert big["tried"] is True and big["train"] > 0 and big["test"] > 0 and "gain" in big
    with session_scope(db) as s:
        assert s.scalar(select(func.count()).select_from(Experiment).where(Experiment.name == "lifecycle:meta_label")) == 1


def test_a_meta_filter_is_kept_only_when_it_beats_taking_every_trade_on_later_trades(db, cfg):
    useful = OUT.meta_label(db, cfg, trades(1500, 2, True), write=False)
    noise = OUT.meta_label(db, cfg, trades(1500, 3, False), write=False)
    assert useful["useful"] is True and useful["gain"]["lower_bound"] > 0
    assert noise["useful"] is False and "discarded" in noise["verdict"]


def test_recalibration_is_a_proposal_that_needs_enough_trades_and_a_later_improvement(cfg):
    assert OUT.recalibrate(trades(100, 4, True), cfg)["applied"] is False
    over = trades(1500, 5, True)
    over["p_model"] = (over["p_model"] * 0.5 + 0.55).clip(0, 1)                                       # systematically over-confident and compressed
    over["hit"] = over["win"]
    r = OUT.recalibrate(over, cfg)
    assert r["applied"] is False and r["recommend"] is True and r["brier"]["recalibrated"] < r["brier"]["stated"]


def test_assess_reports_calibration_only_with_enough_events(cfg):
    a = OUT.assess(trades(80, 6, True), cfg)
    assert a["trades"] == 80 and "no calibration figure" in a["calibration"]["note"]
    b = OUT.assess(trades(600, 6, True), cfg)
    assert b["calibration"]["n"] == 600 and 0 <= b["calibration"]["ece"] <= 1 and b["holding"]["expected_mean"] == 6.0
