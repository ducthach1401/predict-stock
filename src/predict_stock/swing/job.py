"""The `swing run` job: pre-registration -> tuning (once, first fold) -> walk-forward -> final model -> evaluation against the baselines -> report."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select

from predict_stock.backtest import metrics as M
from predict_stock.backtest.baselines import BASELINES
from predict_stock.backtest.job import noise_floor
from predict_stock.backtest.report import save_equity_artifacts, write_if_changed
from predict_stock.backtest.runner import Setup, clean, load_setup
from predict_stock.config import PROJECT_ROOT, AppConfig, SwingConfig
from predict_stock.db.models import ModelMetric, Universe
from predict_stock.db.session import session_scope
from predict_stock.features.registry import spec_hash
from predict_stock.runs import tracked_run
from predict_stock.swing import calibration as cal
from predict_stock.swing import metrics as SM
from predict_stock.swing.evaluate import decide, evaluate_baselines, evaluate_signals, run_signals
from predict_stock.swing.folds import plan_folds, split_fold, with_ends
from predict_stock.swing.registry import (
    build_details, find_experiment, log_experiment, register_model, save_predictions, trial_sink,
)
from predict_stock.swing.strategies import barrier_signals, topk_signals
from predict_stock.swing.tuning import run_study
from predict_stock.swing.walkforward import FoldResult, concat_predictions, fit_final, fit_fold

PRIMARY, SECONDARY = "swing_lgbm_topk", "swing_lgbm_barrier"
NOISE_SEEDS = 20


AMENDMENTS = [
    {"id": 1, "made_after": "first walk-forward run (run 87080007196f206b)",
     "found": "isotonic calibration on raw validation points output a probability of 1.0 in 3 of 11 folds (a few top-scored rows were all events)",
     "change": "isotonic is fitted on equal-count bins of at least swing.isotonic_min_bin (150) validation rows",
     "affects": "the calibrated probability, hence the secondary (barrier) strategy and the calibration report; NOT the rank score, hence not the primary strategy or the verdict",
     "first_run_result": "isotonic ECE 0.0725 vs raw 0.0694 (calibration did not help out of sample)"},
    {"id": 2, "made_after": "first walk-forward run (run 87080007196f206b)",
     "found": "the barrier strategy was flat (no position) for whole test windows: it required probability >= the TRAINING base rate but the calibrated probability is centred on the "
              "VALIDATION base rate, which was lower in 4 of 11 folds, so no name could pass",
     "change": "the entry threshold is a multiple of the base rate of the validation rows the calibrator was fitted on",
     "affects": "the secondary (barrier) strategy and its sensitivities only",
     "first_run_result": "barrier trades: net CAGR 4.4%, Sharpe 0.32, no position at all in the test windows of folds 5, 6, 9 and 10"},
]


def preregistration(cfg: AppConfig, setup: Setup) -> dict:
    """What is decided BEFORE any model is trained. Stored in ``experiments`` (``swing:preregistration``) and in docs/SWING.md."""
    sw = cfg.swing
    return {
        "primary_strategy": {"name": PRIMARY, "rule": "top-K by the rank score at the close, equal weight 1/K, full rebalance on the first session of each ISO week, "
                                                       "orders at the next open through the Phase 4 engine", "top_k": sw.strategy.top_k, "rebalance": sw.strategy.rebalance,
                             "probability_filter": None},
        "secondary_strategy": {"name": SECONDARY, "target_atr": sw.strategy.barrier_target_mult, "stop_atr": sw.strategy.barrier_stop_mult,
                               "time_stop": sw.strategy.barrier_max_hold, "max_positions": sw.strategy.max_positions, "min_prob_multiple": sw.strategy.min_prob_multiple},
        "decision_rule": sw.decision.model_dump(),
        "decision_text": "The held-out period is opened only if the primary strategy passes ALL of: net Sharpe above every portfolio baseline over the same window; "
                         "above the 95th percentile of random weekly portfolios; rank-IC t-statistic >= min_ic_tstat; positive net Sharpe in >= min_positive_fold_share of the "
                         "test windows; positive net Sharpe at stress_cost_multiple x costs. Otherwise the model is reported as not beating the baselines.",
        "tuning": {"protocol": "one Optuna study on the training/validation rows of the first fold, objective = mean daily rank IC of the ranking booster on the validation "
                               "rows, then frozen for every fold", **sw.optuna.model_dump()},
        "folds": sw.folds.model_dump(), "targets": {"rank": sw.rank_target, "return": sw.return_target, "event": f"{sw.event_label} == 1", "hold": sw.hold_column},
        "feature_set": sw.feature_set, "label_spec": sw.label_spec, "calibration": sw.calibration, "seed": sw.seed, "lgbm_defaults": sw.lgbm,
        "sensitivities": "K, entry-probability threshold, ATR target/stop and cost multiples are reported for information and never used to pick the configuration",
        "holdout": [str(setup.holdout.start.date()), str(setup.holdout.end.date())], "dataset_content_hash": setup.dataset_hashes["swing"],
    }


def _persist_fold(engine: Engine, cfg: AppConfig, res: FoldResult, setup: Setup, wf_exp: int, tune_key: str, universe_id: int, run_id: int) -> dict:
    sw = cfg.swing
    feats = res.bundle.features
    test = res.data.test
    pred = res.pred
    x = test[feats].to_numpy(float)
    shap_r, _ = res.bundle.contributions(test, "rank")
    shap_e, _ = res.bundle.contributions(test, "event")
    pred = pred.assign(details=build_details(pred, shap_r, shap_e, feats, x))
    mid, version, sha, reused = register_model(engine, cfg, res.bundle, f"swing_lgbm_f{res.fold.index:02d}", dataset_id=setup.dataset_ids.get("swing"), experiment_id=wf_exp,
                                               extra_params={"fold": res.fold.as_dict(), "purged_rows": res.data.purged, "unlabelled_rows": res.data.unlabelled, "fit_dropped_rows": res.data.fit_dropped,
                                                             "n_fit": len(res.data.fit), "n_val": len(res.data.val), "tuning_study": tune_key,
                                                             "train_range": [str(res.fold.train_start.date()), str(res.fold.train_end.date())]})
    n = save_predictions(engine, mid, universe_id, sw.horizon, pred, run_id)
    return {"fold": res.fold.index, "model_id": mid, "version": version, "sha256": sha, "reused_model": reused, "predictions": n}


def _universe_id(engine: Engine, code: str) -> int:
    with session_scope(engine) as s:
        return s.scalar(select(Universe.id).where(Universe.code == code))


def _calibration_report(preds: pd.DataFrame, cfg: SwingConfig) -> dict:
    ok = preds["tb_label"].notna()
    y = (preds.loc[ok, "tb_label"] == 1).astype(float).to_numpy()
    out = {"n": int(ok.sum()), "base_rate": float(y.mean()), "used": cfg.calibration, "variants": {}, "curves": {}}
    for name, col in (("raw", "proba_raw"), ("isotonic", "proba_isotonic"), ("platt", "proba_platt")):
        p = preds.loc[ok, col].to_numpy(float)
        out["variants"][name] = cal.summary(y, p)
        out["curves"][name] = cal.reliability(y, p, 10).round(6).to_dict("records")
    out["by_fold"] = []
    for f, g in preds[ok].groupby("fold"):
        sm = cal.summary((g["tb_label"] == 1).astype(float), g["proba"])
        out["by_fold"].append({"fold": int(f), "brier": sm["brier"], "ece": sm["ece"], "auc": cal.auc((g["tb_label"] == 1).astype(float), g["proba_raw"]),
                               "max_proba": float(g["proba"].max()), "observed": sm["base_rate"], "calibration_base_rate": float(g["calib_base_rate"].iloc[0])})
    # constants a forecaster could actually have used at the time (the rate of its own training rows / of its calibration rows), next to the pooled OOS rate
    for name, col in (("train_base_rate", "base_rate"), ("calibration_base_rate", "calib_base_rate")):
        c = preds.loc[ok, col].to_numpy(float)
        out.setdefault("constants", {})[name] = {"brier": cal.brier(y, c), "log_loss": cal.log_loss(y, c)}
    return out


def _ic_report(preds: pd.DataFrame, frame: pd.DataFrame, cfg: SwingConfig) -> dict:
    target = cfg.return_target
    h = int(target.rsplit("_", 1)[1])
    keyed = preds[["trade_date", "instrument_id", "score", target, "fold"]].merge(
        frame[["trade_date", "instrument_id", "ret_10", "zscore_20", "ret_5"]], on=["trade_date", "instrument_id"], how="left")
    keyed["mean_reversion"] = -keyed["zscore_20"]
    scores = {"swing_lgbm": "score", "mom_short (ret_10)": "ret_10", "mean_reversion (-zscore_20)": "mean_reversion", "ret_5": "ret_5"}
    out = {"target": target, "horizon": h, "scores": {}, "by_fold": []}
    daily = {}
    for name, col in scores.items():
        ic = SM.daily_rank_ic(keyed, col, target)
        daily[name] = ic
        out["scores"][name] = SM.ic_summary(ic, h)
    fold_of = keyed.groupby("trade_date")["fold"].first()
    for f, dates in fold_of.groupby(fold_of).groups.items():
        ic = daily["swing_lgbm"][daily["swing_lgbm"].index.isin(dates)]
        out["by_fold"].append({"fold": int(f), **SM.ic_summary(ic, h)})
    return out


def run_swing(engine: Engine, cfg: AppConfig, *, trials: int | None = None, noise_seeds: int = NOISE_SEEDS, write: bool = True) -> dict:
    if trials is not None:
        cfg = cfg.model_copy(update={"swing": cfg.swing.model_copy(update={"optuna": cfg.swing.optuna.model_copy(update={"trials": trials})})})
    sw = cfg.swing
    with tracked_run(engine, "swing_walkforward", cfg, {"trials": sw.optuna.trials, "noise_seeds": noise_seeds}) as (run_id, stats):
        setup = load_setup(engine, cfg)
        frame = with_ends(setup.frames["swing"])
        holdout = setup.holdout
        # ---- 0. pre-registration, before anything is trained -------------------------------------------------------------------
        prereg = preregistration(cfg, setup)
        prereg_key = spec_hash(prereg)
        log_experiment(engine, cfg, "swing:amendments", key=spec_hash(AMENDMENTS), params={"universe": setup.universe}, summary={"amendments": AMENDMENTS}, status="registered",
                       description="Mechanical corrections made after the first run had been looked at (they do not touch the primary strategy)", run_id=run_id)
        prereg_id = log_experiment(engine, cfg, "swing:preregistration", key=prereg_key, params={"universe": setup.universe}, summary=clean(prereg), status="registered",
                                   description="Decision rule, primary strategy and tuning protocol, fixed before any SWING model was trained", run_id=run_id)
        folds = plan_folds(frame, sw, holdout.start)
        if not folds:
            raise RuntimeError("no walk-forward fold fits inside the development data")
        # ---- 1. tuning: once, on the first fold's training/validation rows ---------------------------------------------------------
        first = split_fold(frame, folds[0], sw)
        tune_key = spec_hash({"dataset": setup.dataset_hashes["swing"], "fold0": folds[0].as_dict(), "optuna": sw.optuna.model_dump(), "lgbm": sw.lgbm, "n_estimators": sw.n_estimators,
                              "early_stopping": sw.early_stopping_rounds, "target": [sw.rank_target, sw.return_target], "seed": sw.seed, "v": 1})[:16]
        prior = find_experiment(engine, f"swing:optuna:{tune_key}", key=tune_key)
        if prior is not None and prior.status == "success":
            tuned, tune_summary = prior.summary["best_params"], {**prior.summary, "reused": True}
        else:
            res = run_study(first.fit, first.val, sw, trial_sink(engine, cfg, tune_key, run_id))
            tuned = res.best_params
            tune_summary = {"best_params": tuned, "best_validation_ic": res.best_value, "n_complete": res.n_complete, "n_failed": res.n_failed,
                            "trials": sw.optuna.trials, "fit_rows": len(first.fit), "validation_rows": len(first.val), "reused": False,
                            "fell_back_to_defaults": not tuned}
            log_experiment(engine, cfg, f"swing:optuna:{tune_key}", key=tune_key, params={"study": tune_key, "fold": folds[0].as_dict()}, summary=clean(tune_summary),
                           status="success" if tuned else "failed", description="Optuna study (first fold train/validation only)", run_id=run_id)
        # ---- 2. walk-forward ---------------------------------------------------------------------------------------------------------
        wf_key = spec_hash({"dataset": setup.dataset_hashes["swing"], "swing": sw.model_dump(), "tuned": tuned, "holdout": str(holdout.start.date()), "v": 1})[:16]
        wf_exp = log_experiment(engine, cfg, f"swing:walkforward:{wf_key}", key=wf_key, params={"universe": setup.universe, "tuned": tuned}, status="running",
                                description="SWING walk-forward (expanding folds, purged, embargoed)", run_id=run_id)
        universe_id = _universe_id(engine, setup.universe)
        results, registered = [], []
        for fold in folds:
            r = fit_fold(split_fold(frame, fold, sw), sw, tuned)
            results.append(r)
            registered.append(_persist_fold(engine, cfg, r, setup, wf_exp, tune_key, universe_id, run_id))
        bundle, fdata = fit_final(frame, sw, tuned, holdout.start, holdout.end)
        fid, fver, fsha, freused = register_model(engine, cfg, bundle, "swing_lgbm_final", dataset_id=setup.dataset_ids.get("swing"), experiment_id=wf_exp,
                                                  extra_params={"fold": fdata.fold.as_dict(), "purged_rows": fdata.purged, "n_fit": len(fdata.fit), "n_val": len(fdata.val),
                                                                "tuning_study": tune_key, "trained_on": "all development data (before the held-out period)"})
        final = {"model_id": fid, "version": fver, "sha256": fsha, "reused_model": freused, "train": [str(fdata.fold.train_start.date()), str(fdata.fold.train_end.date())]}
        # ---- 3. evaluation -----------------------------------------------------------------------------------------------------------
        preds = concat_predictions(results)
        windows = [f.as_dict() for f in folds]
        first_idx = int(setup.calendar.get_loc(folds[0].test_start))
        mults = list(setup.cfg.backtest.cost_multipliers)
        ic = _ic_report(preds, frame, sw)
        calib = _calibration_report(preds, sw)
        quant = SM.quantile_report(preds, sw.return_target, tuple(sw.quantiles))
        hold = SM.holding_report(preds, sw.hold_column)
        st = sw.strategy
        cal_idx = setup.calendar
        sig_a = topk_signals(preds, cal_idx, first_idx, setup.dev_end_idx, k=st.top_k, max_weight=cfg.backtest.max_weight, rebalance=st.rebalance)
        a_sum, a_eq = evaluate_signals(setup, sig_a, first_idx, windows, key=PRIMARY, title="SWING LightGBM, top-K weekly", multipliers=mults)
        barrier = dict(max_positions=st.max_positions, max_hold=st.barrier_max_hold)
        sig_b = barrier_signals(preds, cal_idx, first_idx, setup.dev_end_idx, target_mult=st.barrier_target_mult, stop_mult=st.barrier_stop_mult,
                                min_prob_multiple=st.min_prob_multiple, **barrier)
        b_sum, b_eq = evaluate_signals(setup, sig_b, first_idx, windows, key=SECONDARY, title="SWING LightGBM, barrier trades (ATR stop / target)", multipliers=mults,
                                       max_positions=st.max_positions)
        base = evaluate_baselines(setup, first_idx, windows, mults)
        noise = noise_floor(replace(setup, start_idx=first_idx), noise_seeds) if noise_seeds > 0 else None
        verdict = decide(a_sum, {k: v[0] for k, v in base.items()}, noise, ic["scores"]["swing_lgbm"], sw.decision)
        sens = _sensitivities(setup, preds, sw, windows, first_idx, [1.0, 2.0])
        # ---- 4. store ----------------------------------------------------------------------------------------------------------------
        equities = {PRIMARY: a_eq, SECONDARY: b_eq, **{k: v[1] for k, v in base.items()}}
        payload = clean({
            "run_key": wf_key, "preregistration": prereg, "amendments": AMENDMENTS, "preregistration_experiment": prereg_id, "tuning": {"key": tune_key, **tune_summary},
            "window": [str(setup.calendar[first_idx].date()), str(setup.calendar[setup.dev_end_idx].date())], "universe": setup.universe,
            "holdout": [str(holdout.start.date()), str(holdout.end.date())], "folds": [{**w, "purged_rows": r.data.purged, "unlabelled_rows": r.data.unlabelled, "n_fit": len(r.data.fit), "n_val": len(r.data.val),
                                                                                      "n_test": len(r.data.test), "best_iteration": r.bundle.best_iteration,
                                                                                      "base_rate": r.bundle.base_rate, "calibration_base_rate": r.bundle.calibration_base_rate}
                                                                                     for w, r in zip(windows, results)],
            "models": {"folds": registered, "final": final}, "ic": ic, "calibration": calib, "quantiles": quant, "holding": hold,
            "primary": a_sum, "secondary": b_sum, "baselines": {k: v[0] for k, v in base.items()}, "noise": noise, "sensitivity": sens, "decision": verdict,
            "feature_importance": _importance(bundle), "predictions": {"rows": int(len(preds)), "dates": [str(preds.trade_date.min().date()), str(preds.trade_date.max().date())]},
            "dataset_hash": setup.dataset_hashes["swing"], "seed": sw.seed})
        arts = save_equity_artifacts(cfg, f"swing_{wf_key}", equities)
        payload["artifacts"] = arts
        log_experiment(engine, cfg, f"swing:walkforward:{wf_key}", key=wf_key, params={"universe": setup.universe, "tuned": tuned}, summary=payload, status="success",
                       description="SWING walk-forward (expanding folds, purged, embargoed)", run_id=run_id)
        for name, s in ((PRIMARY, a_sum), (SECONDARY, b_sum)):
            log_experiment(engine, cfg, f"swing:strategy:{name}", key=f"{wf_key}:{name}", params={"run": wf_key, "window": s["window"]}, summary=s, status="success",
                           description=s["title"], run_id=run_id)
        _store_metrics(engine, registered, ic, calib, wf_exp)
        out_dir = PROJECT_ROOT / sw.artifacts_dir / "runs" / wf_key
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "payload.json").write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        stats.update(run_key=wf_key, experiment=wf_exp, passed=verdict["passed"], folds=len(folds))
    if write:
        from predict_stock.swing.report import write_report
        write_report(cfg, payload, equities, engine)
    return {"payload": payload, "equities": equities, "predictions": preds, "run_id": run_id}


def _importance(bundle) -> list[dict]:
    imp = bundle.boosters["rank"].feature_importance("gain")
    tot = float(imp.sum()) or 1.0
    return [{"feature": f, "gain_share": float(v) / tot} for f, v in sorted(zip(bundle.features, imp), key=lambda t: -t[1])]


def _sensitivities(setup: Setup, preds: pd.DataFrame, sw: SwingConfig, windows: list[dict], first_idx: int, mults: list[float]) -> dict:
    """Report-only. Nothing here selects the configuration."""
    st, cal_idx, bt = sw.strategy, setup.calendar, setup.cfg.backtest
    slim = lambda s: {"net": {k: s["net"].get(k) for k in ("cagr", "sharpe", "max_drawdown", "turnover_annual", "trades")},
                      "gross": {k: s["gross"].get(k) for k in ("cagr", "sharpe")}, "stress2": s["sensitivity"].get("2.0", {}).get("sharpe")}
    out = {"top_k": {}, "top_k_min_prob": {}, "barrier_atr": {}, "barrier_min_prob": {}, "cost_components": {}}
    for k in sw.sensitivity_k:
        sig = topk_signals(preds, cal_idx, first_idx, setup.dev_end_idx, k=k, max_weight=bt.max_weight, rebalance=st.rebalance)
        out["top_k"][str(k)] = slim(evaluate_signals(setup, sig, first_idx, windows, key=f"k{k}", title="", multipliers=mults)[0])
    for m in sw.sensitivity_min_prob:
        sig = topk_signals(preds, cal_idx, first_idx, setup.dev_end_idx, k=st.top_k, max_weight=bt.max_weight, rebalance=st.rebalance, min_prob_multiple=m)
        out["top_k_min_prob"][str(m)] = slim(evaluate_signals(setup, sig, first_idx, windows, key=f"p{m}", title="", multipliers=mults)[0])
    sig_a = topk_signals(preds, cal_idx, first_idx, setup.dev_end_idx, k=st.top_k, max_weight=bt.max_weight, rebalance=st.rebalance)
    r0 = setup.rules
    for label, variant in (("slippage x0", r0.with_costs(slippage=0.0)), ("slippage x2", r0.with_costs(slippage=2 * r0.slippage_rate)), ("slippage x3", r0.with_costs(slippage=3 * r0.slippage_rate)),
                           ("fee and tax x0", r0.with_costs(fee=0.0, tax=0.0)), ("fee and tax x2", r0.with_costs(fee=2 * r0.fee_rate, tax=2 * r0.sell_tax_rate))):
        local = replace(setup, rules=variant)                              # only one component changes; the other two stay at the configured values
        r = run_signals(local, sig_a, first_idx, 1.0)
        m = M.compute_metrics(r.equity, trips=r.round_trips, fills=r.fills, exposure=r.exposure, rf_annual=bt.risk_free_annual)
        out["cost_components"][label] = clean({k: m.get(k) for k in ("cagr", "sharpe", "max_drawdown", "turnover_annual")})
    for tgt, stp in sw.sensitivity_atr:
        sig = barrier_signals(preds, cal_idx, first_idx, setup.dev_end_idx, max_positions=st.max_positions, max_hold=st.barrier_max_hold, target_mult=tgt, stop_mult=stp,
                              min_prob_multiple=st.min_prob_multiple)
        out["barrier_atr"][f"{tgt:g}/{stp:g}"] = slim(evaluate_signals(setup, sig, first_idx, windows, key="atr", title="", multipliers=mults, max_positions=st.max_positions)[0])
    for m in sw.sensitivity_min_prob:
        sig = barrier_signals(preds, cal_idx, first_idx, setup.dev_end_idx, max_positions=st.max_positions, max_hold=st.barrier_max_hold, target_mult=st.barrier_target_mult,
                              stop_mult=st.barrier_stop_mult, min_prob_multiple=m)
        out["barrier_min_prob"][str(m)] = slim(evaluate_signals(setup, sig, first_idx, windows, key="bp", title="", multipliers=mults, max_positions=st.max_positions)[0])
    return out


def _store_metrics(engine: Engine, registered: list[dict], ic: dict, calib: dict, exp_id: int) -> None:
    """Per-fold metrics of each fold model in ``model_metrics`` (replaced on a re-run of the same experiment)."""
    by_ic = {r["fold"]: r for r in ic["by_fold"]}
    by_cal = {r["fold"]: r for r in calib["by_fold"]}
    with session_scope(engine) as s:
        for reg in registered:
            f = reg["fold"]
            s.execute(delete(ModelMetric).where(ModelMetric.model_id == reg["model_id"], ModelMetric.experiment_id == exp_id))
            for name, val in (("rank_ic_mean", by_ic.get(f, {}).get("mean")), ("rank_ic_t_stat", by_ic.get(f, {}).get("t_stat")),
                              ("brier_calibrated", by_cal.get(f, {}).get("brier")), ("ece_calibrated", by_cal.get(f, {}).get("ece"))):
                if val is not None and np.isfinite(val):
                    s.add(ModelMetric(model_id=reg["model_id"], experiment_id=exp_id, split=f"fold_{f}", metric_name=name, value=float(val)))


# ---- the held-out period: only after a PASS, once, together with the baselines --------------------------------------------------------
class HoldoutClosed(RuntimeError):
    """The pre-registered criteria were not met (or no matching development run exists): the held-out period must not be opened."""


def run_swing_oos(engine: Engine, cfg: AppConfig) -> dict:
    from predict_stock.backtest.job import run_holdout_once
    from predict_stock.db.models import Model
    from predict_stock.swing.model import load_bundle
    from predict_stock.swing.report import latest_payload_path
    from predict_stock.swing.walkforward import predict_rows
    sw = cfg.swing
    try:
        payload = json.loads(latest_payload_path(cfg).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HoldoutClosed(f"no stored swing run to base the decision on ({exc}); run `swing run` first") from exc
    setup = load_setup(engine, cfg)
    if payload["preregistration"] != clean(preregistration(cfg, setup)):
        raise HoldoutClosed("the stored development run was made with a different pre-registration (configuration or data changed): run `swing run` again")
    if not payload["decision"]["passed"]:
        failed = [c["name"] for c in payload["decision"]["criteria"] if not c["ok"]]
        raise HoldoutClosed("the pre-registered criteria were not met on the development walk-forward, so the held-out period stays closed. Failed: " + "; ".join(failed))
    with session_scope(engine) as s:
        row = s.scalars(select(Model).where(Model.name == "swing_lgbm_final").order_by(Model.version.desc())).first()
        path, sha = PROJECT_ROOT / row.artifact_path, row.artifact_sha256
    bundle = load_bundle(path, sha)
    frame = setup.frames["swing"]
    d = pd.to_datetime(frame["trade_date"])
    rows = frame[(d >= setup.holdout.start).to_numpy()]
    preds = predict_rows(bundle, rows)
    first = int(setup.calendar.get_loc(setup.holdout.start))
    last = len(setup.calendar) - 1
    st = sw.strategy
    sigs = {PRIMARY: (topk_signals(preds, setup.calendar, first, last, k=st.top_k, max_weight=cfg.backtest.max_weight, rebalance=st.rebalance), None),
            SECONDARY: (barrier_signals(preds, setup.calendar, first, last, max_positions=st.max_positions, max_hold=st.barrier_max_hold, target_mult=st.barrier_target_mult,
                                        stop_mult=st.barrier_stop_mult, min_prob_multiple=st.min_prob_multiple), st.max_positions)}
    return run_holdout_once(engine, cfg, extra_candidates=list(sigs), extra_signals=sigs, setup=setup)
