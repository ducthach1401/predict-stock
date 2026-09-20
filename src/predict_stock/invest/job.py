"""The `invest run` job for B1 and B2: pre-registration -> hyper-parameter grid (first fold) -> walk-forward -> final models -> evaluation against the
baselines with bootstrap intervals -> variants -> report -> database. The held-out period is never touched."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select

from predict_stock.backtest.baselines import rebalance_sessions
from predict_stock.backtest.report import save_equity_artifacts
from predict_stock.backtest.runner import Setup, clean, load_setup
from predict_stock.config import PROJECT_ROOT, AppConfig, InvestConfig, InvestPreset
from predict_stock.db.models import Model, ModelMetric, Universe
from predict_stock.db.session import session_scope
from predict_stock.features.registry import spec_hash
from predict_stock.invest import evaluate as E
from predict_stock.invest import stats as S
from predict_stock.invest import walkforward as W
from predict_stock.invest.strategy import invest_signals, regime_off
from predict_stock.runs import tracked_run
from predict_stock.swing import metrics as SM
from predict_stock.swing.evaluate import evaluate_baselines, evaluate_signals
from predict_stock.swing.registry import find_experiment, log_experiment, register_artifact, save_predictions

ALGO = "invest_candidate"
NOISE_SEEDS = 20


def preregistration(cfg: AppConfig, setup: Setup) -> dict:
    iv = cfg.invest
    return {
        "presets": {k: p.model_dump() for k, p in iv.presets.items()},
        "candidates": iv.candidates, "factors": [[c, s] for c, s in iv.factors],
        "learned": {"ridge_alphas": iv.ridge_alphas, "enet_alphas": iv.enet_alphas, "enet_l1_ratios": iv.enet_l1_ratios, "lgbm": iv.lgbm, "n_estimators": iv.n_estimators,
                    "grid_protocol": "one grid on the first fold's train/validation rows, objective = mean daily rank IC on the validation rows, best point per family frozen"},
        "portfolio": {"top_k": iv.top_k, "weighting": iv.weighting, "max_weight": iv.max_weight, "tranches": {k: p.tranches for k, p in iv.presets.items()},
                      "rebalance_threshold": setup.cfg.backtest.rebalance_threshold, "regime_filter": "variant only", "dca": "variant only"},
        "folds": iv.folds.model_dump(), "embargo": "= the preset's label horizon", "bootstrap": iv.bootstrap.model_dump(),
        "decision_rule": iv.decision.model_dump(),
        "decision_text": "Per preset the candidate with the best net Sharpe is judged. The held-out period is opened only if a preset passes ALL of: net Sharpe above every portfolio "
                         "baseline over the same window; White's reality check over the candidates (Sharpe vs equal-weight) p <= reality_check_p; rank-IC t-statistic >= min_ic_tstat; "
                         "net return ahead of equal-weight in >= min_rolling_share of rolling 1-year windows; positive net Sharpe at stress_cost_multiple x costs. Otherwise the "
                         "result is reported as not beating the baselines and nothing is tuned further.",
        "sensitivities": "K, weighting scheme, tranches, regime filter and DCA are reported for information and never used to choose the configuration",
        "holdout": [str(setup.holdout.start.date()), str(setup.holdout.end.date())], "dataset_content_hash": setup.dataset_hashes["invest"],
    }


def _universe_id(engine: Engine, code: str) -> int:
    with session_scope(engine) as s:
        return s.scalar(select(Universe.id).where(Universe.code == code))


def _trial_sink(engine: Engine, cfg: AppConfig, study: str, run_id: int | None):
    def sink(rec: dict) -> None:
        status = {"COMPLETE": "success", "FAIL": "failed"}.get(rec["state"], rec["state"].lower())
        log_experiment(engine, cfg, f"invest:grid:{study}:{rec['number']:03d}", key=f"{study}:{rec['number']}", status=status, run_id=run_id,
                       description=f"{rec['kind']} {rec['hyper']} ({rec['state']})", params={"study": study, "family": rec["kind"], "hyper": rec["hyper"]},
                       summary={"validation_rank_ic": rec["value"], "state": rec["state"], "error": rec["error"]})
    return sink


def _persist_fold(engine: Engine, cfg: AppConfig, key: str, preset: InvestPreset, run: W.FoldRun, sched_dates: set, setup: Setup, exp_id: int, tune_key: str,
                  universe_id: int, run_id: int) -> list[dict]:
    iv = cfg.invest
    out = []
    test = run.data.test
    for name, model in run.cands.models.items():
        raw = run.cands.artifact(name)
        mid, ver, sha, reused = register_artifact(
            engine, cfg, raw, f"invest_{key}_{name}_f{run.fold.index:02d}", algo=ALGO, feature_set=iv.feature_set, label_spec=iv.label_spec, dataset_id=setup.dataset_ids.get("invest"),
            experiment_id=exp_id, seed=iv.seed, artifact_dir=PROJECT_ROOT / iv.artifacts_dir,
            params={"preset": key, "candidate": name, "horizon": preset.horizon, "fold": run.fold.as_dict(), "n_fit": len(run.data.fit), "n_val": len(run.data.val),
                    "tuning_study": tune_key, "train_range": [str(run.fold.train_start.date()), str(run.fold.train_end.date())], "hypers": run.cands.meta["hypers"].get(name),
                    "inputs": model.inputs()})
        pred = run.preds[name]
        keep = pred["trade_date"].isin(sched_dates)                        # decisions are taken on the rebalance dates: only those are stored
        p = pred[keep]
        rows = test.loc[p.index]
        p = p.assign(proba=np.nan, details=E.build_details(rows, p, model, preset, iv))
        n = save_predictions(engine, mid, universe_id, preset.horizon, p, run_id)
        out.append({"candidate": name, "fold": run.fold.index, "model_id": mid, "version": ver, "sha256": sha, "reused": reused, "predictions": n})
    return out


def _by_fold_ic(ev: pd.DataFrame, ret: str, horizon: int) -> list[dict]:
    out = []
    for f, g in ev.groupby("fold"):
        out.append({"fold": int(f), **SM.ic_summary(SM.daily_rank_ic(g, "score", ret), horizon)})
    return out


def run_preset(engine: Engine, cfg: AppConfig, setup: Setup, key: str, *, run_id: int, noise_seeds: int) -> tuple[dict, dict]:
    iv, bt = cfg.invest, cfg.backtest
    preset = iv.presets[key]
    rank, ret, _ = W.targets(preset)
    holdout = setup.holdout
    frame = W.prepare(setup.frames["invest"], preset)
    folds = W.plan_folds(frame, iv, preset, holdout.start)
    if not folds:
        raise RuntimeError(f"{key}: no walk-forward fold fits inside the development data")
    windows = [f.as_dict() for f in folds]
    # the evaluation window ends with the last complete test window: after it there are no predictions, and a portfolio left unrebalanced would be a distortion
    setup = replace(setup, dev_end_idx=int(setup.calendar.get_loc(folds[-1].test_end)))
    # ---- grid on the first fold (frozen afterwards) -----------------------------------------------------------------------------------
    first = W.split(frame, folds[0], iv)
    tune_key = spec_hash({"dataset": setup.dataset_hashes["invest"], "preset": preset.model_dump(), "fold0": folds[0].as_dict(), "grid": W.grid(iv), "seed": iv.seed, "v": 1})[:16]
    prior = find_experiment(engine, f"invest:grid:{tune_key}", key=tune_key)
    if prior is not None and prior.status == "success":
        hypers = prior.summary["hypers"]
    else:
        hypers = W.tune(first, iv, preset, _trial_sink(engine, cfg, tune_key, run_id))
        log_experiment(engine, cfg, f"invest:grid:{tune_key}", key=tune_key, params={"preset": key, "fold": folds[0].as_dict()}, summary=clean({"hypers": hypers}), status="success",
                       description=f"{key}: hyper-parameter grid on the first fold", run_id=run_id)
    # ---- walk-forward ---------------------------------------------------------------------------------------------------------------------
    wf_key = spec_hash({"dataset": setup.dataset_hashes["invest"], "invest": iv.model_dump(), "preset": key, "hypers": hypers, "holdout": str(holdout.start.date()), "v": 1})[:16]
    exp_id = log_experiment(engine, cfg, f"invest:{key}:walkforward:{wf_key}", key=wf_key, params={"preset": key, "hypers": hypers}, status="running",
                            description=f"{preset.label}: walk-forward", run_id=run_id)
    universe_id = _universe_id(engine, setup.universe)
    first_idx = int(setup.calendar.get_loc(folds[0].test_start))
    sched_dates = {setup.calendar[i] for i in rebalance_sessions(setup.calendar, preset.rebalance, first_idx, setup.dev_end_idx)}
    registered = []
    runs = W.run_walk_forward(frame, iv, key, preset, hypers, holdout.start,
                              on_fold=lambda r: registered.extend(_persist_fold(engine, cfg, key, preset, r, sched_dates, setup, exp_id, tune_key, universe_id, run_id)))
    finals = {}
    fc, fd = W.fit_final(frame, iv, key, preset, hypers, holdout.start, holdout.end)
    for name in fc.models:
        mid, ver, sha, reused = register_artifact(
            engine, cfg, fc.artifact(name), f"invest_{key}_{name}_final", algo=ALGO, feature_set=iv.feature_set, label_spec=iv.label_spec, dataset_id=setup.dataset_ids.get("invest"),
            experiment_id=exp_id, seed=iv.seed, artifact_dir=PROJECT_ROOT / iv.artifacts_dir,
            params={"preset": key, "candidate": name, "horizon": preset.horizon, "fold": fd.fold.as_dict(), "n_fit": len(fd.fit), "n_val": len(fd.val), "hypers": hypers.get(name),
                    "trained_on": "all development data (before the held-out period)", "inputs": fc.models[name].inputs()})
        finals[name] = {"model_id": mid, "version": ver, "sha256": sha, "reused": reused}
    # ---- evaluation -------------------------------------------------------------------------------------------------------------------------
    mults = list(bt.cost_multipliers)
    calendar = setup.calendar
    returns = setup.data.close.pct_change()
    ev = {c: W.evaluable(W.concat(runs, c), preset, holdout.start) for c in iv.candidates}
    ic = {c: SM.ic_summary(SM.daily_rank_ic(ev[c], "score", ret), preset.horizon) for c in iv.candidates}
    ic_folds = {c: _by_fold_ic(ev[c], ret, preset.horizon) for c in iv.candidates}
    quant = SM.quantile_report(ev[iv.candidates[0]], ret, tuple(iv.quantiles))
    sig_kw = dict(rebalance=preset.rebalance, k=iv.top_k, weighting=iv.weighting, cap=iv.max_weight, cov_window=iv.cov_window, tranches=preset.tranches, spacing=preset.tranche_spacing)
    models, eqs = {}, {}
    for c in iv.candidates:
        pr = W.concat(runs, c)
        sig = invest_signals(pr, calendar, returns, first_idx, setup.dev_end_idx, **sig_kw)
        models[c], eqs[c] = evaluate_signals(setup, sig, first_idx, windows, key=f"invest_{key}_{c}", title=f"{preset.label}: {c}", multipliers=mults)
    base = evaluate_baselines(setup, first_idx, windows, mults)
    baselines = {k: v[0] for k, v in base.items()}
    noise = E.random_noise(setup, first_idx, preset.rebalance, iv.top_k, noise_seeds) if noise_seeds > 0 else None
    # ---- small-sample statistics ---------------------------------------------------------------------------------------------------------------
    bs = iv.bootstrap
    net = {c: eqs[c]["net"] for c in iv.candidates}
    bench_eq = base["equal_weight"][1]["net"]
    rets = {c: S.daily_returns(net[c]) for c in iv.candidates}
    kw = dict(resamples=bs.resamples, block=bs.block, level=bs.level, seed=bs.seed)
    cis = {c: S.paired_bootstrap(rets[c], S.daily_returns(bench_eq), **kw) for c in iv.candidates}
    rc = S.reality_check(rets, S.daily_returns(bench_eq), resamples=bs.resamples, block=bs.block, seed=bs.seed)
    best = max(iv.candidates, key=lambda c: models[c]["net"]["sharpe"] if models[c]["net"]["sharpe"] is not None else -9)
    bench_cis = {b: S.paired_bootstrap(rets[best], S.daily_returns(base[b][1]["net"]), **kw) for b in ("mom_long", "bh_vn30", "bh_vnindex") if b in base}
    rolling = {str(w): S.rolling_outperformance(net[best], bench_eq, w) for w in iv.rolling_windows}
    roll_all = {c: {str(w): S.rolling_outperformance(net[c], bench_eq, w) for w in iv.rolling_windows} for c in iv.candidates}
    roll_bench = {b: {str(w): S.rolling_outperformance(net[best], base[b][1]["net"], w) for w in iv.rolling_windows} for b in ("mom_long", "bh_vn30") if b in base}
    verdict = E.decide(best, models, baselines, noise, ic, rc, rolling, iv.decision)
    # ---- variants of the best candidate (information only) --------------------------------------------------------------------------------------
    variants = _variants(setup, best, W.concat(runs, best), first_idx, windows, mults, iv, preset, sig_kw, returns, base)
    dca = {"contribution": iv.dca.monthly_contribution, "strategy": S.dca(net[best], iv.dca.monthly_contribution, iv.dca.initial),
           "equal_weight": S.dca(bench_eq, iv.dca.monthly_contribution, iv.dca.initial),
           **({"bh_vn30": S.dca(base["bh_vn30"][1]["net"], iv.dca.monthly_contribution, iv.dca.initial)} if "bh_vn30" in base else {})}
    # ---- thesis-break information and fundamentals -------------------------------------------------------------------------------------------------
    thesis = _thesis(ev[best], sched_dates, preset, iv, ret)
    fundamentals = E.fundamentals_ablation(setup.frames["invest"], iv, key, preset, hypers, holdout.start)
    # ---- store ----------------------------------------------------------------------------------------------------------------------------------
    payload = clean({"preset": key, "label": preset.label, "run_key": wf_key, "window": [str(calendar[first_idx].date()), str(calendar[setup.dev_end_idx].date())],
                     "universe": setup.universe, "holdout": [str(holdout.start.date()), str(holdout.end.date())], "config": preset.model_dump(), "regime_config": iv.regime.model_dump(),
                     "folds": [{**w, "n_fit": len(r.data.fit), "n_val": len(r.data.val), "n_test": len(r.data.test), "purged_rows": r.data.purged, "unlabelled_rows": r.data.unlabelled,
                                "lgbm_trees": r.cands.models["lgbm"].best_iteration if "lgbm" in r.cands.models else None,
                                "scenario_trees": r.cands.scenario.best_iteration}
                               for w, r in zip(windows, runs)],
                     "hypers": hypers, "models": {"folds": registered, "final": finals}, "ic": ic, "ic_by_fold": ic_folds, "quantiles": quant, "candidates": models,
                     "best": best, "baselines": baselines, "noise": noise, "bootstrap": {"config": bs.model_dump(), "vs_equal_weight": cis, "best_vs_others": bench_cis,
                                                                                         "reality_check": rc},
                     "rolling": {"best": rolling, "all": roll_all, "best_vs": roll_bench}, "decision": verdict, "variants": variants, "dca": dca, "thesis": thesis,
                     "fundamentals": fundamentals, "dataset_hash": setup.dataset_hashes["invest"], "seed": iv.seed, "tuning_study": tune_key})
    equities = {**{f"invest_{key}_{c}": e for c, e in eqs.items()}, **{k: v[1] for k, v in base.items()}}
    payload["artifacts"] = save_equity_artifacts(cfg, f"invest_{key}_{wf_key}", equities)
    log_experiment(engine, cfg, f"invest:{key}:walkforward:{wf_key}", key=wf_key, params={"preset": key, "hypers": hypers}, summary=payload, status="success",
                   description=f"{preset.label}: walk-forward", run_id=run_id)
    for c in iv.candidates:
        log_experiment(engine, cfg, f"invest:strategy:{key}:{c}", key=f"{wf_key}:{c}", params={"run": wf_key, "window": models[c]["window"]}, summary=models[c], status="success",
                       description=models[c]["title"], run_id=run_id)
    _store_metrics(engine, registered, ic_folds, exp_id)
    return payload, equities


def _variants(setup: Setup, best: str, preds: pd.DataFrame, first_idx: int, windows: list[dict], mults: list[float], iv: InvestConfig, preset: InvestPreset, sig_kw: dict,
              returns: pd.DataFrame, base: dict) -> dict:
    """Report-only variants of the best candidate: K, weighting, tranches, regime filter."""
    cal = setup.calendar
    slim = lambda s: {"net": {k: s["net"].get(k) for k in ("cagr", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_annual")}, "gross": {k: s["gross"].get(k) for k in ("cagr", "sharpe")},
                      "stress2": s["sensitivity"].get("2.0", {}).get("sharpe"), "exposure": s.get("exposure")}
    out: dict[str, dict] = {"top_k": {}, "weighting": {}, "tranches": {}, "regime": {}}

    def run(label: str, **over) -> dict:
        kw = {**sig_kw, **over}
        sig = invest_signals(preds, cal, returns, first_idx, setup.dev_end_idx, **kw)
        return slim(evaluate_signals(setup, sig, first_idx, windows, key=label, title="", multipliers=mults)[0])
    for k in iv.sensitivity_k:
        out["top_k"][str(k)] = run(f"k{k}", k=k)
    for w in iv.sensitivity_weighting:
        out["weighting"][w] = run(f"w_{w}", weighting=w)
    for t in iv.sensitivity_tranches:
        out["tranches"][str(t)] = run(f"t{t}", tranches=t)
    off = regime_off(setup.index_close, cal, iv.regime.symbol, iv.regime.fallback, iv.regime.sma_window)
    out["regime"]["filter_on"] = run("regime", off=off, equity_share=iv.regime.equity_share)
    out["regime"]["share_of_sessions_risk_off"] = float(off.iloc[first_idx: setup.dev_end_idx + 1].mean())
    return out


def _thesis(ev: pd.DataFrame, sched_dates: set, preset: InvestPreset, iv: InvestConfig, ret: str) -> dict:
    """Among the names the best candidate would hold at each rebalance (its top-K), how much the thesis-break flags tell about the forward return."""
    sel = ev[ev["trade_date"].isin(sched_dates) & (ev["rank_in_universe"] <= iv.top_k)]
    if sel.empty:
        return {}
    flags = S.thesis_flags(sel, preset.drawdown_break, iv.thesis_rs_floor)
    return {"n_holdings": int(len(sel)), "information": S.thesis_information(sel, flags, ret), "limits": {"rs_rank_below": iv.thesis_rs_floor, "drawdown_beyond": preset.drawdown_break}}


def _store_metrics(engine: Engine, registered: list[dict], ic_folds: dict, exp_id: int) -> None:
    with session_scope(engine) as s:
        for reg in registered:
            f = next((x for x in ic_folds[reg["candidate"]] if x["fold"] == reg["fold"]), None)
            s.execute(delete(ModelMetric).where(ModelMetric.model_id == reg["model_id"], ModelMetric.experiment_id == exp_id))
            if f:
                for name in ("mean", "t_stat"):
                    v = f.get(name)
                    if v is not None and np.isfinite(v):
                        s.add(ModelMetric(model_id=reg["model_id"], experiment_id=exp_id, split=f"fold_{reg['fold']}", metric_name=f"rank_ic_{name}", value=float(v)))


def run_invest(engine: Engine, cfg: AppConfig, *, presets: list[str] | None = None, noise_seeds: int = NOISE_SEEDS, write: bool = True) -> dict:
    keys = presets or list(cfg.invest.presets)
    with tracked_run(engine, "invest_walkforward", cfg, {"presets": keys, "noise_seeds": noise_seeds}) as (run_id, stats):
        setup = load_setup(engine, cfg)
        prereg = preregistration(cfg, setup)
        prereg_id = log_experiment(engine, cfg, "invest:preregistration", key=spec_hash(prereg), params={"universe": setup.universe}, summary=clean(prereg), status="registered",
                                   description="Presets, candidates, decision rule and statistics fixed before any INVEST model was trained", run_id=run_id)
        payloads, equities = {}, {}
        for key in keys:
            payloads[key], equities[key] = run_preset(engine, cfg, setup, key, run_id=run_id, noise_seeds=noise_seeds)
            payloads[key]["preregistration"] = clean(prereg)
            payloads[key]["preregistration_experiment"] = prereg_id
            out_dir = PROJECT_ROOT / cfg.invest.artifacts_dir / "runs" / f"invest_{key}_{payloads[key]['run_key']}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "payload.json").write_text(json.dumps(payloads[key], indent=1, sort_keys=True), encoding="utf-8")
        stats.update(presets=keys, passed={k: p["decision"]["passed"] for k, p in payloads.items()})
    if write:
        from predict_stock.invest.report import write_reports
        write_reports(cfg, payloads, equities, engine)
    return {"payloads": payloads, "equities": equities, "run_id": run_id}


# ---- the held-out period: only after a PASS of at least one preset, once, with the baselines ----------------------------------------------------
class HoldoutClosed(RuntimeError):
    """No preset met the pre-registered criteria (or no matching development run exists): the held-out period must not be opened."""


def run_invest_oos(engine: Engine, cfg: AppConfig) -> dict:
    from predict_stock.backtest.job import run_holdout_once
    from predict_stock.invest.models import load_candidate
    from predict_stock.invest.report import latest_payloads
    iv = cfg.invest
    try:
        payloads = latest_payloads(cfg)
    except FileNotFoundError as exc:
        raise HoldoutClosed(f"no stored invest run to base the decision on ({exc})") from exc
    setup = load_setup(engine, cfg)
    prereg = clean(preregistration(cfg, setup))
    stale = [k for k, p in payloads.items() if p["preregistration"] != prereg]
    if stale:
        raise HoldoutClosed(f"the stored run(s) {stale} were made under a different pre-registration (configuration or data changed): run `invest run` again")
    passing = [k for k, p in payloads.items() if p["decision"]["passed"]]
    if not passing:
        why = {k: [c["name"] for c in p["decision"]["criteria"] if not c["ok"]] for k, p in payloads.items()}
        raise HoldoutClosed("no preset met the pre-registered criteria on the development walk-forward, so the held-out period stays closed. Failed: " + json.dumps(why))
    frame = setup.frames["invest"]
    start = int(setup.calendar.get_loc(setup.holdout.start))
    last = len(setup.calendar) - 1
    returns = setup.data.close.pct_change()
    extra = {}
    for key in passing:
        preset, best = iv.presets[key], payloads[key]["best"]
        with session_scope(engine) as s:
            row = s.scalars(select(Model).where(Model.name == f"invest_{key}_{best}_final").order_by(Model.version.desc())).first()
            path, sha = PROJECT_ROOT / row.artifact_path, row.artifact_sha256
        raw = path.read_bytes()
        from predict_stock.invest.models import sha256
        if sha256(raw) != sha:
            raise HoldoutClosed(f"{path}: sha256 does not match the recorded value")
        model, _, _ = load_candidate(raw)
        rows = frame[(pd.to_datetime(frame["trade_date"]) >= setup.holdout.start).to_numpy()]
        preds = rows[["trade_date", "instrument_id"]].assign(score=model.predict(rows))
        sig_kw = dict(rebalance=preset.rebalance, k=iv.top_k, weighting=iv.weighting, cap=iv.max_weight, cov_window=iv.cov_window, tranches=preset.tranches, spacing=preset.tranche_spacing)
        extra[f"invest_{key}_{best}"] = (invest_signals(preds, setup.calendar, returns, start, last, **sig_kw), None)
    return run_holdout_once(engine, cfg, extra_candidates=list(extra), extra_signals=extra, setup=setup)
