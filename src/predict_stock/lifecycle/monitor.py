"""Monitoring of the models in service (champions): drift of every feature against the training reference, rolling IC, hit rate, calibration, paper fill rate, holding-time deviation,
universe turnover. Everything is written to `monitoring_metrics` (one row per model, metric and day; a rerun replaces it) and a threshold breach raises an alert.

Two guards keep the research honest:
* performance figures (IC, hit rate, calibration) are computed ONLY on sessions of the paper record (or later than the champion's training cut-off and the held-out period): the years
  between a model's training cut-off and the paper start are the research's held-out period, which no model may be evaluated on;
* drift (PSI / KS) uses features only, so it may look at the most recent sessions at any time."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, select

from predict_stock.backtest.runner import Setup
from predict_stock.config import AppConfig
from predict_stock.db.models import (
    Model, MonitoringMetric, PaperOrder, Prediction, Recommendation, RecommendationOutcome, UniverseChangeLog,
)
from predict_stock.db.session import session_scope
from predict_stock.lifecycle import reference as REF
from predict_stock.lifecycle import registry as REG
from predict_stock.paper.alerts import raise_alert
from predict_stock.swing import calibration as cal
from predict_stock.swing import metrics as SM

STRAT_TO_KEY = {"swing": "swing", "invest_b1": "invest", "invest_b2": "invest"}
CARD_STRATEGY = {"swing": "SWING", "invest_b1": "INVEST_B1", "invest_b2": "INVEST_B2"}


def feature_list(strategy: str, frame: pd.DataFrame) -> list[str]:
    from predict_stock.swing.model import feature_columns
    cols = feature_columns(frame)
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def ensure_reference(engine: Engine, cfg: AppConfig, setup: Setup, ref: REG.ModelRef) -> dict | None:
    """The training reference of a model: stored in its params, computed from the dataset rows up to its training cut-off (and stored) when missing. It carries, per feature, the
    yardstick of the training period's own 60-session slices (see ``reference.window_baseline``); a reference stored without it (or for another window) is completed."""
    th = cfg.lifecycle.thresholds
    with session_scope(engine) as s:
        m = s.get(Model, ref.id)
        stored = (m.params or {}).get("reference")
    if stored and all(p.get("ref_window") == th.monitor_window for p in stored.values()):
        return stored
    if ref.trained_until is None:
        return None
    frame = setup.frames[STRAT_TO_KEY[ref.strategy]]
    train = frame[pd.to_datetime(frame["trade_date"]) <= pd.Timestamp(ref.trained_until)]
    if len(train) < 500:
        return None
    prof = stored or REF.profile(train, feature_list(ref.strategy, frame), cfg.lifecycle.reference_quantiles)
    REF.window_baseline(train, prof, th.monitor_window, th.baseline_step, th.baseline_quantile)
    with session_scope(engine) as s:
        m = s.get(Model, ref.id)
        m.params = {**(m.params or {}), "reference": prof, "reference_rows": int(len(train))}
    return prof


def _write(engine: Engine, model_id: int, d: date, rows: list[tuple[str, float, dict | None]]) -> None:
    with session_scope(engine) as s:
        s.execute(delete(MonitoringMetric).where(MonitoringMetric.model_id == model_id, MonitoringMetric.metric_date == d, MonitoringMetric.metric_name.in_([r[0] for r in rows])))
        for name, v, det in rows:
            if v is not None and np.isfinite(v):
                s.add(MonitoringMetric(model_id=model_id, metric_name=name[:64], metric_date=d, value=float(v), details=det))


def drift_metrics(engine: Engine, cfg: AppConfig, setup: Setup, ref: REG.ModelRef, d: pd.Timestamp) -> dict:
    th = cfg.lifecycle.thresholds
    prof = ensure_reference(engine, cfg, setup, ref)
    if not prof:
        return {"skipped": "no training reference (the model has no recorded training cut-off or too few training rows)"}
    frame = setup.frames[STRAT_TO_KEY[ref.strategy]]
    dates = pd.to_datetime(frame["trade_date"])
    cal_idx = setup.calendar
    start = cal_idx[max(0, int(cal_idx.get_loc(cal_idx[cal_idx <= d][-1])) - th.monitor_window + 1)]
    recent = frame[(dates >= start) & (dates <= d)]
    if recent["trade_date"].nunique() < 10:
        return {"skipped": "fewer than 10 sessions of recent data"}
    dr = REF.drift(prof, recent)
    psis = {f: v["psi"] for f, v in dr.items() if np.isfinite(v["psi"])}
    kss = {f: v["ks"] for f, v in dr.items() if np.isfinite(v["ks"])}
    lim = lambda f, key, floor: max(floor, prof[f].get(key) or 0.0)      # the floor from the config, or the training period's own 95th percentile if that is higher
    hot = sorted((f for f in psis if psis[f] > lim(f, "psi_ref", th.psi_alert) or kss.get(f, 0) > lim(f, "ks_ref", th.ks_alert)), key=lambda f: -psis[f])
    return {"psi": psis, "ks": kss, "psi_max": max(psis.values()) if psis else None, "ks_max": max(kss.values()) if kss else None, "n_alert": len(hot), "features_alert": hot,
            "rows": int(len(recent)), "window": [str(start.date()), str(d.date())]}


def _baseline_ic(cfg: AppConfig, strategy: str) -> dict | None:
    """The model's own out-of-sample rank IC from the research (mean, std, days): what 'normal' looks like."""
    try:
        if strategy == "swing":
            import json
            from predict_stock.swing.report import latest_payload_path
            ic = json.loads(latest_payload_path(cfg).read_text(encoding="utf-8"))["ic"]["scores"]["swing_lgbm"]
        else:
            from predict_stock.invest.report import latest_payloads
            p = latest_payloads(cfg)[strategy.split("_")[1]]
            ic = p["ic"][cfg.reco.card_model[strategy.split("_")[1]]]
        return {"mean": ic["mean"], "std": ic["std"], "n_days": ic["n_days"]}
    except Exception:                                                          # noqa: BLE001 - no baseline is a fact, not a failure
        return None


def performance_metrics(engine: Engine, cfg: AppConfig, setup: Setup, ref: REG.ModelRef, d: pd.Timestamp, perf_start: pd.Timestamp | None) -> dict:
    th = cfg.lifecycle.thresholds
    if perf_start is None:
        return {"skipped": "no paper record yet: performance is only measured on sessions after the record starts"}
    start = max(pd.Timestamp(perf_start), pd.Timestamp(ref.trained_until) + pd.Timedelta(days=1) if ref.trained_until else pd.Timestamp.min)
    key = STRAT_TO_KEY[ref.strategy]
    frame = setup.frames[key]
    strategy = ref.strategy
    horizon = cfg.lifecycle.promotion.compare_horizon[strategy]
    ret, rank, end = (f"fwd_ret_{horizon}", f"fwd_rank_{horizon}", f"fwd_end_{horizon}")
    if ret not in frame.columns:
        return {"skipped": f"no {ret} label in the dataset"}
    fr = frame[(pd.to_datetime(frame["trade_date"]) >= start) & (pd.to_datetime(frame["trade_date"]) <= d)]
    realised = fr[fr[ret].notna() & (pd.to_datetime(fr[end]) <= d)]
    out: dict = {"start": str(start.date()), "realised_rows": int(len(realised))}
    if realised.empty:
        return {**out, "skipped": "no realised label yet"}
    # ---- scores of the champion on those rows ------------------------------------------------------------------------------------------------------------
    if strategy == "swing":
        with session_scope(engine) as s:
            preds = pd.read_sql(select(Prediction.as_of_date.label("trade_date"), Prediction.instrument_id, Prediction.score, Prediction.proba, Prediction.rank_in_universe)
                                .where(Prediction.model_id == ref.id, Prediction.as_of_date >= start.date()), s.connection())
        preds["trade_date"] = pd.to_datetime(preds["trade_date"])
        merged = realised.assign(trade_date=pd.to_datetime(realised["trade_date"])).merge(preds, on=["trade_date", "instrument_id"], how="inner")
    else:
        from predict_stock.invest.models import load_candidate
        from pathlib import Path
        from predict_stock.config import PROJECT_ROOT
        model, _, _ = load_candidate((PROJECT_ROOT / ref.artifact_path).read_bytes())
        merged = realised.assign(score=model.predict(realised))
        merged["rank_in_universe"] = merged.groupby("trade_date")["score"].rank(ascending=False, method="first")
    if merged.empty:
        return {**out, "skipped": "the champion has no stored prediction with a realised label yet"}
    days = sorted(merged["trade_date"].unique())[-th.ic_window:]
    w = merged[merged["trade_date"].isin(days)]
    ic = SM.daily_rank_ic(w, "score", ret)
    summ = SM.ic_summary(ic, horizon)
    out.update({"ic": summ, "ic_days": int(len(ic)), "baseline_ic": _baseline_ic(cfg, strategy)})
    k = cfg.reco.swing.max_positions if strategy == "swing" else cfg.invest.top_k
    top = w[w["rank_in_universe"] <= k]
    out["hit_rate"] = float((top[ret] > 0).mean()) if len(top) else None
    out["hit_rate_all"] = float((w[ret] > 0).mean())
    if strategy == "swing" and "tb_label" in w.columns:
        ev = w[w["tb_label"].notna() & (pd.to_datetime(w["tb_end"]) <= d)]
        if len(ev):
            y = (ev["tb_label"] == 1).astype(float).to_numpy()
            p = ev["proba"].to_numpy(float)
            out["calibration"] = {"n": int(len(ev)), "gap": float(p.mean() - y.mean()), "ece": cal.ece(y, p), "realised": float(y.mean()), "mean_stated": float(p.mean())}
    return out


def operational_metrics(engine: Engine, cfg: AppConfig, ref: REG.ModelRef, d: pd.Timestamp, perf_start: pd.Timestamp | None) -> dict:
    """Paper fill rate and holding-time deviation of the cards of this strategy over the last 60 sessions; universe turnover over the last 90 days."""
    out: dict = {}
    since = (d - pd.Timedelta(days=90)).date()
    with session_scope(engine) as s:
        out["universe_changes_90d"] = int(s.scalar(select(func.count()).select_from(UniverseChangeLog).where(UniverseChangeLog.effective_date > since, UniverseChangeLog.action.in_(("add", "remove")))) or 0)
        if perf_start is None:
            return out
        cs = CARD_STRATEGY[ref.strategy]
        lo = max(pd.Timestamp(perf_start), d - pd.Timedelta(days=90)).date()
        recs = s.execute(select(Recommendation.id, Recommendation.status, Recommendation.card).where(Recommendation.strategy == cs, Recommendation.action == "BUY", Recommendation.as_of_date >= lo,
                                                                                                       Recommendation.as_of_date < d.date())).all()
        st = pd.Series([r.status for r in recs]) if recs else pd.Series(dtype=str)
        filled = int(st.isin(["holding", "target", "stopped", "time_exit", "kill_switch", "closed"]).sum())
        not_filled = int(st.isin(["expired", "cancelled"]).sum())
        if filled + not_filled:
            out["fill_rate"] = {"value": filled / (filled + not_filled), "orders": filled + not_filled}
        outs = s.execute(select(RecommendationOutcome.holding_days, Recommendation.card).join(Recommendation, Recommendation.id == RecommendationOutcome.recommendation_id)
                         .where(Recommendation.strategy == cs, RecommendationOutcome.holding_days.is_not(None), Recommendation.as_of_date >= lo)).all()
        devs = [h - c["holding"]["expected_sessions"] for h, c in outs if c and c.get("holding", {}).get("expected_sessions")]
        if devs:
            out["hold_dev"] = {"value": float(np.mean(devs)), "trades": len(devs)}
    return out


def run_monitor(engine: Engine, cfg: AppConfig, setup: Setup, as_of: pd.Timestamp, *, perf_start: pd.Timestamp | None = None, strategies: tuple[str, ...] = REG.STRATEGIES,
                alert: bool = True) -> dict:
    """Compute and store the metrics of every champion; alert on breaches. Returns {strategy: {metrics, breaches, triggers}}."""
    th = cfg.lifecycle.thresholds
    d = pd.Timestamp(as_of)
    result: dict = {}
    for strategy in strategies:
        ref = REG.champion(engine, strategy)
        if ref is None:
            result[strategy] = {"skipped": "no champion"}
            continue
        rows: list[tuple[str, float, dict | None]] = []
        breaches: list[dict] = []
        drift = drift_metrics(engine, cfg, setup, ref, d)
        if "psi" in drift:
            for f in drift["psi"]:
                rows += [(f"psi:{f}", drift["psi"][f], None), (f"ks:{f}", drift["ks"].get(f), None)]
            rows += [("psi_max", drift["psi_max"], {"features": drift["features_alert"][:8], "window": drift["window"]}), ("ks_max", drift["ks_max"], None), ("drift_features_alert", float(drift["n_alert"]), {"features": drift["features_alert"]})]
            if drift["n_alert"] >= th.psi_features_trigger:
                breaches.append({"kind": "drift", "severity": "error", "message": f"{strategy}: {drift['n_alert']} features drifted beyond the larger of PSI {th.psi_alert} / KS {th.ks_alert} and what the training period itself showed in 60-session slices ({', '.join(drift['features_alert'][:6])})"})
            elif drift["n_alert"] >= 1:
                breaches.append({"kind": "drift_moderate", "severity": "warn", "message": f"{strategy}: {drift['n_alert']} feature(s) beyond the training period's own range of shifts ({', '.join(drift['features_alert'])}); "
                                 f"the drift retrain needs {th.psi_features_trigger}"})
        perf = performance_metrics(engine, cfg, setup, ref, d, perf_start)
        if "ic" in perf:
            ic = perf["ic"]
            rows += [("rolling_ic", ic["mean"], {"days": perf["ic_days"], "t_stat": ic["t_stat"]}), ("hit_rate", perf.get("hit_rate"), {"all_rows": perf.get("hit_rate_all")})]
            base = perf.get("baseline_ic")
            if perf["ic_days"] >= th.ic_min_days:
                se = (base["std"] / np.sqrt(max(perf["ic_days"] / cfg.lifecycle.promotion.compare_horizon.get(strategy, 5), 1))) if base else None
                low = base["mean"] - th.ic_z * se if base and se else 0.0
                if ic["mean"] < low or ic["mean"] < 0 and (not base or base["mean"] > 0):
                    breaches.append({"kind": "ic", "severity": "warn", "message": f"{strategy}: rolling {perf['ic_days']}-day rank IC {ic['mean']:.3f} is below "
                                     + (f"its out-of-sample norm {base['mean']:.3f} by more than {th.ic_z} standard errors" if base else "zero")})
        if "calibration" in perf:
            c = perf["calibration"]
            rows += [("calibration_gap", c["gap"], {"n": c["n"]}), ("ece", c["ece"], {"n": c["n"]})]
            if c["n"] >= th.calibration_min_events and abs(c["gap"]) > th.calibration_gap:
                breaches.append({"kind": "calibration", "severity": "warn", "message": f"{strategy}: stated probability off by {c['gap'] * 100:+.1f} points over {c['n']} realised events"})
        ops = operational_metrics(engine, cfg, ref, d, perf_start)
        rows.append(("universe_changes_90d", float(ops["universe_changes_90d"]), None))
        if ops["universe_changes_90d"] >= th.universe_changes_trigger:
            breaches.append({"kind": "universe", "severity": "warn", "message": f"{strategy}: {ops['universe_changes_90d']} names joined or left the universe in the last 90 days"})
        if "fill_rate" in ops:
            fr = ops["fill_rate"]
            rows.append(("fill_rate", fr["value"], {"orders": fr["orders"]}))
            if fr["orders"] >= th.fill_min_orders and fr["value"] < th.fill_rate_min:
                breaches.append({"kind": "fill", "severity": "warn", "message": f"{strategy}: paper fill rate {fr['value'] * 100:.0f}% over {fr['orders']} orders (backtest of the cards: 71%)"})
        if "hold_dev" in ops:
            hd = ops["hold_dev"]
            rows.append(("hold_dev", hd["value"], {"trades": hd["trades"]}))
            if hd["trades"] >= th.hold_min_trades and abs(hd["value"]) > th.hold_dev_sessions:
                breaches.append({"kind": "hold", "severity": "warn", "message": f"{strategy}: actual holding time differs from the stated one by {hd['value']:+.1f} sessions on average ({hd['trades']} trades)"})
        _write(engine, ref.id, d.date(), rows)
        if alert:
            for b in breaches:
                raise_alert(engine, b["severity"], f"monitor_{b['kind']}", b["message"], details={"strategy": strategy, "model_id": ref.id, "as_of": str(d.date())})
        result[strategy] = {"model_id": ref.id, "model": f"{ref.name} v{ref.version}", "drift": {k: v for k, v in drift.items() if k not in ("psi", "ks")}, "performance": perf, "operations": ops,
                            "breaches": breaches, "triggers": sorted({b["kind"].split("_")[0] for b in breaches}), "metrics_written": len(rows)}
    return result
