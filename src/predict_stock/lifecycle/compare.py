"""Champion vs challenger, on data neither has seen, with a rule fixed in advance (`lifecycle.promotion`), then promotion (by hand) and rollback.

The comparison
* window = sessions AFTER the later of the two training cut-offs whose forward label has already been realised, never the protected (held-out) period unless it was consumed;
* both models are read from their STORED predictions (the challenger's shadow predictions were written on the day, so hindsight cannot creep in);
* rank IC per day for each model, paired block bootstrap of the difference; the net Sharpe of the top-K portfolio of each model through the backtest engine (costs included), paired
  bootstrap of the difference; max drawdown; calibration error of the SWING probability;
* the one-sided level is alpha / k, k = number of challengers tried so far against this champion (`experiments`): the more you try, the harder it gets to win by luck.

PASS = enough shadow time and realised days, AND (the IC difference OR the Sharpe difference has a bootstrap lower bound above zero), AND no worse max drawdown (beyond the tolerance),
AND no worse calibration (beyond the tolerance). `promote` re-runs the comparison, refuses unless it passes, and only then moves the status: the human decision is the command itself."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select

from predict_stock.backtest import metrics as M
from predict_stock.backtest.runner import Setup, clean
from predict_stock.config import AppConfig
from predict_stock.db.models import Experiment, Model, Prediction
from predict_stock.db.session import session_scope
from predict_stock.invest import stats as IS
from predict_stock.lifecycle import registry as REG
from predict_stock.swing import calibration as cal
from predict_stock.swing import metrics as SM
from predict_stock.swing.registry import log_experiment


def holdout_consumed(engine: Engine) -> bool:
    with session_scope(engine) as s:
        return s.scalar(select(func.count()).select_from(Experiment).where(Experiment.name == "holdout:consumed")) > 0


def attempts(engine: Engine, strategy: str, since: datetime | None = None) -> int:
    """Challengers tried against the current champion of ``strategy``: retrain experiments recorded since it became champion (at least 1: the one being judged)."""
    with session_scope(engine) as s:
        q = select(func.count()).select_from(Experiment).where(Experiment.name == f"lifecycle:retrain:{strategy}")
        if since is not None:
            q = q.where(Experiment.started_at >= since)
        return max(int(s.scalar(q)), 1)


def _champion_since(engine: Engine, strategy: str) -> datetime | None:
    h = [x for x in REG.history(engine, strategy=strategy) if x["to"] == "champion"]
    return h[-1]["at"] if h else None


def _stored(engine: Engine, model_id: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    with session_scope(engine) as s:
        df = pd.read_sql(select(Prediction.as_of_date.label("trade_date"), Prediction.instrument_id, Prediction.score, Prediction.proba).where(
            Prediction.model_id == model_id, Prediction.as_of_date >= start.date(), Prediction.as_of_date <= end.date()), s.connection())
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def protocol_hash(cfg: AppConfig, strategy: str) -> str:
    """Fingerprint of the walk-forward protocol a model must have been trained under: folds, horizon, feature set, label spec, hyper-parameter search and candidates. A challenger
    is only comparable if it was trained under the protocol in force now; the retrain job stores this in its params."""
    import hashlib
    import json
    if strategy == "swing":
        c = cfg.swing
        spec = {"feature_set": c.feature_set, "label_spec": c.label_spec, "rank_target": c.rank_target, "horizon": c.horizon, "folds": c.folds.model_dump(), "optuna": c.optuna.model_dump(),
                "lgbm": c.lgbm, "n_estimators": c.n_estimators, "calibration": c.calibration, "quantiles": c.quantiles}
    else:
        c = cfg.invest
        spec = {"feature_set": c.feature_set, "label_spec": c.label_spec, "preset": c.presets[strategy.split("_")[1]].model_dump(), "folds": c.folds.model_dump(), "candidates": c.candidates,
                "factors": c.factors, "lgbm": c.lgbm, "n_estimators": c.n_estimators}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _portfolio(setup: Setup, cfg: AppConfig, strategy: str, scores: pd.DataFrame, first_idx: int, last_idx: int):
    """Net equity of the strategy's portfolio built from a model's scores, constructed exactly as in the research (SWING: top-K weekly; INVEST: top-K on the preset's schedule, tranches)."""
    from predict_stock.invest.strategy import invest_signals
    from predict_stock.swing.evaluate import run_signals
    from predict_stock.swing.strategies import topk_signals
    local = replace(setup, start_idx=first_idx, dev_end_idx=last_idx)
    if strategy == "swing":
        st = cfg.swing.strategy
        sig = topk_signals(scores, setup.calendar, first_idx, last_idx, k=st.top_k, max_weight=cfg.backtest.max_weight, rebalance=st.rebalance)
    else:
        iv, preset = cfg.invest, cfg.invest.presets[strategy.split("_")[1]]
        sig = invest_signals(scores, setup.calendar, setup.data.close.pct_change(), first_idx, last_idx, rebalance=preset.rebalance, k=iv.top_k, weighting=iv.weighting, cap=iv.max_weight,
                             cov_window=iv.cov_window, tranches=preset.tranches, spacing=preset.tranche_spacing)
    if not sig:
        return None
    return run_signals(local, sig, first_idx, 1.0).equity


def compare(engine: Engine, cfg: AppConfig, setup: Setup, challenger_id: int, as_of: pd.Timestamp, *, write: bool = True) -> dict:
    lc, pr = cfg.lifecycle, cfg.lifecycle.promotion
    ch = REG.get(engine, challenger_id)
    if ch.status != "shadow":
        raise REG.LifecycleError(f"#{challenger_id} is {ch.status}: only a shadow model is compared")
    champ = REG.champion(engine, ch.strategy)
    if champ is None:
        raise REG.LifecycleError(f"{ch.strategy} has no champion to compare with")
    if ch.trained_until is None or champ.trained_until is None:
        raise REG.LifecycleError("both models need a recorded training cut-off")
    with session_scope(engine) as s:
        pc, pn = s.get(Model, champ.id).params or {}, s.get(Model, ch.id).params or {}
    same_protocol = pn.get("protocol") == protocol_hash(cfg, ch.strategy)
    strategy, d = ch.strategy, pd.Timestamp(as_of)
    horizon = pr.compare_horizon[strategy]
    start = max(pd.Timestamp(champ.trained_until), pd.Timestamp(ch.trained_until)) + pd.Timedelta(days=1)
    protected = (pd.Timestamp(lc.protected_period[0]), pd.Timestamp(lc.protected_period[1]))
    if not holdout_consumed(engine) and start <= protected[1] and d >= protected[0]:
        raise REG.LifecycleError(f"the comparison window {start.date()}..{d.date()} touches the protected held-out period {protected[0].date()}..{protected[1].date()}; "
                                 "it is only allowed after that period was consumed by training (`retrain --consume-holdout`)")
    key = "swing" if strategy == "swing" else "invest"
    frame = setup.frames[key]
    ret, end_col = f"fwd_ret_{horizon}", f"fwd_end_{horizon}"
    fd = pd.to_datetime(frame["trade_date"])
    labels = frame[(fd >= start) & (fd <= d) & frame[ret].notna() & (pd.to_datetime(frame[end_col]) <= d)]
    labels = labels.assign(trade_date=pd.to_datetime(labels["trade_date"]))
    a, b = _stored(engine, champ.id, start, d), _stored(engine, ch.id, start, d)
    rep: dict = {"strategy": strategy, "champion": {"id": champ.id, "model": f"{champ.name} v{champ.version}", "trained_until": str(champ.trained_until)},
                 "challenger": {"id": ch.id, "model": f"{ch.name} v{ch.version}", "trained_until": str(ch.trained_until)}, "as_of": str(d.date()), "window_start": str(start.date()),
                 "protocol_identical": same_protocol, "champion_protocol": pc.get("protocol"), "challenger_protocol": pn.get("protocol"), "horizon": horizon}
    merged = labels[["trade_date", "instrument_id", ret] + ([c for c in ("tb_label", "tb_end") if c in labels.columns])].merge(
        a.rename(columns={"score": "s_champ", "proba": "p_champ"}), on=["trade_date", "instrument_id"]).merge(b.rename(columns={"score": "s_chall", "proba": "p_chall"}), on=["trade_date", "instrument_id"])
    weeks = ((b["trade_date"].max() - b["trade_date"].min()).days / 7) if len(b) else 0.0
    days = int(merged["trade_date"].nunique())
    k = attempts(engine, strategy, _champion_since(engine, strategy))
    alpha_adj = pr.alpha / k
    rep.update({"shadow_weeks": weeks, "realised_days": days, "attempts": k, "alpha_adjusted": alpha_adj})
    crit = [
        {"name": "challenger trained under the walk-forward protocol in force now", "value": same_protocol, "threshold": True, "ok": bool(same_protocol)},
        {"name": f"shadow time >= {pr.shadow_min_weeks[strategy]} weeks", "value": weeks, "threshold": pr.shadow_min_weeks[strategy], "ok": weeks >= pr.shadow_min_weeks[strategy]},
        {"name": f"realised days after both training cut-offs >= {pr.min_days[strategy]}", "value": days, "threshold": pr.min_days[strategy], "ok": days >= pr.min_days[strategy]},
    ]
    enough = crit[1]["ok"] and crit[2]["ok"] and merged["trade_date"].nunique() >= 5
    ic_lo = sh_lo = None
    if merged.empty or not enough:
        rep.update({"criteria": crit + [{"name": "IC or Sharpe difference above zero (bootstrap lower bound)", "value": None, "threshold": 0, "ok": False, "note": "not enough shadow data yet"}], "passed": False,
                    "verdict": "NOT ENOUGH SHADOW DATA: the comparison is not run until the challenger has the required weeks and realised days"})
        return _store(engine, cfg, rep, write)
    ic_a = SM.daily_rank_ic(merged.rename(columns={"s_champ": "score"}), "score", ret)
    ic_b = SM.daily_rank_ic(merged.rename(columns={"s_chall": "score"}), "score", ret)
    both = pd.concat([ic_a.rename("champ"), ic_b.rename("chall")], axis=1, join="inner").dropna()
    diff = (both["chall"] - both["champ"]).to_numpy()
    idx = IS.bootstrap_indices(len(diff), pr.bootstrap_resamples, max(pr.bootstrap_block, 1), pr.seed)
    boot = diff[idx].mean(axis=1)
    ic_lo = float(np.quantile(boot, alpha_adj))
    rep["ic"] = {"champion_mean": float(both["champ"].mean()), "challenger_mean": float(both["chall"].mean()), "diff": float(diff.mean()), "lower_bound": ic_lo, "days": int(len(diff)),
                 "champion": SM.ic_summary(ic_a, horizon), "challenger": SM.ic_summary(ic_b, horizon)}
    # ---- portfolios, net Sharpe and drawdown ---------------------------------------------------------------------------------------------------------------------
    first_idx = int(setup.calendar.searchsorted(start))
    last_idx = int(setup.calendar.get_loc(setup.calendar[setup.calendar <= d][-1]))
    eq = {}
    for name, col in (("champ", "s_champ"), ("chall", "s_chall")):
        eq[name] = _portfolio(setup, cfg, strategy, merged[["trade_date", "instrument_id", col]].rename(columns={col: "score"}), first_idx, last_idx)
    if all(v is not None for v in eq.values()) and len(eq["champ"]) > 20:
        ra, rb = IS.daily_returns(eq["champ"]), IS.daily_returns(eq["chall"])
        bs = IS.paired_bootstrap(rb, ra, resamples=pr.bootstrap_resamples, block=max(pr.bootstrap_block, 1), level=1 - 2 * alpha_adj, seed=pr.seed)
        mdd_a, mdd_b = M.max_drawdown(eq["champ"]), M.max_drawdown(eq["chall"])
        sh_lo = bs["sharpe_diff"]["lo"]
        rep["portfolio"] = {"champion": {"sharpe": M.sharpe(ra), "max_drawdown": mdd_a}, "challenger": {"sharpe": M.sharpe(rb), "max_drawdown": mdd_b}, "sharpe_diff": bs["sharpe_diff"],
                            "days": int(len(ra)), "note": "net of costs, the research's portfolio construction, built from each model's STORED predictions"}
        crit.append({"name": f"max drawdown no worse than the champion's by more than {pr.mdd_tolerance:.0%}", "value": mdd_b, "threshold": mdd_a - pr.mdd_tolerance, "ok": bool(mdd_b >= mdd_a - pr.mdd_tolerance)})
    else:
        crit.append({"name": "max drawdown no worse than the champion's", "value": None, "threshold": None, "ok": False, "note": "no portfolio could be built over the window"})
    if strategy == "swing" and "tb_label" in merged.columns:
        ev = merged[merged["tb_label"].notna() & (pd.to_datetime(merged["tb_end"]) <= d)]
        if len(ev) >= 50:
            y = (ev["tb_label"] == 1).astype(float).to_numpy()
            ea, eb = cal.ece(y, ev["p_champ"].to_numpy(float)), cal.ece(y, ev["p_chall"].to_numpy(float))
            rep["calibration"] = {"champion_ece": ea, "challenger_ece": eb, "events": int(len(ev))}
            crit.append({"name": f"probability calibration (ECE) no worse by more than {pr.ece_tolerance}", "value": eb, "threshold": ea + pr.ece_tolerance, "ok": bool(eb <= ea + pr.ece_tolerance)})
        else:
            crit.append({"name": "probability calibration (ECE)", "value": None, "threshold": None, "ok": False, "note": "fewer than 50 realised events"})
    better = (ic_lo is not None and ic_lo > 0) or (sh_lo is not None and sh_lo > 0)
    crit.insert(3, {"name": f"rank IC OR net Sharpe beats the champion (bootstrap lower bound > 0 at one-sided level {alpha_adj:.3f})", "value": {"ic_lower_bound": ic_lo, "sharpe_lower_bound": sh_lo},
                    "threshold": 0, "ok": bool(better)})
    rep["criteria"] = crit
    rep["passed"] = all(c["ok"] for c in crit)
    rep["verdict"] = ("PASS: the challenger may be promoted (manual confirmation required)" if rep["passed"] else
                      "REJECTED: " + "; ".join(c["name"] for c in crit if not c["ok"]))
    return _store(engine, cfg, rep, write)


def _store(engine: Engine, cfg: AppConfig, rep: dict, write: bool) -> dict:
    rep = clean(rep)
    if write:
        rid = log_experiment(engine, cfg, f"lifecycle:compare:{rep['challenger']['id']}", key=f"{rep['challenger']['id']}:{rep['as_of']}", params={"strategy": rep["strategy"], "champion": rep["champion"]["id"],
                             "challenger": rep["challenger"]["id"], "as_of": rep["as_of"], "rule": cfg.lifecycle.promotion.model_dump()}, summary=rep, status="success" if rep["passed"] else "rejected",
                             description=rep["verdict"][:200])
        rep["experiment_id"] = rid
    return rep


# ---- promote / reject / rollback -------------------------------------------------------------------------------------------------------------------------------
def promote(engine: Engine, cfg: AppConfig, setup: Setup, model_id: int, as_of: pd.Timestamp, *, actor: str = "manual", note: str = "") -> dict:
    """Manual promotion: the comparison is re-run NOW and must pass. Refuses otherwise (there is no override)."""
    rep = compare(engine, cfg, setup, model_id, as_of)
    if not rep["passed"]:
        raise REG.LifecycleError(f"the challenger does not meet the promotion rule: {rep['verdict']}")
    ref = REG.set_status(engine, model_id, "champion", actor=actor, reason=note or "promoted after passing the pre-registered comparison", details={"comparison_experiment": rep.get("experiment_id"), "as_of": rep["as_of"]})
    return {"promoted": f"{ref.name} v{ref.version}", "comparison": rep}


def reject(engine: Engine, model_id: int, *, actor: str = "manual", reason: str) -> REG.ModelRef:
    return REG.set_status(engine, model_id, "retired", actor=actor, reason=f"rejected: {reason}")


def rollback(engine: Engine, strategy: str, *, actor: str = "manual", reason: str) -> dict:
    """Back to the champion that the current one replaced; the current champion is retired."""
    cur, prev = REG.champion(engine, strategy), REG.previous_champion(engine, strategy)
    if cur is None or prev is None:
        raise REG.LifecycleError(f"{strategy}: there is no earlier champion to roll back to")
    ref = REG.set_status(engine, prev.id, "champion", actor=actor, reason=f"rollback from #{cur.id}: {reason}", details={"rolled_back": cur.id})
    return {"champion": f"{ref.name} v{ref.version}", "retired": f"{cur.name} v{cur.version}"}
