"""Retraining: WHEN (triggers) and HOW (the current walk-forward protocol, then shadow).

Triggers (`due`): schedule (the champion's training cut-off is older than `retrain_months`), drift (PSI/KS over threshold on enough features), IC decay, calibration drift, many universe
changes, a new feature set in the config, or manual. A trigger only says a retrain is DUE; nothing is replaced by it. The retrain trains a challenger with the same protocol as the research
(same folds, purge and embargo, same hyper-parameter search on the first fold, same candidates, same label spec), registers it as `shadow`, and from then on it only produces predictions.

Held-out guard: the research held out `lifecycle.protected_period` and never looked at it. A model trained on data that reaches into it consumes it (it can no longer be an honest test for
anything trained before). That is refused unless `consume_holdout=True`, and when it is done it is recorded once as the experiment `holdout:consumed`; nothing does this automatically."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Engine, func, select

from predict_stock.backtest.runner import Setup, clean, load_setup
from predict_stock.backtest.walkforward import Holdout
from predict_stock.config import PROJECT_ROOT, AppConfig
from predict_stock.db.models import Experiment, Model
from predict_stock.db.session import session_scope
from predict_stock.lifecycle import compare as CMP
from predict_stock.lifecycle import registry as REG
from predict_stock.lifecycle import shadow as SH
from predict_stock.swing.registry import log_experiment, register_artifact, register_model

TRIGGERS = ("schedule", "drift", "manual")


class RetrainRefused(RuntimeError):
    pass


def _months_between(a: date, b: date) -> float:
    return (b - a).days / 30.44


def due(engine: Engine, cfg: AppConfig, strategy: str, as_of: pd.Timestamp, monitor: dict | None = None) -> list[dict]:
    """Reasons a retrain of ``strategy`` is due now: [{kind, message}]. ``monitor`` = the result of `monitor.run_monitor` for the same day (drift, IC, calibration, universe)."""
    lc = cfg.lifecycle
    champ = REG.champion(engine, strategy)
    out: list[dict] = []
    if champ is None:
        return [{"kind": "no_champion", "message": f"{strategy} has no champion"}]
    if champ.trained_until is not None:
        age = _months_between(champ.trained_until, as_of.date())
        if age >= lc.retrain_months:
            out.append({"kind": "schedule", "message": f"{strategy}: the champion's training data ends {champ.trained_until} ({age:.1f} months ago, schedule = {lc.retrain_months})"})
    with session_scope(engine) as s:
        m = s.get(Model, champ.id)
        from predict_stock.db.models import FeatureSet
        fs = s.get(FeatureSet, m.feature_set_id)
    current = cfg.swing.feature_set if strategy == "swing" else cfg.invest.feature_set
    if fs is not None and f"{fs.name}:{fs.version}" != current:
        out.append({"kind": "new_feature_set", "message": f"{strategy}: the config now uses feature set {current}, the champion was trained on {fs.name}:{fs.version}"})
    res = (monitor or {}).get(strategy) or {}
    for b in res.get("breaches", []):
        if b["kind"] in ("drift", "ic", "calibration", "universe"):
            out.append({"kind": b["kind"], "message": b["message"]})
    return out


def _protected_start(cfg: AppConfig) -> pd.Timestamp:
    return pd.Timestamp(cfg.lifecycle.protected_period[0])


def _consume(engine: Engine, cfg: AppConfig, strategy: str, until: date, reason: str) -> int:
    return log_experiment(engine, cfg, "holdout:consumed", key=f"consumed:{cfg.lifecycle.protected_period[0]}", params={"protected_period": list(cfg.lifecycle.protected_period), "strategy": strategy},
                          summary={"first_by": strategy, "training_data_until": str(until), "reason": reason}, status="success",
                          description="the research's held-out period was used as training data (explicitly confirmed); it is no longer a test set for anything")


def _find_tuning(engine: Engine, prefix: str, fold0: dict, preset: str | None = None) -> dict | None:
    """The tuning result of an earlier run with an IDENTICAL first fold (same problem, so no need to search again)."""
    with session_scope(engine) as s:
        for e in s.scalars(select(Experiment).where(Experiment.name.like(f"{prefix}%"), Experiment.status == "success").order_by(Experiment.id.desc())):
            p = e.params or {}
            if p.get("fold") == clean(fold0) and (preset is None or p.get("preset") == preset):
                return e.summary
    return None


def retrain(engine: Engine, cfg: AppConfig, strategy: str, trigger: str, *, as_of: pd.Timestamp | None = None, candidates: list[str] | None = None, consume_holdout: bool = False,
            reasons: list[str] | None = None, actor: str = "cli", setup: Setup | None = None, run_id: int | None = None) -> dict:
    """Train challenger(s) for ``strategy`` and register them as ``shadow``. Nothing here touches the champion or any recommendation."""
    if strategy not in REG.STRATEGIES:
        raise RetrainRefused(f"unknown strategy {strategy!r}: {REG.STRATEGIES}")
    if trigger not in TRIGGERS:
        raise RetrainRefused(f"unknown trigger {trigger!r}: {TRIGGERS}")
    champ = REG.champion(engine, strategy)
    if champ is None:
        raise RetrainRefused(f"{strategy} has no champion: there is nothing to challenge")
    pending = REG.with_status(engine, strategy, "shadow")
    if pending:
        raise RetrainRefused(f"{strategy} already has a challenger in shadow ({', '.join(f'#{p.id}' for p in pending)}): settle it (promote or reject) before training another; "
                             "one challenger at a time keeps the count of attempts honest")
    setup = setup or load_setup(engine, cfg)
    if as_of is None:
        as_of = setup.calendar[-1]
    as_of = pd.Timestamp(as_of)
    # the training frame ends where the labels are still fully known: every row whose forward label ends after `cut` is purged by the fold code
    cut = as_of + pd.Timedelta(days=1)
    to_train_end = None
    if not consume_holdout and not CMP.holdout_consumed(engine):
        if cut > _protected_start(cfg):
            raise RetrainRefused(f"training data up to {as_of.date()} would reach into the protected held-out period ({cfg.lifecycle.protected_period[0]}..{cfg.lifecycle.protected_period[1]}), "
                                 "which the research never looked at. Retraining on it uses it up as a test set for good. Re-run with --consume-holdout to do that knowingly "
                                 "(it is recorded once as the experiment `holdout:consumed`).")
    ho = Holdout(cut, cut + pd.Timedelta(days=1))
    result = {"strategy": strategy, "trigger": trigger, "as_of": str(as_of.date()), "reasons": reasons or [], "champion": f"{champ.name} v{champ.version}", "challengers": []}
    if trigger != "manual" and not reasons:
        result["note"] = "started by a trigger without recorded reasons"
    if strategy == "swing":
        out = _train_swing(engine, cfg, setup, ho, run_id)
    else:
        out = _train_invest(engine, cfg, setup, strategy.split("_")[1], ho, candidates, run_id)
    if consume_holdout and not CMP.holdout_consumed(engine) and out and max(o["trained_until"] for o in out) >= _protected_start(cfg).date():
        result["holdout_consumed_experiment"] = _consume(engine, cfg, strategy, max(o["trained_until"] for o in out), f"retrain ({trigger}) on data up to {as_of.date()}")
    prior = CMP.attempts(engine, strategy, CMP._champion_since(engine, strategy)) if any(not o["reused"] for o in out) else 0
    for o in out:
        if o["reused"]:
            result["challengers"].append({**o, "status": "skipped", "note": "identical to an existing model (same sha256): nothing new to test"})
            continue
        exp = log_experiment(engine, cfg, f"lifecycle:retrain:{strategy}", params={"strategy": strategy, "trigger": trigger, "model_id": o["model_id"], "champion": champ.id, "as_of": str(as_of.date()),
                             "candidate": o.get("candidate")}, summary={"reasons": reasons or [], "trained_until": str(o["trained_until"])}, status="success",
                             description=f"challenger #{o['model_id']} for {strategy} ({trigger})", run_id=run_id)
        ref = REG.set_status(engine, o["model_id"], "shadow", actor=actor, reason=f"retrain ({trigger}): challenger trained with the current protocol", details={"experiment": exp, "reasons": reasons or []})
        stored = SH.store(engine, cfg, setup, ref, as_of, as_of, run_id)
        result["challengers"].append({**o, "status": "shadow", "experiment_id": exp, "predictions_today": stored, "trained_until": str(o["trained_until"])})
    result["attempts_against_this_champion"] = CMP.attempts(engine, strategy, CMP._champion_since(engine, strategy))
    result["attempts_before"] = prior
    return clean(result)


def _train_swing(engine: Engine, cfg: AppConfig, setup: Setup, ho: Holdout, run_id: int | None) -> list[dict]:
    from predict_stock.swing.folds import plan_folds, split_fold, with_ends
    from predict_stock.swing.tuning import run_study
    from predict_stock.swing.walkforward import fit_final
    sw = cfg.swing
    frame = with_ends(setup.frames["swing"])
    folds = plan_folds(frame, sw, ho.start)
    if not folds:
        raise RetrainRefused("no walk-forward fold fits inside the data")
    fold0 = folds[0].as_dict()
    tuning = _find_tuning(engine, "swing:optuna:", fold0)
    if tuning is not None:
        tuned = tuning["best_params"]
    else:
        first = split_fold(frame, folds[0], sw)
        from predict_stock.swing.registry import trial_sink
        res = run_study(first.fit, first.val, sw, trial_sink(engine, cfg, "retrain", run_id))
        tuned = res.best_params
    bundle, fdata = fit_final(frame, sw, tuned, ho.start, ho.end)
    tu = fdata.fold.train_end.date()
    exp = log_experiment(engine, cfg, "swing:retrain:fit", params={"fold": clean(fdata.fold.as_dict()), "tuned": tuned}, summary={"n_fit": len(fdata.fit), "n_val": len(fdata.val)}, status="success",
                         description="final fit of a retrain", run_id=run_id)
    mid, ver, sha, reused = register_model(engine, cfg, bundle, "swing_lgbm_final", dataset_id=setup.dataset_ids.get("swing"), experiment_id=exp,
                                           extra_params={"fold": fdata.fold.as_dict(), "purged_rows": fdata.purged, "n_fit": len(fdata.fit), "n_val": len(fdata.val),
                                                         "protocol": CMP.protocol_hash(cfg, "swing"), "trained_on": f"all data up to {tu} (retrain)", "tuned_from_first_fold": True})
    return [{"model_id": mid, "version": ver, "sha256": sha, "reused": reused, "trained_until": tu, "candidate": "lgbm"}]


def _train_invest(engine: Engine, cfg: AppConfig, setup: Setup, key: str, ho: Holdout, candidates: list[str] | None, run_id: int | None) -> list[dict]:
    from predict_stock.invest import walkforward as W
    from predict_stock.invest.job import ALGO
    iv = cfg.invest
    preset = iv.presets[key]
    frame = W.prepare(setup.frames["invest"], preset)
    folds = W.plan_folds(frame, iv, preset, ho.start)
    if not folds:
        raise RetrainRefused(f"{key}: no walk-forward fold fits inside the data")
    fold0 = folds[0].as_dict()
    prior = _find_tuning(engine, "invest:grid:", fold0, key)
    hypers = prior["hypers"] if prior is not None else W.tune(W.split(frame, folds[0], iv), iv, preset, lambda t: None)
    fc, fd = W.fit_final(frame, iv, key, preset, hypers, ho.start, ho.end)
    tu = fd.fold.train_end.date()
    exp = log_experiment(engine, cfg, f"invest:{key}:retrain:fit", params={"fold": clean(fd.fold.as_dict()), "hypers": hypers}, summary={"n_fit": len(fd.fit), "n_val": len(fd.val)}, status="success",
                         description="final fit of a retrain", run_id=run_id)
    out = []
    for name in fc.models:
        if name == "factor":
            continue                                   # a fixed rule with no fitted parameter: retraining it changes nothing
        if candidates and name not in candidates:
            continue
        mid, ver, sha, reused = register_artifact(
            engine, cfg, fc.artifact(name), f"invest_{key}_{name}_final", algo=ALGO, feature_set=iv.feature_set, label_spec=iv.label_spec, dataset_id=setup.dataset_ids.get("invest"), experiment_id=exp,
            seed=iv.seed, artifact_dir=PROJECT_ROOT / "artifacts/models", strategy=f"invest_{key}", trained_until=tu,
            params={"preset": key, "candidate": name, "horizon": preset.horizon, "fold": fd.fold.as_dict(), "n_fit": len(fd.fit), "n_val": len(fd.val), "hypers": hypers.get(name),
                    "protocol": CMP.protocol_hash(cfg, f"invest_{key}"), "trained_on": f"all data up to {tu} (retrain)", "inputs": fc.models[name].inputs()})
        out.append({"model_id": mid, "version": ver, "sha256": sha, "reused": reused, "trained_until": tu, "candidate": name})
    return out
