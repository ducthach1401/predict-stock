"""Optuna tuning of the shared tree hyper-parameters. Runs ONCE, on the training/validation data of the FIRST walk-forward fold (everything before
its test window); the best parameters are then frozen for every fold. The objective is the mean daily rank IC of the ranking booster on the
validation rows. The number of trials is capped by ``swing.optuna.trials``. EVERY trial - completed, failed or pruned - goes to ``sink``
(which writes an ``experiments`` row), so nothing that was tried disappears."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from predict_stock.config import SwingConfig
from predict_stock.swing.metrics import daily_rank_ic
from predict_stock.swing.model import _has_target, _lgb_params, _target, feature_columns

optuna.logging.set_verbosity(optuna.logging.WARNING)

Sink = Callable[[dict], None]


@dataclass
class TuneResult:
    best_params: dict
    best_value: float | None
    trials: list[dict] = field(default_factory=list)
    n_complete: int = 0
    n_failed: int = 0


def suggest(trial: optuna.Trial) -> dict:
    return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 600, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0), "bagging_freq": 5,
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 100.0, log=True)}


def validation_ic(fit: pd.DataFrame, val: pd.DataFrame, cfg: SwingConfig, tree: dict) -> float:
    """Mean daily rank IC (vs the 5-session forward return) of a ranking booster fitted on ``fit``, on ``val``."""
    feats = feature_columns(fit)
    mf, mv = _has_target(fit, "rank", cfg), _has_target(val, "rank", cfg)
    params = _lgb_params(cfg, "rank", tree)
    dtr = lgb.Dataset(fit.loc[mf, feats].to_numpy(float), _target(fit, "rank", cfg)[mf], free_raw_data=False)
    dva = lgb.Dataset(val.loc[mv, feats].to_numpy(float), _target(val, "rank", cfg)[mv], reference=dtr, free_raw_data=False)
    b = lgb.train(params, dtr, num_boost_round=cfg.n_estimators, valid_sets=[dva], callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
    scored = val.loc[mv, ["trade_date", cfg.return_target]].assign(score=b.predict(val.loc[mv, feats].to_numpy(float)))
    ic = daily_rank_ic(scored, "score", cfg.return_target)
    if ic.empty or not np.isfinite(ic.mean()):
        raise ValueError("no valid daily rank IC on the validation rows")
    return float(ic.mean())


def run_study(fit: pd.DataFrame, val: pd.DataFrame, cfg: SwingConfig, sink: Sink, objective: Callable | None = None) -> TuneResult:
    """``objective(trial, fit, val, cfg)`` can replace the default (tests inject failing trials); it must draw its parameters via ``suggest``."""
    def obj(trial: optuna.Trial) -> float:
        try:
            if objective is not None:
                return objective(trial, fit, val, cfg)
            return validation_ic(fit, val, cfg, suggest(trial))
        except Exception as exc:                                              # recorded, then Optuna marks the trial FAIL and carries on
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:500])
            raise

    records: list[dict] = []

    def log(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        rec = {"number": trial.number, "state": trial.state.name, "value": trial.value, "params": dict(trial.params),
               "error": trial.user_attrs.get("error"), "duration_s": trial.duration.total_seconds() if trial.duration else None}
        records.append(rec)
        sink(rec)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.optuna.seed), pruner=optuna.pruners.NopPruner())
    study.optimize(obj, n_trials=cfg.optuna.trials, timeout=cfg.optuna.timeout_s, catch=(Exception,), callbacks=[log], gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    best = study.best_trial if done else None
    return TuneResult(dict(best.params) if best else {}, float(best.value) if best else None, records, len(done), len(failed))
