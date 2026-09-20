"""Runs the walk-forward: one model per fold, predictions only for that fold's test window, plus the final model on all development data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from predict_stock.backtest.walkforward import Fold, purge_train
from predict_stock.config import SwingConfig
from predict_stock.swing.folds import FoldData, plan_folds, split_fold
from predict_stock.swing.model import SwingBundle, feature_columns, fit_bundle

CARRY = ["fwd_ret_3", "fwd_ret_5", "fwd_rank_5", "tb_label", "tb_time", "tb_ret", "tb_end", "atr_pct_14"]


@dataclass
class FoldResult:
    fold: Fold
    bundle: SwingBundle
    data: FoldData
    pred: pd.DataFrame          # one row per test row: keys, model outputs, rank_in_universe, and the realised labels (for evaluation only)


def rank_within_date(pred: pd.DataFrame) -> pd.Series:
    """1 = best score of the day (ties broken by instrument id, so the ranking is deterministic)."""
    order = pred.sort_values(["trade_date", "score", "instrument_id"], ascending=[True, False, True], kind="mergesort")
    return order.groupby("trade_date").cumcount().add(1).reindex(pred.index)


def predict_rows(bundle: SwingBundle, rows: pd.DataFrame) -> pd.DataFrame:
    out = bundle.predict(rows)
    keys = rows[["trade_date", "instrument_id"]]
    carried = rows[[c for c in CARRY if c in rows.columns]]
    res = pd.concat([keys, out, carried], axis=1)
    res["base_rate"] = bundle.base_rate
    res["calib_base_rate"] = bundle.calibration_base_rate
    res["rank_in_universe"] = rank_within_date(res)
    return res


def fit_fold(data: FoldData, cfg: SwingConfig, tree: dict) -> FoldResult:
    bundle = fit_bundle(data.fit, data.val, cfg, tree, meta={"fold": data.fold.index, "train": [str(data.fold.train_start.date()), str(data.fold.train_end.date())],
                                                              "test": [str(data.fold.test_start.date()), str(data.fold.test_end.date())]})
    return FoldResult(data.fold, bundle, data, predict_rows(bundle, data.test))


def run_walk_forward(frame: pd.DataFrame, cfg: SwingConfig, tree: dict, end: pd.Timestamp, on_fold: Callable[[FoldResult], None] | None = None) -> list[FoldResult]:
    """``frame`` needs ``label_end_max`` (``folds.with_ends``). Nothing dated at or after ``end`` (the held-out start) is used."""
    results = []
    for fold in plan_folds(frame, cfg, end):
        res = fit_fold(split_fold(frame, fold, cfg), cfg, tree)
        if on_fold is not None:
            on_fold(res)
        results.append(res)
    return results


def final_fold(frame: pd.DataFrame, cfg: SwingConfig, holdout_start: pd.Timestamp, holdout_end: pd.Timestamp) -> Fold:
    """A pseudo fold whose 'test window' is the held-out period: train = every session up to ``embargo`` sessions before it."""
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(frame["trade_date"]).unique()))
    t = int(sessions.searchsorted(pd.Timestamp(holdout_start)))
    e = cfg.folds.embargo_sessions
    train_end = sessions[t - e - 1]
    return Fold(-1, "expanding", sessions[0], train_end, pd.Timestamp(holdout_start), pd.Timestamp(holdout_end), e, int(t - e), 0)


def fit_final(frame: pd.DataFrame, cfg: SwingConfig, tree: dict, holdout_start: pd.Timestamp, holdout_end: pd.Timestamp) -> tuple[SwingBundle, FoldData]:
    """The model trained on ALL development data (its labels purged against the held-out start). It is never scored on the held-out period here."""
    fold = final_fold(frame, cfg, holdout_start, holdout_end)
    data = split_fold(frame, fold, cfg)
    data = FoldData(fold, data.fit, data.val, data.test.iloc[0:0], data.purged, data.unlabelled, data.fit_dropped)      # the held-out rows are not even carried along
    bundle = fit_bundle(data.fit, data.val, cfg, tree, meta={"fold": "final", "train": [str(fold.train_start.date()), str(fold.train_end.date())]})
    return bundle, data


def concat_predictions(results: list[FoldResult]) -> pd.DataFrame:
    return pd.concat([r.pred.assign(fold=r.fold.index) for r in results], ignore_index=True)
