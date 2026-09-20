"""Walk-forward for the SWING model: expanding folds over the sessions of the dataset, every training/validation sample PURGED by the end
date of its labels, an embargo of ``folds.embargo_sessions`` between the training window and the test window, nothing at or after the
start of the held-out period.

  fold k:  [ fit .... | purge | val ........ ]  embargo  [ test ]
  * val   = the last ``val_sessions`` sessions of the training window (early stopping, calibration, holding-time table)
  * fit   = the sessions before ``val`` whose labels ended before ``val`` starts
  * a sample of the training window whose label is still open when the test window starts is dropped (``purge_train``)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from predict_stock.backtest.runner import with_label_end
from predict_stock.backtest.walkforward import Fold, make_folds, purge_train
from predict_stock.config import SwingConfig


@dataclass
class FoldData:
    fold: Fold
    fit: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    purged: int          # training-window rows removed because a label was still open at the test start
    unlabelled: int      # training-window rows with no label at all (e.g. a suspended instrument): unusable, not a leak issue
    fit_dropped: int     # rows removed from the fit part because a label was still open at the start of the validation part


def with_ends(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds ``label_end_max`` (latest end among all the row's labels) to a dataset frame."""
    return frame.assign(label_end_max=with_label_end(frame)["label_end_max"].to_numpy())


def plan_folds(frame: pd.DataFrame, cfg: SwingConfig, end: pd.Timestamp) -> list[Fold]:
    f = cfg.folds
    return make_folds(frame["trade_date"].unique(), scheme=f.scheme, train_sessions=f.train_min_sessions if f.scheme == "expanding" else f.train_window_sessions,
                      test_sessions=f.test_sessions, step_sessions=f.step_sessions, embargo_sessions=f.embargo_sessions, end=end)


def split_fold(frame: pd.DataFrame, fold: Fold, cfg: SwingConfig) -> FoldData:
    """``frame`` needs ``label_end_max`` (see ``with_ends``)."""
    d = pd.to_datetime(frame["trade_date"])
    in_train = (d >= fold.train_start) & (d <= fold.train_end)
    train = purge_train(frame, fold, "label_end_max")
    sessions = sorted(pd.to_datetime(train["trade_date"]).unique())
    if len(sessions) <= cfg.folds.val_sessions:
        raise ValueError(f"fold {fold.index}: {len(sessions)} training sessions is not more than the validation window ({cfg.folds.val_sessions})")
    val_start = sessions[-cfg.folds.val_sessions]
    td = pd.to_datetime(train["trade_date"])
    val = train[td >= val_start]
    before = train[td < val_start]
    fit = before[pd.to_datetime(before["label_end_max"]) < val_start]
    test = frame[((d >= fold.test_start) & (d <= fold.test_end)).to_numpy()]
    unlabelled = int((in_train & frame["label_end_max"].isna()).sum())
    return FoldData(fold, fit, val, test, int(in_train.sum() - unlabelled - len(train)), unlabelled, int(len(before) - len(fit)))
