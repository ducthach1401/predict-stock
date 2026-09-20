"""Walk-forward folds with purging and embargo, and a held-out final period that can be evaluated ONCE.

Folds (indices are trading sessions, so holidays never shorten a window)
    expanding : train = [0 .. test_start - embargo - 1]
    rolling   : train = the ``train_sessions`` sessions before the embargo gap
    test      : ``test_sessions`` sessions from ``test_start``; the next fold moves by ``step_sessions``

Embargo = a gap of ``embargo_sessions`` sessions between the last training session and the first test session.
Purging = a training sample is dropped when its LABEL is not yet known at the start of the test window, i.e. when its
label end date >= test start (the label of a sample at t looks forward up to its end date; without purging its outcome
would overlap the test period). Use the ``*_end`` columns of the dataset (``tb_end``, ``fwd_end_h``).

Held-out final period ("out-of-sample"): the last ``months`` of data are excluded from every development fold and every
development backtest. It may be REVEALED once: the first reveal is recorded in the ``experiments`` table and any later one
raises, so the number cannot be tuned against. Reveal it for the final candidate and all baselines in ONE run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from predict_stock.db.models import Experiment


class OOSAlreadyUsed(RuntimeError):
    pass


class OOSLeak(RuntimeError):
    pass


@dataclass(frozen=True)
class Fold:
    index: int
    scheme: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_sessions: int
    n_train: int
    n_test: int

    def as_dict(self) -> dict:
        return {"fold": self.index, "scheme": self.scheme, "train": [str(self.train_start.date()), str(self.train_end.date())],
                "test": [str(self.test_start.date()), str(self.test_end.date())], "embargo_sessions": self.embargo_sessions,
                "n_train_sessions": self.n_train, "n_test_sessions": self.n_test}


def make_folds(sessions, *, scheme: str = "expanding", train_sessions: int, test_sessions: int, step_sessions: int | None = None,
               embargo_sessions: int = 0, end: pd.Timestamp | None = None) -> list[Fold]:
    """Folds over the sorted, unique ``sessions``; nothing after ``end`` (e.g. the start of the held-out period) is used."""
    s = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(sessions)))))
    if end is not None:
        s = s[s < pd.Timestamp(end)]
    if scheme not in ("expanding", "rolling"):
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if min(train_sessions, test_sessions) < 1 or embargo_sessions < 0:
        raise ValueError("train_sessions and test_sessions must be positive, embargo_sessions non-negative")
    step = step_sessions or test_sessions
    folds, k = [], 0
    test_start = train_sessions + embargo_sessions
    while test_start + test_sessions <= len(s):
        train_end = test_start - embargo_sessions - 1
        train_start = 0 if scheme == "expanding" else train_end - train_sessions + 1
        folds.append(Fold(k, scheme, s[train_start], s[train_end], s[test_start], s[test_start + test_sessions - 1], embargo_sessions,
                          train_end - train_start + 1, test_sessions))
        k += 1
        test_start += step
    return folds


def purge_train(frame: pd.DataFrame, fold: Fold, label_end_col: str, date_col: str = "trade_date") -> pd.DataFrame:
    """Training rows for ``fold``: dated within the training window AND whose label ends before the test window starts
    (rows with an unknown label end are unusable and dropped)."""
    d = pd.to_datetime(frame[date_col])
    keep = (d >= fold.train_start) & (d <= fold.train_end) & frame[label_end_col].notna() & (pd.to_datetime(frame[label_end_col]) < fold.test_start)
    return frame[keep.to_numpy()]


def purged_count(frame: pd.DataFrame, fold: Fold, label_end_col: str, date_col: str = "trade_date") -> int:
    d = pd.to_datetime(frame[date_col])
    inside = (d >= fold.train_start) & (d <= fold.train_end)
    return int(inside.sum() - len(purge_train(frame, fold, label_end_col, date_col)))


# ---- held-out final period ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Holdout:
    start: pd.Timestamp      # first session of the held-out period
    end: pd.Timestamp        # last session

    def identifier(self, universe: str) -> str:
        raw = json.dumps({"universe": universe, "start": str(self.start.date()), "end": str(self.end.date())}, sort_keys=True)
        return "oos:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_holdout(sessions, months: int) -> Holdout:
    """The last ``months`` months of ``sessions``."""
    s = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(sessions)))))
    cut = s[-1] - pd.DateOffset(months=months)
    inside = s[s > cut]
    return Holdout(inside[0], s[-1])


def assert_development_only(last_session_used: pd.Timestamp, holdout: Holdout) -> None:
    """Development backtests / model fitting must stay strictly before the held-out period."""
    if pd.Timestamp(last_session_used) >= holdout.start:
        raise OOSLeak(f"a development run reaches {pd.Timestamp(last_session_used).date()}, inside the held-out period starting {holdout.start.date()}")


def oos_status(session: Session, holdout: Holdout, universe: str) -> Experiment | None:
    return session.scalars(select(Experiment).where(Experiment.name == holdout.identifier(universe)).order_by(Experiment.id)).first()


def reveal_holdout(session: Session, holdout: Holdout, universe: str, *, candidates: list[str], summary: dict, run_id: int | None = None,
                   config_snapshot_id: int | None = None) -> Experiment:
    """Record the one allowed evaluation of the held-out period. Raises ``OOSAlreadyUsed`` if it was already revealed."""
    prior = oos_status(session, holdout, universe)
    if prior is not None:
        raise OOSAlreadyUsed(f"the held-out period {holdout.start.date()}..{holdout.end.date()} of {universe} was already evaluated "
                             f"(experiment #{prior.id}, {prior.started_at:%Y-%m-%d}); it may be evaluated only once")
    row = Experiment(name=holdout.identifier(universe), description=f"held-out evaluation of {', '.join(candidates)}", run_id=run_id,
                     config_snapshot_id=config_snapshot_id,
                     params={"universe": universe, "start": str(holdout.start.date()), "end": str(holdout.end.date()), "candidates": candidates},
                     summary=summary, status="used", finished_at=datetime.now(timezone.utc).replace(tzinfo=None))
    session.add(row)
    session.flush()
    return row
