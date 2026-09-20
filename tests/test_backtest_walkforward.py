"""Walk-forward folds, purging, embargo and the held-out final period."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from predict_stock.backtest.walkforward import (
    Fold, OOSAlreadyUsed, OOSLeak, assert_development_only, make_folds, make_holdout, oos_status, purge_train, purged_count, reveal_holdout,
)
from predict_stock.db.models import Experiment
from predict_stock.db.session import session_scope

S = pd.bdate_range("2020-01-01", periods=1000)


def test_expanding_folds_grow_the_training_set_and_never_overlap_the_tests():
    folds = make_folds(S, scheme="expanding", train_sessions=300, test_sessions=100, embargo_sessions=5)
    assert len(folds) == 6 and [f.n_train for f in folds] == [300, 400, 500, 600, 700, 800]
    for a, b in zip(folds, folds[1:]):
        assert b.test_start == S[S.get_loc(a.test_end) + 1]                             # tests are contiguous, none overlaps another
        assert a.train_start == b.train_start == S[0]
    assert folds[-1].test_end == S[904] and S[904] < S[-1]                              # the leftover 95 sessions are too few for another fold


def test_rolling_folds_keep_a_constant_training_window():
    folds = make_folds(S, scheme="rolling", train_sessions=300, test_sessions=100, embargo_sessions=5)
    assert all(f.n_train == 300 for f in folds)
    assert folds[1].train_start == S[S.get_loc(folds[0].train_start) + 100]              # the window slides by the step
    assert len({f.train_start for f in folds}) == len(folds)


def test_the_embargo_is_a_gap_of_exactly_that_many_sessions():
    for emb in (0, 1, 5, 21):
        for f in make_folds(S, train_sessions=300, test_sessions=100, embargo_sessions=emb):
            gap = S.get_loc(f.test_start) - S.get_loc(f.train_end) - 1
            assert gap == emb


def test_step_can_differ_from_the_test_size_and_the_holdout_end_is_respected():
    folds = make_folds(S, train_sessions=300, test_sessions=100, step_sessions=50, end=S[700])
    assert [S.get_loc(f.test_start) - S.get_loc(g.test_start) for g, f in zip(folds, folds[1:])] == [50] * (len(folds) - 1)
    assert all(f.test_end < S[700] for f in folds)                                         # nothing at or after `end` is ever used
    assert make_folds(S, train_sessions=2000, test_sessions=100) == []


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_folds(S, scheme="random", train_sessions=10, test_sessions=5)
    with pytest.raises(ValueError):
        make_folds(S, train_sessions=0, test_sessions=5)
    with pytest.raises(ValueError):
        make_folds(S, train_sessions=10, test_sessions=5, embargo_sessions=-1)


def labelled(horizon):
    """One sample per session; its label ends ``horizon`` sessions later."""
    ends = [S[min(i + horizon, len(S) - 1)] for i in range(len(S))]
    return pd.DataFrame({"trade_date": S, "tb_end": ends})


@pytest.mark.parametrize("horizon", [1, 5, 10, 63])
def test_purging_removes_every_training_sample_whose_label_reaches_the_test_window(horizon):
    frame = labelled(horizon)
    for f in make_folds(S, train_sessions=300, test_sessions=100, embargo_sessions=0):
        train = purge_train(frame, f, "tb_end")
        assert (train["tb_end"] < f.test_start).all() and (train["trade_date"] <= f.train_end).all() and (train["trade_date"] >= f.train_start).all()
        # a label that ENDS on the first test session already used that session's data, so the last `horizon` samples go
        assert purged_count(frame, f, "tb_end") == horizon


def test_an_embargo_at_least_as_long_as_the_label_makes_purging_a_no_op():
    frame = labelled(10)
    for f in make_folds(S, train_sessions=300, test_sessions=100, embargo_sessions=10):
        assert purged_count(frame, f, "tb_end") == 0


def test_purging_drops_unknown_labels_and_uses_the_label_end_not_the_sample_date():
    f = make_folds(S, train_sessions=300, test_sessions=100)[0]
    frame = labelled(10)
    frame.loc[5, "tb_end"] = pd.NaT                                                         # label never resolved
    frame.loc[6, "tb_end"] = f.test_start + pd.Timedelta(days=30)                           # a long label: sample is early, its outcome is in the test period
    kept = purge_train(frame, f, "tb_end")
    assert 5 not in kept.index and 6 not in kept.index and 7 in kept.index


# ---- held-out final period ------------------------------------------------------------------------------------------------------
def test_holdout_is_the_last_months_and_development_must_stay_before_it():
    h = make_holdout(S, 12)
    assert h.end == S[-1] and h.start > S[-1] - pd.DateOffset(months=12) and h.start <= S[-1] - pd.DateOffset(months=12) + pd.Timedelta(days=4)
    assert_development_only(S[S.get_loc(h.start) - 1], h)
    with pytest.raises(OOSLeak, match="inside the held-out period"):
        assert_development_only(h.start, h)
    folds = make_folds(S, train_sessions=200, test_sessions=50, end=h.start)
    assert folds and all(f.test_end < h.start for f in folds)


def test_the_holdout_identifier_depends_on_universe_and_dates_only():
    h = make_holdout(S, 12)
    assert h.identifier("U") == make_holdout(S, 12).identifier("U") and h.identifier("U") != h.identifier("V")
    assert h.identifier("U") != make_holdout(S, 6).identifier("U") and h.identifier("U").startswith("oos:")


def test_the_holdout_can_be_revealed_exactly_once(engine):
    h = make_holdout(S, 12)
    with session_scope(engine) as s:
        assert oos_status(s, h, "U") is None
        row = reveal_holdout(s, h, "U", candidates=["baseline:a", "model:b"], summary={"sharpe": 1.0})
        assert row.status == "used" and row.params["candidates"] == ["baseline:a", "model:b"]
    with session_scope(engine) as s:
        assert oos_status(s, h, "U") is not None
        with pytest.raises(OOSAlreadyUsed, match="only once"):
            reveal_holdout(s, h, "U", candidates=["model:c"], summary={"sharpe": 2.0})      # a second peek, with another candidate
        assert s.scalar(select(func.count()).select_from(Experiment)) == 1
        reveal_holdout(s, h, "OTHER_UNIVERSE", candidates=["x"], summary={})                  # a different holdout is a different budget
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(Experiment)) == 2
