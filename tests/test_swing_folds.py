"""Walk-forward splits of the SWING model: windows, purging by label end, embargo, and that nothing from the test window reaches the model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.config import SwingConfig
from predict_stock.swing.folds import plan_folds, split_fold, with_ends
from predict_stock.swing.model import fit_bundle
from tests.swing_helpers import synthetic_frame

FOLDS = {"train_min_sessions": 300, "test_sessions": 100, "step_sessions": 100, "embargo_sessions": 10, "val_sessions": 60}


@pytest.fixture(scope="module")
def frame():
    return with_ends(synthetic_frame(n_days=800, n_inst=8, seed=11))


def cfg_with(**folds):
    return SwingConfig(folds={**FOLDS, **folds}, n_estimators=40, early_stopping_rounds=8, holding_buckets=4,
                       lgbm={"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30})


def test_folds_are_expanding_and_leave_the_embargo_gap(frame):
    cfg = cfg_with()
    folds = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))
    assert len(folds) >= 4
    sessions = pd.DatetimeIndex(sorted(frame["trade_date"].unique()))
    for f in folds:
        gap = sessions[(sessions > f.train_end) & (sessions < f.test_start)]
        assert len(gap) == cfg.folds.embargo_sessions
    assert all(a.train_start == b.train_start for a, b in zip(folds, folds[1:]))          # expanding: same start
    assert all(a.test_end < b.test_start for a, b in zip(folds, folds[1:]))                 # test windows are disjoint (step = test length)
    assert [b.train_end > a.train_end for a, b in zip(folds, folds[1:])] == [True] * (len(folds) - 1)


def test_nothing_at_or_after_the_holdout_start_is_used(frame):
    hold = pd.Timestamp(sorted(frame["trade_date"].unique())[600])
    for f in plan_folds(frame, cfg_with(), hold):
        assert f.test_end < hold
        d = split_fold(frame, f, cfg_with())
        assert d.test["trade_date"].max() < hold and d.val["label_end_max"].max() < hold


def test_no_training_or_validation_label_is_still_open_when_the_test_window_starts(frame):
    cfg = cfg_with()
    for f in plan_folds(frame, cfg, pd.Timestamp("2100-01-01")):
        d = split_fold(frame, f, cfg)
        for part in (d.fit, d.val):
            assert (pd.to_datetime(part["label_end_max"]) < f.test_start).all()
            assert (part["trade_date"] <= f.train_end).all()
        assert d.test["trade_date"].min() >= f.test_start and d.test["trade_date"].max() <= f.test_end


def test_fit_labels_end_before_validation_starts_and_the_parts_do_not_overlap(frame):
    cfg = cfg_with()
    f = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))[1]
    d = split_fold(frame, f, cfg)
    val_start = d.val["trade_date"].min()
    assert (pd.to_datetime(d.fit["label_end_max"]) < val_start).all() and d.fit["trade_date"].max() < val_start
    assert d.val["trade_date"].nunique() == cfg.folds.val_sessions
    assert set(d.fit.index).isdisjoint(d.val.index) and set(d.val.index).isdisjoint(d.test.index) and set(d.fit.index).isdisjoint(d.test.index)


def test_without_an_embargo_purging_removes_exactly_the_open_labels(frame):
    cfg = cfg_with(embargo_sessions=0)
    f = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))[0]
    d = split_fold(frame, f, cfg)
    assert d.purged > 0                                                        # labels reaching into the test window are dropped ...
    tail = frame[(frame["trade_date"] <= f.train_end) & (pd.to_datetime(frame["label_end_max"]) >= f.test_start)]
    assert d.purged == len(tail)                                               # ... and only those
    assert (tail["trade_date"] > f.train_end - pd.Timedelta(days=20)).all()


def test_with_the_embargo_equal_to_the_horizon_purging_is_a_no_op_for_the_barrier_label(frame):
    cfg = cfg_with(embargo_sessions=10)
    for f in plan_folds(frame, cfg, pd.Timestamp("2100-01-01")):
        assert split_fold(frame, f, cfg).purged == 0


def test_the_model_of_a_fold_does_not_depend_on_anything_in_the_test_window_or_after_it(frame):
    cfg = cfg_with()
    f = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))[1]
    base = split_fold(frame, f, cfg)
    mutated = frame.copy()
    later = pd.to_datetime(mutated["trade_date"]) >= f.test_start
    rng = np.random.default_rng(0)
    for c in ("f_a", "f_b", "f_c", "fwd_ret_5", "fwd_rank_5", "tb_time"):
        mutated.loc[later, c] = rng.normal(size=int(later.sum()))
    mutated.loc[later, "tb_label"] = 1.0
    other = split_fold(mutated, f, cfg)
    a = fit_bundle(base.fit, base.val, cfg)
    b = fit_bundle(other.fit, other.val, cfg)
    assert a.to_bytes() == b.to_bytes()
    # the predictions on the test window use only that window's own features
    pd.testing.assert_series_equal(a.predict(base.test)["score"], b.predict(base.test)["score"])


def test_a_training_window_shorter_than_the_validation_window_is_refused(frame):
    cfg = cfg_with(val_sessions=1000)
    f = plan_folds(frame, cfg_with(), pd.Timestamp("2100-01-01"))[0]
    with pytest.raises(ValueError, match="validation window"):
        split_fold(frame, f, cfg)


# ---- running the walk-forward --------------------------------------------------------------------------------------------------------
def test_walk_forward_predicts_each_test_window_once_and_only_there(frame):
    from predict_stock.swing.walkforward import concat_predictions, run_walk_forward
    cfg = cfg_with()
    hold = pd.Timestamp(sorted(frame["trade_date"].unique())[700])
    tree = {"learning_rate": 0.1}
    results = run_walk_forward(frame, cfg, tree, hold)
    preds = concat_predictions(results)
    assert len(results) >= 3
    assert not preds.duplicated(["trade_date", "instrument_id"]).any()
    assert preds["trade_date"].max() < hold
    for r in results:
        assert r.pred["trade_date"].between(r.fold.test_start, r.fold.test_end).all()
        assert (r.pred["rank_in_universe"] >= 1).all()
    top = preds[preds["rank_in_universe"] == 1]
    assert top["trade_date"].is_unique                                          # exactly one best name per day


def test_predictions_of_earlier_folds_do_not_change_when_later_data_change(frame):
    from predict_stock.swing.walkforward import run_walk_forward
    cfg = cfg_with()
    hold = pd.Timestamp("2100-01-01")
    base = run_walk_forward(frame, cfg, {"learning_rate": 0.1}, hold)
    k = 1
    cut = base[k].fold.test_end
    mutated = frame.copy()
    later = pd.to_datetime(mutated["trade_date"]) > cut
    rng = np.random.default_rng(1)
    for c in ("f_a", "f_b", "f_c", "fwd_ret_5", "fwd_rank_5", "tb_time"):
        mutated.loc[later, c] = rng.normal(size=int(later.sum()))
    mutated.loc[later, "tb_label"] = rng.choice([-1.0, 0.0, 1.0], size=int(later.sum()))
    other = run_walk_forward(mutated, cfg, {"learning_rate": 0.1}, hold)
    for a, b in zip(base[: k + 1], other[: k + 1]):
        pd.testing.assert_frame_equal(a.pred, b.pred)                           # identical predictions for every fold up to k
        assert a.bundle.to_bytes() == b.bundle.to_bytes()


def test_the_final_model_never_sees_the_held_out_period(frame):
    from predict_stock.swing.walkforward import fit_final
    dates = sorted(frame["trade_date"].unique())
    hold_start, hold_end = pd.Timestamp(dates[700]), pd.Timestamp(dates[-1])
    bundle, data = fit_final(frame, cfg_with(), {"learning_rate": 0.1}, hold_start, hold_end)
    for part in (data.fit, data.val):
        assert part["trade_date"].max() <= data.fold.train_end < hold_start
        assert (pd.to_datetime(part["label_end_max"]) < hold_start).all()
    assert len(data.test) == 0
    mutated = frame.copy()
    inside = pd.to_datetime(mutated["trade_date"]) >= hold_start
    mutated.loc[inside, "f_a"] = 99.0
    mutated.loc[inside, "tb_label"] = 1.0
    assert fit_final(mutated, cfg_with(), {"learning_rate": 0.1}, hold_start, hold_end)[0].to_bytes() == bundle.to_bytes()


def test_end_to_end_a_planted_signal_is_found_out_of_sample_and_pure_noise_is_not():
    """The whole pipeline (folds, purging, boosters, calibration) on data where the truth is known: a signal must show up in the out-of-sample
    rank IC, and with no signal the IC must be indistinguishable from zero (a leak would inflate it)."""
    from predict_stock.swing.metrics import daily_rank_ic, ic_summary
    from predict_stock.swing.walkforward import concat_predictions, run_walk_forward
    cfg = cfg_with()
    out = {}
    for name, signal in (("planted", 1.0), ("null", 0.0)):
        f = with_ends(synthetic_frame(n_days=800, n_inst=12, seed=31, signal=signal))
        preds = concat_predictions(run_walk_forward(f, cfg, {"learning_rate": 0.1}, pd.Timestamp("2100-01-01")))
        out[name] = ic_summary(daily_rank_ic(preds, "score", "fwd_ret_5"), 5)
        if name == "planted":
            ok = preds["tb_label"].notna()
            from predict_stock.swing.calibration import auc
            assert auc((preds.loc[ok, "tb_label"] == 1).astype(float), preds.loc[ok, "proba_raw"]) > 0.6      # the event classifier finds it too
    assert out["planted"]["mean"] > 0.15 and out["planted"]["t_stat"] > 5
    assert abs(out["null"]["t_stat"]) < 3 and abs(out["null"]["mean"]) < 0.06


def test_the_holding_time_table_is_built_from_the_validation_rows_only(frame):
    cfg = cfg_with()
    f = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))[1]
    base = split_fold(frame, f, cfg)
    a = fit_bundle(base.fit, base.val, cfg, {"learning_rate": 0.1})
    changed_fit = base.fit.assign(tb_time=1.0)                                # rewrite the holding times of the fit rows: the table must not move
    b = fit_bundle(changed_fit, base.val, cfg, {"learning_rate": 0.1})
    assert a.hold.to_dict() == b.hold.to_dict()
    changed_val = base.val.assign(tb_time=(base.val["tb_time"] % 3) + 1.0)    # ... but the validation rows DO define it
    c = fit_bundle(base.fit, changed_val, cfg, {"learning_rate": 0.1})
    assert c.hold.to_dict() != a.hold.to_dict()
    assert sum(a.hold.n) == a.hold.overall["n"] <= len(base.val)


def test_rows_without_any_label_are_counted_as_unlabelled_not_as_purged(frame):
    cfg = cfg_with()
    f = plan_folds(frame, cfg, pd.Timestamp("2100-01-01"))[1]
    rows = frame[(frame["trade_date"] >= f.train_start) & (frame["trade_date"] <= f.train_end)].index[[5, 6, 7]]
    holed = frame.copy()
    holed.loc[rows, ["fwd_end_5", "tb_end", "label_end_max"]] = pd.NaT
    d = split_fold(holed, f, cfg)
    assert d.unlabelled == 3 and d.purged == 0
    assert not set(rows) & (set(d.fit.index) | set(d.val.index))                 # and they are not trained on
