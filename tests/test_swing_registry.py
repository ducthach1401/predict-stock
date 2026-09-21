"""Models and predictions of the SWING model in the database (idempotency, versioning, details), and the model-explanation records."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from predict_stock.config import PROJECT_ROOT, SwingConfig
from predict_stock.db.models import Model, Prediction
from predict_stock.db.session import session_scope
from predict_stock.features.sets import load_definitions, sync_definitions
from predict_stock.swing.model import fit_bundle
from predict_stock.swing.registry import build_details, register_model, save_predictions
from predict_stock.swing.walkforward import rank_within_date
from tests.conftest import apply
from tests.swing_helpers import synthetic_frame


@pytest.fixture
def db(engine, cfg):
    fsets, lspecs = load_definitions(PROJECT_ROOT / cfg.features.definitions_path)
    with session_scope(engine) as s:
        sync_definitions(s, fsets, lspecs)
    apply(engine, "T1", ["AAA", "BBB", "CCC"], date(2024, 1, 1))
    return engine


@pytest.fixture(scope="module")
def small():
    df = synthetic_frame(n_days=330, n_inst=12, seed=9)
    d = sorted(df["trade_date"].unique())
    fit, val = df[df["trade_date"] < d[220]], df[(df["trade_date"] >= d[220]) & (df["trade_date"] < d[300])]
    scfg = SwingConfig(n_estimators=30, early_stopping_rounds=6, holding_buckets=4, lgbm={"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30})
    return fit_bundle(fit, val, scfg), val, scfg


def test_registering_the_same_model_twice_reuses_the_row(db, cfg, small, tmp_path):
    bundle, _, scfg = small
    c = cfg.model_copy(update={"swing": scfg})
    a = register_model(db, c, bundle, "swing_lgbm_f00", dataset_id=None, experiment_id=None, extra_params={"fold": 0}, artifact_dir=tmp_path)
    b = register_model(db, c, bundle, "swing_lgbm_f00", dataset_id=None, experiment_id=None, extra_params={"fold": 0}, artifact_dir=tmp_path)
    assert a[:2] == (b[0], 1) and b[3] is True and a[3] is False and a[2] == b[2]
    with db.connect() as conn:
        assert conn.execute(select(func.count()).select_from(Model)).scalar() == 1


def test_a_different_model_under_the_same_name_gets_the_next_version_and_keeps_the_old_one(db, cfg, small, tmp_path):
    bundle, val, scfg = small
    c = cfg.model_copy(update={"swing": scfg})
    first = register_model(db, c, bundle, "swing_lgbm_f00", dataset_id=None, experiment_id=None, extra_params={}, artifact_dir=tmp_path)
    other = fit_bundle(val.iloc[:400].pipe(lambda v: pd.concat([v] * 3)).reset_index(drop=True), val, scfg.model_copy(update={"seed": 3}))
    second = register_model(db, c, other, "swing_lgbm_f00", dataset_id=None, experiment_id=None, extra_params={}, artifact_dir=tmp_path)
    assert (first[1], second[1]) == (1, 2) and first[0] != second[0] and first[2] != second[2]
    assert (tmp_path / "swing_lgbm_f00" / "v1.json.gz").exists() and (tmp_path / "swing_lgbm_f00" / "v2.json.gz").exists()


def test_the_model_row_carries_everything_needed_to_reproduce_it(db, cfg, small, tmp_path):
    import hashlib
    bundle, _, scfg = small
    c = cfg.model_copy(update={"swing": scfg})
    mid, version, sha, _ = register_model(db, c, bundle, "swing_lgbm_f03", dataset_id=None, experiment_id=None, extra_params={"tuning_study": "abc"}, artifact_dir=tmp_path)
    with db.connect() as conn:
        row = conn.execute(select(Model).where(Model.id == mid)).one()
    assert row.status == "candidate" and row.algo == "lightgbm_swing" and row.seed == scfg.seed and row.feature_set_id and row.label_spec_id
    assert row.artifact_sha256 == sha == hashlib.sha256((tmp_path / "swing_lgbm_f03" / "v1.json.gz").read_bytes()).hexdigest()
    assert row.params["tree_params"]["num_leaves"] == 7 and row.params["features"] == ["f_a", "f_b", "f_c"] and row.params["tuning_study"] == "abc"
    assert row.params["calibration"] == "isotonic" and row.params["horizon"] == 10 and row.params["best_iteration"]["rank"] >= 1


def _pred_frame(bundle, val, ids, n_days=4):
    rows = val[val["trade_date"].isin(sorted(val["trade_date"].unique())[:n_days]) & val["instrument_id"].isin([1, 2, 3])]
    out = pd.concat([rows[["trade_date", "instrument_id"]], bundle.predict(rows)], axis=1)
    out["rank_in_universe"] = rank_within_date(out)
    sr, _ = bundle.contributions(rows, "rank")
    se, _ = bundle.contributions(rows, "event")
    out["details"] = build_details(out, sr, se, bundle.features, rows[bundle.features].to_numpy(float))
    out["instrument_id"] = out["instrument_id"].map(dict(zip([1, 2, 3], ids)))
    return out


def test_predictions_are_saved_with_details_and_upserted_not_duplicated(db, cfg, small, tmp_path):
    bundle, val, scfg = small
    c = cfg.model_copy(update={"swing": scfg})
    with db.connect() as conn:
        from predict_stock.db.models import Instrument
        ids = [r[0] for r in conn.execute(select(Instrument.id).order_by(Instrument.id))]
    mid = register_model(db, c, bundle, "swing_lgbm_f00", dataset_id=None, experiment_id=None, extra_params={}, artifact_dir=tmp_path)[0]
    frame = _pred_frame(bundle, val, ids)
    assert save_predictions(db, mid, None, 10, frame) == len(frame) == 12
    changed = frame.assign(score=frame["score"] + 1.0)
    assert save_predictions(db, mid, None, 10, changed) == 12
    with db.connect() as conn:
        rows = conn.execute(select(Prediction).where(Prediction.model_id == mid).order_by(Prediction.as_of_date, Prediction.rank_in_universe)).all()
    assert len(rows) == 12
    r0 = rows[0]
    assert r0.rank_in_universe == 1 and r0.horizon_days == 10 and 0 <= r0.proba <= 1 and r0.score == pytest.approx(changed.sort_values(["trade_date", "rank_in_universe"]).iloc[0]["score"])
    d = r0.details
    assert d["q10"] <= d["q50"] <= d["q90"] and d["hold_median"] <= d["hold_p75"] and d["hold_n"] >= 100
    assert len(d["shap_rank"]) == 3 and all(len(t) == 3 for t in d["shap_rank"])          # (feature, value, contribution), at most 5 features (3 exist)
    assert abs(d["shap_rank"][0][2]) >= abs(d["shap_rank"][-1][2])                            # ordered by |contribution|


def test_ranks_are_one_for_the_best_score_each_day_and_break_ties_by_instrument_id():
    df = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 2), "instrument_id": [4, 3, 2, 1, 9, 8],
                       "score": [0.5, 0.9, 0.5, 0.1, 1.0, 1.0]})
    assert list(rank_within_date(df)) == [3, 1, 2, 4, 2, 1]


def test_shap_top_features_in_details_explain_the_direction_of_the_score(small):
    bundle, val, _ = small
    rows = val.head(30)
    p = bundle.predict(rows)
    sr, se = bundle.contributions(rows, "rank")[0], bundle.contributions(rows, "event")[0]
    det = build_details(pd.concat([rows[["trade_date", "instrument_id"]], p], axis=1), sr, se, bundle.features, rows[bundle.features].to_numpy(float), top=1)
    # the planted signal lives in f_a: it is the top contributor of most rows, and its contribution has the sign of (value - typical value)
    tops = [d["shap_rank"][0] for d in det]
    assert np.mean([t[0] == "f_a" for t in tops]) > 0.6
    fa = [t for t in tops if t[0] == "f_a"]
    assert np.mean([np.sign(v) == np.sign(c) for _, v, c in fa if abs(v) > 0.3]) > 0.8


def test_an_older_version_is_retired_when_a_new_version_is_registered(db, cfg, small, tmp_path):
    bundle, val, scfg = small
    c = cfg.model_copy(update={"swing": scfg})
    first = register_model(db, c, bundle, "swing_lgbm_f01", dataset_id=None, experiment_id=None, extra_params={}, artifact_dir=tmp_path)
    other = fit_bundle(val.iloc[:400].pipe(lambda v: pd.concat([v] * 3)).reset_index(drop=True), val, scfg.model_copy(update={"seed": 5}))
    second = register_model(db, c, other, "swing_lgbm_f01", dataset_id=None, experiment_id=None, extra_params={}, artifact_dir=tmp_path)
    with db.connect() as conn:
        status = dict(conn.execute(select(Model.version, Model.status).where(Model.name == "swing_lgbm_f01")).all())
    assert status == {1: "retired", 2: "candidate"} and first[1] == 1 and second[1] == 2
