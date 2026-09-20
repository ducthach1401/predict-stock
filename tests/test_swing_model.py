"""Calibration, holding-time table and the SWING model bundle (on synthetic data with a planted signal)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict_stock.config import SwingConfig
from predict_stock.swing import calibration as cal
from predict_stock.swing.holding import fit_hold_table
from predict_stock.swing.model import SwingBundle, SwingModelError, feature_columns, fit_bundle, load_bundle
from tests.swing_helpers import synthetic_frame


# ---- calibration -----------------------------------------------------------------------------------------------------------------
def test_brier_log_loss_and_ece_known_values():
    y, p = np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 1.0, 0.0])
    assert cal.brier(y, p) == 0
    assert cal.brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert cal.log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    # a perfectly calibrated constant forecast has zero ECE; an always-wrong one does not
    assert cal.ece(np.array([1, 0, 0, 0] * 50), np.full(200, 0.25), bins=4) == pytest.approx(0.0, abs=1e-12)
    assert cal.ece(np.array([1, 1, 0, 0]), np.array([0.0, 0.0, 1.0, 1.0]), bins=2) == pytest.approx(1.0)


def test_reliability_bins_are_equal_frequency_and_cover_every_row():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    r = cal.reliability(rng.random(1000) < p, p, bins=10)
    assert list(r["n"]) == [100] * 10 and r["n"].sum() == 1000
    assert (np.diff(r["mean_pred"]) > 0).all()
    assert (r["observed"].sub(r["mean_pred"]).abs() < 0.12).all()       # the data ARE calibrated: observed ~ predicted


def test_isotonic_is_monotone_clipped_and_repairs_a_distorted_probability():
    rng = np.random.default_rng(1)
    true_p = rng.random(4000) * 0.6
    y = (rng.random(4000) < true_p).astype(float)
    raw = np.clip(true_p ** 2 * 1.4 + 0.02, 0, 1)                       # a monotone but wrong (overconfident-shaped) score
    iso = cal.fit_isotonic(raw, y)
    grid = np.linspace(-0.5, 1.5, 300)
    out = cal.apply_calibrator(iso, grid)
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1     # monotone; out-of-range inputs are clipped, not extrapolated
    assert cal.ece(y, cal.apply_calibrator(iso, raw)) < cal.ece(y, raw)


def test_platt_scaling_recovers_a_scaled_logit():
    rng = np.random.default_rng(2)
    z = rng.normal(size=6000)
    y = (rng.random(6000) < 1 / (1 + np.exp(-(0.5 * z - 1.0)))).astype(float)
    raw = 1 / (1 + np.exp(-z))                                          # over-confident by a factor 2, and shifted
    platt = cal.fit_platt(raw, y)
    assert platt["a"] == pytest.approx(0.5, abs=0.06) and platt["b"] == pytest.approx(-1.0, abs=0.06)
    assert cal.ece(y, cal.apply_calibrator(platt, raw)) < 0.03


def test_calibrator_survives_json_round_trip():
    import json
    iso = cal.fit_isotonic(np.linspace(0, 1, 50), (np.linspace(0, 1, 50) > 0.5).astype(float))
    assert json.loads(json.dumps(iso)) == iso


# ---- holding-time table ----------------------------------------------------------------------------------------------------------
def test_hold_table_medians_p75_and_sample_sizes():
    proba = np.repeat([0.1, 0.5], 500)
    hold = np.concatenate([np.full(500, 8.0), np.tile([2.0, 3.0, 4.0, 6.0], 125)])
    t = fit_hold_table(proba, hold, horizon=10, buckets=2, min_n=100)
    med, p75, n = t.lookup(np.array([0.1, 0.5]))
    assert list(med) == [8.0, 3.5] and list(p75) == [8.0, 4.5] and list(n) == [500, 500]
    assert t.overall["n"] == 1000


def test_hold_table_merges_buckets_that_are_too_small_and_never_reports_n_below_min():
    rng = np.random.default_rng(3)
    proba = rng.random(600)
    t = fit_hold_table(proba, rng.integers(1, 11, 600).astype(float), horizon=10, buckets=10, min_n=100)
    assert min(t.n) >= 100 and sum(t.n) == 600 and len(t.n) <= 6


def test_hold_table_with_tied_probabilities_collapses_to_one_bucket():
    t = fit_hold_table(np.full(300, 0.3), np.arange(300) % 10 + 1.0, horizon=10, buckets=10, min_n=50)
    assert len(t.n) == 1 and t.n[0] == 300


def test_hold_table_counts_time_outs_at_the_horizon():
    t = fit_hold_table(np.full(200, 0.3), np.array([10.0] * 50 + [2.0] * 150), horizon=10, buckets=1, min_n=10)
    assert t.timed_out[0] == pytest.approx(0.25) and t.median[0] == 2.0


# ---- the model bundle ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg_small():
    return SwingConfig(n_estimators=60, early_stopping_rounds=10, lgbm={"learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30}, holding_buckets=5)


@pytest.fixture(scope="module")
def split():
    df = synthetic_frame(n_days=360, n_inst=20, seed=5)
    dates = sorted(df["trade_date"].unique())
    cut = dates[240]
    return df[df["trade_date"] < cut].copy(), df[(df["trade_date"] >= cut) & (df["trade_date"] < dates[330])].copy()


@pytest.fixture(scope="module")
def bundle(split, cfg_small):
    fit, val = split
    return fit_bundle(fit, val, cfg_small)


def test_feature_columns_exclude_keys_and_labels(split):
    fit, _ = split
    assert feature_columns(fit) == ["f_a", "f_b", "f_c"]


def test_the_model_finds_a_planted_signal_and_ignores_the_noise_feature(bundle, split):
    _, val = split
    pred = bundle.predict(val)
    from scipy.stats import spearmanr
    assert spearmanr(pred["score"], val["fwd_ret_5"], nan_policy="omit")[0] > 0.15
    imp = dict(zip(bundle.features, bundle.boosters["rank"].feature_importance("gain")))
    assert imp["f_a"] > 3 * imp["f_c"]


def test_output_columns_and_ranges(bundle, split):
    _, val = split
    p = bundle.predict(val)
    assert {"score", "proba_raw", "proba_isotonic", "proba_platt", "proba", "q10", "q50", "q90", "hold_median", "hold_p75", "hold_n"} <= set(p.columns)
    assert p["proba"].between(0, 1).all() and p["proba_raw"].between(0, 1).all()
    assert (p["hold_median"] <= p["hold_p75"]).all() and (p["hold_n"] >= 100).all()
    assert len(p) == len(val)


def test_quantiles_never_cross(bundle, split):
    _, val = split
    p = bundle.predict(val)
    assert (p["q10"] <= p["q50"]).all() and (p["q50"] <= p["q90"]).all()


def test_the_quantile_interval_covers_roughly_the_stated_share(bundle, split):
    _, val = split
    p = bundle.predict(val)
    y = val["fwd_ret_5"]
    ok = y.notna()
    coverage = ((y[ok] >= p.loc[ok, "q10"]) & (y[ok] <= p.loc[ok, "q90"])).mean()
    assert 0.6 < coverage < 0.95                                        # nominal 80%; small synthetic sample, loose bounds


def test_the_calibrated_probability_is_better_calibrated_on_validation_than_the_raw_one(bundle, split):
    _, val = split
    p = bundle.predict(val)
    ok = val["tb_label"].notna()
    y = (val.loc[ok, "tb_label"] == 1).astype(float)
    assert cal.ece(y, p.loc[ok, "proba"]) <= cal.ece(y, p.loc[ok, "proba_raw"]) + 1e-9


def test_shap_contributions_add_up_to_the_raw_prediction(bundle, split):
    _, val = split
    rows = val.head(50)
    for head in ("rank", "event"):
        contrib, base = bundle.contributions(rows, head)
        raw = bundle.boosters[head].predict(rows[bundle.features].to_numpy(float), raw_score=True)
        assert contrib.shape == (50, 3)
        assert np.allclose(contrib.sum(axis=1) + base, raw, atol=1e-8)


def test_the_saved_model_is_byte_identical_for_the_same_data_and_seed(split, cfg_small, tmp_path):
    fit, val = split
    a, b = fit_bundle(fit, val, cfg_small), fit_bundle(fit, val, cfg_small)
    assert a.to_bytes() == b.to_bytes()
    assert a.save(tmp_path / "m.json.gz") == b.save(tmp_path / "m2.json.gz")


def test_a_different_seed_gives_a_different_model(split, cfg_small):
    fit, val = split
    other = cfg_small.model_copy(update={"seed": 7})
    assert fit_bundle(fit, val, cfg_small).to_bytes() != fit_bundle(fit, val, other).to_bytes()


def test_save_and_load_round_trip_gives_identical_predictions(bundle, split, tmp_path):
    _, val = split
    sha = bundle.save(tmp_path / "m.json.gz")
    loaded = load_bundle(tmp_path / "m.json.gz", sha)
    pd.testing.assert_frame_equal(bundle.predict(val), loaded.predict(val))
    assert loaded.features == bundle.features and loaded.base_rate == bundle.base_rate


def test_a_tampered_model_file_is_refused(bundle, tmp_path):
    sha = bundle.save(tmp_path / "m.json.gz")
    (tmp_path / "m.json.gz").write_bytes(b"corrupt")
    with pytest.raises(SwingModelError, match="sha256"):
        load_bundle(tmp_path / "m.json.gz", sha)


def test_predicting_without_a_feature_column_fails_loudly(bundle, split):
    _, val = split
    with pytest.raises(SwingModelError, match="f_b"):
        bundle.predict(val.drop(columns="f_b"))


def test_too_little_data_is_refused(split, cfg_small):
    fit, val = split
    with pytest.raises(SwingModelError, match="too few rows"):
        fit_bundle(fit.head(100), val, cfg_small)


def test_a_validation_set_with_a_single_class_is_refused_instead_of_calibrating_nonsense(split, cfg_small):
    fit, val = split
    with pytest.raises(SwingModelError, match="too few of one class"):
        fit_bundle(fit, val.assign(tb_label=-1.0), cfg_small)


def test_isotonic_on_raw_points_can_reach_one_but_binned_isotonic_cannot():
    rng = np.random.default_rng(4)
    raw = rng.random(3000) * 0.5
    y = (rng.random(3000) < 0.3).astype(float)
    top = np.argsort(raw)[-8:]
    y[top] = 1.0                                                          # the 8 highest scores happen to be all events
    assert cal.apply_calibrator(cal.fit_isotonic(raw, y), np.array([raw.max()]))[0] > 0.9            # raw-point fit chases them
    binned = cal.fit_isotonic(raw, y, min_bin=150)
    assert cal.apply_calibrator(binned, np.array([raw.max()]))[0] < 0.5                                # a 150-row bin dilutes 8 lucky rows
    assert (np.diff(cal.apply_calibrator(binned, np.linspace(0, 0.5, 200))) >= -1e-12).all()


def test_the_bundle_records_the_base_rate_of_its_calibration_rows(bundle, split):
    _, val = split
    ok = val["tb_label"].notna()
    assert bundle.calibration_base_rate == pytest.approx((val.loc[ok, "tb_label"] == 1).mean())
    assert bundle.base_rate != bundle.calibration_base_rate                                # two different populations


def test_the_probability_filter_is_relative_to_the_calibration_base_rate_by_default():
    from predict_stock.swing.strategies import with_probability_filter
    pred = pd.DataFrame({"score": [1.0, 2.0, 3.0], "proba": [0.2, 0.3, 0.4], "base_rate": [0.35] * 3, "calib_base_rate": [0.25] * 3})
    kept = with_probability_filter(pred, None, 1.0)["score"]
    assert list(kept.notna()) == [False, True, True]                       # >= 0.25 (calibration window), not >= 0.35 (training rows)
    assert list(with_probability_filter(pred, 0.35, 1.0)["score"].notna()) == [False, False, True]
